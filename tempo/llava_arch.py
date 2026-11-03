"""TeMPO: frame-causal video token compression (vectorized implementation).

Text-conditioned greedy residual-energy selection with committed-support
memory (a cross-frame warm-start shared basis) plus parameter-free
neighborhood fusion. Vectorized:

* Shared-basis warm-start projection ``for b in shared_basis: cond_var -= (X@b)^2``
  -> single batched matmul ``cond_var -= (X @ B^T) ** 2 .sum(-1)``.
* Inner picked-mask loops ``for s in selected: cond_var[s] = -1`` -> bool
  ``picked_mask`` + ``masked_fill``.
* Modified-Gram-Schmidt loop ``for b in frame_basis: v -= (v@b)*b`` -> CGS2
  (two batched matmuls, stable per Giraud-Langou-Rozloznik 2005).
* Weighted-fuse cluster loop ``for a in range(len(sel_idx))`` ->
  ``index_add_`` for sums + cluster-conditional weighted blending.

Selection indices match the reference implementation up to floating-point ties; output embeddings
match up to machine epsilon due to non-deterministic atomic adds.
"""

import os
import torch
import torch.nn.functional as F

from method_wrappers.common_llava import BaseVideoMethodMixin
from unified_token_pruning.runtime import ensure_nextline_embedding


class LlavaMetaForCausalLM_tempo(BaseVideoMethodMixin):
    def _ensure_method_module(self):
        pass

    @staticmethod
    def _weighted_fuse(tokens_f, sel_idx, T):
        compressed = tokens_f[sel_idx].clone()
        mask = torch.ones(T, dtype=torch.bool, device=tokens_f.device)
        mask[sel_idx] = False
        disc_idx = torch.where(mask)[0]
        if disc_idx.numel() == 0:
            return compressed
        norms = F.normalize(tokens_f, dim=-1)
        sim = norms[disc_idx] @ norms[sel_idx].t()
        assignments = sim.argmax(dim=-1)
        sim_w = sim.gather(1, assignments.unsqueeze(1)).squeeze(1)

        n_sel = sel_idx.numel()
        device = tokens_f.device
        # Cluster sums of weights and weighted sums of tokens_f[disc_idx].
        # weighted neighbor mean: nw[a] = sum(token_disc * w) / sum(w)
        # where w = sim_w within cluster a, normalized within cluster.
        wsum = torch.zeros(n_sel, device=device, dtype=tokens_f.dtype)
        wsum.index_add_(0, assignments, sim_w)
        # numerator: sum(token_disc[i] * sim_w[i]) per cluster
        weighted_tokens = tokens_f[disc_idx] * sim_w.unsqueeze(1)
        wnum = torch.zeros(n_sel, tokens_f.shape[-1], device=device, dtype=tokens_f.dtype)
        wnum.index_add_(0, assignments, weighted_tokens)
        # cluster size n_a
        counts = torch.zeros(n_sel, device=device, dtype=tokens_f.dtype)
        counts.index_add_(0, assignments, torch.ones(disc_idx.numel(), device=device, dtype=tokens_f.dtype))

        has = counts > 0
        if has.any():
            nw = torch.zeros_like(wnum)
            nw[has] = wnum[has] / (wsum[has].unsqueeze(1) + 1e-8)
            n_a = counts[has]  # (h,)
            sw = 1.0 / (1.0 + n_a.pow(0.5))
            comp_h = compressed[has]
            blend = sw.unsqueeze(1) * comp_h + (1 - sw).unsqueeze(1) * (0.5 * comp_h + 0.5 * nw[has])
            compressed = compressed.clone()
            compressed[has] = blend
        return compressed

    def _compress_video_tokens(self, feats, cls_attn, key_metric, text_tokens, nextline_token):
        nextline_token = ensure_nextline_embedding(self.get_model(), nextline_token, feats.device)
        ratio = float(os.environ.get("RETAIN_RATIO", "0.25"))
        alpha = float(os.environ.get("TEXT_ALPHA", "0.15"))
        B_max = int(os.environ.get("TEMPO_BASIS_CAP", "48"))
        F_num, T, D = feats.shape
        k = max(1, int(T * ratio))
        feats_f = feats.float()
        device = feats.device
        outputs = []

        if text_tokens is not None and text_tokens.numel() > 0:
            if text_tokens.device != device:
                text_tokens = text_tokens.to(device)
            text_f = text_tokens.float() if text_tokens.dim() == 2 else text_tokens.unsqueeze(0).float()
            q = F.normalize(text_f.mean(0, keepdim=True), dim=-1)  # (1, D)
        else:
            q = None

        # shared_basis maintained as a Python list of (D,) tensors. We materialize
        # it into a (n_shared, D) stack at each frame for batched ops; this keeps
        # the rolling tail trim semantics simple while still avoiding per-step
        # Python loops in the critical path.
        shared_basis = []  # list of (D,) fp32 tensors

        for f in range(F_num):
            tokens = feats[f]
            if k >= T:
                outputs.extend([tokens, nextline_token])
                continue

            tokens_f = feats_f[f]
            cond_var = (tokens_f ** 2).sum(dim=-1)

            # Warm-start: project out shared_basis as a batch
            if shared_basis:
                shared_T = torch.stack(shared_basis, dim=0)  # (n_shared, D)
                projs_shared = tokens_f @ shared_T.t()  # (T, n_shared)
                cond_var = cond_var - projs_shared.pow(2).sum(dim=-1)
                cond_var.clamp_(min=0)
            else:
                shared_T = None

            n_shared = len(shared_basis)
            # frame_basis tensor preallocated for n_shared + k vectors
            fb = torch.empty(n_shared + k, D, device=device, dtype=tokens_f.dtype)
            if n_shared > 0:
                fb[:n_shared] = shared_T
            n_fb = n_shared

            picked_mask = torch.zeros(T, dtype=torch.bool, device=device)
            selected_idx = torch.empty(k, dtype=torch.long, device=device)

            for step in range(k):
                cv = cond_var.masked_fill(picked_mask, -1.0)
                if q is not None:
                    text_rel = (tokens_f @ q.t()).squeeze(-1).abs()
                    text_rel = text_rel / (text_rel.max() + 1e-8)
                    cv_norm = cv / (cv.max() + 1e-8)
                    score = (1 - alpha) * cv_norm + alpha * text_rel
                    score = score.masked_fill(picked_mask, -1.0)
                    best = score.argmax()
                else:
                    best = cv.argmax()
                selected_idx[step] = best
                picked_mask[best] = True

                v = tokens_f[best].clone()
                if n_fb > 0:
                    B = fb[:n_fb]
                    # CGS2 (two passes for MGS-grade stability)
                    v = v - B.t() @ (B @ v)
                    v = v - B.t() @ (B @ v)
                vn = v.norm()
                if vn.item() > 1e-8:
                    v = v / vn
                    fb[n_fb] = v
                    n_fb += 1
                    projs = tokens_f @ v
                    cond_var = cond_var - projs ** 2
                    cond_var.clamp_(min=0)

            sel_idx_sorted = torch.sort(selected_idx).values
            compressed = self._weighted_fuse(tokens_f, sel_idx_sorted, T)
            outputs.extend([compressed.to(tokens.dtype), nextline_token])

            # Append new vectors (those added this frame past n_shared) to shared_basis
            new_count = n_fb - n_shared
            for i in range(new_count):
                shared_basis.append(fb[n_shared + i].detach().clone())
            if len(shared_basis) > B_max:
                shared_basis = shared_basis[-B_max:]

        return outputs
