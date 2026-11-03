"""TeMPO reference implementation (loop form).

Readable, loop-based version of the TeMPO selector:
1. Text-conditioned residual-energy scoring
2. Committed-support memory via a cross-frame shared basis
3. Parameter-free neighborhood fusion

``tempo`` (see llava_arch.py) is the vectorized version; selection
indices match this reference up to floating-point ties.
"""

import os

import torch
import torch.nn.functional as F

from method_wrappers.common_llava import BaseVideoMethodMixin
from unified_token_pruning.runtime import ensure_nextline_embedding


class LlavaMetaForCausalLM_tempo_v1(BaseVideoMethodMixin):
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
        for a in range(len(sel_idx)):
            assigned = assignments == a
            if assigned.sum() == 0:
                continue
            n_a = assigned.sum().item()
            w = sim_w[assigned]
            w = w / (w.sum() + 1e-8)
            nw = (tokens_f[disc_idx[assigned]] * w.unsqueeze(1)).sum(0)
            sw = 1.0 / (1.0 + n_a ** 0.5)
            compressed[a] = sw * compressed[a] + (1 - sw) * (0.5 * compressed[a] + 0.5 * nw)
        return compressed

    def _compress_video_tokens(self, feats, cls_attn, key_metric, text_tokens, nextline_token):
        nextline_token = ensure_nextline_embedding(self.get_model(), nextline_token, feats.device)
        ratio = float(os.environ.get("RETAIN_RATIO", "0.25"))
        alpha = float(os.environ.get("TEXT_ALPHA", "0.15"))
        B_max = int(os.environ.get("TEMPO_BASIS_CAP", "48"))
        F_num, T, D = feats.shape
        k = max(1, int(T * ratio))
        feats_f = feats.float()
        outputs = []

        if text_tokens is not None and text_tokens.numel() > 0:
            text_f = text_tokens.float() if text_tokens.dim() == 2 else text_tokens.unsqueeze(0).float()
            q = F.normalize(text_f.mean(0, keepdim=True), dim=-1)
        else:
            q = None

        shared_basis = []
        for f in range(F_num):
            tokens = feats[f]
            if k >= T:
                outputs.extend([tokens, nextline_token])
                continue

            tokens_f = feats_f[f]
            cond_var = (tokens_f ** 2).sum(dim=-1)
            for b in shared_basis:
                projs = tokens_f @ b
                cond_var = cond_var - projs ** 2
            cond_var.clamp_(min=0)

            selected = []
            frame_basis = list(shared_basis)
            for _ in range(k):
                for s in selected:
                    cond_var[s] = -1.0
                if q is not None:
                    text_rel = (tokens_f @ q.t()).squeeze(-1).abs()
                    text_rel = text_rel / (text_rel.max() + 1e-8)
                    cv_norm = cond_var / (cond_var.max() + 1e-8)
                    score = (1 - alpha) * cv_norm + alpha * text_rel
                    for s in selected:
                        score[s] = -1.0
                    best = score.argmax().item()
                else:
                    best = cond_var.argmax().item()
                selected.append(best)

                v = tokens_f[best].clone()
                for b in frame_basis:
                    v = v - (v @ b) * b
                vn = v.norm()
                if vn > 1e-8:
                    v = v / vn
                    frame_basis.append(v)
                    projs = tokens_f @ v
                    cond_var = cond_var - projs ** 2
                    cond_var.clamp_(min=0)

            sel_idx = torch.tensor(sorted(selected[:k]), device=tokens.device)
            compressed = self._weighted_fuse(tokens_f, sel_idx, T)
            outputs.extend([compressed.to(tokens.dtype), nextline_token])

            new_vecs = frame_basis[len(shared_basis):]
            shared_basis.extend(new_vecs)
            if len(shared_basis) > B_max:
                shared_basis = shared_basis[-B_max:]
        return outputs
