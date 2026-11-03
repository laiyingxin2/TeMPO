from __future__ import annotations

import inspect
from typing import Callable, Optional

import torch

from .common import CompressionResult


def ensure_nextline_embedding(model, nextline_token: torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(nextline_token, torch.Tensor) and nextline_token.dim() == 1 and nextline_token.numel() == 1:
        return model.embed_tokens(nextline_token.to(device))
    return nextline_token.to(device)


def frame_attention_to_4d(frame_attention: Optional[torch.Tensor], num_tokens: int, device: torch.device) -> torch.Tensor:
    if frame_attention is None:
        return torch.ones(1, 1, 1, num_tokens, device=device) / max(1, num_tokens)
    attention = frame_attention.to(device)
    if attention.dim() == 1:
        attention = attention.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif attention.dim() == 2:
        attention = attention.unsqueeze(1).unsqueeze(1)
    elif attention.dim() == 3:
        attention = attention.unsqueeze(1)
    return attention


def compress_video_frames(
    compress_fn: Callable,
    feats: torch.Tensor,
    cls_attn: Optional[torch.Tensor],
    query_tokens: Optional[torch.Tensor],
    nextline_token: torch.Tensor,
    retention_ratio: float,
) -> list[torch.Tensor]:
    device = feats.device
    outputs: list[torch.Tensor] = []
    for frame_idx in range(feats.shape[0]):
        frame_tokens = feats[frame_idx : frame_idx + 1]
        frame_attn = None if cls_attn is None else cls_attn[frame_idx : frame_idx + 1]
        attention_4d = frame_attention_to_4d(frame_attn, frame_tokens.shape[1], device)
        query = None if query_tokens is None else query_tokens.unsqueeze(0)
        if hasattr(compress_fn, 'compress'):
            compress_method = compress_fn.compress
            query_arg = query if query is not None else frame_tokens.mean(dim=1, keepdim=True)
            try:
                retained_tokens, info = compress_method(frame_tokens, attention_4d, query_arg, retention_ratio)
            except TypeError:
                retained_tokens, info = compress_method(frame_tokens, attention_4d, retention_ratio)

            retained_indices = info.get('selected_indices')
            if retained_indices is None:
                retained_indices = torch.arange(retained_tokens.shape[1], device=retained_tokens.device).unsqueeze(0)
            selected_mask = info.get('selected_mask')
            if selected_mask is None:
                selected_mask = torch.zeros(frame_tokens.shape[0], frame_tokens.shape[1], dtype=torch.bool, device=frame_tokens.device)
                selected_mask.scatter_(1, retained_indices, True)
            importance_scores = info.get('final_scores', info.get('d_scores', info.get('adjusted_importance', info.get('risk', torch.zeros(frame_tokens.shape[0], frame_tokens.shape[1], device=frame_tokens.device)))))
            result = CompressionResult(
                retained_tokens=retained_tokens,
                retained_indices=retained_indices,
                selected_mask=selected_mask,
                importance_scores=importance_scores,
                method_name=compress_fn.__class__.__name__,
                retention_ratio=retention_ratio,
                extra_info=info,
            )
        else:
            result = compress_fn(frame_tokens, attention_4d, query, retention_ratio)
        outputs.append(result.retained_tokens.squeeze(0))
        outputs.append(nextline_token)
    return outputs
