"""Shared helpers used by both encoder implementations, so the two encoders stay
interchangeable from the World Model Core's point of view (spec Section 3: only
the encoder changes between V1 and V2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


def frame_presence_from_mask(presence_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Collapses a per-channel presence mask (B, T, D) down to a per-frame
    presence indicator (B, T). A frame is 'present' if any channel in it is
    present — this is deliberately permissive; a frame with partial occlusion
    (e.g. some fingers masked) is still usable context, just noisier.
    """
    if presence_mask is None:
        return None
    if presence_mask.dim() == 2:
        return presence_mask
    return presence_mask.any(dim=-1).float()


def masked_mean_pool(h: torch.Tensor, frame_presence: torch.Tensor | None) -> torch.Tensor:
    """Mean-pools (B, T, d_model) -> (B, d_model), respecting frame presence so
    that masked/missing frames (spec edge case 12.1) don't silently drag the
    pooled summary toward zero.
    """
    if frame_presence is None:
        return h.mean(dim=1)
    weights = frame_presence.unsqueeze(-1)  # (B, T, 1)
    denom = weights.sum(dim=1).clamp(min=1e-6)
    return (h * weights).sum(dim=1) / denom
