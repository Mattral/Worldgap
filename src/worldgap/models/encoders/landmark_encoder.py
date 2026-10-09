"""Landmark-sequence encoder for perception rollouts (V1), per TECHNICAL_SPEC.md
Section 6.1. A Transformer encoder — appropriate at this scale since inputs are
~258-dim landmark vectors per frame, not raw pixels.
"""

from __future__ import annotations

import torch
from torch import nn

from ...config import EncoderConfig
from .common import SinusoidalPositionalEncoding, frame_presence_from_mask


class LandmarkEncoder(nn.Module):
    d_model: int

    def __init__(self, input_dim: int, config: EncoderConfig):
        super().__init__()
        self.d_model = config.d_model
        self.input_proj = nn.Linear(input_dim, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.n_layers)

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            presence_mask: (B, T, input_dim) or (B, T) — per spec 5.1, missing
                frames/landmarks MUST be indicated via this mask, not by zero-filling
                and pretending presence (edge case 12.1).
        Returns:
            (B, T, d_model) per-frame embeddings.
        """
        frame_presence = frame_presence_from_mask(presence_mask)
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        key_padding_mask = None
        if frame_presence is not None:
            key_padding_mask = ~frame_presence.bool()  # True = ignore, per torch convention
            # A window where MediaPipe detected nothing in ANY frame would be
            # fully masked, which attention cannot handle (NaN in training mode;
            # a crash in PyTorch's inference fast path). Unmask such windows:
            # their frame outputs are discarded by masked_mean_pool anyway, so
            # every empty window still maps to the same "nothing detected"
            # summary. Found by the run-2 blind test, where the harshest
            # degradations lost the whole body for 48 frames.
            fully_missing = key_padding_mask.all(dim=1)
            if fully_missing.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[fully_missing] = False
        return self.transformer(h, src_key_padding_mask=key_padding_mask)
