"""Actuator-state-sequence encoder for actuation rollouts (V2), per
TECHNICAL_SPEC.md Section 6.2. A small TCN — SHOULD start simpler than the
perception encoder given the much lower input dimensionality; only move to a
Transformer if validation shows the TCN underfits the hysteresis dynamics.

Same forward(x, presence_mask) -> (B, T, d_model) contract as LandmarkEncoder,
so the World Model Core is genuinely interchangeable between the two (spec
Section 3's reusability claim, tested in test_modality_swap.py).

Known limitation (documented, not silently ignored): unlike the Transformer
encoder's attention mask, this TCN's convolutions can let information leak
across a masked/missing timestep from its neighbors. Masked inputs are
zeroed before convolution to reduce this, but it is not a full fix — revisit
if V2 validation shows the encoder is insensitive to injected occlusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...config import EncoderConfig
from .common import frame_presence_from_mask


class ActuatorEncoder(nn.Module):
    d_model: int

    def __init__(self, input_dim: int, config: EncoderConfig):
        super().__init__()
        self.d_model = config.d_model
        self.input_proj = nn.Linear(input_dim, config.d_model)
        self.tcn = nn.Sequential(
            nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, presence_mask: torch.Tensor | None = None) -> torch.Tensor:
        frame_presence = frame_presence_from_mask(presence_mask)
        h = self.input_proj(x)  # (B, T, d_model)
        if frame_presence is not None:
            h = h * frame_presence.unsqueeze(-1)
        h = h.transpose(1, 2)  # (B, d_model, T)
        h = self.tcn(h)
        return h.transpose(1, 2)  # (B, T, d_model)
