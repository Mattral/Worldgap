"""Exponential moving average target-encoder update, per TECHNICAL_SPEC.md Section 6.3.

The target encoder MUST only ever be updated via EMA of the context (online)
encoder's weights — gradients must never flow into it directly. This is what
prevents the trivial-collapse solution where predictor and target co-adapt to
output a constant (spec 12.7).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EMA:
    def __init__(self, decay: float):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = decay

    @torch.no_grad()
    def update(self, online: nn.Module, target: nn.Module) -> None:
        online_params = dict(online.named_parameters())
        for name, target_param in target.named_parameters():
            online_param = online_params[name]
            target_param.mul_(self.decay).add_(online_param, alpha=1.0 - self.decay)
