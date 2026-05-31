"""Shared JEPA-style World Model Core, per TECHNICAL_SPEC.md Section 6.3.

This module MUST be identical regardless of modality (spec Section 3) — only the
encoder passed in via `encoder_factory` differs between perception (V1) and
actuation (V2). If a future change to support V2 or V3 requires touching this
file rather than swapping the encoder, that is a design defect to fix at the
architecture level (spec Section 15), not a per-version patch.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import WorldModelConfig
from .collapse import CollapseSafeguard
from .ema import EMA
from .encoders.common import frame_presence_from_mask, masked_mean_pool

EncoderFactory = Callable[[], nn.Module]


class WorldModel(nn.Module):
    def __init__(self, encoder_factory: EncoderFactory, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.context_encoder = encoder_factory()
        self.target_encoder = encoder_factory()
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        d_model = self.context_encoder.d_model
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.summary_head = nn.Linear(d_model, config.summary_dim)

        self._ema = EMA(config.ema_decay)
        self.collapse_safeguard = CollapseSafeguard(
            variance_threshold=config.collapse_variance_threshold,
            patience_checks=config.collapse_patience_checks,
        )

    def forward(
        self,
        context_x: torch.Tensor,
        context_mask: torch.Tensor,
        future_x: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """One JEPA training step.

        Returns:
            loss: scalar tensor to backprop through the context encoder + predictor.
            diagnostics: dict including 'latent_variance', which the caller MUST
                feed into `self.collapse_safeguard.record(...)` every step (spec 6.3
                requires this be an automated, always-on check, not opt-in).
        """
        ctx_h = self.context_encoder(context_x, context_mask)  # (B, Tc, d)
        ctx_presence = frame_presence_from_mask(context_mask)
        ctx_summary = masked_mean_pool(ctx_h, ctx_presence)  # (B, d)

        predicted = self.predictor(ctx_summary).unsqueeze(1)  # (B, 1, d)
        predicted = predicted.expand(-1, future_x.shape[1], -1)  # (B, Tp, d)

        with torch.no_grad():
            target = self.target_encoder(future_x, future_mask)  # (B, Tp, d), stop-gradient

        loss = F.smooth_l1_loss(predicted, target)
        latent_variance = target.var(dim=0).mean().item()

        return loss, {"latent_variance": latent_variance}

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """MUST be called after every optimizer step, per spec 6.3 — the target
        encoder is never updated by gradient descent directly.
        """
        self._ema.update(self.context_encoder, self.target_encoder)

    @torch.no_grad()
    def encode_rollout_summary(
        self, states: torch.Tensor, presence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encodes a full rollout to the small summary vector consumed by the
        Divergence Module (spec 6.3: summary_dim deliberately small, 32-64).
        """
        h = self.context_encoder(states, presence_mask)
        pooled = masked_mean_pool(h, frame_presence_from_mask(presence_mask))
        return self.summary_head(pooled)
