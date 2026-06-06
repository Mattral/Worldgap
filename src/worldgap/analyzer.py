"""Top-level GapAnalyzer API, per TECHNICAL_SPEC.md Section 9.1.

This is the one class both V1 (perception) and V2 (actuation) go through — only
`config.modality` changes which encoder gets built. `tests/test_modality_swap.py`
is the concrete, runnable test of the reusability claim in spec Section 3: if
that test ever requires touching this file to pass for a new modality, the
architecture has a hidden assumption that needs fixing here, not around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import GapConfig
from .data.rollout import Rollout
from .metrics.frechet import FrechetResult, frechet_distance
from .metrics.mmd import MMDResult, mmd_squared
from .models.encoders.actuator_encoder import ActuatorEncoder
from .models.encoders.landmark_encoder import LandmarkEncoder
from .models.world_model import WorldModel


@dataclass
class GapResult:
    """Per spec 9.1. `n_source`/`n_target`/`confidence` are exposed as
    top-level read-only properties (proxying `frechet`) so callers can write
    `result.confidence` as the spec's own API example shows, without this
    class duplicating storage for values `FrechetResult` already owns.
    """

    frechet: FrechetResult
    mmd: MMDResult
    warnings: list[str] = field(default_factory=list)

    @property
    def n_source(self) -> int:
        return self.frechet.n_source

    @property
    def n_target(self) -> int:
        return self.frechet.n_target

    @property
    def confidence(self) -> str:
        return self.frechet.confidence


class _WindowDataset(Dataset):
    """Slices one (context, future) window from the start of each rollout.

    Known simplification, documented rather than hidden: rollouts shorter than
    context_frames + predict_frames are skipped and counted in `.skipped`, and
    only one window per rollout is taken (no sliding-window augmentation yet).
    Both are reasonable v0.1 choices, not silent bugs — surfaced to the caller
    via GapAnalyzer.fit()'s returned dict.
    """

    def __init__(self, rollouts: list[Rollout], context_frames: int, predict_frames: int):
        self.windows: list[tuple] = []
        self.skipped = 0
        total_len = context_frames + predict_frames
        for r in rollouts:
            if r.states.shape[0] < total_len:
                self.skipped += 1
                continue
            self.windows.append(
                (
                    r.states[:context_frames],
                    r.presence_mask[:context_frames],
                    r.states[context_frames:total_len],
                    r.presence_mask[context_frames:total_len],
                )
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        ctx_x, ctx_m, fut_x, fut_m = self.windows[idx]
        return (
            torch.as_tensor(ctx_x, dtype=torch.float32),
            torch.as_tensor(ctx_m, dtype=torch.float32),
            torch.as_tensor(fut_x, dtype=torch.float32),
            torch.as_tensor(fut_m, dtype=torch.float32),
        )


class GapAnalyzer:
    def __init__(self, config: GapConfig):
        self.config = config
        self.model = WorldModel(encoder_factory=self._build_encoder, config=config.world_model)
        self.optimizer = torch.optim.AdamW(
            list(self.model.context_encoder.parameters()) + list(self.model.predictor.parameters()),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        self._fitted = False

    def _build_encoder(self) -> torch.nn.Module:
        if self.config.modality == "perception":
            return LandmarkEncoder(input_dim=self.config.state_dim, config=self.config.encoder)
        if self.config.modality == "actuation":
            return ActuatorEncoder(input_dim=self.config.state_dim, config=self.config.encoder)
        raise ValueError(f"unknown modality: {self.config.modality!r}")  # pragma: no cover — GapConfig already validates this

    def fit(self, rollouts: list[Rollout]) -> dict:
        torch.manual_seed(self.config.training.seed)
        wm_cfg = self.config.world_model
        dataset = _WindowDataset(rollouts, wm_cfg.context_frames, wm_cfg.predict_frames)
        if len(dataset) == 0:
            raise ValueError(
                "no rollouts long enough for the configured context+predict window "
                f"({wm_cfg.context_frames + wm_cfg.predict_frames} frames); "
                f"{dataset.skipped} rollouts were too short and 0 were usable."
            )
        loader = DataLoader(
            dataset, batch_size=min(self.config.training.batch_size, len(dataset)), shuffle=True
        )

        losses: list[float] = []
        for _epoch in range(self.config.training.max_epochs):
            for ctx_x, ctx_m, fut_x, fut_m in loader:
                loss, diag = self.model(ctx_x, ctx_m, fut_x, fut_m)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.update_target_encoder()
                self.model.collapse_safeguard.record(diag["latent_variance"])
                losses.append(loss.item())

        self._fitted = True
        return {
            "final_loss": losses[-1] if losses else float("nan"),
            "n_steps": len(losses),
            "n_skipped_rollouts": dataset.skipped,
            "collapsed": self.model.collapse_safeguard.has_collapsed(),
        }

    def _rollouts_to_summary_latents(self, rollouts: list[Rollout]) -> np.ndarray:
        self.model.eval()
        summaries = []
        with torch.no_grad():
            for r in rollouts:
                x = torch.as_tensor(r.states, dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(r.presence_mask, dtype=torch.float32).unsqueeze(0)
                summary = self.model.encode_rollout_summary(x, m)
                summaries.append(summary.squeeze(0).numpy())
        return np.stack(summaries)

    def compute_gap(
        self, source_rollouts: list[Rollout], target_rollouts: list[Rollout]
    ) -> GapResult:
        if not self._fitted:
            raise RuntimeError(
                "call fit() before compute_gap() — otherwise the world model has "
                "random, untrained weights and any gap number is meaningless"
            )
        source_latents = self._rollouts_to_summary_latents(source_rollouts)
        target_latents = self._rollouts_to_summary_latents(target_rollouts)

        fd = frechet_distance(source_latents, target_latents)
        mmd = mmd_squared(source_latents, target_latents)

        warnings: list[str] = []
        if fd.confidence == "low":
            warnings.append(
                f"low sample-size confidence (n_source={fd.n_source}, n_target={fd.n_target}, "
                f"latent_dim={fd.latent_dim}) — see spec Section 7.3"
            )
        return GapResult(frechet=fd, mmd=mmd, warnings=warnings)
