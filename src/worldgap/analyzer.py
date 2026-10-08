"""Top-level GapAnalyzer API, per TECHNICAL_SPEC.md Section 9.1.

This is the one class both V1 (perception) and V2 (actuation) go through — only
`config.modality` changes which encoder gets built. `tests/test_modality_swap.py`
is the concrete, runnable test of the reusability claim in spec Section 3: if
that test ever requires touching this file to pass for a new modality, the
architecture has a hidden assumption that needs fixing here, not around it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import GapConfig
from .data.normalization import REMEDY as NORMALIZATION_REMEDY
from .data.normalization import require_single_scheme
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
        # Seed BEFORE the model is constructed, not only in fit().
        #
        # This was a real reproducibility bug up to 0.1.0: `fit()` called
        # torch.manual_seed(), but by then every encoder weight and the
        # predictor had already been initialized from whatever global RNG state
        # happened to exist when GapAnalyzer was constructed. The seed therefore
        # controlled only batch shuffling and dropout, and two runs of the same
        # script with the same config produced different gap scores -- which
        # makes a config file exactly not "a complete record of what produced a
        # given result" (spec 12.18). Caught by running the V2 script twice.
        torch.manual_seed(config.training.seed)
        self.model = WorldModel(encoder_factory=self._build_encoder, config=config.world_model)
        # What is and isn't optimized here, spelled out because it is not
        # obvious from the parameter list alone:
        #
        # - `context_encoder`, `predictor` and `future_mask_tokens` all receive
        #   gradient from the JEPA loss and MUST be in this list. The mask
        #   tokens in particular: leaving them out (as an earlier version of
        #   this line did, before they existed) would freeze them at their
        #   random init, so per-future-frame predictions would differ from each
        #   other arbitrarily rather than in a learned way.
        # - `target_encoder` is deliberately absent: spec 6.3 requires it be
        #   updated only by EMA, never by gradient descent.
        # - `summary_head` is also absent, and this is a real property of the
        #   design worth knowing: it is only ever applied inside
        #   `encode_rollout_summary()`, which runs under `torch.no_grad()`, so
        #   no gradient ever reaches it and it stays a FIXED RANDOM LINEAR
        #   PROJECTION from d_model to summary_dim. That is defensible (a
        #   random projection approximately preserves relative distances, which
        #   is all the divergence metrics need) but it is a design choice, not
        #   an accident, and it means the summary space is not learned. See
        #   docs/TECHNICAL_SPEC.md Section 6.3's implementation note.
        self.optimizer = torch.optim.AdamW(
            list(self.model.context_encoder.parameters())
            + list(self.model.predictor.parameters())
            + [self.model.future_mask_tokens],
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        self._fitted = False
        # Landmark normalization scheme the model was trained on (None = the
        # training rollouts were not normalized, e.g. actuation). Unknown only
        # for checkpoints written before this was recorded.
        self._normalization_scheme: str | None = None
        self._normalization_scheme_known = False

    def _build_encoder(self) -> torch.nn.Module:
        if self.config.modality == "perception":
            return LandmarkEncoder(input_dim=self.config.state_dim, config=self.config.encoder)
        if self.config.modality == "actuation":
            return ActuatorEncoder(input_dim=self.config.state_dim, config=self.config.encoder)
        raise ValueError(f"unknown modality: {self.config.modality!r}")  # pragma: no cover — GapConfig already validates this

    def fit(self, rollouts: list[Rollout]) -> dict:
        # One normalization scheme per study (worldgap.data.normalization):
        # a model trained across anchors would learn the anchor change.
        scheme = require_single_scheme(rollouts, "train on")
        torch.manual_seed(self.config.training.seed)
        wm_cfg = self.config.world_model
        dataset = _WindowDataset(rollouts, wm_cfg.context_frames, wm_cfg.predict_frames)
        if len(dataset) == 0:
            static = sum(1 for r in rollouts if r.temporal_provenance == "static_pose")
            hint = ""
            if static:
                hint = (
                    f" {static} of {len(rollouts)} are temporal_provenance='static_pose' "
                    "(single still images, T=1). A JEPA objective predicts a future "
                    "window from a context window; a photograph has neither. Fit on "
                    "video rollouts and use the fitted model to encode these, or use "
                    "the non-temporal comparison path -- see "
                    "docs/temporal_provenance.md."
                )
            raise ValueError(
                "no rollouts long enough for the configured context+predict window "
                f"({wm_cfg.context_frames + wm_cfg.predict_frames} frames); "
                f"{dataset.skipped} rollouts were too short and 0 were usable.{hint}"
            )

        # Temporal-provenance guard. Deliberately asymmetric, because the two
        # cases deserve different treatment:
        #
        #  - A rollout that *declares* a non-temporal provenance is a caller
        #    stating, in writing, that its frame ordering is not observation
        #    order. Training a next-window predictor on that is modelling an
        #    artifact, so it raises.
        #  - A rollout that declares nothing is legacy or hand-constructed. It
        #    warns rather than raising: this field arrived after 0.1.0, and
        #    breaking every existing caller to enforce a metadata key would be
        #    a worse trade than saying so loudly.
        declared_unordered = [
            r
            for r in rollouts
            if r.temporal_provenance is not None and not r.has_ordered_time_axis
        ]
        if declared_unordered and len(declared_unordered) == len(rollouts):
            provenances = sorted({str(r.temporal_provenance) for r in rollouts})
            raise ValueError(
                "every rollout passed to fit() declares a time axis that does not "
                f"reflect real sequential observation (temporal_provenance in "
                f"{provenances}). The world model's objective is predicting how a "
                "trajectory continues, so training it on ordering that isn't temporal "
                "produces a model of an artifact. Fit on video (or simulated) "
                "rollouts -- see docs/temporal_provenance.md."
            )
        if not any(r.temporal_provenance for r in rollouts):
            warnings.warn(
                "none of these rollouts declare metadata['temporal_provenance'], so "
                "worldgap cannot tell whether their frame ordering is real observation "
                "order. If they came from a video or a simulator, set it to 'video' or "
                "'simulated'; if they are unrelated stills, they should not be trained "
                "on at all. See docs/temporal_provenance.md.",
                UserWarning,
                stacklevel=2,
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
        self._normalization_scheme = scheme
        self._normalization_scheme_known = True
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
        # Same spirit as the temporal-provenance guard: refuse rather than
        # return a number that silently measures something else. A gap across
        # normalization schemes includes the change of anchor itself.
        scheme = require_single_scheme(
            list(source_rollouts) + list(target_rollouts), "compare"
        )
        if self._normalization_scheme_known and scheme != self._normalization_scheme:
            raise ValueError(
                f"refusing to compare rollouts with normalization scheme "
                f"{scheme or 'unnormalized'!r}: this model was trained on "
                f"{self._normalization_scheme or 'unnormalized'!r}. The normalization "
                f"scheme is fixed per study. {NORMALIZATION_REMEDY}"
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
        if mmd.below_noise_floor:
            warnings.append(
                f"MMD² is negative ({mmd.mmd_squared:.6f}). This is the unbiased "
                "estimator behaving correctly, not a bug: it means no difference "
                "between these domains is detectable at this sample size. Do not rank "
                "conditions by the magnitude of a negative MMD² — see spec Section 7.2."
            )
        return GapResult(frechet=fd, mmd=mmd, warnings=warnings)

    # -- Persistence -----------------------------------------------------
    #
    # Spec 9.2's CLI splits `train` and `analyze` into separate commands, which
    # only makes sense if a fitted GapAnalyzer can be written to disk by one
    # process and reloaded by another. Neither the CLI example nor the rest of
    # Section 9 spells out a checkpoint format, so this is a documented
    # implementation decision, not a literal spec requirement: a single
    # torch.save() of the model/optimizer state plus the GapConfig needed to
    # reconstruct this object, since GapConfig (Pydantic) is picklable.

    def save_checkpoint(self, path: str | Path) -> None:
        """Writes model + optimizer state and the config to a single file."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "fitted": self._fitted,
                "normalization_scheme": self._normalization_scheme,
                "normalization_scheme_known": self._normalization_scheme_known,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> GapAnalyzer:
        """Reconstructs a GapAnalyzer from a checkpoint written by
        `save_checkpoint`. Loads with `weights_only=False` since the
        checkpoint intentionally carries a GapConfig object, not just
        tensors — only load checkpoints you trust, same as any pickle-backed
        format.
        """
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        analyzer = cls(checkpoint["config"])
        analyzer.model.load_state_dict(checkpoint["model_state_dict"])
        analyzer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        analyzer._fitted = checkpoint["fitted"]
        # absent in checkpoints written before the scheme was recorded
        analyzer._normalization_scheme = checkpoint.get("normalization_scheme")
        analyzer._normalization_scheme_known = checkpoint.get("normalization_scheme_known", False)
        return analyzer
