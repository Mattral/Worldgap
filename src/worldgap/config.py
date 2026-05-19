"""Configuration schemas for WorldGap, per TECHNICAL_SPEC.md Section 6/9.

All experiment configuration MUST go through these Pydantic models rather than
loose dicts or CLI-only flags, so that a config file is a complete, versioned
record of exactly what produced a given result (spec Section 12.18,
reproducibility requirements).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Modality = Literal["perception", "actuation"]


class EncoderConfig(BaseModel):
    """Per spec 6.1 / 6.2. Defaults are starting points to validate empirically,
    not fixed requirements (spec explicitly flags hidden-dim/layer counts as tunable).
    """

    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1

    @model_validator(mode="after")
    def _check_heads_divide_dmodel(self) -> "EncoderConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        return self


class WorldModelConfig(BaseModel):
    """Per spec 6.3. The JEPA-style core: context encoder + EMA target encoder + predictor."""

    context_frames: int = 16
    predict_frames: int = 8
    ema_decay: float = 0.996
    summary_dim: int = 64  # MUST stay small — see spec 7.3, sample-size-vs-dim tradeoff
    collapse_variance_threshold: float = 1e-4
    collapse_patience_checks: int = 3


class TrainingConfig(BaseModel):
    """Per spec 6.4."""

    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 100
    seed: int = 0


class GapConfig(BaseModel):
    """Top-level config passed to GapAnalyzer. One of these fully specifies a run."""

    modality: Modality
    state_dim: int = Field(
        default=258,
        description=(
            "Per-frame feature dimension. 258 for perception (33*4 pose + 21*3 + 21*3 hands, "
            "spec 5.1.1). Set explicitly for actuation — depends on DOF count modeled."
        ),
    )
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def _check_actuation_dim(self) -> "GapConfig":
        if self.modality == "actuation" and self.state_dim == 258:
            raise ValueError(
                "state_dim=258 is the perception default; set an explicit state_dim "
                "for actuation (spec 5.1.1: 2-6 depending on DOF count modeled)."
            )
        return self
