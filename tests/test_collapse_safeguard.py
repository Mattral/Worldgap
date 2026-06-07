"""Per TECHNICAL_SPEC.md Section 12.7 / 13: 'test_collapse_safeguard.py' MUST
exist and the safeguard MUST be an automated check, not a manual inspection.

Note on test design: reliably forcing an actual JEPA representation collapse
within a handful of training steps is not something you can guarantee
deterministically (that's rather the point — collapse is a failure mode you
detect, not one you can script on demand). So this file tests two things
separately, both required:
  1. The CollapseSafeguard logic itself, directly and deterministically.
  2. That GapAnalyzer.fit() actually wires latent-variance recording into the
     safeguard end-to-end (i.e. the plumbing works), without asserting
     collapse occurs.
"""

from __future__ import annotations

import numpy as np

from worldgap import GapAnalyzer, GapConfig, Rollout
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig
from worldgap.models.collapse import CollapseSafeguard


def test_safeguard_flags_sustained_low_variance():
    guard = CollapseSafeguard(variance_threshold=1e-3, patience_checks=3)
    for v in [1e-4, 1e-4, 1e-4]:
        guard.record(v)
    assert guard.has_collapsed() is True


def test_safeguard_does_not_flag_a_single_dip():
    guard = CollapseSafeguard(variance_threshold=1e-3, patience_checks=3)
    guard.record(1e-4)  # one low reading
    guard.record(0.5)  # recovers
    guard.record(0.4)
    assert guard.has_collapsed() is False


def test_safeguard_requires_enough_history_before_flagging():
    guard = CollapseSafeguard(variance_threshold=1e-3, patience_checks=5)
    guard.record(1e-4)
    guard.record(1e-4)
    assert guard.has_collapsed() is False  # only 2 of 5 required checks recorded


def test_fit_wires_variance_into_collapse_safeguard():
    config = GapConfig(
        modality="perception",
        state_dim=32,
        encoder=EncoderConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32),
        world_model=WorldModelConfig(context_frames=4, predict_frames=2, summary_dim=8),
        training=TrainingConfig(max_epochs=2, batch_size=4, seed=0),
    )
    analyzer = GapAnalyzer(config)

    rng = np.random.default_rng(0)
    rollouts = [
        Rollout(
            modality="perception",
            source="synthetic",
            condition={"idx": i},
            frame_rate_hz=30.0,
            states=rng.normal(size=(8, 32)),
            presence_mask=np.ones((8, 32)),
            timestamps_ms=np.arange(8) * (1000.0 / 30.0),
        )
        for i in range(6)
    ]

    analyzer.fit(rollouts)
    # the plumbing worked iff the safeguard actually received readings
    assert len(analyzer.model.collapse_safeguard.history) > 0
