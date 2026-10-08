"""The normalization scheme is fixed per study, and GapAnalyzer enforces it.

A gap measured across two anchor schemes (or between normalized and raw
rollouts) includes the change of anchor itself, so it would silently measure
something other than the change of conditions. Same spirit as the
temporal-provenance guard: refuse, with a reason, rather than return a number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldgap import GapAnalyzer, GapConfig, Rollout
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig
from worldgap.data.normalization import normalize_rollout, require_single_scheme
from worldgap.data.rollout import PERCEPTION_STATE_DIM, TEMPORAL_PROVENANCE_KEY


def _raw(n: int, seed: int) -> list[Rollout]:
    rng = np.random.default_rng(seed)
    return [
        Rollout(
            modality="perception",
            source="real",
            condition={"i": i, "seed": seed},
            frame_rate_hz=30.0,
            states=rng.uniform(0.2, 0.8, size=(20, PERCEPTION_STATE_DIM)),
            presence_mask=np.ones((20, PERCEPTION_STATE_DIM)),
            timestamps_ms=np.arange(20) * (1000 / 30),
            metadata={TEMPORAL_PROVENANCE_KEY: "video"},
        )
        for i in range(n)
    ]


def _normed(n: int, seed: int, scheme: str) -> list[Rollout]:
    return [normalize_rollout(r, scheme) for r in _raw(n, seed)]


def _analyzer() -> GapAnalyzer:
    return GapAnalyzer(
        GapConfig(
            modality="perception",
            state_dim=PERCEPTION_STATE_DIM,
            encoder=EncoderConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32),
            world_model=WorldModelConfig(context_frames=4, predict_frames=2, summary_dim=4),
            training=TrainingConfig(max_epochs=1, batch_size=4, seed=0),
        )
    )


@pytest.fixture
def fitted_on_shoulder() -> GapAnalyzer:
    a = _analyzer()
    a.fit(_normed(6, 0, "shoulder_midpoint"))
    return a


def test_same_scheme_compares(fitted_on_shoulder):
    result = fitted_on_shoulder.compute_gap(
        _normed(6, 1, "shoulder_midpoint"), _normed(6, 2, "shoulder_midpoint")
    )
    assert np.isfinite(result.frechet.distance)


def test_compute_gap_refuses_source_and_target_with_different_schemes(fitted_on_shoulder):
    with pytest.raises(ValueError, match="different landmark normalization schemes"):
        fitted_on_shoulder.compute_gap(
            _normed(6, 1, "shoulder_midpoint"), _normed(6, 2, "hip_midpoint")
        )


def test_compute_gap_refuses_normalized_mixed_with_raw(fitted_on_shoulder):
    with pytest.raises(ValueError, match="unnormalized"):
        fitted_on_shoulder.compute_gap(_normed(6, 1, "shoulder_midpoint"), _raw(6, 2))


def test_compute_gap_refuses_a_scheme_other_than_the_training_one(fitted_on_shoulder):
    with pytest.raises(ValueError, match="trained on 'shoulder_midpoint'"):
        fitted_on_shoulder.compute_gap(_normed(6, 1, "hip_midpoint"), _normed(6, 2, "hip_midpoint"))


def test_fit_refuses_mixed_schemes():
    mixed = _normed(3, 0, "shoulder_midpoint") + _normed(3, 1, "hip_midpoint")
    with pytest.raises(ValueError, match="refusing to train on"):
        _analyzer().fit(mixed)


def test_training_scheme_survives_a_checkpoint(fitted_on_shoulder, tmp_path: Path):
    path = tmp_path / "ckpt.pt"
    fitted_on_shoulder.save_checkpoint(path)
    reloaded = GapAnalyzer.load_checkpoint(path)
    with pytest.raises(ValueError, match="trained on 'shoulder_midpoint'"):
        reloaded.compute_gap(_normed(6, 1, "hip_midpoint"), _normed(6, 2, "hip_midpoint"))
    reloaded.compute_gap(_normed(6, 1, "shoulder_midpoint"), _normed(6, 2, "shoulder_midpoint"))


def test_unnormalized_sets_report_no_scheme():
    assert require_single_scheme(_raw(3, 0), "compare") is None
    assert require_single_scheme([], "compare") is None
