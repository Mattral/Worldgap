"""Tests for GapAnalyzer.save_checkpoint/load_checkpoint.

Needed so `worldgap train` and `worldgap analyze` can be separate CLI
invocations (spec 9.2) -- a fitted GapAnalyzer written by one process must be
reloadable by another and produce byte-for-byte identical gap results.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldgap import GapAnalyzer, GapConfig, Rollout
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig


def _tiny_config(modality: str = "perception", state_dim: int = 258) -> GapConfig:
    return GapConfig(
        modality=modality,
        state_dim=state_dim,
        encoder=EncoderConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32),
        world_model=WorldModelConfig(context_frames=4, predict_frames=2, summary_dim=8),
        training=TrainingConfig(max_epochs=1, batch_size=4, seed=0),
    )


def _make_rollouts(n: int, t: int, d: int, modality: str, seed: int) -> list[Rollout]:
    rng = np.random.default_rng(seed)
    return [
        Rollout(
            modality=modality,
            source="synthetic",
            condition={"idx": i},
            frame_rate_hz=30.0,
            states=rng.normal(size=(t, d)),
            presence_mask=np.ones((t, d)),
            timestamps_ms=np.arange(t) * (1000.0 / 30.0),
        )
        for i in range(n)
    ]


def test_checkpoint_roundtrip_produces_identical_gap_result(tmp_path):
    config = _tiny_config()
    analyzer = GapAnalyzer(config)
    train_rollouts = _make_rollouts(6, t=8, d=258, modality="perception", seed=1)
    analyzer.fit(train_rollouts)

    source = _make_rollouts(5, t=8, d=258, modality="perception", seed=2)
    target = _make_rollouts(5, t=8, d=258, modality="perception", seed=3)
    result_before = analyzer.compute_gap(source, target)

    ckpt_path = tmp_path / "checkpoint.pt"
    analyzer.save_checkpoint(ckpt_path)
    reloaded = GapAnalyzer.load_checkpoint(ckpt_path)

    assert reloaded.config == config
    result_after = reloaded.compute_gap(source, target)

    assert result_after.frechet.distance == pytest.approx(result_before.frechet.distance)
    assert result_after.mmd.mmd_squared == pytest.approx(result_before.mmd.mmd_squared)
    assert result_after.confidence == result_before.confidence


def test_load_checkpoint_preserves_fitted_state(tmp_path):
    config = _tiny_config()
    analyzer = GapAnalyzer(config)
    analyzer.fit(_make_rollouts(6, t=8, d=258, modality="perception", seed=1))

    ckpt_path = tmp_path / "checkpoint.pt"
    analyzer.save_checkpoint(ckpt_path)
    reloaded = GapAnalyzer.load_checkpoint(ckpt_path)

    rollouts = _make_rollouts(3, t=8, d=258, modality="perception", seed=4)
    # Should NOT raise "call fit() before compute_gap()" -- fitted state must
    # survive the round trip.
    reloaded.compute_gap(rollouts, rollouts)


def test_unfitted_analyzer_checkpoint_stays_unfitted(tmp_path):
    config = _tiny_config()
    analyzer = GapAnalyzer(config)
    ckpt_path = tmp_path / "checkpoint.pt"
    analyzer.save_checkpoint(ckpt_path)

    reloaded = GapAnalyzer.load_checkpoint(ckpt_path)
    rollouts = _make_rollouts(3, t=8, d=258, modality="perception", seed=0)
    with pytest.raises(RuntimeError):
        reloaded.compute_gap(rollouts, rollouts)
