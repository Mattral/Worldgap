"""Per TECHNICAL_SPEC.md: 'a unit test asserting both modalities run through the
identical GapAnalyzer class and produce a GapResult of the same schema is a
MUST, not optional' (Section 9.1), and 'test_modality_swap.py — asserts V1/V2
share GapAnalyzer unmodified' (Section 10/13).

This is the single most important test in the repo: it's the difference
between 'the architecture is reusable' being a design intention versus a
checked fact.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldgap import GapAnalyzer, GapConfig, GapResult, Rollout
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig


def _tiny_config(modality: str, state_dim: int) -> GapConfig:
    # Deliberately tiny so the test runs in seconds, not minutes — this is a
    # wiring/contract test, not a convergence test.
    return GapConfig(
        modality=modality,
        state_dim=state_dim,
        encoder=EncoderConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32),
        world_model=WorldModelConfig(context_frames=4, predict_frames=2, summary_dim=8),
        training=TrainingConfig(max_epochs=1, batch_size=4, seed=0),
    )


def _make_rollouts(n: int, t: int, d: int, modality: str, seed: int) -> list[Rollout]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        states = rng.normal(size=(t, d))
        out.append(
            Rollout(
                modality=modality,
                source="synthetic",
                condition={"idx": i},
                frame_rate_hz=30.0,
                states=states,
                presence_mask=np.ones((t, d)),
                timestamps_ms=np.arange(t) * (1000.0 / 30.0),
            )
        )
    return out


@pytest.mark.parametrize(
    "modality,state_dim",
    [
        ("perception", 258),
        ("actuation", 4),
    ],
)
def test_gap_analyzer_runs_identically_across_modalities(modality, state_dim):
    """Same GapAnalyzer class, same fit()/compute_gap() call sequence, only the
    config's modality/state_dim differ — this is the reusability claim, checked.
    """
    config = _tiny_config(modality, state_dim)
    analyzer = GapAnalyzer(config)

    train_rollouts = _make_rollouts(6, t=8, d=state_dim, modality=modality, seed=1)
    fit_report = analyzer.fit(train_rollouts)
    assert fit_report["n_steps"] > 0
    assert fit_report["n_skipped_rollouts"] == 0

    source = _make_rollouts(10, t=8, d=state_dim, modality=modality, seed=2)
    target = _make_rollouts(10, t=8, d=state_dim, modality=modality, seed=3)
    result = analyzer.compute_gap(source, target)

    assert isinstance(result, GapResult)
    assert result.frechet.latent_dim == config.world_model.summary_dim
    assert result.mmd.mmd_squared >= 0.0 - 1e-6


def test_compute_gap_before_fit_raises():
    config = _tiny_config("perception", 258)
    analyzer = GapAnalyzer(config)
    rollouts = _make_rollouts(3, t=8, d=258, modality="perception", seed=0)
    with pytest.raises(RuntimeError):
        analyzer.compute_gap(rollouts, rollouts)


def test_too_short_rollouts_raise_with_clear_message():
    config = _tiny_config("perception", 258)  # needs 4+2=6 frames
    analyzer = GapAnalyzer(config)
    too_short = _make_rollouts(3, t=3, d=258, modality="perception", seed=0)
    with pytest.raises(ValueError, match="too short"):
        analyzer.fit(too_short)
