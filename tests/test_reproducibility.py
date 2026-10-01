"""Regression guard for spec 12.18: a config must fully determine a result.

This was a real bug up to 0.1.0. `GapAnalyzer.fit()` called
`torch.manual_seed(config.training.seed)`, but `GapAnalyzer.__init__` had
already built the encoders and predictor by then, from whatever global RNG
state happened to be current. The seed controlled batch shuffling and dropout
and nothing else, so two runs of the same script with the same config returned
different gap scores.

It surfaced by running `scripts/run_v2_actuation.py` twice and noticing the
numbers move. These tests make sure it cannot come back quietly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from worldgap import GapAnalyzer, GapConfig, Rollout
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig


def _config(seed: int = 0) -> GapConfig:
    return GapConfig(
        modality="actuation",
        state_dim=3,
        encoder=EncoderConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32),
        world_model=WorldModelConfig(context_frames=6, predict_frames=3, summary_dim=4),
        training=TrainingConfig(max_epochs=3, batch_size=4, seed=seed),
    )


def _rollouts(n: int = 8, t: int = 20, offset: float = 0.0) -> list[Rollout]:
    rng = np.random.default_rng(123)
    out = []
    for i in range(n):
        base = np.linspace(0, 1, t)[:, None] * np.ones((1, 3))
        states = base + offset + rng.normal(0, 0.01, size=(t, 3))
        out.append(
            Rollout(
                modality="actuation",
                source="sim",
                condition={"i": i},
                frame_rate_hz=1.0,
                states=states,
                presence_mask=np.ones((t, 3)),
                timestamps_ms=np.arange(t) * 1000.0,
                metadata={"temporal_provenance": "simulated"},
            )
        )
    return out


def test_same_config_gives_bit_identical_initial_weights():
    """The specific failure: init happened before the seed was applied."""
    a = GapAnalyzer(_config())
    b = GapAnalyzer(_config())
    for pa, pb in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_construction_is_not_affected_by_ambient_rng_state():
    """Constructing something else first must not change the model."""
    a = GapAnalyzer(_config())
    torch.randn(1000)  # disturb the global RNG
    _ = GapAnalyzer(_config())
    torch.randn(37)
    c = GapAnalyzer(_config())
    for pa, pc in zip(a.model.parameters(), c.model.parameters(), strict=True):
        assert torch.equal(pa, pc)


def test_same_config_gives_identical_gap_scores_end_to_end():
    train = _rollouts()
    source = _rollouts(offset=0.0)
    target = _rollouts(offset=0.4)

    results = []
    for _ in range(2):
        analyzer = GapAnalyzer(_config())
        analyzer.fit(train)
        r = analyzer.compute_gap(source, target)
        results.append((r.frechet.distance, r.mmd.mmd_squared))

    assert results[0] == pytest.approx(results[1], rel=0, abs=0)


def test_different_seeds_actually_do_something():
    """Guards the opposite mistake: a seed that is ignored entirely would also
    make the test above pass.
    """
    a = GapAnalyzer(_config(seed=0))
    b = GapAnalyzer(_config(seed=1))
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(a.model.parameters(), b.model.parameters(), strict=True)
    )
