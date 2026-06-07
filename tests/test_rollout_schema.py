import numpy as np
import pytest

from worldgap.data.rollout import (
    PERCEPTION_STATE_DIM,
    Rollout,
    validate_perception_rollout,
)


def _make_rollout(t=20, d=PERCEPTION_STATE_DIM, modality="perception"):
    rng = np.random.default_rng(0)
    return Rollout(
        modality=modality,
        source="synthetic",
        condition={"lighting": "low", "occlusion": "none"},
        frame_rate_hz=30.0,
        states=rng.normal(size=(t, d)),
        presence_mask=np.ones((t, d)),
        timestamps_ms=np.arange(t) * (1000.0 / 30.0),
    )


def test_rollout_id_is_deterministic_content_hash():
    r1 = _make_rollout()
    r2 = _make_rollout()
    # same states/condition/modality (fixed seed) -> same id, per spec 5.1
    assert r1.rollout_id == r2.rollout_id


def test_rollout_id_changes_with_condition():
    r1 = _make_rollout()
    rng = np.random.default_rng(0)
    r2 = Rollout(
        modality="perception",
        source="synthetic",
        condition={"lighting": "high", "occlusion": "none"},  # different condition
        frame_rate_hz=30.0,
        states=rng.normal(size=(20, PERCEPTION_STATE_DIM)),
        presence_mask=np.ones((20, PERCEPTION_STATE_DIM)),
        timestamps_ms=np.arange(20) * (1000.0 / 30.0),
    )
    assert r1.rollout_id != r2.rollout_id


def test_mismatched_states_and_mask_length_raises():
    with pytest.raises(ValueError):
        Rollout(
            modality="perception",
            source="synthetic",
            condition={},
            frame_rate_hz=30.0,
            states=np.zeros((10, PERCEPTION_STATE_DIM)),
            presence_mask=np.ones((9, PERCEPTION_STATE_DIM)),  # mismatched T
            timestamps_ms=np.arange(10),
        )


def test_validate_perception_rollout_rejects_wrong_dim():
    bad = _make_rollout(d=100)  # not PERCEPTION_STATE_DIM
    with pytest.raises(ValueError):
        validate_perception_rollout(bad)


def test_validate_perception_rollout_accepts_correct_dim():
    good = _make_rollout()
    validate_perception_rollout(good)  # should not raise


def test_save_and_load_roundtrip(tmp_path):
    rollout = _make_rollout()
    path = rollout.save(tmp_path)
    loaded = Rollout.load(
        path,
        modality="perception",
        source="synthetic",
        condition=rollout.condition,
        frame_rate_hz=rollout.frame_rate_hz,
    )
    np.testing.assert_allclose(loaded.states, rollout.states)
    np.testing.assert_allclose(loaded.presence_mask, rollout.presence_mask)
