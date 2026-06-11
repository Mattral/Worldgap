"""Tests for data/index.py, per TECHNICAL_SPEC.md Section 5.3."""

from __future__ import annotations

import numpy as np
import pytest

from worldgap.data.index import RolloutIndex
from worldgap.data.rollout import Rollout


def _rollout(condition_tag="clean", t=20, d=6, source="synthetic"):
    rng = np.random.default_rng(0)
    return Rollout(
        modality="perception",
        source=source,
        condition={"lighting": condition_tag},
        frame_rate_hz=30.0,
        states=rng.normal(size=(t, d)),
        presence_mask=np.ones((t, d)),
        timestamps_ms=np.arange(t) * (1000.0 / 30.0),
        metadata={"subject_id": "s01"},
    )


def test_add_and_load_rollout_roundtrip(tmp_path):
    processed_dir = tmp_path / "processed"
    r = _rollout()
    r.save(processed_dir)

    with RolloutIndex(tmp_path / "index.db") as index:
        index.add(r)
        loaded = index.load_rollout(r.rollout_id, processed_dir)

    assert loaded.rollout_id == r.rollout_id
    assert loaded.modality == r.modality
    assert loaded.source == r.source
    assert loaded.condition == r.condition
    assert loaded.metadata == r.metadata
    assert loaded.frame_rate_hz == r.frame_rate_hz
    np.testing.assert_array_equal(loaded.states, r.states)
    np.testing.assert_array_equal(loaded.presence_mask, r.presence_mask)
    np.testing.assert_array_equal(loaded.timestamps_ms, r.timestamps_ms)


def test_load_all_returns_every_indexed_rollout(tmp_path):
    processed_dir = tmp_path / "processed"
    rollouts = [_rollout(condition_tag=f"cond_{i}") for i in range(4)]
    with RolloutIndex(tmp_path / "index.db") as index:
        for r in rollouts:
            r.save(processed_dir)
            index.add(r)
        loaded = index.load_all(processed_dir)

    assert {r.rollout_id for r in loaded} == {r.rollout_id for r in rollouts}


def test_list_filters_by_modality(tmp_path):
    processed_dir = tmp_path / "processed"
    perception_r = _rollout()
    actuation_r = Rollout(
        modality="actuation",
        source="synthetic",
        condition={"unit": "pgm_01"},
        frame_rate_hz=100.0,
        states=np.random.default_rng(1).normal(size=(10, 2)),
        presence_mask=np.ones((10, 2)),
        timestamps_ms=np.arange(10) * 10.0,
    )
    with RolloutIndex(tmp_path / "index.db") as index:
        for r in (perception_r, actuation_r):
            r.save(processed_dir)
            index.add(r)

        perception_only = index.list(modality="perception")
        actuation_only = index.list(modality="actuation")

    assert [row["rollout_id"] for row in perception_only] == [perception_r.rollout_id]
    assert [row["rollout_id"] for row in actuation_only] == [actuation_r.rollout_id]


def test_add_upserts_rather_than_duplicates(tmp_path):
    processed_dir = tmp_path / "processed"
    r = _rollout()
    r.save(processed_dir)
    with RolloutIndex(tmp_path / "index.db") as index:
        index.add(r)
        index.add(r)  # same rollout_id again -- must update, not duplicate
        rows = index.list(modality="perception")

    assert len(rows) == 1


def test_load_rollout_missing_id_raises_keyerror(tmp_path):
    with RolloutIndex(tmp_path / "index.db") as index:
        with pytest.raises(KeyError):
            index.load_rollout("does_not_exist", tmp_path / "processed")


def test_index_persists_across_reopen(tmp_path):
    processed_dir = tmp_path / "processed"
    r = _rollout()
    r.save(processed_dir)
    db_path = tmp_path / "index.db"

    index = RolloutIndex(db_path)
    index.add(r)
    index.close()

    reopened = RolloutIndex(db_path)
    loaded = reopened.load_rollout(r.rollout_id, processed_dir)
    reopened.close()

    np.testing.assert_array_equal(loaded.states, r.states)
