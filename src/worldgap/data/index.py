"""SQLite metadata index, per TECHNICAL_SPEC.md Section 5.3.

`Rollout.save()` only persists the array data (states, presence_mask,
timestamps_ms) to a `.npz` file, and `Rollout.load()` requires the caller to
already know `modality`/`source`/`condition`/`frame_rate_hz`/`metadata` out of
band. Without an index, "load every rollout in a directory" isn't actually
possible from the `.npz` files alone. This module is that index: one row per
rollout, keyed by `rollout_id`, holding exactly the non-array fields
`Rollout.load()` needs -- so `RolloutIndex.load_all()` can reconstruct full
`Rollout` objects from a bare directory plus this one file, per the spec 5.3
directory convention (`data/index.db` alongside `data/processed/{modality}/`).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .rollout import Rollout

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id     TEXT PRIMARY KEY,
    modality       TEXT NOT NULL,
    source         TEXT NOT NULL,
    frame_rate_hz  REAL NOT NULL,
    n_frames       INTEGER NOT NULL,
    condition_json TEXT NOT NULL,
    metadata_json  TEXT NOT NULL
);
"""


class RolloutIndex:
    """Thin wrapper around a single-table SQLite index. Not an ORM -- spec 5.3
    only asks for "one row per rollout, columns matching the non-array
    fields," so a hand-rolled schema is the right amount of machinery here.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RolloutIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add(self, rollout: Rollout) -> None:
        """Upserts one rollout's metadata. MUST be called alongside
        `rollout.save(processed_dir)` -- the index and the `.npz` files are
        two halves of one storage layout, not independent (spec 5.3).
        """
        self._conn.execute(
            """
            INSERT INTO rollouts
                (rollout_id, modality, source, frame_rate_hz, n_frames, condition_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(rollout_id) DO UPDATE SET
                modality=excluded.modality,
                source=excluded.source,
                frame_rate_hz=excluded.frame_rate_hz,
                n_frames=excluded.n_frames,
                condition_json=excluded.condition_json,
                metadata_json=excluded.metadata_json
            """,
            (
                rollout.rollout_id,
                rollout.modality,
                rollout.source,
                rollout.frame_rate_hz,
                rollout.states.shape[0],
                json.dumps(rollout.condition, sort_keys=True),
                json.dumps(rollout.metadata, sort_keys=True),
            ),
        )
        self._conn.commit()

    def get(self, rollout_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT rollout_id, modality, source, frame_rate_hz, n_frames, condition_json, metadata_json "
            "FROM rollouts WHERE rollout_id = ?",
            (rollout_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list(self, modality: str | None = None) -> list[dict]:
        """Returns metadata rows, optionally filtered by modality. Used both
        for CLI directory-loading and for building the validation harness's
        pre-registered condition table (spec 8.3/8.4).
        """
        if modality is None:
            rows = self._conn.execute(
                "SELECT rollout_id, modality, source, frame_rate_hz, n_frames, condition_json, metadata_json "
                "FROM rollouts ORDER BY rollout_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT rollout_id, modality, source, frame_rate_hz, n_frames, condition_json, metadata_json "
                "FROM rollouts WHERE modality = ? ORDER BY rollout_id",
                (modality,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: tuple) -> dict:
        rollout_id, modality, source, frame_rate_hz, n_frames, condition_json, metadata_json = row
        return {
            "rollout_id": rollout_id,
            "modality": modality,
            "source": source,
            "frame_rate_hz": frame_rate_hz,
            "n_frames": n_frames,
            "condition": json.loads(condition_json),
            "metadata": json.loads(metadata_json),
        }

    def load_rollout(self, rollout_id: str, processed_dir: str | Path) -> Rollout:
        """Reconstructs a full Rollout (arrays + metadata) from the `.npz`
        file plus this index -- the round trip `Rollout.load()` alone can't do
        without the caller separately supplying condition/source/frame_rate.
        """
        meta = self.get(rollout_id)
        if meta is None:
            raise KeyError(f"no index entry for rollout_id={rollout_id!r}")
        path = Path(processed_dir) / meta["modality"] / f"{rollout_id}.npz"
        return Rollout.load(
            path,
            modality=meta["modality"],
            source=meta["source"],
            condition=meta["condition"],
            frame_rate_hz=meta["frame_rate_hz"],
            metadata=meta["metadata"],
        )

    def load_all(self, processed_dir: str | Path, modality: str | None = None) -> list[Rollout]:
        """Reconstructs every indexed rollout (optionally filtered by
        modality) from a bare `processed_dir` -- this is what makes
        `worldgap train --data-dir ...` and `worldgap analyze --source ...`
        possible without the CLI needing to know per-rollout metadata itself.
        """
        return [
            self.load_rollout(row["rollout_id"], processed_dir)
            for row in self.list(modality=modality)
        ]
