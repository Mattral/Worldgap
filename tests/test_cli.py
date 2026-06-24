"""End-to-end CLI smoke tests, per TECHNICAL_SPEC.md Section 9.2.

Runs `worldgap train` / `analyze` / `validate` as subprocess-free calls to
`main()` directly (faster than spawning a real subprocess, and still
exercises the real argparse wiring), against small local rollout stores built
entirely from synthetic data -- no network/Kaggle/GPU access needed.
"""

from __future__ import annotations

import numpy as np
import yaml

from worldgap.cli import main
from worldgap.data.index import RolloutIndex
from worldgap.data.rollout import Rollout

_STATE_DIM = 12  # small, not the real 258 -- this is a wiring test, not a convergence test


def _build_store(store_dir, n=6, t=10, seed=0, mean_shift=0.0):
    rng = np.random.default_rng(seed)
    with RolloutIndex(store_dir / "index.db") as index:
        for i in range(n):
            r = Rollout(
                modality="perception",
                source="synthetic",
                condition={"idx": i},
                frame_rate_hz=30.0,
                states=rng.normal(size=(t, _STATE_DIM)) + mean_shift,
                presence_mask=np.ones((t, _STATE_DIM)),
                timestamps_ms=np.arange(t) * (1000.0 / 30.0),
            )
            r.save(store_dir)
            index.add(r)


def _tiny_config_yaml(path):
    path.write_text(
        yaml.safe_dump(
            {
                "state_dim": _STATE_DIM,
                "encoder": {"d_model": 8, "n_layers": 1, "n_heads": 2, "dim_feedforward": 16},
                "world_model": {"context_frames": 4, "predict_frames": 2, "summary_dim": 4},
                "training": {"max_epochs": 1, "batch_size": 4, "seed": 0},
            }
        )
    )


def test_train_then_analyze_end_to_end(tmp_path, capsys):
    train_dir = tmp_path / "train_store"
    source_dir = tmp_path / "clean"
    target_dir = tmp_path / "perturbed"
    for d, shift in [(train_dir, 0.0), (source_dir, 0.0), (target_dir, 2.0)]:
        _build_store(d, seed=hash(str(d)) % 1000)

    config_path = tmp_path / "config.yaml"
    _tiny_config_yaml(config_path)

    train_exit = main(
        ["train", "--modality", "perception", "--data-dir", str(train_dir), "--config", str(config_path)]
    )
    assert train_exit == 0
    checkpoint_path = train_dir / "checkpoint_perception.pt"
    assert checkpoint_path.exists()

    report_path = tmp_path / "report.html"
    analyze_exit = main(
        [
            "analyze",
            "--source",
            str(source_dir),
            "--target",
            str(target_dir),
            "--modality",
            "perception",
            "--output",
            str(report_path),
            "--checkpoint",
            str(checkpoint_path),
        ]
    )
    assert analyze_exit == 0
    assert report_path.exists()
    assert "<table>" in report_path.read_text()


def test_analyze_without_checkpoint_falls_back_to_fresh_fit(tmp_path):
    source_dir = tmp_path / "clean"
    target_dir = tmp_path / "perturbed"
    _build_store(source_dir, seed=1)
    _build_store(target_dir, seed=2, mean_shift=1.0)

    config_path = tmp_path / "config.yaml"
    _tiny_config_yaml(config_path)

    exit_code = main(
        [
            "analyze",
            "--source",
            str(source_dir),
            "--target",
            str(target_dir),
            "--modality",
            "perception",
            "--output",
            str(tmp_path / "report.md"),
            "--config",
            str(config_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "report.md").exists()


def test_train_on_empty_store_gives_actionable_error(tmp_path, capsys):
    empty_dir = tmp_path / "empty_store"
    empty_dir.mkdir()
    with RolloutIndex(empty_dir / "index.db"):
        pass  # creates an empty but valid index

    exit_code = main(["train", "--modality", "perception", "--data-dir", str(empty_dir)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "no rollouts" in captured.err


def test_train_on_missing_store_gives_actionable_error(tmp_path, capsys):
    exit_code = main(["train", "--modality", "perception", "--data-dir", str(tmp_path / "nonexistent")])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "index.db" in captured.err


def test_validate_end_to_end(tmp_path, capsys):
    conditions = [{"lighting": f"c{i}", "occlusion": "none"} for i in range(10)]
    gap_scores = np.linspace(0.1, 1.0, 10)
    ground_truth = gap_scores * 2 + 0.01  # near-perfect monotonic relationship

    gap_csv = tmp_path / "gap_scores.csv"
    truth_csv = tmp_path / "ground_truth.csv"
    with open(gap_csv, "w") as f:
        f.write("lighting,occlusion,gap_score\n")
        for c, g in zip(conditions, gap_scores):
            f.write(f"{c['lighting']},{c['occlusion']},{g}\n")
    with open(truth_csv, "w") as f:
        f.write("lighting,occlusion,ground_truth_degradation\n")
        for c, g in zip(conditions, ground_truth):
            f.write(f"{c['lighting']},{c['occlusion']},{g}\n")

    exit_code = main(["validate", "--gap-scores", str(gap_csv), "--ground-truth", str(truth_csv)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Spearman rho=" in captured.err


def test_validate_mismatched_conditions_raises(tmp_path):
    gap_csv = tmp_path / "gap_scores.csv"
    truth_csv = tmp_path / "ground_truth.csv"
    gap_csv.write_text("lighting,gap_score\n" + "\n".join(f"c{i},0.{i}" for i in range(10)))
    # ground truth is missing condition c9
    truth_csv.write_text("lighting,ground_truth_degradation\n" + "\n".join(f"c{i},0.{i}" for i in range(9)))

    exit_code = main(["validate", "--gap-scores", str(gap_csv), "--ground-truth", str(truth_csv)])
    assert exit_code == 1
