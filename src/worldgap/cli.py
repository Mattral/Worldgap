"""CLI entry point, per TECHNICAL_SPEC.md Section 9.2.

    worldgap train    --modality perception --data-dir ./data/processed --config ...
    worldgap analyze  --source ... --target ... --modality ... --output report.html
    worldgap validate --gap-scores results.csv --ground-truth degradation.csv

v0.2 status: all three subcommands are wired end-to-end against local rollout
stores (see `_load_rollout_store` below) -- no network/Kaggle/GPU access is
needed to run this against your own .npz + index.db data. What's still out of
scope here: producing that data in the first place from HaGRID/EgoHands raw
frames (blocked on MediaPipe + real downloads, see data/loaders/hagrid.py).

**Documented decision** (spec 5.3's diagram shows one repo-wide `data/`
root with a single `index.db` shared across both modalities' `processed/`
subdirectories). This CLI instead treats each of `--data-dir`/`--source`/
`--target` as an independent, self-contained rollout store:

    {dir}/
      index.db
      perception/{rollout_id}.npz
      actuation/{rollout_id}.npz

This is what lets `analyze` compare two independently-produced stores (e.g.
one directory for "clean" rollouts, another for "perturbed" ones) without
requiring them to share one index -- the shared-index diagram in 5.3 remains
the right layout for the primary download+preprocess script's own output, but
`--source`/`--target` need to be comparable regardless of where each came
from. The spec's inline CLI example (`--data-dir ./data/processed/perception`)
has been corrected in docs/TECHNICAL_SPEC.md to `--data-dir ./data/processed`
to match: `--data-dir` is the store root, not a modality-specific subdirectory
(modality is still selected via `--modality`, filtering within that store).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import GapAnalyzer
from .config import GapConfig
from .data.index import RolloutIndex
from .report import ReportEntry, generate_report
from .validation.harness import ConditionResult, ValidationHarness


def _condition_key(condition: dict) -> tuple:
    return tuple(sorted(condition.items()))


def _load_rollout_store(store_dir: Path, modality: str, index_db: Path | None) -> list:
    """Loads every rollout for `modality` out of a self-contained store
    directory (see module docstring). Fails with an actionable message rather
    than a bare empty list or a raw sqlite/OS error -- an empty rollout store
    is almost always a setup mistake, not a valid input.
    """
    db_path = index_db if index_db is not None else store_dir / "index.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"no index.db found at {db_path} -- a rollout store needs both "
            f"'{{store}}/index.db' and '{{store}}/{{modality}}/*.npz' "
            "(save rollouts with Rollout.save() + RolloutIndex.add(), see "
            "tests/test_index.py for the exact pattern)"
        )
    with RolloutIndex(db_path) as index:
        rollouts = index.load_all(store_dir, modality=modality)
    if not rollouts:
        raise ValueError(
            f"index at {db_path} has no rollouts for modality={modality!r} -- "
            "double check --modality matches what was indexed, and that this "
            "isn't an empty/placeholder store"
        )
    return rollouts


def _load_config(config_path: str | None, modality: str) -> GapConfig:
    if config_path is None:
        return GapConfig(modality=modality)
    return GapConfig.from_yaml(config_path, modality=modality)


def _default_checkpoint_path(data_dir: Path, modality: str) -> Path:
    return data_dir / f"checkpoint_{modality}.pt"


def _cmd_train(args: argparse.Namespace) -> int:
    config = _load_config(args.config, args.modality)
    rollouts = _load_rollout_store(args.data_dir, args.modality, args.index_db)
    print(f"[worldgap train] loaded {len(rollouts)} {args.modality} rollouts from {args.data_dir}", file=sys.stderr)

    analyzer = GapAnalyzer(config)
    fit_report = analyzer.fit(rollouts)

    checkpoint_path = args.checkpoint or _default_checkpoint_path(args.data_dir, args.modality)
    analyzer.save_checkpoint(checkpoint_path)

    print(
        f"[worldgap train] done: final_loss={fit_report['final_loss']:.4f} "
        f"n_steps={fit_report['n_steps']} n_skipped_rollouts={fit_report['n_skipped_rollouts']} "
        f"checkpoint={checkpoint_path}",
        file=sys.stderr,
    )
    if fit_report["collapsed"]:
        print(
            "[worldgap train] WARNING: collapse safeguard flagged sustained low latent "
            "variance during training (spec 6.3/12.7) -- gap scores from this checkpoint "
            "are likely meaningless (a collapsed encoder maps everything to ~the same "
            "point, which trivially minimizes distance regardless of real domain gap). "
            "Do not trust downstream results without addressing this first.",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    source_rollouts = _load_rollout_store(args.source, args.modality, args.source_index_db)
    target_rollouts = _load_rollout_store(args.target, args.modality, args.target_index_db)
    print(
        f"[worldgap analyze] source={len(source_rollouts)} rollouts from {args.source}, "
        f"target={len(target_rollouts)} rollouts from {args.target}",
        file=sys.stderr,
    )

    if args.checkpoint is not None:
        print(f"[worldgap analyze] loading trained model from {args.checkpoint}", file=sys.stderr)
        analyzer = GapAnalyzer.load_checkpoint(args.checkpoint)
    else:
        print(
            "[worldgap analyze] NOTE: no --checkpoint given -- fitting a fresh world model "
            "on source+target combined as self-supervised pretraining (a documented fallback, "
            "not a spec requirement: the CLI's own usage example doesn't show an explicit "
            "checkpoint flag for `analyze`). Pass --checkpoint from a prior `worldgap train` "
            "run for a properly-trained model instead.",
            file=sys.stderr,
        )
        config = _load_config(args.config, args.modality)
        analyzer = GapAnalyzer(config)
        analyzer.fit(source_rollouts + target_rollouts)

    result = analyzer.compute_gap(source_rollouts, target_rollouts)
    for w in result.warnings:
        print(f"[worldgap analyze] WARNING: {w}", file=sys.stderr)

    entry = ReportEntry(condition_label=f"{args.source.name}_vs_{args.target.name}", result=result)
    out_path = generate_report([entry], args.output)
    print(
        f"[worldgap analyze] frechet={result.frechet.distance:.4f} mmd={result.mmd.mmd_squared:.4f} "
        f"confidence={result.confidence} report={out_path}",
        file=sys.stderr,
    )
    return 0


def _load_condition_value_csv(path: Path, value_col: str) -> tuple[list[tuple[dict, float]], list[str]]:
    import pandas as pd

    df = pd.read_csv(path)
    if value_col not in df.columns:
        raise ValueError(f"{path}: missing required column {value_col!r} (found {list(df.columns)})")
    condition_cols = [c for c in df.columns if c != value_col]
    if not condition_cols:
        raise ValueError(f"{path}: no condition columns found besides {value_col!r}")
    rows = [
        ({c: row[c] for c in condition_cols}, float(row[value_col])) for _, row in df.iterrows()
    ]
    return rows, condition_cols


def _cmd_validate(args: argparse.Namespace) -> int:
    gap_rows, gap_cond_cols = _load_condition_value_csv(args.gap_scores, "gap_score")
    truth_rows, truth_cond_cols = _load_condition_value_csv(args.ground_truth, "ground_truth_degradation")

    if set(gap_cond_cols) != set(truth_cond_cols):
        raise ValueError(
            f"condition columns don't match between {args.gap_scores} ({gap_cond_cols}) and "
            f"{args.ground_truth} ({truth_cond_cols}) -- both files must key on the same "
            "condition columns to be joined"
        )

    print(
        "[worldgap validate] NOTE: this command validates that gap-scores and ground-truth "
        "cover the same condition set; it treats the gap-scores file's conditions as the "
        "registered set. The actual anti-cherry-picking discipline spec 8.3 requires -- "
        "deciding the condition list BEFORE running experiments -- has to happen upstream, "
        "at data-collection time; this CLI can't retroactively enforce that.",
        file=sys.stderr,
    )

    truth_by_key = {_condition_key(c): v for c, v in truth_rows}
    conditions = [c for c, _ in gap_rows]

    harness = ValidationHarness(min_conditions=args.min_conditions)
    harness.pre_register_conditions(conditions)

    condition_results = []
    missing = []
    for condition, gap_score in gap_rows:
        key = _condition_key(condition)
        if key not in truth_by_key:
            missing.append(condition)
            continue
        condition_results.append(
            ConditionResult(condition=condition, gap_score=gap_score, ground_truth_degradation=truth_by_key[key])
        )
    if missing:
        raise ValueError(
            f"{len(missing)} condition(s) in {args.gap_scores} have no matching row in "
            f"{args.ground_truth}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    report = harness.run(condition_results)
    s = report.spearman
    print(
        f"[worldgap validate] Spearman rho={s.rho:.3f} "
        f"(95% CI [{s.ci_low:.3f}, {s.ci_high:.3f}], n_bootstrap={s.n_bootstrap}) "
        f"across n_conditions={report.n_conditions}",
        file=sys.stderr,
    )
    if s.ci_low <= 0.0 <= s.ci_high:
        print(
            "[worldgap validate] NOTE: the confidence interval includes zero -- this is a "
            "legitimate, reportable outcome per spec Section 16 ('validation correlation may "
            "simply be weak'), not a failure to hide.",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worldgap")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train the world model on a set of rollouts")
    train_p.add_argument("--modality", required=True, choices=["perception", "actuation"])
    train_p.add_argument("--data-dir", required=True, type=Path)
    train_p.add_argument("--config", default=None)
    train_p.add_argument("--index-db", type=Path, default=None, help="Default: <data-dir>/index.db")
    train_p.add_argument("--checkpoint", type=Path, default=None, help="Default: <data-dir>/checkpoint_<modality>.pt")
    train_p.set_defaults(func=_cmd_train)

    analyze_p = subparsers.add_parser("analyze", help="Compute a gap score between two rollout sets")
    analyze_p.add_argument("--source", required=True, type=Path)
    analyze_p.add_argument("--target", required=True, type=Path)
    analyze_p.add_argument("--modality", required=True, choices=["perception", "actuation"])
    analyze_p.add_argument("--output", default="report.html")
    analyze_p.add_argument("--checkpoint", type=Path, default=None, help="From a prior `worldgap train` run")
    analyze_p.add_argument("--config", default=None, help="Only used if --checkpoint is omitted")
    analyze_p.add_argument("--source-index-db", type=Path, default=None, help="Default: <source>/index.db")
    analyze_p.add_argument("--target-index-db", type=Path, default=None, help="Default: <target>/index.db")
    analyze_p.set_defaults(func=_cmd_analyze)

    validate_p = subparsers.add_parser("validate", help="Run the validation harness")
    validate_p.add_argument("--gap-scores", required=True, type=Path)
    validate_p.add_argument("--ground-truth", required=True, type=Path)
    validate_p.add_argument("--min-conditions", type=int, default=10)
    validate_p.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
