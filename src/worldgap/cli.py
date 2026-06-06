"""CLI entry point, per TECHNICAL_SPEC.md Section 9.2.

    worldgap train    --modality perception --data-dir ... --config ...
    worldgap analyze  --source ... --target ... --modality ... --output report.html
    worldgap validate --gap-scores results.csv --ground-truth degradation.csv

v0.1 status: this is an argparse skeleton only. `train`/`analyze`/`validate`
parse their arguments and print guidance to stderr, but do NOT yet call
GapAnalyzer, ValidationHarness, or any data loader end-to-end — see
ROADMAP.md Phase 5. For programmatic use today, call GapAnalyzer /
ValidationHarness directly; tests/test_modality_swap.py shows the exact
fit()/compute_gap() call sequence this CLI will eventually wrap. Report
generation (HTML/Markdown output, spec 9.3) is also not yet implemented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import GapConfig


def _load_config(config_path: str | None, modality: str) -> GapConfig:
    if config_path is None:
        return GapConfig(modality=modality)
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    raw["modality"] = modality
    return GapConfig(**raw)


def _cmd_train(args: argparse.Namespace) -> int:
    print(
        f"[worldgap train] modality={args.modality} data_dir={args.data_dir} "
        f"config={args.config}",
        file=sys.stderr,
    )
    print(
        "NOTE: wiring this to a real data_dir loader is a Phase 1 item — see "
        "ROADMAP.md. For now, use GapAnalyzer.fit() directly with Rollout objects "
        "you've constructed yourself (see notebooks/ once they exist).",
        file=sys.stderr,
    )
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    print(
        f"[worldgap analyze] source={args.source} target={args.target} "
        f"modality={args.modality} output={args.output}",
        file=sys.stderr,
    )
    print(
        "NOTE: report generation (spec 9.3) is not yet implemented — see ROADMAP.md. "
        "Use GapAnalyzer.compute_gap() directly for now.",
        file=sys.stderr,
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    print(
        f"[worldgap validate] gap_scores={args.gap_scores} "
        f"ground_truth={args.ground_truth}",
        file=sys.stderr,
    )
    print(
        "NOTE: CSV-driven validation run is not yet implemented — see ROADMAP.md. "
        "Use worldgap.validation.harness.ValidationHarness directly for now.",
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
    train_p.set_defaults(func=_cmd_train)

    analyze_p = subparsers.add_parser("analyze", help="Compute a gap score between two rollout sets")
    analyze_p.add_argument("--source", required=True, type=Path)
    analyze_p.add_argument("--target", required=True, type=Path)
    analyze_p.add_argument("--modality", required=True, choices=["perception", "actuation"])
    analyze_p.add_argument("--output", default="report.html")
    analyze_p.set_defaults(func=_cmd_analyze)

    validate_p = subparsers.add_parser("validate", help="Run the validation harness")
    validate_p.add_argument("--gap-scores", required=True, type=Path)
    validate_p.add_argument("--ground-truth", required=True, type=Path)
    validate_p.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
