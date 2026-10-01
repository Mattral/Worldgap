#!/usr/bin/env python3
"""End-to-end real V1 run: video recordings -> landmarks -> rollout stores ->
gap scores -> pre-registered validation against MediaPipe's own quality signals.

This is the script that turns worldgap from "tested on synthetic data" into
"run on real measurements". It is written for the user's own machine, because
it needs two things no CI sandbox has: the MediaPipe model bundle host, and
real recordings.

    # 0. preflight (do this first)
    python scripts/check_mediapipe_setup.py --model models/holistic_landmarker.task

    # 1. record (see docs/v1_real_data_runbook.md for what to record)
    #    recordings/clean/*.mp4, recordings/<condition>/*.mp4

    # 2. run
    python scripts/run_v1_real_data.py \\
        --recordings ./recordings \\
        --model ./models/holistic_landmarker.task \\
        --out ./v1_run

What it does NOT do, on purpose: invent data. If a condition folder is empty,
it says so and skips it rather than substituting a synthetic stand-in.

STATUS: the landmark-extraction, rollout-store, gap and validation stages are
all exercised by the test suite against fakes and real synthetic data. The one
path that has NOT been executed anywhere yet is a real `HolisticLandmarker`
over real frames, because that needs the model bundle. Treat the first run as
a run to watch, not to trust blindly -- and read `--dry-run` output first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from worldgap import GapAnalyzer, GapConfig
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig
from worldgap.data.index import RolloutIndex
from worldgap.data.loaders.video import (
    extract_rollout_from_video,
    landmark_quality_ground_truth,
    list_videos,
    split_into_windows,
)
from worldgap.data.rollout import PERCEPTION_STATE_DIM
from worldgap.report import ReportEntry, generate_report
from worldgap.validation.harness import ConditionResult, ValidationHarness

CLEAN_DIRNAME = "clean"


def build_landmarker(model_path: Path, video_mode: bool):
    """Constructs a real HolisticLandmarker.

    Kept in one place so `--dry-run` can skip it and so the exact API surface
    is easy to audit against MediaPipe's docs. Verified against mediapipe
    1.0.1: `vision.HolisticLandmarker` does not exist in 0.10.x wheels.
    """
    from mediapipe.tasks.python import BaseOptions, vision

    if not hasattr(vision, "HolisticLandmarker"):
        raise RuntimeError(
            "this mediapipe build has no vision.HolisticLandmarker -- you are on "
            "0.10.x. Run scripts/check_mediapipe_setup.py; upgrade with "
            "pip install -U 'mediapipe>=1.0'."
        )
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def discover_conditions(recordings_root: Path) -> dict[str, list[Path]]:
    """One subdirectory per capture condition; `clean/` is the source domain."""
    if not recordings_root.exists():
        raise FileNotFoundError(f"{recordings_root} does not exist")
    conditions: dict[str, list[Path]] = {}
    for child in sorted(recordings_root.iterdir()):
        if not child.is_dir():
            continue
        videos = list_videos(child)
        if videos:
            conditions[child.name] = videos
        else:
            print(f"  (skipping '{child.name}': no video files found)")
    return conditions


def extract_condition(
    name: str,
    videos: list[Path],
    landmarker,
    window_frames: int,
    max_frames: int | None,
    stride: int,
) -> tuple[list, list[dict]]:
    """Returns (windowed rollouts, per-recording ground-truth dicts)."""
    rollouts = []
    ground_truths = []
    for path in videos:
        print(f"    {path.name} ... ", end="", flush=True)
        full = extract_rollout_from_video(
            path,
            landmarker,
            condition={"capture_condition": name},
            max_frames=max_frames,
            stride=stride,
        )
        gt = landmark_quality_ground_truth(full)
        ground_truths.append({"recording": path.name, **gt})
        windows = split_into_windows(full, window_frames=window_frames)
        rollouts.extend(windows)
        print(
            f"{full.states.shape[0]} frames -> {len(windows)} windows "
            f"(hand dropout {gt['hand_dropout_rate']:.1%}, "
            f"pose visibility {gt['mean_pose_visibility']:.2f})"
        )
    return rollouts, ground_truths


def save_store(rollouts: list, store_dir: Path) -> None:
    """Writes a self-contained rollout store: {dir}/index.db + {dir}/{modality}/*.npz."""
    store_dir.mkdir(parents=True, exist_ok=True)
    index = RolloutIndex(store_dir / "index.db")
    for r in rollouts:
        r.save(store_dir)
        index.add(r)


def aggregate_ground_truth(rows: list[dict]) -> float:
    """One degradation number per condition, for the Spearman correlation.

    Deliberately simple and deliberately explained: the mean hand-dropout rate
    across that condition's recordings. It is the signal a deployed
    confidence-threshold safety layer would actually trip on, it comes entirely
    from MediaPipe's own output (spec 8.1: independent of the model under
    test), and it is one number rather than a composite so that nothing is
    hidden inside a weighting nobody chose.
    """
    return float(np.mean([r["hand_dropout_rate"] for r in rows]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/holistic_landmarker.task"))
    parser.add_argument("--out", type=Path, default=Path("v1_run"))
    parser.add_argument("--window-frames", type=int, default=48)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="discover recordings and print the plan without loading MediaPipe",
    )
    args = parser.parse_args()

    print("worldgap V1 real-data run")
    print("=" * 70)

    conditions = discover_conditions(args.recordings)
    if CLEAN_DIRNAME not in conditions:
        print(
            f"ERROR: no '{CLEAN_DIRNAME}/' directory under {args.recordings}. "
            "The source domain must be your clean-condition recordings; every other "
            "subdirectory is treated as a target condition."
        )
        return 1

    target_names = [n for n in conditions if n != CLEAN_DIRNAME]
    print(f"\nSource domain : {CLEAN_DIRNAME} ({len(conditions[CLEAN_DIRNAME])} recordings)")
    print(f"Target conditions ({len(target_names)}):")
    for n in target_names:
        print(f"  - {n} ({len(conditions[n])} recordings)")

    if len(target_names) < 10:
        print(
            f"\nNOTE: spec 8.3 requires at least 10 pre-registered conditions for the "
            f"validation correlation; you have {len(target_names)}. The gap scores "
            "below will still be computed and reported -- the validation step will be "
            "skipped rather than run underpowered and presented as if it counted."
        )

    if args.dry_run:
        print("\n--dry-run: stopping before MediaPipe is loaded.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    # Pre-register BEFORE computing anything (spec 8.3). Written to disk first
    # so the order is auditable afterwards, not just asserted.
    harness = ValidationHarness(min_conditions=10)
    pre_registered = [{"capture_condition": n} for n in target_names]
    (args.out / "preregistered_conditions.json").write_text(
        json.dumps(pre_registered, indent=2)
    )
    if len(pre_registered) >= 10:
        harness.pre_register_conditions(pre_registered)
    print(f"\nPre-registered {len(pre_registered)} conditions -> "
          f"{args.out / 'preregistered_conditions.json'}")

    print("\nExtracting landmarks")
    landmarker = build_landmarker(args.model, video_mode=args.stride == 1)
    all_ground_truth: dict[str, list[dict]] = {}
    store_rollouts: dict[str, list] = {}
    try:
        for name, videos in conditions.items():
            print(f"  [{name}]")
            rollouts, gts = extract_condition(
                name, videos, landmarker, args.window_frames, args.max_frames, args.stride
            )
            store_rollouts[name] = rollouts
            all_ground_truth[name] = gts
            save_store(rollouts, args.out / "stores" / name)
    finally:
        close = getattr(landmarker, "close", None)
        if close:
            close()

    (args.out / "ground_truth.json").write_text(json.dumps(all_ground_truth, indent=2))
    print(f"\nGround truth (MediaPipe's own signals) -> {args.out / 'ground_truth.json'}")

    source = store_rollouts[CLEAN_DIRNAME]
    if not source:
        print("ERROR: clean condition produced zero rollouts.")
        return 1

    print(f"\nTraining world model on {len(source)} clean windows")
    config = GapConfig(
        modality="perception",
        state_dim=PERCEPTION_STATE_DIM,
        encoder=EncoderConfig(d_model=128, n_layers=2, n_heads=4, dim_feedforward=256),
        world_model=WorldModelConfig(context_frames=16, predict_frames=8, summary_dim=32),
        training=TrainingConfig(max_epochs=args.epochs, batch_size=16, seed=0),
    )
    analyzer = GapAnalyzer(config)
    fit_stats = analyzer.fit(source)
    print(f"  {fit_stats}")
    if fit_stats["collapsed"]:
        print(
            "\nSTOP: the collapse safeguard fired (spec 12.7). Every gap number from "
            "this model is meaningless. Do not report these results. Try a lower EMA "
            "decay or more/longer recordings."
        )
        return 1
    analyzer.save_checkpoint(args.out / "checkpoint.pt")

    print("\nComputing gaps")
    entries = []
    condition_results = []
    for name in target_names:
        target = store_rollouts[name]
        if not target:
            continue
        result = analyzer.compute_gap(source, target)
        entries.append(ReportEntry(name, result))
        gt_value = aggregate_ground_truth(all_ground_truth[name])
        condition_results.append(
            ConditionResult(
                condition={"capture_condition": name},
                gap_score=result.frechet.distance,
                ground_truth_degradation=gt_value,
            )
        )
        print(
            f"  {name:<24} frechet={result.frechet.distance:.6f}  "
            f"mmd2={result.mmd.mmd_squared:.6f}  conf={result.confidence}  "
            f"hand_dropout={gt_value:.1%}"
        )

    report_path = generate_report(
        entries, args.out / "v1_report.html", title="worldgap V1 -- real recordings"
    )
    print(f"\nReport -> {report_path}")

    if len(condition_results) >= 10:
        print("\nValidation (Spearman, pre-registered conditions)")
        validation = harness.run(condition_results)
        s = validation.spearman
        print(f"  rho={s.rho:.3f}  95% CI=[{s.ci_low:.3f}, {s.ci_high:.3f}]  p={s.p_value:.4f}")
        if s.ci_low <= 0.0 <= s.ci_high:
            print(
                "  The CI includes zero. That is a real, reportable result (spec "
                "Section 14), not a failure to hide or re-run until it moves."
            )
        (args.out / "validation.json").write_text(
            json.dumps(
                {"rho": s.rho, "ci_low": s.ci_low, "ci_high": s.ci_high, "p_value": s.p_value},
                indent=2,
            )
        )
    else:
        print(
            f"\nValidation SKIPPED: {len(condition_results)} conditions, 10 required "
            "(spec 8.3). The gap scores above stand on their own; the correlation "
            "does not exist yet and is not being approximated."
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
