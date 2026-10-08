#!/usr/bin/env python3
"""Run-2 pilot: how badly does each graded image degradation break MediaPipe?

Takes one clean recording, applies each degradation in a severity grid (see
`worldgap.data.degradations`), runs the real landmarker, and reports per
condition:

- `hand_dropout_rate` and `longest_dropout_run_s` (spec 8.1 signals),
- `hand_landmark_error`: distance from the same frame's clean detection, in
  hand sizes (`paired_landmark_error`),
- the trivial baselines the pre-registration compares the gap score against:
  `no_hand_fraction` and `presence_density` (mean of the presence mask).

Its only job is to pick severities that span the ground truth's range, so it
deliberately computes **no gap scores and no gap correlations**: choosing
severities while looking at the tool's performance would bias the confirmatory
run. It does report how well the dropout baseline alone ranks landmark error,
because that is a property of the ground truth, and it decides how high the
bar is for the gap score.

Use footage that will NOT be part of run 2 (e.g. run 1's clean take). Pilot
output is labelled as pilot and never enters the confirmatory analysis.

    python scripts/run_v1_pilot.py --video recordings/clean/take0.mp4 \
        --model models/holistic_landmarker.task --max-frames 900
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from worldgap.data.degradations import ImageDegradation
from worldgap.data.loaders.video import (
    extract_rollout_from_video,
    landmark_quality_ground_truth,
    paired_landmark_error,
)

DEFAULT_GRID = {
    "darken": [0.5, 0.3, 0.2, 0.12, 0.08, 0.05, 0.03],
    "downscale": [320, 192, 128, 96, 64, 48, 32],
    "blur": [1, 2, 4, 6, 9, 13, 18],
    "noise": [10, 20, 35, 50, 70, 100, 140],
}


def build_landmarker(model_path: Path):
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def presence_baselines(rollout) -> dict[str, float]:
    """The trivial scores: what you get by just counting what is missing."""
    gt = landmark_quality_ground_truth(rollout)
    return {
        "no_hand_fraction": gt["hand_dropout_rate"],
        "presence_density": float(np.asarray(rollout.presence_mask).mean()),
    }


def extract(video: Path, model: Path, max_frames: int | None, transform=None):
    landmarker = build_landmarker(model)  # fresh per video: VIDEO-mode timestamps
    try:
        return extract_rollout_from_video(
            video, landmarker, max_frames=max_frames, frame_transform=transform
        )
    finally:
        close = getattr(landmarker, "close", None)
        if close:
            close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, default=Path("recordings/clean/take0.mp4"))
    ap.add_argument("--model", type=Path, default=Path("models/holistic_landmarker.task"))
    ap.add_argument("--max-frames", type=int, default=900, help="pilot on the first N frames")
    ap.add_argument("--kinds", nargs="*", default=None, choices=list(DEFAULT_GRID))
    ap.add_argument(
        "--grid",
        type=json.loads,
        default=None,
        help='custom severities as JSON, e.g. \'{"darken": [0.07, 0.06]}\' (refining a cliff)',
    )
    ap.add_argument("--out", type=Path, default=Path("v1_pilot/pilot.json"))
    args = ap.parse_args()
    grid = args.grid if args.grid is not None else DEFAULT_GRID
    if args.kinds is not None:
        grid = {k: grid[k] for k in args.kinds}
    for kind in grid:
        if kind not in DEFAULT_GRID:
            print(f"ERROR: unknown degradation {kind!r}")
            return 1

    if not args.video.exists():
        print(f"ERROR: {args.video} not found")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"worldgap run-2 pilot on {args.video} (first {args.max_frames} frames)")
    print("No gap scores are computed here, by design.\n")
    t0 = time.perf_counter()
    reference = extract(args.video, args.model, args.max_frames)
    rows = [{
        "kind": "none", "severity": None,
        **landmark_quality_ground_truth(reference),
        **paired_landmark_error(reference, reference),
        **presence_baselines(reference),
    }]
    print(f"  {'none':<10} {'-':>7}  dropout {rows[0]['hand_dropout_rate']:6.1%}")

    for kind, severities in grid.items():
        for severity in severities:
            d = ImageDegradation(kind, severity)
            rollout = extract(args.video, args.model, args.max_frames, d)
            row = {
                "kind": kind,
                "severity": severity,
                **landmark_quality_ground_truth(rollout),
                **paired_landmark_error(reference, rollout),
                **presence_baselines(rollout),
            }
            rows.append(row)
            print(
                f"  {kind:<10} {severity:>7}  dropout {row['hand_dropout_rate']:6.1%}  "
                f"landmark error {row['hand_landmark_error']:.3f}  "
                f"longest gap {row['longest_dropout_run_s']:.2f}s",
                flush=True,
            )

    degraded = [r for r in rows if r["kind"] != "none" and np.isfinite(r["hand_landmark_error"])]
    baseline_vs_error = None
    if len(degraded) >= 3:
        rho, p = spearmanr([r["no_hand_fraction"] for r in degraded], [r["hand_landmark_error"] for r in degraded])
        baseline_vs_error = {"spearman_rho": float(rho), "p_value": float(p), "n_conditions": len(degraded)}
        print(f"\nDropout baseline vs landmark error across pilot conditions: rho={rho:+.3f} (n={len(degraded)})")

    args.out.write_text(json.dumps({
        "pilot": True,
        "video": str(args.video),
        "max_frames": args.max_frames,
        "grid": grid,
        "rows": rows,
        "dropout_baseline_vs_landmark_error": baseline_vs_error,
        "seconds": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"\nPilot results -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
