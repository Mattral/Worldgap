#!/usr/bin/env python3
"""V1 run 2: implements docs/v1_run2_preregistration.md, sections 5-8 (with
Amendment 1). Change nothing here that the pre-registration fixes.

    python scripts/run_v1_run2.py --recordings ./recordings_run2 --dry-run
    python scripts/run_v1_run2.py --recordings ./recordings_run2 \
        --model ./models/holistic_landmarker.task --out ./v1_run2

Pipeline: extract every (condition, take) once, with caching, so an overnight
run can be stopped and resumed without redoing work; fit one world model on
the clean windows; score the 24 software-degraded conditions (confirmatory)
and the physical ones (descriptive); then apply Rules A and B with a single
paired bootstrap.

Order matters and is auditable: `study_settings.json` and
`preregistered_conditions.json` are written before any extraction or score.
`--max-frames` and `--epochs` exist for smoke-testing the pipeline only; any
run that changes them is not run 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from worldgap import GapAnalyzer, GapConfig, Rollout, __version__
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig
from worldgap.data.degradations import ImageDegradation
from worldgap.data.loaders.video import (
    extract_rollout_from_video,
    landmark_quality_ground_truth,
    list_videos,
    paired_landmark_error,
)
from worldgap.data.rollout import (
    PERCEPTION_FEATURE_LAYOUT,
    PERCEPTION_STATE_DIM,
    split_into_windows,
)
from worldgap.report import ReportEntry, generate_report
from worldgap.validation.stats import paired_spearman_bootstrap

# -- fixed by the pre-registration ----------------------------------------------

SEVERITIES = {
    "darken": [0.3, 0.2, 0.12, 0.08, 0.07, 0.06],
    "downscale": [192, 128, 96, 64, 48, 32],
    "blur": [2, 4, 6, 9, 13, 18],
    "noise": [10, 15, 20, 24, 28, 32],
}
PRIMARY = {f"{k}_{s}": ImageDegradation(k, s, seed=0) for k, ss in SEVERITIES.items() for s in ss}
PHYSICAL = ["occlusion_25", "occlusion_50", "occlusion_75", "distance_2m", "distance_3m", "distance_4m"]
CLEAN = "clean"
N_CLEAN_TAKES = 3
SCHEME = "shoulder_midpoint"
WINDOW_FRAMES = 48
SUMMARY_DIM = 16
EPOCHS = 30
N_BOOTSTRAP = 10_000
SEED = 0
UNDERPOWERED_RHO = 0.80  # Amendment 1, section 8.4
PRACTICAL_RHO = 0.60

assert len(PRIMARY) == 24


def model_config(epochs: int) -> GapConfig:
    return GapConfig(
        modality="perception",
        state_dim=PERCEPTION_STATE_DIM,
        encoder=EncoderConfig(d_model=128, n_layers=2, n_heads=4, dim_feedforward=256),
        world_model=WorldModelConfig(context_frames=16, predict_frames=8, summary_dim=SUMMARY_DIM),
        training=TrainingConfig(max_epochs=epochs, batch_size=16, seed=SEED),
    )


def build_landmarker(model_path: Path):
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


# -- caching ----------------------------------------------------------------------


def _cache_key(video: Path, transform: ImageDegradation | None, max_frames: int | None) -> str:
    st = video.stat()
    ident = {
        "video": str(video.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "transform": transform.describe() if transform else None,
        "max_frames": max_frames,
        "scheme": SCHEME,
        "worldgap": __version__,
    }
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:24]


def load_or_extract(video, transform, make_landmarker, cache_dir: Path, max_frames) -> tuple[Rollout, bool]:
    """Returns (rollout, from_cache). A cache entry is the same extraction,
    stored; it is written atomically so an interrupted run leaves no partial
    entry behind."""
    key = _cache_key(video, transform, max_frames)
    npz, meta = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
    if npz.exists() and meta.exists():
        arrays = np.load(npz)
        info = json.loads(meta.read_text())
        return Rollout(
            modality="perception", source="real", condition=info["condition"],
            frame_rate_hz=info["frame_rate_hz"], states=arrays["states"],
            presence_mask=arrays["presence_mask"], timestamps_ms=arrays["timestamps_ms"],
            metadata=info["metadata"],
        ), True

    landmarker = make_landmarker()  # fresh per video: VIDEO-mode timestamps
    try:
        rollout = extract_rollout_from_video(
            video, landmarker, condition={"video": video.name}, max_frames=max_frames,
            normalization_scheme=SCHEME, frame_transform=transform,
        )
    finally:
        close = getattr(landmarker, "close", None)
        if close:
            close()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_npz, tmp_meta = npz.with_suffix(".tmp.npz"), meta.with_suffix(".tmp.json")
    np.savez_compressed(tmp_npz, states=rollout.states, presence_mask=rollout.presence_mask,
                        timestamps_ms=rollout.timestamps_ms)
    tmp_meta.write_text(json.dumps({"condition": rollout.condition, "frame_rate_hz": rollout.frame_rate_hz,
                                    "metadata": rollout.metadata}))
    os.replace(tmp_npz, npz)
    os.replace(tmp_meta, meta)
    return rollout, False


# -- measures -----------------------------------------------------------------------


def _hand_present_per_frame(rollout: Rollout) -> np.ndarray:
    mask = np.asarray(rollout.presence_mask) > 0
    present = np.zeros(mask.shape[0], dtype=bool)
    for name in ("left_hand", "right_hand"):
        present |= mask[:, PERCEPTION_FEATURE_LAYOUT[name]["start"]]
    return present


def windows_of(rollouts: list[Rollout]) -> list[Rollout]:
    out = []
    for r in rollouts:
        out.extend(split_into_windows(r, window_frames=WINDOW_FRAMES))
    return out


def fully_present(windows: list[Rollout]) -> list[Rollout]:
    """Windows in which every frame has at least one hand (section 8.3 item 2)."""
    return [w for w in windows if _hand_present_per_frame(w).all()]


def take_ground_truth(clean: Rollout, degraded: Rollout | None, observed: Rollout) -> dict:
    gt = landmark_quality_ground_truth(observed)
    row = {
        "hand_dropout_rate": gt["hand_dropout_rate"],
        "longest_dropout_run_s": gt["longest_dropout_run_s"],
        "presence_density": float(np.asarray(observed.presence_mask).mean()),
    }
    if degraded is not None:
        pe = paired_landmark_error(clean, degraded)
        row.update(hand_landmark_error=pe["hand_landmark_error"], n_hand_comparisons=pe["n_hand_comparisons"],
                   frames_compared_fraction=pe["frames_compared_fraction"])
    return row


def pooled_landmark_error(takes: list[dict]) -> float:
    """Section 7.2: pooled over all compared (frame, hand) pairs of the takes."""
    n = sum(t["n_hand_comparisons"] for t in takes)
    if n == 0:
        return float("nan")
    return sum(t["hand_landmark_error"] * t["n_hand_comparisons"] for t in takes if t["n_hand_comparisons"]) / n


def gap(analyzer: GapAnalyzer, source: list[Rollout], target: list[Rollout]) -> dict:
    r = analyzer.compute_gap(source, target)
    return {
        "frechet": r.frechet.distance, "mmd2": r.mmd.mmd_squared, "confidence": r.confidence,
        "n_source": r.n_source, "n_target": r.n_target, "warnings": r.warnings, "_result": r,
    }


def interpret(rule_a: bool, rule_b: bool, rho_gap: float) -> str:
    """Section 8.4 as amended (Amendment 1)."""
    if not rule_a:
        return "Rule A failed: the gap score did not track landmark degradation across these conditions."
    if rule_b:
        return ("Rules A and B passed: on one subject and one camera, across 24 pre-registered software-degraded "
                "conditions, the gap score ranked MediaPipe landmark degradation, and did so better than counting "
                "dropouts. Not 'validated' in general.")
    if rho_gap >= UNDERPOWERED_RHO:
        return (f"Rule A passed, Rule B failed with rho_gap >= {UNDERPOWERED_RHO}: the test was underpowered to "
                "settle whether the gap score adds value over counting dropouts. This is not evidence of no "
                "added value.")
    return ("Rule A passed, Rule B failed: the gap score tracks degradation, but not demonstrably better than "
            "counting dropouts. No claim of added value.")


# -- provenance -----------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(root: Path) -> dict[str, list[Path]]:
    found = {}
    for name in [CLEAN, *PHYSICAL]:
        d = root / name
        found[name] = list_videos(d) if d.is_dir() else []
    return found


# -- main ---------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("models/holistic_landmarker.task"))
    ap.add_argument("--out", type=Path, default=Path("v1_run2"))
    ap.add_argument("--cache", type=Path, default=None, help="default: <out>/cache")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="SMOKE TESTS ONLY")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="SMOKE TESTS ONLY")
    ap.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help="SMOKE TESTS ONLY")
    ap.add_argument(
        "--blind",
        action="store_true",
        help="run every stage but write only a health report, discarding all scores and correlations "
        "(for testing on real footage without previewing the result)",
    )
    args = ap.parse_args()
    smoke = args.blind or args.max_frames is not None or args.epochs != EPOCHS or args.bootstrap != N_BOOTSTRAP

    print("worldgap V1 run 2 (docs/v1_run2_preregistration.md)")
    print("=" * 70)
    if smoke:
        print("SMOKE MODE: settings differ from the pre-registration. These results are NOT run 2.")
    found = discover(args.recordings)
    clean_videos = found[CLEAN]
    print(f"clean takes: {len(clean_videos)} {[v.name for v in clean_videos]}")
    for name in PHYSICAL:
        print(f"physical {name:<14} {len(found[name])} takes" + ("" if found[name] else "  (not recorded)"))
    n_passes = len(clean_videos) * (1 + len(PRIMARY)) + sum(len(found[p]) for p in PHYSICAL)
    print(f"primary conditions: {len(PRIMARY)} x {len(clean_videos)} takes; total extraction passes: {n_passes}")
    if len(clean_videos) != N_CLEAN_TAKES:
        print(f"ERROR: the pre-registration fixes {N_CLEAN_TAKES} clean takes; found {len(clean_videos)}.")
        return 1
    if args.dry_run:
        print("--dry-run: stopping before MediaPipe is loaded.")
        return 0

    out = args.out
    cache = args.cache or out / "cache"
    out.mkdir(parents=True, exist_ok=True)

    # Written before any extraction or score (auditable order).
    import mediapipe

    settings = {
        "preregistration": "docs/v1_run2_preregistration.md",
        "smoke_mode": smoke,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "script_sha256": _sha256(Path(__file__)),
        "worldgap": __version__,
        "mediapipe": getattr(mediapipe, "__version__", None),
        "model_sha256": _sha256(args.model),
        "normalization_scheme": SCHEME,
        "window_frames": WINDOW_FRAMES,
        "summary_dim": SUMMARY_DIM,
        "epochs": args.epochs,
        "max_frames": args.max_frames,
        "n_bootstrap": args.bootstrap,
        "seed": SEED,
        "primary_conditions": {k: d.describe() for k, d in PRIMARY.items()},
        "physical_conditions": PHYSICAL,
        "recordings": {k: [v.name for v in vs] for k, vs in found.items()},
    }
    (out / "study_settings.json").write_text(json.dumps(settings, indent=2))
    (out / "preregistered_conditions.json").write_text(
        json.dumps({"primary": list(PRIMARY), "physical": PHYSICAL}, indent=2)
    )

    def make_landmarker():
        return build_landmarker(args.model)

    # -- extraction --------------------------------------------------------------------
    t0 = time.perf_counter()
    done = 0

    def progress(label: str, cached: bool):
        nonlocal done
        done += 1
        el = time.perf_counter() - t0
        print(f"  [{done:>3}/{n_passes}] {label:<34} {'cached' if cached else f'{el / 60:6.1f} min elapsed'}",
              flush=True)

    clean = []
    for v in clean_videos:
        r, c = load_or_extract(v, None, make_landmarker, cache, args.max_frames)
        clean.append(r)
        progress(f"clean/{v.name}", c)

    degraded: dict[str, list[Rollout]] = {}
    for name, d in PRIMARY.items():
        degraded[name] = []
        for v in clean_videos:
            r, c = load_or_extract(v, d, make_landmarker, cache, args.max_frames)
            degraded[name].append(r)
            progress(f"{name}/{v.name}", c)

    physical: dict[str, list[Rollout]] = {}
    for name in PHYSICAL:
        physical[name] = []
        for v in found[name]:
            r, c = load_or_extract(v, None, make_landmarker, cache, args.max_frames)
            physical[name].append(r)
            progress(f"{name}/{v.name}", c)

    # -- model ----------------------------------------------------------------------------
    clean_windows = windows_of(clean)
    print(f"\nTraining on {len(clean_windows)} clean windows (summary_dim {SUMMARY_DIM}, {args.epochs} epochs)")
    analyzer = GapAnalyzer(model_config(args.epochs))
    fit = analyzer.fit(clean_windows)
    print(f"  {fit}")
    if fit["collapsed"]:
        (out / "results.json").write_text(json.dumps({"status": "void: collapse safeguard fired", "fit": fit}, indent=2))
        print("STOP: collapse safeguard fired. Per section 6 no gap score is reported.")
        return 2
    analyzer.save_checkpoint(out / "checkpoint.pt")
    clean_present = fully_present(clean_windows)

    # -- primary conditions -----------------------------------------------------------------
    rows, entries = [], []
    for name, rollouts in degraded.items():
        takes = [take_ground_truth(c, d, d) for c, d in zip(clean, rollouts)]
        cond_windows = windows_of(rollouts)
        g = gap(analyzer, clean_windows, cond_windows)
        entries.append(ReportEntry(name, g.pop("_result")))
        present = fully_present(cond_windows)
        present_gap = None
        if len(present) >= 3 and len(clean_present) >= 3:
            try:
                pg = gap(analyzer, clean_present, present)
                pg.pop("_result")
                present_gap = {**pg, "below_confidence_floor": len(present) < 5 * SUMMARY_DIM}
            except FloatingPointError as e:
                present_gap = {"error": str(e)}
        rows.append({
            "condition": name, **PRIMARY[name].describe(), **g,
            "landmark_error": pooled_landmark_error(takes),
            "n_hand_comparisons": sum(t["n_hand_comparisons"] for t in takes),
            "frames_compared_fraction": float(np.mean([t["frames_compared_fraction"] for t in takes])),
            "dropout": float(np.mean([t["hand_dropout_rate"] for t in takes])),
            "longest_dropout_run_s": float(np.mean([t["longest_dropout_run_s"] for t in takes])),
            "no_hand_fraction": float(np.mean([t["hand_dropout_rate"] for t in takes])),
            "one_minus_presence_density": 1.0 - float(np.mean([t["presence_density"] for t in takes])),
            "n_present_windows": len(present),
            "present_only_gap": present_gap,
            "takes": takes,
        })

    # -- analysis (sections 8.1-8.3) -----------------------------------------------------------
    err = np.array([r["landmark_error"] for r in rows])
    undefined = [r["condition"] for r in rows if not np.isfinite(r["landmark_error"])]
    truth = np.where(np.isfinite(err), err, np.inf)  # section 4.1: undefined ranks worst
    scores = {
        "gap": np.array([r["frechet"] for r in rows]),
        "no_hand_fraction": np.array([r["no_hand_fraction"] for r in rows]),
        "one_minus_presence_density": np.array([r["one_minus_presence_density"] for r in rows]),
        "mmd2": np.array([r["mmd2"] for r in rows]),
    }
    main_stats = paired_spearman_bootstrap(truth, scores, [("gap", "no_hand_fraction")],
                                           n_bootstrap=args.bootstrap, seed=SEED)
    rho_gap = main_stats["rho"]["gap"]
    rule_a = main_stats["ci"]["gap"][0] > 0
    rule_b = main_stats["diff"]["gap-no_hand_fraction"]["ci"][0] > 0

    dropout = np.array([r["dropout"] for r in rows])
    dropout_stats = paired_spearman_bootstrap(
        dropout, {"gap": scores["gap"], "no_hand_fraction": scores["no_hand_fraction"]},
        n_bootstrap=args.bootstrap, seed=SEED)

    pg = [(r["present_only_gap"] or {}).get("frechet") for r in rows]
    have = [i for i, v in enumerate(pg) if v is not None]
    present_stats = None
    if len(have) >= 3:
        present_stats = paired_spearman_bootstrap(
            truth[have], {"gap_present_only": np.array([pg[i] for i in have])},
            n_bootstrap=args.bootstrap, seed=SEED)
        present_stats["n_conditions"] = len(have)

    per_factor = {}
    for kind in SEVERITIES:
        sub = [r for r in rows if r["kind"] == kind]
        e = np.array([r["landmark_error"] for r in sub])
        rho = spearmanr([r["frechet"] for r in sub], np.where(np.isfinite(e), e, np.inf))[0]
        per_factor[kind] = float(rho)

    physical_rows = []
    for name, rollouts in physical.items():
        if not rollouts:
            physical_rows.append({"condition": name, "status": "not recorded"})
            continue
        takes = [take_ground_truth(r, None, r) for r in rollouts]
        g = gap(analyzer, clean_windows, windows_of(rollouts))
        entries.append(ReportEntry(name, g.pop("_result")))
        physical_rows.append({
            "condition": name, **g,
            "dropout": float(np.mean([t["hand_dropout_rate"] for t in takes])),
            "no_hand_fraction": float(np.mean([t["hand_dropout_rate"] for t in takes])),
            "takes": takes,
        })

    verdict = interpret(rule_a, rule_b, rho_gap)
    results = {
        "status": "smoke" if smoke else "run2",
        "fit": fit,
        "rules": {
            "A": {"rho_gap": rho_gap, "ci": main_stats["ci"]["gap"], "pass": bool(rule_a)},
            "B": {"rho_baseline": main_stats["rho"]["no_hand_fraction"],
                  **main_stats["diff"]["gap-no_hand_fraction"], "pass": bool(rule_b)},
            "interpretation": verdict,
        },
        "primary_stats": main_stats,
        "secondary": {
            "dropout_ground_truth": dropout_stats,
            "present_frames_only": present_stats,
            "per_factor_rho": per_factor,
            "rho_gap_at_least_0.6": bool(rho_gap >= PRACTICAL_RHO),
            "undefined_landmark_error_ranked_worst": undefined,
        },
        "conditions": rows,
        "physical": physical_rows,
        "extraction_minutes": round((time.perf_counter() - t0) / 60, 1),
    }
    if args.blind:
        # Every stage above ran; keep only whether it worked, never the numbers.
        report_probe = out / "_blind_report_probe.html"
        generate_report(entries, report_probe, title="blind probe")
        report_ok = report_probe.exists()
        report_probe.unlink(missing_ok=True)
        health = {
            "blind": True,
            "note": "All scores and correlations were computed and discarded unread.",
            "extraction_passes": done,
            "collapsed": fit["collapsed"],
            "n_primary_conditions": len(rows),
            "all_gap_scores_finite": bool(np.isfinite(scores["gap"]).all()),
            "all_mmd_finite": bool(np.isfinite(scores["mmd2"]).all()),
            "n_undefined_landmark_error": len(undefined),
            "n_present_only_gap_computed": sum(1 for r in rows if (r["present_only_gap"] or {}).get("frechet") is not None),
            "rule_a_ci_finite": bool(np.isfinite(main_stats["ci"]["gap"]).all()),
            "rule_b_ci_finite": bool(np.isfinite(main_stats["diff"]["gap-no_hand_fraction"]["ci"]).all()),
            "bootstrap_valid_resamples": main_stats["n_valid"],
            "secondary_computed": {
                "dropout": dropout_stats is not None,
                "present_frames_only": present_stats is not None,
                "per_factor": len(per_factor),
            },
            "physical_rows": len(physical_rows),
            "report_generated": report_ok,
            "extraction_minutes": round((time.perf_counter() - t0) / 60, 1),
        }
        (out / "results_blind.json").write_text(json.dumps(health, indent=2))
        print("\nBLIND MODE: every stage ran; scores and correlations were discarded unread.")
        print(json.dumps(health, indent=2))
        return 0

    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    generate_report(entries, out / "run2_report.html", title="worldgap V1 run 2" + (" (SMOKE)" if smoke else ""))

    print(f"\nRule A: rho_gap = {rho_gap:+.3f}, 95% CI [{main_stats['ci']['gap'][0]:+.3f}, "
          f"{main_stats['ci']['gap'][1]:+.3f}] -> {'PASS' if rule_a else 'FAIL'}")
    d = main_stats["diff"]["gap-no_hand_fraction"]
    print(f"Rule B: rho_base = {main_stats['rho']['no_hand_fraction']:+.3f}, delta = {d['delta']:+.3f}, "
          f"95% CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] -> {'PASS' if rule_b else 'FAIL'}")
    print(f"\n{verdict}")
    if undefined:
        print(f"Undefined landmark error (ranked worst): {undefined}")
    print(f"\nResults -> {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
