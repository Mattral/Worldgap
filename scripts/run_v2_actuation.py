#!/usr/bin/env python3
"""V2 end-to-end: simulated PGM vs. real digitized Ogawa 2017 characterization.

    python scripts/run_v2_actuation.py --out ./v2_run

Unlike V1, this needs no recordings, no downloads and no model bundle -- the
real reference data ships inside the package. It runs anywhere worldgap
installs, which makes it the reproducible half of the project.

What it reports, in this order, because the order matters:

1. **Physical residuals in millimetres**, per pressure level, against each
   curve's own digitization noise floor. This is the interpretable number. A
   residual below the noise floor is not evidence of anything.
2. **Latent-space gap scores** (Frechet + MMD) from the same `GapAnalyzer`
   class V1 uses -- the reusability claim, exercised on real data rather than
   on a toy signal.
3. **Saturation and sample-size caveats**, not as footnotes.

The 0 MPa level is expected to saturate completely: the ideal McKibben model
produces zero force at zero pressure, so it cannot hold any load at any length.
That is a real finding about the model, reported as such rather than dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldgap import GapAnalyzer, GapConfig
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig
from worldgap.data.loaders.pgm_actuator import (
    OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA,
    OGAWA_2017_SCALE_CAVEAT,
)
from worldgap.data.loaders.pgm_sim import (
    ACTUATION_STATE_DIM,
    ACTUATION_STATE_LAYOUT,
    OGAWA_2017_IDEAL_MCKIBBEN,
    digitized_pgm_rollout,
    length_residuals_mm,
    simulated_pgm_rollout,
)
from worldgap.data.rollout import split_into_windows
from worldgap.report import ReportEntry, generate_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("v2_run"))
    parser.add_argument("--window-frames", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--tube-stiffness",
        type=float,
        default=0.0,
        help=(
            "N/mm linear elastic term for the gel-foam inner tube. Default 0 -- the "
            "baseline's inability to produce force at zero pressure is one of the "
            "errors being measured. Set non-zero to test whether tube elasticity "
            "explains the low-pressure gap."
        ),
    )
    args = parser.parse_args()

    params = OGAWA_2017_IDEAL_MCKIBBEN
    if args.tube_stiffness:
        from dataclasses import replace

        params = replace(params, tube_stiffness_n_per_mm=args.tube_stiffness)

    args.out.mkdir(parents=True, exist_ok=True)

    print("worldgap V2 -- simulated PGM vs. real digitized characterization")
    print("=" * 74)
    print(f"\nSimulator : ideal McKibben, {params}")
    print("Real data : Ogawa et al. (2017) Figure 4(a), digitized, bundled with worldgap")
    print(f"State     : {ACTUATION_STATE_LAYOUT} (normalized)")
    print(f"\nSCALE CAVEAT: {OGAWA_2017_SCALE_CAVEAT}")

    # --- 1. physical residuals ------------------------------------------------
    print("\n1. Physical residuals (simulated length - real length), per pressure")
    print(f"   {'P (MPa)':>8} {'mean|res|':>10} {'max|res|':>9} {'noise floor':>12} "
          f"{'> floor?':>9} {'saturated':>10}")
    residuals = []
    for p in OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA:
        r = length_residuals_mm(p, params)
        residuals.append(r)
        print(
            f"   {p:>8} {r['mean_abs_residual_mm']:>10.2f} {r['max_abs_residual_mm']:>9.2f} "
            f"{r['digitization_noise_floor_mm']:>12.2f} "
            f"{'yes' if r['residual_exceeds_noise_floor'] else 'NO':>9} "
            f"{r['saturation']['fraction_saturated']:>10.0%}"
        )
    (args.out / "residuals_mm.json").write_text(json.dumps(residuals, indent=2))

    saturated = [r for r in residuals if r["saturation"]["fraction_saturated"] > 0.5]
    if saturated:
        levels = ", ".join(f"{r['pressure_mpa']} MPa" for r in saturated)
        print(
            f"\n   NOTE: {levels} is mostly saturated -- the ideal model cannot "
            "represent\n         that condition at all (it predicts zero force at zero "
            "pressure, so it\n         holds no load at any length). That is a finding "
            "about the model, not a\n         plateau in the actuator. Excluded from the "
            "latent gap below."
        )

    usable = [
        r["pressure_mpa"] for r in residuals if r["saturation"]["fraction_saturated"] <= 0.5
    ]

    # --- 2. latent-space gap --------------------------------------------------
    print(f"\n2. Latent-space gap over {len(usable)} usable pressure levels")

    sim_rollouts, real_rollouts = [], []
    for p in usable:
        sim_rollouts.extend(
            split_into_windows(simulated_pgm_rollout(p, params=params), args.window_frames)
        )
        real_rollouts.extend(
            split_into_windows(digitized_pgm_rollout(p), args.window_frames)
        )
    print(f"   {len(sim_rollouts)} simulated windows, {len(real_rollouts)} real windows")

    config = GapConfig(
        modality="actuation",
        state_dim=ACTUATION_STATE_DIM,
        encoder=EncoderConfig(d_model=32, n_layers=2, n_heads=4, dim_feedforward=64),
        # summary_dim deliberately small: spec 7.3 wants n >= 5 * dim, and the
        # real side has only 7 measured pressure levels in existence. Raising it
        # would not add information, only lower the confidence flag.
        world_model=WorldModelConfig(context_frames=8, predict_frames=4, summary_dim=8),
        training=TrainingConfig(max_epochs=args.epochs, batch_size=16, seed=0),
    )
    analyzer = GapAnalyzer(config)
    fit_stats = analyzer.fit(sim_rollouts)
    print(f"   fit: {fit_stats}")
    if fit_stats["collapsed"]:
        print("\n   STOP: collapse safeguard fired (spec 12.7). Gap numbers meaningless.")
        return 1

    entries = []
    overall = analyzer.compute_gap(sim_rollouts, real_rollouts)
    entries.append(ReportEntry("all_pressures", overall))
    print(
        f"\n   ALL LEVELS   frechet={overall.frechet.distance:.6f}  "
        f"mmd2={overall.mmd.mmd_squared:.6f}  confidence={overall.confidence}"
    )
    for w in overall.warnings:
        print(f"                warning: {w}")

    print("\n   Per pressure level:")
    per_level = {}
    for p in usable:
        s = split_into_windows(simulated_pgm_rollout(p, params=params), args.window_frames)
        r = split_into_windows(digitized_pgm_rollout(p), args.window_frames)
        res = analyzer.compute_gap(s, r)
        entries.append(ReportEntry(f"{p} MPa", res))
        per_level[str(p)] = {
            "frechet": res.frechet.distance,
            "mmd_squared": res.mmd.mmd_squared,
            "confidence": res.confidence,
        }
        print(
            f"     {p:>5} MPa  frechet={res.frechet.distance:.6f}  "
            f"mmd2={res.mmd.mmd_squared:.6f}  conf={res.confidence}"
        )

    (args.out / "gap_scores.json").write_text(json.dumps(per_level, indent=2))

    # --- 2b. internal consistency check --------------------------------------
    #
    # Does the latent gap score track the physically interpretable error, or is
    # it responding to something else? Rank-correlate the two across pressure
    # levels.
    #
    # This is emphatically NOT spec 8.1 validation and must never be reported as
    # such: 8.1 requires ground truth computed INDEPENDENTLY of the model under
    # test, and both quantities here are derived from the same two curve
    # families. It is a consistency check -- a way for the gap score to fail
    # visibly if it were measuring encoder noise. Passing it is necessary, not
    # sufficient.
    consistency = None
    if len(usable) >= 4:
        from scipy.stats import spearmanr

        residual_by_p = {str(r["pressure_mpa"]): r["mean_abs_residual_mm"] for r in residuals}
        xs = [residual_by_p[k] for k in per_level]
        rho_f, p_f = spearmanr(xs, [v["frechet"] for v in per_level.values()])
        rho_m, p_m = spearmanr(xs, [v["mmd_squared"] for v in per_level.values()])
        # If every per-level MMD^2 came out negative, the estimator is saying it
        # cannot detect a difference at this sample size -- so a rank
        # correlation over those values is ranking noise. Reported, then
        # explicitly disqualified, rather than quoted as a second confirmation.
        n_negative = sum(1 for v in per_level.values() if v["mmd_squared"] < 0)
        mmd_is_meaningful = n_negative < len(per_level)

        consistency = {
            "n_levels": len(usable),
            "spearman_frechet_vs_residual_mm": float(rho_f),
            "p_value_frechet": float(p_f),
            "spearman_mmd_vs_residual_mm": float(rho_m),
            "p_value_mmd": float(p_m),
            "mmd_correlation_is_interpretable": mmd_is_meaningful,
            "n_levels_with_negative_mmd": n_negative,
            "is_spec_8_1_validation": False,
            "why_not": (
                "Both quantities derive from the same two curve families, so this is "
                "internal consistency, not independent ground truth."
            ),
        }
        (args.out / "consistency_check.json").write_text(json.dumps(consistency, indent=2))
        print(
            f"\n   Internal consistency (NOT spec 8.1 validation): across {len(usable)} "
            f"pressure levels,\n     latent Frechet gap vs. physical mean|residual| in mm: "
            f"Spearman rho={rho_f:+.3f} (p={p_f:.4f})"
        )
        if mmd_is_meaningful:
            print(f"     MMD^2 vs. the same: rho={rho_m:+.3f} (p={p_m:.4f})")
        else:
            print(
                f"     MMD^2 vs. the same: rho={rho_m:+.3f} -- DISREGARD. All "
                f"{n_negative} levels have a\n       negative MMD^2, meaning the "
                "estimator detects no difference at this sample\n       size, so "
                "ranking them ranks noise. Only the Frechet correlation counts here."
            )
        print(
            "     Both quantities come from the same two curve families, so this shows "
            "the gap\n     score tracks a physically meaningful error rather than encoder "
            "noise -- it does\n     not show the score predicts real-world transfer "
            "degradation."
        )
    report = generate_report(
        entries,
        args.out / "v2_report.html",
        title="worldgap V2 -- ideal McKibben vs. Ogawa 2017 characterization",
    )
    print(f"\nReport -> {report}")

    # --- 3. how to read this --------------------------------------------------
    print("\n3. How to read this")
    print(
        "   - The residuals in millimetres are the interpretable result; the latent\n"
        "     gap scores are the reusability demonstration. Neither is 'validated' in\n"
        "     spec 8.1's sense: that needs an independent measurement of real transfer\n"
        "     degradation, which does not exist for this actuator yet.\n"
        "   - The real side is digitized from a published figure, not measured here.\n"
        "     Its own noise floor is printed above; residuals below it mean nothing.\n"
        "   - Sample size is bounded by physics-of-the-literature: only 7 pressure\n"
        "     levels were ever published. A 'low' confidence flag here is a fact about\n"
        "     the available data, not a fixable configuration problem."
    )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
