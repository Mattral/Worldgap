#!/usr/bin/env python3
"""Power of run 2's Rule B (docs/v1_run2_preregistration.md, Amendment 1).

Rule B passes when the 95% paired-bootstrap CI of
    Δ = Spearman(gap, truth) − Spearman(baseline, truth)
has its lower bound above 0, across n conditions. This estimates how often
that happens for a given true ρ_gap, with ρ_base fixed at the pilot's value.

Model (an assumption, stated so the numbers are reproducible):
- The n conditions are exchangeable draws from a Gaussian copula over
  (truth, gap, baseline). Target Spearman correlations are converted to the
  latent Pearson correlations with r = 2·sin(π·ρ_s / 6).
- Scores are continuous (no ties). Real dropout has many ties at 0%, so this
  is optimistic about the baseline's resolution.
- Each simulated study is analysed exactly as pre-registered: percentile
  bootstrap over conditions, paired (the same resample for both
  correlations), 10,000 resamples, 95%; resamples where a correlation is
  undefined are discarded.
- corr(gap, baseline) is unknown in advance, so power is reported across a
  grid of values; cells whose correlation matrix is not positive definite
  are skipped.

    python scripts/power_rule_b.py --sims 300
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.stats import rankdata


def spearman_to_pearson(rho_s: float) -> float:
    return 2.0 * np.sin(np.pi * rho_s / 6.0)


def rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman per row of two (B, n) arrays (average ranks for ties)."""
    ra = rankdata(a, axis=1)
    rb = rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def rule_b_passes(truth, gap, base, n_boot: int, rng) -> bool:
    n = truth.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    t = truth[idx]
    d = rowwise_spearman(gap[idx], t) - rowwise_spearman(base[idx], t)
    d = d[np.isfinite(d)]
    return d.size > 0 and np.percentile(d, 2.5) > 0


def power(n: int, rho_gap: float, rho_base: float, rho_gap_base: float, sims: int, n_boot: int, seed: int):
    r = np.array([
        [1.0, spearman_to_pearson(rho_gap), spearman_to_pearson(rho_base)],
        [spearman_to_pearson(rho_gap), 1.0, spearman_to_pearson(rho_gap_base)],
        [spearman_to_pearson(rho_base), spearman_to_pearson(rho_gap_base), 1.0],
    ])
    if np.linalg.eigvalsh(r).min() <= 1e-9:
        return None  # infeasible combination
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(r)
    hits = 0
    for _ in range(sims):
        z = rng.standard_normal((n, 3)) @ chol.T
        hits += rule_b_passes(z[:, 0], z[:, 1], z[:, 2], n_boot, rng)
    return hits / sims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--rho-base", type=float, default=0.56)
    ap.add_argument("--rho-gap", type=float, nargs="*", default=[0.70, 0.80, 0.85, 0.90, 0.95])
    ap.add_argument("--rho-gap-base", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the table as JSON")
    args = ap.parse_args()

    table = {}
    for i, rg in enumerate(args.rho_gap):
        for j, rgb in enumerate(args.rho_gap_base):
            p = power(args.n, rg, args.rho_base, rgb, args.sims, args.boot, args.seed + 1000 * i + j)
            table[(rg, rgb)] = p

    if args.json:
        print(json.dumps([{"rho_gap": k[0], "rho_gap_base": k[1], "power": v} for k, v in table.items()]))
        return 0

    head = "".join(f"  r(gap,base)={c:<4}" for c in args.rho_gap_base)
    print(f"Rule B power, n={args.n}, rho_base={args.rho_base}, {args.sims} sims/cell, {args.boot} resamples")
    print(f"rho_gap {head}   range")
    for rg in args.rho_gap:
        vals = [table[(rg, c)] for c in args.rho_gap_base]
        cells = "".join(f"  {'infeasible':>16}" if v is None else f"  {v:>16.2f}" for v in vals)
        ok = [v for v in vals if v is not None]
        rng_txt = f"{min(ok):.2f}-{max(ok):.2f}" if ok else "-"
        print(f"{rg:<7} {cells}   {rng_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
