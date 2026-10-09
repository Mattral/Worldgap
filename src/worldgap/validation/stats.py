"""Statistical helpers for the Validation Harness, per TECHNICAL_SPEC.md Section 8.2.

Spearman rank correlation (not Pearson) — we care about monotonic risk-ranking
between gap score and ground-truth degradation, not a linear relationship. A
bootstrap confidence interval is required alongside the point estimate; a bare
rho is not acceptable output (spec 8.2/8.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class SpearmanResult:
    rho: float
    p_value: float
    ci_low: float
    ci_high: float
    n_conditions: int
    n_bootstrap: int

    @property
    def ci_excludes_zero(self) -> bool:
        """A convenience check, NOT a pass/fail gate — spec 14 explicitly treats
        a CI that includes zero as a legitimate, reportable outcome, not a
        failure to hide.
        """
        return self.ci_low > 0 or self.ci_high < 0


def spearman_with_bootstrap_ci(
    gap_scores: np.ndarray,
    ground_truth_degradation: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> SpearmanResult:
    if len(gap_scores) != len(ground_truth_degradation):
        raise ValueError("gap_scores and ground_truth_degradation must be the same length")
    n = len(gap_scores)
    if n < 3:
        raise ValueError("need at least 3 conditions to compute a rank correlation at all")

    rho, p_value = spearmanr(gap_scores, ground_truth_degradation)

    rng = np.random.default_rng(seed)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(gap_scores[idx], ground_truth_degradation[idx])
        if not np.isnan(r):
            boot_rhos.append(r)

    if not boot_rhos:
        raise ValueError(
            "every bootstrap resample produced an undefined (NaN) Spearman rho — "
            "this happens when gap_scores or ground_truth_degradation are constant "
            "(no variation to rank). Check the input data before trusting a "
            "correlation claim here; a bootstrap CI cannot be computed from no "
            "valid resamples."
        )

    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot_rhos, alpha * 100))
    ci_high = float(np.percentile(boot_rhos, (1 - alpha) * 100))

    return SpearmanResult(
        rho=float(rho),
        p_value=float(p_value),
        ci_low=ci_low,
        ci_high=ci_high,
        n_conditions=n,
        n_bootstrap=len(boot_rhos),
    )


def _rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman per row of two (B, n) arrays, average ranks for ties. Rows
    where either side is constant give NaN."""
    from scipy.stats import rankdata

    ra = rankdata(a, axis=1)
    rb = rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den == 0] = np.nan
    return out


def paired_spearman_bootstrap(
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    differences: list[tuple[str, str]] = (),
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Spearman ρ of every score against `truth`, plus differences between
    scores, all from **one** percentile bootstrap over conditions.

    Every resample draws the same conditions for every score, so a difference
    ρ_a − ρ_b gets a proper paired CI (run 2's Rule B). A resample is
    discarded for a quantity if any correlation that quantity needs is
    undefined there (constant input); the number of valid resamples is
    reported per quantity.

    `+inf` in `truth` is allowed and ranks as the worst value (run 2's rule
    for conditions whose landmark error is undefined).

    Returns ``{"rho": {name: ...}, "ci": {name: (lo, hi)}, "n_valid": {name: ...},
    "diff": {"a-b": {"delta", "ci", "n_valid"}}}``.
    """
    truth = np.asarray(truth, dtype=float)
    n = truth.shape[0]
    if n < 3:
        raise ValueError("need at least 3 conditions to compute a rank correlation")
    for name, s in scores.items():
        if np.asarray(s).shape != truth.shape:
            raise ValueError(f"score {name!r} has a different length from truth")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_truth = truth[idx]
    point = {k: float(_rowwise_spearman(np.asarray(s, float)[None], truth[None])[0]) for k, s in scores.items()}
    boot = {k: _rowwise_spearman(np.asarray(s, float)[idx], boot_truth) for k, s in scores.items()}

    alpha = (1.0 - ci) / 2.0
    lo_q, hi_q = alpha * 100, (1.0 - alpha) * 100

    def _ci(values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return float("nan"), float("nan")
        return float(np.percentile(values, lo_q)), float(np.percentile(values, hi_q))

    out: dict = {"rho": point, "ci": {}, "n_valid": {}, "diff": {}, "n_bootstrap": n_bootstrap}
    for k, b in boot.items():
        valid = b[np.isfinite(b)]
        out["ci"][k] = _ci(valid)
        out["n_valid"][k] = int(valid.size)
    for a, b in differences:
        both = np.isfinite(boot[a]) & np.isfinite(boot[b])
        d = boot[a][both] - boot[b][both]
        out["diff"][f"{a}-{b}"] = {
            "delta": point[a] - point[b],
            "ci": _ci(d),
            "n_valid": int(d.size),
        }
    return out
