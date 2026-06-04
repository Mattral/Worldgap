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
