"""Maximum Mean Discrepancy cross-check, per TECHNICAL_SPEC.md Section 7.2.

Frechet distance (frechet.py) assumes approximately Gaussian latents. MMD makes
no such assumption. The spec requires both be reported together — disagreement
between them is itself a diagnostic signal (non-Gaussian latent structure) and
MUST be surfaced, not hidden (spec 7.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MMDResult:
    mmd_squared: float
    """The UNBIASED MMD^2 estimate, which **can legitimately be negative**.

    This looks like a bug and is not. The unbiased estimator (below: the
    kernel-matrix diagonals are excluded from the within-domain sums) is
    unbiased for the true MMD^2, and an unbiased estimator of a non-negative
    quantity must take negative values whenever the true value is near zero --
    otherwise it would be biased upward. See Gretton et al. (2012), "A Kernel
    Two-Sample Test", JMLR 13.

    Interpretation: a negative value means "no detectable difference between
    these two domains at this sample size", not "a negative distance". The
    magnitude of a negative value is sampling noise and carries no meaning, so
    do not rank conditions by it in that regime.

    The biased (V-statistic) estimator is always non-negative but overstates
    small gaps, which is the worse failure for this tool's purpose: a
    domain-gap number that can never say "I see nothing" is not much use.
    """
    bandwidth: float

    @property
    def below_noise_floor(self) -> bool:
        """True when the estimate is negative, i.e. indistinguishable from
        no difference at this sample size.
        """
        return self.mmd_squared < 0.0


def _median_heuristic_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.vstack([x, y])
    n = combined.shape[0]
    # Subsample for the pairwise-distance computation if n is large; not a concern
    # at the rollout-count scale this tool operates at (spec: hundreds, not thousands).
    dists = np.linalg.norm(combined[:, None, :] - combined[None, :, :], axis=-1)
    iu = np.triu_indices(n, k=1)
    median = np.median(dists[iu])
    return float(median) if median > 0 else 1.0


def _rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    sq_dists = np.sum(x**2, axis=1)[:, None] + np.sum(y**2, axis=1)[None, :] - 2 * x @ y.T
    return np.exp(-sq_dists / (2 * bandwidth**2))


def mmd_squared(
    source_latents: np.ndarray, target_latents: np.ndarray, bandwidth: float | None = None
) -> MMDResult:
    """Unbiased MMD^2 estimator with an RBF kernel, bandwidth via median heuristic
    unless explicitly provided (spec 7.2).
    """
    if source_latents.shape[1] != target_latents.shape[1]:
        raise ValueError("source and target latents must share latent_dim")

    if bandwidth is None:
        bandwidth = _median_heuristic_bandwidth(source_latents, target_latents)

    n, m = source_latents.shape[0], target_latents.shape[0]
    if n < 2 or m < 2:
        raise ValueError("MMD requires at least 2 samples per domain for the unbiased estimator")

    k_xx = _rbf_kernel(source_latents, source_latents, bandwidth)
    k_yy = _rbf_kernel(target_latents, target_latents, bandwidth)
    k_xy = _rbf_kernel(source_latents, target_latents, bandwidth)

    # Unbiased: exclude diagonal terms in the within-domain sums.
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    sum_xy = k_xy.sum() / (n * m)

    value = float(sum_xx + sum_yy - 2 * sum_xy)
    return MMDResult(mmd_squared=value, bandwidth=bandwidth)
