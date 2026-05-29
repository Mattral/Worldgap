"""Frechet-style divergence metric, per TECHNICAL_SPEC.md Section 7.1 / 7.3.

FD(A, B) = ||mu_A - mu_B||^2 + Tr(Sigma_A + Sigma_B - 2 * sqrtm(Sigma_A @ Sigma_B))

Two requirements from the spec that are NOT optional:
  1. Covariance MUST use Ledoit-Wolf shrinkage, not naive empirical covariance
     (spec 7.1, 12.9) — with rollout counts in the hundreds rather than tens of
     thousands, naive covariance is poorly conditioned or singular.
  2. matrix sqrt MUST have its small complex component discarded, not treated
     as an error (spec 7.1) — this is standard floating-point noise in sqrtm,
     not a sign of a broken computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg
from sklearn.covariance import LedoitWolf

# spec 7.3: n >= 5 * latent_dim per domain, else confidence is "low".
MIN_SAMPLES_PER_DIM = 5


@dataclass
class FrechetResult:
    distance: float
    n_source: int
    n_target: int
    latent_dim: int
    confidence: str  # "low" | "medium" | "high"
    sqrtm_had_complex_component: bool


def _confidence(n_source: int, n_target: int, latent_dim: int) -> str:
    min_n = min(n_source, n_target)
    if min_n < MIN_SAMPLES_PER_DIM * latent_dim:
        return "low"
    if min_n < 2 * MIN_SAMPLES_PER_DIM * latent_dim:
        return "medium"
    return "high"


def frechet_distance(source_latents: np.ndarray, target_latents: np.ndarray) -> FrechetResult:
    """
    Args:
        source_latents: (n_source, latent_dim) array of pooled latent summary vectors.
        target_latents: (n_target, latent_dim) array of pooled latent summary vectors.

    Returns:
        FrechetResult with the distance plus the sample-size/confidence metadata
        that spec 7.3 requires be reported alongside every gap score.
    """
    if source_latents.ndim != 2 or target_latents.ndim != 2:
        raise ValueError("latents must be 2D arrays of shape (n_samples, latent_dim)")
    if source_latents.shape[1] != target_latents.shape[1]:
        raise ValueError(
            "source and target latents must share latent_dim: "
            f"{source_latents.shape[1]} vs {target_latents.shape[1]}"
        )

    n_source, latent_dim = source_latents.shape
    n_target = target_latents.shape[0]

    mu_source = source_latents.mean(axis=0)
    mu_target = target_latents.mean(axis=0)

    # Ledoit-Wolf shrinkage covariance — MUST NOT be replaced with np.cov here (spec 7.1).
    cov_source = LedoitWolf().fit(source_latents).covariance_
    cov_target = LedoitWolf().fit(target_latents).covariance_

    mean_term = float(np.sum((mu_source - mu_target) ** 2))

    cov_product = cov_source @ cov_target
    sqrt_cov_product = linalg.sqrtm(cov_product)

    had_complex = False
    if np.iscomplexobj(sqrt_cov_product):
        imag_magnitude = np.max(np.abs(sqrt_cov_product.imag))
        real_magnitude = np.max(np.abs(sqrt_cov_product.real)) + 1e-12
        if imag_magnitude / real_magnitude > 1e-3:
            # Large imaginary component is NOT the expected floating-point noise
            # case — surface it rather than silently discarding.
            raise FloatingPointError(
                "sqrtm produced a large imaginary component "
                f"(ratio={imag_magnitude / real_magnitude:.4g}); covariance matrices "
                "may be ill-conditioned even after shrinkage — investigate before trusting "
                "this result."
            )
        had_complex = True
        sqrt_cov_product = sqrt_cov_product.real

    trace_term = float(np.trace(cov_source) + np.trace(cov_target) - 2 * np.trace(sqrt_cov_product))
    distance = mean_term + trace_term

    return FrechetResult(
        distance=distance,
        n_source=n_source,
        n_target=n_target,
        latent_dim=latent_dim,
        confidence=_confidence(n_source, n_target, latent_dim),
        sqrtm_had_complex_component=had_complex,
    )
