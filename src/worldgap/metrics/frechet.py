"""Frechet-style divergence metric, per TECHNICAL_SPEC.md Section 7.1 / 7.3.

FD(A, B) = ||mu_A - mu_B||^2 + Tr(Sigma_A + Sigma_B - 2 * sqrtm(Sigma_A @ Sigma_B))

Covariance MUST use Ledoit-Wolf shrinkage, not naive empirical covariance
(spec 7.1, 12.9): with rollout counts in the hundreds rather than tens of
thousands, naive covariance is poorly conditioned or singular.

The trace of the matrix square root is computed with the symmetric
formulation tr((Σ_B^½ Σ_A Σ_B^½)^½) = Σᵢ √λᵢ (see `_trace_sqrt_product`), not
`sqrtm` of the non-symmetric product. This is a documented deviation from spec
7.1, which describes discarding `sqrtm`'s small complex component: there is no
complex component to discard. The old ratio test that judged when that
component was "small" misfired whenever a covariance was singular, because it
divided by a near-zero real part. Well-conditioned results agree with the old
computation to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    # Always False since the symmetric formulation (there is no complex
    # component); kept so code written against 0.2.0 still works.
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
    trace_term = float(
        np.trace(cov_source) + np.trace(cov_target) - 2 * _trace_sqrt_product(cov_source, cov_target)
    )
    distance = mean_term + trace_term

    return FrechetResult(
        distance=distance,
        n_source=n_source,
        n_target=n_target,
        latent_dim=latent_dim,
        confidence=_confidence(n_source, n_target, latent_dim),
        sqrtm_had_complex_component=False,
    )


def _psd_sqrt(cov: np.ndarray) -> np.ndarray:
    """Symmetric square root of a symmetric PSD matrix via eigh, with tiny
    negative eigenvalues (floating-point noise) clipped to zero."""
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def _trace_sqrt_product(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """tr((Σ_a Σ_b)^½), computed as tr((Σ_b^½ Σ_a Σ_b^½)^½) = Σᵢ √λᵢ.

    Σ_a Σ_b is similar to the symmetric PSD matrix Σ_b^½ Σ_a Σ_b^½, so they
    share eigenvalues, which are real and non-negative. Taking `sqrtm` of the
    non-symmetric product instead (the original implementation) produces
    complex round-off that has to be judged and discarded, and that judgement
    breaks down when a covariance is singular. With Σ_b = 0, e.g. a condition
    in which every window encoded to the same latent, this returns exactly 0,
    so the distance reduces to ||μ_a − μ_b||² + tr(Σ_a) as it should.
    """
    root_b = _psd_sqrt(cov_b)
    middle = root_b @ cov_a @ root_b
    eig = np.linalg.eigvalsh((middle + middle.T) / 2)
    return float(np.sum(np.sqrt(np.clip(eig, 0.0, None))))
