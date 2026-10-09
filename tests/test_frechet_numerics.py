import numpy as np
import pytest

from worldgap.metrics.frechet import MIN_SAMPLES_PER_DIM, frechet_distance


def test_identical_distributions_give_near_zero_distance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 8))
    result = frechet_distance(x, x.copy())
    assert result.distance == pytest.approx(0.0, abs=1e-8)


def test_distance_increases_with_mean_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 8))
    y_near = x + rng.normal(scale=0.01, size=(500, 8))
    y_far = x + 5.0

    near = frechet_distance(x, y_near)
    far = frechet_distance(x, y_far)
    assert far.distance > near.distance


def test_confidence_flag_low_when_undersampled():
    latent_dim = 32
    n_small = MIN_SAMPLES_PER_DIM * latent_dim - 1  # just under the spec 7.3 threshold
    rng = np.random.default_rng(2)
    x = rng.normal(size=(n_small, latent_dim))
    y = rng.normal(size=(n_small, latent_dim))
    result = frechet_distance(x, y)
    assert result.confidence == "low"


def test_confidence_flag_high_when_well_sampled():
    latent_dim = 8
    n_large = 2 * MIN_SAMPLES_PER_DIM * latent_dim + 50
    rng = np.random.default_rng(3)
    x = rng.normal(size=(n_large, latent_dim))
    y = rng.normal(size=(n_large, latent_dim))
    result = frechet_distance(x, y)
    assert result.confidence == "high"


def test_mismatched_latent_dim_raises():
    x = np.zeros((10, 4))
    y = np.zeros((10, 5))
    with pytest.raises(ValueError):
        frechet_distance(x, y)


def test_result_reports_required_metadata():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 4))
    y = rng.normal(size=(60, 4))
    result = frechet_distance(x, y)
    # spec 7.3: n_source, n_target, latent_dim, confidence are required output, not optional.
    assert result.n_source == 50
    assert result.n_target == 60
    assert result.latent_dim == 4
    assert result.confidence in {"low", "medium", "high"}


# -- symmetric formulation (run-2 blind test: singular covariance) --------------------


def _old_trace_sqrt(cov_a, cov_b):
    from scipy import linalg

    return float(np.trace(linalg.sqrtm(cov_a @ cov_b).real))


def test_symmetric_trace_matches_sqrtm_on_well_conditioned_covariances():
    from worldgap.metrics.frechet import _trace_sqrt_product

    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=(40, 8))
        b = rng.normal(size=(40, 8)) * rng.uniform(0.5, 2.0, 8)
        ca, cb = np.cov(a.T), np.cov(b.T)
        assert _trace_sqrt_product(ca, cb) == pytest.approx(_old_trace_sqrt(ca, cb), rel=1e-9)


def test_zero_target_covariance_gives_the_exact_closed_form():
    """Every target latent identical (e.g. every window empty): Σ_B = 0, and
    FD = ||μ_A − μ_B||² + tr(Σ_A) exactly. The old sqrtm ratio test raised here."""
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(1)
    source = rng.normal(size=(30, 6))
    target = np.tile(rng.normal(size=6), (18, 1))
    result = frechet_distance(source, target)
    cov_a = LedoitWolf().fit(source).covariance_
    expected = float(np.sum((source.mean(0) - target[0]) ** 2) + np.trace(cov_a))
    assert result.distance == pytest.approx(expected, rel=1e-12)


def test_near_degenerate_targets_are_finite_and_continuous():
    """17 of 18 windows identical: no blow-up, and the distance moves smoothly
    toward the fully degenerate value."""
    rng = np.random.default_rng(2)
    source = rng.normal(size=30 * 6).reshape(30, 6)
    point = rng.normal(size=6)
    distances = []
    for spread in (1e-1, 1e-3, 1e-6, 0.0):
        target = np.tile(point, (18, 1))
        target[0] += spread * rng.normal(size=6)
        distances.append(frechet_distance(source, target).distance)
    assert np.isfinite(distances).all()
    assert abs(distances[-2] - distances[-1]) < 1e-3


def test_distance_is_non_negative_and_symmetric():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(40, 5)), rng.normal(1.0, 2.0, size=(40, 5))
    ab, ba = frechet_distance(a, b).distance, frechet_distance(b, a).distance
    assert ab >= 0 and ab == pytest.approx(ba, rel=1e-9)
