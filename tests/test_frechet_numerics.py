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
