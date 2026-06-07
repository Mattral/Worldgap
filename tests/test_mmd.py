import numpy as np
import pytest

from worldgap.metrics.mmd import mmd_squared


def test_identical_distributions_give_near_zero_mmd():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 8))
    y = rng.normal(size=(300, 8))  # same distribution, different draw
    result = mmd_squared(x, y)
    assert abs(result.mmd_squared) < 0.05  # not exactly zero with finite samples, but small


def test_mmd_increases_with_separation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(300, 8))
    y_near = x + rng.normal(scale=0.1, size=(300, 8))
    y_far = x + 5.0

    near = mmd_squared(x, y_near)
    far = mmd_squared(x, y_far)
    assert far.mmd_squared > near.mmd_squared


def test_too_few_samples_raises():
    x = np.zeros((1, 4))
    y = np.zeros((5, 4))
    with pytest.raises(ValueError):
        mmd_squared(x, y)
