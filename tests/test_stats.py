"""Tests for validation/stats.py, per TECHNICAL_SPEC.md Section 8.2.

Previously this module was only exercised indirectly through
test_validation_harness.py. This file tests spearman_with_bootstrap_ci
directly, including the degenerate-input guard added after a bare
np.percentile call on an empty bootstrap list would otherwise crash with an
unhelpful error.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldgap.validation.stats import spearman_with_bootstrap_ci


def test_perfect_monotonic_relationship_gives_rho_near_one():
    gap_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    degradation = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = spearman_with_bootstrap_ci(gap_scores, degradation, n_bootstrap=200, seed=0)
    assert result.rho > 0.99
    assert result.n_conditions == 5
    assert result.ci_low <= result.rho <= result.ci_high


def test_ci_report_includes_required_metadata():
    rng = np.random.default_rng(1)
    gap_scores = rng.normal(size=12)
    degradation = rng.normal(size=12)
    result = spearman_with_bootstrap_ci(gap_scores, degradation, n_bootstrap=500, seed=1)
    # spec 8.4: a bare rho is not acceptable -- CI and n_conditions MUST be present.
    assert hasattr(result, "ci_low")
    assert hasattr(result, "ci_high")
    assert result.n_conditions == 12
    assert result.n_bootstrap <= 500


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="same length"):
        spearman_with_bootstrap_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_too_few_conditions_raise():
    with pytest.raises(ValueError, match="at least 3"):
        spearman_with_bootstrap_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_constant_inputs_raise_clear_error_instead_of_crashing():
    """Regression test: if gap_scores and/or ground_truth_degradation are
    constant, every bootstrap resample's Spearman rho is NaN (no variation to
    rank), and boot_rhos ends up empty. Before the fix this hit
    np.percentile([], ...) and raised a confusing IndexError; it must now
    raise a clear, actionable ValueError instead.
    """
    gap_scores = np.array([1.0, 1.0, 1.0, 1.0])
    degradation = np.array([2.0, 5.0, 3.0, 9.0])
    with pytest.raises(ValueError, match="undefined \\(NaN\\) Spearman rho"):
        spearman_with_bootstrap_ci(gap_scores, degradation, n_bootstrap=50, seed=0)


# -- paired bootstrap (run 2's Rules A and B) ----------------------------------------


def _three_scores(seed: int = 1, n: int = 24):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    return truth, truth + rng.normal(0, 0.5, n), truth + rng.normal(0, 1.5, n)


def test_paired_bootstrap_point_estimates_match_scipy():
    from scipy.stats import spearmanr

    from worldgap.validation.stats import paired_spearman_bootstrap

    truth, a, b = _three_scores()
    out = paired_spearman_bootstrap(truth, {"a": a, "b": b}, [("a", "b")], n_bootstrap=500)
    assert out["rho"]["a"] == pytest.approx(spearmanr(a, truth)[0])
    assert out["rho"]["b"] == pytest.approx(spearmanr(b, truth)[0])
    assert out["diff"]["a-b"]["delta"] == pytest.approx(out["rho"]["a"] - out["rho"]["b"])


def test_paired_bootstrap_ci_agrees_with_the_single_score_bootstrap():
    """Same seed, same resampling scheme: the paired function's CI for one score
    must equal the existing harness's, so Rule A uses the established method."""
    from worldgap.validation.stats import paired_spearman_bootstrap

    truth, a, _ = _three_scores()
    paired = paired_spearman_bootstrap(truth, {"a": a}, n_bootstrap=2000, seed=0)
    single = spearman_with_bootstrap_ci(a, truth, n_bootstrap=2000, seed=0)
    assert paired["ci"]["a"] == pytest.approx((single.ci_low, single.ci_high))


def test_paired_difference_ci_is_narrower_than_unpaired_when_scores_correlate():
    """Pairing is the point of Rule B: two correlated scores measured on the same
    resampled conditions give a tighter CI on their difference."""
    from worldgap.validation.stats import paired_spearman_bootstrap

    rng = np.random.default_rng(3)
    truth = rng.normal(size=24)
    shared = truth + rng.normal(0, 0.8, 24)
    a, b = shared + rng.normal(0, 0.2, 24), shared + rng.normal(0, 0.2, 24)
    paired = paired_spearman_bootstrap(truth, {"a": a, "b": b}, [("a", "b")], n_bootstrap=3000)
    lo, hi = paired["diff"]["a-b"]["ci"]
    # unpaired: independent resamples for each score
    ra = paired_spearman_bootstrap(truth, {"a": a}, n_bootstrap=3000, seed=11)
    rb = paired_spearman_bootstrap(truth, {"b": b}, n_bootstrap=3000, seed=12)
    unpaired_width = (ra["ci"]["a"][1] - ra["ci"]["a"][0]) + (rb["ci"]["b"][1] - rb["ci"]["b"][0])
    assert hi - lo < unpaired_width / 2


def test_paired_bootstrap_ranks_infinite_truth_as_worst():
    from worldgap.validation.stats import paired_spearman_bootstrap

    truth = np.array([0.1, 0.2, 0.3, np.inf])
    score = np.array([1.0, 2.0, 3.0, 4.0])
    assert paired_spearman_bootstrap(truth, {"s": score}, n_bootstrap=200)["rho"]["s"] == pytest.approx(1.0)


def test_paired_bootstrap_reports_valid_resample_counts_and_rejects_bad_input():
    from worldgap.validation.stats import paired_spearman_bootstrap

    truth, a, _ = _three_scores(n=5)
    out = paired_spearman_bootstrap(truth, {"a": a, "flat": np.ones(5)}, [("a", "flat")], n_bootstrap=300)
    assert 0 < out["n_valid"]["a"] <= 300
    assert out["n_valid"]["flat"] == 0 and np.isnan(out["rho"]["flat"])
    assert out["diff"]["a-flat"]["n_valid"] == 0
    with pytest.raises(ValueError, match="at least 3"):
        paired_spearman_bootstrap(np.array([1.0, 2.0]), {"a": np.array([1.0, 2.0])})
    with pytest.raises(ValueError, match="different length"):
        paired_spearman_bootstrap(truth, {"a": a[:3]})
