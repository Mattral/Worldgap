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
