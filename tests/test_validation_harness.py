import numpy as np
import pytest

from worldgap.validation.harness import ConditionResult, ValidationHarness


def _conditions(n):
    return [{"lighting": f"cond_{i}"} for i in range(n)]


def test_run_without_preregistration_raises():
    harness = ValidationHarness(min_conditions=3)
    results = [ConditionResult(c, gap_score=1.0, ground_truth_degradation=1.0) for c in _conditions(3)]
    with pytest.raises(RuntimeError):
        harness.run(results)


def test_preregistration_below_minimum_raises():
    harness = ValidationHarness(min_conditions=10)
    with pytest.raises(ValueError):
        harness.pre_register_conditions(_conditions(5))


def test_run_with_mismatched_conditions_raises():
    harness = ValidationHarness(min_conditions=3)
    conds = _conditions(3)
    harness.pre_register_conditions(conds)
    # Submitting a different, smaller set is exactly the cherry-picking pattern to block.
    bad_results = [ConditionResult(conds[0], gap_score=1.0, ground_truth_degradation=1.0)]
    with pytest.raises(ValueError, match="prohibits"):
        harness.run(bad_results)


def test_run_with_matching_conditions_succeeds():
    harness = ValidationHarness(min_conditions=3)
    conds = _conditions(12)
    harness.pre_register_conditions(conds)

    rng = np.random.default_rng(0)
    gap_scores = rng.uniform(size=12)
    # construct degradation to be monotonically related to gap_scores so the
    # test also sanity-checks that a real signal produces a positive rho
    degradation = gap_scores + rng.normal(scale=0.05, size=12)

    results = [
        ConditionResult(c, gap_score=float(g), ground_truth_degradation=float(d))
        for c, g, d in zip(conds, gap_scores, degradation)
    ]
    report = harness.run(results)
    assert report.n_conditions == 12
    assert report.spearman.rho > 0.5
