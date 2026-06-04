"""Validation Harness, per TECHNICAL_SPEC.md Section 8.

The anti-circularity requirement (8.1) is enforced by convention: callers MUST
supply a ground_truth_degradation that was computed independently of the world
model (e.g. raw MediaPipe confidence/dropout for V1) — this module cannot check
that for you, it can only refuse to run without the supporting data.

The anti-cherry-picking requirement (8.3) IS enforced here in code: conditions
MUST be pre-registered before results are submitted, and `run()` refuses to
proceed if the submitted condition set doesn't exactly match what was
pre-registered. Selecting or trimming conditions after seeing results is not
just discouraged — it raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .stats import SpearmanResult, spearman_with_bootstrap_ci

MIN_CONDITIONS = 10  # spec 8.2: at least 10-15 distinct conditions required


@dataclass
class ConditionResult:
    condition: dict[str, Any]
    gap_score: float
    ground_truth_degradation: float


@dataclass
class ValidationReport:
    conditions: list[ConditionResult]
    spearman: SpearmanResult
    n_conditions: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_conditions = len(self.conditions)


def _condition_key(condition: dict[str, Any]) -> tuple:
    return tuple(sorted(condition.items()))


class ValidationHarness:
    def __init__(self, min_conditions: int = MIN_CONDITIONS):
        self.min_conditions = min_conditions
        self._registered_conditions: list[dict[str, Any]] | None = None

    def pre_register_conditions(self, conditions: list[dict[str, Any]]) -> None:
        """MUST be called before run() — spec 8.3. Fixes the condition set before
        any gap score or degradation number is looked at, so the correlation
        analysis can't be quietly narrowed to whichever conditions look best.
        """
        if len(conditions) < self.min_conditions:
            raise ValueError(
                f"spec 8.2 requires at least {self.min_conditions} distinct conditions, "
                f"got {len(conditions)}"
            )
        keys = [_condition_key(c) for c in conditions]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate conditions in the pre-registered set")
        self._registered_conditions = conditions

    def run(self, condition_results: list[ConditionResult]) -> ValidationReport:
        if self._registered_conditions is None:
            raise RuntimeError(
                "call pre_register_conditions() before run() — spec 8.3 anti-cherry-picking "
                "requirement; there is no override for this"
            )

        submitted_keys = {_condition_key(c.condition) for c in condition_results}
        registered_keys = {_condition_key(c) for c in self._registered_conditions}
        if submitted_keys != registered_keys:
            missing = registered_keys - submitted_keys
            extra = submitted_keys - registered_keys
            raise ValueError(
                "submitted condition_results do not exactly match the pre-registered set "
                f"(missing={len(missing)}, unexpected extra={len(extra)}) — this is exactly "
                "the selective-reporting pattern spec 8.3 prohibits"
            )

        gap_scores = np.array([c.gap_score for c in condition_results])
        degradation = np.array([c.ground_truth_degradation for c in condition_results])
        spearman = spearman_with_bootstrap_ci(gap_scores, degradation)

        return ValidationReport(conditions=condition_results, spearman=spearman)
