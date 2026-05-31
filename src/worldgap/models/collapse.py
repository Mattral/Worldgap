"""Representation-collapse safeguard, per TECHNICAL_SPEC.md Section 6.3 / edge case 12.7.

JEPA-family self-supervised models have a known failure mode: the predictor
learns a trivial constant output and the target encoder's latents collapse to
near-zero variance. This MUST be an automated check during training, not a
manual post-hoc inspection (spec 6.3).
"""

from __future__ import annotations


class CollapseSafeguard:
    def __init__(self, variance_threshold: float, patience_checks: int):
        self.variance_threshold = variance_threshold
        self.patience_checks = patience_checks
        self._history: list[float] = []

    def record(self, latent_variance: float) -> None:
        self._history.append(latent_variance)

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def has_collapsed(self) -> bool:
        """True iff the last `patience_checks` recorded variances are all below
        `variance_threshold` — i.e. collapse is sustained, not a single noisy dip.
        """
        if len(self._history) < self.patience_checks:
            return False
        recent = self._history[-self.patience_checks :]
        return all(v < self.variance_threshold for v in recent)
