"""Graded image degradations, applied to real frames before MediaPipe sees them.

Run 2's primary conditions (see the run-2 pre-registration) are real clean
recordings degraded in software at graded severities. Every condition reuses
the *same* frames as `clean`, which removes the take-to-take noise that made
run 1 underpowered, and gives each degraded frame a clean reference for the
paired landmark-error ground truth (`paired_landmark_error`).

These are **simulated** degradations of real footage, not real deployment
conditions, and results built on them must be labelled that way.

Each degradation is a frozen, JSON-describable value and deterministic for a
given (seed, frame index), so a condition can be re-created exactly from its
pre-registered description.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

KINDS = {
    "darken": "multiply intensity by `severity` in (0, 1]; lower is darker",
    "downscale": "resize so the frame is `severity` pixels wide; smaller is worse",
    "blur": "Gaussian blur with sigma = `severity` pixels",
    "noise": "add Gaussian noise with std = `severity` intensity levels (0-255)",
}


@dataclass(frozen=True)
class ImageDegradation:
    kind: str
    severity: float
    seed: int = 0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown degradation {self.kind!r}; choose one of {sorted(KINDS)}")
        if self.kind == "darken" and not 0 < self.severity <= 1:
            raise ValueError("darken severity must be in (0, 1]")
        if self.kind in ("downscale", "blur", "noise") and self.severity <= 0:
            raise ValueError(f"{self.kind} severity must be > 0")

    def describe(self) -> dict:
        return asdict(self)

    def __call__(self, frame_bgr: np.ndarray, frame_index: int) -> np.ndarray:
        import cv2

        if self.kind == "darken":
            out = np.rint(frame_bgr.astype(np.float64) * self.severity)
            return np.clip(out, 0, 255).astype(np.uint8)
        if self.kind == "downscale":
            h, w = frame_bgr.shape[:2]
            new_w = int(self.severity)
            if new_w >= w:
                return frame_bgr
            new_h = max(1, round(h * new_w / w))
            return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if self.kind == "blur":
            return cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=float(self.severity))
        # noise: seeded per frame so a condition is exactly reproducible
        rng = np.random.default_rng([self.seed, frame_index])
        out = frame_bgr.astype(np.float64) + rng.normal(0, self.severity, frame_bgr.shape)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
