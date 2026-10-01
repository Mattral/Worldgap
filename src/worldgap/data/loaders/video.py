"""Video loader -- the real source of V1 trajectories.

Why this module exists: V1's claim is about how camera-based hand tracking
degrades *over time* under deployment conditions. Tremor is an oscillation at a
frequency. Dropout is a run of consecutive frames where MediaPipe lost the
hand. Reduced range of motion is a shrunken trajectory envelope. None of those
are properties of a photograph, so none of them can be measured on a still
image dataset, however large. They need video.

This module reads an actual video file, runs the landmarker over its decoded
frames in order, and produces a `Rollout` whose time axis is real -- tagged
`temporal_provenance="video"`.

The practical consequence worth knowing: **you do not need a dataset download
to do a real V1 run.** A webcam and this module are sufficient. Recording the
same gesture set twice -- once in good light, once in the deployment-like
condition you care about -- gives a genuine source/target pair of real measured
trajectories, which is a stronger artifact than anything HaGRID can provide for
this particular claim. See `docs/v1_real_data_runbook.md`.

Video decoding uses OpenCV, which ships as a dependency of `mediapipe`, so the
`perception` extra covers both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..rollout import (
    PERCEPTION_STATE_DIM,
    TEMPORAL_PROVENANCE_KEY,
    WALL_CLOCK_PROVENANCES,
    Rollout,
    split_into_windows,
)

__all__ = [
    "VIDEO_SUFFIXES",
    "extract_rollout_from_video",
    "landmark_quality_ground_truth",
    "list_videos",
    "split_into_windows",  # re-exported: it is modality-agnostic and lives in rollout.py
]
from .mediapipe_extract import holistic_result_to_feature_vector

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def list_videos(root: Path) -> list[Path]:
    """Returns video files under `root`, recursively, sorted for reproducibility."""
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def extract_rollout_from_video(
    video_path: Path,
    landmarker,
    condition: dict | None = None,
    metadata: dict | None = None,
    max_frames: int | None = None,
    stride: int = 1,
    use_video_mode: bool = True,
) -> Rollout:
    """Decodes `video_path` and runs `landmarker` over its frames in order.

    Args:
        landmarker: a constructed `HolisticLandmarker`, or any duck-typed object
            exposing `.detect(image)` and (if `use_video_mode`) `.detect_for_video(
            image, timestamp_ms)`. Injected rather than constructed here so this
            module is testable without a model bundle.
        stride: keep every Nth decoded frame. The rollout's `frame_rate_hz` is
            divided accordingly, so the time axis stays truthful.
        use_video_mode: call `detect_for_video()` when available. MediaPipe's
            VIDEO running mode tracks across frames rather than re-detecting each
            one independently, which is both faster and closer to how a live
            deployment behaves. Falls back to `detect()` automatically. A
            VIDEO-mode landmarker must be fresh for each file: timestamps
            restart at 0 per video and MediaPipe requires them to increase.

    Frames where nothing was detected are kept with `presence_mask=False`, never
    dropped or interpolated -- spec 8.1's ground truth *is* that dropout, so
    patching over it would destroy the signal validation depends on.

    Raises rather than returning an empty rollout if the file decodes to nothing,
    since a silently-empty rollout would sail through the rest of the pipeline
    and quietly shrink `n_source`.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading video needs mediapipe (which brings OpenCV): "
            "pip install 'worldgap[perception]'"
        ) from e

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open {video_path}")

    native_fps = capture.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0 or not np.isfinite(native_fps):
        raise ValueError(
            f"{video_path} reports no usable frame rate (got {native_fps!r}). "
            "Refusing to assume 30 fps: frame_rate_hz is what converts a tremor "
            "frequency in Hz into frames, so guessing it silently corrupts every "
            "time-domain comparison. Re-encode the file with an explicit rate, or "
            "construct the Rollout yourself with a rate you actually know."
        )

    states: list[np.ndarray] = []
    presences: list[np.ndarray] = []
    timestamps_ms: list[float] = []

    frame_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = frame_index * (1000.0 / native_fps)
                if use_video_mode and hasattr(landmarker, "detect_for_video"):
                    result = landmarker.detect_for_video(image, int(ts_ms))
                else:
                    result = landmarker.detect(image)
                features, presence = holistic_result_to_feature_vector(result)
                states.append(features)
                presences.append(presence)
                timestamps_ms.append(ts_ms)
                if max_frames is not None and len(states) >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()

    if not states:
        raise ValueError(f"{video_path} decoded to zero usable frames")

    return Rollout(
        modality="perception",
        source="real",
        condition={**(condition or {}), "video": video_path.name},
        frame_rate_hz=native_fps / stride,
        states=np.stack(states),
        presence_mask=np.stack(presences).astype(np.float64),
        timestamps_ms=np.asarray(timestamps_ms, dtype=np.float64),
        metadata={
            **(metadata or {}),
            TEMPORAL_PROVENANCE_KEY: "video",
            "native_fps": float(native_fps),
            "stride": stride,
        },
    )


def landmark_quality_ground_truth(rollout: Rollout) -> dict[str, float]:
    """Spec 8.1 ground-truth signals, computed from MediaPipe's own output.

    Spec 8.1 requires validation ground truth that is **independent of the
    world model under test**. These signals qualify: they come straight from
    the pose estimator's own reporting, never touch the encoder, and are
    exactly what a deployed confidence-threshold safety layer would be
    thresholding on.

    Returns:
        hand_dropout_rate: fraction of frames in which neither hand was
            detected at all. The bluntest and most operationally meaningful
            failure -- a gesture controller with no hand has nothing to act on.
        any_dropout_rate: fraction of frames with any missing block (pose or
            either hand).
        mean_pose_visibility: mean of MediaPipe's own per-landmark visibility
            score over the pose block, across present frames only. Lower means
            the estimator itself is less sure.
        longest_dropout_run_s: duration of the longest consecutive stretch with
            no hand. Two short blips and one long blackout can share a dropout
            rate while being very different to control through, so the run
            length is reported separately rather than averaged away.

    Raises if the rollout has no ordered time axis, because
    `longest_dropout_run_s` is meaningless without one -- see
    `docs/temporal_provenance.md`.
    """
    from ..rollout import PERCEPTION_FEATURE_LAYOUT

    if rollout.modality != "perception":
        raise ValueError(f"expected a perception rollout, got {rollout.modality!r}")
    if rollout.states.shape[1] != PERCEPTION_STATE_DIM:
        raise ValueError(
            f"expected D={PERCEPTION_STATE_DIM}, got D={rollout.states.shape[1]}"
        )
    if rollout.temporal_provenance not in WALL_CLOCK_PROVENANCES:
        raise ValueError(
            "landmark_quality_ground_truth() needs a real time axis (it reports a "
            f"dropout run length in seconds); this rollout's temporal_provenance is "
            f"{rollout.temporal_provenance!r}. An ordered-but-unclocked axis such as "
            "'quasi_static_sweep' is not enough. See docs/temporal_provenance.md."
        )

    mask = np.asarray(rollout.presence_mask) > 0

    def _block(name: str) -> slice:
        g = PERCEPTION_FEATURE_LAYOUT[name]
        return slice(g["start"], g["start"] + g["n_landmarks"] * len(g["dims"]))

    left = mask[:, _block("left_hand")].any(axis=1)
    right = mask[:, _block("right_hand")].any(axis=1)
    pose = mask[:, _block("pose")].any(axis=1)

    no_hand = ~(left | right)
    n_frames = mask.shape[0]

    # longest run of consecutive hand-less frames
    longest_run = 0
    current = 0
    for missing in no_hand:
        current = current + 1 if missing else 0
        longest_run = max(longest_run, current)

    pose_g = PERCEPTION_FEATURE_LAYOUT["pose"]
    vis_cols = np.arange(
        pose_g["start"] + pose_g["dims"].index("visibility"),
        pose_g["start"] + pose_g["n_landmarks"] * len(pose_g["dims"]),
        len(pose_g["dims"]),
    )
    present_pose = rollout.states[pose, :][:, vis_cols]
    mean_visibility = float(present_pose.mean()) if present_pose.size else 0.0

    fps = rollout.frame_rate_hz or float("nan")
    return {
        "hand_dropout_rate": float(no_hand.mean()),
        "any_dropout_rate": float((~(left & right & pose)).mean()),
        "mean_pose_visibility": mean_visibility,
        "longest_dropout_run_s": float(longest_run / fps) if fps else float("nan"),
        "n_frames": float(n_frames),
    }
