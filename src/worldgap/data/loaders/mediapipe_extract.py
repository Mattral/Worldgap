"""Shared MediaPipe Holistic Landmarker extraction, per TECHNICAL_SPEC.md
Section 5.4 and Section 11 (API surface note: this targets the current Tasks
API, `mediapipe.tasks.python.vision.HolisticLandmarker`, not the legacy
`mp.solutions` API most existing tutorials still show).

Both `hagrid.py` and `egohands.py` delegate here rather than duplicating this
logic -- spec 5.4 requires both loaders to produce Rollout objects with an
identical feature-vector layout, since source and target domains only mean
anything if they're comparable in the same latent space to begin with.

What IS implemented and unit-tested here, with zero network access: the
conversion from a HolisticLandmarker detection result into the project's
PERCEPTION_FEATURE_LAYOUT-shaped array, including graceful handling of
frames where a hand/pose wasn't detected at all. Tests use small duck-typed
fake landmark/result objects (see tests/test_mediapipe_extract.py) rather
than real `mediapipe` types, so this test file runs without the `perception`
extra installed, matching how `landmarker` is dependency-injected below
rather than constructed internally.

What is NOT implemented, because it needs network access this sandbox's
allowlist blocks (`storage.googleapis.com`, where MediaPipe's `.task` model
bundles are hosted -- confirmed via a direct request, not assumed) and real
downloaded frame images: actually constructing a working `HolisticLandmarker`
and running it against real HaGRID/EgoHands frames. `extract_rollout_from_frames`
below takes an already-constructed `landmarker` as a parameter for exactly
this reason -- the caller (in an environment with real network access) is
responsible for downloading the model and constructing the landmarker; this
module only needs something that responds to `.detect(image)`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..rollout import PERCEPTION_FEATURE_LAYOUT, PERCEPTION_STATE_DIM, Rollout


def _landmark_block(landmarks, n_landmarks: int, dims: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Converts one detected landmark list (pose, left hand, or right hand)
    into a flat (n_landmarks * len(dims),) feature block plus a same-shaped
    presence mask.

    `landmarks` is falsy (None or an empty list) when MediaPipe didn't detect
    that part in this frame at all -- a real, common outcome (e.g. a hand
    briefly out of frame), not an error condition. That case gets zeros and
    an all-False presence mask for this block, exactly like the `presence_mask`
    convention the rest of the library already uses for synthetic occlusion
    (see `simulate_occlusion` in synthetic_perturb.py) -- MediaPipe dropout and
    synthetic occlusion MUST look the same downstream, since the whole point
    of spec 8.1's validation ground truth is comparing against real
    MediaPipe dropout rate.
    """
    n_dims = len(dims)
    block_len = n_landmarks * n_dims
    if not landmarks:
        return np.zeros(block_len, dtype=np.float64), np.zeros(block_len, dtype=bool)

    if len(landmarks) != n_landmarks:
        raise ValueError(
            f"expected {n_landmarks} landmarks, got {len(landmarks)} -- MediaPipe's "
            "topology for this landmark type should be fixed; a mismatch here means "
            "either the wrong landmarker output was passed in, or MediaPipe's model "
            "changed its output topology since this was written"
        )

    features = np.zeros(block_len, dtype=np.float64)
    for i, lm in enumerate(landmarks):
        for d, dim_name in enumerate(dims):
            features[i * n_dims + d] = getattr(lm, dim_name)
    return features, np.ones(block_len, dtype=bool)


def holistic_result_to_feature_vector(result) -> tuple[np.ndarray, np.ndarray]:
    """Converts one frame's `HolisticLandmarkerResult` (or any duck-typed
    equivalent with `.pose_landmarks`/`.left_hand_landmarks`/
    `.right_hand_landmarks` attributes) into a (PERCEPTION_STATE_DIM,) feature
    vector plus a same-shaped presence mask, per `PERCEPTION_FEATURE_LAYOUT`.
    """
    features = np.zeros(PERCEPTION_STATE_DIM, dtype=np.float64)
    presence = np.zeros(PERCEPTION_STATE_DIM, dtype=bool)

    attr_by_group = {
        "pose": "pose_landmarks",
        "left_hand": "left_hand_landmarks",
        "right_hand": "right_hand_landmarks",
    }
    for group_name, group in PERCEPTION_FEATURE_LAYOUT.items():
        start = group["start"]
        n_landmarks = group["n_landmarks"]
        dims = group["dims"]
        landmarks = getattr(result, attr_by_group[group_name], None)
        block_features, block_presence = _landmark_block(landmarks, n_landmarks, dims)
        end = start + len(block_features)
        features[start:end] = block_features
        presence[start:end] = block_presence

    return features, presence


def extract_rollout_from_frames(
    frame_paths: list[Path],
    landmarker,
    frame_rate_hz: float = 30.0,
    condition: dict | None = None,
    source: str = "real",
    metadata: dict | None = None,
) -> Rollout:
    """Runs `landmarker` (an already-constructed object with a `.detect(image)`
    method -- typically `mediapipe.tasks.python.vision.HolisticLandmarker`,
    but any duck-typed equivalent works, which is exactly what makes this
    testable without a real model file) over a sequence of frame image files
    and packages the landmark trajectory as a `Rollout`.

    Frames where nothing was detected are NOT dropped or interpolated --
    they're kept with `presence_mask=False` for the relevant columns, so
    dropout rate stays measurable (spec 8.1 needs real MediaPipe dropout as
    ground truth; silently patching over it here would destroy exactly the
    signal validation depends on).
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty -- nothing to extract")

    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            "mediapipe is required to load frame images and run the landmarker "
            "(worldgap[perception]). Install with: pip install worldgap[perception]"
        ) from e

    n_frames = len(frame_paths)
    states = np.zeros((n_frames, PERCEPTION_STATE_DIM), dtype=np.float64)
    presence_mask = np.zeros((n_frames, PERCEPTION_STATE_DIM), dtype=bool)

    for i, frame_path in enumerate(frame_paths):
        image = mp.Image.create_from_file(str(frame_path))
        result = landmarker.detect(image)
        features, presence = holistic_result_to_feature_vector(result)
        states[i] = features
        presence_mask[i] = presence

    timestamps_ms = np.arange(n_frames) * (1000.0 / frame_rate_hz)

    return Rollout(
        modality="perception",
        source=source,
        condition=condition or {},
        frame_rate_hz=frame_rate_hz,
        states=states,
        presence_mask=presence_mask.astype(np.float64),
        timestamps_ms=timestamps_ms,
        metadata=metadata or {},
    )
