#!/usr/bin/env python3
"""Preflight check for a real V1 run. Run this BEFORE recording anything.

Every failure it reports is one that would otherwise surface partway through a
capture session, which is the expensive moment to find out.

    python scripts/check_mediapipe_setup.py --model ./models/holistic_landmarker.task

Exits 0 if everything needed for a real V1 run is present, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

OK = "  ok   "
BAD = " FAIL  "
WARN = " warn  "


def _report(status: str, message: str) -> None:
    print(f"[{status}] {message}")


def check_mediapipe() -> bool:
    try:
        import mediapipe as mp
    except ImportError:
        _report(BAD, "mediapipe is not installed. pip install 'worldgap[perception]'")
        return False

    version = getattr(mp, "__version__", "unknown")
    try:
        from mediapipe.tasks.python import vision
    except ImportError:
        _report(BAD, f"mediapipe {version} has no mediapipe.tasks.python.vision")
        return False

    if not hasattr(vision, "HolisticLandmarker"):
        _report(
            BAD,
            f"mediapipe {version} does NOT provide vision.HolisticLandmarker.\n"
            "         This is the known 0.10.x gap: those wheels ship only "
            "PoseLandmarker and\n"
            "         HandLandmarker, and the legacy mp.solutions API is gone. "
            "Upgrade:\n"
            "           pip install -U 'mediapipe>=1.0'",
        )
        return False

    _report(OK, f"mediapipe {version} provides vision.HolisticLandmarker")
    return True


def check_model_bundle(model_path: Path) -> bool:
    if not model_path.exists():
        _report(
            BAD,
            f"model bundle not found at {model_path}\n"
            f"         Download it (about 130 MB) with:\n"
            f"           curl -L -o {model_path} \\\n"
            f"             {MODEL_URL}",
        )
        return False

    size_mb = model_path.stat().st_size / 1e6
    if size_mb < 1.0:
        _report(
            BAD,
            f"{model_path} is only {size_mb:.2f} MB -- that is almost certainly an "
            "HTML error page saved under a .task name, not the model. Delete it and "
            "re-download.",
        )
        return False

    _report(OK, f"model bundle present at {model_path} ({size_mb:.0f} MB)")
    return True


def check_landmarker_constructs(model_path: Path) -> bool:
    """The check that actually matters: build the thing and run it on one frame."""
    try:
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:
        _report(BAD, f"import failed: {e}")
        return False

    try:
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
        )
        with vision.HolisticLandmarker.create_from_options(options) as landmarker:
            blank = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.zeros((256, 256, 3), dtype=np.uint8),
            )
            result = landmarker.detect(blank)
    except Exception as e:  # noqa: BLE001 - we want to report anything at all here
        _report(BAD, f"could not construct or run HolisticLandmarker: {type(e).__name__}: {e}")
        return False

    for attr in ("pose_landmarks", "left_hand_landmarks", "right_hand_landmarks"):
        if not hasattr(result, attr):
            _report(BAD, f"result has no .{attr} -- worldgap's feature layout expects it")
            return False

    _report(OK, "HolisticLandmarker constructs and runs (blank frame detected nothing, as expected)")
    return True


def check_opencv() -> bool:
    try:
        import cv2
    except ImportError:
        _report(BAD, "OpenCV not importable -- needed to decode video. It ships with mediapipe.")
        return False
    _report(OK, f"OpenCV {cv2.__version__} available for video decoding")
    return True


def check_camera(index: int) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    capture = cv2.VideoCapture(index)
    try:
        if not capture.isOpened():
            _report(
                WARN,
                f"no camera at index {index}. Fine if you are working from recorded "
                "files; a blocker if you planned to capture live.",
            )
            return True  # not fatal
        ok, _frame = capture.read()
        if not ok:
            _report(WARN, f"camera {index} opened but returned no frame")
            return True
        fps = capture.get(cv2.CAP_PROP_FPS)
        _report(OK, f"camera {index} opens and delivers frames (reported fps: {fps or 'unknown'})")
    finally:
        capture.release()
    return True


def check_worldgap() -> bool:
    try:
        import worldgap
        from worldgap.data.loaders import video  # noqa: F401
    except ImportError as e:
        _report(BAD, f"worldgap not importable: {e}")
        return False
    _report(OK, f"worldgap {worldgap.__version__} importable, video loader present")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/holistic_landmarker.task"),
        help="path to the holistic_landmarker.task bundle",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--skip-camera", action="store_true", help="skip the webcam check entirely"
    )
    args = parser.parse_args()

    print("worldgap V1 preflight\n" + "-" * 60)

    results = [check_worldgap(), check_mediapipe(), check_opencv()]

    if results[1]:  # only worth checking the bundle if mediapipe is usable
        model_ok = check_model_bundle(args.model)
        results.append(model_ok)
        if model_ok:
            results.append(check_landmarker_constructs(args.model))

    if not args.skip_camera:
        check_camera(args.camera_index)

    print("-" * 60)
    if all(results):
        print("All required checks passed. You can record and run V1.")
        return 0
    print("Some checks FAILED -- fix those before recording, not after.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
