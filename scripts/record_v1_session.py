#!/usr/bin/env python3
"""Guided webcam capture for docs/v1_real_data_runbook.md (Step 1).

Shows a mirrored preview with on-screen gesture prompts and saves the RAW
camera frames (no overlay, not mirrored) to
`<recordings>/<condition>/take<N>.mp4`, one folder per capture condition, in
the layout `scripts/run_v1_real_data.py` expects.

    python scripts/record_v1_session.py --test            10 s take + MediaPipe check
    python scripts/record_v1_session.py --session         the whole plan, resumable
    python scripts/record_v1_session.py --session --only dim_light

Keys in the window: SPACE starts a take, S skips it, Q quits. A take aborted
with Q is deleted, never kept half-recorded. Takes that already exist are
skipped, so a session can be stopped and resumed.

Why this is part of the pipeline, not a convenience: every file is written
with the frame rate the camera ACTUALLY delivered, measured while recording.
`extract_rollout_from_video` converts frames to seconds with the file's own
rate, so `landmark_quality_ground_truth`'s `longest_dropout_run_s` is only as
right as that number. Webcams drop frames in low light while still reporting
their nominal rate; a file stamped "30 fps" holding 20 fps of footage would
stretch every duration by half. Run 1's recordings were made with this
script (camera delivered 29.99-30.01 fps; no take needed correcting).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2

WINDOW = "worldgap V1 capture"
TAKES = 3
COUNTDOWN_S = 5
# Rewrite the file with the measured rate when it differs from the camera's
# nominal rate by more than this fraction.
FPS_TOLERANCE = 0.005

GESTURES = [
    ("FIST", "closed, thumb across fingers"),
    ("PALM", "fingers spread wide"),
    ("STOP", "fingers together, upright"),
    ("LIKE", "thumbs up"),
]

# (folder, setup instruction, seconds per gesture, downscale width or None).
# The routine is identical everywhere except `fast_motion` (same gestures,
# twice as fast); `low_resolution` is downscaled as it is saved.
CONDITIONS = [
    ("clean", "Normal light, plain background, arm's length. Camera fixed.", 2.0, None),
    ("dim_light", "Lights off, screen glow only.", 2.0, None),
    ("backlit", "Window or lamp BEHIND you.", 2.0, None),
    ("side_light", "One lamp at 90 deg to the side: hard shadows on the hand.", 2.0, None),
    ("hand_partial_occlusion", "A book or sheet partly covering the gesturing hand (not your other hand).", 2.0, None),
    ("hand_edge_of_frame", "Gesture at the frame border, drifting in and out.", 2.0, None),
    ("far_from_camera", "Sit/stand 2-3 m from the camera.", 2.0, None),
    ("fast_motion", "Same routine at double speed (prompts come faster).", 1.0, None),
    ("cluttered_background", "Busy background behind you instead of a wall.", 2.0, None),
    ("low_resolution", "Same setup as clean. Saved downscaled to 320 px wide.", 2.0, 320),
    ("camera_shake", "Hold or gently tilt the laptop/camera by hand throughout.", 2.0, None),
]


def session_plan(only: str | None = None) -> list[tuple]:
    """`clean` first and last (runbook: spot drift across the session)."""
    by_name = {c[0]: c for c in CONDITIONS}
    if only:
        if only not in by_name:
            raise SystemExit(f"unknown condition {only!r}; choose from {sorted(by_name)}")
        return [(by_name[only], t) for t in range(TAKES)]
    plan = [(by_name["clean"], 0), (by_name["clean"], 1)]
    for c in CONDITIONS[1:]:
        plan += [(c, t) for t in range(TAKES)]
    plan.append((by_name["clean"], 2))
    return plan


def _put(img, text, y, scale=1.0, color=(255, 255, 255), thick=2, x=20):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def show(frame, lines_top, big=None, sub=None, progress=None) -> int:
    """Draws on a mirrored COPY for the person; the saved frame is untouched."""
    disp = cv2.flip(frame, 1)
    h, w = disp.shape[:2]
    for i, (text, scale) in enumerate(lines_top):
        _put(disp, text, 40 + i * 38, scale)
    if big:
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 3.2, 8)[0]
        _put(disp, big, h // 2 + size[1] // 2, 3.2, (80, 255, 120), 8, x=(w - size[0]) // 2)
    if sub:
        size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        _put(disp, sub, h // 2 + 70, 1.0, x=(w - size[0]) // 2)
    if progress is not None:
        cv2.rectangle(disp, (20, h - 30), (w - 20, h - 14), (60, 60, 60), -1)
        cv2.rectangle(disp, (20, h - 30), (20 + int((w - 40) * progress), h - 14), (80, 255, 120), -1)
    cv2.imshow(WINDOW, disp)
    return cv2.waitKey(1) & 0xFF


def _read(cap):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("camera stopped delivering frames")
    return frame


def record_take(
    cap,
    out_path: Path,
    seconds: float,
    gesture_s: float,
    downscale: int | None,
    header: list,
    clock: Callable[[], float] = time.perf_counter,
) -> dict | None:
    """Records one take. Returns stats, or None if aborted (partial file removed).

    `clock` is injectable so the frame-rate logic can be tested deterministically.
    """
    start = clock()
    while clock() - start < COUNTDOWN_S:  # countdown; nothing is written
        left = COUNTDOWN_S - int(clock() - start)
        if show(_read(cap), header, big=str(left), sub="get ready: first gesture is FIST") == ord("q"):
            return None

    frame = _read(cap)
    h, w = frame.shape[:2]
    size = (downscale, round(h * downscale / w)) if downscale else (w, h)
    nominal = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), nominal, size)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an mp4 writer")

    stamps: list[float] = []
    t0 = clock()
    aborted = False
    while True:
        t = clock() - t0
        if t >= seconds:
            break
        out = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if downscale else frame
        writer.write(out)
        stamps.append(t)
        name, desc = GESTURES[int(t // gesture_s) % len(GESTURES)]
        key = show(frame, [*header, (f"REC {t:4.0f}/{seconds:.0f}s", 0.9)], big=name, sub=desc, progress=t / seconds)
        if key == ord("q"):
            aborted = True
            break
        frame = _read(cap)
    writer.release()
    if aborted or len(stamps) < 10:
        tmp.unlink(missing_ok=True)
        return None

    actual = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    if abs(actual - nominal) / nominal > FPS_TOLERANCE:
        src = cv2.VideoCapture(str(tmp))
        fixed = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), actual, size)
        while True:
            ok, f = src.read()
            if not ok:
                break
            fixed.write(f)
        src.release()
        fixed.release()
        tmp.unlink()
        fps_written = actual
    else:
        tmp.replace(out_path)
        fps_written = nominal
    return {
        "file": str(out_path),
        "frames": len(stamps),
        "seconds": round(stamps[-1], 2),
        "camera_fps_nominal": nominal,
        "camera_fps_actual": round(actual, 3),
        "fps_written": round(fps_written, 3),
        "size": list(size),
    }


def _wait_for_start(cap, header: list) -> str:
    while True:
        key = show(_read(cap), [*header, ("SPACE = start   S = skip   Q = quit", 0.8)])
        if key in (ord(" "), ord("s"), ord("q")):
            return chr(key)


def _open_camera(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    return cap


def mediapipe_check(path: Path, model: Path) -> dict:
    """Runs the real landmarker over a test take: is the setup good enough?"""
    from mediapipe.tasks.python import BaseOptions, vision

    from worldgap.data.loaders.video import (
        extract_rollout_from_video,
        landmark_quality_ground_truth,
    )

    opts = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
    )
    with vision.HolisticLandmarker.create_from_options(opts) as lm:
        rollout = extract_rollout_from_video(path, lm, condition={"capture_condition": "test"})
    gt = landmark_quality_ground_truth(rollout)
    return {
        "frames": int(rollout.states.shape[0]),
        "frame_rate_hz": round(rollout.frame_rate_hz, 2),
        **{k: round(float(v), 3) for k, v in gt.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help="10 s test take, then a MediaPipe check")
    mode.add_argument("--session", action="store_true", help="record the session plan")
    ap.add_argument("--only", help="record just this condition's takes")
    ap.add_argument("--recordings", type=Path, default=Path("recordings"))
    ap.add_argument("--model", type=Path, default=Path("models/holistic_landmarker.task"))
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument(
        "--test-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "worldgap_test_take",
        help="where the test take goes; never inside --recordings, where it would become a condition",
    )
    args = ap.parse_args()

    cap = _open_camera(args.camera)
    try:
        if args.test:
            args.test_dir.mkdir(parents=True, exist_ok=True)
            out = args.test_dir / "test_take.mp4"
            header = [("TEST take (10 s) - not saved to recordings/", 0.8)]
            if _wait_for_start(cap, header) != " ":
                print(json.dumps({"event": "cancelled"}))
                return 0
            stats = record_take(cap, out, 10.0, 2.0, None, header)
            cap.release()
            cv2.destroyAllWindows()
            if stats is None:
                print(json.dumps({"event": "aborted"}))
                return 1
            print(json.dumps({"event": "test_take", **stats}))
            print(json.dumps({"event": "mediapipe", **mediapipe_check(out, args.model)}))
            return 0

        plan = session_plan(args.only)
        for i, ((name, how, gesture_s, downscale), take) in enumerate(plan, 1):
            out = args.recordings / name / f"take{take}.mp4"
            if out.exists():
                print(json.dumps({"event": "exists", "file": str(out)}), flush=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            header = [(f"[{i}/{len(plan)}]  {name}  take {take + 1}/{TAKES}", 1.0), (how, 0.7)]
            key = _wait_for_start(cap, header)
            if key == "q":
                print(json.dumps({"event": "quit"}), flush=True)
                break
            if key == "s":
                print(json.dumps({"event": "skipped", "condition": name, "take": take}), flush=True)
                continue
            stats = record_take(cap, out, args.seconds, gesture_s, downscale, header)
            event = "saved" if stats else "aborted"
            print(json.dumps({"event": event, "condition": name, "take": take, **(stats or {})}), flush=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
