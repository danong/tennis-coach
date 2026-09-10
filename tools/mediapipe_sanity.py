#!/usr/bin/env python3
"""Load a local MediaPipe Pose Landmarker model and run one ordered VIDEO call."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mediapipe as mp
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_heavy.task"))
    parser.add_argument("--print-sha256", action="store_true")
    args = parser.parse_args()
    model = args.model.expanduser()
    if not model.is_file():
        parser.error(f"model does not exist: {model}")
    if args.print_sha256:
        print(sha256(model))
        return 0

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    # A synthetic RGB frame is intentionally person-free: this verifies model/API
    # loading and monotonic VIDEO timestamps without reading private footage.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), dtype=np.uint8))
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        first = landmarker.detect_for_video(image, 0)
        second = landmarker.detect_for_video(image, 33)
    print(f"MediaPipe Pose Landmarker loaded: {model.name}")
    print(f"ordered VIDEO calls succeeded; detected poses: {len(first.pose_landmarks)}, {len(second.pose_landmarks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
