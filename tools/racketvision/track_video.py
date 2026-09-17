"""Legacy video/CSV renderer using the shared in-process RacketVision provider."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import pandas as pd

from serve_review.media.frames import SampledFrame
from serve_review.tracking.racketvision import (
    RACKET_KEYPOINT_NAMES,
    RacketVisionConfig,
    RacketVisionTracker,
)

POSE_CONNECTIONS = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
    (23, 24),
)
RACKET_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (2, 4))
POSE_LANDMARK_COUNT = 33


def load_mediapipe_cache(path: Path, frame_count: int):
    """Load normalized 2D landmarks from the existing JSONL cache."""
    frames = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") != "dense-world-frame":
                continue
            persons = record["observation"]["frame_2d"]["persons"]
            frames.append(persons[0]["keypoints"] if persons else None)
    if len(frames) != frame_count:
        raise RuntimeError(
            f"MediaPipe cache has {len(frames)} frames, expected {frame_count}"
        )
    return frames


def make_sdr_video(input_path: Path, output_path: Path) -> None:
    """Convert the known BT.2020 HLG prototype input to BT.709 model pixels."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            "colorspace=iall=bt2020:all=bt709:fast=0",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ],
        check=True,
    )


def decode_frames(video_path: Path, *, crop_top: int = 0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if crop_top:
            frame = frame[crop_top:]
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("Video contains no readable frames")
    return frames, float(fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mediapipe-cache", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path("vendor/racketvision"))
    parser.add_argument("--model-root", type=Path, default=Path("models/racketvision"))
    parser.add_argument("--bbox-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batchsize", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--crop-top", type=int, default=0)
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Input does not exist: {args.video}")
    if args.mediapipe_cache and not args.mediapipe_cache.is_file():
        parser.error(f"MediaPipe cache does not exist: {args.mediapipe_cache}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="racketvision-video-") as temporary:
        sdr_video = Path(temporary) / "input-sdr.mp4"
        print("Converting HDR/HLG input to BT.709 SDR...")
        make_sdr_video(args.video, sdr_video)
        frames, fps = decode_frames(sdr_video, crop_top=max(0, args.crop_top))

    height, width = frames[0].shape[:2]
    pose_rows = (
        load_mediapipe_cache(args.mediapipe_cache, len(frames))
        if args.mediapipe_cache
        else [None] * len(frames)
    )
    sampled = (
        SampledFrame(
            time_seconds=index / fps,
            timestamp_ms=round(index * 1000 / fps),
            width=width,
            height=height,
            image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        for index, frame in enumerate(frames)
    )
    config = RacketVisionConfig(
        source_root=args.source_root,
        ball_checkpoint=args.model_root / "balltrack.pth",
        racket_detector_checkpoint=args.model_root / "racket-detector.pth",
        racket_keypoints_checkpoint=args.model_root / "racket-keypoints.pth",
        ball_threshold=args.threshold,
        racket_bbox_threshold=args.bbox_threshold,
        ball_batch_size=args.batchsize,
        device=args.device,
    )
    observations = RacketVisionTracker(config).track_frames(sampled)

    rows = []
    for index, (observation, pose) in enumerate(zip(observations, pose_rows)):
        row = {"Frame": index, "X": 0, "Y": 0, "Visibility": 0, "Confidence": 0.0}
        if observation.ball_2d is not None:
            row.update(
                X=round(observation.ball_2d.x * width),
                Y=round(observation.ball_2d.y * height),
                Visibility=1,
                Confidence=observation.ball_2d.confidence,
            )
        if pose:
            for pose_index, point in enumerate(pose[:POSE_LANDMARK_COUNT]):
                row[f"Pose{pose_index}X"] = point["x"] * width
                row[f"Pose{pose_index}Y"] = point["y"] * height
                row[f"Pose{pose_index}Confidence"] = point.get("visibility", 0.0)
        racket = observation.racket_2d
        if racket is not None:
            row["RacketIndex"] = 0
            row["BBox1"], row["BBox2"], row["BBox3"], row["BBox4"] = (
                racket.bbox[0] * width,
                racket.bbox[1] * height,
                racket.bbox[2] * width,
                racket.bbox[3] * height,
            )
            row["BBoxConfidence"] = racket.bbox_confidence
            for name in RACKET_KEYPOINT_NAMES:
                point = racket.keypoints.get(name)
                if point is not None:
                    row[f"{name}X"] = point.x * width
                    row[f"{name}Y"] = point.y * height
                    row[f"{name}Confidence"] = point.confidence
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".csv"), index=False)

    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {args.output}")
    try:
        for frame, observation, pose in zip(frames, observations, pose_rows):
            if observation.ball_2d is not None:
                center = (
                    round(observation.ball_2d.x * width),
                    round(observation.ball_2d.y * height),
                )
                cv2.circle(frame, center, 14, (0, 255, 255), 3)
            if pose:
                points = [
                    (round(point["x"] * width), round(point["y"] * height))
                    for point in pose[:POSE_LANDMARK_COUNT]
                ]
                for a, b in POSE_CONNECTIONS:
                    if (
                        pose[a].get("visibility", 0.0) >= 0.5
                        and pose[b].get("visibility", 0.0) >= 0.5
                    ):
                        cv2.line(frame, points[a], points[b], (0, 165, 255), 2)
            racket = observation.racket_2d
            if racket is not None:
                points = []
                for name in RACKET_KEYPOINT_NAMES:
                    point = racket.keypoints.get(name)
                    points.append(
                        None
                        if point is None
                        else (round(point.x * width), round(point.y * height))
                    )
                for point in points:
                    if point is not None:
                        cv2.circle(frame, point, 7, (255, 0, 255), -1)
                for a, b in RACKET_CONNECTIONS:
                    if points[a] is not None and points[b] is not None:
                        cv2.line(frame, points[a], points[b], (255, 0, 255), 2)
            writer.write(frame)
    finally:
        writer.release()

    print(f"Wrote {args.output.with_suffix('.csv')}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
