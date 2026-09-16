#!/usr/bin/env python3
"""Run RacketVision BallTrack on one video and create an annotated MP4."""

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RACKETVISION = HERE / "RacketVision"
BALLTRACK = RACKETVISION / "source" / "BallTrack"
RACKETPOSE = RACKETVISION / "source" / "RacketPose"
DEFAULT_CFG = BALLTRACK / "configs" / "tracknetv3_base.py"
# download_checkpoints.py was run from the RacketVision repository root.
DEFAULT_CKPT = RACKETVISION / "BallTrack" / "checkpoints" / "balltrack_best.pth"
DEFAULT_DET_CFG = RACKETPOSE / "configs" / "detection" / "rtmdet_m_racket_infer.py"
DEFAULT_DET_CKPT = RACKETVISION / "RacketPose" / "checkpoints" / "epoch_300.pth"
DEFAULT_POSE_CFG = RACKETPOSE / "configs" / "pose" / "rtmpose_m_racket_infer.py"
DEFAULT_POSE_CKPT = RACKETVISION / "RacketPose" / "checkpoints" / "best_PCK_epoch_90.pth"


def make_sdr_video(input_path, output_path):
    """Convert BT.2020 HLG video to ordinary BT.709 SDR with ffmpeg."""
    filter_expr = "colorspace=iall=bt2020:all=bt709:fast=0"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path), "-vf", filter_expr,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-an", str(output_path),
    ], check=True)


def letterbox(frame, width, height):
    """Fit a frame into the model canvas without changing its aspect ratio."""
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, round(src_w * scale))
    resized_h = max(1, round(src_h * scale))
    resized = cv2.resize(frame, (resized_w, resized_h))
    left = (width - resized_w) // 2
    top = (height - resized_h) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, scale, left, top


def extract_frames_and_median(video_path, frame_dir, width=512, height=288,
                              samples=64, crop_top=0):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    model_dir = frame_dir / "model"
    model_dir.mkdir()
    sample_indices = set(np.linspace(0, max(0, total - 1), samples, dtype=int)) if total else set()
    native_samples = []
    model_samples = []
    frame_paths = []
    model_frame_paths = []
    frame_index = 0
    output_width = output_height = None
    transform = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Preserve the source orientation. OpenCV gives us the actual portrait
        # pixels; rotating them would make the output 90 degrees sideways.
        if crop_top:
            frame = frame[crop_top:]
        if output_width is None:
            output_height, output_width = frame.shape[:2]
            _, scale, left, top = letterbox(frame, width, height)
            transform = (scale, left, top)
        model_frame, _, _, _ = letterbox(frame, width, height)

        path = frame_dir / f"{frame_index:04d}.jpg"
        model_path = model_dir / f"{frame_index:04d}.jpg"
        if not cv2.imwrite(str(path), frame) or not cv2.imwrite(str(model_path), model_frame):
            raise RuntimeError(f"Could not write frame {frame_index}")
        frame_paths.append(path)
        model_frame_paths.append(model_path)

        if frame_index in sample_indices:
            native_samples.append(frame.copy())
            model_samples.append(model_frame)
        frame_index += 1

    cap.release()
    if not frame_paths:
        raise RuntimeError("Video contains no readable frames")
    if not native_samples:
        native_samples.append(cv2.imread(str(frame_paths[0])))
        model_samples.append(cv2.imread(str(model_frame_paths[0])))

    native_median = np.median(np.stack(native_samples), axis=0).astype(np.uint8)
    model_median = np.median(np.stack(model_samples), axis=0).astype(np.uint8)
    return (frame_paths, model_frame_paths, native_median, model_median,
            fps, output_width, output_height, transform)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--det-cfg", type=Path, default=DEFAULT_DET_CFG)
    parser.add_argument("--det-ckpt", type=Path, default=DEFAULT_DET_CKPT)
    parser.add_argument("--pose-cfg", type=Path, default=DEFAULT_POSE_CFG)
    parser.add_argument("--pose-ckpt", type=Path, default=DEFAULT_POSE_CKPT)
    parser.add_argument("--bbox-threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--crop-top", type=int, default=0,
                        help="Pixels to remove from the top before inference")
    args = parser.parse_args()

    if not args.video.exists():
        parser.error(f"Input does not exist: {args.video}")
    if not args.cfg.exists():
        parser.error(f"Config does not exist: {args.cfg}")
    for path in (args.ckpt, args.det_cfg, args.det_ckpt, args.pose_cfg, args.pose_ckpt):
        if not path.exists():
            parser.error(f"Required file does not exist: {path}")

    sys.path.insert(0, str(BALLTRACK))
    from inference import BallInferencer

    pose_spec = importlib.util.spec_from_file_location(
        "racketvision_pose_inference", RACKETPOSE / "tools" / "inference.py"
    )
    pose_module = importlib.util.module_from_spec(pose_spec)
    pose_spec.loader.exec_module(pose_module)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")

    with tempfile.TemporaryDirectory(prefix="racketvision-") as temp:
        temp_dir = Path(temp)
        sdr_video = temp_dir / "input-sdr.mp4"
        print("Converting HDR/HLG input to BT.709 SDR...")
        make_sdr_video(args.video, sdr_video)
        (frame_paths, model_frame_paths, native_median, model_median,
         fps, width, height, transform) = extract_frames_and_median(
            sdr_video, temp_dir, samples=max(1, args.samples),
            crop_top=max(0, args.crop_top)
        )
        median_path = temp_dir / "median.npz"
        np.savez(median_path, median=model_median)
        cv2.imwrite(str(temp_dir / "median.png"), model_median)

        # Keep inspectable artifacts after the temporary frames have been
        # cleaned up. The default median is native-orientation; the model
        # version shows exactly what is passed to TrackNet.
        persistent_median_path = args.output.with_name(args.output.stem + "-median.npz")
        persistent_median_png = args.output.with_name(args.output.stem + "-median.png")
        persistent_model_median_png = args.output.with_name(args.output.stem + "-model-median.png")
        np.savez(persistent_median_path, median=native_median)
        cv2.imwrite(str(persistent_median_png), native_median)
        cv2.imwrite(str(persistent_model_median_png), model_median)

        inferencer = BallInferencer(
            str(args.cfg), str(args.ckpt), device=args.device,
            thre=args.threshold, batchsize=args.batchsize,
        )
        results = inferencer([str(path) for path in model_frame_paths], str(median_path))
        scale, left, top = transform

        print("Loading racket detector and pose models...")
        with pose_module.DefaultScope.overwrite_default_scope("mmdet"):
            det_model = pose_module.init_detector(
                str(args.det_cfg), str(args.det_ckpt), device=args.device
            )
        pose_model = pose_module.init_pose_model(
            str(args.pose_cfg), str(args.pose_ckpt), device=args.device
        )
        racket_results = []
        for path in frame_paths:
            frame = cv2.imread(str(path))
            racket_results.append(pose_module.detect_and_estimate_pose(
                det_model, pose_model, frame, cat_id=2,
                bbox_thr=args.bbox_threshold
            ))

        for result in results:
            if result["Visibility"]:
                result["X"] = round((result["X"] - left) / scale)
                result["Y"] = round((result["Y"] - top) / scale)
                if not (0 <= result["X"] < width and 0 <= result["Y"] < height):
                    result["X"] = result["Y"] = 0
                    result["Visibility"] = 0
        rows = []
        keypoint_names = ["Top", "Bottom", "Handle", "Left", "Right"]
        for ball, rackets in zip(results, racket_results):
            rackets = rackets[:1]  # one racket is enough for this prototype
            if not rackets:
                rows.append(ball.copy())
                continue
            for racket_index, racket in enumerate(rackets):
                row = ball.copy()
                row["RacketIndex"] = racket_index
                bbox = racket["bbox"][0]
                for i, value in enumerate(bbox):
                    row[f"BBox{i + 1}"] = value
                row["BBoxConfidence"] = racket["bbox_score"]
                for name, point, score in zip(
                    keypoint_names, racket["keypoints"], racket["keypoint_scores"]
                ):
                    row[f"{name}X"], row[f"{name}Y"] = point
                    row[f"{name}Confidence"] = score
                rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video: {args.output}")
        try:
            for result, frame_path, rackets in zip(results, frame_paths, racket_results):
                frame = cv2.imread(str(frame_path))
                if result["Visibility"]:
                    center = (int(result["X"]), int(result["Y"]))
                    cv2.circle(frame, center, 14, (0, 255, 255), 3)
                    cv2.putText(
                        frame, f"ball {result['Confidence']:.2f}",
                        (center[0] + 16, center[1] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    )
                for racket in rackets[:1]:
                    x1, y1, x2, y2 = map(int, racket["bbox"][0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    points = [tuple(map(int, point)) for point in racket["keypoints"]]
                    for point in points:
                        cv2.circle(frame, point, 7, (255, 0, 255), -1)
                    for a, b in ((0, 1), (1, 2), (2, 3), (2, 4)):
                        cv2.line(frame, points[a], points[b], (255, 0, 255), 2)
                writer.write(frame)
        finally:
            writer.release()

    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output}")
    print(f"Wrote {persistent_median_png}")
    print(f"Wrote {persistent_median_path}")
    print(f"Wrote {persistent_model_median_png}")


if __name__ == "__main__":
    main()
