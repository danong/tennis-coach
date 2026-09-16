#!/usr/bin/env python3
"""Interpolate and lightly smooth RacketVision metadata, then render it."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                    (24, 26), (26, 28), (27, 29), (29, 31), (28, 30),
                    (30, 32), (23, 24))
RACKET_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (2, 4))


def smooth(values, valid, window, polyorder):
    """Interpolate internal gaps and apply Savitzky-Golay to the result."""
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    result = values.copy()
    indices = np.arange(len(values))
    known = indices[valid]
    if not len(known):
        return result
    # Do not invent positions before the first or after the last observation.
    result[indices >= known[0]] = np.interp(
        indices[indices >= known[0]], known, values[valid]
    )
    usable = indices <= known[-1]
    result[usable] = np.interp(indices[usable], known, values[valid])
    n = int(valid.sum())
    size = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if size > polyorder and size >= 3 and n >= 2:
        result[known[0]:known[-1] + 1] = savgol_filter(
            result[known[0]:known[-1] + 1], size, polyorder, mode="interp"
        )
    return result


def process(csv_path, output_csv, window, polyorder):
    data = pd.read_csv(csv_path)
    smoothed = data.copy()

    ball_valid = data["Visibility"].fillna(0).to_numpy() > 0
    for axis in ("X", "Y"):
        smoothed[f"Smooth{axis}"] = smooth(
            data[axis], ball_valid, window, polyorder
        )

    for prefix, count, confidence in (("Pose", 33, "Pose{index}Confidence"),
                                       ("", 5, "{name}Confidence")):
        names = [str(i) for i in range(count)] if prefix else [
            "Top", "Bottom", "Handle", "Left", "Right"
        ]
        for index, name in enumerate(names):
            valid = data[confidence.format(index=index, name=name)].fillna(0).to_numpy() > 0
            for axis in ("X", "Y"):
                source = f"{prefix}{name}{axis}"
                if source not in data:
                    continue
                smoothed[f"Smooth{source}"] = smooth(
                    data[source], valid, window, polyorder
                )

    for index in range(1, 5):
        valid = data["BBoxConfidence"].fillna(0).to_numpy() > 0
        smoothed[f"SmoothBBox{index}"] = smooth(
            data[f"BBox{index}"], valid, window, polyorder
        )
    smoothed.to_csv(output_csv, index=False)
    return smoothed


def render(video_path, data, output_video):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output_video}")
    try:
        for frame_index, row in data.iterrows():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended before CSV at frame {frame_index}")
            if np.isfinite(row["SmoothX"]) and np.isfinite(row["SmoothY"]):
                center = (round(row["SmoothX"]), round(row["SmoothY"]))
                cv2.circle(frame, center, 14, (0, 255, 255), 3)
                cv2.putText(frame, "ball smooth", (center[0] + 16, center[1] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            pose = [(row.get(f"SmoothPose{i}X"), row.get(f"SmoothPose{i}Y"))
                    for i in range(33)]
            pose_ok = [np.isfinite(x) and np.isfinite(y) for x, y in pose]
            for a, b in POSE_CONNECTIONS:
                if pose_ok[a] and pose_ok[b]:
                    cv2.line(frame, tuple(map(round, pose[a])), tuple(map(round, pose[b])),
                             (0, 165, 255), 2)
            for point, valid in zip(pose, pose_ok):
                if valid:
                    cv2.circle(frame, tuple(map(round, point)), 5, (0, 165, 255), -1)
            racket = [(row.get(f"Smooth{name}X"), row.get(f"Smooth{name}Y"))
                      for name in ("Top", "Bottom", "Handle", "Left", "Right")]
            racket_ok = [np.isfinite(x) and np.isfinite(y) for x, y in racket]
            bbox = [row.get(f"SmoothBBox{i}") for i in range(1, 5)]
            if all(np.isfinite(v) for v in bbox):
                cv2.rectangle(frame, (round(bbox[0]), round(bbox[1])),
                              (round(bbox[2]), round(bbox[3])), (255, 0, 255), 2)
            for a, b in RACKET_CONNECTIONS:
                if racket_ok[a] and racket_ok[b]:
                    cv2.line(frame, tuple(map(round, racket[a])), tuple(map(round, racket[b])),
                             (255, 0, 255), 2)
            for point, valid in zip(racket, racket_ok):
                if valid:
                    cv2.circle(frame, tuple(map(round, point)), 7, (255, 0, 255), -1)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=int, default=11,
                        help="Odd Savitzky-Golay window length")
    parser.add_argument("--polyorder", type=int, default=2)
    args = parser.parse_args()
    video = args.video or args.csv.with_suffix(".mp4")
    output_csv = args.output or args.csv.with_name(args.csv.stem + "-smoothed.csv")
    output_video = output_csv.with_suffix(".mp4")
    if args.window < 3 or args.window % 2 == 0:
        parser.error("--window must be an odd number >= 3")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    data = process(args.csv, output_csv, args.window, args.polyorder)
    render(video, data, output_video)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_video}")


if __name__ == "__main__":
    main()
