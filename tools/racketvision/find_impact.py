#!/usr/bin/env python3
"""Find and render a likely racket/ball impact frame from smoothed metadata."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


POSE_CONNECTIONS = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 25), (25, 27), (12, 24),
                    (24, 26), (26, 28), (23, 24), (27, 29), (29, 31),
                    (28, 30), (30, 32))
RACKET_NAMES = ("Top", "Bottom", "Handle", "Left", "Right")
HOOP_NAMES = ("Top", "Bottom", "Left", "Right")
RACKET_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (2, 4))


def point(row, prefix, index=None):
    name = f"{prefix}{index}" if index is not None else prefix
    return np.array([row[f"Smooth{name}X"], row[f"Smooth{name}Y"]], dtype=float)


def finite_point(row, prefix, index=None):
    value = point(row, prefix, index)
    return np.all(np.isfinite(value))


def choose_frame(data):
    candidates = []
    wrist_elevation = []
    racket_elevation = []
    for _, row in data.iterrows():
        if not finite_point(row, "") or not finite_point(row, "Pose", 16):
            wrist_elevation.append(np.nan)
            racket_elevation.append(np.nan)
            continue
        hoop = np.array([point(row, name) for name in HOOP_NAMES], dtype=np.float32)
        ball = point(row, "")
        wrist_elevation.append(-point(row, "Pose", 16)[1])
        racket_elevation.append(-float(hoop[:, 1].mean()))
        hull = cv2.convexHull(hoop)
        inside = cv2.pointPolygonTest(hull, tuple(ball), False) >= 0
        distance = np.linalg.norm(hoop - ball, axis=1).min()
        scale = max(1.0, np.linalg.norm(hoop.max(0) - hoop.min(0)))
        candidates.append((int(row["Frame"]), distance / scale, inside))

    wrist_elevation = np.asarray(wrist_elevation)
    racket_elevation = np.asarray(racket_elevation)
    wrist_max = np.nanmax(wrist_elevation)
    racket_max = np.nanmax(racket_elevation)
    wrist_range = max(1.0, np.nanmax(wrist_elevation) - np.nanmin(wrist_elevation))
    racket_range = max(1.0, np.nanmax(racket_elevation) - np.nanmin(racket_elevation))

    scored = []
    for frame, proximity, inside in candidates:
        row_index = data.index[data["Frame"] == frame][0]
        score = (4.0 * proximity +
                 abs(wrist_elevation[row_index] - wrist_max) / wrist_range +
                 abs(racket_elevation[row_index] - racket_max) / racket_range)
        # Being inside the four hoop points is a strong collision signal.
        scored.append((score - (2.0 if inside else 0.0), frame, proximity, inside))
    if not scored:
        raise RuntimeError("No frame has complete smoothed ball, wrist, and racket data")
    return min(scored), wrist_max, racket_max


def render(frame, row, output):
    # Draw into a separate layer so the annotations remain visible but do not
    # obscure the underlying racket and ball. Alpha is intentionally subtle.
    overlay = frame.copy()
    # Body pose.
    pose = [point(row, "Pose", i) for i in range(33)]
    for a, b in POSE_CONNECTIONS:
        if np.all(np.isfinite(pose[a])) and np.all(np.isfinite(pose[b])):
            cv2.line(overlay, tuple(pose[a].round().astype(int)), tuple(pose[b].round().astype(int)),
                     (0, 165, 255), 3)
    for value in pose:
        if np.all(np.isfinite(value)):
            cv2.circle(overlay, tuple(value.round().astype(int)), 7, (0, 165, 255), -1)

    # Racket: four hoop points plus handle.
    racket = [point(row, name) for name in RACKET_NAMES]
    for a, b in RACKET_CONNECTIONS:
        if np.all(np.isfinite(racket[a])) and np.all(np.isfinite(racket[b])):
            cv2.line(overlay, tuple(racket[a].round().astype(int)), tuple(racket[b].round().astype(int)),
                     (255, 0, 255), 3)
    for value in racket:
        if np.all(np.isfinite(value)):
            cv2.circle(overlay, tuple(value.round().astype(int)), 10, (255, 0, 255), -1)

    ball = point(row, "")
    cv2.circle(overlay, tuple(ball.round().astype(int)), 20, (0, 255, 255), 4)
    cv2.putText(overlay, f"impact candidate: frame {int(row['Frame'])}", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)
    cv2.imwrite(str(output), frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    video = args.video or args.csv.with_name(args.csv.name.replace("-smoothed.csv", ".mp4"))
    output = args.output or args.csv.with_name(args.csv.stem + "-impact.png")
    data = pd.read_csv(args.csv)
    selected, wrist_max, racket_max = choose_frame(data)
    _, frame_number, proximity, inside = selected
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    render(frame, data.loc[data["Frame"] == frame_number].iloc[0], output)
    print(f"Selected frame {frame_number} (hoop distance={proximity:.3f}, inside_hoop={inside})")
    print(f"Wrist global elevation max={wrist_max:.1f}; racket global elevation max={racket_max:.1f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
