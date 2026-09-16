#!/usr/bin/env python3
"""Small image-space rigid-pose and two-arc ball-fitting experiment."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

POSE_EDGES = ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
              (24, 26), (26, 28), (27, 29), (29, 31), (28, 30),
              (30, 32))
RACKET_NAMES = ("Top", "Bottom", "Handle", "Left", "Right")
RACKET_EDGES = ((0, 1), (1, 2), (2, 3), (2, 4))


def xy(row, name):
    return np.array([row[f"Smooth{name}X"], row[f"Smooth{name}Y"]], float)


def constrained_pose(data):
    """Project each frame onto median bone lengths, keeping the hips anchored."""
    points = np.array([
        [[row[f"SmoothPose{i}X"], row[f"SmoothPose{i}Y"]] for i in range(33)]
        for _, row in data.iterrows()
    ], float)
    lengths = {edge: np.nanmedian(np.linalg.norm(points[:, edge[1]] - points[:, edge[0]], axis=1))
               for edge in POSE_EDGES}
    # A few passes are enough for this visualization baseline.
    for frame in points:
        for _ in range(4):
            for a, b in POSE_EDGES:
                if not (np.all(np.isfinite(frame[a])) and np.all(np.isfinite(frame[b]))):
                    continue
                delta = frame[b] - frame[a]
                distance = np.linalg.norm(delta)
                if distance < 1e-6 or not np.isfinite(lengths[(a, b)]):
                    continue
                correction = delta / distance * (distance - lengths[(a, b)])
                # Move both joints, except keep the hip midpoint as a stable anchor.
                if a not in (23, 24):
                    frame[a] += correction * 0.35
                if b not in (23, 24):
                    frame[b] -= correction * 0.65
    for i in range(33):
        data[f"PhysicsPose{i}X"] = points[:, i, 0]
        data[f"PhysicsPose{i}Y"] = points[:, i, 1]
    return data


def robust_parabola(frames, values):
    """Fit a quadratic while repeatedly rejecting large candidate jumps."""
    frames = np.asarray(frames, float)
    values = np.asarray(values, float)
    keep = np.isfinite(values)
    if keep.sum() < 3:
        return None
    for _ in range(4):
        coefficients = np.polyfit(frames[keep], values[keep], 2)
        residual = np.abs(values - np.polyval(coefficients, frames))
        threshold = max(20.0, np.nanpercentile(residual[keep], 75) * 2.5)
        keep = keep & (residual <= threshold)
    return coefficients


def fit_vertex_curve(frames, values, vertex):
    """Fit y = a*(frame-vertex)^2 + c, rejecting candidate jumps."""
    frames = np.asarray(frames, float)
    values = np.asarray(values, float)
    keep = np.isfinite(values)
    if keep.sum() < 3:
        return None
    design = np.column_stack(((frames - vertex) ** 2, np.ones(len(frames))))
    for _ in range(4):
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = np.abs(values - design @ coefficients)
        threshold = max(20.0, np.nanpercentile(residual[keep], 75) * 2.5)
        keep = keep & (residual <= threshold)
    coefficients[0] = max(0.0, coefficients[0])
    return coefficients


def fit_ball(data, toss_start, release, impact, bounce, peak_frame):
    observed = data["Visibility"].fillna(0).to_numpy() > 0
    ball = data[["X", "Y"]].to_numpy(float)
    smooth = data[["SmoothX", "SmoothY"]].to_numpy(float)
    wrist = data[["SmoothPose15X", "SmoothPose15Y"]].to_numpy(float)
    distance = np.linalg.norm(smooth - wrist, axis=1)
    bounce = min(bounce, len(data))
    data["BallPhase"] = "held"
    data.loc[toss_start:release - 1, "BallPhase"] = "toss (coupled)"
    data.loc[release:impact - 1, "BallPhase"] = "pre-impact flight"
    data.loc[impact:bounce - 1, "BallPhase"] = "post-impact flight"
    data.loc[bounce:, "BallPhase"] = "post-bounce"
    data["FitBallX"] = np.nan
    data["FitBallY"] = np.nan
    # During the toss, use the smoothed left wrist as the coupled ball proxy.
    toss = np.arange(toss_start, release)
    data.loc[toss_start:release - 1, "FitBallX"] = data.loc[toss_start:release - 1, "SmoothPose15X"]
    data.loc[toss_start:release - 1, "FitBallY"] = data.loc[toss_start:release - 1, "SmoothPose15Y"]
    for start, end, phase in ((release, impact, "pre-impact"), (impact, bounce, "post-impact")):
        mask = observed & (np.arange(len(data)) >= start) & (np.arange(len(data)) < end)
        if mask.sum() < 3:
            continue
        for axis, name in enumerate(("FitBallX", "FitBallY")):
            frames = data.loc[mask, "Frame"]
            if phase == "pre-impact" and axis == 1:
                coefficients = fit_vertex_curve(frames, ball[mask, axis], peak_frame)
                if coefficients is not None:
                    # Make the pre-impact arc arrive at the racket contact point.
                    coefficients[1] = (smooth[impact, axis] -
                                      coefficients[0] * (impact - peak_frame) ** 2)
                values = (coefficients[0] * (data.loc[start:end - 1, "Frame"] - peak_frame) ** 2
                          + coefficients[1]) if coefficients is not None else None
            else:
                coefficients = robust_parabola(frames, ball[mask, axis])
                if coefficients is not None:
                    # Avoid an artificial jump between the two fitted arcs.
                    coefficients[-1] += (smooth[impact, axis] -
                                         np.polyval(coefficients, impact))
                values = (np.polyval(coefficients, data.loc[start:end - 1, "Frame"])
                          if coefficients is not None else None)
            if values is not None:
                if phase == "pre-impact":
                    # Join the flight arc to the hand at release while keeping
                    # the racket contact point fixed; this avoids a visible
                    # discontinuity from noisy release coordinates.
                    anchor = wrist[release, axis]
                    if np.isfinite(anchor):
                        frames_out = data.loc[start:end - 1, "Frame"].to_numpy(float)
                        blend = (impact - frames_out) / max(1, impact - release)
                        # Fade the join correction to zero at the supplied peak.
                        peak_fade = ((frames_out - peak_frame) / max(1, abs(release - peak_frame))) ** 2
                        blend *= peak_fade
                        first_value = values.iloc[0] if hasattr(values, "iloc") else values[0]
                        values = values + (anchor - first_value) * blend
                data.loc[start:end - 1, name] = values
    return data, release


def find_impact(data):
    ball_x = "FitBallX" if "FitBallX" in data else "SmoothX"
    ball_y = "FitBallY" if "FitBallY" in data else "SmoothY"
    valid = data[ball_x].notna()
    valid &= data[ball_y].notna()
    hoop = data[[f"Smooth{n}{axis}" for n in ("Top", "Bottom", "Left", "Right")
                 for axis in ("X", "Y")]].to_numpy(float).reshape(-1, 4, 2)
    ball = data[[ball_x, ball_y]].to_numpy(float)
    wrist_elevation = -data["SmoothPose16Y"].to_numpy(float)
    racket_elevation = -np.nanmean(hoop[:, :, 1], axis=1)
    max_wrist = np.nanmax(wrist_elevation)
    max_racket = np.nanmax(racket_elevation)
    score = np.full(len(data), np.inf)
    for i in np.where(valid)[0]:
        distance = np.linalg.norm(hoop[i] - ball[i], axis=1).min()
        scale = max(1.0, np.linalg.norm(hoop[i].max(0) - hoop[i].min(0)))
        score[i] = 4 * distance / scale
        score[i] += abs(wrist_elevation[i] - max_wrist) / 300
        score[i] += abs(racket_elevation[i] - max_racket) / 300
    return int(data.iloc[np.argmin(score)]["Frame"])


def render(video, data, impact, output):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # OpenCV does not apply this file's -90 degree display rotation. The CSV
    # coordinates are in the displayed portrait orientation.
    width, height = 1080, 1920
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output}")
    try:
        for frame_index, (_, row) in enumerate(data.iterrows()):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended before CSV at frame {frame_index}")
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            overlay = frame.copy()
            pose = [np.array([row[f"PhysicsPose{i}X"], row[f"PhysicsPose{i}Y"]])
                    for i in range(33)]
            for a, b in POSE_EDGES:
                if np.all(np.isfinite(pose[a])) and np.all(np.isfinite(pose[b])):
                    cv2.line(overlay, tuple(pose[a].round().astype(int)),
                             tuple(pose[b].round().astype(int)), (0, 165, 255), 3)
            for p in pose:
                if np.all(np.isfinite(p)):
                    cv2.circle(overlay, tuple(p.round().astype(int)), 7, (0, 165, 255), -1)

            racket = [xy(row, n) for n in RACKET_NAMES]
            for a, b in RACKET_EDGES:
                if np.all(np.isfinite(racket[a])) and np.all(np.isfinite(racket[b])):
                    cv2.line(overlay, tuple(racket[a].round().astype(int)),
                             tuple(racket[b].round().astype(int)), (255, 0, 255), 3)
            for p in racket:
                if np.all(np.isfinite(p)):
                    cv2.circle(overlay, tuple(p.round().astype(int)), 10, (255, 0, 255), -1)

            if np.isfinite(row["FitBallX"]):
                ball = np.array([row["FitBallX"], row["FitBallY"]])
            elif np.isfinite(row["SmoothX"]):
                ball = xy(row, "")
            else:
                ball = None
            if ball is not None:
                cv2.circle(overlay, tuple(ball.round().astype(int)), 20, (0, 255, 255), 4)
            cv2.putText(overlay, f"physics frame {frame_index} ({row['BallPhase']})",
                        (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            writer.write(cv2.addWeighted(frame, 0.75, overlay, 0.25, 0))
    finally:
        cap.release()
        writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--toss-start", type=int, default=283)
    parser.add_argument("--release-frame", type=int, default=339)
    parser.add_argument("--impact-frame", type=int, default=439)
    parser.add_argument("--bounce-frame", type=int, default=560)
    parser.add_argument("--peak-frame", type=int, default=382,
                        help="Known pre-impact ball peak frame")
    args = parser.parse_args()
    video = args.video or args.csv.with_name(args.csv.name.replace("-smoothed.csv", ".mp4"))
    output_csv = args.output or args.csv.with_name(args.csv.stem + "-physics.csv")
    output_video = output_csv.with_suffix(".mp4")
    data = pd.read_csv(args.csv)
    data = constrained_pose(data)
    impact = args.impact_frame
    data, release = fit_ball(data, args.toss_start, args.release_frame,
                             impact, args.bounce_frame, args.peak_frame)
    data.to_csv(output_csv, index=False)
    render(video, data, impact, output_video)
    print(f"release={release}, impact={impact}")
    print(f"Wrote {output_csv}\nWrote {output_video}")


if __name__ == "__main__":
    main()
