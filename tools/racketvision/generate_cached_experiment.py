#!/usr/bin/env python3
"""Generate a full-rate MediaPipe CSV/overlay from an existing ball CSV."""
import argparse
from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path); p.add_argument("ball_csv", type=Path)
    p.add_argument("--model", type=Path, default=Path("models/pose_landmarker_heavy.task"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = pd.read_csv(args.ball_csv)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened(): raise RuntimeError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows = []
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1)) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, round(i * 1000 / fps))
            row = data.iloc[i].to_dict()
            h, w = frame.shape[:2]
            points = result.pose_landmarks[0] if result.pose_landmarks else []
            for j in range(33):
                if j < len(points):
                    row[f"Pose{j}X"] = points[j].x * w
                    row[f"Pose{j}Y"] = points[j].y * h
                    row[f"Pose{j}Confidence"] = points[j].visibility
                else:
                    row[f"Pose{j}X"] = row[f"Pose{j}Y"] = row[f"Pose{j}Confidence"] = float("nan")
            rows.append(row); i += 1
    cap.release()
    if i != len(data): raise RuntimeError(f"video has {i} frames, CSV has {len(data)}")
    out = pd.DataFrame(rows); out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({i} full-rate pose frames)")

if __name__ == "__main__": main()
