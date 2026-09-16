# RacketVision video prototype

Runs RacketVision BallTrack and RacketPose on one video, producing an annotated
video and a per-frame CSV. This is intentionally a small, CPU-only prototype.

## Setup

The prototype uses a separate uv environment:

```bash
uv venv --python 3.11 tools/racketvision/.venv
VENV=tools/racketvision/.venv/bin/python
uv pip install --python "$VENV" torch==2.1.2 torchvision==0.16.2
uv pip install --python "$VENV" mmengine==0.10.7
uv pip install --python "$VENV" mmcv==2.1.0 \
  --find-links https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
uv pip install --python "$VENV" mmdet==3.3.0 mmpose==1.3.2
uv pip install --python "$VENV" \
  pandas tqdm parse huggingface_hub terminaltables pycocotools \
  shapely scipy xtcocotools munkres json_tricks albumentations chumpy
```

Download the RacketVision checkpoints (from the cloned repository root):

```bash
cd tools/racketvision/RacketVision
../.venv/bin/python source/download_checkpoints.py --module BallTrack
../.venv/bin/python source/download_checkpoints.py --module RacketPose
cd ../../..
```

`ffmpeg` must also be available on `PATH`.

## Run

```bash
tools/racketvision/.venv/bin/python tools/racketvision/track_video.py \
  input.MOV --output output/racketvision/tracked.mp4 \
  --threshold 0.10
```

To overlay the existing MediaPipe 2D pose cache for `single-serve-01`:

```bash
tools/racketvision/.venv/bin/python tools/racketvision/track_video.py \
  refs/anchors/single-serve-01.mov \
  --output output/racketvision/single-serve-01-mediapipe.mp4 \
  --threshold 0.10 \
  --mediapipe-cache output/m4-frozen-confirmation/single-serve-01/cache/kinematic-track-v1.jsonl
```

The cache is reused directly; MediaPipe is not run by the prototype. It must
match the input video’s decoded frame count and order.

Optional `--crop-top N` removes `N` native pixels from the top before model
inference. Outputs include:

```text
tracked.mp4
tracked.csv
tracked-median.png / tracked-median.npz
tracked-model-median.png
```

The CSV contains ball coordinates/confidence and, when detected, one racket
bounding box plus five racket keypoints (`Top`, `Bottom`, `Handle`, `Left`,
`Right`) and their confidences.

## Baseline smoothing

Post-process an existing run without re-running any models:

```bash
tools/racketvision/.venv/bin/python tools/racketvision/smooth_tracks.py \
  output/racketvision/single-serve-01-mediapipe.csv
```

This writes `-smoothed.csv` and `-smoothed.mp4`. Internal gaps are linearly
interpolated, then coordinates are filtered with a Savitzky-Golay filter
(default window 11, polynomial order 2). Adjust with `--window` and
`--polyorder`.

Find and render an impact candidate from the smoothed metadata:

```bash
tools/racketvision/.venv/bin/python tools/racketvision/find_impact.py \
  output/racketvision/single-serve-01-mediapipe-smoothed.csv
```

The heuristic favors the frame where the ball is closest to, or inside, the
four racket hoop points, while the MediaPipe right wrist (`Pose16`) and racket
hoop center are near their maximum upward elevation. It writes a single
`-impact.png` frame for inspection.

## Physics-fitting experiment

Apply fixed median bone lengths and fit separate quadratic ball arcs after
release and impact, without re-running inference:

```bash
tools/racketvision/.venv/bin/python tools/racketvision/fit_physics.py \
  output/racketvision/single-serve-01-mediapipe-smoothed.csv \
  --video refs/anchors/single-serve-01.mov
```

This writes `-physics.csv` and a full-length `-physics.mp4` overlay using the
original, unannotated video as its base. For the current clip, the supplied
phase hints are toss start 283, release 339, peak 382, impact 439, and bounce
560. Use the script options to change these assumptions.

## What we are testing

This directory is an intentionally disposable research prototype. The goal is
to compare simple approaches visually before deciding what, if anything, should
be incorporated into the main project.

### Stage 1: model detections

`track_video.py` runs TrackNet/BallTrack for the ball and RacketPose for the
racket. It writes the original per-frame detections. BallTrack is single-ball
and can jump between tennis balls or lose the ball entirely.

### Stage 2: cached body pose

The MediaPipe overlay reads existing `kinematic-track-v1.jsonl` coordinates
when `--mediapipe-cache` is provided. It does not run MediaPipe. Cache frames
must match the video frame order and count.

### Stage 3: baseline smoothing

`smooth_tracks.py` fills internal gaps with linear interpolation, then applies
a Savitzky-Golay filter (default window 11, polynomial order 2). It processes
ball coordinates, pose landmarks, racket keypoints, and racket boxes. This is a
visual baseline only: interpolation can connect unrelated BallTrack candidates.
The original columns remain in the CSV beside `Smooth*` columns.

### Stage 4: contact heuristic

`find_impact.py` searches for a frame where the ball is near the four racket
hoop points (preferably inside them), while the right wrist and racket are near
their maximum upward image elevation. This is a heuristic, not a classifier.
It currently identifies frame 439 for the sample clip.

### Stage 5: physics-fitting experiment

`fit_physics.py` is a second post-processing experiment. It:

- projects pose landmarks toward fixed median bone lengths;
- treats frames 283–338 as a ball/left-wrist coupled toss;
- fits a robust quadratic to the pre-impact flight from release 339 to impact
  439, with a supplied peak at 382;
- fits a separate robust quadratic from impact 439 to bounce 560; and
- leaves the post-bounce phase unmodeled.

Only original `Visibility > 0` ball observations are used as fitting samples in
the two flight phases. Large residuals are rejected iteratively. The generated
`FitBall*` columns are the modeled trajectory; `Smooth*` columns remain
available for comparison. The fitted arcs are anchored at the release hand
and racket-contact position to avoid visible discontinuities.

## Prototype notes

- The input test video was 10-bit BT.2020 HLG HDR. We first convert it to
  BT.709 SDR with ffmpeg; this fixed the washed-out appearance but did not
  materially improve ball detection.
- The source is vertical. Frames remain vertical for output, while model frames
  are letterboxed to TrackNet's required `512x288` input rather than stretched.
- A top crop experiment increased detections somewhat but did not eliminate
  dropouts, especially near the crop boundary.
- BallTrack's thresholded heatmap decoder is single-ball and has no temporal
  smoothing. At a low threshold it found a useful continuous sequence, but
  still has occasional gaps/jumps.
- RacketPose uses the tennis detector class and runs on CPU. No audio is
  preserved in the generated MP4.
