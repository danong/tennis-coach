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
