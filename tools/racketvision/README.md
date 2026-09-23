# RacketVision video prototype

Runs RacketVision BallTrack and RacketPose on one video, producing an annotated
video and a per-frame CSV. This is intentionally a small, CPU-only prototype.

## Setup

Create the project's Python environment, then fetch the pinned, ignored
RacketVision source and model files:

```bash
mise run setup
mise run bootstrap
```

Bootstrap checks out RacketVision commit
`c44af2a08524d3cb54d818f19686f4cdea4d2793` under `vendor/racketvision/` and
downloads the four artifacts in `models/manifest.json`. It verifies each
artifact's SHA-256 before keeping it. Valid local files are reused. If an
existing vendor checkout is at another commit, bootstrap stops and asks you to
move it aside; it does not overwrite local source changes. The vendor source
and downloaded models are ignored by Git.

The current environment configuration pins Torch/OpenMMLab, NumPy 1.26, and
the older Setuptools API required by MMPose. It selects a CUDA 12.1 MMCV wheel
on Linux, but the full locked environment has not yet been verified on WSL2.
`mise run setup` creates the environment; `ffmpeg` must also be available on
`PATH`.

To recreate just these ignored runtime assets later, run `mise run bootstrap`.

## Run

```bash
uv run --locked --all-groups python tools/racketvision/track_video.py \
  input.MOV --output output/racketvision/tracked.mp4 \
  --threshold 0.10
```

To overlay the existing MediaPipe 2D pose cache for `single-serve-01`:

```bash
uv run --locked --all-groups python tools/racketvision/track_video.py \
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
```

The CSV contains ball coordinates/confidence and, when detected, one racket
bounding box plus five racket keypoints (`Top`, `Bottom`, `Handle`, `Left`,
`Right`) and their confidences.

## Baseline smoothing

Post-process an existing run without re-running any models:

```bash
uv run --locked --all-groups python tools/racketvision/smooth_tracks.py \
  output/racketvision/single-serve-01-mediapipe.csv
```

This writes `-smoothed.csv` and `-smoothed.mp4`. Internal gaps are linearly
interpolated, then coordinates are filtered with a Savitzky-Golay filter
(default window 11, polynomial order 2). The smoothed video draws the raw
tracking layer at 25% alpha underneath the fully opaque smoothed layer, with
no tracking text. Adjust with `--window` and `--polyorder`.

Find and render an impact candidate from the smoothed metadata:

```bash
uv run --locked --all-groups python tools/racketvision/find_impact.py \
  output/racketvision/single-serve-01-mediapipe-smoothed.csv
```

The heuristic favors the frame where the ball is closest to, or inside, the
four racket hoop points, while the MediaPipe right wrist (`Pose16`) and racket
hoop center are near their maximum upward elevation. It writes a clean, unannotated `-impact.png` source frame for inspection.

## Physics-fitting experiment

Apply fixed median bone lengths and fit separate quadratic ball arcs after
release and impact, without re-running inference:

```bash
uv run --locked --all-groups python tools/racketvision/fit_physics.py \
  output/racketvision/single-serve-01-mediapipe-smoothed.csv \
  --video refs/anchors/single-serve-01.mov
```

This writes `-physics.csv` and a full-length `-physics.mp4` overlay using the
original, unannotated video as its base. For the current clip, the supplied
phase hints are toss start 283, release 339, peak 382, impact 439, and bounce
560. Use the script options to change these assumptions.

## Four-stage experiment and observations

The experiment is organized around four numbered stages. The numbering reflects
artifact lineage rather than a perfectly contiguous sequence:

### 00 — Raw ball and racket tracking

`00-X-raw.mp4` shows the direct BallTrack/RacketPose observations.

Regenerate the raw overlays:

```bash
uv run --locked --all-groups python tools/racketvision/render_raw.py \
  refs/anchors/single-serve-01.mov output/racketvision/cache/single-serve-01-raw.csv \
  --output output/racketvision/00-single-serve-01-raw.mp4

uv run --locked --all-groups python tools/racketvision/render_raw.py \
  /Users/danielong/Documents/Tennis/Serves/2026-09-15/cropped.MOV \
  output/racketvision/cache/cropped-raw.csv \
  --output output/racketvision/00-cropped-raw.mp4
```

Observations:

- **Ball:** largely useless before toss release; generally accurate from
  release to contact with a couple of missed observations; generally accurate
  after contact but with substantially more missed observations and jitter.
- **Racket:** quite jittery, but it locks onto the racket perfectly every now
  and then.

### 01 — Interpolation and smoothing

`01-X-smoothed.mp4` fills internal gaps and applies a Savitzky-Golay filter.
The raw layer is shown underneath at 25% opacity and the smoothed layer is
opaque.

The smoothed ball and racket have generally the same strengths and weaknesses
as the raw tracks, but are a little better visually. Smoothing can also hide
whether a point came from a real observation or an interpolation.

Regenerate the smoothed outputs:

```bash
uv run --locked --all-groups python tools/racketvision/smooth_tracks.py \
  output/racketvision/cache/single-serve-01-raw.csv \
  --video output/racketvision/00-single-serve-01-raw.mp4 \
  --output output/racketvision/01-single-serve-01-smoothed.csv

uv run --locked --all-groups python tools/racketvision/smooth_tracks.py \
  output/racketvision/cache/cropped-raw.csv \
  --video output/racketvision/00-cropped-raw.mp4 \
  --output output/racketvision/01-cropped-smoothed.csv
```

### 02 — Stage/contact detection

`02-X-00-start.png`, `02-X-01-release.png`,
`02-X-02-peak.png`, and `02-X-03-impact.png` are clean source frames selected
from cached stage metadata and the experiment’s heuristics.

Detection is generally strong. However, `02-cropped-01-release.png` is totally
wrong: the ball is nowhere near the hand. The likely failure is that the
existing stage detector uses body-pose signals such as left-arm elevation and
extension, not direct ball/hand distance, so an arm-motion event can be labeled
as release even when the ball is elsewhere.

Regenerate the stage frames:

```bash
uv run --locked --all-groups python tools/racketvision/find_stages.py \
  output/racketvision/cache/single-serve-01-smoothed.csv \
  refs/anchors/single-serve-01.mov \
  output/racketvision/cache/single-serve-01-stages.json \
  --prefix output/racketvision/02-single-serve-01

uv run --locked --all-groups python tools/racketvision/find_stages.py \
  output/racketvision/cache/cropped-smoothed.csv \
  /Users/danielong/Documents/Tennis/Serves/2026-09-15/cropped.MOV \
  output/racketvision/cache/cropped-stages.json \
  --prefix output/racketvision/02-cropped
```

### 03 — Physics fitting

`03-X-physics.mp4` adds the experimental rigid-body and trajectory models.

Regenerate the physics videos:

```bash
uv run --locked --all-groups python tools/racketvision/fit_physics.py \
  output/racketvision/cache/single-serve-01-smoothed.csv \
  --video refs/anchors/single-serve-01.mov \
  --toss-start 272 --release-frame 327 --impact-frame 439 --peak-frame 380 \
  --output output/racketvision/03-single-serve-01-physics.csv

uv run --locked --all-groups python tools/racketvision/fit_physics.py \
  output/racketvision/cache/cropped-smoothed.csv \
  --video /Users/danielong/Documents/Tennis/Serves/2026-09-15/cropped.MOV \
  --toss-start 107 --release-frame 141 --impact-frame 170 --peak-frame 153 \
  --bounce-frame 314 --output output/racketvision/03-cropped-physics.csv
```

Observations:

- **Physics ball:** the general shape is right but the trajectory is very
  inaccurate; the smoothed ball matches the real ball better.
- **Physics racket:** the points align with the racket location better, but do
  not match the actual hoop/handle landmarks. They are scaled down toward the
  center of mass.
- **Physics body:** generally looks quite good.

These results suggest that rigid pose constraints may be worth exploring for
body landmarks, while the current physics ball and racket models should not
replace the smoothed observations yet.

### Why is there no 02?

Yes—this was effectively a specification/naming omission. Stage `02` was never
assigned a separate artifact-producing operation. The current useful sequence
is therefore `00` raw, `01` smoothed, `02` detected stages, and `03` physics.
A future diagnostic stage can be added after the physics output if needed.

## Experiment cache

Reusable metadata for the current two exemplars is kept in
`output/racketvision/cache/` so the rendered artifacts can be regenerated
without searching through older output files:

```text
cache/single-serve-01-raw.csv
cache/single-serve-01-smoothed.csv
cache/single-serve-01-pose.jsonl
cache/cropped-raw.csv
cache/cropped-smoothed.csv
cache/cropped-pose.jsonl
cache/single-serve-01-stages.json
cache/cropped-stages.json
```

The impact PNG must be rendered from the original source video, not from a
smoothed/annotated MP4. The source MOV rotation is normalized before frame
extraction.

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
