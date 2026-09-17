# Multimodal SceneTrack

> **Status:** Current · **State:** Maintained · **Work:** Deferred · **As of:** 2026-09-16

## Implemented design

`SceneTrack` is a small in-memory view that aligns existing native-frame body,
ball, racket, and audio observations on the dense MediaPipe world-track PTS
grid. It is data plumbing only: it does not run inference, own caches,
orchestrate processing, or select checkpoints.

```python
@dataclass(frozen=True)
class Point2D:
    x: float                    # normalized upright source coordinate
    y: float
    confidence: float

@dataclass(frozen=True)
class Racket2D:
    bbox: tuple[float, float, float, float]
    bbox_confidence: float
    keypoints: Mapping[str, Point2D | None]

@dataclass(frozen=True)
class SceneFrame:
    time_seconds: float
    body_2d: FrameObservation | None
    body_3d: tuple[WorldLandmark | None, ...] | None
    ball_2d: Point2D | None
    racket_2d: Racket2D | None
    audio_energy: float | None
    audio_transient: bool | None

@dataclass(frozen=True)
class SceneTrack:
    source_fingerprint: str
    available_modalities: frozenset[str]
    frames: tuple[SceneFrame, ...]
```

`time_seconds` is canonical source PTS. There is no FPS-derived join,
nearest-frame matching, interpolation, truncation, smoothing, or scene
serialization. `available_modalities` distinguishes an unavailable model from
a valid frame with no detection: when `ball_2d` is available,
`frame.ball_2d is None` means RacketVision did not detect a ball in that frame.

## Raw observation artifacts

Kinematic and RacketVision observations remain independent reusable caches:

```text
metadata/<video>/attempts/<serve-id>/cache/
  kinematic-track-v1.jsonl
  racketvision-track-v1.jsonl
```

The RacketVision JSONL cache is source/range/timeline/tracker/preprocessing
bound, atomically published, and contains one explicit typed observation per
native world-frame PTS. It retains null ball/racket detections and confidence
verbatim; it does not smooth or fill missing observations.

RacketVision receives the exact dense native PTS schedule. Tagged iPhone HLG
sources are converted to SDR only for model pixels; original source media,
exports, fingerprints, and PTS remain authoritative.

## Construction

`build_scene_track_with_racketvision_observations()` accepts a dense world
cache snapshot plus typed `RacketVisionFrameObservation` values. It requires
identical row counts and exact equal timestamps, then attaches ball/racket
observations without changing body or audio evidence.

This strict construction rejects misaligned input rather than guessing. A
separate production CSV adapter no longer exists; CSV remains limited to
experimental tools under `tools/racketvision/`.

## Processing integration

`run_analyze_serve()` automatically loads a compatible RacketVision cache or
generates it on a cache miss, then constructs a `SceneTrack`. `process` treats
the tracking artifact as part of attempt completion and lazily shares one
RacketVision tracker across attempt cache misses. A completed rerun reuses both
kinematic and RacketVision caches without decoding model frames or opening the
tracker.

The review/source-frame path also uses SDR pixels for tagged HLG sources, but
review images are generated output rather than a reusable cache.

## Current use and deferred decision

`SceneTrack` currently supplies visual diagnostics around selected release and
contact checkpoints. Existing checkpoint scoring remains body/audio based:

- release does not yet use ball-to-toss-hand evidence;
- contact does not yet use ball-to-racket evidence.

Direct evidence weighting is deliberately deferred until automatic tracking is
reliable across normal `process` runs. No generalized track registry, scene
cache, independent cadence alignment, court coordinates, homography, physics
fitting, multi-camera support, or production smoothing is planned by this
implementation.
