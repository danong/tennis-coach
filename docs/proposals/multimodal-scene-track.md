# MVP multimodal scene track

> **Status:** Proposed · **State:** Draft · **Work:** Next · **As of:** 2026-09-15

## Decision

Introduce a small, in-memory `SceneTrack` that joins existing native-frame body, ball, racket, and audio observations on the MediaPipe world-cache timeline. It will be an adapter over current artifacts, not a new cache format or a replacement for their schemas.

The first consumer will be an optional `analyze-serve --racketvision-csv PATH` diagnostic path. Existing checkpoint selection will remain unchanged. The experiment will report whether direct ball/hand and ball/racket evidence agrees with the currently selected release and contact checkpoints.

## Motivation

Serve cutting, stage-checkpoint analysis, and the RacketVision prototype currently use different per-frame representations. This has already led to duplicated pose columns and frame-order assumptions in the prototype. A minimal joined representation gives analysis code one canonical PTS-bearing view while preserving the independently reusable caches and CSVs.

The MVP is intentionally limited to the current experiment. It does not attempt to solve independent sampling schedules, generalized sensor fusion, court geometry, or stereo vision.

## Model

The implementation should reuse `FrameObservation`, `WorldLandmark`, and existing audio types rather than redefine them. New types are approximately:

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
    time_seconds: float         # canonical source PTS
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

`available_modalities` distinguishes an unavailable modality from a model that ran but produced no detection. For example, when `ball_2d` is available, `frame.ball_2d is None` means no ball was detected on that frame. Raw and smoothed RacketVision CSVs are separate inputs; the MVP will not mix them or add per-value provenance.

All exposed 2D coordinates must be normalized to the upright, unmirrored source display frame. Body 3D retains the existing hip-centered MediaPipe world convention. The scene representation makes no claim that 2D and 3D coordinates share a spatial frame.

`time_seconds` is the scene API's only time value. MediaPipe's integer-millisecond timestamp is derived and validated at the MediaPipe adapter boundary; it is not duplicated on `SceneFrame`. Existing nested `FrameObservation` values retain their compatibility field, but scene logic must not read it as canonical time.

## Construction and validation

Add one builder that accepts a loaded dense world cache, optional already-aligned audio, and an optional RacketVision CSV. The dense world observations define the frame count, order, and PTS. RacketVision row `i` is attached to world observation `i`; its pixel coordinates are normalized using probed source dimensions. This gives the prototype canonical source PTS without deriving time from frame index or nominal FPS.

The builder must reject:

- non-increasing or inconsistent body PTS;
- RacketVision row counts that differ from the world-cache frame count;
- audio counts or timestamps that differ from the body timeline;
- non-finite or out-of-range normalized coordinates and confidences; and
- cropped RacketVision output, because its CSV does not yet carry a reliable source-coordinate transform.

There will be no nearest-frame matching, interpolation, implicit truncation, or new scene serialization. Existing source/model cache validation remains authoritative.

## First integration

When `--racketvision-csv` is supplied, `analyze-serve` will construct the scene and append diagnostic evidence to `serve-3d-diagnostics.json` without changing checkpoint scores or output checkpoint PTS. At minimum, diagnostics should include:

- ball-to-left-wrist 2D distance at and around selected release;
- the nearby frame with the strongest observed ball/hand separation evidence;
- ball-to-racket-hoop distance, or ball-inside-hoop status, around selected contact;
- the nearby frame with the strongest observed ball/racket contact evidence; and
- observation availability, confidence, candidate PTS, and signed delta from the selected checkpoint.

The existing path without the option must behave identically. Tests should cover joining, missing detections, coordinate normalization, mismatch rejection, and unchanged analysis without RacketVision input.

## Success criterion and next decision

Run the integration on the two current exemplars, including the known incorrect cropped-video release. The MVP succeeds if it produces auditable, correctly aligned evidence that helps explain agreement or disagreement with body/audio checkpoints. Only then decide whether to add ball/racket scoring cues, support another timeline, or expand the model.

## Non-goals

No existing cache migration; no `cut` integration; no generic track registry; no configurable alignment policies; no new smoothing; no multi-person or multi-camera model; no court coordinates, homography, or physics integration; and no production claim for RacketVision accuracy.

---

# Appendix A: Possible generalized design

If the MVP proves valuable and concrete consumers require more flexibility, `SceneTrack` could evolve from one strict joined frame sequence into a source-bound collection of independently versioned modality tracks.

## Independent observation tracks

A generalized model would store each modality at its native cadence:

```python
@dataclass(frozen=True)
class TimedSample(Generic[T]):
    time_seconds: float
    observation: T

@dataclass(frozen=True)
class ObservationTrack(Generic[T]):
    identity: TrackIdentity
    coordinate_space: CoordinateSpace | None
    samples: tuple[TimedSample[T], ...]
```

`TrackIdentity` could record source fingerprint, modality, schema and method versions, model/configuration identity, and parent track IDs. Raw, interpolated, filtered, smoothed, and physics-fitted values would then be separate tracks with explicit derivation lineage rather than columns blended into one artifact.

## Projected scene frames and explicit alignment

`SceneFrame` would become a view projected at a requested PTS rather than the stored form. Each value could carry alignment metadata such as `exact`, `nearest`, `interpolated`, `no_detection`, `outside_tolerance`, or `track_unavailable`, together with source PTS and temporal delta. Consumers would request an explicit per-modality alignment policy; no interpolation would be silent.

This would support sampled cutting poses, native-frame stage analysis, dense audio, and independently sampled visual models without forcing them onto one physical timeline.

## Coordinate spaces and transforms

Coordinate spaces could become first-class identities: upright source pixels, normalized image coordinates, MediaPipe hip-centered world coordinates, calibrated camera coordinates, or a court ground plane. Versioned transforms could connect spaces over a validity interval. A future court homography, for example, would be an independently replaceable image-to-court transform rather than a mutation of every observation.

Stereo support could add camera identities, synchronization metadata, calibration, and triangulated tracks while preserving each camera's raw 2D observations.

## Richer provenance and support

Track-level provenance could answer which model and configuration generated a series. Per-sample or per-keypoint support could separately describe whether a value was observed, interpolated, filtered, predicted, or fitted, along with confidence and temporal uncertainty. The existing filtered-world quality metadata is a useful precedent.

## Possible physical layout

Only if repeated joins become expensive or operationally awkward should the logical model gain a manifest or materialized cache, for example:

```text
scene/
  manifest.json
  timeline.jsonl
  tracks/
    body-mediapipe-2d.jsonl
    body-mediapipe-world-3d.jsonl
    ball-racketvision.jsonl
    racket-racketvision.jsonl
    audio-transients.jsonl
  derived/
    body-filtered-3d.jsonl
    ball-smoothed-2d.jsonl
```

This remains future work. The MVP should reveal which parts, if any, are justified by actual consumers.