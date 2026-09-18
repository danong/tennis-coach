# Processing lifecycle

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-16

This is the high-level lifecycle of:

```sh
mise run process -- VIDEO_OR_DIRECTORY [--dry-run] [--force]
```

It explains artifact ownership and data flow. Detailed detection/export behavior
is in [serve extraction and cutting](../reference/serve-extraction-and-cutting.md);
detailed checkpoint scoring is in [serve stage-checkpoint analysis](../reference/serve-stage-checkpoint-analysis.md).

## One source

```text
source video
  -> discover and probe source metadata/fingerprint
  -> detect serve attempts and export compilation
  -> for every accepted attempt:
       body-world observations + RacketVision observations + aligned audio
       -> SceneTrack
       -> SceneFeatureSeries + KinematicWaveformTrack
       -> CompositeAnchorSet
       -> SixAnchorSolver
       -> checkpoints, review frames, diagnostics
  -> metadata/index.html
```

A directory target processes its immediate MOV/MP4 children independently.
Original source media is never changed.

## Cut and export

`process` first delegates to `pipeline.run_cut`:

```text
source metadata
  -> sampled body/audio detection evidence
  -> accepted attempt ranges
  -> attempts.json
  -> exports/<video>/serves.mov
```

Detection owns ranges and exports. It does not select stage checkpoints.

## Per-attempt analysis

For each accepted range, `run_analyze_serve` owns the following in-memory
transformations:

```text
native-PTS MediaPipe body observations
native-PTS RacketVision ball/racket observations
aligned audio energy/transient observations
  -> SceneTrack                 # aligned observation-level rows

body-world observations
  -> FilteredWorldTrack
  -> KinematicWaveformTrack

SceneTrack
  -> SceneFeatureSeries

KinematicWaveformTrack + SceneFeatureSeries
  -> CompositeAnchorSet         # exact PTS sequences must match
  -> SixAnchorSolution          # chronology only
  -> eight checkpoint stages    # two are derived midpoints
```

`SceneTrack` owns aligned observations, not scoring: it has no normalized cues,
weights, stage labels, or solver decisions. Missing observations remain `None`.

## Reuse and force

Reusable per-attempt model artifacts live under:

```text
metadata/<video>/attempts/<serve-id>/cache/
  kinematic-track-v1.jsonl
  racketvision-track-v1.jsonl
```

A normal rerun skips complete artifacts. `--dry-run` reports intended actions
without writing. `--force` clears generated work and regenerates it. Cache
identity and exact source PTS, rather than nominal FPS, determine compatibility.

## Outputs

`process` publishes local generated artifacts beside the source:

```text
metadata/<video>/
  source.json
  attempts.json
  attempts/<serve-id>/checkpoints.json
  attempts/<serve-id>/serve-3d-diagnostics.json
  attempts/<serve-id>/review-serve-3d/
  index.html
exports/<video>/serves.mov
```

The final index links the attempt reviews. Checkpoint/review artifacts are
regenerated output; the two cache files are the reusable model-observation
artifacts.

## Module boundaries

| Module | Owns |
| --- | --- |
| `process.py` | Multi-source coordination, completeness, local summary |
| `pipeline.py` | Attempt detection and export orchestration |
| `analyze_serve.py` | Per-attempt I/O composition and output publication |
| `scene.py` | Exact PTS-aligned observation rows |
| `checkpoints/scene_features.py` | Image-space scene feature table |
| `checkpoints/kinematic_waveforms.py` | Body/audio waveform feature table |
| `checkpoints/composite_anchors.py` | Cue transforms, normalization, weighted candidates |
| `checkpoints/six_anchor_solver.py` | Chronological selection only |
