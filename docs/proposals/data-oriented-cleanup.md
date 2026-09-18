# A one-day, data-oriented cleanup of the serve analyzer

> **Status:** Proposed · **State:** Draft · **Work:** Next · **As of:** 2026-09-16

## The point of this cleanup

The prototype now does the useful thing: `mise run process VIDEO` finds an
attempt, builds body/ball/racket observations on native source PTS, scores
serve checkpoints, and writes a review. The next improvement should not be a
new model or a generalized framework. It should make the data path obvious
and delete code that cannot participate in that path.

This is a single-user, local, experimental tool. The target is fewer moving
parts and a codebase that can be understood by following immutable data from
one transformation to the next.

## 1. Where we are now

The active per-attempt path is:

```text
source video + requested attempt range
  -> dense native-PTS MediaPipe world/2D observations
  -> kinematic-track-v1.jsonl
  -> native-PTS RacketVision ball/racket observations
  -> racketvision-track-v1.jsonl
  -> aligned native-PTS audio energy/transient observations
  -> SceneTrack (strict aligned body, ball, racket, and audio observations)

body-world observations -> FilteredWorldTrack -> KinematicWaveformTrack
SceneTrack audio ----------------------------------------------^

SceneTrack -> SceneFeatureSeries
KinematicWaveformTrack + SceneFeatureSeries -> CompositeAnchorSet
CompositeAnchorSet -> SixAnchorSolution -> AttemptPhase
AttemptPhase -> checkpoints.json + review JPEGs/HTML
```

The durable artifacts are deliberately few: the two source/model-observation
caches and the final user-facing checkpoint/review output. Filtering,
waveforms, scene features, candidates, and DP solutions are in-memory values.
This is the right storage policy for a prototype.

The core data values already have useful names:

| Value | Current owner |
|---|---|
| Dense body observations | `pose/world.py` |
| Ball/racket observations | `tracking/racketvision.py` |
| Strict aligned multimodal observations | `scene.py` |
| Aligned audio sampling/qualification | currently `analyze_serve.py` + `media/audio.py` |
| Filtered body track | `checkpoints/world_filter.py` |
| Body/audio waveform table | `checkpoints/kinematic_waveforms.py` |
| Scene-derived feature series | currently `checkpoints/composite_anchors.py` |
| Dense scored candidates | `checkpoints/composite_anchors.py` |
| Chronological selected anchors | `checkpoints/six_anchor_solver.py` |
| I/O orchestration and final projection | `analyze_serve.py` |

There are also two sources of avoidable confusion:

1. `analyze_serve.py` is a large orchestration file that contains the only
   visible end-to-end flow, while its data transformations are spread across
   modules with historical M4/phase terminology.
2. The old sparse `phase_features -> evidence -> phase_solver` stack is not
   reachable from production analysis. It remains with a large test surface
   and re-exports, even though the active pipeline uses waveform composites
   and `six_anchor_solver` instead.

The just-added visual features expose a smaller version of the same issue:
`SceneFeatureSeries` is a real intermediate table, but it is currently built
inside the composite scorer and takes the waveform PTS sequence as an input.
That makes the feature projection depend on the table it should later be
combined with, rather than preserving SceneTrack's own canonical timeline.

There is a related canonical-data gap: `SceneTrack` can represent aligned
audio, but active orchestration builds it before audio sampling and passes
audio directly to the waveform builder. The result is two parallel views of
the attempt timeline rather than one aligned multimodal observation table.

## 2. Where we should go

The desired architecture is a short chain of typed tables, with each module
owning one transformation:

```text
source/model observations
  -> aligned observation table (`SceneTrack`)
  -> body/audio feature table     ->\
  -> scene feature table           -> scored candidate table
                                     -> selected-anchor table
                                     -> final checkpoint document
```

The rules are simple:

- **Observation-level aligned data owns no stage meaning.** `SceneTrack`
  remains an exact PTS-aligned view of body, ball, racket, and optional audio.
  Its audio transient flag may be qualified, but it is still an observation,
  not a biomechanical feature or decision. Do not add smoothing-derived
  biomechanics, normalized cues, weights, stage labels, or selection state to
  it.
- **Feature tables derive values, not decisions.** A scene-feature table owns
  handle/hoop orientation, ball/wrist distance, and ball/hoop distance. It
  takes only `SceneTrack` and preserves its PTS verbatim. A waveform table
  owns 3D body and audio quantities, using the audio rows already represented
  by `SceneTrack`. Missing inputs remain `None`; no interpolation or fallback
  policy is hidden here.
- **The scorer owns only cue transforms and weights.** It first rejects a
  waveform table and scene-feature table whose complete PTS sequences are not
  exactly equal. It can then turn a distance into a forward
  separation-onset cue or negate it for proximity, normalize it within an
  attempt, and make dense candidates. It must not decode media, inspect
  caches, or join observations.
- **The DP owns only chronology.** It consumes candidate scores and PTS; it
  must not know whether a score came from a wrist, microphone, ball, or
  racket.
- **The orchestrator owns I/O and composition only.** It loads/generates
  source/model observation artifacts, calls transformations in order, and
  writes final outputs. It
  does not define biomechanical features or solver recurrence.

This is data-oriented programming in the useful small sense: explicit,
immutable, PTS-aligned sequences cross narrow boundaries. It does not require
a feature registry, plugin system, generalized cache manifest, conversion
context, or another layer of factories.

## 3. A less-than-one-day path

### A. Make the scene-feature table explicit (about 1–2 hours)

Move `SceneFeatureSeries` and `build_scene_feature_series()` out of
`composite_anchors.py` into a small `checkpoints/scene_features.py` module.

Its only input is `SceneTrack`; its only output is an immutable table carrying
that exact PTS sequence:

```text
PTS
handle-to-hoop vertical orientation
left-wrist-to-ball distance
hoop-center-to-ball distance
```

`analyze_serve.py` builds this table from `SceneTrack` and passes it to
`build_composite_anchor_set(track, config, scene_features=...)`.
`build_composite_anchor_set()` owns the exact full-sequence PTS equality
check, because it is the first place that combines the two tables.
`composite_anchors.py` no longer imports `SceneTrack` or MediaPipe joint
indices. This makes the boundary between raw observations, derived features,
and weighted scoring visible without changing behavior or caches.

In the same small change, move SceneTrack construction until after aligned
source-audio sampling and transient qualification. Extend the scene builder to
accept one optional audio row per PTS (rather than requiring audio to be
all-or-nothing), then build the waveform's audio inputs from those same scene
rows. Audio sampling failure remains an honestly absent audio modality; it
must not prevent body/ball/racket analysis.

### B. Remove the unreachable sparse stage stack (about 2–3 hours)

Delete the production-unreachable modules and their dedicated tests:

```text
checkpoints/phase_features.py
checkpoints/evidence.py
legacy solving portions of checkpoints/phase_solver.py
```

Replace `PhaseSolverConfig`, the only active dependency from that area, with
`SixAnchorSolverConfig` beside `six_anchor_solver.py` (or in a tiny adjacent
chronology-config module). It contains only the active six-anchor fields:
`skip_penalty`, `contact_skip_penalty`, `min_transition_gap_seconds`,
`max_transition_gap_seconds`, `max_contact_to_finish_seconds`, and
`transition_bonus`, plus the current config identity. Do not copy sparse-path
observation floors, confidence bonuses/penalties, or body thresholds into the
new config. Remove stale package re-exports and tests that exist only for the
deleted sparse path.

This is deletion, not compatibility preservation. There is no production CLI
or artifact that consumes those modules.

### C. Add one current data-flow reference (about 30–45 minutes)

Replace the stale body-only section of
`docs/reference/serve-phase-analysis.md` with the actual table flow above:
raw body/racket caches, `SceneTrack`, body/audio and scene feature tables,
composites, DP, final stages. Link to this proposal only if the cleanup is
still in progress; otherwise fold the diagram directly into the reference.

### D. Stop (about 30 minutes for focused checks)

First capture a deterministic pre-refactor regression fixture: fixed
synthetic waveform and scene-feature tables, the resulting composite
candidate set, and the six-anchor solution. After the move/deletion work,
assert byte-equivalent candidate/solution serializations and selected PTS for
that fixture. Then run the focused active analysis tests, one forced anchor
process run, and inspect the generated checkpoint/review output. Do not
undertake a broad module rename, cache migration, new diagnostics,
score-reliability framework, or API compatibility layer in this cleanup.

## Acceptance criteria

At the end of the day:

1. A reader can trace one `process` attempt through no more than these values:
   source/model observations, aligned observations, scene/waveform features,
   candidates, solution, output.
2. `SceneTrack` remains observation-level aligned data, including optional audio rows, and
   `build_scene_feature_series()` depends only on it.
3. The composite builder rejects unequal waveform/scene-feature PTS sequences
   before scoring; it consumes feature tables rather than raw model
   representations.
4. No production-reachable code imports the removed sparse stage stack.
5. `mise run process refs/anchors/single-serve-01.mov --force` still creates
   the same artifact layout and successfully reaches a selected stage result.
6. The deterministic pre-refactor candidate/solution fixture is unchanged.
7. No new generic framework, persistence format, fallback mode, or process
   behavior is introduced.

This leaves room for later experimental weight tuning, but makes that tuning
a local edit to explicit feature and score tables rather than archaeology
through legacy phase code.
