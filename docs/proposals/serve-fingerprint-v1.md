# ServeFingerprintV1

> **Status:** Proposed · **State:** Ready for implementation review · **Scope:** Body-only serve description and alignment

## 1. V1 product contract

`ServeFingerprintV1` is a deterministic, versioned description of one detected serve attempt. It reuses the existing body waveform table and the six directly selected serve anchors. It does not replace checkpoint detection and does not introduce a parallel geometry pipeline.

The fingerprint has two complementary parts:

1. a small inventory of interpretable scalar measurements; and
2. a fixed-shape, phase-normalized sequence containing selected waveform channels.

### Supported comparison domain

V1 fingerprints are comparable only for:

```text
right-handed serves
+
compatible rear-view capture conditions
```

“Compatible” means broadly similar camera side, orientation, framing, and player scale in the image. V1 does not infer compatibility. A later comparison layer must not compare fingerprints outside this declared domain as though they were camera-invariant biomechanics.

### Included in V1

- body-only measurements already represented by `KinematicWaveformTrack`;
- five durations between the six directly selected anchors;
- anchor-sampled body geometry;
- interval extrema and extrema timing where they have a clear interpretation;
- a small fixed-shape sequence in existing waveform units;
- explicit missingness;
- compact source, extractor, coordinate, model, waveform, and checkpoint provenance.

### Excluded from V1

- ball, racket, image-space scene, and audio features;
- generalized handedness;
- cross-view normalization, camera calibration, and pose canonicalization;
- technique grades, serve-quality scores, or coaching judgments;
- calibrated uncertainty or probabilistic confidence;
- per-serve amplitude normalization, corpus scaling, imputation, PCA, clusters, embeddings, and similarity scores;
- ball outcomes and mechanics/outcome associations.

Existing ball/racket/audio cues remain available to checkpoint selection. Their exclusion applies only to the persisted fingerprint.

### Safe assumptions for later consumers

A consumer may assume that:

- schema and extractor identities define the exact metric and sequence inventory;
- source time is canonical PTS-based time;
- sequence values retain the units of the source waveform channels;
- only the sequence time axis has been normalized;
- `null` and `available: false` mean unavailable, never numeric zero;
- all five sequence segments have a fixed declared shape;
- actual tempo remains available from anchors and duration metrics;
- no statistical transform has been applied;
- comparison is supported only within the declared right-handed rear-view domain.

A consumer may not assume that MediaPipe world coordinates are calibrated biomechanics, that checkpoint scores are probabilities, or that V1 is camera invariant.

## 2. Persisted schema

Artifact filename:

```text
serve-fingerprint-v1.json
```

Top-level sketch:

```json
{
  "schema": "serve-fingerprint-v1",
  "schema_version": 1,
  "extractor": {
    "config_id": "serve-fingerprint-default-v1",
    "method_version": "serve-fingerprint-v1"
  },
  "identity": {
    "source_fingerprint": "sha256:...",
    "attempt_start_seconds": 12.345,
    "attempt_end_seconds": 15.678
  },
  "comparison_domain": "right-handed-compatible-rear-view-v1",
  "provenance": {
    "coordinate_convention": "mediapipe-world-hip-centered-v2",
    "waveform_method_version": "kinematic-waveforms-v3",
    "waveform_config_id": "kinematic-waveforms-default-v3",
    "checkpoint_method_version": "serve-waveform-v1",
    "checkpoint_config_id": "phase-solver-serve-default-v2",
    "body_model_name": "...",
    "body_model_version": "..."
  },
  "phase_layout": {
    "layout_id": "serve-five-segment-layout-v1",
    "samples_per_segment": 16,
    "resampling_method": "linear-exact-pts-complete-support-v1",
    "segments": [
      {"id": "start_to_release", "start_anchor": "start", "end_anchor": "release"},
      {"id": "release_to_loading", "start_anchor": "release", "end_anchor": "loading"},
      {"id": "loading_to_cocking", "start_anchor": "loading", "end_anchor": "cocking"},
      {"id": "cocking_to_contact", "start_anchor": "cocking", "end_anchor": "contact"},
      {"id": "contact_to_finish", "start_anchor": "contact", "end_anchor": "finish"}
    ]
  },
  "anchors": {
    "start": {"available": true, "time_seconds": 12.51},
    "release": {"available": true, "time_seconds": 13.04},
    "loading": {"available": true, "time_seconds": 13.37},
    "cocking": {"available": true, "time_seconds": 13.71},
    "contact": {"available": true, "time_seconds": 13.84},
    "finish": {"available": true, "time_seconds": 14.42}
  },
  "metrics": {
    "cocking_to_contact_duration": {
      "available": true,
      "value": 0.13,
      "unit": "seconds"
    },
    "right_wrist_speed_peak_cocking_to_contact": {
      "available": true,
      "value": 8.24,
      "unit": "body_lengths/second"
    }
  },
  "normalized_sequence": {
    "channel_order": ["knee_flexion_left", "..."],
    "channel_units": ["degrees", "..."],
    "shape": [5, 16, 12],
    "segments": [
      {
        "id": "start_to_release",
        "available": true,
        "duration_metric": "start_to_release_duration",
        "values": [[0.0, null], [0.0, null]],
        "availability": [[true, false], [true, false]]
      }
    ]
  }
}
```

The abbreviated `values` and `availability` arrays above are illustrative. Each persisted segment contains exactly 16 rows and 12 columns. The full tensor shape is always `[5, 16, 12]`.

### Identity and provenance rules

- `source_fingerprint` plus exact attempt range is the attempt identity. The synthetic `serve-001` identifier is not identity.
- No filesystem path is persisted.
- The body model identity is included because it directly changes persisted waveform values and is not encoded by the fingerprint extractor ID.
- The fingerprint does not store a dependency graph or every internal configuration.
- Anchor records contain only availability and selected PTS. Existing heuristic checkpoint scores are not copied or relabeled as probabilities.

### Missingness rules

- Scalar metrics always exist by name. Unavailable metrics use `available: false` and `value: null` while retaining their unit.
- Available anchors contain finite `time_seconds`; unavailable anchors contain `time_seconds: null`.
- Sequence cells use both `null` and a parallel boolean availability mask. A missing value is never encoded as zero.
- When either segment boundary is unavailable, the segment has `available: false`, and all values and masks are `null`/`false`.

## 3. Initial scalar metric inventory

### Shared extraction rules

- Anchor samples use the exact waveform row at the selected anchor PTS. Direct selected anchors already lie on the waveform PTS grid. Failure to find an exact row makes the dependent metric unavailable.
- Intervals are closed at both selected anchor PTS values.
- An interval reduction requires both anchors, at least three native waveform rows, and complete channel availability at every native row in the interval. This strict V1 rule avoids claiming an extremum when an unsupported portion could contain it.
- Extrema ties select the earliest PTS.
- `*_time_relative_to_contact` is `extremum_pts - contact_pts`; values before contact are negative.
- Flexion is `0` at full extension. “Peak extension rate” is `max(-flexion_velocity)`, reported as a positive magnitude.
- Availability follows only existing waveform availability and anchor availability. V1 adds no new confidence score.

| # | Name | Meaning | Source waveform channel(s) | Anchor / interval / reduction | Unit | Availability requirement |
|---:|---|---|---|---|---|---|
| 1 | `start_to_release_duration` | Time from start to release | none | `release_pts - start_pts` | seconds | Both anchors available |
| 2 | `release_to_loading_duration` | Time from release to loading | none | `loading_pts - release_pts` | seconds | Both anchors available |
| 3 | `loading_to_cocking_duration` | Time from loading to cocking | none | `cocking_pts - loading_pts` | seconds | Both anchors available |
| 4 | `cocking_to_contact_duration` | Time from cocking to contact | none | `contact_pts - cocking_pts` | seconds | Both anchors available |
| 5 | `contact_to_finish_duration` | Time from contact to finish | none | `finish_pts - contact_pts` | seconds | Both anchors available |
| 6 | `knee_flexion_left_at_loading` | Left-knee bend at loading | `knee_flexion_left` | Exact loading sample | degrees | Loading and channel sample available |
| 7 | `knee_flexion_right_at_loading` | Right-knee bend at loading | `knee_flexion_right` | Exact loading sample | degrees | Loading and channel sample available |
| 8 | `knee_flexion_left_max_release_to_cocking` | Deepest left-knee bend before cocking | `knee_flexion_left` | Maximum over release–cocking | degrees | Complete qualified interval |
| 9 | `knee_flexion_right_max_release_to_cocking` | Deepest right-knee bend before cocking | `knee_flexion_right` | Maximum over release–cocking | degrees | Complete qualified interval |
| 10 | `knee_extension_rate_left_peak_loading_to_contact` | Fastest left-knee extension | `knee_flexion_velocity_left` | Maximum of negated velocity over loading–contact | degrees/second | Complete qualified interval |
| 11 | `knee_extension_rate_right_peak_loading_to_contact` | Fastest right-knee extension | `knee_flexion_velocity_right` | Maximum of negated velocity over loading–contact | degrees/second | Complete qualified interval |
| 12 | `shoulder_hip_separation_max_release_to_contact` | Largest unsigned transverse shoulder/hip separation | `shoulder_hip_separation_transverse_deg` | Maximum over release–contact | degrees | Complete qualified interval |
| 13 | `shoulder_hip_separation_max_time_relative_to_contact` | Timing of maximum separation | same as #12 | PTS of #12 minus contact PTS | seconds | Metric #12 and contact available |
| 14 | `shoulder_tilt_at_loading` | Shoulder-line tilt at loading | `shoulder_tilt_deg` | Exact loading sample | degrees | Loading and channel sample available |
| 15 | `shoulder_tilt_at_contact` | Shoulder-line tilt at contact | `shoulder_tilt_deg` | Exact contact sample | degrees | Contact and channel sample available |
| 16 | `hip_tilt_at_loading` | Hip-line tilt at loading | `hip_tilt_deg` | Exact loading sample | degrees | Loading and channel sample available |
| 17 | `hip_tilt_at_contact` | Hip-line tilt at contact | `hip_tilt_deg` | Exact contact sample | degrees | Contact and channel sample available |
| 18 | `torso_verticality_at_loading` | Shoulder/hip vertical extent at loading | `torso_verticality` | Exact loading sample | body_lengths | Loading and channel sample available |
| 19 | `hip_ankle_vertical_extent_at_loading` | Hip/ankle vertical extent at loading | `hip_ankle_vertical_extent` | Exact loading sample | body_lengths | Loading and channel sample available |
| 20 | `toss_arm_elevation_at_release` | Left wrist elevation relative to left shoulder at release | `left_arm_elevation` | Exact release sample | body_lengths | Release and channel sample available |
| 21 | `toss_arm_elevation_max_release_to_cocking` | Highest relative toss-arm elevation after release | `left_arm_elevation` | Maximum over release–cocking | body_lengths | Complete qualified interval |
| 22 | `toss_arm_extension_at_release` | Left shoulder-to-wrist reach at release | `left_arm_extension` | Exact release sample | body_lengths | Release and channel sample available |
| 23 | `hitting_elbow_flexion_at_cocking` | Right-elbow bend at cocking | `elbow_flexion_right` | Exact cocking sample | degrees | Cocking and channel sample available |
| 24 | `hitting_elbow_flexion_at_contact` | Right-elbow bend at contact | `elbow_flexion_right` | Exact contact sample | degrees | Contact and channel sample available |
| 25 | `hitting_elbow_extension_rate_peak_cocking_to_contact` | Fastest right-elbow extension toward contact | `elbow_flexion_velocity_right` | Maximum of negated velocity over cocking–contact | degrees/second | Complete qualified interval |
| 26 | `hitting_elbow_extension_peak_time_relative_to_contact` | Timing of fastest elbow extension | same as #25 | PTS of #25 minus contact PTS | seconds | Metric #25 and contact available |
| 27 | `right_wrist_speed_peak_cocking_to_contact` | Peak normalized right-wrist speed toward contact | `right_wrist_speed` | Maximum over cocking–contact | body_lengths/second | Complete qualified interval |
| 28 | `right_wrist_speed_peak_time_relative_to_contact` | Timing of peak right-wrist speed | same as #27 | PTS of #27 minus contact PTS | seconds | Metric #27 and contact available |
| 29 | `right_wrist_shoulder_distance_at_cocking` | Right wrist distance from right shoulder at cocking | `right_wrist_rel_shoulder_distance` | Exact cocking sample | body_lengths | Cocking and channel sample available |
| 30 | `right_wrist_shoulder_distance_at_contact` | Right wrist distance from right shoulder at contact | `right_wrist_rel_shoulder_distance` | Exact contact sample | body_lengths | Contact and channel sample available |

### Camera-sensitive retained metrics

The following are retained because they are useful within the supported rear-view domain, but they are not camera-invariant biomechanics:

- transverse shoulder/hip separation;
- shoulder and hip tilt;
- vertical extent/elevation channels;
- MediaPipe-world wrist speed.

No raw `dx` or `dz` wrist component is included in V1. If empirical review shows one of the retained quantities is unstable even within compatible rear views, remove it by introducing a new extractor/config identity rather than silently changing V1 semantics.

## 4. Normalized sequence inventory

### Channel order

The sequence contains exactly these 12 channels:

1. `knee_flexion_left`
2. `knee_flexion_right`
3. `knee_flexion_velocity_left`
4. `knee_flexion_velocity_right`
5. `elbow_flexion_right`
6. `elbow_flexion_velocity_right`
7. `shoulder_hip_separation_transverse_deg`
8. `shoulder_tilt_deg`
9. `hip_tilt_deg`
10. `left_arm_elevation`
11. `left_arm_extension`
12. `right_wrist_speed`

This inventory intentionally excludes audio, local peak/trough flags, settling flags, ball/racket features, and camera-axis wrist components.

### Segment order

1. `start_to_release`
2. `release_to_loading`
3. `loading_to_cocking`
4. `cocking_to_contact`
5. `contact_to_finish`

The layout identity is `serve-five-segment-layout-v1`. This identity, rather than an implicit dependency on `STAGE_ORDER`, defines V1 sequence semantics.

### Sampling and interpolation

- Each segment contains 16 samples, including both endpoints.
- Target PTS values are evenly spaced in time between the two selected anchor PTS values: `start + i * duration / 15` for `i = 0..15`.
- Values are interpolated linearly in exact PTS time from the bracketing waveform rows.
- An exact target PTS uses the exact source value.
- A channel/segment is eligible only when both anchors exist, at least three native waveform rows lie in the closed segment, and every native waveform row in the segment has that channel available.
- If eligibility fails, all 16 values for that channel/segment are `null`, and its 16 mask entries are `false`.
- No interpolation crosses an unavailable interior row. V1 does not fill, smooth, extrapolate, or borrow from another segment.
- Shared anchors appear at the end of one segment and the start of the next. This intentional duplication keeps each segment independently interpretable.

The strict complete-support policy may produce more missing channel/segments than a future masked partial-resampling policy. It is preferred for V1 because it is simple and cannot fabricate motion through an unsupported interval.

### Retained absolute timing

Temporal normalization does not erase tempo because the artifact also stores:

- all six selected anchor PTS values; and
- the five scalar segment-duration metrics.

No per-segment speed adjustment is applied to values. Existing derivative values remain in degrees/second or body_lengths/second.

## 5. Code design and file-level changes

### Add

#### `src/serve_review/fingerprint.py`

Owns only the V1 contract and pure transformation:

- schema, method, config, layout, and filename constants;
- ordered metric and sequence inventories;
- small frozen schema value types with validation and deterministic JSON encoding;
- `build_serve_fingerprint_v1(...)`;
- exact-anchor lookup, strict interval reduction, and sequence resampling helpers.

Expected main signature:

```python
build_serve_fingerprint_v1(
    track: KinematicWaveformTrack,
    attempt_phase: AttemptPhase,
    *,
    source_fingerprint: str,
    body_model_name: str,
    body_model_version: str,
    config: ServeFingerprintConfig | None = None,
) -> ServeFingerprintV1
```

`AttemptPhase` supplies the effective attempt range, checkpoint method/config identity, and selected direct-anchor PTS. The extractor must reject an available direct anchor that is not exactly present on the waveform PTS grid.

#### `tests/test_fingerprint.py`

Focused synthetic tests for:

- schema validation and deterministic round-trip JSON;
- exact-anchor metrics and duration metrics;
- extrema values, timing, signed extension-rate conversion, and earliest tie handling;
- stable `[5, 16, 12]` sequence shape;
- exact-PTS linear interpolation;
- missing anchor and missing interior support behavior;
- rejection of ball/racket/audio channels from the declared inventories.

#### `docs/reference/serve-fingerprints.md`

A concise maintained reference derived from this approved proposal once implementation lands. It should document the artifact contract and operational output, not repeat implementation planning.

### Modify

#### `src/serve_review/analyze_serve.py`

- Import the fingerprint builder and filename constant.
- After `AttemptPhase` is constructed, build the fingerprint from the already existing `KinematicWaveformTrack` and selected checkpoints.
- Add fingerprint output collision handling and atomic JSON writing beside checkpoints and diagnostics.
- Add `fingerprint_path` to `AnalyzeServeResult`.
- Include the fingerprint temporary-file stem in cancellation cleanup.
- Do not move geometry, metric definitions, or interpolation logic into this module.

Resulting composition:

```text
FilteredWorldTrack
  -> KinematicWaveformTrack
  -> existing checkpoint scoring/selection
  -> AttemptPhase

KinematicWaveformTrack + AttemptPhase
  -> build_serve_fingerprint_v1()
  -> serve-fingerprint-v1.json
```

#### `src/serve_review/analysis_artifacts.py`

- Add `fingerprint_path` using the filename owned by `fingerprint.py`.
- Require a readable, supported V1 fingerprint in `is_complete()`.
- Keep `process.py` unaware of the filename.

#### `src/serve_review/process.py`

No production-flow change is expected. Test fixtures that synthesize a complete attempt must create a valid fingerprint artifact. Existing use of `AttemptAnalysisArtifacts.is_complete()` automatically adopts the new requirement.

#### `tests/test_analyze_serve.py`

- Assert the fingerprint is emitted at the expected path.
- Assert its source/range/anchor linkage and stable shape.
- Extend collision, force, and cancellation expectations to the new artifact.
- Confirm checkpoint, diagnostics, and review outputs retain existing behavior.

#### `tests/test_process.py`

- Update complete-attempt fixtures to include a valid fingerprint.
- Assert a missing fingerprint makes an attempt incomplete and schedules analysis.
- Continue asserting that `process.py` does not encode artifact filenames in workflow logic.

#### Documentation

- `docs/architecture/processing-lifecycle.md`: add fingerprint extraction and output.
- `docs/reference/serve-stage-checkpoint-analysis.md`: list the new output and note that checkpoints provide alignment landmarks.
- `docs/README.md`: link the maintained fingerprint reference after implementation.
- `README.md`: replace the vague automated-analysis future note with the now-supported descriptive fingerprint capability when the milestone is complete.

### Leave alone

- `src/serve_review/scene.py`
- `src/serve_review/checkpoints/scene_features.py`
- `src/serve_review/checkpoints/composite_anchors.py`
- `src/serve_review/checkpoints/six_anchor_solver.py`
- `src/serve_review/checkpoints/kinematic_waveforms.py`
- RacketVision and ball/racket caches
- checkpoint scoring weights and checkpoint selection behavior

The fingerprint consumes `KinematicWaveformTrack`; it does not add geometry to it or change its channel semantics.

## 6. Existing analyses and completeness

A missing `serve-fingerprint-v1.json` makes an attempt incomplete.

This is the simplest behavior consistent with the current pipeline:

- `process.py` already delegates completeness to `AttemptAnalysisArtifacts`;
- the fingerprint is produced during normal per-attempt analysis;
- the persisted waveform table itself is not currently an artifact;
- body-only fingerprint data could theoretically be reconstructed from the world cache and checkpoints, but no backfill path exists today.

No migration or backfill command will be added for V1. The consequence is that analyses created before this artifact exists will be scheduled for reanalysis. Current `process.py` clears the attempt destination before rerunning, including its colocated model caches, so this is a one-time full re-inference cost for existing attempts. The dataset is currently small, and accepting that cost is less scope than adding cache-preserving migration behavior.

## 7. Ordered implementation work items

### 1. Add fingerprint schema and fixed inventories

**Purpose:** Establish the persisted contract before extraction logic.

**Main locations:**

- add `src/serve_review/fingerprint.py`;
- add initial schema tests in `tests/test_fingerprint.py`.

**Observable result:** A valid `ServeFingerprintV1` can deterministically serialize/deserialize; invalid versions, shapes, units, missingness pairs, channel order, or segment order are rejected.

**Dependencies:** None.

### 2. Add scalar extraction

**Purpose:** Produce the exact 30 metrics from an existing waveform track and `AttemptPhase` without recomputing body geometry.

**Main locations:**

- `src/serve_review/fingerprint.py`;
- `tests/test_fingerprint.py`.

**Observable result:** Synthetic tracks produce expected duration, anchor-sampled, extrema, extension-rate, and extrema-timing metrics; unavailable support produces explicit null metrics.

**Dependencies:** Item 1.

### 3. Add phase sequence resampling

**Purpose:** Produce the fixed `[5, 16, 12]` phase-normalized sequence.

**Main locations:**

- `src/serve_review/fingerprint.py`;
- `tests/test_fingerprint.py`.

**Observable result:** Exact-PTS linear interpolation is deterministic; boundaries are retained; missing anchors or any unsupported interior source sample make the affected channel/segment unavailable; no values are imputed.

**Dependencies:** Item 1. It may reuse anchor/interval lookup helpers from item 2.

### 4. Compose fingerprint extraction into per-attempt analysis

**Purpose:** Build the fingerprint from the already computed waveform and selected checkpoints.

**Main locations:**

- `src/serve_review/analyze_serve.py`;
- `tests/test_analyze_serve.py`.

**Observable result:** `run_analyze_serve()` returns an `AnalyzeServeResult` with `fingerprint_path`, and the in-memory fingerprint matches source identity, effective attempt range, selected direct anchors, waveform identity, and body-model identity.

**Dependencies:** Items 1–3.

### 5. Persist the fingerprint atomically

**Purpose:** Publish the fingerprint as a normal per-attempt artifact with existing collision, force, and cancellation semantics.

**Main locations:**

- `src/serve_review/analyze_serve.py`;
- `tests/test_analyze_serve.py`.

**Observable result:** `serve-fingerprint-v1.json` is written beside checkpoints and diagnostics; collisions fail without `--force`; forced runs replace it; failures do not leave complete-looking temporary artifacts.

**Dependencies:** Item 4.

### 6. Integrate artifact completeness

**Purpose:** Make fingerprint presence part of a complete analyzed attempt without leaking filename knowledge into `process.py`.

**Main locations:**

- `src/serve_review/analysis_artifacts.py`;
- `tests/test_process.py`;
- fixture updates in `tests/test_analyze_serve.py` as needed.

**Observable result:** A valid fingerprint is required for completeness; a missing or invalid fingerprint schedules reanalysis; complete new attempts continue to skip; `process.py` uses only `AttemptAnalysisArtifacts`.

**Dependencies:** Item 5.

### 7. Update maintained documentation

**Purpose:** Make the new artifact, supported comparison domain, and exclusions discoverable.

**Main locations:**

- add `docs/reference/serve-fingerprints.md`;
- update `docs/architecture/processing-lifecycle.md`;
- update `docs/reference/serve-stage-checkpoint-analysis.md`;
- update `docs/README.md` and `README.md`.

**Observable result:** Documentation describes the implemented schema, body-only scope, phase layout, output location, completeness consequence, and downstream/non-goals accurately.

**Dependencies:** Items 1–6, so documentation reflects final names and behavior.

### 8. Run focused and repository checks

**Purpose:** Verify integration without opening a broader evaluation project.

**Main locations:** No new production locations.

**Observable result:** Fingerprint, analyze-serve, and process tests pass, followed by the repository's normal test/check commands. Existing checkpoint JSON, diagnostics, and review assertions remain unchanged except for the additional artifact.

**Dependencies:** Items 1–7.

## 8. Exit criteria

The milestone is complete when all of the following are true:

- The same analysis inputs produce byte-deterministic fingerprint JSON.
- The artifact carries `serve-fingerprint-v1`, schema version 1, extractor/config identity, phase-layout identity, source identity, exact attempt range, coordinate convention, body model identity, waveform identity, checkpoint identity, and six selected-anchor records.
- All 30 named scalar metrics are always present and use the specified units and availability semantics.
- The normalized sequence always declares and validates shape `[5, 16, 12]`.
- Every available sequence value is in the original waveform unit; phase normalization changes only target time positions.
- Actual durations remain available as scalar metrics.
- Missing anchors, missing anchor samples, insufficient interval support, and unsupported interior rows remain explicitly unavailable/null.
- No ball, racket, scene-image, audio, detector-flag, statistical-normalization, imputation, PCA, clustering, or similarity field appears in V1.
- Fingerprint extraction reads existing waveform channels and contains no body-landmark geometry implementation.
- `run_analyze_serve()` persists the fingerprint and exposes its path.
- `AttemptAnalysisArtifacts.is_complete()` requires a valid fingerprint.
- `process.py` contains no fingerprint filename knowledge.
- Existing checkpoint selection, checkpoint JSON, diagnostics, and review behavior are unchanged.
- Focused fingerprint/analyze/process tests and normal repository checks pass.

A small synthetic fixture should additionally verify one known piecewise-linear channel end to end, including expected scalar extrema, expected extrema PTS, 16 resampled values, and missing-gap behavior. No corpus-level accuracy benchmark is required for V1 completion.

## 9. Deferred work

The following are explicitly outside this milestone:

- left-handed or generalized handedness support;
- camera calibration, cross-view comparison, and camera/view normalization;
- pose canonicalization;
- ball/racket fingerprint channels and their reliability contract;
- audio fingerprint channels;
- calibrated checkpoint or metric uncertainty;
- partial-gap interpolation or learned imputation;
- personal baseline construction and robust deviation statements;
- statistical scaling and feature weighting;
- PCA and dimensionality reduction;
- nearest-neighbor retrieval;
- clustering and anomaly detection;
- learned temporal embeddings;
- outcome ingestion or association with speed, placement, or serve result;
- coaching judgments or generic technique/quality scores;
- migration/backfill infrastructure for pre-fingerprint attempts.

## Final review

No P0 findings.

No P1 findings.

[P2] Existing complete attempts incur one-time re-inference

Problem:
Adding the fingerprint to `AttemptAnalysisArtifacts.is_complete()` makes old attempts incomplete, and current `process.py` clears the whole attempt directory, including model caches, before rerunning.

Why it matters:
The first post-milestone processing run can be expensive for existing recordings.

Recommended change:
Accept and document the one-time cost for V1 because the dataset is small. Do not add migration or cache-preservation machinery unless actual rerun cost proves unacceptable.

Relevant code/docs:
`src/serve_review/process.py`
`src/serve_review/analysis_artifacts.py`
`docs/architecture/processing-lifecycle.md`

Verdict: READY TO IMPLEMENT

Blocking changes:
- None.

Implementation can begin with:
- Add the versioned fingerprint schema, fixed metric inventory, and fixed phase-layout contract in `src/serve_review/fingerprint.py` with focused schema tests.
