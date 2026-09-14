# M4 3D Kinematic Waveform Remediation

> **Status:** approved replacement remediation design. This supersedes the 2D sparse-feature repair sequence in `m4-remediation-plan.md`. M3 improvement, including use of phase coherence to reduce false positives, is explicitly out of scope here; the frozen development annotations remain the only tuning evidence; held-out footage remains unread until configuration is frozen.

## Goal

The failed sparse 2D checkpoint implementation is replaced with one native-frame MediaPipe Heavy **3D kinematic phase path**. Its input is a per-frame kinematic track: exact presentation timestamps (PTS, the source video time at which each frame is displayed), `pose_world_landmarks`, and observation quality/missingness. Raw-source audio remains an optional contact cue. There are no new learned models, ball/racket trackers, multi-view inputs, or training.

Normalized `pose_landmarks` may be retained from the same MediaPipe inference solely for an optional pixel-aligned review overlay. They are never phase features, candidate evidence, or solver input. Dropping `z` from hip-centered world landmarks is not a valid projection onto the source video; a correct 3D projection would require unavailable camera calibration.

The checkpoint pipeline is:

```text
one explicit serve video/range → native-PTS 3D kinematic track
→ segment-safe Butterworth filter → 3D feature matrix
→ six composite anchor candidate sets → existing DP chronology search
→ two derived midpoint stages → source-frame review
```

## 1. Native 3D kinematic track

For one explicit serve video/range, decode every native source frame, preserve its exact PTS, and run the approved Heavy model once per frame. A source-bound kinematic-track cache exists only to avoid repeated inference; it is not a parallel analytical pipeline. It stores:

- `pose_world_landmarks` as the sole phase-analysis coordinates;
- direct-observation mask, source support time, interpolation span, temporal uncertainty, and model/configuration identity;
- optionally, synchronized normalized 2D landmarks for review-overlay rendering only.

Missing landmarks stay missing. No long gap is bridged. The legacy sparse 2D checkpoint cache is not an input to this path.

## 2. Filtering

Filter each 3D coordinate in each continuous visibility-qualified segment using a versioned low-pass Butterworth design. Filtering never crosses a missing span. It records edge/padding confidence separately from observed support.

M4.8 is authorized to add SciPy and use a second-order-sections Butterworth implementation. The filter operates on a canonical native-time grid built from exact PTS; short qualified resampling is explicit and cannot masquerade as direct observation. The implementation leaf must specify and version filter order, cutoff, zero-phase offline policy, minimum segment length, and boundary handling before any tuning.

## 3. Kinematic waveform matrix

All metrics are timestamped, quality-qualified waveform channels, not one-frame image extrema. Initial required channels are:

- bilateral knee flexion and extension velocity;
- bilateral elbow flexion/extension;
- shoulder-line and hip-line tilt;
- shoulder--hip transverse-plane separation;
- ``torso_verticality`` and ``hip_ankle_vertical_extent`` (upward-positive
  body-relative vertical extents; MediaPipe world landmarks are hip-centered,
  so neither measures absolute court height or global body translation);
- left-arm elevation and extension;
- right-wrist position relative to shoulder, elbow, torso, and pelvis;
- wrist/forearm velocity, acceleration, and inflection signals;
- post-contact whole-body settling;
- optional aligned raw-audio transient energy.

Coordinate-frame conventions, signs, normalization, derivative method, quality gates, and missingness behavior are versioned with the waveform configuration.

## 4. Composite anchor checkpoints

The detected anchor stages are `start`, `release`, `loading`, `cocking`, `contact`, and `finish`. Each candidate has a generic weighted composite unary score from multiple anatomical cue families plus an explicit support/coverage factor. Missing cues lower confidence; contradictory cues lower score. Weights and broad thresholds are tuned on frozen development exemplar 1, then frozen before a single confirmation run on exemplar 2.

- **start:** toss-arm rest/low state, general stillness, sustained left-wrist rise, and left-elbow extension onset.
- **release:** left-wrist mid-head-height crossing, sustained rise, left-arm extension, and continuing preparation support.
- **loading:** terminal trophy configuration: bilateral knee flexion, shoulder/hip tilt, shoulder--hip separation, elevated/extended left arm, and bent right elbow near shoulder height.
- **cocking:** relative right-wrist elevation trough or inflection before major upward acceleration, rising shoulder/hip reference, knee/hip unload, and loading-unwind signal.
- **contact:** right-wrist elevation apex followed by fall, wrist velocity/acceleration transition, late arm/torso configuration, and audio transient when present.
- **finish:** sustained right-wrist, torso, and ankle/foot image-plane settling.

## 5. DP and derived stages

The existing dynamic-programming chronology search remains the selection mechanism for the six composite anchor stages. Its recurrence, chronology constraints, transition logic, and explicit skip behavior are not redesigned in this remediation.

`acceleration` and `deceleration` are not independently searched. After anchor selection:

- `acceleration` is the exact temporal midpoint of selected `cocking` and `contact`;
- `deceleration` is the exact temporal midpoint of selected `contact` and `finish`.

A derived stage is unavailable when either bounding anchor is unavailable. Its confidence/provenance derives from those anchors and supported wrist waveform data at the midpoint.

## 6. Execution and gate

1. Implement/cache native world tracks.
2. Implement/filter the 3D waveform representation.
3. Implement composite six-anchor candidate scoring.
4. Adapt the existing DP integration to six searched anchors plus two derived midpoint stages.
5. Tune only on frozen development exemplar 1.
6. Freeze configuration and run frozen development exemplar 2 once; inspect phase-debug and unchanged evaluation outputs.
7. Only if that confirmation is acceptable, run once on session-disjoint held-out footage. Do not tune afterward.

Filtering, waveform construction, and candidate scoring may be in-memory analysis stages. Persist the reusable kinematic track, final checkpoints, and compact debug/review evidence rather than treating every intermediate representation as a separate user-facing pipeline artifact. Every development run retains exact PTS, candidate support, DP reconciliation, and manual-versus-selected review evidence.
