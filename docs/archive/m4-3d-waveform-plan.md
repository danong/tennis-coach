# M4 3D Kinematic Waveform Remediation

> **Status:** Historical · **State:** Complete · **Work:** None · **As of:** 2026-09-14

> **Status note:** implemented 3D production path; development configuration frozen and second-exemplar confirmation accepted. This supersedes the 2D sparse-feature repair sequence in `m4-remediation-plan.md`. M3 improvement, including use of phase coherence to reduce false positives, is explicitly out of scope here; frozen development annotations remain the only tuning evidence; held-out footage remains unread pending its one permitted evaluation run.

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

Filter each 3D coordinate in each continuous world-observation-qualified segment using a versioned low-pass Butterworth design. Filtering never crosses a missing span. It records edge/padding confidence separately from observed support. The implemented filter is order 4, 12 Hz, SciPy SOS zero-phase filtering, with a minimum 21-sample segment and short interior-gap interpolation only; every emitted row retains its exact source PTS.

The current implementation designs its filter from the median native-PTS step and applies it by sample index inside each qualified segment. It does not claim that irregular samples are uniformly resampled. Timestamp-uniform resampling/evaluation-back-to-PTS is deferred work, not an implemented property. The implementation leaf versions filter order, cutoff, zero-phase offline policy, minimum segment length, and boundary handling.

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

The detected anchor stages are `start`, `release`, `loading`, `cocking`, `contact`, and `finish`. Each candidate has a weighted composite unary score from multiple anatomical cue families plus an explicit support/coverage factor. Missing cues lower confidence; contradictory cues lower score. The current path still preserves a dense candidate at every native frame; stage-specific event eligibility is deferred.

Current development identities are `kinematic-waveforms-default-v3` / `kinematic-waveforms-v3`, `composite-anchors-default-v5` / `composite-anchors-v4`, and `phase-solver-serve-default-v2` for `analyze-serve`. These identities, together with the serialized schema versions, distinguish the corrected vertical convention, audio/peak cue behavior, start/finish calibration, and M4-only contact-to-finish bound from earlier output. The shared legacy solver default remains `phase-solver-default-v1` and is not given the M4 finish bound.

- **start:** toss-arm rest/low state, general stillness, sustained left-wrist rise, and left-elbow extension onset. Current development weights prioritize arm-low (`0.45`) and stillness (`0.35`) over rise (`0.15`) and elbow extension (`0.05`).
- **release:** left-wrist mid-head-height crossing, sustained rise, left-arm extension, and continuing preparation support.
- **loading:** terminal trophy configuration: bilateral knee flexion, shoulder/hip tilt, shoulder--hip separation, elevated/extended left arm, and bent right elbow near shoulder height.
- **cocking:** relative right-wrist elevation trough or inflection before major upward acceleration, rising shoulder/hip reference, knee/hip unload, and loading-unwind signal.
- **contact:** right-wrist elevation apex followed by fall, wrist velocity/acceleration transition, late arm/torso configuration, and audio transient when present.
- **finish:** post-contact right-wrist speed trough with whole-body and torso settling support. The current development composite weights the trough (`0.60`) above whole-body settling (`0.25`) and torso settling (`0.15`); `analyze-serve` additionally requires selected finish to be within `0.80 s` of selected contact. This is an M4 development guard against selecting eventual standing rest, not a claim of foot contact, jump landing, or image-plane evidence.

## 5. DP and derived stages

The existing dynamic-programming chronology search remains the selection mechanism for the six composite anchor stages. Its recurrence, chronology constraints, transition logic, and explicit skip behavior are not redesigned in this remediation.

`acceleration` and `deceleration` are not independently searched. After anchor selection:

- `acceleration` is the exact temporal midpoint of selected `cocking` and `contact`;
- `deceleration` is the exact temporal midpoint of selected `contact` and `finish`.

A derived stage is unavailable when either bounding anchor is unavailable. Its confidence/provenance derives from those anchors and supported wrist waveform data at the midpoint.

## 6. Current gate and deferred work

Native world caching, filtering, waveform construction, composite anchors, six-anchor DP integration, checkpoint writing, diagnostics, and source-frame review are implemented through `serve-review analyze-serve VIDEO`. Current calibration evidence is development-only; it is not held-out performance and does not complete the confirmation gate.

The configuration is frozen after direct calibration on the first development exemplar and an accepted no-change confirmation run on the second. Do not tune afterward. Deferred work, deliberately not folded into the current production path, is:

1. timestamp-uniform filtering with evaluation back at exact PTS;
2. 3D geometry reliability/limb-length consistency and angle-quality gating;
3. an arbitrary-time audit view in diagnostics/review;
4. sparse stage-specific event eligibility, including a low-before-sustained-rise start and first meaningful post-contact wrist trough finish;
5. a formally approved companion-2D observation-quality contract, if desired. Normalized 2D coordinates remain overlay-only unless that contract is explicitly revised.

The remaining gate is one session-disjoint held-out run. Its result must be reported without further tuning.

Filtering, waveform construction, and candidate scoring may be in-memory analysis stages. Persist the reusable kinematic track, final checkpoints, and compact debug/review evidence rather than treating every intermediate representation as a separate user-facing pipeline artifact. Every development run retains exact PTS, candidate support, DP reconciliation, and manual-versus-selected review evidence.
