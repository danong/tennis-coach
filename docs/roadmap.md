# Serve Review offline roadmap

Status: active plan, 2026-09-09. This replaces the archived iOS-first roadmap in `old-docs/`.

Every numbered leaf below is scoped for exactly one isolated Tau run. Run only one at a time with `opencode/muse-spark-1.3-contributor-free`, review its exact candidate diff, integrate it, and make the integrated revision the dependency for the next run. Gates are orchestrator/user reviews, not delegated implementation items.

## Tau execution contract

Before dispatch, the orchestrator must have a committed base and stable bookmark. For each leaf:

1. Create a task JSON outside the repository. Copy the leaf's objective, dependency, allowed/forbidden scope, behavior, and checks into `goal`; workers do not inherit this roadmap or conversation automatically.
2. Tell the worker to read `AGENT.md`, `docs/design.md`, and the named adjacent code/tests. Setup is `mise install && mise run setup` in its fresh workspace.
3. Set the task verifier to the leaf's focused check followed by `mise run check`, with no skipped checks. Workers do not commit, move bookmarks, edit docs to excuse divergence, access `refs/`, or add dependencies unless the leaf explicitly allows it.
4. Run with model `opencode/muse-spark-1.3-contributor-free`, JJ workspace isolation, a fixed base revision, and that base as an immutable head.
5. Inspect and review the exact captured revision. Integrate only approved work, rerun `mise run check`, record the run ID/status in the table, append bounded feedback to `docs/tau-dogfood.md`, and release only after retaining or rejecting the candidate safely.

Default worker yield: summary, changed files, checks/results, deviations, and remaining risks. Request rescue rather than inventing a contract, weakening a test, touching private footage, or introducing an undeclared package.

## Tracking

| Leaf | Status | Tau run | Integrated revision | Notes |
| --- | --- | --- | --- | --- |
| Foundation | done | direct | initial base | mise/uv/CLI scaffold |
| M1.1 | done | `9caa1fad-e7b3-4deb-9a95-c91caef22e19` | `40540f728c0b` | reviewed, integrated, and released after topology repair |
| M1.2 | done | `99203706-0a25-4975-a8b3-9954f5d4243e` | `4bb98f3bbe28` | accepted; Display Matrix rotation coverage and integrated check passed |
| M1.3 | accepting | `0356c0d0-15f5-4cfb-81e8-0e2c3bb7fbcb` | — | reviewed synthetic fixture generator |
| M1.4 | accepting | `eb2adaa8-2201-4dfb-9a42-c672c7f56e07` | — | reviewed individual clip exporter |
| M1.5 | accepting | `447ec8e3-d675-4a39-b2f2-d2575d198169` | — | reviewed compilation exporter and manual export CLI |
| M2.1 | not started | — | — | — |
| M2.2 | not started | — | — | — |
| M2.3 | deferred | — | — | reproducible downloader deferred; local approved Heavy artifact is present |
| M2.4 | not started | — | — | — |
| M2.5 | not started | — | — | — |
| M2.6 | not started | — | — | — |
| M3.1 | not started | — | — | — |
| M3.2 | not started | — | — | — |
| M3.3 | not started | — | — | — |
| M3.4 | not started | — | — | — |
| M3.5 | not started | — | — | — |
| M4.1 | done | `73efa0fb-71b0-4f75-b953-db3b20a9e3da` | `e769cb6b98d6` | 720-test candidate verification; reviewed and accepted |
| M4.2 | done | `2591024a-7455-4fd5-8b71-137b46e7bd6f`; repairs `ec74444b-3383-4ba5-8896-cbf67ba76258`, `cf4d4edb-e266-430c-b439-dc461c4510cb` | `a38e4ab88563`; repairs `436e6f48f5b1`, `673b7105c7c5` | real-anchor smoke accepted: 145/145 direct-supported, contact audio -14 ms |
| M4.3 | done | `00055168-4215-428d-878f-60b9918627ab` | `8d1b1be6f873` | scale-invariant contact repair included; reviewed and accepted |
| M4.4 | done | `4c684400-913c-457a-9065-cd84a1096d7e` | `1d84e958b852` | DP skip states and advisory audio/body contact compatibility accepted |
| M4.5 | done | `45e93d07-87a1-4002-b305-59c64c022ef7` | `304544f3cb0a` | atomic analyze/checkpoints report accepted |
| M4.6 | not started | — | — | phase evaluation and manual gate |
| M4.7 | conditional | — | — | focused denser inference only if the M4 gate requires it |

## M1 — Trusted media path

Goal: prove probing, precise trimming, and compilation independently of machine learning.

### M1.1 — Media domain schemas

- **Objective:** Add versioned immutable source metadata, media range, and export-plan types with JSON codecs and validation.
- **Dependency:** Foundation base.
- **Allowed:** `src/serve_review/domain.py`, `tests/test_domain.py`.
- **Forbidden:** subprocesses, CLI changes, dependencies, inference, private media.
- **Behavior:** Half-open finite nonnegative ranges; `start < end`; clamp to source duration; deterministic JSON; reject unknown newer schema versions.
- **Focused check:** `uv run pytest tests/test_domain.py`.
- **Exit:** Tests cover boundaries, overlap/adjacency, clamping, invalid values, ordering, and JSON round trips; `mise run check` passes.

### M1.2 — ffprobe adapter and probe command

- **Objective:** Inspect a video with ffprobe and emit normalized `source.json` metadata.
- **Dependency:** M1.1 integrated.
- **Allowed:** `src/serve_review/media/__init__.py`, `src/serve_review/media/probe.py`, `src/serve_review/cli.py`, `tests/test_probe.py`, `tests/test_cli.py`.
- **Forbidden:** FFmpeg export, frame decoding, inference, new packages, private media.
- **Behavior:** Argument-array subprocess; parse fixture JSON; preserve rational frame/time values; actionable missing-tool/process/malformed-output errors; atomic JSON output.
- **Focused check:** `uv run pytest tests/test_probe.py tests/test_cli.py`.
- **Exit:** Fake-process tests cover HEVC metadata, rotation, variable-rate fields, missing streams, corrupt input, and ffprobe failure; CLI help documents `probe`.

### M1.3 — Synthetic media fixture generator

- **Objective:** Generate tiny portrait and landscape test videos locally using FFmpeg for integration tests.
- **Dependency:** M1.2 integrated.
- **Allowed:** `tests/media_factory.py`, `tests/test_media_factory.py`, `.gitignore` if needed.
- **Forbidden:** committing generated binaries, private media, production source changes, dependencies.
- **Behavior:** Deterministic command construction, bounded duration/resolution, visible timestamp/segment identity, skip nothing when FFmpeg is available.
- **Focused check:** `uv run pytest tests/test_media_factory.py`.
- **Exit:** Tests generate and probe both orientations in temporary directories; no generated media appears in JJ status.

### M1.4 — Individual clip exporter

- **Objective:** Export each validated source range as an independently playable clip.
- **Dependency:** M1.3 integrated.
- **Allowed:** `src/serve_review/media/export.py`, `tests/test_export_clips.py`.
- **Forbidden:** compilation, CLI wiring, stream-copy claims, inference, dependencies.
- **Behavior:** Use original source, exact filter-based trims, collision-safe names, temporary files plus atomic rename, cleanup on cancellation/failure, and explicit encoder choice.
- **Focused check:** `uv run pytest tests/test_export_clips.py`.
- **Exit:** Synthetic tests verify count, durations within one source sample, dimensions/orientation, ordering, invalid/empty plans, collision behavior, and no partial final files.

### M1.5 — Compilation exporter and manual export CLI

- **Objective:** Concatenate validated ranges without dead-time gaps and expose `serve-review export` using a ranges JSON file.
- **Dependency:** M1.4 integrated.
- **Allowed:** `src/serve_review/media/export.py`, `src/serve_review/cli.py`, `tests/test_export_compilation.py`, `tests/test_cli.py`.
- **Forbidden:** automatic detection, pose code, dependencies, private media.
- **Behavior:** Support `compilation`, `clips`, and `both`; preserve source order; reject overlap/empty output unless normalized by the domain plan; report output paths; input remains immutable.
- **Focused check:** `uv run pytest tests/test_export_compilation.py tests/test_cli.py`.
- **Exit:** Generated-media integration proves segment identity/order, expected duration, all output modes, padding-independent manual ranges, and actionable FFmpeg failures.

**M1 gate:** The user supplies a small manual ranges JSON for one ignored MOV and inspects clips plus compilation. Record dimensions, duration, cadence, orientation, and whether high-rate detail remains useful. Do not begin inference until export is accepted.

## M2 — Reproducible pose observations

Goal: produce cached body observations from timestamped frames without coupling model output to serve rules.

### M2.1 — Timestamped frame sampler

- **Objective:** Decode RGB frames on an explicit source-time schedule with bounded memory.
- **Dependency:** M1 gate accepted.
- **Allowed:** `pyproject.toml`, `uv.lock`, `src/serve_review/media/frames.py`, `tests/test_frames.py`.
- **Forbidden:** pose models, detector rules, CLI changes, private media. Add only the selected decode dependency documented in the task.
- **Behavior:** Decode sequential source frames and feed selected frames in strictly increasing source-time order; preserve canonical source timestamps separately from MediaPipe milliseconds; orientation normalization, cancellation, no full-video retention, and requested-rate sampling independent of nominal FPS. Initial policy is 30 Hz, with 60 Hz evaluation allowed on the hand-cut single-serve fixture.
- **Focused check:** `uv run pytest tests/test_frames.py`.
- **Exit:** Synthetic portrait/landscape and variable-schedule tests verify timestamps, coverage, cancellation, and a documented queue/memory bound.

### M2.2 — Pose observation and cache schemas

- **Objective:** Define portable person boxes, body keypoints, visibility, frame observations, cache identity, and atomic JSONL cache behavior.
- **Dependency:** M2.1 integrated.
- **Allowed:** `src/serve_review/pose/__init__.py`, `src/serve_review/pose/schema.py`, `src/serve_review/pose/cache.py`, `tests/test_pose_schema.py`, `tests/test_pose_cache.py`.
- **Forbidden:** model execution, detector logic, CLI, dependencies.
- **Behavior:** Upright unmirrored normalized coordinates; explicit missing joints; source fingerprint/model/sampling identity; interrupted cache never presents as complete.
- **Focused check:** `uv run pytest tests/test_pose_schema.py tests/test_pose_cache.py`.
- **Exit:** Round-trip, stale identity, partial resume, corrupt cache quarantine, coordinate validation, and atomic-write tests pass.

### M2.3 — Model manifest and verified downloader

- **Objective:** Define a model manifest and explicit command that downloads only approved artifacts and verifies SHA-256/license metadata.
- **Dependency:** M2.2 integrated. The project owner approved the local Heavy MediaPipe model/license for M2; the task records its pinned manifest contract.
- **Allowed:** `src/serve_review/models.py`, `src/serve_review/cli.py`, `tests/test_models.py`, `tests/test_cli.py`, `models/.gitkeep`.
- **Forbidden:** choosing a model/license, committing weights, implicit network during tests/setup, inference, unrelated dependencies.
- **Behavior:** No download without explicit command; temporary file and atomic rename; reject hash mismatch; models remain ignored; tests use a local fake transport.
- **Focused check:** `uv run pytest tests/test_models.py tests/test_cli.py`.
- **Exit:** Tests cover valid artifact, mismatch, interruption, existing valid file, offline error, and attribution output.

### M2.4 — MediaPipe Pose Landmarker adapter

- **Objective:** Run the approved local Pose Landmarker `.task` model and map its 33 landmarks to portable normalized observations.
- **Dependency:** M2.2 integrated and the approved local Heavy artifact is available at `models/pose_landmarker_heavy.task`; M2.3 downloader work is deferred.
- **Allowed:** `src/serve_review/pose/backend.py`, `src/serve_review/pose/mediapipe.py`, `tests/test_mediapipe_pose.py`, sanitized tiny fixtures.
- **Forbidden:** serve rules, CLI, model downloads, model/license changes, private frames, new dependencies.
- **Behavior:** Use ordered `VIDEO` calls; retain canonical source time separately from integer MediaPipe milliseconds; serialize calls to one landmarker; expose no-person/missing landmarks honestly.
- **Focused check:** `uv run pytest tests/test_mediapipe_pose.py`.
- **Exit:** Tests cover coordinate mapping, monotonically ordered timestamps, missing poses/landmarks, metadata, and deterministic adapter behavior.

### M2.5 — Pose extraction coordinator and diagnostic command

- **Objective:** Connect frame sampling, the MediaPipe adapter, and resumable cache; add a diagnostic extraction command.
- **Dependency:** M2.4 integrated.
- **Allowed:** `src/serve_review/pose/extract.py`, `src/serve_review/cli.py`, `tests/test_pose_extract.py`, `tests/test_cli.py`.
- **Forbidden:** serve detection/checkpoints, visualization, model downloads, dependencies, private media.
- **Behavior:** Bounded processing, progress, cancellation, cache hit/resume, serialized VIDEO inference, and explicit no-person frames.
- **Focused check:** `uv run pytest tests/test_pose_extract.py tests/test_cli.py`.
- **Exit:** Fake-backend end-to-end tests cover complete, partial resume, cancellation, stale cache, inference error, and CLI reporting.

**M2 gate:** The orchestrator runs approved models on short ignored footage, records processing seconds/source minute and representative failures, and the user confirms body tracks follow the server. Reject or change models before M3 if tracking is unusable. Do not commit footage or raw local paths.

## M3 — Automatic serve cutting

Goal: convert pose observations into useful attempt ranges and wire the public `cut` command.

### M3.1 — Temporal feature extraction

- **Objective:** Convert ordered pose observations into normalized visibility, motion, overhead, and rest features.
- **Dependency:** M2 gate accepted.
- **Allowed:** `src/serve_review/detection/__init__.py`, `src/serve_review/detection/features.py`, `tests/test_features.py`.
- **Forbidden:** candidate state machine, CLI, media, dependencies, private data.
- **Behavior:** Central versioned configuration; translation/scale normalization; source-time derivatives; smoothing with explicit boundaries; missing-data propagation.
- **Focused check:** `uv run pytest tests/test_features.py`.
- **Exit:** Synthetic sequences prove determinism, normalization invariance, unequal time-step handling, missing joints, and no fabricated evidence.

### M3.2 — Candidate range state machine

- **Objective:** Produce unpadded serve candidates from feature sequences using explicit hysteresis and duration rules.
- **Dependency:** M3.1 integrated.
- **Allowed:** `src/serve_review/detection/ranges.py`, `tests/test_detection_ranges.py`.
- **Forbidden:** CLI/media/cache changes, dependencies, hidden threshold tuning.
- **Behavior:** Reset/feed/finish semantics; minimum/maximum duration; merge gap; truncated first/last attempt; isolated toss rejection where feature evidence permits; abbreviated motion allowed.
- **Focused check:** `uv run pytest tests/test_detection_ranges.py`.
- **Exit:** Synthetic fixtures cover zero/one/multiple attempts, pauses, aborted toss, missing samples, nearby unrelated motion, back-to-back attempts, and end flushing.

### M3.3 — Attempt document and padding planner

- **Objective:** Turn candidates into versioned attempts with stable IDs, confidence/evidence, configured padding, and source-bound clamping.
- **Dependency:** M3.2 integrated.
- **Allowed:** `src/serve_review/domain.py`, `src/serve_review/detection/plan.py`, `tests/test_attempts.py`.
- **Forbidden:** export invocation, CLI, dependencies, model inference.
- **Behavior:** Preserve unpadded and effective ranges; deterministic IDs for one analysis run; union overlapping effective export ranges without rewriting detected ranges.
- **Focused check:** `uv run pytest tests/test_attempts.py`.
- **Exit:** Tests cover zero padding, source boundaries, overlap, ordering, confidence serialization, stable rerun, and empty detection.

### M3.4 — Evaluation matching and reports

- **Objective:** Validate session-disjoint annotation manifests and calculate deterministic range metrics.
- **Dependency:** M3.3 integrated.
- **Allowed:** `src/serve_review/evaluation.py`, `tests/test_evaluation.py`, `tests/fixtures/annotations/`.
- **Forbidden:** tuning detector thresholds, private manifests, CLI, dependencies.
- **Behavior:** Reject absolute paths/session leakage/invalid labels; one-to-one IoU matching; precision/recall, boundary error, retained duration, extra seconds, and machine-readable report.
- **Focused check:** `uv run pytest tests/test_evaluation.py`.
- **Exit:** Hand-calculated synthetic cases verify matching/ties/strata/ambiguity and invalid manifests.

### M3.5 — End-to-end cut command

- **Objective:** Replace the placeholder with probe → cached pose → detection → padding → JSON → requested export.
- **Dependency:** M3.4 integrated and frozen detector configuration supplied by orchestrator.
- **Allowed:** `src/serve_review/pipeline.py`, `src/serve_review/cli.py`, `tests/test_pipeline.py`, `tests/test_cli.py`.
- **Forbidden:** checkpoint logic, detector threshold changes, new dependencies, private media.
- **Behavior:** Existing CLI contract; progress and stage-specific errors; no fake full-video result on empty detection; atomic documents/outputs; cache reuse; source immutable.
- **Focused check:** `uv run pytest tests/test_pipeline.py tests/test_cli.py`.
- **Exit:** Fully faked orchestration plus generated-media integration verifies all output modes, padding, empty result, stage failure cleanup, cache hit, and deterministic attempts JSON.

**M3 gate:** Label development sessions, freeze configuration without viewing the held-out session, then run once on held-out footage. Target 95% recall and 90% precision for this personal setup and inspect every error. The user compares compilation review against raw scrubbing. Failed gates produce a new narrowly specified experiment rather than weaker metrics.

## M4 — Kovacs eight-stage serve phases

Goal: estimate the eight stages of the Kovacs & Ellenbecker serve model inside accepted, unpadded attempt ranges without destabilizing serve cutting. The stages are `start`, `release`, `loading`, `cocking`, `acceleration`, `contact`, `deceleration`, and `finish`.

Each stage is an estimated biomechanical phase, not necessarily one exact visual event. Output therefore includes a half-open source-time interval, an optional representative keyframe, confidence, availability, provenance, and limitations. Public JSON uses concise stage names; method limitations belong in provenance/metadata and documentation. Body pose must not claim direct observation of the ball, racket head, racket orientation, shoulder internal/external rotation, or visually observed contact.

M4 is non-blocking with respect to M3: phase completeness and structural anomaly flags never suppress, shorten, relabel, or otherwise change macro attempts or exports. Contact is anchored by raw-source audio with explicit uncertainty and may be supported by body pose, but it is not described as an exact visually observed frame.

### M4.1 — Eight-stage domain schema and rubric

- **Objective:** Add versioned attempt-phase documents and codify the observable body/audio rubric for all eight stages.
- **Dependency:** M3.5 integrated and the development serve-cutting baseline reviewed; the held-out cutting gate may remain pending because M4 cannot alter M3 output.
- **Allowed:** `src/serve_review/domain.py`, `src/serve_review/checkpoints/__init__.py`, `tests/test_checkpoints.py`.
- **Forbidden:** inference, feature heuristics, solver logic, CLI, dependencies, detector changes.
- **Behavior:** Every stage supports a half-open interval, optional keyframe, confidence, availability (`available`, `partial`, `unavailable`), provenance (`body_pose`, `audio_transient`, `body_pose_audio`, `manual`), evidence, and limitations. Documents include method/configuration identity, `structural_status` (`complete`, `partial`, `incomplete`, `unavailable`), and stable anomaly identifiers. Stage times lie inside the unpadded attempt and preserve canonical source time. Unsupported newer versions fail safely.
- **Rubric constraints:** `contact` is audio-anchored with temporal uncertainty; `release` is a body-pose estimate unless ball evidence exists; `cocking` is a body-pose stage estimate and cannot claim observed racket drop; camera-axis motion is not called anatomical posterior/forward without calibrated viewpoint; lead/toss/dominant-side identity is configured or explicitly uncertain.
- **Focused check:** `uv run pytest tests/test_checkpoints.py`.
- **Exit:** Golden round trips and invalid stage order, range, provenance, availability, uncertainty, anomaly, and version cases pass; all eight concise stage keys are covered.

### M4.2 — Attempt-local phase features

- **Objective:** Convert existing timestamped pose/audio observations inside each unpadded attempt into a uniform, confidence-aware feature grid suitable for phase inference.
- **Dependency:** M4.1 integrated.
- **Allowed:** `src/serve_review/checkpoints/phase_features.py`, `tests/test_phase_features.py`.
- **Forbidden:** stage selection, DP solver, CLI, detector changes, whole-session re-inference, new dependencies.
- **Behavior:** Slice strictly to the attempt range; resample only short visibility-qualified spans onto a uniform grid; retain `observed`, observation-quality, and interpolation-span channels; never bridge configured long gaps. Record source frame rate, pose observation rate, and analysis-grid rate. Define Savitzky–Golay/local-polynomial smoothing windows in seconds and derive legal odd sample windows from cadence. Compute timestamp-aware positions, angles, velocities, and accelerations with explicit boundary/gap confidence. Interpolation may support smoothing but cannot manufacture visual precision; uncertainty is never narrower than supporting observation spacing.
- **Signals:** Visibility-gated normalized wrist/elbow/shoulder/hip/knee/ankle trajectories, elbow and knee angles, shoulder/hip/torso axes, torso displacement/rotation proxies, and aligned audio transient candidates. Camera-relative quantities remain named camera-relative.
- **Focused check:** `uv run pytest tests/test_phase_features.py`.
- **Exit:** Synthetic 30/60/120 Hz and irregular-time sequences produce cadence-consistent extrema; tests cover short interpolation, forbidden long gaps, visibility loss, derivative boundaries, quality propagation, and no artificial precision from upsampling.

### M4.3 — Stage evidence and candidate generation

- **Objective:** Generate a bounded set of deterministic candidate frames/intervals and unary evidence scores `S_i(t)` for each of the eight stages.
- **Dependency:** M4.2 integrated and a written stage rubric supplied in the task.
- **Allowed:** `src/serve_review/checkpoints/evidence.py`, `tests/test_phase_evidence.py`.
- **Forbidden:** joint sequence solving, CLI, detector/export changes, racket/ball claims, dependencies.
- **Behavior:** Combine multiple visibility-qualified cues rather than define a stage by one global extremum. Weight evidence by observation quality; interpolated points cannot be the sole support for a high-confidence visual keyframe. Anchor contact candidates to audio transients with uncertainty. Support multiple candidates, explicit unavailable candidates, deterministic tie-breaking, configurable broad physiological search windows, and side/viewpoint uncertainty. Thresholds are centralized and versioned.
- **Focused check:** `uv run pytest tests/test_phase_evidence.py`.
- **Exit:** Synthetic full, abbreviated, occluded, truncated, low-knee-bend, and aborted sequences demonstrate sensible candidate sets without forcing missing evidence; audio-only contact cannot create unsupported surrounding body stages.

### M4.4 — Constrained joint phase solver

- **Objective:** Select the globally consistent eight-stage sequence using dynamic programming instead of greedy chained extrema.
- **Dependency:** M4.3 integrated.
- **Allowed:** `src/serve_review/checkpoints/phase_solver.py`, `tests/test_phase_solver.py`.
- **Forbidden:** feature extraction, CLI, detector/export changes, dependencies, private tuning data.
- **Behavior:** Maximize `sum S_i(T_i) + sum Q_i(T_i, T_{i+1})` under chronological ordering and broad versioned transition constraints. Every stage has an unavailable/skip state with an explicit penalty; the solver never fabricates a stage merely to complete the sequence. Contact is a strong uncertain anchor, not an unconstrained exact frame. Produce intervals, representative keyframes, confidence/evidence, structural status, and non-blocking anomaly flags. Deterministic tie-breaking is mandatory.
- **Focused check:** `uv run pytest tests/test_phase_solver.py`.
- **Exit:** Hand-calculated paths verify global-over-local choices, ordering, ties, skip states, uncertainty propagation, contact anchoring, missing data, truncated attempts, and deterministic output.

### M4.5 — Analyze command and phase report

- **Objective:** Add `analyze` to load accepted attempts and cached pose/audio observations, run phase features/evidence/solver, and atomically emit `checkpoints.json`.
- **Dependency:** M4.4 integrated.
- **Allowed:** `src/serve_review/analysis_pipeline.py`, `src/serve_review/cli.py`, `tests/test_analysis_pipeline.py`, `tests/test_cli.py`.
- **Forbidden:** detector threshold changes, suppression of attempts/exports, output-video overlays, dense re-inference, dependencies, private media.
- **Behavior:** Accept explicit attempts or prior cut output; reuse compatible caches; retain cutting usability on analysis failure; emit honest full/partial/unavailable results and stage-derived metrics only when their required stages are available. Never modify `attempts.json`, clips, or compilation.
- **Focused check:** `uv run pytest tests/test_analysis_pipeline.py tests/test_cli.py`.
- **Exit:** Fake end-to-end tests cover full/partial/no phases, stale inputs, cancellation, malformed data, atomic output, deterministic reruns, and byte-unchanged serve exports.

### M4.6 — Phase evaluation and manual gate

- **Objective:** Validate session-disjoint phase annotation manifests and report interval/keyframe quality without tuning on held-out footage.
- **Dependency:** M4.5 integrated.
- **Allowed:** `src/serve_review/checkpoint_evaluation.py`, `tests/test_checkpoint_evaluation.py`, sanitized annotation fixtures.
- **Forbidden:** solver/feature threshold changes, CLI, dependencies, private manifests.
- **Behavior:** Compare representative keyframes with manually accepted intervals; report availability, interval overlap, median and high-percentile timing error, stage-order violations, confidence calibration, structural completeness, and serve-versus-aborted structural separation. Do not score an unavailable body-only claim as if racket/ball ground truth existed.
- **Focused check:** `uv run pytest tests/test_checkpoint_evaluation.py`.
- **Exit:** Hand-calculated synthetic cases verify interval acceptance, uncertainty, missing stages, ambiguity, ties, split leakage, and deterministic reports.

**M4 gate:** The user labels representative stage intervals on development attempts, configuration is frozen, and the system is run once on session-disjoint footage. Review every stage and anomaly, recording per-stage availability, timing usefulness, systematic errors, and whether 30 Hz pose observations are adequate. Structural status remains advisory and cannot alter M3 exports.

### M4.7 — Conditional focused denser pose extraction

- **Objective:** Re-run pose inference at a higher observed cadence only inside accepted attempt windows if the M4 gate shows that existing observations are temporally inadequate.
- **Dependency:** M4 gate evidence demonstrating a concrete cadence-related failure; otherwise this leaf remains deferred.
- **Allowed:** `src/serve_review/pose/extract.py`, `src/serve_review/pose/cache.py`, `src/serve_review/analysis_pipeline.py`, `tests/test_dense_pose.py`, `tests/test_analysis_pipeline.py`.
- **Forbidden:** whole-session dense passes, model changes, solver threshold changes, dependencies, private media.
- **Behavior:** Separate attempt-window cache identity, bounded source-frame scheduling, actual inference on requested frames, overlap deduplication, cancellation, and explicit observed cadence metadata. Resampling never substitutes for requested visual observations.
- **Focused check:** `uv run pytest tests/test_dense_pose.py tests/test_analysis_pipeline.py`.
- **Exit:** Tests verify requested observed coverage, cache reuse/invalidation, overlapping windows, processing bounds, and truthful distinction between observed and interpolated samples.

## Completion rules

A leaf is not done because a worker says it is done. It requires a terminal Tau result, exact-diff review, passing verifier with candidate binding, approved integration, integrated `mise run check`, roadmap/run record update, dogfood entry, and safe workspace release. Hardware/private-video gates cannot be replaced by mocks. Never run dependent leaves concurrently or move the protected base while a run may resume.
