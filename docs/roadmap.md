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
| M2.3 | not started | — | — | — |
| M2.4 | not started | — | — | — |
| M2.5 | not started | — | — | — |
| M2.6 | not started | — | — | — |
| M3.1 | not started | — | — | — |
| M3.2 | not started | — | — | — |
| M3.3 | not started | — | — | — |
| M3.4 | not started | — | — | — |
| M3.5 | not started | — | — | — |
| M4.1 | not started | — | — | — |
| M4.2 | not started | — | — | — |
| M4.3 | not started | — | — | — |
| M4.4 | not started | — | — | — |

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
- **Behavior:** Ordered source timestamps, orientation normalization, cancellation, no full-video retention, requested-rate sampling independent of nominal FPS.
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
- **Dependency:** M2.2 integrated and orchestrator supplies approved URLs, hashes, licenses, and model contracts in the task.
- **Allowed:** `src/serve_review/models.py`, `src/serve_review/cli.py`, `tests/test_models.py`, `tests/test_cli.py`, `models/.gitkeep`.
- **Forbidden:** choosing a model/license, committing weights, implicit network during tests/setup, inference, unrelated dependencies.
- **Behavior:** No download without explicit command; temporary file and atomic rename; reject hash mismatch; models remain ignored; tests use a local fake transport.
- **Focused check:** `uv run pytest tests/test_models.py tests/test_cli.py`.
- **Exit:** Tests cover valid artifact, mismatch, interruption, existing valid file, offline error, and attribution output.

### M2.4 — MediaPipe Pose Landmarker adapter

- **Objective:** Run the approved local Pose Landmarker `.task` model and map its 33 landmarks to portable normalized observations.
- **Dependency:** M2.3 integrated and the manifest-approved artifact is available locally.
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

## M4 — Automatic checkpoints

Goal: add honest, optional review anchors without destabilizing serve cutting.

### M4.1 — Checkpoint domain schema

- **Objective:** Add versioned optional checkpoint intervals, confidence, provenance, visibility, and method identity.
- **Dependency:** M3 gate accepted.
- **Allowed:** `src/serve_review/domain.py`, `tests/test_checkpoints.py`.
- **Forbidden:** inference/heuristics, CLI, dependencies.
- **Behavior:** Checkpoints lie within their attempt; unavailable is representable; estimated contact cannot use visual-contact provenance; unsupported newer schemas fail safely.
- **Focused check:** `uv run pytest tests/test_checkpoints.py`.
- **Exit:** Golden round trips and invalid provenance/time/version cases pass.

### M4.2 — Focused dense pose extraction

- **Objective:** Reuse the pose backend at a denser schedule only inside accepted attempt windows.
- **Dependency:** M4.1 integrated.
- **Allowed:** `src/serve_review/pose/extract.py`, `src/serve_review/pose/cache.py`, `tests/test_dense_pose.py`.
- **Forbidden:** checkpoint selection, CLI, dependencies, model changes.
- **Behavior:** Attempt-window cache identity, bounded scheduling, deduplication at boundaries, cancellation, and no whole-session dense pass.
- **Focused check:** `uv run pytest tests/test_dense_pose.py`.
- **Exit:** Tests verify requested coverage, cache reuse/invalidation, overlapping windows, and processing bounds.

### M4.3 — Body-derived checkpoint baseline

- **Objective:** Emit optional loading, upward-swing, estimated-contact, follow-through, and landing intervals from dense body features.
- **Dependency:** M4.2 integrated and checkpoint rubric supplied in the task.
- **Allowed:** `src/serve_review/checkpoints/__init__.py`, `src/serve_review/checkpoints/body.py`, `tests/test_body_checkpoints.py`.
- **Forbidden:** racket/ball claims, CLI, dependencies, detector range changes.
- **Behavior:** Deterministic temporal windows; confidence/provenance; missing phases allowed; estimated contact never labeled visually observed.
- **Focused check:** `uv run pytest tests/test_body_checkpoints.py`.
- **Exit:** Synthetic motions cover full/abbreviated serve, occlusion, truncated range, absent landing, ambiguous contact, and determinism.

### M4.4 — Analyze command and checkpoint report

- **Objective:** Add `analyze` to load attempts, run dense extraction/body checkpoints, and atomically emit `checkpoints.json`.
- **Dependency:** M4.3 integrated.
- **Allowed:** `src/serve_review/analysis_pipeline.py`, `src/serve_review/cli.py`, `tests/test_analysis_pipeline.py`, `tests/test_cli.py`.
- **Forbidden:** racket model integration, output-video overlays, dependencies, private media.
- **Behavior:** Accept explicit attempts or prior cut output; retain cutting usability on analysis failure; cache reuse; honest partial/unavailable results.
- **Focused check:** `uv run pytest tests/test_analysis_pipeline.py tests/test_cli.py`.
- **Exit:** Fake end-to-end tests cover full/partial/no checkpoints, stale attempts, cancellation, malformed input, atomic output, and unchanged serve exports.

**M4 gate:** The user reviews checkpoint frames against manually chosen intervals. Record per-checkpoint usefulness and errors. Racket/ball model discovery becomes a new milestone only if body-derived anchors are insufficient and a licensed mobile-capable model has been explicitly approved.

## Completion rules

A leaf is not done because a worker says it is done. It requires a terminal Tau result, exact-diff review, passing verifier with candidate binding, approved integration, integrated `mise run check`, roadmap/run record update, dogfood entry, and safe workspace release. Hardware/private-video gates cannot be replaced by mocks. Never run dependent leaves concurrently or move the protected base while a run may resume.
