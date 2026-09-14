# Serve Review roadmap: MVP1–MVP2

Status: implementation-ready plan, 2026-09-04. This roadmap assumes no application code exists.

## How to use this roadmap

Milestones are ordered release gates. A milestone is complete only when every leaf work unit is complete and its milestone exit evidence has been reviewed. IDs are stable; reference them in branches, commits, and review notes.

Each leaf unit is deliberately narrow enough for a low-cost implementation model. Assign exactly one leaf unit at a time. The assignee receives the relevant design section, prerequisite commits, allowed paths, and the named checks. Unless a unit explicitly says otherwise, it may modify only the files named in its deliverable and their matching tests.

Statuses should be tracked beside each ID when implementation begins: `not started`, `in progress`, `blocked`, or `done`. “Done” means the exit criteria pass on the worker's checkout and the orchestrator has reviewed the diff. Generated `.xcodeproj` output, private footage, signing configuration, and local evaluation manifests are never committed.

### Milestone map

| Milestone | Outcome | Gate owner |
| --- | --- | --- |
| M0 | Reproducible project and commands | Orchestrator review |
| M1 | Stable portable domain and adapter contracts | Orchestrator architecture review |
| M2 | Offline attempt detector and honest metrics | Orchestrator detector gate |
| M3 | Reliable on-device recording and persistence | Device evidence + orchestrator review |
| M4 | Automatic post-stop analysis | Device evidence + detector regression |
| M5 | Fast attempt review and manual repair | User workflow test |
| M6 | Gap-free export to Photos | Device evidence + media review |
| M7 | MVP1 release validation | User acceptance |
| M8 | MVP2 annotation and model feasibility | Explicit go/no-go review |
| M9 | Hand/racket tracking and phase analysis | Accuracy gate |
| M10 | MVP2 inspection experience | User acceptance |

## Delegation contract

Use gpt-5.6-luna for leaf units with frozen inputs and mechanical exit checks. Reserve a stronger model or direct orchestrator work for contract design, detector interpretation, media/concurrency review, model licensing, evaluation leakage checks, and milestone acceptance.

Every delegated prompt must contain:

- One leaf ID and one-sentence objective.
- The exact prerequisite commits or statement that dependencies are present.
- Allowed files/directories and explicit forbidden scope.
- Relevant contracts copied or linked from the [design](design.md).
- Fixtures available and commands to run.
- Exit criteria copied verbatim.
- A requirement to report changed files, checks and results, assumptions, and remaining risks.

Workers must stop and report when a public contract is missing, a prerequisite fails, a new dependency appears necessary, or acceptance criteria conflict. They must not redesign adjacent modules, weaken tests or thresholds, edit docs to legitimize divergence, add compatibility wrappers speculatively, use private footage in commits, or claim device validation from simulator results.

The orchestrator reviews every leaf for scope, behavior, tests, and accidental generated/private files. At each milestone gate, run the full available `mise run check`, inspect the diff, and record device/evaluation evidence where required. Shared contract and project files have a single writer at a time; parallelize only tickets with disjoint ownership after their interfaces are frozen.

## M0 — Bootstrap and reproducible commands

Goal: a new checkout can install auxiliary tools, generate the project, and run truthful checks.

### M0.1 — Repository hygiene

**Work:** Initialize Git if still absent; add `.gitignore` entries for generated projects/build products, mise local state, signing overrides, evaluation output, local manifests, and private footage. Add empty source/test directory placeholders only where Git requires them.

**Exit:** `git status --short` contains the intended docs/configuration only; a fixture placed in the documented private-footage path is ignored; source fixtures remain trackable. No reference document is deleted or rewritten.

### M0.2 — Pin the tool contract

**Work:** Add `mise.toml` and a committed Xcode-version policy file. Pin XcodeGen to an exact tested version through a mise-supported backend. Define task names from the README, using placeholder failures only for tasks whose implementation belongs to later milestones. Do not install Swift through mise.

**Exit:** `mise config` parses the file; `mise install` installs XcodeGen; tool output matches the pin; every advertised task is either functional or exits nonzero with a message naming its enabling milestone.

### M0.3 — Add environment diagnostics

**Work:** Implement `mise run doctor` as a small checked-in script. Verify full Xcode selection, compatible Xcode/Swift/iOS SDK, an installed simulator runtime, XcodeGen, and available disk space. Report physical-device signing as a manual prerequisite rather than failure unless device mode is explicitly requested.

**Exit:** Unit-test parseable helper logic where applicable; run once with a valid setup and once with one dependency hidden or overridden. Both results are actionable and the invalid case exits nonzero.

### M0.4 — Generate a minimal app and Swift package

**Work:** Add `project.yml`, an iOS 18 SwiftUI application target, unit/UI test targets, and `Packages/ServeCore/Package.swift` with empty `Domain`, `Detection`, and `Editing` library targets. Keep the bundle identifier configurable and signing out of source control.

**Exit:** `mise run generate`, `mise run test-core`, and `mise run build-ios` succeed with the selected toolchain; regenerating causes no tracked diff; the app launches in a simulator.

### M0.5 — Make the check task authoritative

**Work:** Implement `test-ios` and `check`; select an available simulator deterministically or accept an override. Print tool/SDK/destination at the start and fail on missing prerequisites.

**Exit:** `mise run check` performs core tests plus generation and simulator build/tests; a deliberately invalid destination fails rather than skips; README commands match actual behavior.

**M0 gate:** Review pins, generated-file policy, task truthfulness, Swift 6 strict-concurrency settings, and a clean regeneration. Record the first known-good Xcode build.

## M1 — Domain model and frozen contracts

Goal: downstream workers can implement modules without inventing shared semantics.

### M1.1 — Rational media time and ranges

**Work:** Implement portable `MediaTime` and half-open `MediaRange` value types with checked construction, comparison across timescales, duration, intersection, clamp, and JSON coding. Avoid floating-point storage.

**Exit:** Tests cover unequal timescales, invalid/zero timescales, boundaries at zero/duration, adjacent versus overlapping ranges, large values, and coding round trips; `mise run test-core` passes.

### M1.2 — Recording and attempt documents

**Work:** Implement the version-1 persistent structs described in the design: recording metadata/status/revision, attempt detection and optional override, analysis run, export receipt, source fingerprint, and versioned method identifiers. Add centralized validation.

**Exit:** Tests cover valid round trips, unknown enum values where forward-compatible, rejected invalid ranges/references, effective-range behavior, and refusal of unsupported newer schema versions.

### M1.3 — Pure edit operations

**Work:** Implement add, boundary-adjust, include/exclude/restore, split, selection, and effective-range ordering as pure operations returning a new validated revision. Define stable-ID behavior in tests.

**Exit:** Tests demonstrate stale revision rejection, unchanged detected ranges after overrides, split IDs and bounds, excluded-item navigation policy, empty state, and deterministic ordering.

### M1.4 — Export planning

**Work:** Implement a pure planner that validates included effective ranges, sorts and unions overlap, preserves adjacency without inserting gaps, and returns expected output duration and edit revision.

**Exit:** Table-driven tests cover unordered, overlapping, adjacent, excluded, overridden, out-of-bounds, empty, and mixed-timescale inputs. No AVFoundation import exists in `ServeCore`.

### M1.5 — Adapter protocol surface and fakes

**Work:** Add the platform-side protocols from the design and minimal test fakes. Keep framework payload types inside the adapter layer. Add a composition-root placeholder showing injection into one trivial feature coordinator.

**Exit:** App and test targets compile; fakes can model success, cancellation, interruption, stale save, and failure; no feature view directly constructs an adapter.

### M1.6 — Schema fixtures and compatibility checks

**Work:** Commit small version-1 JSON fixtures and golden decoding/encoding tests. Document the migration rule beside the codec implementation.

**Exit:** Golden fixtures decode, round-trip semantically, and reject a fabricated newer schema without deleting or rewriting it.

**M1 gate:** Orchestrator reviews naming, time semantics, revision rules, Sendable/isolation decisions, and adapter boundaries. Freeze these contracts before delegating M2–M6 work; later changes require a focused design update and compatibility test.

## M2 — Offline attempt detection and evaluation

Goal: establish on real held-out footage that pose-derived ranges are useful before coupling detection to recording.

### M2.1 — Local dataset manifest and annotation format

**Work:** Define a versioned manifest with relative asset paths, session-level split, view/motion tags, visibility, attempt ranges, and ambiguous/excluded intervals. Provide synthetic examples and validation; keep real manifests ignored.

**Exit:** Validator rejects absolute paths, split leakage by session ID, invalid ranges, duplicate IDs, and missing labels; synthetic manifest passes.

### M2.2 — Evaluation matching and metrics

**Work:** Implement deterministic one-to-one interval matching and metrics specified in the design, independent of Vision. Emit machine-readable JSON and concise text.

**Exit:** Hand-calculated fixtures verify match tie-breaking, IoU threshold, precision/recall, boundary errors, containment, retained-duration ratio, extraneous seconds, strata, and ambiguous counts.

### M2.3 — Offline asset frame reader

**Work:** Implement an AVFoundation adapter that decodes source-relative frames at an explicit sampling schedule and normalizes rotation metadata. Add a tiny generated landscape and portrait fixture.

**Exit:** Integration tests verify ordered timestamps, requested coverage, duration bounds, normalized orientation, cancellation, and no unbounded frame retention.

### M2.4 — Apple Vision body-pose adapter

**Work:** Map Vision body observations to portable normalized observations with confidence/visibility and explicit missing points. Record API revision and configuration.

**Exit:** Adapter tests with stored non-private frames verify coordinate transform, missing-person/missing-joint behavior, ordering, cancellation, and repeatable metadata. No detector thresholds appear in this adapter.

### M2.5 — Pose cache and resumable coverage

**Work:** Define and implement a versioned local pose sidecar keyed by source fingerprint, estimator version, transform, and sampling policy. Track completed sample intervals and atomic writes.

**Exit:** Tests cover cache hit/miss, interrupted writes, corrupt sidecar quarantine without source deletion, partial coverage resume, and invalidation after source/configuration change.

### M2.6 — Pure temporal features

**Work:** Convert ordered observations into a documented small feature vector such as joint visibility, normalized wrist/elbow/shoulder motion, torso scale, overhead evidence, and rest/motion evidence. Keep configuration centralized.

**Exit:** Synthetic sequences verify invariance to image translation/scale within tolerance, honest missing-data propagation, smoothing boundaries, and deterministic output.

### M2.7 — Candidate range state machine

**Work:** Implement reset/feed/finish detection over feature sequences, including start/end hysteresis, minimum/maximum duration, merge rules, and configurable padding. It must permit abbreviated motions.

**Exit:** Synthetic tests cover one/multiple attempts, long pauses, aborted toss, missing samples, truncated first/last attempt, false nearby motion, back-to-back attempts, and finish flushing.

### M2.8 — Evaluation command

**Work:** Wire manifest validation, frame reader, cached pose extraction, detector, and metrics into `mise run evaluate`. Write versioned reports outside tracked source.

**Exit:** The command processes the synthetic manifest end to end, supports each split, refuses cross-split tuning inputs, resumes after cancellation, and returns nonzero for invalid data or unmet requested gates.

### M2.9 — Development-corpus tuning

**Owner:** Orchestrator or reviewed detector specialist; do not delegate as an unconstrained Luna task.

**Work:** Label supplied sessions, reserve session-disjoint held-out data, inspect errors, and tune only the development split. Record configuration and declared supported strata.

**Exit:** Dataset floors from the design are met or gaps are recorded; a frozen detector/configuration is selected without inspecting held-out outcomes for further tuning.

### M2.10 — Held-out detector gate

**Owner:** Orchestrator.

**Work:** Run the frozen detector once on held-out footage, inspect representative false positives/negatives and boundary failures, and issue a go/no-go report.

**Exit:** Metrics meet the design gates overall and per adequately sized declared stratum. Otherwise MVP1 automatic detection is blocked and a separately scoped detector experiment is approved before M3 integration; thresholds remain unchanged.

**M2 gate:** Commit code, schemas, synthetic fixtures, frozen configuration, and aggregate report. Keep private assets/manifests local. Review dataset leakage, padding metrics, supported-setup wording, and reproducibility.

## M3 — On-device capture and durable recordings

Goal: record high-frame-rate silent source media without analysis harming capture.

### M3.1 — Camera capability discovery

**Work:** Implement format enumeration and deterministic selection for rear wide 1080p/120 fps or visible 60 fps fallback. Produce user-facing reasons for unavailable configurations.

**Exit:** Unit tests over captured format descriptors cover supported/fallback/no-camera cases; a device log records the selected format, dimensions, nominal frame rate, and camera on the target iPhone.

### M3.2 — Capture state machine

**Work:** Implement permission and lifecycle states: idle, preparing, recording, stopping, finalized, interrupted, and failed. Keep transitions separate from AVFoundation callbacks.

**Exit:** Deterministic tests cover denied permission, double start/stop, background interruption, writer failure, low-storage signal, and recovery to a non-recording state.

### M3.3 — AVFoundation writer adapter

**Work:** Configure capture, serialize sample appends, normalize timestamps to the first video sample, write silent `.mov` media, and finalize atomically into a recording directory. Analysis callbacks receive bounded references only.

**Exit:** Device tests produce playable 60 and supported 120 fps files with monotonic timestamps and correct orientation; injected append/finalize failures preserve/report recoverable artifacts; no microphone permission appears.

### M3.4 — Analysis sampling queue

**Work:** Add the 15 Hz configurable sampler with at most one in-flight and one replaceable pending frame. Record intended samples, analysis skips, and capture-quality events separately.

**Exit:** Stress tests with a deliberately slow consumer demonstrate bounded memory/queue depth and unaffected writer completion; counters distinguish analysis replacement from capture drops.

### M3.5 — Atomic recording store

**Work:** Implement per-recording directories, atomic JSON revision saves, listing, opening, and explicit deletion. Handle missing/corrupt documents without deleting source media.

**Exit:** Integration tests cover relaunch, stale revision, interrupted temp write, orphaned finalized source, missing source, corrupt document, deletion, and unsupported future schema.

### M3.6 — Capture UI

**Work:** Build recordings list and camera screen with framing guide, actual format indicator, Record/Stop, elapsed time, storage/error states, orientation lock, and idle-timer handling.

**Exit:** UI tests cover permissions and state presentation; on device, a user can create, stop, reopen, and delete a recording without entering practice metadata.

### M3.7 — Interruption recovery

**Work:** On launch, inspect incomplete recording directories, open any finalized media, reconstruct minimal metadata when safe, and surface unrecoverable artifacts for explicit deletion.

**Exit:** Fixture/device scenarios for normal termination, interruption after finalization, and deliberately incomplete media yield accurate recoverability messages and never silently discard a source.

**M3 gate:** Review media queue confinement and lifecycle failure paths. Record a 30-minute target-device run showing bounded analysis queues, source playability, thermal state history, storage use, and capture-drop counters.

## M4 — Automatic post-stop analysis

Goal: every finalized recording produces editable attempt ranges without blocking or corrupting capture.

### M4.1 — Live observation ingestion

**Work:** Connect sampled capture frames to the pose adapter/cache using normalized source times. Persist coverage incrementally without running range extraction on partial unordered data.

**Exit:** Tests verify source-time agreement, bounded ingestion, cancellation, and recovery from cache write failure; a short device recording yields inspectable cached observations.

### M4.2 — Missing-coverage catch-up

**Work:** After finalization, compare intended schedule to cached coverage and fill missing samples through the offline reader. Deduplicate by source time and produce one ordered observation sequence.

**Exit:** Tests with forced live-analysis starvation yield the same ordered inputs and candidate output as offline-only analysis within documented timestamp tolerance.

### M4.3 — Analysis job coordinator

**Work:** Implement persisted queued/running/completed/failed/cancelled states, progress, resume after relaunch, and source/configuration invalidation. Serialize work per recording.

**Exit:** Tests cover stop-to-analysis transition, cancellation, relaunch during catch-up, retry, stale cache, deleted recording, and two recordings. No state reports completion before candidates are saved.

### M4.4 — Candidate adoption

**Work:** Convert detector output into stable attempt documents on the first completed analysis. Preserve source-relative ranges, method version, and configuration ID.

**Exit:** Tests verify deterministic initial IDs within a saved run, no partial publication, empty results, rerun stored separately, and no silent replacement of manual edits.

### M4.5 — Post-stop progress UI

**Work:** Connect capture Stop to finalization/analysis progress and then review. Provide retry, cancel, empty-result recovery, and explicit failure details.

**Exit:** UI tests cover fast completion, catch-up progress, empty result, cancellation, failure, and relaunch; the displayed count equals the committed candidate set.

### M4.6 — On-device latency characterization

**Owner:** Orchestrator/device tester.

**Work:** Measure finalization plus catch-up on five ten-minute target-device recordings, including one with forced live backlog. Save aggregate results and profiling notes.

**Exit:** p95 review-ready latency meets the design gate, or a reviewed optimization ticket is added with a measured bottleneck. No threshold is weakened to close the milestone.

**M4 gate:** Re-run the held-out detector suite through the integrated path and compare ranges to the offline evaluator. Review timestamp equivalence, cancellation, and honest progress states.

## M5 — Attempt review and repair

Goal: make detected attempts faster to inspect than raw video while preserving manual recovery.

### M5.1 — Attempt-bounded player

**Work:** Implement playback for a selected source range with exact stop-at-end and explicit loop behavior. Keep rate independent from export.

**Exit:** Synthetic-media tests verify start seek, end stop, repeated loops, last attempt, interrupted playback, and no escape into dead time beyond a small documented decoder tolerance.

### M5.2 — Attempt navigation

**Work:** Build index/count, previous/next, included-only navigation, first/last behavior, and selection persistence using `ReviewSession`.

**Exit:** UI/unit tests cover zero, one, multiple, excluded, restored, and deleted selections; stable IDs rather than indices drive selection.

### M5.3 — Scrubbing and frame stepping

**Work:** Add an attempt-local scrubber, 1x/0.5x/0.25x controls, and previous/next source-sample stepping. Show source-relative time without claiming frame-exactness when a sample is unavailable.

**Exit:** Tests verify scrubber mapping at range boundaries and rate reset rules; device inspection confirms forward/back stepping on 60 and 120 fps sources.

### M5.4 — Review zoom

**Work:** Add pinch/double-tap zoom with pan bounds and per-review presentation persistence. Keep source and export transforms untouched.

**Exit:** UI tests verify reset and attempt switching policy; a golden export before/after zoom is byte/visual-equivalent apart from normal encoder nondeterminism.

### M5.5 — Boundary editor

**Work:** Add handles that expose adjacent source context and save a revisioned override. Enforce valid source bounds and minimum sample-containing duration.

**Exit:** UI/domain tests cover both handles, cancellation, stale save conflict, first/last source bounds, and persistence after relaunch.

### M5.6 — Include, restore, add, and split

**Work:** Expose exclude/restore, full-source recovery timeline for adding a range, and split at the current playhead. Require an explicit save for edits.

**Exit:** UI tests cover false-positive removal, missed-attempt recovery, split validation, undo-before-save, empty included set, and reopening saved edits.

### M5.7 — Review accessibility and court usability

**Work:** Ensure large controls, VoiceOver labels/order, high-contrast states, portrait/landscape review, and one-handed navigation. Avoid precision controls that require tiny targets.

**Exit:** Accessibility audit has no unlabeled actionable controls; Dynamic Type does not hide core actions; a five-minute outdoor use test records issues and resolved blockers.

**M5 gate:** The user reviews a representative six-attempt session in about one minute, can repair one false and one missed detection, and confirms the app avoids raw-video searching for normal use. Record time and friction notes.

## M6 — Composition export and Photos

Goal: save one normal-speed, correctly oriented video containing included attempts in order.

### M6.1 — AV composition builder

**Work:** Convert an immutable pure export plan into an AV composition while preserving source timing/transform and concatenating ranges without artificial gaps.

**Exit:** Synthetic integration tests verify segment order, expected duration within one source sample, overlap unioning, orientation, silence, and normal playback duration.

### M6.2 — File exporter

**Work:** Export the composition to a temporary movie with progress, cancellation, collision-safe paths, and capability checks. Report any frame-rate fallback before starting.

**Exit:** Device tests cover 60 and 120 fps inputs, cancellation cleanup, disk exhaustion simulation where practical, retry, and no leaked temporary files. Media inspection records output nominal/average frame rate and duration.

### M6.3 — Add-only Photos writer

**Work:** Request add-only authorization at the moment of save and write a completed export. Return explicit authorized/denied/restricted/cancelled/failure outcomes.

**Exit:** UI/adapter tests cover every authorization result via fakes; physical-device test confirms a playable asset in Photos without read-library permission.

### M6.4 — Export coordinator and revision snapshot

**Work:** Snapshot the source/edit revision, prevent duplicate concurrent requests, coordinate plan/build/write, retain local data, and persist an export receipt.

**Exit:** Tests cover edit attempts during export, repeated intentional export, cancellation at each stage, Photos failure, stale revision, empty selection, relaunch, and successful retry.

### M6.5 — Export UI

**Work:** Add Save to Photos, progress/cancel, disclosed format fallback, completion, and actionable errors. Keep review usable after completion/failure.

**Exit:** UI tests cover all coordinator states; no error path removes the local recording or saved edits.

### M6.6 — Export fidelity matrix

**Owner:** Orchestrator/device tester.

**Work:** Inspect portrait/landscape, 60/120 fps, adjusted boundaries, exclusions, overlaps, and split attempts on the target phone and in Photos.

**Exit:** For every matrix row, order/orientation/playback speed/duration are correct and source action is retained. Any fallback is user-visible and documented.

**M6 gate:** Media/concurrency review confirms immutable export plans, no edit revision mixing, no source deletion, and accurate format claims.

## M7 — MVP1 release gate

Goal: decide from measured use whether MVP1 delivers immediate court value.

### M7.1 — Full regression and privacy audit

**Work:** Run all automated checks and private evaluation; inspect application permissions, network behavior, logs, artifacts, and deletion paths.

**Exit:** `mise run check` passes; detector gates still pass; app requests camera and add-only Photos only; no footage/network upload or sensitive paths appear in logs or committed files.

### M7.2 — Sustained device session

**Work:** Run a 30-minute representative practice, stop, analyze, review, repair one range, export, relaunch, and reopen.

**Exit:** No crash/source loss/unbounded backlog; latency and thermal/storage evidence are recorded; the Photos result remains correct and local state survives relaunch.

### M7.3 — User value trial

**Work:** Compare the app against raw Camera review for at least two sessions. Measure time from Stop to first useful clip, six-attempt review time, repair count/time, and subjective interruptions.

**Exit:** User confirms normal sessions avoid manual dead-time searching and exported compilations are useful. If not, file specific observed problems before authorizing MVP2 work.

### M7.4 — MVP1 release notes and known limits

**Work:** Update README status, supported setup/strata, measured performance, installation, privacy, and known failure cases. Tag the frozen detector/config and schema versions.

**Exit:** Claims trace to evidence, no unvalidated view/drill is presented as supported, and a clean checkout follows the documented setup.

**MVP1 release gate:** User accepts the court workflow; orchestrator signs off detector, media, privacy, and durability evidence. Release can remain personal/TestFlight; App Store work is separate.

## M8 — MVP2 annotation and feasibility

Goal: freeze what phase markers and visual tracks mean, and prove available methods can produce them before adding UI.

### M8.1 — Annotation rubric

**Owner:** Orchestrator plus user.

**Work:** Turn the design's loading, racket-drop, contact-candidate, and follow-through definitions into a visual guide with assessable/unassessable and uncertainty-interval rules. Define hitting-hand identity and racket box/handle/throat/tip points.

**Exit:** Both annotators independently label a pilot set, resolve disagreements, and update the guide until remaining ambiguity is explicitly representable.

### M8.2 — MVP2 annotation schema and validator

**Work:** Add versioned phase intervals, point/box landmarks, visibility, annotator, and adjudication fields to a separate local manifest/sidecar schema. Add synthetic examples.

**Exit:** Validator rejects points outside coordinates, phase times outside attempts, impossible uncertainty intervals, missing provenance, session leakage, and mixed source fingerprints.

### M8.3 — Held-out corpus preparation

**Owner:** User labels; orchestrator audits split.

**Work:** Assemble the minimum corpus from the design across separate sessions and visibility conditions. Reserve held-out sessions before method selection/tuning.

**Exit:** Counts and strata meet the floor or limitations are recorded; private footage remains ignored; held-out labels are sealed from tuning work.

### M8.4 — Hand-pose feasibility adapter

**Work:** Prototype Apple's hand-pose request on high-resolution hitting-hand crops derived from body tracks. Retain crop transforms and confidence; do not smooth into invented detections.

**Exit:** Evaluation reports coverage/error on development data, runtime per frame on target device, failure examples, and a recommendation against the frozen gate.

### M8.5 — Racket model discovery and licensing matrix

**Owner:** Strong-model research/review task.

**Work:** Compare a small set of on-device-capable candidates for racket boxes and landmarks. Record model source, weights license, redistribution/commercial terms, input/output, conversion path, binary/runtime cost, and evidence on sample footage.

**Exit:** Select one legally usable candidate for a bounded prototype, or issue a no-go report with the exact data/training work required. Repository dependencies remain unchanged until selection approval.

### M8.6 — Racket prototype and evaluation adapter

**Work:** Integrate the approved candidate in an isolated evaluation target, including crop/source transforms and unavailable results. No product UI.

**Exit:** Development report covers box/landmark accuracy, coverage, visibility failures, identity switches, device runtime/memory, and package size. Model/license attribution files are complete.

### M8.7 — Phase baseline experiment

**Work:** Build an offline baseline using body/hand/racket time series to emit optional uncertainty intervals for each phase anchor. Keep features and inference separate.

**Exit:** Development metrics and failure clips exist per phase; abbreviated motions legitimately omit inapplicable phases; contact candidates distinguish visual from kinematic provenance.

### M8.8 — MVP2 go/no-go review

**Owner:** Orchestrator plus user.

**Work:** Review usefulness of prototype overlays/anchors, development evidence, licensing, runtime, and remaining validation cost. Freeze selected versions and acceptance gates.

**Exit:** A written go decision names exact model/config/contracts and authorized dependencies, or a no-go decision stops M9 and identifies the smallest next experiment.

**M8 gate:** No worker begins model-dependent product code before this review. Tool additions for conversion/evaluation are pinned through mise after approval.

## M9 — MVP2 analysis pipeline

Goal: produce versioned, optional phase and tracking results on finalized attempts without destabilizing MVP1.

### M9.1 — Analysis domain types

**Work:** Add versioned optional phase anchors/intervals, observations, provenance, visibility, transforms, and stale status in `ServeCore/Analysis`.

**Exit:** Golden coding tests cover missing phases/points, intervals, versions, and source-time/coordinate invariants; MVP1 documents still decode unchanged.

### M9.2 — Analysis cache and invalidation

**Work:** Persist attempt analysis by source fingerprint, effective range, and body/hand/racket/phase versions. Mark affected results stale after boundary edits.

**Exit:** Tests cover exact hit, partial-version miss, boundary edit, source replacement, corrupt sidecar, cancelled replacement, and retention of the last valid result.

### M9.3 — Production hand tracker

**Work:** Harden the approved M8 hand adapter with batching/cancellation and source-coordinate outputs. Restrict to the hitting-hand region while preserving unavailable results.

**Exit:** Adapter passes frozen synthetic/integration tests and reproduces development metrics within tolerance; no wrist-angle interpretation is emitted.

### M9.4 — Production racket tracker

**Work:** Integrate the approved model, conversion artifacts, license notices, crop mapping, and confidence/visibility policy. Keep box and landmarks distinguishable.

**Exit:** Deterministic fixtures and device profiling pass frozen limits; missing/occluded racket points remain unavailable; package contents match approved licensing.

### M9.5 — Track association

**Work:** Associate body, hitting hand, and racket observations over an attempt with explicit gaps and identity-switch detection. Optional visualization interpolation is marked separately.

**Exit:** Synthetic sequences cover occlusion, crossing distractor, missing hand, lost racket, reappearance, and identity switch; raw observations remain accessible.

### M9.6 — Phase-anchor production implementation

**Work:** Implement the frozen M8 phase method over time series, emitting timestamp/interval, confidence, provenance, and availability.

**Exit:** Unit fixtures cover all anchors, absent contact, abbreviated motion, low visibility, boundary truncation, and determinism; no generic wrist-speed peak is labeled visual contact.

### M9.7 — MVP2 job coordinator

**Work:** Run analysis on demand per attempt/recording with progress, cancellation, resume, versioning, and retention of previous valid results until replacement completes.

**Exit:** Tests cover concurrent requests, playback during analysis, edit invalidation, cancel/retry, relaunch, model failure, and MVP1 export while MVP2 is unavailable.

### M9.8 — Held-out accuracy and performance gate

**Owner:** Orchestrator.

**Work:** Run frozen components once on sealed held-out sessions and profile the target device. Inspect false markers, misses, landmarks, visibility, and identity switches.

**Exit:** Per-phase and tracking gates from the design pass without threshold retuning, or M10 is blocked pending a separately reviewed method change.

**M9 gate:** Architecture review confirms all MVP2 results are optional/versioned, stale results cannot attach to edited ranges, and MVP1 review/export remain functional when models fail.

## M10 — MVP2 inspection experience

Goal: make relevant serve moments materially faster to reach and understand than raw scrubbing.

### M10.1 — Phase marker navigation

**Work:** Add labeled available anchors to the attempt scrubber and next/previous phase controls. Show uncertainty and provenance without overstating exactness.

**Exit:** UI tests cover all/partial/no markers, estimated contact, uncertainty intervals, stale analysis, and markers at range boundaries.

### M10.2 — Phase loop controls

**Work:** Create adjustable short loops centered on a selected anchor/interval, clamped to the effective attempt range.

**Exit:** Playback tests cover first/last anchor, wide uncertainty, missing phase, adjusted attempt bounds, and switching back to full-attempt playback.

### M10.3 — Body, hand, and racket overlays

**Work:** Render selectable overlays in source coordinates through rotation/zoom/letterboxing transforms. Distinguish raw, visual interpolation, low-confidence, and unavailable results.

**Exit:** Golden snapshot/geometry tests cover orientations and zoom; no line bridges unavailable gaps as if observed; overlays never appear in export.

### M10.4 — Analysis status and recovery UI

**Work:** Surface not analyzed, processing, available, partial, stale, failed, and unsupported states with analyze/retry actions. Ordinary playback stays available.

**Exit:** UI tests cover every state and relaunch; cancelling or failing analysis leaves MVP1 review and export intact.

### M10.5 — Inspection usability comparison

**Owner:** User plus orchestrator.

**Work:** On representative attempts, compare time to reach requested moments and confidence in inspection using raw scrubbing, MVP1, and MVP2. Record overlay obstruction/errors.

**Exit:** Phase navigation reduces median time to requested moments and the user judges markers/overlays useful. Any misleading failure is fixed or the affected output is disabled before release.

### M10.6 — MVP2 regression, privacy, and release notes

**Work:** Run all MVP1/MVP2 checks; audit models/licenses, permissions, network behavior, storage growth/deletion, schema compatibility, and supported claims. Update README.

**Exit:** MVP1 acceptance still passes; frozen MVP2 held-out gates pass; notices ship; no new permission/network use is introduced; MVP1 recordings open and export correctly after upgrade.

**MVP2 release gate:** User accepts the faster inspection workflow; orchestrator signs off evaluation, model provenance, on-device performance, backward compatibility, and absence of automated technique claims.

## Cross-cutting completion rules

A milestone cannot be closed with skipped checks. If hardware, private footage, or Xcode is unavailable, code work may be complete but the milestone remains pending its named evidence. Record blockers rather than replacing real tests with mocks.

New findings follow this path:

1. Reproduce and record the failing fixture or device evidence.
2. Decide whether the issue violates the current design or reveals a missing decision.
3. For a missing decision, update design/contracts and compatibility expectations before implementation.
4. Add one bounded remediation leaf with dependencies, allowed files, and exit criteria.
5. Re-run the affected milestone gate and all prior release gates that could regress.

Do not add future features opportunistically. Speed estimation, technique scoring, feedback timing, cloud sync, multiple players, generalized strokes, and real-time alerts each require a separate design and evidence gate after MVP2.
