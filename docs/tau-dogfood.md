# Tau dogfood journal

Append one entry per attempt to the **app's** `docs/tau-dogfood.md` (or its existing
dogfood journal), before calling the task complete. Preserve failures, pauses,
no-edit outcomes, repairs and unavailable measurements. Archive the exact task
under the app's `dogfood/tasks/` only after checking it for private material.
Keep local `.tau` state and raw Pi sessions ignored; they may contain secrets or
private reasoning. The app's roadmap should point to the run and remaining work.

### 2026-09-09 — M1.1 Media domain schemas

- **Identity:** run `9caa1fad-e7b3-4deb-9a95-c91caef22e19`; Tau revision unavailable; base `de469e6fc442`; captured candidate `3fe6a16b561c`; adopted `40540f728c0b` (rebased onto the original foundation during graph repair); task stored outside the app.
- **Environment:** macOS/Apple Silicon; JJ 0.45.1; mise Python 3.11.16; uv 0.12.9; model `opencode/muse-spark-1.3-contributor-free`.
- **Allocation/result:** 75 turns / 20 minutes allowed; observed 12 turns; verified.
- **Evidence:** worker changed only `src/serve_review/domain.py` and `tests/test_domain.py`; exact diff review found no dependency or private-media changes; verifier `uv run pytest tests/test_domain.py && mise run check` passed; stored candidate binding valid with empty successor.
- **Review/integration:** approved by orchestrator; candidate duplicated into an independent adopted revision; integrated `mise run check` passed with 65 tests. Human acceptance unavailable.
- **Cleanup:** initial `tau release` returned `blocked-active-work` with `topology-violation`. The empty workspace successor was reparented to Tau's recorded captured candidate, then `tau release` succeeded. The detached rewritten candidate left by the earlier integration mistake had no descendants and was abandoned after release. `jj workspace list` now contains only the default workspace. See `docs/tau-feedback/m1.1-cleanup-blocked.md`.
- **Cost/attention:** measured inference cost unavailable; one integration-command targeting error was caught because the first integrated check collected only four tests, then corrected before acceptance.
- **Finding:** JJ duplicate creates a sibling revision; target the emitted duplicate revision, not `@-`, when describing/moving a bookmark. Its descendant rewrite also blocked Tau cleanup; restoring the workspace successor to Tau's recorded candidate enabled safe release. Next action: dispatch M1.2 from the current `main` and use a topology-preserving adoption flow.

### 2026-09-09 — M1.2 ffprobe adapter and probe command (rejected parent)

- **Identity:** run `2788cd7f-cea8-4914-97e9-6af38c75ef34`; Tau revision unavailable; base `202f1034`; captured candidate `fe71af158949`; repair task stored outside the app.
- **Environment:** macOS/Apple Silicon; JJ 0.45.1; mise Python 3.11.16; uv 0.12.9; model `opencode/muse-spark-1.3-contributor-free`.
- **Allocation/result:** 75 turns / 20 minutes allowed; observed 21 turns; verified, then rejected.
- **Evidence:** candidate changed only the five allowed M1.2 files; stored verifier passed and binding was valid. Exact review found that rotation parsing accepts direct/tag fields but does not parse ffprobe MOV Display Matrix data in `side_data_list`.
- **Review/lifecycle:** rejected by orchestrator. `tau repair` was attempted with the narrow rotation task but returned `cannot repair: parent current candidate is not inspectable`; no child was created. Tau `reject` then safely disposed the candidate and run-owned workspace. Integration/human acceptance unavailable.
- **Cleanup:** released by `tau reject`.
- **Cost/attention:** unavailable.
- **Finding:** a verified candidate can have a meaningful media-metadata gap despite broad unit coverage; repair must test ffprobe's actual alternate metadata shape. Repair was unavailable for this pre-upgrade parent record, so the safe fallback was reject then a fresh retry.

### 2026-09-09 — M1.2 ffprobe adapter and probe command (retry)

- **Identity:** run `99203706-0a25-4975-a8b3-9954f5d4243e`; Tau revision unavailable; base `fcfa9af2`; captured candidate `5b3edc2d8b24`; task stored outside the app.
- **Environment:** macOS/Apple Silicon; JJ 0.45.1; mise Python 3.11.16; uv 0.12.9; model `opencode/muse-spark-1.3-contributor-free`.
- **Allocation/result:** 75 turns / 20 minutes allowed; observed 28 turns; verified.
- **Evidence:** candidate changed only the five allowed M1.2 files; stored verifier and binding passed. Exact review confirmed argument-array ffprobe invocation, rational metadata parsing, atomic source JSON writing, and explicit direct/tag/Display-Matrix rotation precedence. Tests cover positive/negative/malformed/unsupported/conflicting `side_data_list` values.
- **Review/lifecycle:** approved by orchestrator; `tau accept --target main` accepted candidate `4bb98f3bbe28`. The local empty working-copy child was rebased onto accepted `main` before the integrated retest. Human acceptance unavailable.
- **Cleanup:** accepted worker workspace released; `jj workspace list` contains only the default workspace.
- **Cost/attention:** unavailable.
- **Finding:** requiring `integration_verification` and using Tau acceptance avoided manual candidate topology manipulation. Tau moved `main` correctly with the approval-journal commit as its ancestor; the caller still must rebase its empty default working-copy child to `main` before local commands test the accepted tree.

### 2026-09-09 — M1.3 synthetic media fixture generator

- **Identity:** run `0356c0d0-15f5-4cfb-81e8-0e2c3bb7fbcb`; base `419b17c8`; captured candidate `98e88fc47427`.
- **Environment/result:** macOS/Apple Silicon; JJ 0.45.1; Python 3.11.16; model `opencode/muse-spark-1.3-contributor-free`; 14 turns; verified.
- **Evidence/review:** exact two-file diff only, no generated binary or dependency change; deterministic bounded FFmpeg fixture factory and real temporary portrait/landscape tests; stored verifier/binding passed; approved, acceptance pending.
- **Finding:** generated tiny fixtures give M1.4 media integration tests a private-footage-free foundation.

### 2026-09-11 — M4.1 Kovacs eight-stage schema

- **Identity:** accepted run `73efa0fb-71b0-4f75-b953-db3b20a9e3da`; base `a920ed2c`; captured candidate `28e751d61a2a`; integrated `e769cb6b98d6`; model `opencode/muse-spark-1.3-contributor-free`; 35 turns.
- **Evidence/review:** exactly three allowed files changed. Candidate verification passed 720 tests. Exact review confirmed canonical eight-stage names; immutable stage/attempt/source documents; strict availability, provenance, range, ordering, status, source, and version validation; audio contact requires an anchor plus uncertainty; direct mutation of the stage mapping is rejected.
- **Lifecycle:** approved and accepted through Tau; worker workspace released; default working copy rebased onto `main`. Per owner instruction, no redundant post-acceptance full-suite run was performed.
- **Intervention:** first candidate (`c0d3b280-e8dc-4524-95d3-e5b4c875058e`) was rejected for missing audio-anchor and top-level source-document constraints; repair preparation failed before binding (`64c8ff43-976a-45ae-8208-73ad5fdc1425`). A corrected candidate (`98763a68-35aa-474f-ae2d-d1df2f9344c9`) passed 720 tests but Tau acceptance verification timed out twice and left no integration. A second corrected candidate (`fbd11111-d256-4909-af17-ed8e146cecac`) was rejected because its frozen dataclass exposed a mutable stage dictionary. The accepted retry added a read-only mapping and direct regression coverage.
- **Cost/attention:** three full candidate runs were required; measured inference cost unavailable. Human acceptance unavailable.
- **Finding:** schema immutability must cover nested containers, not only frozen dataclass attributes; focused integration verification avoids repeating an already-passed full candidate suite during acceptance.

### 2026-09-11 — M4.2 attempt-local phase features

- **Identity:** accepted run `2591024a-7455-4fd5-8b71-137b46e7bd6f`; base `0ecbda71`; captured candidate `b93210a00c2e`; integrated `a38e4ab88563`; model `opencode/muse-spark-1.3-contributor-free`; 17 turns.
- **Evidence/review:** exactly three allowed files changed. Candidate verification passed 725 tests. Exact review confirmed strict half-open attempt slicing, physical-duration window configuration, uniform grid metadata, visibility-gated normalized camera-relative geometry, short-gap-only reduced-quality interpolation, timestamp-aware derivatives, audio alignment, and synthetic cadence/gap/boundary/precision-honesty coverage.
- **Lifecycle:** approved and accepted through Tau; worker workspace released; default working copy rebased onto `main`. Per owner instruction, acceptance ran only focused M4.2 verification rather than another full suite.
- **Cost/attention:** measured inference cost unavailable; human acceptance unavailable.
- **Finding:** local-polynomial derivatives and explicit observed/interpolated quality channels satisfy cadence invariance without requiring a higher-rate pose-cache architecture; actual denser inference remains conditional on the M4 gate.

### 2026-09-11 — M4.2 anchor smoke / alignment repair attempts

- **Smoke evidence:** ignored anchor `refs/anchors/single-serve-01.mov`, using a matching existing 145-frame pose cache, built a 145-sample 30 Hz grid. Raw audio peak was 3.655 s, 14 ms before the known ~3.669 s contact; the candidate grid time was 3.667 s. This validates source-time/audio alignment.
- **Finding:** only 4 grid samples were marked observed and 141 interpolated. The pose cadence measured 29.979 Hz while the configured grid is 30 Hz; strict exact-time matching accumulated small drift and misclassified genuine observations. This is an M4.2 repair requirement before M4.3 evidence gating—not an inference-quality conclusion.
- **Tau operational outcomes:** `ce3fe3cd-3e8d-49fc-948b-ab674c099151` failed before worker start because the dispatch base hash was mistyped; `c0eba7a4-56c5-40f4-85f0-288175a20fcb` exhausted 45 turns without candidate/readiness. Session JSONL inspection shows `1bc50aa4-e99d-4fa3-bda0-bd078623ceb2` and `e1266c98-01b7-4e84-b5dd-459c7fb9bcb2` were provider `429 FreeUsageLimitError` failures, misreported by Tau as missing handoff; see `docs/tau-feedback/2026-09-11-free-model-429-misclassified.md`. No candidate was captured, reviewed, or integrated; unsafe resumes were not attempted.
- **Next action:** make a fresh, bounded M4.2 alignment repair candidate with a direct-observation snap tolerance and explicit support-offset/uncertainty semantics; do not proceed to M4.3 until it is verified and reviewed.

### 2026-09-11 — M4.2 alignment and physical-angle repairs

- **Identity:** accepted alignment repair `ec74444b-3383-4ba5-8896-cbf67ba76258`, candidate `3748cb89f25c`, integrated `436e6f48f5b1`; accepted angle-bounds repair `cf4d4edb-e266-430c-b439-dc461c4510cb`, candidate `0e7706fce1bf`, integrated `673b710597b4`. Both used `opencode/muse-spark-1.3-contributor-free` after its free quota recovered.
- **Evidence/review:** alignment repair changes only the three permitted M4.2 files; it introduces cadence-capped one-to-one direct-observation matching, support offsets, and conservative temporal uncertainty with 29.979 Hz regression coverage. The two-file angle repair clamps only post-smoothing emitted physical elbow/knee angles and leaves schema validation and pre-clamp derivative fits strict. Both candidates passed their focused and full verification and were reviewed/accepted with focused acceptance verification.
- **Hardware smoke:** the ignored single-serve anchor now builds a 145-sample 30 Hz grid from 145 29.979 Hz pose observations with `145 observed`, `0 interpolated`, `0 missing`. Raw audio peak is 3.655 s, 14 ms before the known ~3.669 s contact; the nearest candidate grid time is 3.667 s. A local-polynomial elbow overshoot (180.055 degrees) discovered by the first repair smoke is bounded by the second repair.
- **Finding:** private-video smoke immediately after a feature-schema change is necessary: synthetic cadence tests found the mask issue, while real pose geometry exposed the bounded-angle overshoot. Both are numerical representation defects, not detector tuning results.

### 2026-09-11 — M4.3 stage evidence and candidate generation

- **Identity:** accepted run `00055168-4215-428d-878f-60b9918627ab`; base `95b7fb64`; candidate `4cf47da91212`; integrated `8d1b1be6f873`; model `opencode/muse-spark-1.3-contributor-free`; 23 turns.
- **Evidence/review:** exactly two allowed files changed. Candidate verification passed. M4.3 provides immutable bounded unary evidence/candidate sets for all eight canonical stages, with broad versioned windows, quality/missingness handling, honest body-pose camera-relative language, and no joint solver or macro-export changes. Contact is restricted to audio-candidate samples, uses audio-transient provenance and uncertainty, and normalizes only among flagged transient energies; synthetic native-scale 0.005/0.053 RMS coverage prevents the earlier absolute-amplitude failure.
- **Intervention:** initial run `306c7a22-c58a-475f-a97c-136910fb41db` was rejected during exact review because `audio_energy / 2.0` plus the candidate floor discarded real-scale RMS peaks. Its repair preparation `d61e15b6-cbf9-463a-af72-46473707ef8d` failed before a candidate due to the immutable-workspace repair limitation. The parent was safely rejected and a corrected full leaf was dispatched from `main`.
- **Lifecycle:** reviewed and accepted through Tau with focused acceptance verification; worker workspace released; default working copy rebased. Human acceptance unavailable.
- **Finding:** an audio-candidate flag reflects an upstream transient decision; candidate-stage scoring must be amplitude-scale-invariant rather than reapplying an arbitrary raw RMS threshold.

### 2026-09-11 — M4.4 constrained joint phase solver

- **Identity:** accepted run `4c684400-913c-457a-9065-cd84a1096d7e`; base `c3c63e2b`; candidate `b32db753fd88`; integrated `1d84e958b852`; model `opencode/muse-spark-1.3-contributor-free`; 50 turns.
- **Evidence/review:** exactly two allowed files changed and candidate verification passed. The solver performs deterministic candidate-or-skip DP over M4.3 unary evidence, with broad span-aware chronology bounds, explicit skip penalties, deterministic ties, source/range/method validation, and honest M4.1 unavailable outputs. Hand-calculated tests cover global-over-local selection, skips, bounds, ties, truncation, mismatch rejection, and no macro side effects.
- **Contact semantics:** selected audio contact is retained through missing/occluded/incompatible body support. Existing-grid wrist elevation and camera-relative motion support can upgrade provenance to `body_pose_audio`; otherwise it remains `audio_transient` with lowered confidence and a stable advisory anomaly. This is not an M3 export filter.
- **Lifecycle:** reviewed and accepted through Tau with focused acceptance verification; worker workspace released; default working copy rebased. Human acceptance unavailable.
- **Finding:** DP solves the expected local-extrema problem: a lower unary candidate can win when it makes the entire phase path chronologically feasible, while unsupported stages remain unavailable rather than being invented.

### 2026-09-11 — M4.5 analyze command and phase report

- **Identity:** accepted run `45e93d07-87a1-4002-b305-59c64c022ef7`; base `21e56544`; candidate `de1dd754ce83`; integrated `304544f3cb0a`; model `opencode/muse-spark-1.3-contributor-free`; 33 turns.
- **Evidence/review:** exactly four allowed files changed and candidate verification passed. `analyze` validates source-bound attempts, requires complete matching pose cache identity, demuxes raw audio once, max-pools to cached timestamps, reuses decoder relative transient policy, runs M4.2–M4.4 over unpadded ranges, isolates per-attempt failures, and atomically emits source-bound `checkpoints.json`. Tests cover explicit/discovered attempts, cache failures, audio policy, partial/empty outcomes, cancellation/collisions, determinism, and unchanged M3 artifacts.
- **Lifecycle:** reviewed and accepted through Tau with focused acceptance verification; worker workspace released; default working copy rebased. Human acceptance unavailable.
- **Finding:** phase analysis can be added without destabilizing serve cutting by treating the existing pose cache and attempts document as immutable inputs and phase output as a separate atomic artifact.

### 2026-09-11 — M4.6 phase evaluation and manual-gate support

- **Identity:** accepted run `3ccf4265-c1ac-423a-90e2-6b20469b0148`; base `530e24d3`; candidate `4262fe5189f7`; integrated `f5d3d8e455ca`; model `opencode/muse-spark-1.3-contributor-free`; 21 turns.
- **Evidence/review:** exactly four allowed files changed and candidate verification passed. Strict versioned annotations are source-relative and session-disjoint; they support available, unavailable, and ambiguous stage labels. Evaluation reports per-stage availability, accepted keyframes, interval IoU, signed/absolute timing and percentiles, order violations, confidence calibration, structural completeness, and optional serve-versus-aborted separation. Synthetic tests cover hand-calculated cases, leakage/path/version/conflict rejection, deterministic reports, and input immutability.
- **Lifecycle:** reviewed and accepted through Tau with focused acceptance verification; worker workspace released; default working copy rebased. Human acceptance unavailable.
- **Finding:** phase evaluation must compare keyframes against manually accepted uncertainty intervals, not pretend every body-only stage has an exact racket/ball ground-truth frame.

### 2026-09-11 — labeled phase-review renderer

- **Identity:** accepted run `a013079d-8dae-484d-b68e-00c63d2e4810`; base `0b684cfd`; candidate `6540a66fe304`; integrated `aedb6861`; model `opencode/muse-spark-1.3-contributor-free`; 32 turns.
- **Evidence/review:** four allowed files changed and candidate verification passed. `review-phases` validates source-bound phase output and a complete matching pose cache, samples raw upright source at selected keyframes, overlays existing landmarks, and burns stage/time/confidence/provenance/anomaly caption text into JPEGs. It gives unavailable phases manifest rows instead of inventing images, labels contact as an estimate rather than visual observation, and publishes its review directory atomically. Tests cover source/cache failure, bounded pose support, cancellation/collision, determinism, and unchanged inputs.
- **Development execution:** cut, analyze, and review succeeded locally on a private single-serve development exemplar. Five selected frames rendered; three stages were honestly unavailable. Local source paths, frame labels, and output paths remain ignored.
- **Lifecycle:** reviewed and accepted through Tau with focused acceptance verification; worker workspace released; default working copy rebased. Human acceptance unavailable.

### 2026-09-11 — M4.6 optional manual-confidence repair and failed anchor gate

- **Identity:** accepted repair `db6b077d-8d30-4541-a075-9ca1fdd3ebff`; candidate `915cc29342f6`; integrated `f025a824`; model `opencode/muse-spark-1.3-contributor-free`; 16 turns.
- **Repair:** real manual labels exposed that the annotation codec incorrectly required a manual confidence field. The field now decodes as `null` when omitted, while supplied values retain strict validation; machine-confidence calibration semantics are unchanged. Tests cover absent/null equivalence and unchanged calibration.
- **Development gate:** user labels were converted from 0-based source frames to an ignored exact-decoded-PTS manifest. Evaluation gave 0/5 selected keyframes accepted, three manually available stages unavailable, and only contact close (about 40 ms). Pre-contact errors are hundreds of milliseconds to >1.5 s, so this is a heuristic/chronology failure rather than evidence for M4.7 cadence work. No configuration changed.

### 2026-09-12 — multi-source phase review renderer dispatch

- **Identity:** interrupted run `a82fe44f-ec35-4f38-974d-a11bd38f7d34`; base `6c48242c`; model `opencode/muse-spark-1.3-contributor-free`; task `/tmp/tennis-coach-review-annotations-task.json`.
- **Environment:** local macOS Pi/Tau harness; workspace adapter `jj`; measurements unavailable.
- **Allocation/result:** 75 turns / 1200-second budget; 0 turns observed; paused after caller interruption.
- **Evidence:** worker session stopped after initial file reads; no candidate, handoff, verification, or binding was stored; no app files changed by the worker.
- **Review/lifecycle:** no review or acceptance; lifecycle paused, not integrated. Human acceptance and downstream survival unavailable.
- **Cleanup:** workspace/state retained for diagnosis; no retry performed.
- **Cost/attention:** measured totals unavailable; intervention was interrupting a stale run after confirming no worker/controller process remained.
- **Finding:** Tau remained in `running` with a stale 0-turn snapshot and no handoff after the foreground controller timed out. Next action is a fresh bounded dispatch after diagnosing the stalled startup; do not infer completion from status.
