# UX and workflow design

> **Status:** Current · **State:** Accepted · **Work:** None · **As of:** 2026-09-14

This document specifies the accepted command-line workflow and user-visible contracts for the implemented delivery-stage-2 compatibility workflow. It uses the [glossary](../reference/glossary.md) and the current [domain model](../architecture/domain-model.md).

## 0. Delivery stages

The delivery strategy is deliberately incremental:

1. **Design (complete):** specify the UX and contracts without changing runtime behavior.
2. **Compatibility workflow (current/accepted):** compose the existing cutting and stage-checkpoint services behind the new workflow, preserving their algorithms, caches, and artifacts.
3. **Tooling and naming unification (later):** consolidate commands, schemas, names, artifact invalidation, and compatibility aliases after the workflow has been exercised.
4. **Product polish (last):** improve presentation, packaging, progress reporting, and other refinements after the contracts and engineering foundations stabilize.

References to a delivery stage in this document mean the numbered stages above. Separate engineering projects—such as unified pose extraction, stage-checkpoint-informed detection, and the full review application—may be scheduled between or alongside stages 3 and 4, but are not part of stages 1 or 2.

This design does not authorize a new detector, unified pose representation, cache-format rewrite, or full review application. Those are separate engineering projects.

## 1. Goals

A user returning from practice should be able to point Serve Review at one source, several sources, or a directory and receive an understandable, resumable result without managing internal cache paths or running a command chain manually.

The workflow must support:

- one-off analysis of a standalone source;
- explicit analysis of a source range containing one attempt;
- ordered multi-source recording sessions;
- reusable collections and ad hoc comparisons;
- application-managed artifacts and cache reuse;
- clear preflight, progress, status, failure, and cleanup behavior;
- direct discovery and opening of review artifacts; and
- continued access to lower-level diagnostic commands without presenting them as the primary workflow.

## 2. Non-goals

This UX project does not include:

- changing serve-detection or stage-checkpoint heuristics;
- unifying sparse 2D and dense 3D inference;
- changing MediaPipe models or inference cadence;
- redesigning pose or kinematic cache formats;
- building the final session review page or stage gallery;
- adding browser-based annotation persistence;
- introducing a database before manifests prove insufficient;
- copying or managing source media as an asset library; or
- silently renaming current commands, schemas, types, or artifacts.

Delivery stage 2 may call existing commands or services and link their existing artifacts. It must not claim capabilities that require the deferred engineering projects.

## 3. UX principles

### One obvious entrypoint

The intended public syntax is:

```sh
serve-review COMMAND ...
```

The project is not packaged as an independently installed application yet. From a repository checkout, the canonical direct invocation of that same executable is:

```sh
uv run --locked serve-review COMMAND ...
```

Mise is **not** being abandoned. It remains the repository task runner for environment setup, diagnostics, tests, and convenient aliases such as `mise run cut -- ...`. Delivery stage 2 may add a `mise run process -- ...` alias. A mise task must delegate to the same CLI and must not define different behavior.

Documentation should not mix invocation styles within one workflow. User-facing target examples use `serve-review`; contributor instructions use either the direct `uv run --locked serve-review` form consistently or one documented mise alias.

### Sources are the unit of computation

A source is fingerprinted once. Moving a source or placing it in several sessions or collections must not duplicate compatible analysis. Sessions and collections organize references; they do not own caches or copied media.

### Sessions are optional

A standalone source does not require a synthetic one-source session. A session is created only when recording context or source ordering is meaningful.

### Safe repetition is the default

Repeating a command should inspect existing state, reuse compatible work, and continue incomplete work. It should not require `--overwrite`, create sibling output trees, or recompute expensive inference without explaining why.

### Uncertainty is visible

No-match, uncertain, rejected, incomplete, stale, and failed states must be represented explicitly. The workflow must never substitute an entire source when no attempt is detected.

### Generated data stays managed

The default workflow writes only beneath the managed workspace. User-selected exports are the exception. Source directories remain untouched.

## 4. Public command surface

The intended public commands are:

```text
serve-review process TARGET... [OPTIONS]
serve-review review [SELECTORS]
serve-review compare [SELECTORS]
serve-review session COMMAND ...
serve-review collection COMMAND ...
serve-review status [SELECTORS]
serve-review clean [SELECTORS] --dry-run
serve-review doctor
```

`process` is the primary all-in-one verb. It avoids colliding with the current legacy `analyze` command and does not imply that a user understands the internal detection and checkpoint-analysis split.

### Diagnostic namespace

The intended post-unification namespace for lower-level operations is:

```text
serve-review debug probe ...
serve-review debug extract-poses ...
serve-review debug export-ranges ...
serve-review debug analyze-attempt ...
serve-review debug review-checkpoints ...
```

During delivery stage 2, all current command names remain available. The compatibility workflow may call their Python services directly without implementing the `debug` namespace yet. Moving commands under `debug`, adding aliases, and deprecating old names belongs to delivery stage 3.

## 5. Target selection

`TARGET` may be:

- one MOV or MP4 path;
- multiple explicit media paths; or
- a directory, searched non-recursively by default.

Directory discovery must:

- include only documented media suffixes;
- use deterministic filename ordering;
- ignore generated workspace artifacts;
- report unsupported files rather than treating them as corrupt sources; and
- deduplicate sources by fingerprint after probing.

Recursive discovery requires explicit `--recursive`.

## 6. One-off workflows

### One source containing possible attempts

```sh
serve-review process practice.mov --open
```

The command fingerprints and registers the source, detects attempts, and invokes the existing `analyze-serve` service once for each detected attempt range. An attempt-level analysis failure is recorded without disguising it as success. The command then publishes a source-level review entrypoint, prints its path, and optionally opens it.

No session is created.

### One source known to contain one attempt

```sh
serve-review process serve.mov --single-attempt --open
```

`--single-attempt` treats the entire source as one explicit attempt and bypasses range detection. It must be mutually exclusive with `--range`.

### One explicit source range

```sh
serve-review process practice.mov --range 12.5:17.0 --open
```

The range uses canonical source seconds and half-open `[start, end)` semantics. It creates or reuses one explicit-range attempt without first running automatic range detection.

### Several unrelated sources

```sh
serve-review process first.mov second.mov third.mov --open
```

The sources are processed together for convenience, but no recording session is inferred. The command creates an unnamed, temporary review selection linking the source results. It is not listed as a session or collection, and a later unnamed selection may replace it. Any processing run provenance remains recorded independently. Reprocessing and cache identity remain source-based.

## 7. Recording-session workflows

### Create or update while processing

```sh
serve-review process ~/Imports/2026-09-14-practice \
  --session "2026-09-14 practice" \
  --open
```

If the named session does not exist, it is created. If it exists, newly discovered sources are appended in deterministic order after existing members; existing source membership is not duplicated. A source belongs to at most one recording session because a particular media asset has one recording context. Cross-session and thematic grouping belongs in collections. Adding a source that already belongs to another session fails with instructions for an explicit move.

A destructive replacement of session membership is never inferred from directory contents. Sources that disappeared from the directory remain session members but may be reported as unavailable at their known path.

### Explicit management

```sh
serve-review session list
serve-review session show "2026-09-14 practice"
serve-review session create "2026-09-14 practice"
serve-review session add "2026-09-14 practice" VIDEO...
serve-review session remove "2026-09-14 practice" --source SOURCE_ID
serve-review session tag "2026-09-14 practice" --tag drill:cocking
```

`session remove` removes membership only. It never deletes the source, analysis, cache, or source media.

Session display names need not be globally permanent identifiers. Commands should accept an unambiguous display name or stable `session_id` and report ambiguity instead of guessing.

## 8. Collections and comparisons

Collections are non-exclusive saved selections. They may contain sources, attempts, or a query over metadata.

```sh
serve-review collection create "Carlos Alcaraz reference serves"
serve-review collection add "Carlos Alcaraz reference serves" VIDEO...
serve-review collection show "Carlos Alcaraz reference serves"
serve-review collection list
serve-review collection delete "Carlos Alcaraz reference serves"
```

Deleting a collection removes only the selection manifest.

### Ad hoc comparison

```sh
serve-review compare first.mov second.mov --stage contact --open
serve-review compare --session "2026-09-14 practice" --source 3 --stage cocking
serve-review compare --collection "Carlos Alcaraz reference serves" --stage contact
```

Different selector types narrow the result (`AND`): for example, `--session practice --source 3 --stage contact` selects contact checkpoints from the third source in that session. Repeating the same selector type broadens that part of the query (`OR`): two `--session` options select from either session. Ambiguous names or unsupported combinations are rejected rather than guessed.

In delivery stage 2, `compare` produces a simple index from artifact paths present in workflow-run provenance. The renderer can embed explicitly supplied clips and checkpoint keyframes, but current workflow runs do not expose every cutting output to `compare`. Use the printed cutting artifacts directly when they are absent from the comparison index. A filterable stage gallery and synchronized player are deferred to the review-application project.

A comparison can be saved without duplicating artifacts:

```sh
serve-review compare --session "2026-09-14 practice" \
  --stage cocking \
  --save-as "September cocking review"
```

The saved object is a collection, not a session.

## 9. Managed workspace

### Default location

The workspace follows the XDG data-directory convention on every development platform:

```text
$XDG_DATA_HOME/serve-review/    # when XDG_DATA_HOME is set
~/.local/share/serve-review/   # otherwise
```

`SERVE_REVIEW_WORKSPACE` overrides that default globally, and `--workspace PATH` overrides both for one invocation. The resolved workspace path must be printed by `serve-review status` and included in diagnostics. The CLI does not use a macOS-specific Application Support default.

Delivery stage 2 uses the following ownership boundaries. Exact nested filenames may remain provisional because users discover artifacts through commands rather than by depending on private workspace paths:

```text
workspace/
  sources/       source registrations and source-scoped artifacts
  sessions/      recording-session manifests
  collections/   saved collection manifests
  runs/          processing-run provenance
  temporary/     replaceable unnamed review artifacts
  trash/         recoverable cleanup staging, if implemented
```

The layout is private implementation detail unless a file is documented as a versioned interchange contract. Commands that create or select artifacts must print the primary review page and relevant artifact paths; `--open` opens the primary page directly. Users therefore receive their artifacts as part of the command result rather than needing to inspect the workspace layout. `review` and `status` provide a safe way to find those paths again later.

### Source handling

By default, the workspace stores a fingerprint, metadata, and known path to the source; it does not copy source media. A missing path is reported as `source unavailable`, not as deleted analysis. Relocation and path-repair UX are deferred but must remain possible because identity is content-based.

### Compatibility artifacts

Existing `source.json`, `attempts.json`, `shadows.json`, `run.json`, pose caches, kinematic caches, clips, compilations, checkpoint documents, diagnostics, and review pages are stored beneath a source-scoped compatibility directory. Their internal schemas do not change in delivery stage 2.

## 10. Identity

### Source identity

`source_id` is an opaque stable identifier derived from or permanently bound to the full source fingerprint. It must not depend on basename, absolute path, directory order, session membership, or collection membership.

The complete fingerprint remains available for collision checking. User-facing output shows a short ID plus the current filename; manifests retain the full stable identity.

### Session and collection identity

`session_id` and `collection_id` are opaque persistent identifiers assigned at creation. Display names may change without changing identity.

### Attempt identity

An attempt belongs to exactly one source. Within delivery stage 2:

- repeating detection with the same source, detector identity, configuration, and resulting range must preserve `attempt_id`;
- moving or renaming the source must not change `attempt_id`;
- session or collection membership must not change `attempt_id`; and
- changing detector behavior may create a new attempt revision rather than silently attaching old annotations to a materially different range.

Cross-version reconciliation of shifted or split ranges requires an explicit design. Until then, the system must preserve old attempts, identify the producing run/configuration, and report that redetection produced a distinct revision. It must not use fuzzy matching silently.

The existing `serve-001` identifiers remain source-document-local compatibility IDs. In delivery stage 2, the workspace maps them to opaque stable IDs without changing `attempts.json`.

## 11. Processing plan and progress

Before expensive work, `process` prints a deterministic plan:

```text
Workspace: ~/.local/share/serve-review
Sources: 10 discovered, 9 new, 1 already registered
Probe: 1 reusable, 9 required
Range detection: 2 reusable, 8 required
Stage-checkpoint analysis: pending detection
Review index: will update
```

`--dry-run` performs discovery, identity lookup, compatibility checks that do not require expensive inference, and plan rendering. It creates no run or artifacts.

Progress is reported by source and pipeline step. Completion output always includes:

- accepted, uncertain, rejected, and failed counts where available;
- cache reuse and inference counts where available;
- the stable source/session/collection IDs involved;
- the review entrypoint; and
- the command to inspect status after interruption.

Compatibility wrappers must not fabricate unavailable counts from current schemas. They may report `unknown` with an explanation.

## 12. Resume, refresh, and collision semantics

### Default: resume and reuse

`process` defaults to safe resume:

- reuse compatible complete artifacts;
- resume supported incomplete caches;
- rerun missing downstream steps;
- preserve completed artifacts from prior runs; and
- create a new run record when actual processing is attempted.

A cache miss must include a reason such as `not found`, `source changed`, `model changed`, `configuration changed`, `schema changed`, `incomplete and not resumable`, or `corrupt`.

### Refresh

Explicit refresh options select work to recompute:

```sh
serve-review process VIDEO --refresh detection
serve-review process VIDEO --refresh checkpoints
serve-review process VIDEO --refresh review
```

Refreshing one step invalidates only dependent artifacts. In delivery stage 2, if the current services cannot perform narrow invalidation safely, the command must explain the broader recomputation before starting.

### Overwrite

The public workflow does not expose a general `--overwrite` switch. Replacement is scoped through `--refresh` or explicit export commands. Existing low-level `--overwrite` options remain diagnostic compatibility behavior.

### Failure and interruption

A failed run records its last pipeline step and actionable error. Published artifacts from earlier successful runs remain available. Temporary artifacts are cleaned according to current atomic-write contracts. Repeating the original command resumes where compatibility permits.

## 13. Status

```sh
serve-review status
serve-review status VIDEO
serve-review status --session "2026-09-14 practice"
serve-review status --collection "Carlos Alcaraz reference serves"
```

Status is read-only and reports:

- registered sources and known/missing paths;
- latest successful and failed runs;
- current detector and checkpoint artifacts;
- stale or incompatible artifacts;
- cache hit eligibility and incompatibility reasons;
- attempt counts and classifications available from current schemas;
- session/collection membership; and
- review entrypoints.

Human-readable output is the default. `--json` emits a versioned machine-readable status document.

## 14. Cleanup

Cleanup operates only on application-managed derived artifacts:

```sh
serve-review clean --dry-run
serve-review clean --source SOURCE_ID --dry-run
serve-review clean --stale --dry-run
serve-review clean --failed-runs --dry-run
```

`--dry-run` is mandatory unless `--confirm` is supplied. Cleanup must distinguish:

- recomputable caches;
- generated review media and pages;
- failed-run temporary data;
- stale compatibility artifacts;
- manifests and provenance;
- user annotations; and
- source media.

Source media and user annotations are never cleanup targets. Session and collection manifests require their own explicit delete commands. Delivery stage 2 implements cleanup reporting and `--dry-run` only. Destructive cleanup with `--confirm` is deferred until retention rules have been tested.

## 15. Review behavior and discovery

Every successful `process`, `review`, or `compare` command prints one primary review entrypoint. `--open` opens it with the operating system; omission never opens a browser unexpectedly.

```sh
serve-review review VIDEO --open
serve-review review --session "2026-09-14 practice" --open
serve-review review --collection "September cocking review" --open
```

### Compatibility review

Delivery stage 2 generates a deliberately minimal static landing page that:

- identifies the selected sources and attempts;
- embeds clips or compilations supplied explicitly to the renderer with basic HTML video controls;
- displays supplied checkpoint JPEGs inline;
- links existing per-attempt review pages;
- links every other generated artifact, including JSON manifests and diagnostics, for manual inspection;
- identifies missing, failed, or stale analysis; and
- provides the exact command needed to produce missing artifacts.

If a browser cannot play a source codec, the page still provides a direct media link and explains that browser-compatible proxies are deferred. It need not implement interactive filtering, annotations, synchronized playback, or media proxies.

### Deferred polished review

A separate review-application project will define the canonical review manifest and implement:

- serve-by-serve playback;
- eight-checkpoint timelines;
- stage galleries across attempts;
- source/session/collection/tag filtering;
- browser-compatible media proxies;
- comparison layouts; and
- optional persisted human decisions.

The compatibility landing page should link artifacts through stable IDs or manifests where practical so it can later be replaced without changing source analysis.

## 16. Error and exit behavior

Errors name the source and pipeline step, explain whether prior artifacts remain usable, and suggest one next action. Batch processing isolates source failures where safe and summarizes them at the end.

Recommended exit codes are:

```text
0  requested work completed, including an honest no-attempt result
1  processing or external-tool failure
2  invalid command, selector, target, range, or configuration
3  partial batch success with one or more isolated source failures
```

Delivery stage 2 adopts these exit codes for the new public workflow. Existing lower-level compatibility commands retain their current exit behavior until delivery stage 3. Machine-readable status must never report a failed source as successful.

## 17. Delivery stage 2: compatibility workflow

Delivery stage 2 is a thin orchestration layer over existing Python services.

### Allowed work

- add workspace and manifest domain objects;
- register sources by existing probe fingerprint;
- add `process`, `status`, session, and collection command shells;
- invoke existing `run_cut` and `run_analyze_serve` services;
- select explicit ranges and existing attempt ranges;
- route current output directories beneath the workspace;
- record compatibility artifact paths and run status;
- create a minimal static review landing page; and
- preserve current lower-level commands.

### Explicitly deferred work

- pose-cache or kinematic-cache redesign;
- sparse/dense inference unification;
- stage-checkpoint-informed range validation;
- detector threshold changes;
- renamed persisted schemas;
- attempt fuzzy reconciliation across detector versions;
- final review manifest and application;
- browser media-proxy pipeline; and
- performance optimization beyond avoiding known duplicate work.

### Compatibility boundaries

The orchestrator must call Python services directly rather than shelling out to its own CLI. Existing FFmpeg/ffprobe subprocess boundaries remain unchanged. It may translate service results into workspace manifests but must not reinterpret detector confidence or stage-checkpoint evidence.

## 18. Later unification and polish

### Tooling and naming project

After the compatibility workflow is proven:

- decide whether `process` remains the primary verb;
- move diagnostics under `debug` with compatibility aliases;
- execute the future renames listed in the glossary;
- version unified run/status schemas;
- remove duplicate output-directory concepts; and
- make artifact invalidation dependency-aware.

### Engineering projects

Separately design and evaluate:

- shared sparse/dense 2D/3D pose extraction and cache coverage;
- stage-coherent attempt validation;
- canonical review manifest and session review application;
- observability, retention, and garbage collection; and
- broader corpus/viewpoint support.

### Polish project

Only after the contracts and engineering foundations stabilize:

- concise progress display;
- completion-time estimates;
- richer browser UX;
- source relocation assistance;
- interactive tagging and annotations;
- packaging and installation refinement; and
- accessibility and visual design.

## 19. Acceptance criteria for delivery stage 1

Delivery stage 1 is complete when:

- one-off, explicit-range, session, collection, and comparison workflows are specified;
- the public and diagnostic command boundaries are explicit;
- workspace ownership and source-preservation rules are explicit;
- identity, resume, refresh, status, cleanup, and review-discovery semantics are explicit;
- delivery stage 2 is separated from deferred engineering work;
- terminology matches the glossary; and
- open disagreements found during review are resolved or recorded below.

## 20. Decisions resolved during review

1. `process` remains the primary verb. An `analyze` alias may be added after the legacy command retires, but no rename is planned.
2. A source belongs to zero or one recording session. Collections provide all non-exclusive and cross-session grouping.
3. The workspace uses the XDG data-directory convention, with explicit environment and CLI overrides; it does not use a macOS-specific default.
4. An unnamed multi-source review selection is temporary and replaceable. It does not become a session or collection unless explicitly saved. Processing run provenance remains independent.
5. The delivery-stage-2 landing page is minimal HTML: inline available images and video, direct links to all other generated artifacts, and visible missing/failure states. Rich interaction and media proxies remain deferred.

No unresolved decision blocks delivery stage 2.
