# Delivery stage 2 Tau dispatch plan

> **Status:** Current · **State:** Accepted · **Work:** Active · **As of:** 2026-09-14

This plan decomposes delivery stage 2 of the accepted [UX and workflow design](ux-workflow-design.md) into small, sequential Tau tasks for `opencode/muse-spark-1.3-contributor-free`. It specifies tasks for review; it does not dispatch them.

The goal is a thin compatibility workflow over existing services. No task may change detection, stage-checkpoint scoring, pose inference, cache formats, media encoding, or existing command behavior.

## 1. Dispatch policy

### Immutable sequential bases

Before the first dispatch, commit the accepted UX design and this reviewed plan. Each task starts from the accepted result of its predecessor. Do not dispatch these tasks concurrently: later tasks intentionally build on earlier schemas and stores.

For every task:

- use the explicit model `opencode/muse-spark-1.3-contributor-free`;
- use JJ isolation from one committed immutable base;
- fallback to open-codex/gpt-5.6-luna if opencode/muse-spark-1.3-contributor-free is down;
- archive the sanitized task JSON under `dogfood/tasks/` only after review;
- record every outcome in `docs/development/tau-dogfood.md`; and
- use Tau's `accept`, `repair`, or `reject` lifecycle rather than manually moving candidate changes.

### Common worker contract

Every task JSON must state:

```text
Read only the named context pack before editing. Do not read the deferred
roadmap, archive, research notes, private refs, or unrelated implementation.
Do not add dependencies. Do not modify existing detector, checkpoint, pose,
cache, media, or export behavior. Keep current commands backward compatible.
Use pure/injected collaborators in tests; no private footage, model inference,
network, or real FFmpeg is required. Modify only Allowed files. If an allowed
file or requirement is insufficient, call tau_yield with needs_rescue rather
than broadening scope. Run only the focused worker check. Finish with tau_yield
reporting summary, files_changed, checks, assumptions, deviations, and risks.
```

### Verification contract

The worker runs only the ticket's focused check. Tau independently verifies every candidate with:

```json
{
  "verification": {
    "command": "mise run check",
    "cwd": ".",
    "timeout_ms": 300000,
    "output_limit": 6000
  },
  "integration_verification": {
    "command": "mise run check",
    "cwd": ".",
    "timeout_ms": 300000,
    "output_limit": 6000
  },
  "timeout_ms": 2700000,
  "max_turns": 75
}
```

A passing verifier is evidence, not approval; inspect the exact diff against each ticket's review criteria before acceptance.

### Shared implementation shape

Delivery stage 2 adds a narrow package without reorganizing current modules:

```text
src/serve_review/workflow/
  __init__.py
  workspace.py       workspace paths and atomic manifest I/O
  records.py         workflow-facing versioned records and stable IDs
  sources.py         source registration
  sessions.py        recording-session storage
  collections.py     explicit collection storage
  planning.py        target discovery and dry-run plans
  processing.py      compatibility orchestration
  landing.py         minimal static HTML
  status.py          read-only status projection
  cleanup.py         cleanup inventory/dry-run only
  cli.py             new public command handlers/parsers
```

Tests mirror these modules as `tests/test_workflow_*.py`. This shape keeps most changes out of the existing large `cli.py`; DS2.8 makes one bounded integration edit there.

## 2. DS2.1 — Workspace paths and atomic manifest I/O

**Dependency:** accepted documentation base only.

**Outcome:** Add a small workflow package that resolves the managed workspace and provides atomic JSON file primitives. No domain records or CLI changes.

**Allowed files:**

- `src/serve_review/workflow/__init__.py` (new)
- `src/serve_review/workflow/workspace.py` (new)
- `tests/test_workflow_workspace.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §§3 and 9
3. `docs/reference/glossary.md` entries for source, run, cache, artifact, and review
4. `src/serve_review/pipeline.py` helpers `_write_text_atomic` and `_remove_quietly` only, as behavioral examples; do not edit that file
5. atomic-write tests located by `rg -n "atomic|tmp-|os.replace" tests/test_pipeline.py tests/test_pose_cache.py`

**Required behavior:**

- `resolve_workspace(explicit=None, environ=None, home=None)` precedence is explicit path, `SERVE_REVIEW_WORKSPACE`, `$XDG_DATA_HOME/serve-review`, then `~/.local/share/serve-review`.
- Empty environment overrides are rejected, not silently ignored.
- Resolution performs no filesystem mutation.
- `WorkspacePaths` exposes `root`, `sources`, `sessions`, `collections`, `runs`, `temporary`, and `trash`.
- `ensure_workspace` creates only those directories and is idempotent.
- UTF-8 JSON writes use a sibling temporary file, flush, and `os.replace`; failure removes the temporary file.
- JSON read errors identify the path and never return partial/default data.
- No wall-clock timestamps, global mutable state, database, locking, or dependencies.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_workspace.py
```

**Review criteria:** Exact precedence, no mutation during resolution, no shell calls, actionable errors, temporary cleanup, and no copied atomic-I/O refactor in existing modules.

## 3. DS2.2 — Workflow records and source registration

**Dependency:** DS2.1 accepted.

**Outcome:** Add strict versioned workflow records, stable ID helpers, and source registration using existing source probing. Do not process media.

**Allowed files:**

- `src/serve_review/workflow/__init__.py`
- `src/serve_review/workflow/records.py` (new)
- `src/serve_review/workflow/sources.py` (new)
- `tests/test_workflow_records.py` (new)
- `tests/test_workflow_sources.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/architecture/domain-model.md` through “Identity and ownership rules”
3. `docs/plans/ux-workflow-design.md` §§9–10
4. `docs/reference/glossary.md` entries for source, session, collection, run, and artifact
5. `src/serve_review/domain.py` `SourceMetadata` codec only
6. `src/serve_review/media/probe.py` `fingerprint_for_path` and `probe_source` signatures only
7. `src/serve_review/workflow/workspace.py`

**Required behavior:**

- Add strict, deterministic JSON codecs for `SourceRecord` and shared reference/ID value objects only; reject unknown keys and unsupported schema versions.
- `source_id` is deterministic from the complete `sha256:<hex>` fingerprint and independent of path/name.
- `attempt_id` helper is deterministic for exact source identity, producer method/config identity, and exact unpadded range. It performs no fuzzy matching.
- `register_source` accepts injectable probe/fingerprint collaborators, records the full fingerprint and normalized `SourceMetadata`, and stores a source record beneath `sources/<source_id>/`.
- Re-registering the same bytes at another path adds a known path without duplicating the source.
- Detect and reject a short-ID collision against a different full fingerprint.
- Registration stores references only; it never copies or modifies source media.
- Tests use fake source bytes and injected metadata; no ffprobe subprocess.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_records.py tests/test_workflow_sources.py
```

**Review criteria:** Stable IDs do not depend on path/order, schema codecs are strict, known paths are deterministic/deduplicated, and registration cannot touch media content.

## 4. DS2.3 — Recording-session store

**Dependency:** DS2.2 accepted.

**Outcome:** Implement recording-session manifests and storage with zero-or-one session membership per source. No CLI.

**Allowed files:**

- `src/serve_review/workflow/records.py`
- `src/serve_review/workflow/sessions.py` (new)
- `tests/test_workflow_sessions.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/architecture/domain-model.md` Session, Source, and identity rules
3. `docs/plans/ux-workflow-design.md` §7 and decision 2 in §20
4. `docs/reference/glossary.md` Session and Collection
5. `src/serve_review/workflow/workspace.py`
6. `src/serve_review/workflow/records.py`
7. `src/serve_review/workflow/sources.py`

**Required behavior:**

- Strict versioned `SessionRecord` with opaque persistent `session_id`, mutable display name, ordered unique source IDs, and deterministic string tags.
- Inject ID creation in tests; do not use a wall clock.
- Create, list, load by ID, resolve an unambiguous display name, rename, tag, add, remove, and delete membership manifest operations.
- A source belongs to zero or one session. Adding a member of another session fails and identifies both sessions; no implicit move.
- Removing/deleting session membership never deletes source records or source artifacts.
- Missing source IDs, duplicate/ambiguous names, malformed manifests, and duplicate additions have explicit behavior and tests.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_sessions.py
```

**Review criteria:** Ordering is stable, session uniqueness is enforced by the store, no source deletion path exists, and no collection/query behavior leaks into this task.

## 5. DS2.4 — Explicit collection store

**Dependency:** DS2.3 accepted.

**Outcome:** Implement non-exclusive saved collections with explicit source and attempt references. Query-backed collections remain deferred. No CLI.

**Allowed files:**

- `src/serve_review/workflow/records.py`
- `src/serve_review/workflow/collections.py` (new)
- `tests/test_workflow_collections.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/architecture/domain-model.md` Collection and identity rules
3. `docs/plans/ux-workflow-design.md` §8 and decision 4 in §20
4. `docs/reference/glossary.md` Collection, Corpus, Source, and Attempt
5. workflow records/source/session modules from accepted base

**Required behavior:**

- Strict versioned `CollectionRecord` with opaque persistent ID, display name, ordered unique source IDs, ordered unique attempt IDs, and deterministic tags.
- Create, list, load/resolve, rename, tag, add/remove references, and delete manifest operations.
- Membership is non-exclusive across collections and independent of session membership.
- Deleting a collection removes only its manifest.
- Validate referenced source IDs; attempt references are opaque until processing records exist, but duplicates and malformed IDs are rejected.
- Do not implement query persistence, comparison rendering, source processing, or source copying.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_collections.py
```

**Review criteria:** Collections do not enforce session-like exclusivity, deletion cannot reach source/attempt artifacts, and scope stays explicit-membership only.

## 6. DS2.5 — Target discovery and process planning

**Dependency:** DS2.4 accepted.

**Outcome:** Add pure/injected target discovery and a read-only `ProcessPlan` suitable for `--dry-run`. No inference, artifact writes, HTML, or CLI.

**Allowed files:**

- `src/serve_review/workflow/records.py`
- `src/serve_review/workflow/planning.py` (new)
- `tests/test_workflow_planning.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §§5–6, 9, and 11
3. `docs/reference/glossary.md` Source, Range, Attempt, Run, Cache, and Artifact
4. accepted workflow workspace/record/source modules
5. `src/serve_review/domain.py` `MediaRange` only
6. CLI argument-validation tests found with `rg -n "range|recursive|missing input" tests/test_cli.py`

**Required behavior:**

- Discover explicit files or immediate directory children; recursion requires an explicit flag.
- Support documented MOV/MP4 suffixes case-insensitively, deterministic path order, and fingerprint-based deduplication through injected source lookup.
- Unsupported directory entries are reported separately; an explicit unsupported target is an input error.
- Validate `--single-attempt` versus one explicit half-open range as mutually exclusive and require exactly one source for either mode.
- `ProcessPlan` reports discovered/new/registered sources and whether probe, detection, checkpoint analysis, and landing-page work are required/reusable/unknown based only on injected state inspection.
- Rendering the human plan is deterministic.
- Dry planning performs no workspace writes and invokes no processing service.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_planning.py
```

**Review criteria:** No side effects, deterministic discovery, no stdin/speculative modes, no cache internals guessed, and `unknown` is used when compatibility artifacts cannot prove reuse.

## 7. DS2.6 — One-source compatibility processing service

**Dependency:** DS2.5 accepted.

**Outcome:** Implement one-source orchestration over injected/existing `run_cut` and `run_analyze_serve` services, with source-scoped paths, stable workflow attempt mappings, safe reuse, and run provenance. This is the highest-risk service task and must remain single-source.

**Allowed files:**

- `src/serve_review/workflow/records.py`
- `src/serve_review/workflow/processing.py` (new)
- `tests/test_workflow_processing.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §§6, 9–12, 16, and 17
3. `docs/architecture/domain-model.md` Attempt, Run, identity, and deferred decisions
4. accepted workflow modules
5. `src/serve_review/pipeline.py` `CutResult`, `CutError`, and `run_cut` signature/return paths only
6. `src/serve_review/analyze_serve.py` `AnalyzeServeResult`, errors, and `run_analyze_serve` signature/return paths only
7. `src/serve_review/domain.py` `AttemptDocument`, `Attempt`, and `MediaRange` codecs only
8. fake-orchestration patterns in `tests/test_pipeline.py` and `tests/test_analyze_serve.py`, located with `rg`; do not read media integration tests

**Required behavior:**

- Accept exactly one registered source plus normal-detection, whole-source single-attempt, or explicit-range mode.
- Call Python services directly; never invoke the project's CLI as a subprocess.
- Normal mode calls `run_cut` with source-scoped compatibility output and `mode="both"`; explicit/single mode bypasses `run_cut`.
- Analyze each accepted detected range with a separate attempt-scoped output directory and kinematic cache path so hard-coded compatibility `serve-001` output cannot collide across attempts.
- Map each compatibility attempt to the stable workflow `attempt_id` helper without modifying current `attempts.json` or checkpoint schemas.
- Reuse only when existing artifacts parse successfully and match source fingerprint, exact range, producer/config identity available from current artifacts, and required files. Otherwise report an explicit stale/collision/refresh-required reason; do not pass broad `overwrite=True` silently.
- Record strict versioned `WorkflowRunRecord` status and per-pipeline-step/per-attempt outcomes atomically.
- Isolate checkpoint-analysis failure per attempt and return partial status; a detection failure fails the source.
- Preserve honest empty detection with no synthetic attempt.
- Tests inject all heavy services and create only tiny fake JSON/files.
- Do not generate HTML or implement multi-source looping.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_processing.py
```

**Review criteria:** Existing services are unmodified, output/cache paths cannot collide across attempts, reuse validation is conservative, no silent overwrite/fuzzy matching, empty/partial/failure states are honest, and run provenance is sufficient for later status projection.

## 8. DS2.7 — Minimal landing-page renderer

**Dependency:** DS2.6 accepted.

**Outcome:** Add a deterministic, static, minimal HTML renderer over explicit artifact inputs. No selection lookup or processing.

**Allowed files:**

- `src/serve_review/workflow/records.py`
- `src/serve_review/workflow/landing.py` (new)
- `tests/test_workflow_landing.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §15 and decision 5 in §20
3. `docs/reference/glossary.md` Artifact, Clip, Compilation, Review, Checkpoint, and Keyframe
4. accepted workflow records/workspace modules
5. HTML escaping/atomic directory patterns located with `rg -n "html.escape|index.html|os.replace" src/serve_review/analyze_serve.py src/serve_review/phase_review.py tests/test_phase_review.py`

**Required behavior:**

- Renderer receives explicit source/attempt/artifact records; it does not scan arbitrary directories or infer semantic status from filenames.
- Escape all user/path/status text and percent-encode local links appropriately.
- Embed available clips/compilations with basic `<video controls>` elements.
- Display available checkpoint JPEGs inline with stage, checkpoint time, and attempt labels.
- Link per-attempt review pages and every supplied generated artifact, including JSON and diagnostics.
- Show missing/stale/failed states and supplied remediation commands as text.
- Include a browser-codec warning and direct media links; do not transcode or proxy.
- Deterministic ordering and output; write atomically.
- No JavaScript, CSS framework, server, annotation UI, filtering, synchronization, or dependency.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_landing.py
```

**Review criteria:** HTML injection is prevented, all supplied artifacts remain discoverable, inline media degrades to links, output is deterministic, and renderer has no workspace/process side effects.

## 9. DS2.8 — Public `process` command and batch compatibility workflow

**Dependency:** DS2.7 accepted.

**Outcome:** Wire the new package into one `process` command supporting one-off, explicit-range, directory/batch, optional session association, dry-run, minimal landing page, and documented exit codes. Preserve every existing command.

**Allowed files:**

- `src/serve_review/cli.py` (one import/hook plus any unavoidable top-level dispatch integration only)
- `src/serve_review/workflow/cli.py` (new)
- `src/serve_review/workflow/processing.py`
- `src/serve_review/workflow/landing.py`
- `tests/test_workflow_cli_process.py` (new)
- `mise.toml` (optional `process` alias only)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §§3–7, 9, 11–12, 15–17
3. accepted workflow modules and their focused tests
4. `src/serve_review/cli.py` `build_parser`, `main`, `cut`, and `analyze_serve` only
5. parser/handler fake patterns in the first relevant tests located by `rg -n "build_parser|monkeypatch.*run_" tests/test_cli.py`
6. `mise.toml`

**Required behavior:**

- Add `process TARGET...` with `--workspace`, `--recursive`, `--session`, `--single-attempt`, `--range START:END`, `--padding`, `--open`, and `--dry-run`.
- Keep all existing parser forms and handlers unchanged.
- Batch sources in deterministic order; isolate source failures where safe.
- Exit 0 for complete/honest-empty, 1 for processing failure, 2 for invalid input, and 3 for partial batch success.
- Create/update an explicitly named session only after source registration; enforce one-session membership.
- Unnamed multi-source landing output goes under replaceable `temporary/`, not sessions, collections, or run identity.
- Print the resolved workspace, concise plan/progress/result summary, primary landing-page path, relevant artifact paths, and status recovery command.
- `--open` uses an injected platform opener after successful publication; omission never opens anything.
- `--dry-run` creates no workspace/run/artifact and calls no processing service.
- Add `mise run process -- ...` only as a direct alias to `uv run --locked serve-review process "$@"`; no alternate behavior.
- Heavy services and opener are faked in tests.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_cli_process.py tests/test_cli.py
```

**Review criteria:** Existing CLI tests remain untouched/passing, integration edit in `cli.py` is narrow, no subprocess self-invocation, dry-run is truly read-only, batch exits are exact, and output always identifies the landing page/artifacts.

## 10. DS2.9 — Session and collection public commands

**Dependency:** DS2.8 accepted.

**Outcome:** Expose the already-tested session and explicit-collection stores through public CLI subcommands. Do not add query-backed collections or processing behavior.

**Allowed files:**

- `src/serve_review/workflow/cli.py`
- `tests/test_workflow_cli_organization.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §§7–8
3. accepted session/collection stores and tests
4. accepted workflow CLI process parser/handler patterns

**Required behavior:**

- Add the specified `session list/show/create/add/remove/tag` commands.
- Add `collection list/show/create/add/remove/tag/delete` commands; rename may be included only if already exposed cleanly by the store.
- Accept stable IDs or unambiguous names; ambiguity exits 2 and lists matching IDs.
- Path arguments are registered as sources but are not automatically processed by organizational commands.
- Mutating commands print stable IDs and affected manifest paths.
- Removing session membership or deleting collections never removes source records/artifacts.
- No HTML, comparisons, query language, fuzzy names, or source moving.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_cli_organization.py tests/test_workflow_sessions.py tests/test_workflow_collections.py
```

**Review criteria:** Commands are thin store adapters, destructive scope is manifest-only, name ambiguity is safe, and no processing side effect is hidden in `add`.

## 11. DS2.10 — Read-only status projection and command

**Dependency:** DS2.9 accepted.

**Outcome:** Implement human and versioned JSON status over existing workflow records and compatibility artifacts. Status performs no repair or processing.

**Allowed files:**

- `src/serve_review/workflow/records.py`
- `src/serve_review/workflow/status.py` (new)
- `src/serve_review/workflow/cli.py`
- `tests/test_workflow_status.py` (new)
- `tests/test_workflow_cli_status.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §§9, 11–13, and 16
3. accepted workflow records/stores/processing modules
4. current artifact codecs named by `rg -n "class (AttemptDocument|PhaseDocument)|run.json" src/serve_review/domain.py src/serve_review/pipeline.py src/serve_review/analyze_serve.py`
5. accepted workflow CLI tests

**Required behavior:**

- Status selectors: whole workspace, source path/ID, session, or collection.
- Report known/missing source paths, latest workflow run outcome, compatibility detection/checkpoint artifact presence, stale/corrupt/unavailable reasons proven by current records, cache eligibility when provable, attempt counts/classifications available from current schemas, memberships, and landing-page paths.
- Unknown facts are `unknown` with a reason; do not infer accepted/uncertain/rejected classifications absent from current schemas.
- Human output is deterministic and concise.
- `--json` is a strict versioned status document with deterministic serialization.
- Status creates no directories, writes, cache quarantine, refresh, processing, or browser open.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_status.py tests/test_workflow_cli_status.py
```

**Review criteria:** Entire path is read-only, missing/corrupt/stale differ, unknown remains honest, JSON is strict/versioned, and source selectors cannot escape the workspace registry.

## 12. DS2.11 — `review`, `compare`, and temporary selections

**Dependency:** DS2.10 accepted.

**Outcome:** Resolve existing artifacts into landing pages for standalone sources, sessions, explicit collections, and ad hoc comparisons. No new analysis and no polished review application.

**Allowed files:**

- `src/serve_review/workflow/landing.py`
- `src/serve_review/workflow/cli.py`
- `tests/test_workflow_cli_review.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §§8, 13, and 15 plus decisions 4–5 in §20
3. accepted workflow stores, status projection, landing renderer, and CLI tests
4. `docs/reference/glossary.md` Review, Collection, Stage, Checkpoint, and Keyframe

**Required behavior:**

- `review` resolves one source, session, or collection and republishes/prints a landing page from existing records.
- `compare` accepts explicit sources and/or session/collection selectors plus `--source`, `--stage`, `--stages`, `--open`, and `--save-as` within the simple semantics already specified.
- Different selector types use `AND`; repeated values of one type use `OR`.
- Unnamed selections publish beneath replaceable `temporary/` and create no collection.
- `--save-as` writes an explicit collection manifest; it does not copy/reprocess artifacts.
- Commands never trigger processing. Missing artifacts appear in HTML with the exact `process` command needed.
- `--open` behavior matches `process` and is injected in tests.
- No JavaScript filters, synchronized playback, annotations, media proxying, fuzzy matching, or query-backed saved collection.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_cli_review.py tests/test_workflow_landing.py
```

**Review criteria:** Read-only unless explicitly saving a collection/landing page, temporary selection does not clutter manifests/runs, selector semantics are tested, and absent analysis is linked to remediation rather than silently computed.

## 13. DS2.12 — Cleanup inventory and dry-run command

**Dependency:** DS2.11 accepted.

**Outcome:** Add cleanup reporting only. No deletion implementation.

**Allowed files:**

- `src/serve_review/workflow/cleanup.py` (new)
- `src/serve_review/workflow/cli.py`
- `tests/test_workflow_cleanup.py` (new)
- `tests/test_workflow_cli_cleanup.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. `docs/plans/ux-workflow-design.md` §14 and decision 4 in §20
3. accepted workspace, source, run/status, and CLI modules
4. `docs/reference/glossary.md` Cache, Artifact, Run, Session, and Collection

**Required behavior:**

- `clean --dry-run` inventories only application-managed derived artifacts.
- Support selectors for source, stale compatibility artifacts, failed runs, and replaceable temporary review artifacts.
- Classify candidates as recomputable cache, generated review/media, failed-run artifact, stale artifact, manifest/provenance, user annotation, or source media.
- Only the first four classes may be reported as future deletion candidates.
- Source media, user annotations, source/session/collection manifests, and required run provenance are protected and clearly shown as excluded.
- Reject `clean` without `--dry-run`; do not expose `--confirm` until deletion exists.
- No unlink, rename-to-trash, mutation, quarantine, processing, or artifact repair.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_cleanup.py tests/test_workflow_cli_cleanup.py
```

**Review criteria:** Search remains under resolved workspace roots, symlinks/path traversal cannot escape inventory boundaries, no deletion primitive exists, and protected classes cannot be marked deletable.

## 14. DS2.13 — Current documentation and acceptance pass

**Dependency:** DS2.12 accepted.

**Outcome:** Document only the implemented delivery-stage-2 behavior, mark accepted design status appropriately, and add focused end-to-end fake tests for the public workflow. No production behavior change except tiny defects required to make documented existing behavior true; request rescue for anything larger.

**Allowed files:**

- `README.md`
- `docs/README.md`
- `docs/plans/ux-workflow-design.md`
- `docs/plans/delivery-stage-2-tau-plan.md`
- `docs/architecture/domain-model.md`
- `docs/reference/cli.md`
- `docs/reference/workspace-and-artifacts.md`
- `docs/reference/schemas.md`
- `docs/reference/configuration.md`
- `docs/user-guide/getting-started.md`
- `docs/user-guide/sessions-and-collections.md`
- `docs/user-guide/reviewing-serves.md`
- `docs/user-guide/troubleshooting.md`
- `tests/test_workflow_acceptance.py` (new)

**Context pack, in order:**

1. `AGENT.md`
2. accepted UX design and domain model
3. all accepted `tests/test_workflow_*.py` test names using `rg -n "^def test_"`; read only tests needed to verify claims
4. `src/serve_review/workflow/cli.py` public parser definitions and handler signatures
5. current documentation status/glossary

**Required behavior:**

- Replace relevant placeholders with concise current user/reference documentation.
- Clearly distinguish installed syntax, repository `uv run --locked` syntax, and optional mise aliases.
- Document exact stage-2 limitations and deferred engineering work.
- Update statuses: implemented references/guides become Current/Maintained; completed plan becomes Historical/Complete only after acceptance.
- Acceptance tests use injected services and temporary workspaces to cover one-off dry-run, one source, partial batch exit 3, named session, temporary comparison, saved collection, status JSON, and cleanup dry-run.
- No private media, model, network, or real FFmpeg.
- Do not rename commands/types/artifacts or revise algorithms.

**Focused worker check:**

```sh
uv run --locked pytest tests/test_workflow_acceptance.py && git diff --check
```

**Review criteria:** Documentation matches parser/service behavior exactly, no placeholder remains for implemented stage-2 contracts, limitations are explicit, tests cover public composition rather than internals, and no runtime refactor is hidden in the documentation ticket.

## 15. Integration gates

After every accepted task, the orchestrator reviews:

1. exact allowed-file compliance;
2. focused test quality and failure-path coverage;
3. absence of dependency, inference, detector, cache-format, and media changes;
4. Tau verifier binding to the exact candidate;
5. integrated `mise run check` result from `tau accept`;
6. durable dogfood record and safe workspace lifecycle; and
7. whether the next task's context pack still matches the integrated code.

Delivery stage 2 is complete only after DS2.13 is accepted and a manual CLI smoke test confirms, with fake or disposable source data, that command output directly identifies/open the landing page and relevant artifacts. Private-footage quality or detector accuracy is not an acceptance gate for this compatibility stage.
