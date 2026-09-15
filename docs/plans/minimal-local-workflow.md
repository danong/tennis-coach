# Minimal local processing workflow

> **Status:** Proposed · **State:** Draft · **Work:** Next · **As of:** 2026-09-14

## 1. Purpose

Serve Review has one user, who is also its developer. The primary workflow should optimize for that fact rather than model a multi-user media library or a future installed product.

The useful job is:

1. Record one or more tennis-practice videos.
2. Copy them into a deliberately named directory on the development Mac.
3. Run one command against that directory.
4. Follow one printed link to review compilations and detected attempts.

The workflow is a small coordinator over the existing serve-cutting and stage-checkpoint analysis tools. It should add one useful HTML summary, straightforward progress, and simple continuation after interruption. It should not introduce a second domain model around the existing pipeline.

## 2. User interface

The canonical user-facing invocation is:

```sh
mise run process -- ~/videos/tennis/2026-09-10
```

A single source may also be selected:

```sh
mise run process -- ~/videos/tennis/2026-09-10/court-1.mov
```

Force regeneration with either spelling:

```sh
mise run process -- TARGET --force
mise run process -- TARGET --rebuild
```

`--force` is the canonical spelling. `--rebuild` is an exact alias. Supplying both has the same meaning as supplying either one.

User documentation uses Mise commands. Internal and partial pipeline commands remain available to the developer through `uv run --locked serve-review ...`, but they are not presented as a separately installed application.

A self-documenting entrypoint exposes the complete command inventory:

```sh
mise run help
```

It should show the primary `process` workflow first, followed by the internal cutting, analysis, evaluation, and diagnostic commands with their repository invocation syntax.

## 3. Input directory

The input directory is the recording session. Its name and parent hierarchy are chosen by the user at import time. Serve Review does not create, name, register, move, or catalog sessions.

Directory processing discovers immediate regular-file children with `.mov` or `.mp4` suffixes, case-insensitively, in deterministic filename order. It is not recursive. `metadata/`, `exports/`, nested directories, and unsupported files are ignored.

A single-video target uses its parent as the recording directory. Generated artifacts therefore remain with related raw videos regardless of whether processing starts from the video or its directory.

The workflow fails during preflight if two discovered videos have the same filename stem. The user resolves the ambiguity by renaming a source. There is no generated disambiguation ID.

Source videos are never modified, moved, or deleted.

## 4. Owned output

For input such as:

```text
~/videos/tennis/2026-09-10/
  court-1.mov
  court-2.mov
```

Serve Review owns two plainly named child directories:

```text
~/videos/tennis/2026-09-10/
  court-1.mov
  court-2.mov

  metadata/
    index.html                         # process: recording-directory summary
    court-1/                           # run_cut: exact metadata destination
      source.json                      # run_cut probe: source metadata/fingerprint
      cache/
        pose-v1.jsonl                  # run_cut detection: sparse pose cache
      attempts.json                    # run_cut planning: accepted detected/export ranges
      shadows.json                     # run_cut decoder: rejected/incomplete hypotheses
      run.json                         # run_cut: current cut result and diagnostics
      attempts/                        # process: accepted-attempt analysis grouping
        serve-001/                     # run_analyze_serve: exact output destination
          cache/
            kinematic-track-v1.jsonl   # analyze: dense world-landmark cache
          checkpoints.json             # analyze: selected stage checkpoints
          serve-3d-diagnostics.json    # analyze: scoring/solver diagnostics
          review-serve-3d/             # analyze: static checkpoint review
            index.html
            review.json
            start.jpg
            release.jpg
            loading.jpg
            cocking.jpg
            acceleration.jpg
            contact.jpg
            deceleration.jpg
            finish.jpg
        serve-002/
          ...
      manual-analysis/                 # direct analyze-serve default, not process output
        ...
    court-2/
      ...

  exports/
    court-1/                           # run_cut: exact media destination
      serves.mov                       # padded gap-free compilation
    court-2/
      serves.mov
```

The comments identify the owner and pipeline stage for each file. Exact existing diagnostic filenames beneath an attempt may remain as produced by the analyzer. The stable boundaries are:

- `metadata/index.html` is the process summary;
- `metadata/STEM/` is the exact cut metadata/cache destination;
- `metadata/STEM/attempts/serve-NNN/` is the exact automated analysis destination;
- `metadata/STEM/manual-analysis/` is the replaceable default for direct range analysis; and
- `exports/STEM/serves.mov` is the only retained cut video.

No individual serve clips are retained. `metadata/STEM/run.json` is the cut command's current diagnostic result, not workflow run history. No global application workspace is involved. There are no source, session, collection, attempt, or workflow-run identities beyond the source filename and existing source-local `serve-NNN` labels.

## 5. Producer destination contract and processing behavior

The replacement intentionally changes the current producer destination behavior. Existing `output/` defaults and the convention of silently appending the source stem to a caller-provided analysis directory are not preserved.

`run_cut` derives these defaults from the source path:

```text
metadata destination = VIDEO.parent / "metadata" / VIDEO.stem
media destination    = VIDEO.parent / "exports" / VIDEO.stem
```

Its source metadata, detection documents, diagnostic result, and sparse pose cache go to the metadata destination. Requested compilation/clips go to the media destination. `process` requests compilation only. Explicit exact metadata and media destinations remain injectable Python options for tests and unusual internal use; ordinary CLI use requires no output-directory arguments.

`run_analyze_serve` treats a supplied output directory as the exact destination and no longer appends the source stem. With no explicit destination, direct range analysis writes to:

```text
VIDEO.parent / "metadata" / VIDEO.stem / "manual-analysis"
```

Its default kinematic cache is `cache/kinematic-track-v1.jsonl` beneath that exact destination. The process coordinator supplies `metadata/STEM/attempts/serve-NNN/` as the exact destination for detected attempts.

Accordingly, the direct partial commands remain simple:

```sh
uv run --locked serve-review cut VIDEO --padding 1 --output compilation
uv run --locked serve-review analyze-serve VIDEO --start-seconds S --end-seconds E
```

They produce the same local directory schema as `process`. Explicit output overrides, if retained in the internal CLI, mean exact destinations and are not part of the primary workflow.

The implementation does not stage work in XDG storage, copy generated media or caches, create symlinks, rewrite producer manifests, or migrate old outputs. The old `output/` tree and removed global Stage-2 workspace are unrelated generated data and may be deleted manually after replacement acceptance.

For each selected source, `process` performs the following sequence using direct Python service calls:

1. Probe and fingerprint the source using the existing media rules.
2. Detect accepted serve-attempt ranges across the source.
3. Export one gap-free compilation with one second of padding around each accepted attempt.
4. Run the existing one-attempt stage-checkpoint analyzer for every accepted detected range.
5. Generate the recording-directory summary page.

The compilation uses padded export ranges. Stage-checkpoint analysis uses each attempt's original unpadded detected range in canonical source seconds.

An honest no-attempt result is successful. Its summary entry says that no serves were detected and has no compilation or attempt links. Rejected, incomplete, and shadow hypotheses are not analyzed as accepted attempts.

Processing one attempt must use an attempt-specific output directory so analyzer filenames cannot collide with another attempt.

An individual source or attempt failure does not prevent later independent sources or attempts from running. Failures appear both in terminal output and on the summary page.

## 6. Continuation and force

Existing artifacts are the current state. The replacement workflow adds no state database, global manifests, historical run records, invalidation graph, or new `state.json` schema.

Normal processing follows three rules:

1. If a unit's small required output set is complete, skip it.
2. If a unit is incomplete, remove only that unit's generated output and run that unit again.
3. If existing `source.json` identifies different source bytes at the same source location, fail and instruct the user to use `--force`.

A cut unit is complete when the existing cut result reports successful detection, its source and attempts documents are readable, and the expected compilation exists when accepted attempts exist. An honest-empty successful cut does not require a compilation.

An attempt-analysis unit is complete when its checkpoints, diagnostics, review document, and review HTML are readable. Individual JPEG presence follows the analyzer's recorded stage availability; the coordinator does not invent missing stages.

A directory existing by itself never proves completion. If an interruption leaves only some required files, the next invocation clears that source's incomplete cut unit or that attempt's incomplete analysis unit and retries it. Completed sibling attempts remain untouched.

`--force` and `--rebuild` remove the selected generated units before processing:

- for a video target, only that stem's metadata and compilation are removed;
- for a directory target, generated metadata and exports for all discovered source stems are removed.

Raw media is outside every deletion boundary.

The workflow does not automatically invalidate output after an algorithm change. While the pipeline is evolving, the user explicitly requests `--force` when new results are wanted.

## 7. Progress and completion

Progress is concise and stage-level. It does not claim frame-level percentages or completion-time estimates that the existing services cannot provide.

```text
[1/2 videos] court-1.mov
  detecting serves…
  found 7 serves
  exporting compilation…
  [1/7 attempts] analyzing 00:12.4–00:16.8…
```

Native model-runtime diagnostics should not obscure normal progress. Unexpected warnings remain available in an explicit verbose/debug mode or diagnostic log if suppression at the dependency boundary is practical.

At the end, print a compact result and one clickable absolute file URL:

```text
Complete: 2 videos, 13 attempts
Failed: 1 attempt
Review: file:///Users/me/videos/tennis/2026-09-10/metadata/index.html
```

The summary is regenerated on every invocation, including invocations where all computation is skipped. A truncated or manually damaged page is therefore repaired by running `process` again; `--force` is also acceptable when the user wants to start over. No special transactional HTML machinery is required.

Exit status is deliberately small:

- `0`: all selected work completed, including honest no-attempt results;
- `1`: processing finished with one or more source or attempt failures;
- `2`: invalid command, target, source collision, or configuration.

The summary path is printed whenever a summary can be generated, including after processing failures.

## 8. Summary page

`metadata/index.html` is a static page with relative links so the recording directory can be moved as a unit. It uses source filenames, not opaque identifiers.

For each source it shows:

- source filename and a link to the raw video;
- complete, empty, incomplete, or failed status;
- accepted attempt count;
- a link to the compilation when present;
- each `serve-NNN` label and exact detected range;
- a link to that attempt's existing stage-checkpoint review HTML; and
- actionable failure text when processing did not complete.

A representative section is:

```text
court-1.mov — 7 serves — complete
[Source] [Compilation]

serve-001 — 00:12.4–00:16.8 — [Checkpoint review]
serve-002 — 00:31.1–00:35.7 — [Checkpoint review]
serve-003 — failed: source sampling returned 7 of 8 requested frames
```

The page does not implement comparison, stage filtering, synchronized playback, annotation, JavaScript persistence, media proxies, or an asset browser. Those features require a separate demonstrated need.

## 9. Explicit removals and non-goals

The replacement removes rather than extends the generalized workflow. It does not retain:

- a global managed workspace;
- source registration or opaque workflow source IDs;
- recording-session manifests;
- collection manifests;
- selectors;
- `review` republishing;
- `compare` or stage-filtered comparison;
- generalized status projection;
- cleanup inventory or trash staging;
- temporary selections;
- workflow run provenance or run history;
- generalized schema/version frameworks for workflow records;
- recursive discovery or multiple unrelated target roots;
- installed-CLI documentation; or
- cross-platform behavior beyond the development Mac.

Existing detector, compilation, stage-checkpoint, cache-content, diagnostic-content, annotation, and evaluation behavior remains authoritative. This replacement changes producer destination defaults and path interpretation, but does not redesign media or analysis algorithms.

## 10. Acceptance

The replacement is useful only when the actual user workflow succeeds. Acceptance requires a disposable real recording directory containing multiple videos and confirms:

1. one Mise command processes every source and prints one summary URL;
2. the summary uses filenames and links each compilation and attempt review;
3. an honest-empty source is understandable;
4. an actual attempt failure is visible without opening private manifests;
5. interruption followed by the same command preserves completed attempts and resumes incomplete work;
6. an immediate completed rerun performs no model inference or media export;
7. `--force` on one video leaves sibling generated results untouched;
8. directory `--force` regenerates all selected results;
9. source videos remain byte-for-byte unchanged; and
10. terminal output remains readable in the presence of model-runtime diagnostics.

Automated tests use temporary directories and injected expensive collaborators. At least one manual acceptance run must use the real cutting and analysis stack. Passing injected tests alone is not acceptance.

# Replacement work units

The implementation must be a net deletion. Each unit has one externally reviewable outcome and should avoid reusable abstractions that are not needed by the next unit.

## Work unit 1 — Reset producer destinations

**Outcome:** Make the existing cut and one-attempt analysis commands write the §4 schema by default, without changing detection, export encoding, timestamps, checkpoint scoring, or artifact contents.

**Primary files:**

- `src/serve_review/pipeline.py`
- `src/serve_review/analyze_serve.py`
- the `cut` and `analyze-serve` handlers/parsers in `src/serve_review/cli.py`
- their existing focused tests

**Required behavior:**

- `run_cut(VIDEO)` defaults metadata to `VIDEO.parent/metadata/STEM/` and media to `VIDEO.parent/exports/STEM/`.
- Cut metadata/cache files and `run.json` stay in metadata; `run.json` and `CutResult` name the actual export path.
- Compilation mode creates only `exports/STEM/serves.mov`; existing clips/both debug modes place their media beneath the same export destination.
- `run_analyze_serve(VIDEO)` defaults to the exact `metadata/STEM/manual-analysis/` destination.
- A supplied analyzer output directory is exact; no source stem is appended.
- The analyzer's default kinematic cache is beneath its exact output directory.
- CLI defaults use these paths without `--output-dir`. Internal exact-path overrides may remain but must be described as overrides, not normal usage.
- Existing collision and `--overwrite` behavior remains within these lower-level commands.

**Focused check:** Existing pipeline, analyzer, and CLI tests updated for destination behavior, plus `git diff --check`.

**Review gate:** The diff changes path calculation and returned/recorded paths only. It must not add XDG handling, copying, symlinks, migration, workflow records, or algorithm changes.

## Work unit 2 — Build the local coordinator

**Dependency:** Work unit 1 accepted.

**Outcome:** Add one small coordinator for discovery, artifact-based completion, force boundaries, and direct producer calls. No HTML or public command wiring yet.

**Primary files:**

- one new production module, preferably `src/serve_review/process.py`
- one focused test module, preferably `tests/test_process.py`

**Required behavior:**

- Accept exactly one existing video or directory target.
- Discover immediate MOV/MP4 children deterministically; ignore nested directories, `metadata/`, `exports/`, and unsupported files.
- Reject duplicate source stems before processing.
- Derive only the paths specified in §4; create no opaque IDs or global state.
- Classify cut and attempt units from their fixed required artifacts.
- Reject a source fingerprint mismatch with a `--force` instruction.
- Skip complete units without invoking expensive collaborators.
- Clear and retry only an incomplete cut unit or incomplete attempt unit.
- Invoke `run_cut` once per needed source with padding `1` and compilation mode.
- Invoke `run_analyze_serve` once per needed accepted attempt, using its unpadded detected range and exact `metadata/STEM/attempts/serve-NNN/` destination.
- Continue independent attempts and sources after operational failure and retain actionable failure details for presentation.
- Implement identical `force=True` behavior for the future `--force` and `--rebuild` spellings; deletion can target only generated paths for selected source stems.

**Focused check:** Temporary-directory tests with injected cut/analyze collaborators; no model, FFmpeg, network, private media, HTML, CLI, or old workflow package.

**Review gate:** The module should remain a direct loop over sources and attempts. If it introduces stores, selectors, generalized records, provenance history, cleanup policy, or a plugin/service framework, reduce it.

## Work unit 3 — Add summary and user entrypoints

**Dependency:** Work unit 2 accepted.

**Outcome:** Make the coordinator useful through one summary page, concise progress, `mise run process`, and `mise run help`.

**Primary files:**

- `src/serve_review/process.py`
- its focused tests
- the narrow process parser/handler in `src/serve_review/cli.py`
- `mise.toml`

**Required behavior:**

- Render `metadata/index.html` directly from discovered source names, current artifacts, and current invocation failures.
- Use escaped text and relative links to raw sources, compilations, and per-attempt analyzer pages.
- Regenerate the summary even when all computation is skipped.
- Print progress in the exact granularity described in §7; do not invent frame percentages or estimates.
- Print complete/failed counts and one absolute `file://` summary URL.
- Return `0`, `1`, or `2` with the §7 meanings.
- Parse `--force` and `--rebuild` as aliases for one boolean.
- Make `mise run process -- TARGET [--force|--rebuild]` the user entrypoint.
- Make `mise run help` show process first and then repository-only internal commands.
- Do not open a browser automatically or add an installed-application promise.

**Focused check:** Coordinator/CLI tests covering complete, empty, partial failure, skip, force, HTML escaping, relative links, and both force spellings.

**Review gate:** The page is one index over existing outputs, not a new review application. No source-ID pages, comparison, filtering, JavaScript, media proxy, or additional command family.

## Work unit 4 — Remove Stage 2 and accept the replacement

**Dependency:** Work unit 3 accepted.

**Outcome:** Delete the generalized Stage-2 implementation and stale workflow documentation, update the README to the one-user workflow, and prove the replacement against real media.

**Deletion scope:**

- `src/serve_review/workflow/` and its integration hook, except no code that the new coordinator actually uses may be moved wholesale;
- `tests/test_workflow_*.py` superseded by the bounded replacement tests;
- Stage-2 workflow plans, generalized CLI/workspace/schema guides, and session/collection/review guides;
- workflow-only terminology from maintained architecture documentation; and
- obsolete Mise workflow aliases.

Before deletion, inventory exact inbound imports and links. Retain detector, cutting, analyzer, cache, evaluation, and reusable procedure documentation. Do not archive deleted workflow prose elsewhere.

**README outcome:** One usage section shows creating/importing a recording directory, `mise run process -- DIRECTORY`, the resulting `metadata/` and `exports/`, the summary URL, `--force`, and `mise run help`. Internal `uv run --locked` syntax appears only in the development/debug command inventory.

**Automated gate:** Focused replacement tests, full `mise run check`, exact diff review, and a clear net reduction in production code, tests, and maintained documentation.

**Real acceptance gate:** Use a disposable multi-video recording directory and complete all checks in §10, including interrupt/resume, a no-op rerun with no model initialization, one-video force, directory force, visible failure details, valid compilation links, and valid attempt review links. Passing mocks alone is not acceptance.
