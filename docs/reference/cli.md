# CLI reference

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

The workflow is local and offline. In an installed environment use `serve-review`; from a checkout use `uv run --locked serve-review`. `mise run process -- ...` is an optional repository alias for `process`.

## Grammar

```text
serve-review process TARGET... [--workspace PATH] [--recursive] [--session NAME]
  [--single-attempt | --range START:END] [--padding SECONDS] [--dry-run] [--open]
serve-review status [SOURCE] [--session SELECTOR | --collection SELECTOR] [--json] [--workspace PATH]
serve-review review (SOURCE | --session SELECTOR | --collection SELECTOR) [--open] [--workspace PATH]
serve-review compare [SOURCE_PATH_OR_ID ...] [--source ID_OR_PATH] [--session SELECTOR] [--collection SELECTOR]
  [--stage STAGE] [--stages STAGE,...] [--save-as NAME] [--open] [--workspace PATH]
serve-review clean --dry-run [--workspace PATH] [--source ID] [--stale] [--failed-runs] [--temporary]
serve-review session {list|show SELECTOR|create NAME|add SELECTOR VIDEO...|remove SELECTOR --source ID|tag SELECTOR --tag TAG} [--workspace PATH]
serve-review collection {list|show SELECTOR|create NAME|add SELECTOR [VIDEO...] [--attempt ID]|remove SELECTOR [--source ID] [--attempt ID]|tag SELECTOR --tag TAG|delete SELECTOR} [--workspace PATH]
```

`START:END` is finite source seconds with `0 <= START < END`; ranges are half-open. Directory targets are non-recursive unless `--recursive` is supplied. Supported media discovery and deterministic ordering are applied by the planner; paths and source IDs select registered sources.

Selectors accept an unambiguous display name or stable ID. Within repeated options, selection is OR; combining selector types is intersection. `status SOURCE` accepts a source path or `source-...` ID. `review` requires exactly one selector. `compare` requires at least one selector; it is read-only except `--save-as`, which writes an explicit collection manifest. Stage names are exact (`start`, `release`, `loading`, `cocking`, `acceleration`, `contact`, `deceleration`, `finish`).

`process --dry-run` performs planning only and creates no workspace files. `--open` opens the printed primary static landing page; without it no browser is opened. Successful `process`, `review`, and `compare` print the landing path and workflow-run artifact paths currently available to the command. Process output also prints a recovery command (`serve-review status --workspace ...`); copy it even if a future status interface changes.

## Exit codes

- `0`: completed, including an honest empty result.
- `1`: processing, rendering, or external-tool failure.
- `2`: invalid command, target, selector, range, or configuration (and `clean` without `--dry-run`).
- `3`: partial batch success with isolated source failures.

The workflow never moves source media. `clean` only inventories candidates: there is no deletion operation in stage 2.

Stage 2 is a compatibility wrapper: batch `process` invokes the existing single-source cutting and stage-checkpoint services once per registered source. It does not prove reuse when provenance is insufficient, and it does not claim that `status`, `review`, or `compare` process media. There is no automatic source move, fuzzy matching, query-backed collection, annotation UI/JS server, or media proxy. Multiple runs have no claimed latest/order when provenance lacks chronology.

Lower-level commands (`probe`, `cut`, `export`, `extract-poses`, `analyze`, `analyze-serve`, and diagnostic review commands) remain compatibility commands; their individual options and exit behavior are documented by their help output and existing reference pages.
