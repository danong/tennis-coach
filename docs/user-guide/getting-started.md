# Getting started

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

Serve Review is an offline/local macOS command-line workflow. Install the package and invoke the installed command:

```sh
serve-review process ~/Movies/practice.mov --open
```

In a repository checkout, use the locked environment instead:

```sh
uv run --locked serve-review process ~/Movies/practice.mov --open
```

The optional repository alias is equivalent:

```sh
mise run process -- ~/Movies/practice.mov --open
```

Set up a checkout with `mise install`, `mise run setup`, then `mise run doctor`. Doctor checks Python, FFmpeg, and ffprobe. `process` registers the source, discovers attempts (or accepts `--single-attempt`/`--range`), records provenance, and prints a static landing page. `--dry-run` is safe for checking discovery and creates no files.

The source video stays where it is. Generated files live below the managed workspace; use the printed `status` recovery command or `serve-review status --json` to locate them. Stage 2 is local and compatibility-focused: it does not provide a polished annotation UI, media proxy, query collections, or fuzzy attempt matching.
