# Workspace and artifacts

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

The workspace is private application storage; use command output and `status` to find artifacts. Resolution precedence is `--workspace PATH`, then `SERVE_REVIEW_WORKSPACE`, then `$XDG_DATA_HOME/serve-review`, then `~/.local/share/serve-review`.

```text
WORKSPACE/
  sources/       registered source manifests and source-scoped compatibility artifacts
  sessions/      session manifests
  collections/   explicit collection manifests
  runs/          workflow provenance
  temporary/     replaceable unnamed review pages
  trash/         reserved cleanup staging
```

`process` registers a source by full content fingerprint, metadata, and one or more absolute known paths. It does not copy or move source media. A source is identified by an opaque `source-...` ID. Sessions and collections reference IDs; they do not own computation or media. A named session gets session pages; an unnamed multi-source process or compare uses `temporary/` and creates no session or collection unless `--save-as` is requested.

Source-scoped compatibility artifacts may include `source.json`, `attempts.json`, `shadows.json`, `run.json`, clips, compilations, checkpoint documents, diagnostics, caches, and review pages. Their nested filenames and layout are private and may change. Only versioned schemas are interchange contracts. Runs record provenance and artifact paths; missing or stale artifacts are reported rather than silently repaired.

`review` and `compare` publish static landing pages from recorded manifests and artifact paths. They do not process media. `--open` opens the printed page. `clean --dry-run` inventories candidates and protected files and never deletes anything; source media, manifests, provenance, and user annotations are protected.
