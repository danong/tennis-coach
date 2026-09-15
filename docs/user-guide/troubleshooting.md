# Troubleshooting

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

- **Check prerequisites:** `uv run --locked serve-review doctor`. Install FFmpeg/ffprobe and use the pinned Python setup if it fails.
- **Find a prior result:** copy the printed `serve-review status --workspace ...` recovery command, or run `serve-review status --workspace PATH`; add `--json` for strict machine-readable output.
- **Target rejected:** verify the path exists and is a supported media file. Use `--recursive` for nested directory discovery. Use `--range START:END` for one known attempt or `--single-attempt` when the whole source is one attempt.
- **Partial batch (exit 3):** inspect status; successful sources and prior artifacts remain usable. Re-run the source rather than assuming another source succeeded.
- **Selector error (exit 2):** use the complete `source-...`/`session-...`/`collection-...` ID or an unambiguous name. `review` needs exactly one selector; `compare` needs at least one.
- **No browser playback:** the page is static and links the original artifact. No media proxy is currently provided.
- **Cleanup:** run `serve-review clean --dry-run [--stale|--failed-runs|--temporary]`. Stage 2 never deletes; source media and annotations are protected.

The workflow is deliberately conservative: it does not claim cache reuse when provenance cannot prove compatibility, does not automatically move sources, and does not infer chronology or a latest run when multiple runs lack chronological provenance. There is no fuzzy matching, query-backed collection, annotation UI/JS server, or private network/media service. Heavy MediaPipe/NumPy/SciPy dependencies remain current prerequisites.
