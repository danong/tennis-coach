# Configuration

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

Configuration is intentionally small. Workspace selection has this precedence:

1. `--workspace PATH` on the workflow command;
2. `SERVE_REVIEW_WORKSPACE`;
3. `$XDG_DATA_HOME/serve-review`;
4. `~/.local/share/serve-review`.

`--workspace` is available on `process`, `status`, `review`, `compare`, `clean`, and session/collection subcommands. It is expanded as a user path and does not create directories for read-only commands.

`process` flags are `--recursive`, `--session NAME`, `--single-attempt`, `--range START:END`, `--padding SECONDS` (default `1.0`), `--dry-run`, and `--open`. `--single-attempt` and `--range` are mutually exclusive; `--range` uses finite half-open source seconds. `--dry-run` plans without writes. `--open` opens only the resulting landing page.

The Python version is 3.11 (pinned by Mise). `ffmpeg` and `ffprobe` are system prerequisites for real media processing; `serve-review doctor` checks them. Heavy local dependencies (MediaPipe, NumPy, and SciPy) remain required by the current package even though documentation and acceptance tests use injected boundaries and no real media.
