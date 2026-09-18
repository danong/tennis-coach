# Serve Review

A local command-line tool for finding and reviewing tennis serve attempts in source videos, ideally slow-motion recordings. See the [glossary](docs/reference/glossary.md) for canonical terminology.

**Status:** Local offline workflow.

## What we are building

### Complete

**Serve cutting:** From a source containing multiple serve attempts, detect accepted attempts, remove the intermediate footage, and produce either one gap-free compilation, one clip per attempt, or both. The initial development footage was 120 fps rear-tripod slow motion, but frame rate is not an input requirement: the pipeline also processes ordinary 30 fps video.

**Stage-checkpoint analysis:** From a source containing one attempt, automatically estimate checkpoints for eight stages: start, release, loading, cocking, acceleration, contact, deceleration, and finish. While these stage names are taken from [An 8-Stage Model for Evaluating the Tennis Serve](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445225/), the implemented definitions and heuristics differ slightly. On my manually labeled attempts, we achieved a mean absolute error (MAE) of about 70 ms, which is generally useful for review.

### Future Work

**iOS app:** Process, cut, and analyze videos from my phone directly on the tennis court.

**Automated analysis:** Maybe some transformer based thing? TBD.

## Documents

Start with the [documentation index](docs/README.md). The main operational references are:

- [Processing lifecycle](docs/architecture/processing-lifecycle.md): what happens when `mise run process` runs.
- [Serve extraction and cutting](docs/reference/serve-extraction-and-cutting.md): attempt detection and export.
- [Serve stage-checkpoint analysis](docs/reference/serve-stage-checkpoint-analysis.md): multimodal checkpoint analysis.

If maintained documents conflict, resolve and update them before implementation.

## Offline development setup

The first implementation target is a macOS command-line pipeline. Mise pins Python and uv and provides the project entrypoints; uv owns the Python environment and lockfile. FFmpeg/ffprobe are system prerequisites checked by `doctor`.

```sh
mise install
mise run setup
mise run doctor
mise run check
```

## Process a recording directory

Copy one or more `.mov` or `.mp4` source videos into a meaningfully named directory, then run:

```sh
mise run process -- ~/videos/tennis/2026-09-10
```

The command leaves source videos unchanged, writes detection and checkpoint data under `metadata/`, writes one compilation per source under `exports/`, and prints a `file://` link to `metadata/index.html`. Repeating the command skips complete artifacts and regenerates incomplete work.

Preview without writing, or explicitly regenerate selected output:

```sh
mise run process -- TARGET --dry-run
mise run process -- TARGET --force
```

A video path processes only that source; a directory processes its immediate MOV/MP4 children. Run `mise run help` for the primary and internal command inventory.

Current development commands:

| Command | Purpose |
| --- | --- |
| `mise run setup` | Create/update the Python environment exactly from `uv.lock`. |
| `mise run doctor` | Verify Python, FFmpeg, and ffprobe. |
| `mise run test` | Run offline pipeline tests. |
| `mise run check` | Run diagnostics and tests. |
| `mise run help` | Show the repository command inventory. |
| `mise run process -- TARGET [--dry-run] [--force]` | Process one source or a recording directory and publish its local summary. |
| `uv run --locked serve-review cut <video> --padding 1 --output <compilation\|clips\|both> [--dry-run] [--force]` | Detect accepted attempts and export a compilation, clips, or both. |
| `uv run --locked serve-review analyze-serve <video> [--start-seconds S --end-seconds E] [--dry-run] [--force]` | One-attempt native-PTS stage-checkpoint analysis; writes checkpoints, diagnostics, review, and reusable body-world and ball/racket observation caches. |
| `mise run phase-annotate -- <video> --attempts <attempts.json> --labels <labels.json> --output <annotations.json>` | Convert zero-based decoded-frame labels to exact-PTS private annotations. |
| `mise run phase-evaluate -- --checkpoints <checkpoints.json> --annotations <annotations.json> --output <report.json>` | Write the deterministic local checkpoint-evaluation report. |
| `uv run python tools/export_segments.py <video> <segments.json> --output-dir <dir>` | Export manually selected corpus segments. |

Use `uv run python` for ad-hoc Python commands rather than an unversioned system `python`. Keep user labels and generated annotation manifests under local `refs/annotations/`; use decoded source timestamps, never `frame / assumed_fps`. Large source videos, downloaded models, caches, and generated artifacts are kept out of normal source changes.

Historical iOS-first and superseded remediation plans are retained in [`docs/archive/`](docs/archive/) for reference only; they are not active implementation specifications. For current stage-checkpoint behavior, use [Serve stage-checkpoint analysis](docs/reference/serve-stage-checkpoint-analysis.md).

## Footage and local storage

Development footage and downloaded reference videos live in local `refs/` storage because they are large. Keep source videos, absolute machine paths, credentials, and generated media out of normal source changes; a future Git LFS setup can provide versioned media provenance when useful. Synthetic fixtures, annotation schemas, and aggregate test results are suitable for the repository.

## Working with models

Roadmap leaves are designed as single isolated Tau runs using the explicitly authorized free model `opencode/muse-spark-1.3-contributor-free`. Run one leaf at a time. The orchestrator reviews and integrates exact candidate diffs and owns architecture, model selection, local-footage evaluation, and milestone gates.

For work performed under the deferred roadmap, follow its [Tau execution contract](docs/plans/offline-roadmap.md#tau-execution-contract); its tickets require a verifier, exact-diff review, integrated checks, dogfood record, and cleanup evidence.
