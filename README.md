# Serve Review

A local command-line tool for finding and reviewing tennis serve attempts in source videos, ideally slow-motion recordings. See the [glossary](docs/reference/glossary.md) for canonical terminology.

**Status:** Research and prototyping. 

## What we are building

### Complete

**Serve cutting:** From a source containing multiple serve attempts, detect accepted attempts, remove the intermediate footage, and produce either one gap-free compilation, one clip per attempt, or both. Proven to work on 120 fps sources of my own serve filmed from the rear on a tripod.

**Stage-checkpoint analysis:** From a source containing one attempt, automatically estimate checkpoints for eight stages: start, release, loading, cocking, acceleration, contact, deceleration, and finish. While these stage names are taken from [An 8-Stage Model for Evaluating the Tennis Serve](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445225/), the implemented definitions and heuristics differ slightly. On my manually labeled attempts, we achieved a mean absolute error (MAE) of about 70 ms, which is generally useful for review.

### Future Work

**iOS app:** Process, cut, and analyze videos from my phone directly on the tennis court.

**Automated analysis:** Maybe some transformer based thing? TBD. 

## Documents

- [Design](docs/architecture/offline-pipeline.md): offline behavior, media rules, architecture, detection, checkpoints, and evaluation.
- [UX and workflow design](docs/plans/ux-workflow-design.md): proposed source, session, collection, workspace, command, resume, status, cleanup, and review-discovery contracts.
- [Deferred roadmap](docs/plans/offline-roadmap.md): earlier single-run leaves, dependencies, allowed scope, checks, and milestone gates.
- [Serve stage-checkpoint analysis](docs/reference/serve-phase-analysis.md): current native-PTS 3D checkpoint pipeline, cues, weights, filtering, artifacts, limitations, and code pointers.
- [Archive](docs/archive/): superseded M4 remediation and iOS-first planning documents.
- [M5 TCN proposal](docs/proposals/m5-tcn-phase-detection.md): deferred learned multi-view phase-detection experiment.

The design is the general behavioral source of truth. [Serve stage-checkpoint analysis](docs/reference/serve-phase-analysis.md) is the operational source of truth for the current native-3D checkpoint path; the deferred roadmap records its earlier delivery order and gates. If maintained documents conflict, resolve and update them before implementation.

## Offline development setup

The first implementation target is a macOS command-line pipeline. Mise pins Python and uv and provides the project entrypoints; uv owns the Python environment and lockfile. FFmpeg/ffprobe are system prerequisites checked by `doctor`.

```sh
mise install
mise run setup
mise run doctor
mise run check
```

Current commands:

| Command | Purpose |
| --- | --- |
| `mise run setup` | Create/update the Python environment exactly from `uv.lock`. |
| `mise run doctor` | Verify Python, FFmpeg, and ffprobe. |
| `mise run test` | Run offline pipeline tests. |
| `mise run check` | Run diagnostics and tests. |
| `mise run cut -- <video> --padding 1 --output <compilation\|clips\|both>` | Detect accepted attempts and export a compilation, clips, or both. |
| `mise run analyze -- <video>` | Legacy attempt-based analysis; requires prior `cut` output or `--attempts`. |
| `uv run --locked serve-review analyze-serve <video> [--start-seconds S --end-seconds E]` | Current one-attempt native-PTS 3D stage-checkpoint analysis; writes checkpoints, diagnostics, a review page, and a reusable world cache. |
| `mise run review-phases -- <video>` | Render labeled pose-overlay keyframes for manual checkpoint review. |
| `mise run phase-annotate -- <video> --attempts <attempts.json> --labels <labels.json> --output <annotations.json>` | Convert zero-based decoded-frame labels to exact-PTS private annotations. |
| `mise run phase-evaluate -- --checkpoints <checkpoints.json> --annotations <annotations.json> --output <report.json>` | Write the deterministic local checkpoint-evaluation report. |
| `uv run python tools/export_segments.py <video> <segments.json> --output-dir <dir>` | Export manually selected corpus segments. |

Use `uv run python` for ad-hoc Python commands rather than an unversioned system `python`. Keep user labels and generated annotation manifests under local `refs/annotations/`; use decoded source timestamps, never `frame / assumed_fps`. Large source videos, downloaded models, caches, and generated artifacts are kept out of normal source changes.

Historical iOS-first and superseded remediation plans are retained in [`docs/archive/`](docs/archive/) for reference only; they are not active implementation specifications. For current stage-checkpoint behavior, use [Serve stage-checkpoint analysis](docs/reference/serve-phase-analysis.md).

## Footage and local storage

Development footage and downloaded reference videos live in local `refs/` storage because they are large. Keep source videos, absolute machine paths, credentials, and generated media out of normal source changes; a future Git LFS setup can provide versioned media provenance when useful. Synthetic fixtures, annotation schemas, and aggregate test results are suitable for the repository.

## Working with models

Roadmap leaves are designed as single isolated Tau runs using the explicitly authorized free model `opencode/muse-spark-1.3-contributor-free`. Run one leaf at a time. The orchestrator reviews and integrates exact candidate diffs and owns architecture, model selection, local-footage evaluation, and milestone gates.

For work performed under the deferred roadmap, follow its [Tau execution contract](docs/plans/offline-roadmap.md#tau-execution-contract); its tickets require a verifier, exact-diff review, integrated checks, dogfood record, and cleanup evidence.
