# Serve Review

A local command-line tool for finding serve attempts in long iPhone slow-motion recordings and exporting the useful footage without dead time.

**Status:** M1–M3 cutting and export, pose extraction, detection, and evaluation are complete. M4 phase-analysis code and review tooling are implemented, but the current development phase-timing gate has failed and needs a narrow heuristic/chronology repair before held-out evaluation. A later iPhone application is out of the current roadmap.

## What we are building

**Serve cutting:** pass one MOV/MP4 recording, context padding, and an output mode to produce either one gap-free serve compilation, one clip per serve, or both.

**Checkpoint analysis:** after cutting is reliable, identify optional review moments such as loading, upward swing, estimated contact, follow-through, and landing. Racket/ball-dependent claims remain unavailable until supported by appropriate visual evidence.

Everything runs locally. The project does not prescribe practice protocols, score technique, generate coaching, or estimate speed.

## Documents

- [Design](docs/design.md): offline behavior, media rules, architecture, detection, checkpoints, and evaluation.
- [Roadmap](docs/roadmap.md): single-run leaves, dependencies, allowed scope, checks, and milestone gates.
- [M4 remediation plan](docs/m4-remediation-plan.md): current dense body-pose/audio repair under a narrow 120 fps rear-view contract.
- [M5 TCN proposal](docs/m5-tcn-phase-detection.md): deferred learned multi-view phase-detection experiment.

The design is the behavioral source of truth. The roadmap defines delivery order and evidence required to mark work complete. If they conflict, resolve and update both before implementation.

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
| `mise run cut -- <video> --padding 1 --output <compilation\|clips\|both>` | Detect accepted serves and export a compilation, clips, or both. |
| `mise run analyze -- <video>` | Reuse compatible cached poses/audio and write `checkpoints.json`. |
| `mise run review-phases -- <video>` | Render labeled pose-overlay keyframes for manual phase review. |
| `mise run phase-annotate -- <video> --attempts <attempts.json> --labels <labels.json> --output <annotations.json>` | Convert zero-based decoded-frame labels to exact-PTS private annotations. |
| `mise run phase-evaluate -- --checkpoints <checkpoints.json> --annotations <annotations.json> --output <report.json>` | Write the deterministic local phase-gate report. |
| `uv run python tools/export_segments.py <video> <segments.json> --output-dir <dir>` | Export manually selected corpus segments. |

Use `uv run python` for ad-hoc Python commands rather than an unversioned system `python`. Keep user labels and generated annotation manifests under local `refs/annotations/`; use decoded source timestamps, never `frame / assumed_fps`. Large source videos, downloaded models, caches, and generated outputs are kept out of normal source changes.

The archived iOS-first plan is retained in `old-docs/` for reference but is not an active implementation specification.

## Footage and local storage

Development footage and downloaded reference videos live in local `refs/` storage because they are large. Keep source videos, absolute machine paths, credentials, and generated media out of normal source changes; a future Git LFS setup can provide versioned media provenance when useful. Synthetic fixtures, annotation schemas, and aggregate test results are suitable for the repository.

## Working with models

Roadmap leaves are designed as single isolated Tau runs using the explicitly authorized free model `opencode/muse-spark-1.3-contributor-free`. Run one leaf at a time. The orchestrator reviews and integrates exact candidate diffs and owns architecture, model selection, local-footage evaluation, and milestone gates.

See the [Tau execution contract](docs/roadmap.md#tau-execution-contract). A ticket is complete only when its verifier, exact-diff review, integrated checks, dogfood record, and cleanup evidence are complete.
