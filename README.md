# Serve Review

A local command-line tool for aiding the analysis of videos, ideally slow-motion recordings, of tennis serves.

**Status:** Research and prototyping. 

## What we are building

### Complete

**Serve cutting:** From a video with multiple serves, detect serve attempts, cut out the intermediate footage, and produce either one gap-free serve compilation, one clip per serve, or both. Proven to work on 120 fps videos of my own serve filmed from the rear on a tripod.

**Checkpoint analysis:** From a video of a single serve, automatically detect the following checkpoints (8): start, release, loading, cocking, acceleration, contact, deceleration, finish. Note that while these stage names are taken from [An 8-Stage Model for Evaluating the Tennis Serve](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445225/), the implemented definition and heuristics differ slightly. On my manually labeled serves, we achieved a MAE of ~70ms, which is generally good enough to be useful for analysis.

### Future Work

**iOS app:** Process, cut, and analyze videos from my phone directly on the tennis court.

**Automated analysis:** Maybe some transformer based thing? TBD. 

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
