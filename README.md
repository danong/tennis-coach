# Serve Review

A local command-line tool for finding and reviewing tennis serve attempts in source videos, ideally slow-motion recordings. See the [glossary](docs/reference/glossary.md) for canonical terminology.

**Status:** Local offline workflow.

## What we are building

### Complete

**Serve cutting:** From a source containing multiple serve attempts, detect accepted attempts, remove the intermediate footage, and produce either one gap-free compilation, one clip per attempt, or both. The initial development footage was 120 fps rear-tripod slow motion, but frame rate is not an input requirement: the pipeline also processes ordinary 30 fps video.

**Stage-checkpoint analysis:** From a source containing one attempt, automatically estimate checkpoints for eight stages: start, release, loading, cocking, acceleration, contact, deceleration, and finish. While these stage names are taken from [An 8-Stage Model for Evaluating the Tennis Serve](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445225/), the implemented definitions and heuristics differ slightly. On my manually labeled attempts, we achieved a mean absolute error (MAE) of about 70 ms, which is generally useful for review.

**Serve fingerprints:** Serve analysis writes a deterministic, body-only
measurement record for each attempt, combining 30 fixed interpretable metrics
with a `[5, 16, 12]` phase-aligned motion sequence. V1 supports comparison only
among right-handed serves under compatible rear-view capture conditions; see
the [serve fingerprint reference](docs/reference/serve-fingerprints.md).

**Interpretable serve comparison:** Persisted fingerprints can be compared
metric by metric or against a caller-selected robust baseline. Comparisons keep
missingness and compatibility explicit and do not collapse mixed units into a
single score; see the [serve comparison reference](docs/reference/serve-comparisons.md).

### Future Work

**iOS app:** Process, cut, and analyze videos from my phone directly on the tennis court.

## Documents

Start with the [documentation index](docs/README.md). The main operational references are:

- [Processing lifecycle](docs/architecture/processing-lifecycle.md): what happens when `mise run process` runs.
- [Serve extraction and cutting](docs/reference/serve-extraction-and-cutting.md): attempt detection and export.
- [Serve stage-checkpoint analysis](docs/reference/serve-stage-checkpoint-analysis.md): multimodal checkpoint analysis.
- [Serve fingerprints](docs/reference/serve-fingerprints.md): persisted body-only attempt description and comparison domain.

If maintained documents conflict, resolve and update them before implementation.

## Offline development setup

On macOS, install [mise](https://mise.jdx.dev/) and FFmpeg (including
`ffprobe`), then prepare and check the project:

```sh
mise install
mise run setup
mise run doctor
```

Run `mise run test` before changing code, or `mise run check` for the full
local check.

On WSL2 with the RTX 5070, install FFmpeg and use
[NVIDIA's WSL repository](https://docs.nvidia.com/cuda/wsl-user-guide/) to
install `cuda-toolkit-12-8` without a Linux driver. Install `gcc-14` and
`g++-14` too, then run:

```sh
mise install
mise run setup
mise run setup-wsl-cuda
mise run bootstrap
mise run check
```

The WSL task installs Torch 2.7.1 with CUDA 12.8 and builds MMCV against it.
It writes an ignored `.mise.local.toml` that keeps `mise run process -- TARGET`
from restoring the shared Torch 2.1 lock. The macOS setup and lockfile remain
unchanged. If you rerun `mise run setup` on WSL, rerun `mise run setup-wsl-cuda`
before processing footage.

## Process a recording directory

Put one or more `.mov` or `.mp4` recordings in a directory and run:

```sh
mise run process -- ~/videos/tennis/2026-09-10
```

`process` leaves source videos unchanged. It detects attempts, writes metadata
and per-attempt reviews under `metadata/`, writes a compilation under
`exports/`, and prints a link to the local review index. A video path processes
that one source; a directory processes its immediate MOV/MP4 children.

Rerunning skips complete generated work. Use `--dry-run` to see planned work
without writing, or `--force` to regenerate it:

```sh
mise run process -- TARGET --dry-run
mise run process -- TARGET --force
```

For individual cutting, one-attempt analysis, and development tools, run
`mise run help`; the [documentation index](docs/README.md) links to the
corresponding references.

Historical iOS-first and superseded remediation plans are retained in [`docs/archive/`](docs/archive/) for reference only; they are not active implementation specifications. For current stage-checkpoint behavior, use [Serve stage-checkpoint analysis](docs/reference/serve-stage-checkpoint-analysis.md).

## Footage and local storage

Development footage and downloaded reference videos live in local `refs/` storage because they are large. Keep source videos, absolute machine paths, credentials, and generated media out of normal source changes; a future Git LFS setup can provide versioned media provenance when useful. Synthetic fixtures, annotation schemas, and aggregate test results are suitable for the repository.

## Working with models

Run `mise run bootstrap` after cloning to fetch the pinned, ignored RacketVision
checkout and download model artifacts from `models/manifest.json`; every
artifact is checked against its recorded SHA-256. See the
[RacketVision setup notes](tools/racketvision/README.md#setup) for environment
and GPU setup details.

Roadmap leaves are designed as single isolated Tau runs using the explicitly authorized free model `opencode/muse-spark-1.3-contributor-free`. Run one leaf at a time. The orchestrator reviews and integrates exact candidate diffs and owns architecture, model selection, local-footage evaluation, and milestone gates.

For work performed under the deferred roadmap, follow its [Tau execution contract](docs/plans/offline-roadmap.md#tau-execution-contract); its tickets require a verifier, exact-diff review, integrated checks, dogfood record, and cleanup evidence.
