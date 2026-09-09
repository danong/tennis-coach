# Agent instructions

## Project

Serve Review is currently a local macOS command-line pipeline for finding tennis serves in long iPhone slow-motion videos, exporting clips/a compilation, and later identifying review checkpoints. Read `docs/design.md` for behavior and `docs/roadmap.md` for ordered leaf work. `old-docs/` is archived iOS-first planning and is not the current specification.

## Commands

```sh
mise install
mise run setup
mise run doctor
mise run check
```

Use mise task entrypoints and `uv run ...`; do not rely on system Python or manually activate `.venv`. uv owns Python dependencies and `uv.lock`. Add no dependency unless the assigned roadmap leaf explicitly authorizes it.

## Boundaries

- Work on exactly one assigned roadmap leaf.
- Modify only paths explicitly allowed by that leaf/task.
- Do not redesign contracts, alter adjacent modules, weaken checks, or broaden scope.
- Do not make VCS commits, move bookmarks, or modify JJ configuration as a worker.
- Do not access, inspect, copy, rename, or report metadata from `refs/`; it contains private footage/reference material.
- Do not commit media, model weights, caches, outputs, local manifests, absolute private paths, credentials, or Tau state.
- Tests must be deterministic, offline, and independent of private footage. Generate tiny media fixtures in temporary directories when media is required.
- Use subprocess argument arrays rather than shell interpolation. Never mutate source videos.
- Preserve source timestamps and rational media metadata; do not derive canonical source time from frame index/assumed FPS.
- Inference output must represent missing/uncertain observations honestly. Do not label body-kinematic estimates as visually observed racket/ball contact.

## Code conventions

- Python is `>=3.11,<3.12`, pinned by mise.
- Keep domain/detection logic pure and separate from file I/O, FFmpeg, and model runtimes.
- Keep external model/runtime objects behind adapters; version persisted schemas/configurations.
- Write focused tests for errors, boundaries, cancellation/cleanup, and determinism.
- Run the leaf's focused check and `mise run check` before yielding.

## Tau worker completion

Report changed files, exact checks/results, assumptions, deviations, and remaining risks through `tau_yield`. If requirements conflict, a prerequisite is absent, private data appears necessary, or an undeclared dependency seems required, stop and use `tau_request_rescue` instead of improvising.
