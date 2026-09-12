# Agent instructions

## Project

Serve Review is currently a local macOS command-line pipeline for finding tennis serves in long iPhone slow-motion videos, exporting clips/a compilation, and later identifying review checkpoints. Read `docs/design.md` for behavior and `docs/roadmap.md` for ordered leaf work. `old-docs/` is archived iOS-first planning and is not the current specification. Reusable operational procedures are under `docs/procedures/`; consult that directory for anchor-video and downloaded slow-motion corpus workflows when asked to repeat those tasks.

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
- Treat `refs/` as local working storage for footage and reference material; do not add large media artifacts to normal source changes.
- Keep generated media, caches, outputs, local manifests, absolute machine paths, credentials, and Tau state out of commits unless a task explicitly calls for a small, portable fixture or aggregate result.
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

For delegated work, give Tau a concise task specification with allowed files and checks, run the required focused and regression checks, and finish with `tau_yield` reporting changed files, results, assumptions, deviations, and risks. Ask for rescue instead of improvising when requirements conflict, private footage is required, or an undeclared dependency seems necessary.
