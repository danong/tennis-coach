# Agent instructions

## Project

Serve Review is currently a local macOS command-line pipeline for finding serve attempts in long iPhone slow-motion source videos, exporting clips or a compilation, and estimating stage checkpoints. Use the canonical terminology in `docs/reference/glossary.md`. Read `docs/architecture/offline-pipeline.md` for current behavior and `docs/plans/offline-roadmap.md` only for the deferred leaf plan. `docs/archive/old-docs/` is archived iOS-first planning and is not the current specification. Read `docs/reference/serve-phase-analysis.md` for the current native-3D stage-checkpoint path. Reusable operational procedures are under `docs/user-guide/procedures/`; consult that directory for anchor-video and downloaded slow-motion corpus workflows when asked to repeat those tasks.

## Commands

```sh
mise install
mise run setup
mise run doctor
mise run check
```

Use mise task entrypoints and `uv run ...`; do not rely on system Python or manually activate `.venv`. uv owns Python dependencies and `uv.lock`. For roadmap work, add no dependency unless the assigned roadmap leaf explicitly authorizes it.

## Boundaries

- When performing roadmap work, work on exactly one assigned roadmap leaf.
- Modify only paths explicitly allowed by that leaf/task.
- Do not redesign contracts, alter adjacent modules, weaken checks, or broaden scope.
- Do not make VCS commits, move bookmarks, or modify JJ configuration as a worker.
- Treat `refs/` as local working storage for footage and reference material; do not add large media artifacts to normal source changes.
- Keep generated media, caches, outputs, local manifests, absolute machine paths, credentials, and Tau state out of commits unless a task explicitly calls for a small, portable fixture or aggregate result.
- Tests must be deterministic, offline, and independent of private footage. Generate tiny media fixtures in temporary directories when media is required.
- Use subprocess argument arrays rather than shell interpolation. Never mutate a source.
- Preserve source timestamps and rational media metadata; do not derive canonical source time from frame index/assumed FPS.
- Inference output must represent missing/uncertain observations honestly. Do not label body-kinematic estimates as visually observed racket/ball contact.

## Code conventions

- Python is `>=3.11,<3.12`, pinned by mise.
- Keep domain/detection logic pure and separate from file I/O, FFmpeg, and model runtimes.
- Keep external model/runtime objects behind adapters; version persisted schemas/configurations.
- Write focused tests for errors, boundaries, cancellation/cleanup, and determinism.
- Run the leaf's focused check and `mise run check` before yielding.

## Tau worker completion

For delegated work, give Tau a concise task specification with allowed files and checks. The worker runs only its focused check, then finishes with `tau_yield` reporting changed files, results, assumptions, deviations, and risks; Tau's independent task verifier runs exactly one `mise run check`. The orchestrator reviews the exact diff and stored verification rather than rerunning tests. Use generous wall-clock budgets: set the Tau task timeout and outer harness timeout to at least twice the comparable observed duration, with the harness timeout exceeding the task timeout; Tau caps each verifier command at 600 seconds. Ask for rescue instead of improvising when requirements conflict, private footage is required, or an undeclared dependency seems necessary.
