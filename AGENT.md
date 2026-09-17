# Agent instructions

## Project

Serve Review is a project that aims to help me, the developer and sole-user of the project, analyze videos of my serve. Right now, this includes: (1) detecting serve attempts in a video and (2) analyzing videos of serves. To get oriented, read the README.md and docs/README.md. Do targetted reads of other docs and files as needed depending on your current task.

## Working style

- This is a single-user prototype. Optimize for getting the requested behavior working, not for hypothetical future users or failures.
- Make the smallest coherent change that solves the request. Do not expand the task into adjacent cleanup, redesign, documentation, compatibility, or defensive hardening unless asked.
- Apply YAGNI aggressively: prefer deleting code or making a direct edit over adding an abstraction, option, fallback, migration path, or generalized framework.
- Keep tests proportional. Add a regression test when it protects meaningful behavior; do not test that removed or absent behavior remains absent.
- Keep verification proportional too. Run focused checks for small changes. Do not broadly reformat, lint, or clean unrelated code.
- If a simple implementation causes a real problem, the sole user can report it and we can improve it then.

## Conventions

- Use mise task entrypoints and `uv run ...`; do not rely on system Python or manually activate `.venv` except when prototyping and testing things.
- uv owns Python dependencies and `uv.lock`.
- Keep domain/detection logic pure and separate from file I/O, FFmpeg, and model runtimes.
- Keep external model/runtime objects behind adapters; version persisted schemas/configurations.

## Boundaries

- Keep generated media, caches, outputs, local manifests, absolute machine paths, credentials, and Tau state out of commits unless a task explicitly calls for a small, portable fixture or aggregate result.
- Tests must be deterministic, offline, and independent of private footage. Generate tiny media fixtures in temporary directories when media is required.

