# Agent instructions

## Project

Serve Review is a project that aims to help me, the developer and sole-user of the project, analyze videos of my serve. Right now, this includes: (1) detecting serve attempts in a video and (2) analyzing videos of serves. To get oriented, read the README.md and docs/README.md. Do targetted reads of other docs and files as needed depending on your current task.

## Conventions

- Use mise task entrypoints and `uv run ...`; do not rely on system Python or manually activate `.venv` except when prototyping and testing things. 
- uv owns Python dependencies and `uv.lock`.
- Remember that this project has a single user and is in the prototype stage. This means: (1) concentrate on building the minimum viable product first and let that inform the next unit of work; do not make plans too far into the future. (2) Strike the balance between keeping the codebase simple and maintainable.
- Keep domain/detection logic pure and separate from file I/O, FFmpeg, and model runtimes.
- Keep external model/runtime objects behind adapters; version persisted schemas/configurations.

## Boundaries

- Keep generated media, caches, outputs, local manifests, absolute machine paths, credentials, and Tau state out of commits unless a task explicitly calls for a small, portable fixture or aggregate result.
- Tests must be deterministic, offline, and independent of private footage. Generate tiny media fixtures in temporary directories when media is required.

