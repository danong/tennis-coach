# Serve Review

A local command-line tool for finding serve attempts in long iPhone slow-motion recordings and exporting the useful footage without dead time.

**Status:** the offline project foundation is implemented; trusted media export and automatic detection are next. A later iPhone application is out of the current roadmap.

## What we are building

**Serve cutting:** pass one MOV/MP4 recording, context padding, and an output mode to produce either one gap-free serve compilation, one clip per serve, or both.

**Checkpoint analysis:** after cutting is reliable, identify optional review moments such as loading, upward swing, estimated contact, follow-through, and landing. Racket/ball-dependent claims remain unavailable until supported by appropriate visual evidence.

Everything runs locally. The project does not prescribe practice protocols, score technique, generate coaching, estimate speed, or upload footage.

## Documents

- [Design](docs/design.md): offline behavior, media rules, architecture, detection, checkpoints, and evaluation.
- [Roadmap](docs/roadmap.md): single-run Tau leaves, dependencies, allowed scope, checks, and milestone gates.

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
| `mise run cut -- <video> --padding 1 --output <compilation\|clips\|both>` | Intended offline cutter interface. It currently fails explicitly until detection/export is implemented. |

Use `uv run python` for ad-hoc Python commands rather than an unversioned system `python`. Private source videos, downloaded models, caches, and generated outputs are ignored.

The archived iOS-first plan is retained in `old-docs/` for reference but is not an active implementation specification.

## Footage and privacy

Development uses user-supplied practice footage kept in ignored local storage. Commit synthetic fixtures, annotation schemas, and aggregate test results; do not commit private videos, absolute private paths, credentials, or signing settings. Do not upload footage or send it to a model/service without explicit authorization.

The existing self-analysis Markdown, NeuraSkill HTML, and serve-evaluation PDF are reference material, not validated training labels or executable requirements. Do not bulk-print the self-analysis file: it embeds large base64 images.

## Working with models

Roadmap leaves are designed as single isolated Tau runs using the explicitly authorized free model `opencode/muse-spark-1.3-contributor-free`. Run one leaf at a time. The orchestrator reviews and integrates exact candidate diffs and owns architecture, model/license selection, private-footage evaluation, and milestone gates.

See the [Tau execution contract](docs/roadmap.md#tau-execution-contract). A ticket is complete only when its verifier, exact-diff review, integrated checks, dogfood record, and cleanup evidence are complete.
