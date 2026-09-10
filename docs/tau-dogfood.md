# Tau dogfood journal

Append one entry per attempt to the **app's** `docs/tau-dogfood.md` (or its existing
dogfood journal), before calling the task complete. Preserve failures, pauses,
no-edit outcomes, repairs and unavailable measurements. Archive the exact task
under the app's `dogfood/tasks/` only after checking it for private material.
Keep local `.tau` state and raw Pi sessions ignored; they may contain secrets or
private reasoning. The app's roadmap should point to the run and remaining work.

### 2026-09-09 — M1.1 Media domain schemas

- **Identity:** run `9caa1fad-e7b3-4deb-9a95-c91caef22e19`; Tau revision unavailable; base `de469e6fc442`; captured candidate `3fe6a16b561c`; adopted `40540f728c0b` (rebased onto the original foundation during graph repair); task stored outside the app.
- **Environment:** macOS/Apple Silicon; JJ 0.45.1; mise Python 3.11.16; uv 0.12.9; model `opencode/muse-spark-1.3-contributor-free`.
- **Allocation/result:** 75 turns / 20 minutes allowed; observed 12 turns; verified.
- **Evidence:** worker changed only `src/serve_review/domain.py` and `tests/test_domain.py`; exact diff review found no dependency or private-media changes; verifier `uv run pytest tests/test_domain.py && mise run check` passed; stored candidate binding valid with empty successor.
- **Review/integration:** approved by orchestrator; candidate duplicated into an independent adopted revision; integrated `mise run check` passed with 65 tests. Human acceptance unavailable.
- **Cleanup:** initial `tau release` returned `blocked-active-work` with `topology-violation`. The empty workspace successor was reparented to Tau's recorded captured candidate, then `tau release` succeeded. The detached rewritten candidate left by the earlier integration mistake had no descendants and was abandoned after release. `jj workspace list` now contains only the default workspace. See `docs/tau-feedback/m1.1-cleanup-blocked.md`.
- **Cost/attention:** measured inference cost unavailable; one integration-command targeting error was caught because the first integrated check collected only four tests, then corrected before acceptance.
- **Finding:** JJ duplicate creates a sibling revision; target the emitted duplicate revision, not `@-`, when describing/moving a bookmark. Its descendant rewrite also blocked Tau cleanup; restoring the workspace successor to Tau's recorded candidate enabled safe release. Next action: dispatch M1.2 from the current `main` and use a topology-preserving adoption flow.

### 2026-09-09 — M1.2 ffprobe adapter and probe command (parent)

- **Identity:** run `2788cd7f-cea8-4914-97e9-6af38c75ef34`; Tau revision unavailable; base `202f1034`; captured candidate `fe71af158949`; repair task stored outside the app.
- **Environment:** macOS/Apple Silicon; JJ 0.45.1; mise Python 3.11.16; uv 0.12.9; model `opencode/muse-spark-1.3-contributor-free`.
- **Allocation/result:** 75 turns / 20 minutes allowed; observed 21 turns; verified, then rejected for repair.
- **Evidence:** candidate changed only the five allowed M1.2 files; stored verifier passed and binding was valid. Exact review found that rotation parsing accepts direct/tag fields but does not parse ffprobe MOV Display Matrix data in `side_data_list`.
- **Review/lifecycle:** rejected by orchestrator. `tau repair` was attempted with the narrow rotation task but returned `cannot repair: parent current candidate is not inspectable`; no child was created. Integration/human acceptance unavailable.
- **Cleanup:** parent is being rejected through Tau before a fresh retry.
- **Cost/attention:** unavailable.
- **Finding:** a verified candidate can have a meaningful media-metadata gap despite broad unit coverage; repair must test ffprobe's actual alternate metadata shape. Repair is unavailable for this pre-upgrade parent record, so the safe fallback is reject/release then a fresh leaf retry.
