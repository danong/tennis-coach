# Tau dogfood journal

Append one bounded entry for every Tau attempt, including failed, paused, rescued, rejected, and no-edit runs. Do not include private footage details, credentials, absolute paths, full transcripts, or private reasoning.

Each entry records:

- date and roadmap leaf;
- run ID, Tau revision, base revision, and adopted revision;
- environment and explicitly authorized model;
- allocation and terminal result;
- worker claim, exact diff summary, verifier result, and binding evidence;
- orchestrator review, integration, and integrated retest;
- release/cleanup result and workspace/graph confirmation;
- measured cost/attention when available; and
- one concrete finding and next action.

### 2026-09-09 — M1.1 Media domain schemas

- **Identity:** run `9caa1fad-e7b3-4deb-9a95-c91caef22e19`; Tau revision unavailable; base `de469e6fc442`; captured candidate `3fe6a16b561c`; adopted `5f2645fa4757`; task stored outside the app.
- **Environment:** macOS/Apple Silicon; JJ 0.45.1; mise Python 3.11.16; uv 0.12.9; model `opencode/muse-spark-1.3-contributor-free`.
- **Allocation/result:** 75 turns / 20 minutes allowed; observed 12 turns; verified.
- **Evidence:** worker changed only `src/serve_review/domain.py` and `tests/test_domain.py`; exact diff review found no dependency or private-media changes; verifier `uv run pytest tests/test_domain.py && mise run check` passed; stored candidate binding valid with empty successor.
- **Review/integration:** approved by orchestrator; candidate duplicated into an independent adopted revision; integrated `mise run check` passed with 65 tests. Human acceptance unavailable.
- **Cleanup:** initial `tau release` returned `blocked-active-work` with `topology-violation`. The empty workspace successor was reparented to Tau's recorded captured candidate, then `tau release` succeeded. The detached rewritten candidate left by the earlier integration mistake had no descendants and was abandoned after release. `jj workspace list` now contains only the default workspace. See `docs/tau-feedback/m1.1-cleanup-blocked.md`.
- **Cost/attention:** measured inference cost unavailable; one integration-command targeting error was caught because the first integrated check collected only four tests, then corrected before acceptance.
- **Finding:** JJ duplicate creates a sibling revision; target the emitted duplicate revision, not `@-`, when describing/moving a bookmark. Its descendant rewrite also blocked Tau cleanup; restoring the workspace successor to Tau's recorded candidate enabled safe release. Next action: dispatch M1.2 from the current `main` and use a topology-preserving adoption flow.
