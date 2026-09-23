# Datasets and annotations

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-23

Keep private source videos and labels outside version control. The current local
pilot is `corpus/2026-09-08/`, a single recording session; `corpus/` is ignored.
This session is useful for dogfooding and finding errors, but cannot provide an
independent estimate of detection or checkpoint accuracy. Split future sessions,
not individual serves, into development and holdout sets before tuning.

Run `mise run process -- SESSION_DIR`, then `mise run annotate -- SESSION_DIR`.
The local review site has three steps: classify detected attempts, scan the full
source videos for missed serves, and review release, cocking, and contact on real
serves. It omits rejected detector fragments because they are too noisy to
review. Labels live in `SESSION_DIR/annotations/review-annotations-v1.json`,
separate from regenerable `metadata/` artifacts. Each video annotation records
its source fingerprint, camera view, full-scan flag, labeled source-time ranges,
and checkpoint status/time. A source change requires relabeling that video.

The saved times are video playback timestamps. Google Photos slow-motion exports
can contain baked speed ramps, so these labels can evaluate visual event timing
on the exported timeline, but elapsed time and derived velocity or acceleration
must not be treated as physical capture-time measurements. Use original captures
when validating physical-time metrics.

The current pilot's review counts and remaining evaluation work are in the
[project handoff](../development/2026-09-23-handoff.md). The detection and phase
evaluation protocols have not yet been implemented.
