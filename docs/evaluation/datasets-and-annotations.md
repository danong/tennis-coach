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

The saved times are video playback timestamps and must never be rewritten as
estimated capture timestamps. For the September pilot, assume each serve is
entirely at either normal playback speed (slowdown factor `1`) or quarter-speed
playback (slowdown factor `4`): 120 fps capture encoded for 30 fps playback.
Different serves in the same video can have different factors. Under that
assumption, an interval wholly inside a serve has estimated capture duration
`(playback_end - playback_start) / slowdown_factor`; a video-wide factor or
absolute capture timestamp does not follow from the annotations. Record the
per-serve factor and its assumed/inferred provenance alongside labels when an
evaluator needs capture-time intervals, while retaining source PTS as the
canonical annotation coordinate. Original captures remain preferable for
validating physical-time metrics, especially near speed-ramp boundaries.

The current evaluator infers `1` or `4` from each serve's reviewed
release-to-contact playback interval using a two-second dividing line. This
rule is specific to the clear two-group September pilot and is labeled
exploratory in its output; it is not a reviewed field in the annotation JSON.

The current pilot's review counts and remaining evaluation work are in the
[project handoff](../development/2026-09-23-handoff.md). Current measured
baselines are in [detection](detection.md) and [phase evaluation](phases.md).
