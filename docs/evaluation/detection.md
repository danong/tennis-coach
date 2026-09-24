# Detection evaluation

> **Status:** Current · **State:** Maintained · **Work:** Next · **As of:** 2026-09-23

`mise run evaluate -- SESSION_DIR` compares each fully reviewed video's
`attempts.json` with its saved serve labels. It matches intervals one-to-one at
temporal IoU 0.5 and reports true positives, false positives, false negatives,
and F1. `--video NAME` (repeatable) selects a fixed subset; `--json` includes
per-video details. It reads generated artifacts and annotations without running
models or changing either. Manually added serves count as false negatives if
unmatched; ambiguous labels can excuse an overlapping unmatched prediction.
Review ranges have approximate boundaries, so this F1 is an exploratory
serve-finding score, not a precise boundary-localization benchmark.
Unmatched predictions overlapping reviewed shadow swings, toss/aborts, or
other negative examples remain false positives and are reported by label.
Only genuinely ambiguous ranges can excuse a prediction.

The `2026-09-08` pilot has 39 labeled serves and 30 stored predictions: 24 true
positives, 6 false positives, and 15 false negatives. Precision is 0.800,
recall is 0.615, and **F1 is 0.696**. The six false positives are three shadow
swings, two toss/aborts, and one other. Two videos
(`IMG_1080.MOV`, `IMG_1082.MOV`) form a fixed comparison subset with 8 serves;
the remaining nine videos have 31 serves. We inspected both subsets while
exploring the data, so the eight serves are **not an untouched test
set**. The split stays within one recording day and cannot estimate
generalization to new sessions.

Most accepted serves retain their original generated boundaries, so the tiny
onset/end errors in this pilot mostly measure timestamp serialization, not
independent boundary accuracy. The manifest evaluator scores each recording
separately and pools counts afterward; videos in one session cannot match
across their local timestamp origins.

An experimental rule that doubled the audio validation window after three
playback seconds of preparation was discarded. It recovered two pilot serves,
but preparation duration is not a reliable indicator of playback speed and the
same session was used to choose the rule. All 15 manually added misses are in
the assumed 4× playback group and overlap decoder `shadow` hypotheses; that
overlap alone does not establish why each serve was missed. Inspect the
individual hypotheses before changing detection. Reserve fresh sessions for
an unbiased test.
