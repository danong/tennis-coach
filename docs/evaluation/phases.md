# Phase evaluation

> **Status:** Current · **State:** Maintained · **Work:** Next · **As of:** 2026-09-23

`mise run evaluate -- SESSION_DIR` compares generated release, cocking, and
contact keyframes with timed, accepted/corrected human labels. It reports
proposal coverage and mean absolute error (MAE), both on source playback time
and estimated real-time milliseconds. The real-time estimate divides each
absolute playback error by four when that serve's reviewed release-to-contact
interval exceeds two playback seconds; otherwise it uses a factor of one.
This is the September pilot's provisional 1×/4× assumption, not recovered
capture timing. Missing proposals do not enter MAE and are reported separately.

On `corpus/2026-09-08`, the stored checkpoint predictions have:

| Checkpoint | Real-time MAE | Timed proposals / labeled serves |
| --- | ---: | ---: |
| Release | 49.3 ms | 23 / 39 |
| Cocking | 21.4 ms | 23 / 39 |
| Contact | 18.5 ms | 23 / 39 |

Pooling the available release, cocking, and contact errors gives a **29.7 ms
focused-checkpoint MAE** in estimated real time.

The 16 missing opportunities per checkpoint include detector misses, so these
MAEs alone do not describe end-to-end checkpoint coverage. The fixed two-video
comparison subset has 7/8 proposals, with release/cocking/contact MAEs of
42.9/14.3/21.4 ms respectively. It was inspected during development, so it is
not an untouched test set. All measurements concern visually marked events in
exported video. They do not validate shoulder rotation, serve power, or injury
risk.
