# M4 Remediation: Dense Body-Pose and Audio Checkpoints

> **Status:** active remediation plan after the failed development phase gate. This is not M5 and does not introduce ball/racket tracking, learned scoring, TCN training, new model dependencies, multi-view support, or cross-player generalization claims.

## 1. Narrow operating contract

The remediation target is deliberately constrained:

- locally processed iPhone slow-motion source at **native 120 fps**;
- one **right-handed** player;
- **rear-view** camera only;
- accepted M3 attempt ranges only;
- existing MediaPipe Heavy body pose plus raw-source audio;
- exact source presentation timestamps, never `frame / assumed_fps`;
- M3 attempts, clips, and exports remain immutable and non-blocking.

A result outside this contract is `partial` or `unavailable`; it is not silently generalized. The existing M4 documents, evaluation report, review renderer, and private annotations remain the durable interfaces.

## 2. Frozen failure baseline

The current sparse-pose M4 gate failed on a private development exemplar. Contact was near the manual event, but early body stages selected substantially too early and several manually available stages were skipped. This is evidence of feature/evidence/chronology failure, not proof that a learned model or ball tracking is immediately required.

Private labels and exact-decoded-PTS manifests live under ignored `refs/annotations/dev/`. They are frozen for this remediation. Do not inspect or tune on held-out footage.

## 3. Operational body-only checkpoints

These are reproducible **proxies**, not claims of observed ball release, racket orientation, or shoulder axial rotation.

| Stage | M4 remediation operational definition | Required limitations |
| --- | --- | --- |
| `start` | Last stable local low point of the tossing wrist before sustained upward motion. Image-space low means maximum wrist `y`. | `body_pose_estimate`, `camera_relative_only` |
| `release` | First sustained upward crossing of a face/head-height proxy by the tossing wrist after `start`. It is a toss-elevation proxy, not observed ball separation. | `ball_unobserved`, `release_is_body_proxy` |
| `loading` | Best pre-contact body-loading configuration from knee flexion plus rear-view torso/shoulder configuration, inside a broad contact-relative window. | `body_pose_estimate`, `camera_relative_only` |
| `cocking` | Latest credible pre-contact serving-arm configuration with elbow flexion, low wrist relative to shoulder/elbow, and trajectory reversal into the upward contact path. | `shoulder_axial_rotation_unobserved`, `racket_tip_unobserved`, `cocking_is_body_configuration_proxy` |
| `acceleration` | Half cumulative 2D serving-wrist arc length between selected `cocking` and audio contact. | `derived_wrist_arc_checkpoint`, `camera_relative_only` |
| `contact` | Raw-source audio transient anchor with explicit uncertainty; compatible body pose can support, never visually prove, impact. | `audio_anchor_not_visual_contact` |
| `deceleration` | Half cumulative 2D serving-wrist arc length between contact and selected `finish`. | `derived_wrist_arc_checkpoint`, `camera_relative_only` |
| `finish` | First sustained post-contact low-motion/follow-through completion after the serving wrist and torso leave their kinetic-drive period. | `body_pose_estimate`, `camera_relative_only` |

The serving arm is right and tossing arm is left for this plan. No handedness or view inference is attempted.

## 4. Dense native attempt observations

Within each unpadded accepted attempt:

1. Decode every native 120 fps source frame and preserve its presentation timestamp.
2. Run the existing approved Heavy pose model on each native frame.
3. Store a distinct, source-bound dense attempt cache with observation quality and missingness.
4. Compute smoothing and derivatives using physical timestamp deltas and windows expressed in seconds.
5. Do not downsample before pose inference. Any convenience analysis grid retains explicit support timestamps, observation mask, interpolation span, and uncertainty floor.

This is an evidence-quality experiment, not an assertion that dense cadence alone fixes the failed gate.

## 5. Diagnostic before retuning

Create a private deterministic phase-debug artifact for every manually labeled development attempt. It must show:

- manual source time and current selected source time per stage;
- native observation support and visibility at both times;
- all local evidence candidates, unary scores, and rank at the manual time;
- contact-relative offsets and transition/skip contributions;
- selected DP path, unavailable reason, and stable anomaly IDs;
- feature traces and labeled review frames sufficient to inspect a disagreement.

No thresholds change until this diagnostic identifies whether a failure is missing pose evidence, an invalid proxy, an unsuitable contact-relative search region, unary scoring, or DP skip/transition scoring.

## 6. Decoder remediation

After the diagnostic and dense cache are available:

- retain a full-attempt search for `start`; do not crop it out using a contact-centered fast window;
- score `release`, `loading`, and `cocking` in broad, versioned contact-relative regions;
- select `acceleration` and `deceleration` only after their endpoint anchors, using the defined wrist-arc midpoint;
- retain explicit skip/unavailable states for every stage;
- use no hard-coded timings from one exemplar; derive broad priors from multiple development labels;
- version the new feature/evidence/solver configuration and retain old report comparability.

## 7. Evaluation gates

1. Run the unchanged private annotation manifest through `mise run phase-evaluate` before and after every scoped change.
2. Inspect labeled review frames and every unavailable/anomalous stage.
3. Expand to a small development set only after the frozen exemplar improves without regressions.
4. Freeze dense-pose, proxy, evidence, and decoder configurations.
5. Run once on session-disjoint held-out footage; do not tune afterwards.

Metrics include availability, accepted keyframe rate against manual uncertainty intervals, signed/absolute timing error, P90 error, order violations, and confidence calibration. A one-frame manual interval is intentionally strict; timing error remains the primary diagnostic rather than widening labels to hide failure.

## 8. Deferred work

The following belong to M5 or later, not this remediation:

- ball and racket detection/tracking;
- visual ball-racket contact claims;
- TCN/heatmap training, PyTorch, or learned duration priors;
- multiple camera views, handedness inference, or 30–240 fps generalization;
- coaching, 3D lifting, databases, and LLM integration.

If M4 works under this narrow contract, its dense annotations and diagnostics can accelerate manual labeling for M5. They are proposed labels for review, never automatic ground truth.
