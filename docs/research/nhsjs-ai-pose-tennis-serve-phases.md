# Research: NHSJS AI pose-estimation tennis-serve phases and serve-speed prediction

> **Status:** Historical · **State:** Complete · **Work:** None · **As of:** 2026-09-14

- Source URL: https://nhsjs.com/2026/using-ai-pose-estimation-to-characterize-tennis-serve-phases-and-explore-serve-speed-prediction/
- Retrieval date (UTC): 2026-09-13 (retrieved live via `curl`; HTML parsed to text, ~47 kB extracted text).
- Source identity: Maya Cukras, NHSJS Reports, published 2026-09-09. PDF download link is offered on the page; this report is based on the HTML article body including Abstract / Introduction / Methods / Results / Discussion / References. No private footage, annotations, or local paths are used here.
- Project context read: `README.md`, `docs/architecture/offline-pipeline.md`, `docs/plans/offline-roadmap.md` (M4/M5 leaves), `docs/archive/m4-remediation-plan.md`, plus skim of `docs/proposals/m5-tcn-phase-detection.md` §1–3 for M4/M5 boundary.

> Convention used below: **"Paper claims"** are statements directly supported by the article text/tables/figures. **"Project inference"** is our interpretation for Serve Review and is explicitly labeled as such. Uncertainties are called out; nothing here is a claim of local-footage validation.

## 1. Concise source summary

**Paper claims:** the study asks whether open-source pose estimation on broadcast tennis stills can (a) distinguish serve phases and (b) predict serve speed. Data are manually cropped stills from the publicly available YouTube broadcast feed of the 2025 Australian Open quarter-finalists/semifinalists (Methods, Figure 1, ref. 28). There are 8 athletes and 256 total stills; after manual rejection of gross pose failures, YOLO retains 213/256 and MediaPipe retains 185/256 (Methods "Preprocessing", Tables 2A/2B).

**Paper claims — methods:**

- Two independent static-frame pipelines are compared (Table 1): Ultralytics YOLOv8-Pose (2D X,Y; 640×640 input; confidence 0.60; keypoint confidence 0.75; no preprocessing) vs. MediaPipe BlazePose Heavy in static-image mode (`model_complexity=2`; confidence 0.60; keypoint confidence 0.75; CLAHE + light Gaussian blur preprocessing, described as an empirical robustness measure, explicitly "not claimed to improve joint-level accuracy" and with "no quantitative pose-accuracy validation performed").
- Dominant side is Right with a left override for one athlete ("SHE" in Table 1). Manual interactive review (matplotlib 1/2/3 keys) removes major failures only; it is "not applied as a quantitative measure of pose accuracy."
- Features per frame (6 predictors): dominant-side elbow angle (shoulder–elbow–wrist), knee angle (hip–knee–ankle), hip angle (shoulder–hip–knee) via the standard `arccos((A−B)·(C−B)/(‖A−B‖‖C−B‖))` definition; shoulder-coil angle defined as the angle between the left–right shoulder line and the left–right hip line as a proxy for transverse-plane trunk rotation; dominant wrist and shoulder vertical (Y) position as elevation/extension measures (MediaPipe Y values normalized to torso length). Δ features are extension-minus-loading differences.
- Serve speed comes from on-screen broadcast radar (km/h, e.g. Figure 1a: Sonego 166 km/h), discretized **within each athlete** into Low/Medium/High by that athlete's own 33.3rd/67th percentiles; discrete radar ties make bins slightly uneven (Methods, Table 5).
- Models are all scikit-learn Random Forests with 500 trees, Gini splits, no max depth, fixed a priori, no tuning on test data; no feature scaling; no class reweighting/resampling; impurity (Gini) importance averaged across folds; no permutation importance (Methods "Model Configuration" through "Feature Importance Estimation").

**Paper claims — evaluation design:**

- Phase classification: leave-one-athlete-out cross-validation (train on N−1 athletes, test on the held-out athlete; same-serve frames kept together "when identifiable," with residual correlation caveat for broadcast sampling).
- Speed prediction: within-athlete stratified 5-fold CV, reported as mean accuracy and macro-F1 vs. a strict train-majority baseline (not vs. nominal 1/3 chance), plus Δ-accuracy, sensitivity over minimum-sample thresholds (MIN_N = 12/15/20), and paired within-athlete static-vs-Δ comparisons on shared athletes.

**Paper claims — headline results:**

- Phase (loading vs. extension) generalizes across athletes at near-perfect levels: MediaPipe 3D mean accuracy 0.989 / F1 0.993 / ROC-AUC 1.000; YOLO 2D 0.985 / 0.987 / 0.999 (Table 3).
- Feature importance (Table 4, Gini, mean ± SD over folds): wrist Y-position dominates (MediaPipe 0.34±0.03; YOLO 0.47±0.01); knee angle (0.24 / 0.13) and elbow angle (0.23 / 0.19) are next; hip angle (0.07 / 0.13), shoulder coil (0.11 / 0.06), shoulder Y (0.006 / 0.018) contribute less.
- Static single-frame speed prediction fails to beat the majority baseline: YOLO static 0.336 vs. baseline 0.380 (Δ −0.044); MediaPipe static 0.318 vs. 0.331 (Δ −0.013); macro-F1 ~0.27–0.29 (Table 6).
- Paired loading→extension Δ features modestly beat baseline but remain weak in absolute terms: YOLO Δ 0.456 vs. 0.385 (Δ +0.072, n=4 athletes with ≥12 serves); MediaPipe Δ 0.435 vs. 0.393 (Δ +0.042, n=2); macro-F1 ~0.39–0.41. Paired YOLO static-vs-Δ on the same 4 athletes favors Δ by +0.084 vs. baseline; the MediaPipe paired comparison (n=2) is stated as inconclusive (−0.013, no consistent improvement). The direction persists at MIN_N 15/20 but with sharply reduced n (at MIN_N 20 only n=1 overlap; YOLO Δ 0.476 vs. static 0.402 in that single athlete). Inter-athlete SD of accuracy is 0.055–0.102 (static) and 0.058–0.130 (Δ).

**Paper claims — author interpretation and limits (Discussion):** phase separation is a "relatively coarse and visually distinct task" robust to pose noise, and near-perfect scores reflect task simplicity rather than proof of precise joint centers/angles. Speed depends on coordination/sequencing/energy transfer, not one static posture (citing refs 3–5); Δ gains are "weak but systematic," consistent with static-vs-dynamic (ref. 26) and spatiotemporal-pose (ref. 27) literature. YOLO-vs-MediaPipe differences are confounded by usable-frame counts and class balance and by possible single-camera depth-inference noise (ref. 29); the paper explicitly states the quantity-vs-representation effect "cannot be determined with certainty." Listed degradations: 640×640 athlete crops, dominant-arm motion blur with substantial QC rejection, a small 6-feature set omitting joint rotations/segment rotations/leg force/ad–deuce variation/kinetic-chain interactions (refs 5, 7, 10, 30), and frame-pair Δ rather than continuous sequences. The work is framed as exploratory/feasibility, "not intended to show generalizable predictive modeling."

## 2. Relevant observations (mapped to our vocabulary)

### Pose features and joint angles/speeds

- **Paper claims:** only joint *angles* (elbow/knee/hip/coil) plus two vertical *positions* (wrist/shoulder Y) are used. No joint angular velocities, wrist speed, torso-rotation rate, arc length, smoothing windows, or gap handling are reported; derivatives appear only implicitly as extension-minus-loading Δ. Angle geometry is the textbook three-point arccos; shoulder coil is a 2D/3D shoulder-line-vs-hip-line angle.
- **Project inference:** this is a much sparser feature set than our M4.2 grid (which already has timestamp-aware positions/angles/velocities/accelerations, visibility gating, short-span interpolation with explicit masks, and second-based Savitzky–Golay windows). The paper therefore cannot validate our derivative/smoothing design; at most it corroborates that *simple* wrist-Y + elbow/knee angles carry phase signal even under poor imaging.

### Serve phases

- **Paper claims:** only two coarse phases — loading and extension stills — manually selected from broadcasts (Figure 1b–c; Figure 2 shows YOLO/MediaPipe overlays for the same frames). There is no eight-stage Kovacs model, no start/release/loading/cocking/acceleration/contact/deceleration/finish decomposition, no timestamps, no ordering/transition model, and no interval/keyframe error metrics.
- **Project inference:** phase-classification success does **not** transfer to our eight-stage sequencing problem. The most we can borrow is qualitative: wrist elevation separates coarse pre/post configurations, and knee/elbow flexion carry loading-vs-extension signal — consistent with, but weaker evidence for, our `loading` composite (knee flexion + shoulder/hip tilt + shoulder–hip angle difference) and `cocking` proxy (elbow flexion + wrist-low-relative + reversal into upward motion).

### Contact and release

- **Paper claims:** there is no contact detection (no audio, no ball/racket, no impact frame) and no toss-release observation. "Release" never appears as a predicted event; Δ is just loading-to-extension change. Speed labels come from broadcast radar, not from pose.
- **Project inference:** the paper offers zero evidence about audio-anchored contact (our `contact` rule) or the face-height wrist-crossing toss proxy (our `release` rule). It must not be cited as support for either. Its real lesson is negative: without a contact anchor or ball evidence, even coarse phase pairs only weakly predict an external performance variable.

### Prediction (phase vs. speed)

- **Paper claims:** cross-athlete phase classification ≈ 1.0 AUC (Table 3) vs. within-athlete 3-class speed accuracy < 0.50 and at/below baseline for static frames (Table 6). Δ helps directionally but stays ≈ 0.44–0.46. The author explicitly warns static posture is insufficient and Δ is necessary-but-not-sufficient in this formulation.
- **Project inference:** this supports our existing scope exclusions (design §2: no speed estimation; M4: no biomechanical diagnosis; remediation §8: no coaching/databases). It also supports preferring *change-based* descriptors (velocities, arc-length midpoints for `acceleration`/`deceleration`, displacement runs for `start`/`finish`) over single-frame extrema — which the remediation plan already does. It does not support adding any speed or scoring output in M4.

## 3. Applicability to Serve Review under the narrow M4 contract

The historical M4 remediation contract (see `docs/archive/m4-remediation-plan.md` §1) is: native 120 fps iPhone slow motion, one right-handed player, rear view only, accepted M3 ranges only, existing MediaPipe Heavy body pose + raw-source audio, exact presentation timestamps, M3 outputs immutable, non-blocking advisory phases. Design §2 and roadmap M4 explicitly forbid claiming ball/racket contact, racket drop/orientation, shoulder internal/external rotation, toss release as observed ball event, speed, coaching, or technique scoring from body pose alone.

Under that contract:

- **Directly compatible:** wrist-Y elevation cues, elbow/knee angle configurations, torso-line tilt/separation proxies, visibility-gated evidence with explicit unavailable states, broad contact-relative search regions, and change-based (Δ/velocity/arc-length) descriptors. All are body-kinematic and already in the M4.2/M4.3/remediation vocabulary.
- **Not compatible:** anything requiring broadcast generalization, YOLO pipelines, learned classifiers, radar speed, ball/racket tracking, or 3D depth claims. Our rear-view 120 fps slow-motion source is strictly higher-cadence and more controlled than the paper's 640×640 motion-blurred broadcast stills, so paper accuracy numbers must not be quoted as expected M4 performance.
- **Key uncertainty:** the paper has no rear-view / slow-motion / timestamp / audio / eight-stage equivalent, so every transfer to our setting is analogy, not evidence. Label it as project inference.

## 4. Concrete ideas we can adopt or evaluate in M4 (no ball/racket, learned models, new dependencies, or unsupported claims)

These fit the M4.1–M4.6 + remediation allowed files and need no new packages:

1. **Keep wrist-Y + elbow/knee angles as first-class evidence, and use Table 4 only as a weak diagnostic prior.** *Project inference:* if our phase-debug artifact (remediation §5) shows `loading`/`cocking` evidence dominated by shoulder-Y or shoulder-coil while wrist-Y/knee/elbow disagree, treat that as a smell to inspect, not as a threshold change. Do not reweight to match Table 4 Gini importances — those are dataset- and model-specific and the paper gives no transfer guarantee.
2. **Prefer multi-cue composites over single global extrema (already our M4.3 rule; paper reinforces it).** *Paper claims:* phase separates with several redundant cues (Table 4 spreads weight across wrist/knee/elbow). *Project inference:* evaluate candidate-generation changes by checking whether any stage is decided by one extremum; require at least two visibility-qualified cues per body stage, consistent with remediation §6.
3. **Retain Δ/velocity/arc-length formulations for transitional stages.** *Paper claims:* Δ beats static directionally (Table 6, paired +0.084 for YOLO). *Project inference:* this is circumstantial support for our existing `acceleration`/`deceleration` 50%-arc-length midpoints and `start` upward-velocity-run + displacement rule. No new features are needed; use the debug artifact to verify endpoints exist before deriving midpoints (remediation §6 already requires this).
4. **Adopt the paper's strict-baseline and per-athlete-variability reporting habit in diagnostics only.** *Paper claims:* Table 6 reports accuracy vs. train-majority baseline, Δ-accuracy, macro-F1, MIN_N sensitivity, and inter-athlete SD. *Project inference:* our gate already reports availability, accepted-keyframe rate, signed/absolute error, P90, order violations, and calibration — keep those. Optionally add a trivial-baseline column (e.g., contact-centered naive keyframe) in private debug notes to avoid fooling ourselves with easy stages; do not change the committed gate.
5. **Treat manual-QC rejection rates as justification for visibility gating, not as a preprocessing recipe.** *Paper claims:* 43/256 YOLO and 71/256 MediaPipe frames rejected; MediaPipe more blur-intolerant (Tables 2A/2B); CLAHE+blur was unvalidated robustness theater. *Project inference:* do not add CLAHE/blur (would violate "no unsupported claims" and needs dependency/config churn). Instead, keep M4.2's rule: short visibility-qualified interpolation only, long gaps forbidden, interpolated points never sole support for a high-confidence keyframe.
6. **Use within-attempt normalization cautiously.** *Paper claims:* MediaPipe Y normalized to torso length; speed bins normalized within-athlete via tertiles. *Project inference:* our `loading` composite already normalizes each stream within the attempt before fixed weights (remediation §3). The paper's inter-athlete heterogeneity (SD up to 0.13) is a reminder to keep weights fixed/versioned and derive broad priors from multiple development labels, never one exemplar (remediation §6) — a process point, not a numeric import.

All six are evidence/scoring/diagnostic refinements inside the frozen M4 documents and evaluation tools. None claims ball observation, speed, or coaching.

## 5. Approaches that do not fit M4 (and why)

- **Random Forest phase/speed classifiers (Methods "Classification Model Selection").** Learned models are out of M4 scope by design (roadmap M4.1–M4.4 specify deterministic evidence + DP solver; remediation §8 defers all training). Small-n Gini importances do not transfer to our DP unary/transition scores.
- **Serve-speed prediction of any kind (Tables 5–6).** Design §2 explicitly excludes serve speed; the paper itself shows static speed prediction fails and Δ speed prediction stays weak. Adding speed output would be both out-of-contract and unsupported.
- **YOLOv8-Pose pipeline or any new inference dependency (Table 1).** M4 remediation pins the existing approved MediaPipe Heavy model; roadmap forbids adding dependencies except in the ticket that first uses them. A second pose backend belongs, if anywhere, in a separately gated experiment, not the narrow repair.
- **CLAHE + Gaussian blur preprocessing as a claimed accuracy fix.** The paper explicitly disclaims accuracy benefit and reports no validation. Adopting it would violate the "no unsupported claims" constraint.
- **Broadcast-still generalization claims (cross-athlete AUC ≈ 1, Table 3).** Our contract is one right-handed rear-view player at 120 fps, not multi-athlete broadcast generalization. Quoting Table 3 as an M4 target would be a category error the paper itself warns against (coarse two-class task ≠ precise stage timing).
- **3D joint-depth or shoulder-axial-rotation claims.** MediaPipe 3D depth from single-camera broadcast is flagged by the paper as potentially noisy and inseparable from sample-size effects (Discussion). M4.1 already forbids claiming shoulder internal/external rotation from body pose; this paper does not lift that ban.
- **Ball-racket contact, toss-release-as-ball-event, or radar-speed ground truth.** None exists in the paper's visual evidence; our `contact` stays audio-anchored with uncertainty and `release` stays a body proxy with provenance caveats.

## 6. What belongs in deferred M5 (and why)

Per `docs/proposals/m5-tcn-phase-detection.md` (proposal, not active plan) and remediation §8:

- **Continuous-sequence temporal modeling (TCN/heatmap + DP decoder, M5 §3).** *Why M5:* the paper's core forward-looking lesson is that frame pairs under-model dynamics ("full temporal sequences rather than discrete frame pairs may allow modeling of coordination patterns and energy transfer"). Our M4 solver is fixed-form DP over hand-built evidence; learned spatiotemporal dynamics, modality dropout, and learned duration priors are explicitly deferred.
- **Ball and racket detection/tracking and visual contact claims.** *Why M5-or-later:* the paper has no ball/racket evidence and our M4 contract forbids such claims; M5 is the first place ball channels appear (M5 §3 pipeline), still gated separately.
- **Multi-view / multi-handedness / 30–240 fps generalization and cross-player claims.** *Why M5:* the paper's LOAO phase result is suggestive but built on coarse broadcast stills with manual QC; generalizing our narrow rear-view right-handed contract requires the M5 data, labeling, and evaluation apparatus, not an M4 hotfix.
- **Higher-resolution / higher-frame-rate source studies and any speed–biomechanics modeling.** *Why M5-or-later:* the paper calls for "higher resolution video" and richer features (joint rotations, segment rotations, leg force, ad/deuce context); design excludes speed entirely, so even exploratory speed work waits until after M4/M5 phase timing is trustworthy.
- **Anything requiring PyTorch, new pose backends, or training infrastructure.** *Why M5:* remediation §8 names PyTorch/TCN training as deferred; M4 must not add dependencies.

If M4 succeeds narrowly, its dense 120 fps caches and diagnostics become *proposed* labels for M5 review (remediation §8) — never automatic ground truth.

## 7. Prioritized recommendations

1. **No M4 threshold or weight changes from this paper.** (Paper is coarse two-phase broadcast stills; our failure is eight-stage chronology on 120 fps rear-view slow motion. Changing evidence weights to match Table 4 would be tuning to an incomparable task.)
2. **Use the paper only to sanity-check the debug artifact once it exists:** confirm `loading`/`cocking` candidates respond to knee/elbow/wrist-Y composites rather than a single extremum or shoulder-Y/coil alone. If they do not, that locates the fault in candidate generation (remediation §5) without importing any paper constants.
3. **Keep the arc-length-midpoint definitions of `acceleration`/`deceleration` and the velocity-run definition of `start`/`finish`.** The paper's only positive speed signal comes from change-based Δ, which is the closest available corroboration for preferring change descriptors over static extrema.
4. **Do not adopt CLAHE/blur, YOLO, RF models, or speed outputs.** Each violates the M4 contract or the no-unsupported-claims rule, and the paper itself withholds validation for the first and shows weakness for the last.
5. **File the M5 pointers narrowly:** continuous-sequence dynamics, ball/racket channels, and any speed–biomechanics exploration go to M5; cite this paper's Discussion (frame pairs insufficient; higher resolution needed; 3D-vs-sample-size unresolved) as motivation, not as a specification.

## 8. Limitations of this report

- Based on HTML article text only; the linked PDF was not separately inspected, and no source inspection beyond page-text extraction is claimed. Figure pixel content was not re-analyzed; Figure 1/2 descriptions follow captions/alt-text as extracted.
- Reference list entries [1]–[30] were skimmed for context (kinetic chain, Hawk-Eye ML, static-vs-dynamic, MediaPipe limits) but not independently verified; citation numbers above refer to the article's own numbering.
- Tables 2A/2B athlete naming follows the page as extracted (including "Blake Shelton" and "SHE" override); any page-level typos are reproduced without correction and do not affect the conclusions.
- No local footage, annotations, or metrics were consulted; all M4/M5 applicability statements are contract-based reasoning, not empirical validation.

*Retrieval status: succeeded (HTTP 200 via curl, 2026-09-13 UTC). Source-inspection limitation: HTML only, no PDF comparison, no figure re-measurement.*
