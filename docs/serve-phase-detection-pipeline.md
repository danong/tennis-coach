# Technical Design Document: Automated Tennis Serve Phase Detection Pipeline

## 1. System Overview & Scope

This document specifies the technical architecture for an automated, high-precision tennis serve phase checkpointing engine. The system operates on monocular smartphone video captured across variable frame rates (30–240 fps), arbitrary camera setups (rear, left-side, right-side), and varied serve styles or player demographics (platform vs. pinpoint, full-loop vs. abbreviated takebacks, jumping vs. non-jumping/standing servers).

The pipeline serves a three-tier system architecture:

* **Macro Level (Goal 1 - Active):** Auto-splicing raw video clips into discrete serve attempt boundaries.
* **Micro Level (Goal 2 - Target Scope):** Extracting exact physical presentation timestamps (`timestamp_us`) for the eight Kovacs serve stages with sub-50 ms temporal accuracy.
* **Downstream Analytics (Goal 3 - Future Context):** Supplying deterministic, scale-invariant kinematic data to a relational store and LLM coaching engine. (Downstream database schemas, 3D lifting passes, and LLM prompt design are explicitly out-of-scope for this document).

---

## 2. Operational Event Definitions

To eliminate label ambiguity and handle diverse server mechanics (e.g., non-jumping servers or abbreviated takebacks), each Kovacs phase is mapped to an explicit, single-frame operational target:

| Kovacs Phase | Phase Type | Operational Event Definition | Primary Feature / Anchor Signal |
| --- | --- | --- | --- |
| **Start ($T_1$)** | Boundary | First frame of sustained, intentional serve-motion onset post-settling | Motion onset of tossing wrist or torso |
| **Toss Release ($T_2$)** | Instant | First frame where the ball visibly separates from the tossing hand | Ball-hand spatial separation / wrist velocity inflection |
| **Max Loading ($T_3$)** | Configuration | Peak composite trunk coil and lower-body loading configuration | Composite score: max shoulder lateral tilt ($\theta_{\text{shoulder}}$) + max hip-shoulder separation angle + peak knee flexion |
| **Cocking ($T_4$)** | Configuration | Racket low-point / maximum shoulder external rotation proxy | Lowest vertical position ($Y_{\min}$) of dominant wrist/racket proxy |
| **Acceleration ($T_5$)** | Transition | **Spatial midpoint** ($50\%$ path distance) along dominant wrist arc between Cocking ($T_4$) and Contact ($T_6$) | Cumulative 2D spatial arc length of dominant wrist: $S(T_5) = 0.5 \times S(T_4 \to T_6)$ |
| **Contact ($T_6$)** | Instant | Exact frame of ball-racket impact | Audio RMS transient peak fused with visual impact proximity |
| **Deceleration ($T_7$)** | Transition | **Spatial midpoint** ($50\%$ path distance) along dominant wrist arc between Contact ($T_6$) and Finish ($T_8$) | Cumulative 2D spatial arc length of dominant wrist: $S(T_7) = 0.5 \times S(T_6 \to T_8)$ |
| **Finish ($T_8$)** | Boundary | Completion of follow-through motion and cessation of kinetic drive | Dominant wrist speed drops below threshold ($V_{\text{wrist}} < \epsilon$) AND torso rotation halts |

---

## 3. System Architecture & Processing Pipeline

```
Raw Video + Audio
       │
       ├── Frame Timestamp Extraction (`timestamp_us`)
       ├── Audio RMS Transient Detector ──> Contact Prior (T_audio)
       └── Macro Serve Splicer (Goal 1 Clip)
               │
               ▼
   Conditional Window Router
       ├── High Confidence: [T_audio - 1.2s, T_audio + 0.5s]
       └── Low Confidence / Fallback: Full Macro Serve Clip
               │
               ▼
   Dense Native Pose & Ball Extraction (120–240 fps)
               │
               ▼
   Canonical 120 Hz Resampling Grid + Binary Observation Mask
               │
               ▼
   1D Temporal Convolutional Network (TCN)
       ├── Modality Dropout (Audio/Ball Channels)
       └── Output: 8 × T Gaussian Heatmaps (σ = 20–50 ms)
               │
               ▼
   Constrained Dynamic Programming (DP) Decoder
       └── Enforces Monotonic Order & Soft Duration Priors
               │
               ▼
   Native-Frame Local Refinement (`timestamp_us` Snapping)
               │
               ▼
   8 Validated Checkpoint Timestamps + Confidence Scores

```

### 3.1 Ingestion & Conditional Window Routing

1. **Native Timebase Tracking:** Video frames are decoded preserving exact presentation timestamps (`timestamp_us`) via container metadata. The pipeline never normalizes video frame rates at the ingestion layer.
2. **Audio Contact Prior:** Audio is processed at 200 Hz using short-window RMS energy to identify impact transients ($T_{\text{audio}}$).
3. **Conditional Windowing:**
* *Fast-Path (High Audio Confidence):* When a sharp impact transient is cleanly identified, processing is cropped to $[T_{\text{audio}} - 1.2\text{ s}, T_{\text{audio}} + 0.5\text{ s}]$ (~200 frames at 120 fps) to optimize compute and eliminate pre-serve noise.
* *Fallback Path (Low Audio Confidence / Windy / Muffled):* If $T_{\text{audio}}$ is ambiguous, the fast crop is bypassed and the full 4–5 second clip isolated by the Goal 1 macro splicer is routed directly into downstream processing.



### 3.2 Dense Feature Extraction & Canonical 120 Hz Resampling Grid

Within the active analysis window, features are extracted densely at the video's native source frame rate (120–240 fps):

* **Spatial Normalization:** 2D keypoints are converted to torso-relative coordinates and scaled by torso length:

$$L_{\text{torso}} = \Vert{}\text{Midpoint}_{\text{shoulders}} - \text{Midpoint}_{\text{hips}}\Vert{}_2$$


* **Smoothed Kinematics:** Joint velocities are computed using physical time deltas ($\Delta t = t_k - t_{k-1}$) and passed through a lightweight Savitzky-Golay filter to suppress coordinate jitter. Raw second derivatives (accelerations) are excluded.
* **Canonical Timebase Resampling:**
* To prevent TCN convolutional kernels from expanding or shrinking in physical time across 30–240 fps inputs, extracted features are linearly interpolated onto a fixed **120 Hz canonical temporal grid**.
* **Binary Observation Mask:** A secondary binary channel $M(t) \in \{0, 1\}$ is appended to the feature stream, where $1$ indicates a true native frame observation and $0$ indicates an interpolated grid step.



### 3.3 1D Temporal Convolutional Network (TCN) Event Scorer

The resampled feature matrix is processed by a lightweight, multi-stage dilated 1D TCN.

#### Input Feature Vector ($F_t \in \mathbb{R}^K$ at 120 Hz)

* Torso-normalized 2D coordinates and visibility scores for key upper/lower body joints.
* Smoothed angular streams (knee flexion, dominant elbow, shoulder tilt, hip tilt, hip-shoulder separation).
* Timestamp-correct 2D joint velocities.
* Audio transient energy and audio confidence score.
* Auxiliary ball position, velocity, and tracking confidence (when available).
* Binary observation mask $M(t)$.
* Static view classification vector (Rear, Left-Side, Right-Side) and handedness flag.

#### Target Formulation

The TCN outputs 8 continuous temporal heatmaps $P(\text{event}_i \text{ at } t)$. Ground truth event timestamps $T_i$ are encoded as continuous physical-time Gaussian targets:


$$y_i(t) = \exp\left(-\frac{(t - T_i)^2}{2 \sigma_i^2}\right)$$


where $\sigma_i \in [20\text{ ms}, 50\text{ ms}]$, scaled proportionally to human annotation variance for that specific phase.

#### Training Regimes

* **Modality Dropout:** Audio and ball tracking feature channels are randomly zeroed out during training ($p = 0.3$) to force the TCN to learn robust kinematic representations rather than over-indexing on audio spikes.
* **Data Augmentations:** Temporal scaling ($\pm 15\%$), timestamp jitter, joint coordinate noise, landmark dropout, and horizontal mirroring (with automatic handedness swapping).

### 3.4 Constrained Dynamic Programming Decoder & Local Native Refinement

1. **Probabilistic Sequence Decoding:** The continuous 1D heatmaps $H_i(t)$ act as unary scores in a Viterbi/DP decoder. The decoder maximizes the global path score subject to soft duration priors and strict monotonicity:

$$\hat{\mathbf{t}} = \arg\max_{t_1 < t_2 < \dots < t_8} \sum_{i=1}^{8} \log H_i(t_i) + \sum_{i=1}^{7} \log p_i(t_{i+1} - t_i)$$



where $p_i(\Delta t)$ represents a log-normal distribution of human phase durations.
2. **Native Local Refinement:** The 120 Hz grid timestamp predictions $\hat{t}_i$ are mapped back to the active native video buffer, selecting the exact native frame (`timestamp_us`) that minimizes physical time offset.

---

## 4. Training Data Requirements for TCN

Because the TCN operates on lightweight pre-computed feature vectors rather than raw video pixels, the model can be trained efficiently on a compact, highly diverse dataset.

### 4.1 Dataset Size & Composition Targets

| Dimension | Target Distribution | Architectural Rationale |
| --- | --- | --- |
| **Total Serves** | 80–120 annotated serve clips | Sufficient to train a low-parameter 1D TCN without overfitting |
| **Distinct Players** | 12–20 unique individuals | Captures variance in body dimensions, tempo, age, and style |
| **Camera Views** | 40% Rear, 30% Left-side, 30% Right-side | Enforces cross-view generalization |
| **Serve Styles** | 70% Full-loop, 30% Abbreviated takeback | Ensures TCN handles structural takeback variations |
| **Mechanics** | 80% Jumping (Platform/Pinpoint), 20% Standing/Non-jumping | Validates upper-body coiling definitions ($T_3, T_8$) |
| **Frame Rates** | 25% 30 fps, 25% 60 fps, 30% 120 fps, 20% 240 fps | Validates the canonical 120 Hz resampling grid and observation mask |

### 4.2 Annotation Protocol & Bootstrapping

1. **Semi-Automated Bootstrapping:** Candidate serve clips are initialized using the audio contact anchor ($T_6$) and coarse heuristic bounds.
2. **Human Verification:** Annotators adjust keyframe boundaries according to the **Operational Event Definitions** in Section 2.
3. **Uncertainty Bounds:** Labelers flag occluded or ambiguous phases, automatically broadening the target Gaussian width $\sigma_i$ for those specific samples during loss computation.
4. **Active Learning Loop:** Human annotation priority is driven by model disagreement: serves that produce high DP-to-TCN residual distance or low TCN peak confidence are selected for manual review.

---

## 5. Alternatives Considered & Rejection Rationale

* **Heuristics + Unconstrained Dynamic Programming:**
* *Rejected:* Unconstrained rule-based peak searching fails on pre-contact phases ($T_2, T_3$), drifting up to 2.6 seconds into pre-serve setup noise and ball bounces.


* **2D Dynamic Time Warping (DTW) Template Matching:**
* *Rejected:* 2D spatial trajectories alter fundamentally across camera angles (rear vs. side views). DTW cannot align features that disappear due to projection geometry without maintaining an unscalable combinatorial library of templates.


* **Off-the-Shelf Monocular 3D Pose Lifting (e.g., MotionBERT):**
* *Rejected for Primary Timing:* Published benchmarks confirm that monocular 3D pose estimators degrade severely when estimating joint velocities during high-speed, dynamic athletic movements. 3D lifting is reserved strictly as a downstream metric engine.


* **Temporal Cycle-Consistency (TCC) / Video Transformers:**
* *Rejected:* Unsupervised TCC alignment exhibits a temporal resolution ceiling of ~0.75 seconds, making it incapable of isolating 100 ms micro-phases without large in-domain pre-training datasets.



---

## 6. Implementation Roadmap

### Phase 1: Core Timebase & Preprocessing Infrastructure

* Implement timestamp-preserving video frame decoding (`timestamp_us`).
* Build the conditional window router (Fast-path $T_{\text{audio}}$ crop vs. Goal 1 full-clip fallback).
* Build the native-FPS feature extraction module (torso normalization, smoothed joint velocities).
* Implement feature resampling onto the 120 Hz canonical grid alongside the binary observation mask $M(t)$.

### Phase 2: Annotation Tooling & Dataset Collection

* Build a lightweight human-in-the-loop annotation GUI displaying frame timestamps, joint trajectories, and audio waveforms.
* Collect and annotate 80–120 serve clips spanning required distributions (views, frame rates, abbreviated/full-loop styles, jumping/standing servers).
* Export annotated keyframes as continuous 120 Hz Gaussian targets ($\sigma_i \in [20\text{ ms}, 50\text{ ms}]$).

### Phase 3: TCN Model & DP Decoder Development

* Implement the 1D dilated TCN network architecture in PyTorch.
* Set up training pipeline with modality dropout (audio/ball features) and data augmentations (temporal scaling, coordinate jitter, mirroring/handedness swaps).
* Implement the semi-Markov / DP Viterbi decoder with soft log-normal duration priors.
* Integrate native-frame local timestamp snapping (`timestamp_us`).

### Phase 4: Benchmarking & Pipeline Validation

* Evaluate model against physical-time metrics (Hit@25ms, Hit@50ms, Hit@100ms, and P90/P95 tail error).
* Perform ablation tests on audio modality dropout, canonical grid resampling, and view conditioning.
* Lock Phase 2 checkpointing model for downstream Goal 3 integration.