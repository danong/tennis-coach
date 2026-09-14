# Research: jasnwag tennis_serve_dataset — Large-Scale 3D Pose Estimation of Professional Tennis Serves from Broadcast Video

> **Status:** Historical · **State:** Complete · **Work:** None · **As of:** 2026-09-14

- Source URL: https://github.com/jasnwag/tennis_serve_dataset
- Retrieval date (UTC): 2026-09-13 (retrieved live via `git clone --depth 1` into a temporary directory; repository contents inspected directly, not merely the GitHub landing page).
- Source identity: Jason Wang, Robert Chen, Patrick Ho, Emmy Kim, Samuel Min, Jaden Shim, Vrishak Vemuri, Derek Wang, Natalie Kupperman, Stephen Baek, University of Virginia. Cited paper: `wang2026largescale`, "Large-Scale 3D Pose Estimation of Professional Tennis Serves from Broadcast Video", Proc. IEEE/CVF CVPR Workshops (CVPRW), 2026 (citation block in README; paper PDF itself was not consulted for this report).
- Commit / branch: `7889457cc9b50b0c2a9fdcc81d786bcdb6b8652c` on `main` (`origin/main`), commit date 2026-03-19 09:33:10 -0400, commit subject "Fix LICENSE authors to match paper". Shallow clone (`--depth 1`, grafted); earlier history was not inspected.
- Project context read: `README.md`, `docs/architecture/offline-pipeline.md`, `docs/plans/offline-roadmap.md` (M4/M5 leaves, gate handoff), `docs/archive/m4-remediation-plan.md`, plus skim of `docs/proposals/m5-tcn-phase-detection.md` §§1–3 for the M4/M5 boundary.
- Dataset payload status: **not downloaded or inspected**. The `.npy`/`.parquet` payload is hosted on Google Drive (link in README) and no payload file is present in the repository. Nothing below claims inspection of serve media, pose arrays, or metadata rows.

> Convention used below: **"Repository fact"** is a statement directly supported by a file inspected in the clone listed in §1. **"Project inference"** is our interpretation for Serve Review and is explicitly labeled as such. Nothing here is a claim of local-footage validation.

## 1. Repository facts — what was inspected

Retrieval method: `git clone --depth 1 https://github.com/jasnwag/tennis_serve_dataset` into `/tmp/jasnwag/tennis_serve_dataset` on 2026-09-13 UTC, followed by reading every non-asset text file in the clone.

Complete non-`.git` file inventory (9 files; no code, no data payload):

| Path | Size | Disposition |
|---|---|---|
| `README.md` | ~6.1 kB, 173 lines | Inspected in full |
| `LICENSE` | ~1.4 kB | Inspected in full |
| `documentation/data_dictionary.md` | ~3.5 kB, 83 lines | Inspected in full |
| `documentation/keypoint_mapping.md` | ~4.3 kB, 136 lines | Inspected in full |
| `documentation/analysis_examples.md` | ~11.8 kB, 377 lines | Inspected in full |
| `.gitignore` | ~1.4 kB | Inspected in full |
| `.gitattributes` | 46 bytes (`*.parquet filter=lfs diff=lfs merge=lfs -text`) | Inspected in full |
| `assets/bounding_grid_8x3.gif` | ~1.3 MB | Listed only, not viewed frame-by-frame |
| `assets/skeleton.gif` | ~4.1 MB | Listed only |
| `assets/server.gif` | ~15.8 MB | Listed only |
| `assets/gender.gif` | ~17.0 MB | Listed only |

Repository facts about absence (verified by filename search excluding `.git/`):

- No `.py`, `.ipynb`, `.yaml`, `.toml`, `.json`, `.parquet`, `.npy`, `.npz`, or `.csv` files exist in the clone.
- No training code, inference scripts, model weights, configuration files, evaluation scripts, or annotation files exist in the clone.
- No `SPLITS`, `train/test`, or cross-validation definition exists anywhere in the clone.
- The `.gitignore` explicitly ignores `*.csv`, `*.json`, `*.parquet`, `*.h5/.hdf5`, `*.pkl/.pickle`, `*.npy/.npz`, `data/raw/`, `data/processed/`, `data/models/`, `data/outputs/`, `data/us_open_data/`, plus media (`.mp4/.avi/.mov/.mkv/...`), audio, and archives — i.e., the repository is documentation-plus-GIFs by design; the data lives outside it.

## 2. Repository facts — dataset contents and provenance (as claimed by README)

All values in this section are README claims, not independently verified (payload not downloaded):

- **Scale:** 5,966 tennis serves from the 2024 US Open, 109 unique players, 113 matches. Gender split: 56 male / 53 female players.
- **Provenance:** broadcast video of professional matches (2024 US Open). Construction is described as a "fully automated pipeline — no manual annotation required — making it scalable to any broadcast tennis footage."
- **Pipeline (four stages, prose only, no code or versions pinned):**
  1. Detection — RTMDet localizes the serving player in each frame.
  2. 2D pose estimation — RTMPose extracts 17-joint 2D keypoints (COCO format).
  3. 3D lifting — MotionBERT lifts 2D poses to 3D coordinates.
  4. Temporal alignment — Dynamic Time Warping (DTW) aligns variable-length sequences to a canonical serve template, producing fixed-length 3D pose sequences plus derived biomechanical features.
- **Hosted layout (after Google Drive download; not present in clone):**
  ```
  tennis_serve_dataset/
  ├── keypoints/                  # 3D pose sequences — shape (T, 17, 3)
  ├── joint_angles/               # Joint angle time series — shape (T, 8)
  ├── angular_velocities/         # Angular velocity time series — shape (T, 8)
  ├── angular_accelerations/      # Angular acceleration time series — shape (T, 8)
  └── metadata.parquet            # Serve-level metadata (5,966 rows × 20 columns)
  ```
- **Quick-start contract (README code block):** `metadata = pd.read_parquet("metadata.parquet")`; single-serve loads `np.load("keypoints/0.npy")` with shape `(T, 17, 3)` and `np.load("joint_angles/0.npy")` with shape `(T, 8)`.
- **Download gate:** the full dataset is behind a Google Drive folder link (`drive.google.com/drive/folders/1Wr7UjMvgLwgqCQ09wSaw94ozvRYRO8fB`). No direct raw URL, no checksum, no version tag, and no LFS pointer are given in the repository.

## 3. Repository facts — annotation definitions and formats

### 3.1 Keypoint schema (`documentation/keypoint_mapping.md` + README)

- 17 COCO-ordered joints: 0 Nose, 1 Left Eye, 2 Right Eye, 3 Left Ear, 4 Right Ear, 5 Left Shoulder, 6 Right Shoulder, 7 Left Elbow, 8 Right Elbow, 9 Left Wrist, 10 Right Wrist, 11 Left Hip, 12 Right Hip, 13 Left Knee, 14 Right Knee, 15 Left Ankle, 16 Right Ankle.
- Coordinates are 3D `(x, y, z)` per joint per frame; array shape `(T, 17, 3)`.
- Confidence scores shape `(n_frames, 17)`, range 0.0–1.0.
- Stated normalization: X/Y normalized to video frame dimensions (0.0–1.0); Z is relative depth (negative = closer to camera). `data_dictionary.md` adds "All 3D coordinates are normalized to the video frame."
- The mapping doc gives worked access examples (right wrist = index 10; three-point `arccos` angle helper for the right elbow from indices 6/8/10) and quality guidance: head/shoulders/hips are "high-quality"; wrists/ankles are "potentially noisy"; occluded joints "may have lower confidence scores"; a 0.5 confidence-threshold filter is shown as an example (not a mandated gate).
- *Project inference:* the joint set overlaps our MediaPipe-33 subset in the joints M4 actually uses (shoulders/elbows/wrists/hips/knees/ankles plus head), but index order, 3D-lifted provenance, and frame-normalized scaling are all incompatible with our upright-unmirrored normalized MediaPipe observation schema — any comparison would need an explicit joint-remap plus re-normalization, not a drop-in load.

### 3.2 Biomechanical feature set (README)

Eight joint angles, each with a velocity and an acceleration time series of shape `(T, 8)`:

| Index | Joint |
|---|---|
| 0 | Left elbow |
| 1 | Right elbow |
| 2 | Left shoulder |
| 3 | Right shoulder |
| 4 | Left hip |
| 5 | Right hip |
| 6 | Left knee |
| 7 | Right knee |

No angle-definition equations, no units (degrees vs. radians), no derivative method (finite difference vs. filtered), no smoothing window, and no gap/missingness convention are stated in the repository. *Project inference:* this is a joints-only angle list — it has no shoulder/hip-tilt, shoulder–hip separation, torso-rotation proxy, wrist-arc length, or audio channel, so it does not specify any of our `loading`/`cocking`/`acceleration`/`deceleration` descriptors.

### 3.3 Metadata columns — two inconsistent descriptions (repository fact)

- README's metadata table (20 columns): `match_id`, `server`, `server_gender` (`M`/`F`), `player1`, `player2`, `PointServer` (1/2), `Speed_KMH`, `n_frames`, `SetNo`, `GameNo`, `PointNumber`, `P1Score`, `P2Score`, `ServeNumber` (first/second), `ServeResult` (Ace/In/Fault/…), `match_num`, `round`, `ElapsedTime`.
- `data_dictionary.md` describes a *different, wider* column set: `video_name`, `json_file_path`, `json_file_found`, `player1/2`, `server_name` (lowercase, no special characters — note: README uses `server`), `server_gender`, `PointServer`, `tournament`, `round`, `court`, `date`, `n_frames`, `keypoints_clean` (embedded JSON array `n_frames × 17 × 3`), `keypoint_scores_clean` (`n_frames × 17`), `PointNumber`, `GameScore`, `SetScore`, `MatchScore`, `PointWinner`, `PointType`, `frame_start`, `frame_end`, `video_fps` (example 30.0), `keypoint_quality`, `frame_completeness`. It notes frame counts "vary from 60–120 frames per serve" and that some columns may contain NaN.
- The dictionary's usage examples load `data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv` with embedded JSON keypoint strings — a CSV layout that contradicts the README's `.parquet` + `.npy` layout. No reconciliation note exists. See §8 (missing documentation).

### 3.4 Splits, labels, and phase annotation

- Repository facts: **no split definition** (no train/validation/test, no session- or player-disjoint protocol) exists in the clone. **No serve-phase labels** exist: there is no `start`/`release`/`loading`/`cocking`/`acceleration`/`contact`/`deceleration`/`finish` equivalent, no timestamps, no intervals, no keyframes, no ordering constraints. Serve context is limited to `ServeNumber` (first/second) and `ServeResult` (Ace/In/Fault/…) plus score fields — outcome labels, not phase labels. There is no ball position, no racket position/orientation, no audio/transient channel, and no contact-event field.
- *Project inference:* the dataset as documented is a collection of DTW-warped motion clips with outcome metadata, not a phase-timing benchmark. It cannot score our checkpoint gate (availability, accepted-keyframe rate, signed/absolute error, P90, order violations, calibration) without substantial new labeling work that is out of M4 scope.

## 4. Repository facts — models, code, and evaluation

- **Models:** named only in README prose — RTMDet (detection), RTMPose (2D COCO-17), MotionBERT (2D→3D lifting). No checkpoint names, no config files, no version pins, no license/attribution notes for the models, no runtime or hardware notes. *Project inference:* none of these can be adopted under our "no new dependencies" constraint, and MotionBERT-style 3D lifting is in any case deferred per remediation §8 (no 3D lifting in M4).
- **Code:** none in the repository. `documentation/analysis_examples.md` contains ~377 lines of illustrative `pandas`/`numpy`/`matplotlib`/`seaborn`/`sklearn` snippets (loading, trajectory/angle helpers, gender comparisons, KMeans clustering, 3D trajectory plots). These are recipes, not a runnable package: they reference a CSV path (`data/full/usopen_points_clean_keypoints_cleaned_with_server_gender.csv`) that is not in the repo and cannot execute as cloned.
- **Reported results (README "Key Results" — claims without protocol):**

| Task | Accuracy | Macro F1 | Majority baseline | Lift |
|---|---|---|---|---|
| Gender | 97.3% | 0.972 | 55.0% | +42.3% |
| Player ID (top 14) | 99.2% | 0.992 | 10.6% | +88.6% |
| Serve quality | 84.0% | 0.757 | 65.0% | +19.1% |

  Speed regression: R² 0.253, MAE 17.2 km/h, RMSE 20.8 km/h.

- Repository facts about what is *missing* around those numbers: no split, no cross-validation scheme, no feature list used per task, no model class named for the classifiers/regressor, no confidence intervals, no per-class breakdown, and no definition of "serve quality" (which `ServeResult` values map to which classes is unstated). The Player-ID task is restricted to the "top 14" players without stating the selection rule. *Project inference:* these numbers must not be cited as generalization results or compared to any Serve Review metric; they are unscoped headline figures.

## 5. Repository facts — provenance and licensing

- **License:** CC BY 4.0 (outlined `LICENSE` file restating share/adapt rights with attribution; full legal text by reference to creativecommons.org). Copyright line: "© 2026 Jason Wang, Robert Chen, Patrick Ho, Emmy Kim, Samuel Min, Jaden Shim, Vrishak Vemuri, Derek Wang, Natalie Kupperman, Stephen Baek." README's License section confirms CC BY 4.0 with attribution requirement.
- *Project inference:* CC BY 4.0 permits local research use with credit, but three cautions apply: (a) the license covers the dataset authors' contribution — underlying broadcast footage rights (2024 US Open) are not addressed in the repository and "no warranties" language applies; (b) any derivative or redistributed artifact would need attribution plus a change note; (c) per our media rules, payload files must stay under ignored local storage (`refs/`-style) and must never be committed. No download is recommended for M4 (see §9).
- **Provenance caveats stated in the repo:** fully automated pipeline with no manual annotation; frame counts vary (60–120 per serve pre-alignment); NaNs possible; unmatched JSON files flagged `json_file_found = False`; quality/confidence and completeness columns provided. The mapping doc concedes wrists/ankles and occluded joints are noisy.

## 6. Applicability to M4 (narrow remediation contract)

The M4 contract (remediation §1) is: native 120 fps iPhone slow motion, one right-handed player, rear view only, accepted M3 ranges only, existing MediaPipe Heavy body pose + raw-source audio, exact presentation timestamps, M3 outputs immutable, advisory non-blocking phases. Design §2 and roadmap M4 forbid claiming ball/racket contact, racket drop/orientation, shoulder axial rotation, toss release as an observed ball event, speed, coaching, or technique scoring from body pose alone.

Against that contract, the dataset's fit is poor for direct use — for structural reasons, not just missing labels:

1. **Time is warped, not canonical.** DTW alignment to a "canonical serve template" (README pipeline step 4) destroys the absolute source-time chronology our gate scores. Signed/absolute timing error, P90 error, contact-relative offsets, and transition/skip contributions are all meaningless on DTW-warped sequences. *Project inference:* this alone disqualifies the dataset as M4 evaluation or tuning data. Our remediation explicitly requires exact presentation timestamps and forbids substituting resampled grids for observations (remediation §4); DTW-warped clips are the extreme form of that anti-pattern.
2. **No contact anchor, no audio, no ball/racket.** Our `contact` is a raw-source audio transient with uncertainty and our `release` is an explicit body proxy pending ball evidence. This dataset offers neither channel, so it cannot inform the two hardest M4 decisions.
3. **Wrong camera, wrong subjects, wrong cadence.** Broadcast multi-view pro footage (dictionary example: 30.0 fps) vs. our fixed rear-view 120 fps single-player iPhone contract. The 60–120-frame pre-alignment counts imply ~2–4 s broadcast clips, not our dense native attempt observations. Cross-domain transfer (pros → one amateur; broadcast → rear-view phone) is unquantified.
4. **Fully automated labels with no manual gate.** Our M4 gate rests on frozen manual intervals and a strict one-frame acceptance discipline. A no-manual-annotation pipeline with acknowledged wrist/ankle noise and NaN gaps cannot serve as ground truth for that gate.
5. **No phase decomposition.** The eight Kovacs stages have no counterpart; outcome fields (`ServeResult`, `PointType`) are not phase onsets.

## 7. Concrete adoptable ideas requiring no new dependencies or scope expansion

None of the items below imports data, models, or code from the repository; none changes M3 outputs; all fit the M4.1–M4.6 + remediation allowed files (feature/evidence/solver wording, diagnostics, report language). Each is framed as a diagnostic or wording improvement, not a threshold import (the repo provides no usable constants).

1. **Keep multi-cue composites; cite the angle-set overlap only as weak corroboration.** *Repository fact:* the dataset ships 8 joint angles (elbows/shoulders/hips/knees) plus velocities/accelerations and reports strong gender/player separability from motion alone. *Project inference:* when the phase-debug artifact (remediation §5) is built, check that no body stage rests on a single global extremum; require ≥2 visibility-qualified cues per stage per M4.3. Do not import any weight or threshold — none is specified.
2. **Treat their velocity/acceleration channels as process corroboration for timestamp-aware derivatives.** *Repository fact:* the payload schema carries per-angle velocity and acceleration series alongside angles. *Project inference:* consistent with our M4.2 choice of timestamp-aware derivatives with second-based Savitzky–Golay windows and explicit boundary/gap confidence. No constant to copy; a design-alignment note for the debug writeup at most.
3. **Retain explicit confidence/quality gating — their schema agrees missingness handling matters.** *Repository fact:* per-joint confidence `(n_frames, 17)`, `keypoint_quality`, `frame_completeness`, and `json_file_found` flags are first-class columns; the mapping doc recommends threshold filtering. *Project inference:* supports our existing rule — short visibility-qualified interpolation only, long gaps forbidden, interpolated points never sole support for a high-confidence keyframe, observation/quality/interpolation-span channels retained (M4.2/M4.3). Keep our explicit binary observation-mask semantics, which are stricter than anything specified here.
4. **Use their DTW step as the documented reason never to warp time in M4.** *Repository fact:* variable-length serves are DTW-aligned to a canonical template to make fixed-length sequences. *Project inference:* add one sentence to the remediation diagnostic notes contrasting this with our contract (exact PTS, physical-delta derivatives, uncertainty never narrower than observation spacing). A useful cautionary citation for reviewers, not a method import.
5. **Borrow metadata discipline, not metadata content.** *Repository fact:* serve-level context (tournament/round/scores/serve number/outcome/speed) travels with every motion clip; the dictionary recommends lowercase normalized player names and documents NaN/missingness conventions. *Project inference:* mirror the habit in our `checkpoints.json` provenance (method/configuration identity, availability, anomaly IDs, limitations) which M4.1/M4.5 already require — no schema change needed.
6. **Treat their speed-regression weakness as support for our scope exclusions.** *Repository fact:* speed prediction is weak (R² 0.253, MAE 17.2 km/h) despite 5,966 serves with serve-speed labels. *Project inference:* corroborates design §2 and remediation §8 exclusions (no speed estimation, no coaching/scoring in M4). Do not add any speed output; cite only as negative evidence if a reviewer proposes it.

## 8. Directions that conflict with M4 (do not adopt)

1. **3D lifting via MotionBERT.** Deferred by remediation §8 (no 3D lifting, no databases); M4 uses 2D MediaPipe Heavy body pose only. Importing lifted-Z features would add an unapproved model dependency and unquantified depth noise.
2. **RTMDet / RTMPose adoption or re-implementation.** New model dependencies, forbidden without explicit authorization (roadmap execution contract). Our pose backend is frozen to the approved Heavy artifact.
3. **DTW-aligned features or fixed-length canonical templates.** Directly contradicts remediation §4 (dense native observations, physical-timestamp smoothing) and would invalidate the phase-timing gate.
4. **Gender / player-ID classification or serve-quality prediction.** Out of scope (design §2 excludes technique scoring/coaching; M4 outputs advisory phase anchors only). The 97–99% headline figures are also unscoped (no protocol) and must not motivate a new classification leaf.
5. **Downloading the Google Drive payload for M4 tuning/evaluation.** session-disjoint private iPhone footage with frozen manual intervals is the only M4 evidence base (roadmap gate handoff). Broadcast pro clips cannot substitute, and payload files would bloat ignored storage for no gate purpose.
6. **Visual-contact, racket-drop, toss-release-as-ball-event, or speed claims.** The dataset contains no ball/racket/audio evidence that could support such claims, and our contract forbids them from body pose alone regardless.

## 9. M5 fit (deferred learned multi-view path)

*Project inference:* the dataset is more relevant to M5 than to M4, but only as a reference design — not as a ready training set:

- **Useful precedents for M5 design:** large-scale automated mining of broadcast serves; joint-angle + velocity + acceleration feature organization; per-joint confidence propagation; outcome/speed metadata joined to motion; dimensionality-reduction visualizations (gender/player clustering GIFs) suggesting motion-space structure worth testing.
- **Blockers for direct M5 use:** DTW-warped time (M5 needs native `timestamp_us` + canonical 120 Hz grid with binary observation mask, M5 §3.2); no phase-event labels for the 8-way heatmap targets (M5 §3.3 needs `T_1..T_8` supervision); no ball channel (M5 assumes dense ball extraction); single-tournament pro-only coverage (no multi-view/hand/demographic generality the M5 proposal targets); Google Drive hosting with no checksums/versioning (conflicts with our reproducibility expectations); broadcast-rights provenance unaddressed.
- **Recommendation:** keep M5 deferred per the roadmap. If M5 is ever scoped, evaluate this dataset then — against raw-timestamp availability, a documented split protocol, and a phase-labeling plan — rather than downloading it now. The remediation plan's closing note stands: dense M4 annotations and diagnostics become reviewable proposed labels for M5; warped third-party clips do not.

## 10. Missing documentation and verification limits

Repository facts about gaps (all verified by reading the full clone):

1. **No data-layout reconciliation:** README (parquet + `.npy`, 20 columns, `server`/`Speed_KMH`/`ServeNumber`/`ServeResult`) vs. `data_dictionary.md` + `analysis_examples.md` (CSV with embedded JSON, `server_name`/`keypoints_clean`/`frame_start`/`video_fps`/score strings). Which layout the Drive folder actually contains, and whether both exist, is unstated.
2. **No split or evaluation protocol** for the gender/player-ID/quality/speed results: no train/test definition, no CV scheme, no model classes, no feature lists, no "serve quality" class mapping, no "top 14" selection rule, no confidence intervals.
3. **No pipeline specification:** RTMDet/RTMPose/MotionBERT versions, configs, checkpoints, DTW template construction, distance metric, and warping constraints are all unstated; player–serve association and de-duplication rules are unstated.
4. **No angle/derivative specification:** joint-angle equations, units, reference frames, and velocity/acceleration computation are unstated (only the 8-joint list and `(T, 8)` shapes are given).
5. **No license provenance for inputs:** CC BY 4.0 covers the authors' contribution; broadcast-footage rights, player publicity considerations, and model-output licensing are unaddressed.
6. **No versioning:** no checksum, DOI, or dataset version tag; the Drive folder is mutable with no pinned snapshot.
7. Verification limits of this report: Google Drive payload not downloaded (no `.npy`/`.parquet` inspected); paper PDF not consulted (README citation only); GIF assets listed but not frame-analyzed; commit history beyond the shallow-clone HEAD not inspected. No media or annotation claim above goes beyond the text files read.

## 11. Prioritized recommendations

1. **Do not download the payload for M4.** (No gate value; warped time, no phases, no audio/ball, domain mismatch, storage cost.) Record this report as the M4-relevant output and move on to the remediation diagnostic.
2. **Adopt nothing numerical; adopt at most §7 items 1–6 as diagnostic wording.** Specifically: multi-cue check in the phase-debug artifact, confidence-gating retention, DTW-as-caution note, provenance-discipline confirmation, and the speed-exclusion corroboration. All are zero-dependency and M3-safe.
3. **If a future worker proposes this dataset for M5,** require before any download: (a) Drive layout reconciliation (§10.1) with checksums; (b) raw pre-DTW timestamps or a statement they are unavailable; (c) a written split/phase-label plan compatible with M5 §§3.2–3.3; (d) explicit orchestrator authorization for ignored-storage budget and CC BY 4.0 attribution handling. Default remains: no download.
4. **Attribution hygiene:** if any idea from this repository influences future code or docs, credit "Wang et al., tennis_serve_dataset (CC BY 4.0), https://github.com/jasnwag/tennis_serve_dataset" with a change note, per the license.

## Appendix — retrieval and verification record

- `git clone --depth 1 https://github.com/jasnwag/tennis_serve_dataset` — exit 0, 2026-09-13 UTC.
- `HEAD` = `7889457cc9b50b0c2a9fdcc81d786bcdb6b8652c` (`main`, commit date 2026-03-19 -0400); remote `origin` = the source URL above.
- Files read in full: `README.md`, `LICENSE`, `documentation/data_dictionary.md`, `documentation/keypoint_mapping.md`, `documentation/analysis_examples.md`, `.gitignore`, `.gitattributes`.
- Filename search confirmed zero code/data files in the clone (no `.py/.ipynb/.yaml/.toml/.json/.parquet/.npy/.npz/.csv` outside `.git/`).
- Worker check performed: `test -s docs/research/jasnwag-tennis-serve-dataset.md` (no `mise run check` per task instructions).
