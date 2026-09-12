# Serve Review offline pipeline design

Status: implementation specification. M1–M4 code exists locally; the M4 development gate is currently failed and requires a narrow heuristic/chronology repair before held-out phase evaluation.

## 1. Product contract

Given one unmodified iPhone slow-motion `.MOV`, find serve attempts without manual scrubbing and export either:

- one normal-timeline compilation containing only detected attempts;
- one movie per attempt; or
- both.

The user controls symmetric context padding in seconds. The intended command is:

```sh
mise run cut -- refs/sessions/dev/2026-09-08-00.mov --padding 1 --output both
```

A later analysis command adds review checkpoints within each detected serve. Everything runs locally on an Apple-silicon Mac.

## 2. Scope and success

### Serve cutting

An attempt is a visible serve-like swing, including abbreviated and back-of-strings motions. An isolated toss without a swing is not an attempt. A candidate range begins before useful preparation and ends after follow-through/recovery. Padding is applied after detection and clamped to source bounds.

The first useful release must:

1. accept one MOV/MP4 path;
2. preserve the source file unchanged;
3. emit versioned JSON with unpadded and padded source-time ranges;
4. create the requested clips/compilation from original source samples;
5. make a session materially faster to review than raw scrubbing; and
6. expose uncertain detections rather than silently substituting the full video.

### Checkpoints

Checkpoint analysis runs only inside accepted serve ranges. Initial checkpoint candidates are loading/trophy, upward-swing initiation, estimated contact window, follow-through, and landing/recovery. Toss release, racket drop, racket orientation, and visually observed contact require racket/ball evidence and are not claimed from body pose alone.

Every emitted checkpoint has a timestamp or uncertainty interval, confidence, provenance (`body-kinematic`, `racket-visual`, `ball-racket-visual`, or `manual`), and availability. Missing output is valid.

### Excluded for now

Live capture, iPhone UI, coaching advice, technique scoring, serve speed, cloud services, training a large foundation model, and generalized stroke recognition are out of scope.

## 3. Source-media rules

The supplied originals are 1080p HEVC MOV files containing 120 or 240 fps video according to ffprobe. The pipeline must still inspect every input rather than infer properties from its name.

- Source times are decimal seconds derived from integer media timestamps; never use a frame index divided by an assumed FPS as canonical time.
- Analysis may sample frames sparsely, but export always reads the original source.
- Ranges are half-open `[start, end)` and clamped to source duration.
- Exact non-keyframe cuts may require re-encoding. A stream-copy mode must never masquerade as frame-accurate output.
- Output preserves dimensions, orientation, normal source-timeline playback, and high-rate frame detail where FFmpeg/VideoToolbox supports it.
- Temporary output is written beside the final result and atomically renamed after success.

Apple Photos `.aae` adjustment sidecars are not inputs to detection. The source MOV's sample timeline is authoritative for v1.

## 4. CLI and outputs

```text
serve-review doctor
serve-review probe VIDEO
serve-review export VIDEO --ranges ranges.json --output {compilation,clips,both}
serve-review cut VIDEO --padding SECONDS --output {compilation,clips,both}
serve-review analyze VIDEO [--attempts attempts.json]
serve-review review-phases VIDEO [--checkpoints checkpoints.json]

# local manual-gate tools (through mise)
phase_annotations.py VIDEO --attempts attempts.json --labels labels.json --output annotations.json
evaluate_checkpoints.py --checkpoints checkpoints.json --annotations annotations.json --output report.json
```

`probe` and `export` are explicit lower-level commands so media behavior can be validated before inference. `cut` composes probe, cached pose extraction, detection, and export.

Default output layout:

```text
output/<source-stem>/
  source.json
  attempts.json
  checkpoints.json             # only after checkpoint analysis
  review-phases/               # labeled JPEGs, HTML index, review manifest
  serves.mov                   # compilation/both
  clips/serve-001.mov          # clips/both
  cache/pose-v1.jsonl          # local derived cache
  run.json                     # versions, options, timings, errors
```

Commands reject missing inputs, negative padding, invalid ranges, unsupported/corrupt media, empty exports, and output collisions unless replacement is explicitly requested. Failures never alter the input.

## 5. Architecture

Keep modules narrow and inference backends replaceable:

| Module | Responsibility | Must not do |
| --- | --- | --- |
| `cli` | Parse arguments, report progress/errors, invoke services | Detector rules or subprocess parsing |
| `media.probe` | Run ffprobe and normalize source metadata | Decode inference frames |
| `media.frames` | Yield bounded timestamped RGB frames at a requested schedule | Retain an entire video in memory |
| `media.export` | Validate ranges and invoke FFmpeg for clips/concatenation | Detect attempts |
| `domain` | Versioned metadata, ranges, attempts, checkpoints | File I/O or framework objects |
| `pose.backend` | Person detection and body-pose inference | Serve classification |
| `pose.cache` | Fingerprinted, versioned observations and resumable writes | Treat stale results as valid |
| `detection.features` | Pose sequence to normalized temporal features | Media/export operations |
| `detection.ranges` | Deterministic state machine producing candidates | Hidden threshold changes |
| `checkpoints` | Analyze accepted ranges for optional phase anchors | Biomechanical diagnosis |
| `evaluation` | Session-disjoint labels, matching, and reports | Tune against held-out labels |

All external process calls use argument arrays, not shell interpolation. JSON schemas include `schema_version`, source fingerprint, method/model version, and configuration ID.

## 6. Dependencies and model policy

Mise pins developer tools and exposes all supported commands. uv owns the Python environment and lockfile. FFmpeg/ffprobe are system prerequisites checked by `doctor` until a reliable mise binary distribution is selected.

Add dependencies only in the ticket that first uses them. Expected inference dependencies are:

- NumPy for arrays/features;
- PyAV or OpenCV-headless for timestamped frame decoding;
- ONNX Runtime for portable Mac inference;
- an exported person detector plus RTMPose-family body model;
- Core ML Tools only during a later conversion experiment.

MMPose, MMCV, and OpenMIM are not runtime dependencies. Model files live in local ignored storage with a recorded version, checksum, input/output contract, and attribution. No model is accepted solely because it has a high benchmark score: it must be useful on representative serve footage and have a credible Core ML path.

## 7. Detection strategy

Use two passes:

1. **Coarse pass:** sample approximately 15–30 fps across the session, track the player, extract body keypoints, and find activity windows.
2. **Focused pass:** inspect only candidate windows more densely for boundaries and checkpoints.

Initial range detection is deterministic over pose-derived features: joint visibility, torso scale, wrist/elbow velocity, overhead evidence, whole-body motion, and return-to-rest evidence. Features normalize for translation and player scale and propagate missing observations explicitly. A versioned configuration owns smoothing, thresholds, hysteresis, duration bounds, merge gaps, and detector padding.

Model inference and range extraction are separate so detection can be tested with synthetic pose sequences and retuned from caches without decoding video again.

## 8. Evaluation and gates

Annotations use source-time ranges and are split by recording session. Report one-to-one interval matching at IoU 0.5, precision, recall, onset/end error, retained-duration ratio, and extra seconds per attempt. Report processing seconds per source minute and peak memory.

Development begins with the local supplied sessions. Before calling automatic cutting reliable, reserve at least one entire session from threshold tuning and manually inspect every miss and false positive. Provisional personal-use targets are 95% recall and 90% precision; boundary padding must not hide poor localization.

Checkpoint evaluation reports each checkpoint separately against manually reviewed uncertainty intervals. Manual labels are kept locally under ignored `refs/annotations/<split>/`: label zero-based decoded source frames, then convert them through `mise run phase-annotate`, which resolves actual FFprobe timestamps and validates source fingerprint/duration against `attempts.json`. Never make canonical times from `frame_index / nominal_fps`. `mise run phase-evaluate` writes the deterministic report without mutating checkpoints or annotations. A useful-body-checkpoint gate may pass while racket/contact checkpoints remain unavailable.

## 9. Testing and local storage

Testing layers are:

1. pure unit tests for schemas, times, features, ranges, and command errors;
2. generated tiny media fixtures for probe/export/timestamp behavior;
3. local video smoke tests whose paths/results are not committed;
4. user review of compilations and checkpoints.

`refs/`, `models/`, `output/`, caches, local manifests, `.tau/`, and Tau workspaces/state remain ignored. Tests must not require private footage or network access. Aggregate sanitized metrics may be committed.
