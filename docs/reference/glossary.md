# Glossary and terminology

> **Status:** Current · **State:** Maintained · **Work:** None · **As of:** 2026-09-14

This is the normative vocabulary for current and proposed Serve Review documentation. Literal commands, filenames, schema fields, and Python types retain their implemented names in backticks even when those names predate this vocabulary.

## Media and organization

### Source

One immutable input media asset, identified by its content fingerprint. A filesystem path locates a source but is not its identity. Use **source video** on first mention when that helps a reader, then **source**. Use `VIDEO` only as a CLI metavariable.

### Range

A half-open interval `[start_seconds, end_seconds)` on a source timeline. Qualify it when needed: **candidate range**, **detected range**, **effective range**, or **export range**.

### Candidate

A possible serve-like range that has not completed validation. Candidate status must not be presented as a confirmed serve.

### Attempt

A stable review unit representing one detected serve-like action. The intended model permits an attempt to be accepted, uncertain, rejected, or incomplete. In the current compatibility schema, `attempts.json` contains only exported detections, while `shadows.json` records aborted or incomplete detector hypotheses.

### Serve

Human-readable shorthand for an accepted attempt. In precise detector and schema prose, prefer **candidate** or **attempt** as appropriate.

### Session

An optional, ordered grouping of sources representing a real recording, practice, match, or acquisition context. A source does not require a session. A session is not a processing run or an output directory.

### Collection

A non-exclusive, named or temporary selection of sources or attempts. Collections support corpora and comparisons and may span sessions. Membership never duplicates source media or analysis.

### Corpus

The broader body of reference material available to a workspace. Represent useful subsets of a corpus as collections rather than encoding every grouping as a session.

## Processing and artifacts

### Run

One execution with recorded configuration, software/model versions, status, timing, and errors. Repeating work creates or resumes a run; it does not create a recording session.

### Cache

Reusable derived computation bound to source, model, preprocessing, schedule/coverage, and schema identity. A cache is not an authoritative user result. A cache hit means compatible computation was reused; it does not mean every downstream artifact is current.

### Artifact

Any application-managed generated file, including a cache, manifest, clip, compilation, image, report, or page. Qualify the artifact when possible instead of calling it an “output.”

### Clip

A derived, independently playable media file containing one attempt or explicit source range.

### Compilation

A derived playable file combining multiple export ranges without the intervening source footage.

### Review

The human activity or interface used to inspect attempts and checkpoints. For generated artifacts, say **review page**, **review manifest**, **review image**, or **review clip** rather than using “review” alone.

### Analysis

Derived interpretation of a source or attempt. Qualify the scope when ambiguity is possible: **source analysis**, **attempt detection**, or **stage-checkpoint analysis**.

## Serve-stage analysis

### Stage

One named position in the ordered eight-stage serve model: `start`, `release`, `loading`, `cocking`, `acceleration`, `contact`, `deceleration`, or `finish`. Use **stage** for the conceptual label.

### Checkpoint

The analyzer result associated with one stage. A checkpoint records a timestamp or uncertainty interval plus availability, confidence, and provenance. A checkpoint is data, not an image.

### Keyframe

A decoded or rendered source frame selected to visualize a checkpoint. For example, `contact` is the stage, the estimate at 4.217 seconds is its checkpoint, and the JPEG rendered near 4.217 seconds is its keyframe.

When discussing encoded media, use **codec keyframe** to distinguish an independently decodable video frame from a checkpoint keyframe.

### Phase

A continuous temporal or biomechanical period. Do not use **phase** as a synonym for a point checkpoint. The term remains correct in signal-processing phrases such as **zero-phase filter** and in quoted external terminology.

### Anchor

A solver-internal selected candidate for one of the six directly searched stage checkpoints. `acceleration` and `deceleration` are derived checkpoints, not searched anchors. Do not use **anchor** as a synonym for every checkpoint.

### Anchor video

A fixed development source used for repeatable inspection or comparison. Always qualify this term; it is unrelated to a solver anchor.

## Writing conventions

- Introduce **source video** once, then use **source**.
- Use **candidate** before validation and **attempt** for the persisted review unit.
- Use **serve** only colloquially or for an accepted attempt.
- Use **stage** for the eight labels, **checkpoint** for analyzer data, and **keyframe** only for a frame/image.
- Qualify review artifacts and analysis scope.
- Use **pipeline step** for orchestration timing or errors so it is not confused with a serve stage.
- Put implemented compatibility names such as `PhaseDocument`, `review-phases`, and `stage_timings_seconds` in code formatting.
- Link to this glossary near the beginning of terminology-heavy maintained documents; do not link every occurrence.
- Preserve historical documents as records rather than silently modernizing their terminology.

## Compatibility names and future renames

These changes are intentionally deferred until their code, schemas, CLI compatibility, migrations, and documentation can be handled together.

| Current name | Preferred future name | Notes |
| --- | --- | --- |
| `docs/reference/serve-phase-analysis.md` | `docs/reference/serve-stage-analysis.md` | The maintained document describes stage checkpoints, not continuous phase intervals. |
| `review-phases` | `review-checkpoints` or the future unified `review` command | Keep an alias during CLI migration. |
| `phase-annotate` / `phase-evaluate` | `checkpoint-annotate` / `checkpoint-evaluate` | Coordinate with filenames and annotation schemas. |
| `PhaseDocument`, `AttemptPhase`, `StagePhase` | checkpoint-oriented domain names | Requires a versioned schema/API migration; do not rename prose literals first. |
| `phase-annotations.json` / `phase-evaluation.json` | checkpoint-oriented artifact names | Preserve readers for old local manifests where practical. |
| `review-phases/` and `review-serve-3d/` | one review-artifact layout | Part of the future review-manifest project. |
| `session_dir` in the current one-source cutting path | `source_dir` or `source_artifact_dir` | The directory currently represents a source, not a recording session. |
| `stage_timings_seconds` in `run.json` | `pipeline_step_timings_seconds` | Avoid confusion with serve stages; requires a run-schema version change. |
