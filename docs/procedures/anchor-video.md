# Anchor-video cookbook

Use this workflow to create a new local anchor clip and, optionally, compare
its automatic phases with hand annotations. Anchor footage is private and
ignored; do not commit the video, generated output, caches, or annotations.

## 1. Identify the exact source

List the development sessions and use the complete filename:

```sh
find refs/sessions/dev -maxdepth 1 -type f -name '*.mov' -print | sort
```

For example, the source may be
`refs/sessions/dev/2026-09-08-08.mov`; there is no implicit
`refs/sessions/dev/2026-09-08.mov` alias.

## 2. Run the serve cutter

Use a dedicated output directory and retain individual clips:

```sh
mise run cut -- refs/sessions/dev/2026-09-08-08.mov \
  --padding 1 \
  --output clips \
  --output-dir output/anchor-single-serve-02
```

Inspect the generated clips under:

```text
output/anchor-single-serve-02/2026-09-08-08/clips/
```

Copy the selected clip to the anchors directory with its stable name:

```sh
cp output/anchor-single-serve-02/2026-09-08-08/clips/serve-001.mov \
  refs/anchors/single-serve-02.mov
```

Use `--output both` instead when you also want the compilation. The cutter
may detect more than one serve; choose the clip to promote to an anchor rather
than copying the full compilation.

If the source is already manually segmented by source-time intervals, use the
shared exporter instead:

```sh
uv run python tools/export_segments.py SOURCE.mp4 SEGMENTS.json \
  --output-dir refs/corpus/segments/VIDEO
```

## 3. Analyze automatic phases

Pass the attempts file explicitly when the cut used a custom output directory:

```sh
mise run analyze -- refs/sessions/dev/2026-09-08-08.mov \
  --attempts output/anchor-single-serve-02/2026-09-08-08/attempts.json \
  --output-dir output/anchor-single-serve-02
```

This writes `checkpoints.json`. Render the automatic overlay review:

```sh
mise run review-phases -- refs/sessions/dev/2026-09-08-08.mov \
  --checkpoints output/anchor-single-serve-02/2026-09-08-08/checkpoints.json \
  --output-dir output/anchor-single-serve-02
```

Open:

```text
output/anchor-single-serve-02/2026-09-08-08/review-phases/index.html
```

## 4. Convert hand labels

Create a private labels file under `refs/annotations/dev/` using exactly these
stage keys:

```json
{
  "schema_version": 1,
  "dataset_split": "dev",
  "session_id": "dev-2026-09-08-08",
  "media": "sessions/dev/2026-09-08-08.mov",
  "attempts": [
    {
      "attempt_id": "serve-001",
      "attempt_label": "serve",
      "stages": {
        "start": 0,
        "release": 0,
        "loading": 0,
        "cocking": 0,
        "acceleration": 0,
        "contact": 0,
        "deceleration": 0,
        "finish": 0
      }
    }
  ]
}
```

The frame numbers must be **zero-based decoded frame indices in the original
source video**. If labels were made while watching the exported clip, they are
clip-relative and must be mapped back to source frame indices first; do not
simply pass those numbers to `phase-annotate`. In particular, a one-second
padded clip starts before the detector's unpadded range, so a label that looks
valid in the clip can fail validation against the source attempts document.
Map each clip frame to the original decoded frame using FFprobe timestamps (or
the exact source-frame offset), not an assumed FPS. Keep the original
clip-relative labels separately if they are useful for audit.

Convert them to exact source-PTS annotations:

```sh
mise run phase-annotate -- refs/sessions/dev/2026-09-08-08.mov \
  --attempts output/anchor-single-serve-02/2026-09-08-08/attempts.json \
  --labels refs/annotations/dev/single-serve-02.labels.json \
  --output refs/annotations/dev/single-serve-02.phase-annotations.json
```

The converter validates the source fingerprint, duration, frame bounds, and
that every label lies inside the detector's unpadded attempt range. If a
manually selected start precedes that range, either correct the source attempt
range through a reviewed manual-range workflow or re-check the label; do not
silently discard the early phase.

## 5. Evaluate automatic vs. hand phases

```sh
mise run phase-evaluate -- \
  --checkpoints output/anchor-single-serve-02/2026-09-08-08/checkpoints.json \
  --annotations refs/annotations/dev/single-serve-02.phase-annotations.json \
  --output output/anchor-single-serve-02/2026-09-08-08/phase-evaluation.json
```

The report contains per-stage timing error, interval overlap/IoU, availability,
and ordering information. It does not render images.

## Notes

- Never mutate the original source video.
- Use exact source timestamps, not `frame / assumed_fps`, as canonical times.
- A downloaded/transcoded video may report 60 fps even when its description
  says it was shot at 120 fps. Record delivered FPS separately from any capture
  FPS assumption; do not duplicate frames to manufacture 120-fps media.
- `review-phases` currently renders automatic checkpoints only; hand-annotation
  frame galleries require a separate renderer or manual frame extraction.
- Use `--overwrite` only when intentionally replacing an existing generated
  artifact.
