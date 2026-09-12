#!/usr/bin/env python3
"""Convert zero-based decoded video-frame labels into a phase annotation manifest.

Labels JSON schema v1:
{"schema_version":1,"dataset_split":"dev","session_id":"...",
 "media":"relative/path.mov","attempts":[{"attempt_id":"serve-001",
 "attempt_label":"serve","stages":{"start":265,...,"finish":510}}]}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from serve_review.checkpoint_evaluation import PhaseAnnotation, PhaseAnnotationManifest
from serve_review.domain import AttemptDocument, STAGE_ORDER
from serve_review.media.probe import probe_source


class AnnotationToolError(ValueError):
    pass


def _load_json(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationToolError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnnotationToolError(f"{what} must be a JSON object")
    return value


def _timestamps(video: Path, ffprobe: str) -> list[float]:
    args = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
            "frame=best_effort_timestamp_time", "-of", "json", str(video)]
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        frames = json.loads(result.stdout).get("frames")
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise AnnotationToolError(f"could not read decoded frame timestamps: {exc}") from exc
    if not isinstance(frames, list) or len(frames) < 2:
        raise AnnotationToolError("source needs at least two decoded video frames")
    times: list[float] = []
    for index, frame in enumerate(frames):
        try:
            value = float(frame["best_effort_timestamp_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnnotationToolError(f"frame {index} has no usable timestamp") from exc
        if not math.isfinite(value) or (times and value <= times[-1]):
            raise AnnotationToolError(f"frame timestamps are invalid at frame {index}")
        times.append(value)
    return times


def _atomic_write(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise AnnotationToolError(f"output exists: {path}; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except OSError: pass
        raise


def annotate(video: Path, attempts_path: Path, labels_path: Path, output: Path, *, ffprobe: str = "ffprobe", overwrite: bool = False) -> Path:
    if not video.is_file():
        raise AnnotationToolError(f"video does not exist: {video}")
    try:
        attempts = AttemptDocument.from_json(attempts_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AnnotationToolError(f"invalid attempts document: {exc}") from exc
    metadata = probe_source(video, ffprobe=ffprobe)
    if metadata.fingerprint != attempts.source_fingerprint:
        raise AnnotationToolError("video fingerprint does not match attempts document")
    if not math.isclose(metadata.duration_seconds, attempts.source_duration_seconds, abs_tol=0.001):
        raise AnnotationToolError("video duration does not match attempts document")
    labels = _load_json(labels_path, "labels")
    required = {"schema_version", "dataset_split", "session_id", "media", "attempts"}
    if set(labels) != required or labels["schema_version"] != 1:
        raise AnnotationToolError("labels must use schema_version 1 and exactly the documented keys")
    if labels["dataset_split"] not in {"dev", "heldout"}:
        raise AnnotationToolError("dataset_split must be dev or heldout")
    media = labels["media"]
    if not isinstance(media, str) or not media or PurePosixPath(media).is_absolute() or ".." in PurePosixPath(media).parts:
        raise AnnotationToolError("media must be a non-escaping relative POSIX path")
    if not isinstance(labels["session_id"], str) or not labels["session_id"]:
        raise AnnotationToolError("session_id must be non-empty")
    if not isinstance(labels["attempts"], list) or not labels["attempts"]:
        raise AnnotationToolError("attempts must be a non-empty list")
    times = _timestamps(video, ffprobe)
    by_id = {attempt.attempt_id: attempt for attempt in attempts.attempts}
    rows: list[PhaseAnnotation] = []
    seen: set[str] = set()
    for entry in labels["attempts"]:
        if not isinstance(entry, dict) or set(entry) != {"attempt_id", "attempt_label", "stages"}:
            raise AnnotationToolError("each label attempt needs attempt_id, attempt_label, and stages")
        attempt_id = entry["attempt_id"]
        if attempt_id in seen or attempt_id not in by_id:
            raise AnnotationToolError(f"unknown or duplicate attempt_id: {attempt_id!r}")
        seen.add(attempt_id)
        if entry["attempt_label"] not in {"serve", "aborted"}:
            raise AnnotationToolError("attempt_label must be serve or aborted")
        stages = entry["stages"]
        if not isinstance(stages, dict) or set(stages) != set(STAGE_ORDER):
            raise AnnotationToolError("stages must contain exactly the canonical eight stage keys")
        attempt = by_id[attempt_id]
        for stage in STAGE_ORDER:
            frame = stages[stage]
            if not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame < len(times) - 1:
                raise AnnotationToolError(f"{stage} frame must be a zero-based index before the last frame")
            start, end = times[frame], times[frame + 1]
            if start < attempt.detected_range.start_seconds or end > attempt.detected_range.end_seconds:
                raise AnnotationToolError(f"{stage} frame {frame} lies outside unpadded attempt {attempt_id}")
            rows.append(PhaseAnnotation(session_id=labels["session_id"], media=media, attempt_id=attempt_id,
                attempt_start_seconds=attempt.detected_range.start_seconds, attempt_end_seconds=attempt.detected_range.end_seconds,
                stage=stage, status="available", interval_start_seconds=start, interval_end_seconds=end,
                manual_keyframe_seconds=start, confidence=None, attempt_label=entry["attempt_label"]))
    if set(by_id) != seen:
        raise AnnotationToolError("labels must cover exactly the attempts document's attempts")
    manifest = PhaseAnnotationManifest(dataset_split=labels["dataset_split"], annotations=tuple(rows))
    _atomic_write(output, manifest.to_json(), overwrite)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a phase annotation manifest from zero-based decoded frame labels.")
    parser.add_argument("video", type=Path); parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe"); parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        print(annotate(args.video, args.attempts, args.labels, args.output, ffprobe=args.ffprobe, overwrite=args.overwrite))
    except AnnotationToolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 1
    return 0

if __name__ == "__main__": raise SystemExit(main())
