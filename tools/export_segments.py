#!/usr/bin/env python3
"""Export manually selected video segments from a JSON manifest."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class SegmentExportError(ValueError):
    """An invalid manifest or failed segment export."""


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SegmentExportError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("segments"), list):
        raise SegmentExportError("manifest must be an object with a segments array")
    return value


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SegmentExportError("each segment needs a non-empty string id")
    name = value.strip()
    if name in {".", ".."} or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in name):
        raise SegmentExportError(f"unsafe segment id: {value!r}")
    return name


def _seconds(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SegmentExportError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SegmentExportError(f"{field} must be a finite number >= 0")
    return result


def _export_one(video: Path, segment: dict[str, Any], destination: Path, ffmpeg: str) -> None:
    start = _seconds(segment.get("start_seconds"), "start_seconds")
    end = _seconds(segment.get("end_seconds"), "end_seconds")
    if end <= start:
        raise SegmentExportError(f"segment {segment.get('id')!r} must have end_seconds > start_seconds")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".tmp-", suffix=".mp4", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{start:.6f}",
        "-i",
        str(video),
        "-t",
        f"{end - start:.6f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(temporary_path),
    ]
    try:
        subprocess.run(args, check=True)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise SegmentExportError(f"ffmpeg produced no output for {segment['id']}")
        os.replace(temporary_path, destination)
    except FileNotFoundError as exc:
        raise SegmentExportError(f"ffmpeg executable not found: {ffmpeg}") from exc
    except subprocess.CalledProcessError as exc:
        raise SegmentExportError(f"ffmpeg failed for segment {segment['id']!r} with exit code {exc.returncode}") from exc
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def export_segments(video: Path, manifest_path: Path, output_dir: Path, *, ffmpeg: str = "ffmpeg", overwrite: bool = False) -> list[Path]:
    if not video.is_file():
        raise SegmentExportError(f"input video does not exist: {video}")
    manifest = _load_manifest(manifest_path)
    results: list[Path] = []
    for raw_segment in manifest["segments"]:
        if not isinstance(raw_segment, dict):
            raise SegmentExportError("each segment must be an object")
        segment_id = _safe_id(raw_segment.get("id"))
        destination = output_dir / f"{segment_id}.mp4"
        if destination.exists() and not overwrite:
            raise SegmentExportError(f"output exists: {destination}; pass --overwrite to replace it")
        _export_one(video, raw_segment, destination, ffmpeg)
        results.append(destination)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Export manually selected segments from a JSON manifest.")
    parser.add_argument("video", type=Path, help="source video")
    parser.add_argument("manifest", type=Path, help="segment manifest JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for exported MP4 clips")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable (default: ffmpeg)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing clips")
    args = parser.parse_args()
    try:
        for path in export_segments(args.video, args.manifest, args.output_dir, ffmpeg=args.ffmpeg, overwrite=args.overwrite):
            print(path)
    except SegmentExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
