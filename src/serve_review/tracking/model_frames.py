"""Metadata-gated source-frame preparation for RacketVision inference."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve_review.media.frames import FrameError, SampledFrame, iter_sampled_frames

__all__ = [
    "RACKETVISION_HLG_FILTER",
    "ColorMetadata",
    "build_color_probe_args",
    "iter_racketvision_model_frames",
    "parse_color_metadata",
    "probe_color_metadata",
]

RACKETVISION_HLG_FILTER = "colorspace=iall=bt2020:all=bt709:fast=0"


@dataclass(frozen=True, slots=True)
class ColorMetadata:
    """Relevant normalized color tags from the first video stream."""

    space: str | None
    transfer: str | None
    primaries: str | None
    range: str | None

    @property
    def is_bt2020_hlg(self) -> bool:
        return (
            self.space in {"bt2020", "bt2020nc"}
            and self.transfer == "arib-std-b67"
            and self.primaries == "bt2020"
        )

    @property
    def racketvision_filter(self) -> str | None:
        return RACKETVISION_HLG_FILTER if self.is_bt2020_hlg else None


def build_color_probe_args(video: Path | str, *, ffprobe: str = "ffprobe") -> list[str]:
    """Build the narrow ffprobe command used to classify model pixels."""
    if not isinstance(ffprobe, str) or not ffprobe.strip():
        raise FrameError("invalid ffprobe: expected a non-blank executable name.")
    source = str(video)
    if not source:
        raise FrameError("invalid video: expected a non-blank path.")
    return [
        ffprobe.strip(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=color_space,color_transfer,color_primaries,color_range",
        "-of",
        "json",
        source,
    ]


def parse_color_metadata(payload: object) -> ColorMetadata:
    """Parse first-stream color tags, allowing absent tags for ordinary SDR."""
    if not isinstance(payload, dict):
        raise FrameError("color metadata payload must be a JSON object.")
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise FrameError(
            "color metadata must contain exactly one selected video stream."
        )
    stream = streams[0]
    if not isinstance(stream, dict):
        raise FrameError("color metadata video stream must be a JSON object.")

    def value(field: str) -> str | None:
        raw = stream.get(field)
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip():
            raise FrameError(f"color metadata {field} must be a non-blank string.")
        return raw.strip().lower()

    return ColorMetadata(
        space=value("color_space"),
        transfer=value("color_transfer"),
        primaries=value("color_primaries"),
        range=value("color_range"),
    )


def probe_color_metadata(
    video: Path | str, *, ffprobe: str = "ffprobe"
) -> ColorMetadata:
    """Read color tags without decoding or modifying the source."""
    args = build_color_probe_args(video, ffprobe=ffprobe)
    executable = args[0]
    if "/" in executable or "\\" in executable:
        if not Path(executable).is_file():
            raise FrameError(f"{executable!r} was not found.")
    elif shutil.which(executable) is None:
        raise FrameError(f"{executable!r} was not found on PATH.")
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise FrameError(f"could not run ffprobe as {executable!r}: {exc}.") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise FrameError(f"ffprobe could not read video color metadata{suffix}.")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FrameError("ffprobe returned malformed color metadata JSON.") from exc
    return parse_color_metadata(payload)


def iter_racketvision_model_frames(
    video: Path | str,
    times_seconds: list[float] | tuple[float, ...],
    *,
    source: Any = None,
    rotation_degrees: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[SampledFrame]:
    """Yield upright exact-PTS frames with HLG conversion only when tagged."""
    metadata = probe_color_metadata(video, ffprobe=ffprobe)
    return iter_sampled_frames(
        video,
        times_seconds,
        source=source,
        rotation_degrees=rotation_degrees,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        is_cancelled=is_cancelled,
        filter_expression=metadata.racketvision_filter,
    )
