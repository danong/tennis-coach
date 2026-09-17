"""Metadata-gated HLG-to-SDR frame conversion helpers.

Original media remains authoritative. This module only identifies tagged iPhone
HLG sources and supplies the FFmpeg filter for consumers that need ordinary
8-bit BT.709 RGB pixels, such as computer-vision models or JPEG reviews.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from serve_review.media.frames import FrameError

__all__ = [
    "HLG_TO_SDR_FILTER",
    "ColorMetadata",
    "build_color_probe_args",
    "parse_color_metadata",
    "probe_color_metadata",
]

# Decode the exact tagged iPhone HLG format to linear light, apply a
# deterministic SDR tone map, then encode BT.709 video-range pixels. The final
# consumer selects its own pixel format (for example, raw rgb24 for CV).
HLG_TO_SDR_FILTER = (
    "zscale=pin=bt2020:tin=arib-std-b67:min=bt2020nc:rin=tv:"
    "transfer=linear:npl=100,format=gbrpf32le,"
    "tonemap=tonemap=hable:desat=0,"
    "zscale=primaries=bt709:transfer=bt709:matrix=bt709:range=tv"
)


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
    def sdr_filter(self) -> str | None:
        """Return the required HLG-to-SDR filter, or None for SDR/unknown."""
        return HLG_TO_SDR_FILTER if self.is_bt2020_hlg else None


def build_color_probe_args(video: Path | str, *, ffprobe: str = "ffprobe") -> list[str]:
    """Build the narrow ffprobe command used to classify decoded pixels."""
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
        raise FrameError("color metadata must contain exactly one selected video stream.")
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


def probe_color_metadata(video: Path | str, *, ffprobe: str = "ffprobe") -> ColorMetadata:
    """Read source color tags without decoding or modifying the source."""
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
