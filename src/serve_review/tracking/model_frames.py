"""RacketVision model-frame decoding."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Final

from serve_review.media.color import (
    ColorMetadata,
    HLG_TO_SDR_FILTER,
    build_color_probe_args,
    parse_color_metadata,
    probe_color_metadata,
)
from serve_review.media.frames import SampledFrame, iter_sampled_frames

__all__ = [
    "RACKETVISION_HLG_FILTER",
    "ColorMetadata",
    "build_color_probe_args",
    "iter_racketvision_model_frames",
    "parse_color_metadata",
    "probe_color_metadata",
]

# Compatibility name for the tracking-specific wrapper.
RACKETVISION_HLG_FILTER = HLG_TO_SDR_FILTER
_UNSET_FILTER: Final = object()


def iter_racketvision_model_frames(
    video: Path | str,
    times_seconds: list[float] | tuple[float, ...],
    *,
    source: Any = None,
    rotation_degrees: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    is_cancelled: Callable[[], bool] | None = None,
    filter_expression: str | None | object = _UNSET_FILTER,
) -> Iterator[SampledFrame]:
    """Yield upright exact-PTS frames using a supplied or probed SDR filter."""
    if filter_expression is _UNSET_FILTER:
        filter_expression = probe_color_metadata(video, ffprobe=ffprobe).sdr_filter
    return iter_sampled_frames(
        video,
        times_seconds,
        source=source,
        rotation_degrees=rotation_degrees,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        is_cancelled=is_cancelled,
        filter_expression=filter_expression,
    )
