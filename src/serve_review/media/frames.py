"""Timestamped RGB frame sampler (M2.1).

Decodes a source video sequentially with the system FFmpeg and yields only
the frames selected by an explicit source-time schedule. Each yielded
:class:`SampledFrame` carries its canonical source timestamp (decimal
seconds derived from integer container timestamps via ffprobe) plus a
MediaPipe-style integer ``timestamp_ms`` convenience. The canonical time is
always ``time_seconds``; ``timestamp_ms`` is ``round(time_seconds * 1000)``
and must never replace the float for range logic.

Decoding uses subprocess argument arrays only; the source file is only
read, never modified. Pose models are never invoked here.

Memory bound (documented): at most ``MAX_BUFFERED_FRAMES`` (one) decoded
frame -- roughly ``width * height * 3`` bytes -- plus one yielded output
frame is held at a time. Frames stream through an FFmpeg ``rawvideo``
pipe; the whole video is never retained. The ffprobe per-frame timestamp
list is small floats (about 8 bytes per source frame) and is the only
preloaded structure.

Rotation normalization: stored pixels are rotated into the upright display
orientation using the probe ``rotation_degrees`` (counter-clockwise
display-rotation degrees, matching the FFmpeg display-matrix /
autorotate / export convention: ``-display_rotation`` sets a pure
counter-clockwise rotation). ``0`` is identity, ``90`` rotates 90 degrees
counter-clockwise, ``180`` flips, ``270`` rotates 90 degrees clockwise.
An explicit ``rotation_degrees`` argument overrides probed metadata.

Sampler orientation contract: ``SAMPLER_ORIENTATION_VERSION`` pins this
counter-clockwise convention. Version 1 treated the probed value as
clockwise (90/270 swapped), so upright frames from rotation-90/270
sources came out 180 degrees flipped relative to the FFmpeg
autorotate/export orientation. Any pose cache holding observations
sampled under version 1 from a rotation-90/270 source is stale: its
normalized coordinates sit in the flipped frame and must be
quarantined and re-extracted, never silently reused.

Selection semantics: for each strictly increasing requested source time
``t``, the first decoded frame with canonical time ``>= t - tolerance``
(1 ms, covering ffprobe microsecond rounding plus float error) is
selected; a request at or after the last decoded frame selects that final
frame, which covers through the source duration. Successive requests
mapping to the same decoded frame yield it once, so output canonical
times are strictly increasing and at most one output exists per decoded
frame. Requesting above the source frame rate therefore does not invent
frames. The requested rate only builds a schedule from
``(duration, rate)`` and never uses the nominal FPS.
"""

from __future__ import annotations

import bisect
import json
import math
import shutil
import subprocess
from bisect import bisect_left
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from serve_review.media import probe as probe_module

__all__ = [
    "DEFAULT_SAMPLE_RATE_HZ",
    "FRAME_MATCH_TOLERANCE_SECONDS",
    "MAX_BUFFERED_FRAMES",
    "MAX_SAMPLE_RATE_HZ",
    "SUPPORTED_ROTATIONS",
    "SAMPLER_ORIENTATION_VERSION",
    "FrameCancelled",
    "FrameError",
    "SampledFrame",
    "build_ffprobe_frame_times_args",
    "build_rawvideo_decode_args",
    "build_uniform_schedule",
    "iter_frames_at_rate",
    "iter_sampled_frames",
    "rotate_rgb_frame",
    "validate_schedule",
]

#: Default requested sampling rate; independent of the source nominal FPS.
DEFAULT_SAMPLE_RATE_HZ = 30.0
#: Upper bound for a requested sampling rate in Hz.
MAX_SAMPLE_RATE_HZ = 120.0
#: Maximum number of decoded frames held in memory at once (streaming pipe).
MAX_BUFFERED_FRAMES = 1
#: Tolerance matching a requested time to the next decoded canonical time.
FRAME_MATCH_TOLERANCE_SECONDS = 0.001
#: Supported display-rotation values (counter-clockwise degrees, matching
#: the FFmpeg display-matrix / autorotate / export convention).
SUPPORTED_ROTATIONS = (0, 90, 180, 270)
#: Sampler orientation contract version. Bumped to 2 when the 90/270-degree
#: convention was corrected from clockwise to counter-clockwise (matching
#: FFmpeg autorotate/export). Caches sampled under version 1 from
#: rotation-90/270 sources are 180-degree-flipped and must be
#: quarantined/re-extracted, never silently validated.
SAMPLER_ORIENTATION_VERSION = 2

_DURATION_EPS = 1e-9


class FrameError(Exception):
    """Actionable failure to sample timestamped frames from a source video."""


class FrameCancelled(FrameError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


@dataclass(frozen=True, slots=True, eq=False)
class SampledFrame:
    """One selected RGB frame with its canonical source timestamp.

    ``time_seconds`` is canonical; ``timestamp_ms`` is the
    ``round(time_seconds * 1000)`` convenience for ordered ``VIDEO``
    inference calls. ``image`` is a contiguous ``uint8`` RGB array with
    shape ``(height, width, 3)`` in upright display orientation.
    """

    time_seconds: float
    timestamp_ms: int
    width: int
    height: int
    image: Any

    def __post_init__(self) -> None:
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise FrameError(
                f"invalid frame time_seconds: {self.time_seconds!r}; "
                "expected a finite number of seconds >= 0."
            )
        object.__setattr__(self, "time_seconds", float(self.time_seconds))
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise FrameError(
                f"invalid frame timestamp_ms: {self.timestamp_ms!r}; "
                "expected an integer >= 0."
            )
        for key in ("width", "height"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FrameError(
                    f"invalid frame {key}: {value!r}; expected an integer > 0."
                )
        image = self.image
        if not isinstance(image, np.ndarray):
            raise FrameError(
                "invalid frame image: expected a numpy uint8 RGB array, "
                f"got {type(image).__name__}."
            )
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise FrameError(
                "invalid frame image: expected a uint8 array with shape "
                f"(height, width, 3), got dtype={image.dtype} shape={image.shape}."
            )
        if image.shape[0] != self.height or image.shape[1] != self.width:
            raise FrameError(
                f"invalid frame image shape {image.shape!r}: does not match "
                f"(height={self.height}, width={self.width})."
            )

    @property
    def nbytes(self) -> int:
        """Return the RGB byte count of the yielded image."""
        return int(self.image.nbytes)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"SampledFrame(time_seconds={self.time_seconds!r}, "
            f"timestamp_ms={self.timestamp_ms!r}, "
            f"width={self.width!r}, height={self.height!r}, "
            f"image=<uint8 {(self.height, self.width, 3)}>)"
        )


def _check_executable(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrameError(
            f"invalid {name}: {value!r}; expected a non-blank executable name."
        )
    return value.strip()


def _ensure_tool(executable: str) -> None:
    if "/" in executable or "\\" in executable:
        if not Path(executable).is_file():
            raise FrameError(
                f"{executable!r} was not found; install FFmpeg and ensure "
                "it is on PATH."
            )
    elif shutil.which(executable) is None:
        raise FrameError(
            f"{executable!r} was not found; install FFmpeg and ensure "
            "it is on PATH."
        )


def _check_duration(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise FrameError(
            f"invalid duration_seconds: {value!r}; "
            "expected a finite number greater than zero."
        )
    return float(value)


def _check_rate(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= MAX_SAMPLE_RATE_HZ
    ):
        raise FrameError(
            f"invalid rate_hz: {value!r}; expected a finite rate in "
            f"(0, {MAX_SAMPLE_RATE_HZ}]."
        )
    return float(value)


def _check_rotation(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrameError(
            f"invalid rotation_degrees: {value!r}; "
            f"expected one of {list(SUPPORTED_ROTATIONS)}."
        )
    if value not in SUPPORTED_ROTATIONS:
        raise FrameError(
            f"unsupported rotation_degrees: {value!r}; "
            f"supported degrees are {list(SUPPORTED_ROTATIONS)}."
        )
    return value


def build_uniform_schedule(
    duration_seconds: float,
    rate_hz: float,
    *,
    start_seconds: float = 0.0,
) -> tuple[float, ...]:
    """Build a uniform source-time schedule for ``[start, duration)``.

    The schedule depends only on ``(start_seconds, duration_seconds,
    rate_hz)`` -- never on the source nominal FPS -- so a requested 30 or
    60 Hz means the same source-time grid for any input. Points are
    ``start_seconds + k / rate_hz`` while strictly below ``duration``;
    ``duration`` itself is excluded (half-open source timeline).
    """
    duration = _check_duration(duration_seconds)
    rate = _check_rate(rate_hz)
    if (
        isinstance(start_seconds, bool)
        or not isinstance(start_seconds, (int, float))
        or not math.isfinite(float(start_seconds))
        or float(start_seconds) < 0
        or float(start_seconds) >= duration
    ):
        raise FrameError(
            f"invalid start_seconds: {start_seconds!r}; expected a finite "
            "number with 0 <= start < duration."
        )
    start = float(start_seconds)
    count = int(math.floor((duration - start) * rate - _DURATION_EPS)) + 1
    if count <= 0:
        raise FrameError(
            "empty frame schedule: no uniform sample falls inside "
            f"[start={start}, duration={duration}) at {rate} Hz."
        )
    return tuple(start + index / rate for index in range(count))


def validate_schedule(
    times_seconds: object, duration_seconds: float
) -> tuple[float, ...]:
    """Validate an explicit schedule against the half-open source timeline."""
    duration = _check_duration(duration_seconds)
    if isinstance(times_seconds, (int, float, bool)) or not isinstance(
        times_seconds, (list, tuple)
    ):
        raise FrameError(
            "invalid schedule: expected a non-empty list/tuple of source "
            f"times in seconds, got {type(times_seconds).__name__}."
        )
    items = list(times_seconds)
    if not items:
        raise FrameError(
            "invalid schedule: at least one requested source time is required."
        )
    cleaned: list[float] = []
    for entry in items:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(float(entry))
        ):
            raise FrameError(
                f"invalid schedule time: {entry!r}; "
                "expected a finite number of seconds."
            )
        value = float(entry)
        if value < 0 or value >= duration:
            raise FrameError(
                f"schedule time {value!r} lies outside the source timeline "
                f"[0, {duration}); times must satisfy 0 <= t < duration."
            )
        cleaned.append(value)
    for earlier, later in zip(cleaned, cleaned[1:]):
        if not later > earlier:
            raise FrameError(
                "invalid schedule: requested source times must be strictly "
                f"increasing, got {cleaned!r}."
            )
    return tuple(cleaned)


def rotate_rgb_frame(image: np.ndarray, rotation_degrees: int) -> np.ndarray:
    """Rotate stored ``image`` into upright display orientation.

    ``rotation_degrees`` is the counter-clockwise display rotation reported
    by probing (FFmpeg display-matrix convention, as applied by FFmpeg
    autorotate and the export path): ``0`` is identity, ``90`` rotates 90
    degrees counter-clockwise, ``180`` flips, ``270`` rotates 90 degrees
    clockwise (i.e. 90 degrees counter-clockwise three times). ``90`` and
    ``270`` swap width and height. The result is a contiguous array.
    """
    rotation = _check_rotation(rotation_degrees)
    if not isinstance(image, np.ndarray):
        raise FrameError(
            "invalid image: expected a numpy RGB array, "
            f"got {type(image).__name__}."
        )
    if rotation == 0:
        return np.ascontiguousarray(image)
    if rotation == 180:
        return np.ascontiguousarray(np.rot90(image, k=2))
    if rotation == 90:
        # Counter-clockwise quarter turn (FFmpeg display-matrix convention).
        return np.ascontiguousarray(np.rot90(image, k=1))
    return np.ascontiguousarray(np.rot90(image, k=3))


def build_ffprobe_frame_times_args(
    video: Path | str, *, ffprobe: str = "ffprobe"
) -> list[str]:
    """Build the ffprobe argument array listing per-frame canonical times."""
    executable = _check_executable("ffprobe", ffprobe)
    source = str(video)
    if not source:
        raise FrameError("invalid video: expected a non-blank path.")
    return [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=pkt_pts_time,best_effort_timestamp_time",
        "-of",
        "json",
        source,
    ]


def build_rawvideo_decode_args(
    video: Path | str, *, ffmpeg: str = "ffmpeg", rate_hz: float | None = None
) -> list[str]:
    """Build the FFmpeg argument array decoding stored frames to ``rgb24``.

    Uses ``-fps_mode passthrough`` so each filter-graph output frame
    crosses the pipe once in decode order. ``-hwaccel videotoolbox``
    (input option, before ``-i``) offloads HEVC decode to the Media
    Engine. ``-noautorotate`` (input option, before ``-i``) disables
    FFmpeg's automatic display-rotation so the pipe always carries
    stored-orientation ``rgb24`` bytes; the caller then applies the
    single manual upright rotation. Without it, rotated phone footage
    would arrive pre-rotated and a second manual rotation would
    double-rotate/garble the pose input.

    When ``rate_hz`` is given, an output ``-vf fps=<rate>:round=up``
    stage drops frames in FFmpeg so only sampled-rate frames cross the
    pipe. ``round=up`` selects the first source frame at or after each
    output timestamp, matching the sampler selection rule (first decoded
    frame ``>= t - tolerance``); the default ``round=near`` picks a
    neighbouring frame (up to half a source interval away) and breaks
    the 1 ms match tolerance on non-divisible grids. ``rate_hz`` must be
    a valid sampling rate; ``None`` keeps 1:1 passthrough for arbitrary
    schedules. The same inputs always produce the same array; no shell
    interpolation is used.
    """
    executable = _check_executable("ffmpeg", ffmpeg)
    source = str(video)
    if not source:
        raise FrameError("invalid video: expected a non-blank path.")
    filt: list[str] = []
    if rate_hz is not None:
        rate = _check_rate(rate_hz)
        filt = ["-vf", f"fps={rate:g}:round=up"]
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "videotoolbox",
        "-noautorotate",
        "-i",
        source,
        "-map",
        "0:v:0",
        *filt,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-fps_mode",
        "passthrough",
        "-",
    ]


def _resolve_metadata(
    video_path: Path, source: object, ffprobe: str
) -> Any:
    from serve_review.domain import SourceMetadata as _SourceMetadata

    if source is None:
        try:
            return probe_module.probe_source(video_path, ffprobe=ffprobe)
        except Exception as exc:
            raise FrameError(
                f"could not probe source video {video_path}: {exc}."
            ) from exc
    if not isinstance(source, _SourceMetadata):
        raise FrameError(
            "invalid source metadata: expected SourceMetadata, "
            f"got {type(source).__name__}."
        )
    try:
        fingerprint = probe_module.fingerprint_for_path(video_path)
    except Exception as exc:
        raise FrameError(f"could not read input video {video_path}: {exc}.") from exc
    if fingerprint != source.fingerprint:
        raise FrameError(
            "source metadata fingerprint does not match the video file; "
            "re-probe the source before sampling frames."
        )
    return source


def _decode_frame_times(video_path: Path, ffprobe: str) -> list[float]:
    _ensure_tool(ffprobe)
    args = build_ffprobe_frame_times_args(video_path, ffprobe=ffprobe)
    try:
        completed = subprocess.run(
            args, capture_output=True, text=True, check=False
        )
    except FileNotFoundError as exc:
        raise FrameError(
            f"{ffprobe!r} was not found; install FFmpeg and ensure it is "
            "on PATH."
        ) from exc
    except OSError as exc:
        raise FrameError(f"could not run ffprobe as {ffprobe!r}: {exc}.") from exc
    if completed.returncode != 0:
        detail = ((completed.stderr or "").strip() or (completed.stdout or "").strip())
        suffix = f": {detail}" if detail else ": no output"
        raise FrameError(
            f"ffprobe failed listing frames for {video_path} "
            f"(exit {completed.returncode}){suffix}."
        )
    try:
        payload = json.loads(completed.stdout or "")
    except json.JSONDecodeError as exc:
        raise FrameError(
            f"ffprobe returned malformed JSON for {video_path}: {exc}."
        ) from exc
    if not isinstance(payload, dict):
        raise FrameError(
            f"ffprobe returned malformed output for {video_path}: "
            "JSON object is required."
        )
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise FrameError(
            f"ffprobe found no decodable video frames in {video_path}."
        )
    times: list[float] = []
    for index, entry in enumerate(raw_frames):
        if not isinstance(entry, dict):
            raise FrameError(
                f"ffprobe returned a malformed frame entry at index {index} "
                f"for {video_path}."
            )
        raw = entry.get("pkt_pts_time")
        if raw is None:
            raw = entry.get("best_effort_timestamp_time")
        if raw is None:
            raise FrameError(
                f"ffprobe frame {index} for {video_path} carries no "
                "presentation timestamp."
            )
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise FrameError(
                f"ffprobe frame {index} for {video_path} carries a malformed "
                f"timestamp {raw!r}."
            ) from exc
        if not math.isfinite(value) or value < 0:
            raise FrameError(
                f"ffprobe frame {index} for {video_path} carries an invalid "
                f"timestamp {raw!r}."
            )
        times.append(value)
    for earlier, later in zip(times, times[1:]):
        if later + 1e-9 < earlier:
            raise FrameError(
                f"ffprobe frame timestamps for {video_path} are not in "
                "decode order."
            )
    return times


def _select_frame_indices(
    frame_times: list[float], schedule: tuple[float, ...], video_path: Path
) -> list[int]:
    selected: list[int] = []
    for target in schedule:
        index = bisect_left(frame_times, target - FRAME_MATCH_TOLERANCE_SECONDS)
        if index >= len(frame_times):
            # The final decoded frame covers through the source duration,
            # so tail requests select it (deduped to a single output).
            index = len(frame_times) - 1
        if selected and index <= selected[-1]:
            continue  # same decoded frame as an earlier request: emit once.
        selected.append(index)
    return selected


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _infer_uniform_rate(schedule: tuple[float, ...]) -> float | None:
    """Infer the uniform sampling rate of ``schedule`` if evenly spaced."""
    if len(schedule) < 2:
        return None
    interval = schedule[1] - schedule[0]
    if not math.isfinite(interval) or interval <= 0:
        return None
    for earlier, later in zip(schedule, schedule[1:]):
        if abs((later - earlier) - interval) > 1e-9:
            return None
    rate = 1.0 / interval
    if not math.isfinite(rate) or not 0 < rate <= MAX_SAMPLE_RATE_HZ:
        return None
    return rate


def _select_filter_rate(
    schedule: tuple[float, ...],
    frame_times: list[float],
    selected: list[int],
) -> float | None:
    """Decide the ffmpeg ``fps`` filter rate, or ``None`` for passthrough.

    Returns the inferred uniform rate only for the safe downsampling
    case: an evenly spaced schedule starting at the timeline origin
    whose point count does not exceed the decoded source frame count
    and whose selection maps 1:1 (no dedup). Every other shape --
    arbitrary schedules, non-zero starts, single points, or requested
    rates above the source density (which would duplicate frames and
    invent output) -- returns ``None`` so decoding stays bit-identical
    passthrough with unchanged selection semantics.
    """
    inferred = _infer_uniform_rate(schedule)
    if inferred is None:
        return None
    if abs(schedule[0]) > 1e-9:
        return None
    if len(schedule) > len(frame_times):
        return None
    if len(selected) != len(schedule):
        return None
    try:
        return _check_rate(inferred)
    except FrameError:
        return None


_HWACCEL_FAILURE_MARKERS = (
    "videotoolbox",
    "hwaccel",
    "hardware device",
    "no device available",
    "device creation failed",
    "hardware device setup failed",
)


def _hwaccel_failure_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _HWACCEL_FAILURE_MARKERS)


def _short_read_error(
    *, video_path: Path, done: int, expected: int, detail: str, filtered: bool
) -> FrameError:
    suffix = f": {detail}" if detail else ""
    if detail and _hwaccel_failure_text(detail):
        return FrameError(
            f"VideoToolbox hardware decode failed for {video_path} "
            f"after {done} of {expected} expected frames{suffix}; "
            "ensure this FFmpeg build supports `-hwaccel videotoolbox`, "
            "the source codec is Media Engine compatible, and retry. "
            "No software fallback was attempted; refusing to emit "
            "silently re-decoded output."
        )
    kind = "fps-filtered" if filtered else "passthrough"
    if filtered:
        return FrameError(
            f"ffmpeg produced only {done} of {expected} expected "
            f"({kind}) frames for {video_path}{suffix}."
        )
    return FrameError(
        f"ffmpeg produced only {done} of {expected} expected frames "
        f"for {video_path}{suffix}."
    )


def _check_cancelled(is_cancelled: Callable[[], bool] | None, message: str) -> None:
    if is_cancelled is None:
        return
    if not callable(is_cancelled):
        raise FrameError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    if is_cancelled():
        raise FrameCancelled(message)


def _run_selected(
    video_path: Path,
    stored_width: int,
    stored_height: int,
    rotation: int,
    frame_times: list[float],
    selected: list[int],
    ffmpeg: str,
    is_cancelled: Callable[[], bool] | None,
    filter_rate_hz: float | None = None,
) -> Iterator[SampledFrame]:
    _ensure_tool(ffmpeg)
    args = build_rawvideo_decode_args(
        video_path, ffmpeg=ffmpeg, rate_hz=filter_rate_hz
    )
    frame_size = stored_width * stored_height * 3
    if rotation in (90, 270):
        out_width, out_height = stored_height, stored_width
    else:
        out_width, out_height = stored_width, stored_height
    filtered = filter_rate_hz is not None
    wanted = set(selected)
    last_wanted = selected[-1]
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise FrameError(
            f"{ffmpeg!r} was not found; install FFmpeg and ensure it is "
            "on PATH."
        ) from exc
    except OSError as exc:
        raise FrameError(f"could not run ffmpeg as {ffmpeg!r}: {exc}.") from exc
    assert proc.stdout is not None
    try:
        _check_cancelled(is_cancelled, "frame sampling was cancelled before starting.")
        if filtered:
            # fps-filtered pipe: exactly one output per selected source
            # frame in schedule order; canonical times stay the ffprobe
            # source times so output matches the uniform schedule within
            # FRAME_MATCH_TOLERANCE_SECONDS. Bounded to one frame.
            for position, source_index in enumerate(selected):
                _check_cancelled(
                    is_cancelled,
                    f"frame sampling was cancelled at sampled frame "
                    f"{position} of {len(selected)}.",
                )
                raw = _read_exact(proc.stdout, frame_size)
                if len(raw) < frame_size:
                    detail = b""
                    try:
                        assert proc.stderr is not None
                        detail = proc.stderr.read() or b""
                    except OSError:
                        detail = b""
                    text = detail.decode("utf-8", "replace").strip()
                    raise _short_read_error(
                        video_path=video_path,
                        done=position,
                        expected=len(selected),
                        detail=text,
                        filtered=True,
                    )
                stored = (
                    np.frombuffer(raw, dtype=np.uint8)
                    .reshape((stored_height, stored_width, 3))
                    .copy()
                )
                upright = rotate_rgb_frame(stored, rotation)
                moment = frame_times[source_index]
                yield SampledFrame(
                    time_seconds=moment,
                    timestamp_ms=int(round(moment * 1000)),
                    width=out_width,
                    height=out_height,
                    image=np.ascontiguousarray(upright, dtype=np.uint8),
                )
            return
        for index in range(len(frame_times)):
            _check_cancelled(
                is_cancelled,
                f"frame sampling was cancelled at source frame {index} "
                f"of {len(frame_times)}.",
            )
            if index > last_wanted:
                break
            raw = _read_exact(proc.stdout, frame_size)
            if len(raw) < frame_size:
                detail = b""
                try:
                    assert proc.stderr is not None
                    detail = proc.stderr.read() or b""
                except OSError:
                    detail = b""
                text = detail.decode("utf-8", "replace").strip()
                raise _short_read_error(
                    video_path=video_path,
                    done=index,
                    expected=len(frame_times),
                    detail=text,
                    filtered=False,
                )
            if index not in wanted:
                continue
            # ``raw`` is a fresh immutable bytes object per frame; copy
            # into an owned array so each yielded image owns its pixels
            # and no mutable/reused buffer is shared across outputs.
            stored = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape((stored_height, stored_width, 3))
                .copy()
            )
            upright = rotate_rgb_frame(stored, rotation)
            moment = frame_times[index]
            yield SampledFrame(
                time_seconds=moment,
                timestamp_ms=int(round(moment * 1000)),
                width=out_width,
                height=out_height,
                image=np.ascontiguousarray(upright, dtype=np.uint8),
            )
    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except OSError:
            pass
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except OSError:
            pass
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except OSError:
            pass


def iter_sampled_frames(
    video: Path | str,
    times_seconds: list[float] | tuple[float, ...],
    *,
    source: Any = None,
    rotation_degrees: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[SampledFrame]:
    """Yield selected RGB frames in strictly increasing source-time order.

    Args:
        video: Source video path; read only, never modified.
        times_seconds: Explicit strictly increasing schedule within
            ``[0, source_duration)``. The requested times select decoded
            frames; yielded timestamps are the decoded canonical times.
        source: Optional already-probed ``SourceMetadata``; when omitted
            the source is probed. A supplied fingerprint must match the
            file.
        rotation_degrees: Optional display-rotation override (one of
            ``0/90/180/270``); defaults to the probed source rotation.
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        is_cancelled: Optional hook polled before each decoded frame; a
            true return raises :class:`FrameCancelled` and terminates the
            decoder.

    Returns:
        A lazy iterator of :class:`SampledFrame`; decoding starts on
        iteration and holds at most one frame at a time.
    """
    video_path = Path(video).expanduser()
    if not video_path.is_file():
        raise FrameError(f"input video does not exist: {video_path}.")
    ffmpeg_exe = _check_executable("ffmpeg", ffmpeg)
    ffprobe_exe = _check_executable("ffprobe", ffprobe)
    if is_cancelled is not None and not callable(is_cancelled):
        raise FrameError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    metadata = _resolve_metadata(video_path, source, ffprobe_exe)
    try:
        duration = float(metadata.duration_seconds)
        stored_width = int(metadata.width)
        stored_height = int(metadata.height)
        probed_rotation = int(metadata.rotation_degrees)
    except (AttributeError, TypeError, ValueError) as exc:
        raise FrameError(f"invalid source metadata for {video_path}: {exc}.") from exc
    rotation = probed_rotation if rotation_degrees is None else _check_rotation(
        rotation_degrees
    )
    schedule = validate_schedule(times_seconds, duration)
    frame_times = _decode_frame_times(video_path, ffprobe_exe)
    selected = _select_frame_indices(frame_times, schedule, video_path)
    filter_rate_hz = _select_filter_rate(schedule, frame_times, selected)
    return _run_selected(
        video_path,
        stored_width,
        stored_height,
        rotation,
        frame_times,
        selected,
        ffmpeg_exe,
        is_cancelled,
        filter_rate_hz,
    )


def iter_frames_at_rate(
    video: Path | str,
    rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    *,
    start_seconds: float = 0.0,
    source: Any = None,
    rotation_degrees: int | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[SampledFrame]:
    """Yield frames on a uniform ``rate_hz`` schedule from ``start_seconds``.

    The schedule is built from ``(source_duration, rate_hz)`` only, so a
    requested 30 or 60 Hz is independent of the source nominal FPS.
    """
    video_path = Path(video).expanduser()
    if not video_path.is_file():
        raise FrameError(f"input video does not exist: {video_path}.")
    ffmpeg_exe = _check_executable("ffmpeg", ffmpeg)
    ffprobe_exe = _check_executable("ffprobe", ffprobe)
    rate = _check_rate(rate_hz)
    metadata = _resolve_metadata(video_path, source, ffprobe_exe)
    try:
        duration = float(metadata.duration_seconds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise FrameError(f"invalid source metadata for {video_path}: {exc}.") from exc
    schedule = build_uniform_schedule(
        duration, rate, start_seconds=start_seconds
    )
    return iter_sampled_frames(
        video_path,
        schedule,
        source=metadata,
        rotation_degrees=rotation_degrees,
        ffmpeg=ffmpeg_exe,
        ffprobe=ffprobe_exe,
        is_cancelled=is_cancelled,
    )
