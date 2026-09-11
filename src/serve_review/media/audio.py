"""Video-aligned audio transient feature sampler.

Decodes the raw audio track directly from the source file with the system
FFmpeg and computes band-limited RMS energy aligned to explicit
video-frame source timestamps supplied by the caller. The output
``A_t`` series is keyed by canonical source ``time_seconds``, matching
the pose-frame time base from :mod:`serve_review.media.frames`.

Pipeline (decode-time only; the source file is only read, never
modified):

1. the audio stream is demuxed straight from the source container with
   ``-map 0:a:0`` -- no slow-motion time-stretch, no source re-encode;
2. a 1.0-3.5 kHz bandpass (``highpass=f=1000`` followed by
   ``lowpass=f=3500`` in the FFmpeg filter graph) isolates impact-like
   transients from handling rumble and high-frequency hiss;
3. the filtered mono ``f32le`` stream is converted to RMS energy over a
   short centered window (default 10 ms, covering a 5-15 ms impact
   spike) around each caller-supplied timestamp, using deterministic
   pure-Python arithmetic (no NumPy in this module).

Memory bound (documented): at most ``MAX_BUFFERED_SAMPLES`` decoded
mono samples (one chunk, ``MAX_BUFFERED_SAMPLES * 4`` bytes) plus two
float accumulators per requested timestamp are held at a time. Samples
stream through an FFmpeg ``f32le`` pipe; the whole track is never
retained.

Audio is required for this path, not optional-silent: a source with no
audio stream raises :class:`AudioError`.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve_review.media import probe as probe_module

__all__ = [
    "BANDPASS_HIGH_HZ",
    "BANDPASS_LOW_HZ",
    "DEFAULT_WINDOW_SECONDS",
    "MAX_BUFFERED_SAMPLES",
    "MAX_WINDOW_SECONDS",
    "MIN_WINDOW_SECONDS",
    "AudioCancelled",
    "AudioEnergy",
    "AudioError",
    "AudioStreamInfo",
    "audio_filter_graph",
    "build_audio_decode_args",
    "iter_audio_energy",
    "probe_audio_stream",
    "rms_energy",
    "validate_audio_schedule",
    "validate_window_seconds",
]

#: Bandpass corner frequencies in Hz, applied in the FFmpeg filter graph.
BANDPASS_LOW_HZ = 1000.0
BANDPASS_HIGH_HZ = 3500.0
#: Default RMS window in seconds, centered on each requested timestamp.
DEFAULT_WINDOW_SECONDS = 0.010
#: Smallest accepted RMS window in seconds.
MIN_WINDOW_SECONDS = 0.002
#: Largest accepted RMS window in seconds.
MAX_WINDOW_SECONDS = 0.100
#: Maximum number of decoded mono samples held in memory at once
#: (streaming pipe chunk; ``MAX_BUFFERED_SAMPLES * 4`` bytes of f32le).
MAX_BUFFERED_SAMPLES = 16384

_F32 = struct.Struct("<f")
_CHUNK_BYTES = MAX_BUFFERED_SAMPLES * _F32.size


class AudioError(Exception):
    """Actionable failure to sample audio transient energy."""


class AudioCancelled(AudioError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


@dataclass(frozen=True, slots=True, eq=False)
class AudioEnergy:
    """Band-limited RMS energy at one canonical source timestamp.

    ``time_seconds`` echoes the caller-supplied schedule point (canonical
    source time, matching the pose-frame time base); ``energy`` is the
    RMS of the 1.0-3.5 kHz filtered mono samples inside the centered
    window, in nominal ``0..1`` full-scale units (values above ``1``
    are possible for hot sources and are preserved, never clipped).
    """

    time_seconds: float
    energy: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise AudioError(
                f"invalid audio time_seconds: {self.time_seconds!r}; "
                "expected a finite number of seconds >= 0."
            )
        object.__setattr__(self, "time_seconds", float(self.time_seconds))
        if (
            isinstance(self.energy, bool)
            or not isinstance(self.energy, (int, float))
            or not math.isfinite(float(self.energy))
            or float(self.energy) < 0
        ):
            raise AudioError(
                f"invalid audio energy: {self.energy!r}; "
                "expected a finite number >= 0."
            )
        object.__setattr__(self, "energy", float(self.energy))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"AudioEnergy(time_seconds={self.time_seconds!r}, "
            f"energy={self.energy!r})"
        )


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    """Probed identity of the source audio track used for decoding."""

    sample_rate_hz: int
    duration_seconds: float
    channels: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise AudioError(
                f"invalid audio sample_rate_hz: {self.sample_rate_hz!r}; "
                "expected an integer > 0."
            )
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or float(self.duration_seconds) <= 0
        ):
            raise AudioError(
                f"invalid audio duration_seconds: {self.duration_seconds!r}; "
                "expected a finite number greater than zero."
            )
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))
        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
            or self.channels <= 0
        ):
            raise AudioError(
                f"invalid audio channels: {self.channels!r}; "
                "expected an integer > 0."
            )


def audio_filter_graph() -> str:
    """Return the deterministic 1.0-3.5 kHz bandpass filter graph.

    A two-pole highpass at ``BANDPASS_LOW_HZ`` followed by a two-pole
    lowpass at ``BANDPASS_HIGH_HZ``. The same inputs always produce the
    same graph; no shell interpolation is used.
    """
    return (
        f"highpass=f={BANDPASS_LOW_HZ:g},lowpass=f={BANDPASS_HIGH_HZ:g}"
    )


def _check_executable(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AudioError(
            f"invalid {name}: {value!r}; expected a non-blank executable name."
        )
    return value.strip()


def _ensure_tool(executable: str) -> None:
    if "/" in executable or "\\" in executable:
        if not Path(executable).is_file():
            raise AudioError(
                f"{executable!r} was not found; install FFmpeg and ensure "
                "it is on PATH."
            )
    elif shutil.which(executable) is None:
        raise AudioError(
            f"{executable!r} was not found; install FFmpeg and ensure "
            "it is on PATH."
        )


def build_audio_decode_args(
    video: Path | str, *, ffmpeg: str = "ffmpeg"
) -> list[str]:
    """Build the FFmpeg argument array demuxing raw source audio.

    Maps the first source audio stream (``-map 0:a:0``) with no seeking
    and no slow-motion processing, applies the 1.0-3.5 kHz bandpass in
    the filter graph, downmixes to mono, and emits native-rate ``f32le``
    samples on stdout. The same inputs always produce the same array;
    no shell interpolation is used.
    """
    executable = _check_executable("ffmpeg", ffmpeg)
    source = str(video)
    if not source:
        raise AudioError("invalid video: expected a non-blank path.")
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source,
        "-map",
        "0:a:0",
        "-af",
        audio_filter_graph(),
        "-ac",
        "1",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]


def rms_energy(samples: Sequence[float]) -> float:
    """Return the deterministic RMS energy of ``samples`` (pure Python).

    Empty input yields ``0.0``. Every sample must be a finite number;
    anything else raises :class:`AudioError`.
    """
    if isinstance(samples, (str, bytes, bytearray)):
        raise AudioError(
            "invalid audio samples: expected a sequence of finite "
            f"numbers, got {type(samples).__name__}."
        )
    try:
        values = list(samples)  # type: ignore[arg-type]
    except TypeError as exc:
        raise AudioError(
            "invalid audio samples: expected a sequence of finite "
            f"numbers, got {type(samples).__name__}."
        ) from exc
    total = 0.0
    count = 0
    for entry in values:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(float(entry))
        ):
            raise AudioError(
                f"invalid audio sample: {entry!r}; "
                "expected a finite number."
            )
        value = float(entry)
        total += value * value
        count += 1
    if count == 0:
        return 0.0
    return math.sqrt(total / count)


def validate_audio_schedule(
    times_seconds: object, duration_seconds: float
) -> tuple[float, ...]:
    """Validate explicit audio sample times against the source timeline."""
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or float(duration_seconds) <= 0
    ):
        raise AudioError(
            f"invalid duration_seconds: {duration_seconds!r}; "
            "expected a finite number greater than zero."
        )
    duration = float(duration_seconds)
    if isinstance(times_seconds, (int, float, bool)) or not isinstance(
        times_seconds, (list, tuple)
    ):
        raise AudioError(
            "invalid audio schedule: expected a non-empty list/tuple of "
            f"source times in seconds, got {type(times_seconds).__name__}."
        )
    items = list(times_seconds)
    if not items:
        raise AudioError(
            "invalid audio schedule: at least one requested source time "
            "is required."
        )
    cleaned: list[float] = []
    for entry in items:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(float(entry))
        ):
            raise AudioError(
                f"invalid audio schedule time: {entry!r}; "
                "expected a finite number of seconds."
            )
        value = float(entry)
        if value < 0 or value >= duration:
            raise AudioError(
                f"audio schedule time {value!r} lies outside the source "
                f"timeline [0, {duration}); times must satisfy "
                "0 <= t < duration."
            )
        cleaned.append(value)
    for earlier, later in zip(cleaned, cleaned[1:]):
        if not later > earlier:
            raise AudioError(
                "invalid audio schedule: requested source times must be "
                f"strictly increasing, got {cleaned!r}."
            )
    return tuple(cleaned)


def validate_window_seconds(value: object) -> float:
    """Validate the centered RMS window in seconds."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not MIN_WINDOW_SECONDS <= float(value) <= MAX_WINDOW_SECONDS
    ):
        raise AudioError(
            f"invalid window_seconds: {value!r}; expected a finite number "
            f"in [{MIN_WINDOW_SECONDS}, {MAX_WINDOW_SECONDS}]."
        )
    return float(value)


def _parse_sample_rate(raw: object, video_path: Path) -> int:
    if isinstance(raw, bool):
        raise AudioError(
            f"malformed audio metadata for {video_path}: sample_rate must "
            f"be a positive integer string, got {raw!r}."
        )
    try:
        rate = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise AudioError(
            f"malformed audio metadata for {video_path}: sample_rate must "
            f"be a positive integer string, got {raw!r}."
        ) from exc
    if rate <= 0:
        raise AudioError(
            f"malformed audio metadata for {video_path}: sample_rate must "
            f"be a positive integer, got {raw!r}."
        )
    return rate


def _parse_audio_duration(
    stream: dict, payload: dict, video_path: Path
) -> float:
    raw: object = stream.get("duration")
    if raw is None:
        fmt = payload.get("format")
        if isinstance(fmt, dict):
            raw = fmt.get("duration")
    if isinstance(raw, bool):
        raise AudioError(
            f"malformed audio metadata for {video_path}: duration must be "
            f"a positive number of seconds, got {raw!r}."
        )
    try:
        duration = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AudioError(
            f"malformed audio metadata for {video_path}: duration must be "
            f"a positive number of seconds, got {raw!r}."
        ) from exc
    if not math.isfinite(duration) or duration <= 0:
        raise AudioError(
            f"malformed audio metadata for {video_path}: duration must be "
            f"a positive finite number of seconds, got {raw!r}."
        )
    return duration


def probe_audio_stream(
    video: Path | str, *, ffprobe: str = "ffprobe"
) -> AudioStreamInfo:
    """Probe the first source audio stream with an ffprobe argument array.

    Raises :class:`AudioError` when the file is missing, probing fails,
    metadata is malformed, or -- because audio is required for this
    path -- no audio stream is present.
    """
    video_path = Path(video).expanduser()
    if not video_path.is_file():
        raise AudioError(f"input video does not exist: {video_path}.")
    ffprobe_exe = _check_executable("ffprobe", ffprobe)
    try:
        payload = probe_module.run_ffprobe(video_path, ffprobe=ffprobe_exe)
    except Exception as exc:
        raise AudioError(
            f"could not probe audio for {video_path}: {exc}."
        ) from exc
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise AudioError(
            f"malformed ffprobe output for {video_path}: 'streams' must "
            "be a list."
        )
    audio = next(
        (
            entry
            for entry in streams
            if isinstance(entry, dict) and entry.get("codec_type") == "audio"
        ),
        None,
    )
    if audio is None:
        raise AudioError(
            f"unsupported media: ffprobe found no audio stream in "
            f"{video_path}; audio is required for this path "
            "(impact-transient analysis cannot run silent)."
        )
    rate = _parse_sample_rate(audio.get("sample_rate"), video_path)
    channels_raw = audio.get("channels", 1)
    if isinstance(channels_raw, bool) or not isinstance(
        channels_raw, int
    ) or channels_raw <= 0:
        raise AudioError(
            f"malformed audio metadata for {video_path}: channels must be "
            f"a positive integer, got {channels_raw!r}."
        )
    duration = _parse_audio_duration(audio, payload, video_path)
    return AudioStreamInfo(
        sample_rate_hz=rate,
        duration_seconds=duration,
        channels=channels_raw,
    )


def _check_cancelled(is_cancelled: Callable[[], bool] | None, message: str) -> None:
    if is_cancelled is None:
        return
    if not callable(is_cancelled):
        raise AudioError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    if is_cancelled():
        raise AudioCancelled(message)


def _window_sample_ranges(
    schedule: tuple[float, ...],
    window_seconds: float,
    sample_rate_hz: int,
    duration_seconds: float,
) -> tuple[tuple[int, int], ...]:
    half = window_seconds / 2.0
    ranges: list[tuple[int, int]] = []
    for moment in schedule:
        low = max(0.0, moment - half)
        high = min(duration_seconds, moment + half)
        start = int(math.floor(low * sample_rate_hz))
        end = int(math.ceil(high * sample_rate_hz))
        if end <= start:
            end = start + 1
        ranges.append((start, end))
    return tuple(ranges)


def _run_energy(
    video_path: Path,
    schedule: tuple[float, ...],
    window_seconds: float,
    info: AudioStreamInfo,
    ffmpeg: str,
    is_cancelled: Callable[[], bool] | None,
) -> Iterator[AudioEnergy]:
    _ensure_tool(ffmpeg)
    args = build_audio_decode_args(video_path, ffmpeg=ffmpeg)
    rate = info.sample_rate_hz
    ranges = _window_sample_ranges(
        schedule, window_seconds, rate, info.duration_seconds
    )
    starts = [start for start, _ in ranges]
    ends = [end for _, end in ranges]
    sums = [0.0] * len(schedule)
    counts = [0] * len(schedule)
    try:
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise AudioError(
            f"{ffmpeg!r} was not found; install FFmpeg and ensure it is "
            "on PATH."
        ) from exc
    except OSError as exc:
        raise AudioError(f"could not run ffmpeg as {ffmpeg!r}: {exc}.") from exc
    assert proc.stdout is not None
    try:
        _check_cancelled(
            is_cancelled, "audio sampling was cancelled before starting."
        )
        active: deque[int] = deque()
        add_ptr = 0
        sample_index = 0
        total = 0
        # Pipe reads may split a 4-byte sample across chunks, so carry
        # leftover bytes forward; only trailing bytes at EOF are torn.
        pending = b""
        while True:
            _check_cancelled(
                is_cancelled,
                f"audio sampling was cancelled at sample {sample_index}.",
            )
            chunk = proc.stdout.read(_CHUNK_BYTES)
            if not chunk:
                break
            data = pending + chunk
            whole = (len(data) // _F32.size) * _F32.size
            pending = data[whole:]
            for (value,) in _F32.iter_unpack(data[:whole]):
                while add_ptr < len(schedule) and starts[add_ptr] <= sample_index:
                    active.append(add_ptr)
                    add_ptr += 1
                while active and ends[active[0]] <= sample_index:
                    active.popleft()
                for window in active:
                    sums[window] += value * value
                    counts[window] += 1
                sample_index += 1
                total += 1
        if pending:
            raise AudioError(
                f"ffmpeg produced a torn audio stream for {video_path}: "
                f"{len(pending)} trailing byte(s) do not form one f32le "
                "sample."
            )
        _check_cancelled(is_cancelled, "audio sampling was cancelled.")
        if total == 0:
            raise AudioError(
                f"ffmpeg decoded no audio samples for {video_path}; "
                "verify the source carries a decodable audio track."
            )
        returncode = proc.wait()
        if returncode != 0:
            detail = b""
            try:
                assert proc.stderr is not None
                detail = proc.stderr.read() or b""
            except OSError:
                detail = b""
            text = detail.decode("utf-8", "replace").strip()
            suffix = f": {text}" if text else ""
            raise AudioError(
                f"ffmpeg failed decoding audio for {video_path} "
                f"(exit {returncode}){suffix}."
            )
        for position, moment in enumerate(schedule):
            if counts[position]:
                energy = math.sqrt(sums[position] / counts[position])
            else:
                energy = 0.0
            yield AudioEnergy(time_seconds=moment, energy=energy)
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


def iter_audio_energy(
    video: Path | str,
    times_seconds: list[float] | tuple[float, ...],
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    is_cancelled: Callable[[], bool] | None = None,
) -> Iterator[AudioEnergy]:
    """Yield band-limited RMS energy at explicit video-frame source times.

    Args:
        video: Source video path; read only, never modified. Its raw
            audio track is demuxed directly (no slow-motion processing).
        times_seconds: Explicit strictly increasing schedule within
            ``[0, audio_duration)`` -- the caller's video-frame source
            timestamps. Output ``time_seconds`` echo these exactly.
        window_seconds: Centered RMS window per timestamp (default
            10 ms, covering a 5-15 ms impact spike).
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        is_cancelled: Optional hook polled per decoded chunk; a true
            return raises :class:`AudioCancelled` and terminates the
            decoder.

    Returns:
        A lazy iterator of :class:`AudioEnergy` in schedule order;
        decoding starts on iteration and holds at most one chunk of
        samples at a time.
    """
    video_path = Path(video).expanduser()
    if not video_path.is_file():
        raise AudioError(f"input video does not exist: {video_path}.")
    ffmpeg_exe = _check_executable("ffmpeg", ffmpeg)
    ffprobe_exe = _check_executable("ffprobe", ffprobe)
    if is_cancelled is not None and not callable(is_cancelled):
        raise AudioError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    _ensure_tool(ffmpeg_exe)
    window = validate_window_seconds(window_seconds)
    info = probe_audio_stream(video_path, ffprobe=ffprobe_exe)
    schedule = validate_audio_schedule(times_seconds, info.duration_seconds)
    return _run_energy(
        video_path,
        schedule,
        window,
        info,
        ffmpeg_exe,
        is_cancelled,
    )
