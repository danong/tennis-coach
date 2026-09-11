"""Deterministic synthetic video fixture generator (M1.3 test helper).

This module lives under ``tests/`` and is not production code. It builds
tiny portrait and landscape MOV fixtures in temporary directories using a
local FFmpeg with subprocess argument arrays only.

Design notes:

- The source is ``testsrc2``, which burns in a moving test pattern with a
  running timestamp, so every frame carries visibly distinguishable
  timestamp identity without extra dependencies.
- Per-segment identity (for later multi-segment export tests) comes from a
  deterministic ``hue`` rotation plus a filled ``drawbox`` corner patch in
  a segment-specific color. ``drawtext`` is intentionally avoided: the
  pinned FFmpeg builds may not ship ``libfreetype``, so text burn-in is
  not a portable fixture primitive.
- Command construction is a pure function of the spec: the same inputs
  always produce the same argument array.
- Fixtures are bounded: durations are at most ``MAX_DURATION_SECONDS`` and
  no dimension exceeds ``MAX_DIMENSION`` pixels.
- Callers must pass paths inside temporary directories (for example the
  ``tmp_path`` pytest fixture). Nothing is written outside the given path.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AUDIO_CHANNELS",
    "AUDIO_SAMPLE_RATES",
    "DEFAULT_AUDIO_CHANNELS",
    "DEFAULT_AUDIO_SAMPLE_RATE_HZ",
    "DEFAULT_DURATION_SECONDS",
    "DEFAULT_FPS",
    "FixtureError",
    "FixtureSpec",
    "LANDSCAPE_SIZE",
    "MAX_DIMENSION",
    "MAX_DURATION_SECONDS",
    "MIN_DIMENSION",
    "PORTRAIT_SIZE",
    "SEGMENT_COLORS",
    "SEGMENT_HUE_STEP_DEGREES",
    "audio_input_spec",
    "build_av_fixture_args",
    "build_ffmpeg_args",
    "extract_frame_bytes",
    "ffmpeg_available",
    "filter_graph",
    "generate_av_fixture",
    "generate_fixture",
    "landscape_spec",
    "make_fixture",
    "portrait_spec",
    "segment_color",
    "segment_hue_degrees",
    "source_input_spec",
    "spec_for_orientation",
]

PORTRAIT_SIZE: tuple[int, int] = (240, 320)
LANDSCAPE_SIZE: tuple[int, int] = (320, 240)
DEFAULT_FPS = 30
DEFAULT_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 2.0
MAX_DIMENSION = 320
MIN_DIMENSION = 16
MAX_FPS = 30

#: Corner-patch colors cycling per segment index. All are named FFmpeg colors.
SEGMENT_COLORS: tuple[str, ...] = (
    "red",
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "white",
    "orange",
)
SEGMENT_HUE_STEP_DEGREES = 90


class FixtureError(Exception):
    """Actionable failure to construct or generate a synthetic fixture."""


@dataclass(frozen=True)
class FixtureSpec:
    """Immutable, validated description of one synthetic fixture."""

    width: int
    height: int
    duration_seconds: float = DEFAULT_DURATION_SECONDS
    fps: int = DEFAULT_FPS
    segment_index: int = 0

    def __post_init__(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise FixtureError(
                    f"invalid fixture {name}: {value!r}; "
                    "expected an integer number of pixels."
                )
            if value < MIN_DIMENSION or value > MAX_DIMENSION:
                raise FixtureError(
                    f"invalid fixture {name}: {value!r}; "
                    f"expected {MIN_DIMENSION}..{MAX_DIMENSION} pixels."
                )
            if value % 2 != 0:
                raise FixtureError(
                    f"invalid fixture {name}: {value!r}; "
                    "expected an even dimension for yuv420p output."
                )
        duration = self.duration_seconds
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not (0 < float(duration) <= MAX_DURATION_SECONDS)
        ):
            raise FixtureError(
                f"invalid fixture duration: {duration!r}; "
                f"expected 0 < duration <= {MAX_DURATION_SECONDS}s."
            )
        if isinstance(self.fps, bool) or not isinstance(self.fps, int):
            raise FixtureError(
                f"invalid fixture fps: {self.fps!r}; expected an integer frame rate."
            )
        if self.fps <= 0 or self.fps > MAX_FPS:
            raise FixtureError(
                f"invalid fixture fps: {self.fps!r}; "
                f"expected 1..{MAX_FPS} fps."
            )
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or self.segment_index < 0
        ):
            raise FixtureError(
                f"invalid fixture segment_index: {self.segment_index!r}; "
                "expected a non-negative integer."
            )


def spec_for_orientation(
    orientation: str,
    *,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    fps: int = DEFAULT_FPS,
    segment_index: int = 0,
) -> FixtureSpec:
    """Return the bounded spec for the ``"portrait"``/``"landscape"`` preset."""
    if orientation == "portrait":
        width, height = PORTRAIT_SIZE
    elif orientation == "landscape":
        width, height = LANDSCAPE_SIZE
    else:
        raise FixtureError(
            f"invalid fixture orientation: {orientation!r}; "
            "expected 'portrait' or 'landscape'."
        )
    return FixtureSpec(
        width=width,
        height=height,
        duration_seconds=duration_seconds,
        fps=fps,
        segment_index=segment_index,
    )


def portrait_spec(
    segment_index: int = 0,
    *,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    fps: int = DEFAULT_FPS,
) -> FixtureSpec:
    """Return the tiny portrait fixture spec for ``segment_index``."""
    return spec_for_orientation(
        "portrait",
        duration_seconds=duration_seconds,
        fps=fps,
        segment_index=segment_index,
    )


def landscape_spec(
    segment_index: int = 0,
    *,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    fps: int = DEFAULT_FPS,
) -> FixtureSpec:
    """Return the tiny landscape fixture spec for ``segment_index``."""
    return spec_for_orientation(
        "landscape",
        duration_seconds=duration_seconds,
        fps=fps,
        segment_index=segment_index,
    )


def segment_hue_degrees(segment_index: int) -> int:
    """Return the deterministic hue rotation for ``segment_index``."""
    if (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or segment_index < 0
    ):
        raise FixtureError(
            f"invalid fixture segment_index: {segment_index!r}; "
            "expected a non-negative integer."
        )
    return (segment_index * SEGMENT_HUE_STEP_DEGREES) % 360


def segment_color(segment_index: int) -> str:
    """Return the deterministic corner-patch color for ``segment_index``."""
    if (
        isinstance(segment_index, bool)
        or not isinstance(segment_index, int)
        or segment_index < 0
    ):
        raise FixtureError(
            f"invalid fixture segment_index: {segment_index!r}; "
            "expected a non-negative integer."
        )
    return SEGMENT_COLORS[segment_index % len(SEGMENT_COLORS)]


def _format_seconds(value: float) -> str:
    return f"{float(value):.3f}"


def source_input_spec(spec: FixtureSpec) -> str:
    """Return the deterministic ``testsrc2`` lavfi input description."""
    if not isinstance(spec, FixtureSpec):
        raise FixtureError(
            "invalid fixture spec: "
            f"expected FixtureSpec, got {type(spec).__name__}."
        )
    return (
        f"testsrc2=size={spec.width}x{spec.height}"
        f":rate={spec.fps}"
        f":duration={_format_seconds(spec.duration_seconds)}"
    )


def _corner_box_size(spec: FixtureSpec) -> tuple[int, int]:
    side = max(16, min(spec.width, spec.height) // 4)
    side -= side % 2  # keep even for yuv420p
    return (side, side)


def filter_graph(spec: FixtureSpec) -> str:
    """Return the deterministic ``-vf`` filter graph for ``spec``."""
    if not isinstance(spec, FixtureSpec):
        raise FixtureError(
            "invalid fixture spec: "
            f"expected FixtureSpec, got {type(spec).__name__}."
        )
    box_w, box_h = _corner_box_size(spec)
    return (
        f"hue=h={segment_hue_degrees(spec.segment_index)}:s=1,"
        f"drawbox=x=0:y=0:w={box_w}:h={box_h}"
        f":color={segment_color(spec.segment_index)}:t=fill"
    )


def build_ffmpeg_args(
    output_path: Path | str,
    spec: FixtureSpec,
    *,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the deterministic FFmpeg argument array for ``spec``.

    The result is a list of strings suitable for :func:`subprocess.run`
    without ``shell=True``. The same ``(output_path, spec, ffmpeg)``
    inputs always produce the same array.
    """
    if not isinstance(spec, FixtureSpec):
        raise FixtureError(
            "invalid fixture spec: "
            f"expected FixtureSpec, got {type(spec).__name__}."
        )
    if not isinstance(ffmpeg, str) or not ffmpeg:
        raise FixtureError(
            f"invalid ffmpeg executable: {ffmpeg!r}; expected a non-blank string."
        )
    output = str(output_path)
    if not output:
        raise FixtureError("invalid fixture output path: expected a non-blank path.")
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        source_input_spec(spec),
        "-vf",
        filter_graph(spec),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-t",
        _format_seconds(spec.duration_seconds),
        output,
    ]


def ffmpeg_available(executable: str = "ffmpeg") -> bool:
    """Return True when ``executable`` resolves on PATH or as a file path."""
    if not isinstance(executable, str) or not executable:
        return False
    if "/" in executable or "\\" in executable:
        return Path(executable).is_file()
    return shutil.which(executable) is not None


def _missing_tool_error(ffmpeg: str) -> FixtureError:
    return FixtureError(
        f"ffmpeg was not found as {ffmpeg!r}; "
        "install FFmpeg and ensure it is on PATH."
    )


def generate_fixture(
    output_path: Path | str,
    spec: FixtureSpec,
    *,
    ffmpeg: str = "ffmpeg",
    timeout_seconds: float = 120.0,
) -> Path:
    """Generate the fixture video at ``output_path`` and return its path.

    Runs FFmpeg with an argument array. On failure any partial output is
    removed and a :class:`FixtureError` is raised. The input is synthetic;
    no source media is read or modified.
    """
    if not isinstance(spec, FixtureSpec):
        raise FixtureError(
            "invalid fixture spec: "
            f"expected FixtureSpec, got {type(spec).__name__}."
        )
    output = Path(output_path).expanduser()
    if not str(output):
        raise FixtureError("invalid fixture output path: expected a non-blank path.")
    if "/" in ffmpeg or "\\" in ffmpeg:
        if not Path(ffmpeg).is_file():
            raise _missing_tool_error(ffmpeg)
    elif shutil.which(ffmpeg) is None:
        raise _missing_tool_error(ffmpeg)
    parent = output.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FixtureError(
                f"could not create fixture directory {parent}: {exc}."
            ) from exc
    args = build_ffmpeg_args(output, spec, ffmpeg=ffmpeg)
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise _missing_tool_error(ffmpeg) from exc
    except OSError as exc:
        raise FixtureError(f"could not run ffmpeg as {ffmpeg!r}: {exc}.") from exc
    except subprocess.TimeoutExpired as exc:
        _remove_partial(output)
        raise FixtureError(
            f"ffmpeg timed out generating fixture {output} "
            f"after {timeout_seconds}s."
        ) from exc
    if completed.returncode != 0:
        _remove_partial(output)
        detail = ((completed.stderr or "").strip() or (completed.stdout or "").strip())
        suffix = f": {detail}" if detail else ": no output"
        raise FixtureError(
            f"ffmpeg failed generating fixture {output} "
            f"(exit {completed.returncode}){suffix}."
        )
    if not output.is_file() or output.stat().st_size == 0:
        _remove_partial(output)
        raise FixtureError(
            f"ffmpeg did not produce fixture output at {output}."
        )
    return output


def make_fixture(
    output_path: Path | str,
    orientation: str,
    *,
    segment_index: int = 0,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    fps: int = DEFAULT_FPS,
    ffmpeg: str = "ffmpeg",
    timeout_seconds: float = 120.0,
) -> Path:
    """Generate the ``orientation`` preset fixture at ``output_path``."""
    spec = spec_for_orientation(
        orientation,
        duration_seconds=duration_seconds,
        fps=fps,
        segment_index=segment_index,
    )
    return generate_fixture(
        output_path, spec, ffmpeg=ffmpeg, timeout_seconds=timeout_seconds
    )


def extract_frame_bytes(
    video_path: Path | str,
    time_seconds: float,
    *,
    ffmpeg: str = "ffmpeg",
    timeout_seconds: float = 60.0,
) -> bytes:
    """Decode one RGB frame at ``time_seconds`` and return raw ``rgb24`` bytes.

    Used to prove timestamp/segment identity: frames taken at different
    source times, or from different segment fixtures, must differ visibly.
    """
    video = Path(video_path).expanduser()
    if not video.is_file():
        raise FixtureError(f"fixture video does not exist: {video}.")
    if (
        isinstance(time_seconds, bool)
        or not isinstance(time_seconds, (int, float))
        or float(time_seconds) < 0
    ):
        raise FixtureError(
            f"invalid frame time: {time_seconds!r}; "
            "expected seconds >= 0."
        )
    if "/" in ffmpeg or "\\" in ffmpeg:
        if not Path(ffmpeg).is_file():
            raise _missing_tool_error(ffmpeg)
    elif shutil.which(ffmpeg) is None:
        raise _missing_tool_error(ffmpeg)
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        _format_seconds(time_seconds),
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise _missing_tool_error(ffmpeg) from exc
    except OSError as exc:
        raise FixtureError(f"could not run ffmpeg as {ffmpeg!r}: {exc}.") from exc
    except subprocess.TimeoutExpired as exc:
        raise FixtureError(
            f"ffmpeg timed out extracting a frame from {video}."
        ) from exc
    if completed.returncode != 0:
        detail = ((completed.stderr or b"").decode("utf-8", "replace")).strip()
        suffix = f": {detail}" if detail else ""
        raise FixtureError(
            f"ffmpeg failed extracting a frame from {video} "
            f"(exit {completed.returncode}){suffix}."
        )
    if not completed.stdout:
        raise FixtureError(
            f"ffmpeg produced no frame bytes for {video} at t={time_seconds}s."
        )
    return bytes(completed.stdout)


def _remove_partial(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


#: Sample rates covered by audio fixtures (raw slow-motion MOVs carry
#: 44.1/48 kHz audio alongside high-rate video).
AUDIO_SAMPLE_RATES: tuple[int, ...] = (44100, 48000)
#: Channel counts covered by audio fixtures.
AUDIO_CHANNELS: tuple[int, ...] = (1, 2)
DEFAULT_AUDIO_SAMPLE_RATE_HZ = 44100
DEFAULT_AUDIO_CHANNELS = 1


def audio_input_spec(
    audio_expr: str,
    *,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
) -> str:
    """Return the deterministic ``aevalsrc`` lavfi input description.

    ``audio_expr`` is one ``aevalsrc`` expression for mono, or one
    ``|``-separated expression per channel for stereo. The expression
    may use the ``t`` (seconds) variable, so clicks/tones sit at known
    source times. Single quotes are rejected: the expression is wrapped
    in quotes for lavfi parsing and a quote would break determinism.
    """
    if not isinstance(audio_expr, str) or not audio_expr.strip():
        raise FixtureError(
            "invalid audio expression: expected a non-blank aevalsrc "
            f"expression, got {audio_expr!r}."
        )
    if "'" in audio_expr:
        raise FixtureError(
            "invalid audio expression: single quotes are not allowed "
            f"in {audio_expr!r}."
        )
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz not in AUDIO_SAMPLE_RATES
    ):
        raise FixtureError(
            f"invalid audio sample rate: {sample_rate_hz!r}; "
            f"expected one of {list(AUDIO_SAMPLE_RATES)}."
        )
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not (0 < float(duration_seconds) <= MAX_DURATION_SECONDS)
    ):
        raise FixtureError(
            f"invalid audio duration: {duration_seconds!r}; "
            f"expected 0 < duration <= {MAX_DURATION_SECONDS}s."
        )
    return (
        f"aevalsrc='{audio_expr.strip()}'"
        f":s={sample_rate_hz}"
        f":d={_format_seconds(duration_seconds)}"
    )


def build_av_fixture_args(
    output_path: Path | str,
    spec: FixtureSpec,
    *,
    audio_expr: str,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    channels: int = DEFAULT_AUDIO_CHANNELS,
    audio_codec: str = "aac",
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the deterministic FFmpeg argument array for one A/V fixture.

    The synthetic video track follows ``spec`` (``testsrc2`` plus the
    deterministic segment filter); the audio track is the ``audio_expr``
    signal encoded at ``sample_rate_hz``/``channels``. The same inputs
    always produce the same array; no shell interpolation is used.
    """
    if not isinstance(spec, FixtureSpec):
        raise FixtureError(
            "invalid fixture spec: "
            f"expected FixtureSpec, got {type(spec).__name__}."
        )
    if not isinstance(ffmpeg, str) or not ffmpeg:
        raise FixtureError(
            f"invalid ffmpeg executable: {ffmpeg!r}; expected a non-blank string."
        )
    if (
        isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels not in AUDIO_CHANNELS
    ):
        raise FixtureError(
            f"invalid audio channels: {channels!r}; "
            f"expected one of {list(AUDIO_CHANNELS)}."
        )
    if not isinstance(audio_codec, str) or not audio_codec.strip():
        raise FixtureError(
            f"invalid audio codec: {audio_codec!r}; expected a non-blank string."
        )
    output = str(output_path)
    if not output:
        raise FixtureError("invalid fixture output path: expected a non-blank path.")
    audio_input = audio_input_spec(
        audio_expr,
        sample_rate_hz=sample_rate_hz,
        duration_seconds=spec.duration_seconds,
    )
    return [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        source_input_spec(spec),
        "-f",
        "lavfi",
        "-i",
        audio_input,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        filter_graph(spec),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        "-c:a",
        audio_codec.strip(),
        "-ar",
        str(sample_rate_hz),
        "-ac",
        str(channels),
        "-shortest",
        "-t",
        _format_seconds(spec.duration_seconds),
        output,
    ]


def generate_av_fixture(
    output_path: Path | str,
    spec: FixtureSpec,
    *,
    audio_expr: str,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
    channels: int = DEFAULT_AUDIO_CHANNELS,
    audio_codec: str = "aac",
    ffmpeg: str = "ffmpeg",
    timeout_seconds: float = 120.0,
) -> Path:
    """Generate one audio+video fixture at ``output_path``.

    Runs FFmpeg with an argument array. On failure any partial output
    is removed and a :class:`FixtureError` is raised. Both tracks are
    synthetic; no source media is read or modified.
    """
    if not isinstance(spec, FixtureSpec):
        raise FixtureError(
            "invalid fixture spec: "
            f"expected FixtureSpec, got {type(spec).__name__}."
        )
    output = Path(output_path).expanduser()
    if not str(output):
        raise FixtureError("invalid fixture output path: expected a non-blank path.")
    if "/" in ffmpeg or "\\" in ffmpeg:
        if not Path(ffmpeg).is_file():
            raise _missing_tool_error(ffmpeg)
    elif shutil.which(ffmpeg) is None:
        raise _missing_tool_error(ffmpeg)
    parent = output.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FixtureError(
                f"could not create fixture directory {parent}: {exc}."
            ) from exc
    args = build_av_fixture_args(
        output,
        spec,
        audio_expr=audio_expr,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        audio_codec=audio_codec,
        ffmpeg=ffmpeg,
    )
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise _missing_tool_error(ffmpeg) from exc
    except OSError as exc:
        raise FixtureError(f"could not run ffmpeg as {ffmpeg!r}: {exc}.") from exc
    except subprocess.TimeoutExpired as exc:
        _remove_partial(output)
        raise FixtureError(
            f"ffmpeg timed out generating fixture {output} "
            f"after {timeout_seconds}s."
        ) from exc
    if completed.returncode != 0:
        _remove_partial(output)
        detail = ((completed.stderr or "").strip() or (completed.stdout or "").strip())
        suffix = f": {detail}" if detail else ": no output"
        raise FixtureError(
            f"ffmpeg failed generating fixture {output} "
            f"(exit {completed.returncode}){suffix}."
        )
    if not output.is_file() or output.stat().st_size == 0:
        _remove_partial(output)
        raise FixtureError(
            f"ffmpeg did not produce fixture output at {output}."
        )
    return output
