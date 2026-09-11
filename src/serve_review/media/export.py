"""Individual clip exporter (M1.4).

Each validated :class:`ExportPlan` range is exported as one independently
playable clip read from the original source samples. Trimming is exact and
filter-based (``trim``/``atrim`` with ``setpts`` reset) and always
re-encoded with an explicitly selected video encoder; this module never
uses stream copy (``-c copy``), and callers must not describe its output
as stream-copy or frame-exact.

Behavior:

- The source file is only read, never modified.
- Clip names are deterministic and collision-safe
  (``serve-001.mov``, ``serve-002.mov``, ... in plan order). Pre-existing
  destinations raise :class:`ExportCollisionError` unless
  ``overwrite=True``; the collision check runs before any FFmpeg work.
- Each clip is written to a temporary file beside its final destination
  and atomically renamed with :func:`os.replace` after FFmpeg succeeds.
- Temporary files are removed on FFmpeg failure, timeout, cancellation,
  or keyboard interrupt. A failed clip never leaves a partial final file.
- When the source carries an audio stream, audio is filter-trimmed
  (``atrim``) and encoded with the explicit audio encoder; video-only
  sources produce video-only clips.
- All subprocesses use argument arrays; no shell interpolation.

This module owns FFmpeg invocation only. It never detects attempts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from serve_review.domain import ExportPlan, SourceMetadata
from serve_review.media import probe as probe_module
from serve_review.media.probe import ProbeError

__all__ = [
    "CLIPS_SUBDIR",
    "COMPILATION_FILENAME",
    "DEFAULT_AUDIO_ENCODER",
    "DEFAULT_VIDEO_BITRATE",
    "DEFAULT_VIDEO_BITRATE_BPS",
    "DEFAULT_VIDEO_ENCODER",
    "EXPORT_MODES",
    "SOFTWARE_VIDEO_ENCODER",
    "ExportCancelled",
    "ExportCollisionError",
    "ExportError",
    "build_compilation_ffmpeg_args",
    "build_concat_copy_args",
    "build_ffmpeg_args",
    "clip_filename",
    "concat_join_readiness",
    "export_clips",
    "export_compilation",
    "export_outputs",
]

#: Explicit default video encoder: VideoToolbox hardware H.264.
#: Always re-encoded; never stream-copied. ``libx264`` (see
#: :data:`SOFTWARE_VIDEO_ENCODER`) remains selectable via the existing
#: ``video_encoder`` parameter for the software path.
DEFAULT_VIDEO_ENCODER = "h264_videotoolbox"
#: Software H.264 encoder kept selectable via ``video_encoder``.
SOFTWARE_VIDEO_ENCODER = "libx264"
#: Explicit default target video bitrate for the hardware encoder.
#: Review-band target: 6 Mbps lies inside the 4-8 Mbps review band
#: (see ``REVIEW_VIDEO_BITRATE_MIN_BPS``/``REVIEW_VIDEO_BITRATE_MAX_BPS``).
DEFAULT_VIDEO_BITRATE = "6M"
#: Default target bitrate in bits per second (matches ``DEFAULT_VIDEO_BITRATE``).
DEFAULT_VIDEO_BITRATE_BPS = 6_000_000
#: Lower bound of the review-band bitrate window (bits per second).
REVIEW_VIDEO_BITRATE_MIN_BPS = 4_000_000
#: Upper bound of the review-band bitrate window (bits per second).
REVIEW_VIDEO_BITRATE_MAX_BPS = 8_000_000
#: Explicit audio encoder used only when the source has an audio stream.
DEFAULT_AUDIO_ENCODER = "aac"
#: Compilation file name written inside an output directory.
COMPILATION_FILENAME = "serves.mov"
#: Subdirectory holding per-range clips under an output directory.
CLIPS_SUBDIR = "clips"
#: Supported output modes for :func:`export_outputs` and the export CLI.
EXPORT_MODES = ("compilation", "clips", "both")

_DURATION_TOLERANCE_SECONDS = 1e-6


class ExportError(Exception):
    """Actionable failure to export clips from a source video."""


class ExportCollisionError(ExportError):
    """Raised when a clip destination already exists and ``overwrite`` is False."""


class ExportCancelled(ExportError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


def clip_filename(index: int) -> str:
    """Return the deterministic 1-based clip file name for ``index``."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ExportError(
            f"invalid clip index: {index!r}; expected an integer >= 1."
        )
    return f"serve-{index:03d}.mov"


def _check_encoder(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(
            f"invalid {name}: {value!r}; expected a non-blank encoder name."
        )
    cleaned = value.strip()
    if cleaned.lower() == "copy":
        raise ExportError(
            f"invalid {name}: 'copy' would be a stream copy, which cannot "
            "provide exact filter-based trims; choose a re-encoding encoder "
            f"such as {DEFAULT_VIDEO_ENCODER!r}."
        )
    return cleaned


def _format_seconds(value: float) -> str:
    return f"{float(value):.6f}"


def _parse_video_bitrate_to_bps(value: object) -> int:
    """Parse an FFmpeg-style bitrate (``"6M"``, ``"6000k"``, ``"6000000"``) to bps."""
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ExportError(
            f"invalid video_bitrate: {value!r}; expected a non-blank bitrate "
            "such as '6M'."
        )
    text = value.strip().lower()
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        raise ExportError(
            f"invalid video_bitrate: {value!r}; expected a numeric bitrate "
            "such as '6M'."
        ) from None
    import math

    if not math.isfinite(number) or number <= 0:
        raise ExportError(
            f"invalid video_bitrate: {value!r}; expected a bitrate greater "
            "than zero."
        )
    return int(number * multiplier)


def _check_video_bitrate(name: str, value: object) -> str:
    cleaned = value.strip() if isinstance(value, str) else value
    _parse_video_bitrate_to_bps(cleaned)  # raises ExportError when invalid.
    assert isinstance(cleaned, str)
    return cleaned.strip()


def _video_encoder_args(encoder: str, video_bitrate: str) -> list[str]:
    """Return the deterministic quality/speed flags for ``encoder``.

    Hardware ``h264_videotoolbox`` uses an explicit ``-b:v`` target in
    the review band; the software ``libx264`` path keeps its explicit
    ``-preset veryfast -crf 18`` options. Any other re-encoding encoder
    receives the explicit ``-b:v`` target. ``-preset``/``-crf`` are
    ``libx264``-specific and are never emitted for VideoToolbox.
    """
    if encoder == SOFTWARE_VIDEO_ENCODER:
        return ["-preset", "veryfast", "-crf", "18"]
    return ["-b:v", video_bitrate]


def _hardware_failure_hint(encoder: str) -> str:
    if "videotoolbox" in encoder.lower():
        return (
            f" Hardware encode via {encoder!r} failed; ensure this FFmpeg build "
            "supports VideoToolbox hardware encoding on this Mac and retry, or "
            f"pass video_encoder={SOFTWARE_VIDEO_ENCODER!r} explicitly for the "
            "software path. No silent fallback was attempted."
        )
    return ""


_HWACCEL_FAILURE_MARKERS = (
    "videotoolbox",
    "hwaccel",
    "hardware device",
    "no device available",
    "device creation failed",
    "hardware device setup failed",
)


def _is_hwaccel_failure(text: str) -> bool:
    """Return True when FFmpeg output names a VideoToolbox decode failure."""
    return any(marker in (text or "").lower() for marker in _HWACCEL_FAILURE_MARKERS)


def _hardware_decode_failure_hint(detail: str) -> str:
    """Return the actionable VideoToolbox decode hint for ``detail``."""
    if detail and _is_hwaccel_failure(detail):
        return (
            " VideoToolbox hardware decode via '-hwaccel videotoolbox' failed; "
            "ensure this FFmpeg build supports `-hwaccel videotoolbox`, "
            "the source codec is Media Engine compatible, and retry. "
            "No software fallback was attempted; refusing to emit "
            "silently re-decoded output."
        )
    return ""


def build_ffmpeg_args(
    source_video: Path | str,
    start_seconds: float,
    end_seconds: float,
    temp_output: Path | str,
    *,
    ffmpeg: str = "ffmpeg",
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    audio_encoder: str = DEFAULT_AUDIO_ENCODER,
    include_audio: bool = False,
) -> list[str]:
    """Build the deterministic FFmpeg argument array for one exact trim.

    Hardware VideoToolbox decode (``-hwaccel videotoolbox`` as an input
    option before ``-i``) offloads source decode to the Media Engine.
    Video is trimmed with the ``trim`` filter and re-encoded with
    ``video_encoder`` (default VideoToolbox hardware H.264 at the
    ``video_bitrate`` target in the 4-8 Mbps review band; ``libx264``
    remains selectable and keeps its explicit preset/CRF options).
    When ``include_audio`` is true, audio is trimmed
    with the ``atrim`` filter and encoded with ``audio_encoder``;
    otherwise the output is video-only (``-an``). Stream copy is never
    used. The same inputs always produce the same array.
    """
    import math

    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise ExportError(
            f"invalid ffmpeg executable: {ffmpeg!r}; expected a non-blank string."
        )
    encoder = _check_encoder("video_encoder", video_encoder)
    bitrate = _check_video_bitrate("video_bitrate", video_bitrate)
    if include_audio:
        _check_encoder("audio_encoder", audio_encoder)
    for key, value in (("start_seconds", start_seconds), ("end_seconds", end_seconds)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ExportError(
                f"invalid {key}: {value!r}; expected a finite number of seconds."
            )
    start = float(start_seconds)
    end = float(end_seconds)
    if start < 0 or not start < end:
        raise ExportError(
            f"invalid trim range: [{start!r}, {end!r}); "
            "expected 0 <= start < end."
        )
    source = str(source_video)
    dest = str(temp_output)
    if not source:
        raise ExportError("invalid source_video: expected a non-blank path.")
    if not dest:
        raise ExportError("invalid temp_output: expected a non-blank path.")
    video_filter = (
        f"trim=start={_format_seconds(start)}:end={_format_seconds(end)},"
        "setpts=PTS-STARTPTS"
    )
    args = [
        ffmpeg.strip(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "videotoolbox",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-c:v",
        encoder,
        *_video_encoder_args(encoder, bitrate),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if include_audio:
        audio_filter = (
            f"atrim=start={_format_seconds(start)}:end={_format_seconds(end)},"
            "asetpts=PTS-STARTPTS"
        )
        args.extend(
            [
                "-map",
                "0:a?",
                "-af",
                audio_filter,
                "-c:a",
                str(audio_encoder).strip(),
            ]
        )
    else:
        args.append("-an")
    args.append(dest)
    return args


def _missing_tool_error(ffmpeg: str) -> ExportError:
    return ExportError(
        f"ffmpeg was not found as {ffmpeg!r}; "
        "install FFmpeg and ensure it is on PATH."
    )


def _ensure_ffmpeg(ffmpeg: str) -> None:
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise ExportError(
            f"invalid ffmpeg executable: {ffmpeg!r}; expected a non-blank string."
        )
    cleaned = ffmpeg.strip()
    if "/" in cleaned or "\\" in cleaned:
        if not Path(cleaned).is_file():
            raise _missing_tool_error(cleaned)
    elif shutil.which(cleaned) is None:
        raise _missing_tool_error(cleaned)


def _audio_present(payload: dict) -> bool:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return False
    return any(
        isinstance(entry, dict) and entry.get("codec_type") == "audio"
        for entry in streams
    )


def _fingerprint_or_raise(path: Path) -> str:
    try:
        return probe_module.fingerprint_for_path(path)
    except ProbeError as exc:
        raise ExportError(str(exc)) from exc


def _remove_quietly(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def export_clips(
    source_video: Path | str,
    plan: ExportPlan,
    output_dir: Path | str,
    *,
    source: SourceMetadata | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    audio_encoder: str = DEFAULT_AUDIO_ENCODER,
    include_audio: bool | None = None,
    overwrite: bool = False,
    is_cancelled: Callable[[], bool] | None = None,
    timeout_seconds: float = 300.0,
) -> list[Path]:
    """Export one independently playable clip per plan range, in plan order.

    Args:
        source_video: Original source video; read only, never modified.
        plan: Validated :class:`ExportPlan` whose fingerprint must match
            the file at ``source_video``.
        output_dir: Directory receiving ``serve-001.mov`` and friends.
        source: Optional already-probed metadata for ``source_video``; when
            omitted the source is probed. Either way the fingerprint (and
            duration) must agree with ``plan``.
        ffmpeg: FFmpeg executable.
        ffprobe: ffprobe executable used for validation/audio detection.
        video_encoder: Explicit video encoder (never ``"copy"``; default
            VideoToolbox hardware H.264, ``libx264`` selects software).
        video_bitrate: Explicit target video bitrate for hardware encoders
            (default ``'6M'`` in the 4-8 Mbps review band; ignored by the
            ``libx264`` software path, which uses preset/CRF instead).
        audio_encoder: Explicit audio encoder when audio is included.
        include_audio: ``True`` forces trimmed audio, ``False`` forces
            video-only output, ``None`` (default) auto-detects: audio is
            included only when ffprobe reports an audio stream.
        overwrite: When false (default), any pre-existing destination
            raises :class:`ExportCollisionError` before any FFmpeg work.
        is_cancelled: Optional hook polled before each clip; a true
            return raises :class:`ExportCancelled`.
        timeout_seconds: Per-clip FFmpeg timeout.

    Returns:
        Final clip paths in plan order.

    Raises:
        ExportCollisionError: On destination collisions without overwrite.
        ExportCancelled: When ``is_cancelled`` reports cancellation.
        ExportError: On invalid inputs, probe/FFmpeg failures, or timeouts.
    """
    import math

    if not isinstance(plan, ExportPlan):
        raise ExportError(
            "invalid export plan: "
            f"expected ExportPlan, got {type(plan).__name__}."
        )
    source_path = Path(source_video).expanduser()
    if not source_path.is_file():
        raise ExportError(f"input video does not exist: {source_path}.")
    out_dir = Path(output_dir).expanduser()
    encoder = _check_encoder("video_encoder", video_encoder)
    bitrate = _check_video_bitrate("video_bitrate", video_bitrate)
    if include_audio is True:
        _check_encoder("audio_encoder", audio_encoder)
    elif include_audio is not None and not isinstance(include_audio, bool):
        raise ExportError(
            f"invalid include_audio: {include_audio!r}; "
            "expected True, False, or None."
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ExportError(
            f"invalid timeout_seconds: {timeout_seconds!r}; "
            "expected a finite number greater than zero."
        )
    if is_cancelled is not None and not callable(is_cancelled):
        raise ExportError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    _ensure_ffmpeg(ffmpeg.strip() if isinstance(ffmpeg, str) else ffmpeg)  # type: ignore[arg-type]

    if not isinstance(ffprobe, str) or not ffprobe.strip():
        raise ExportError(
            f"invalid ffprobe executable: {ffprobe!r}; expected a non-blank string."
        )

    fingerprint = _fingerprint_or_raise(source_path)
    if fingerprint != plan.source_fingerprint:
        raise ExportError(
            "export plan fingerprint does not match the source video; "
            "re-probe the source and rebuild the plan from its metadata."
        )

    metadata: SourceMetadata
    audio_payload: dict | None = None
    if source is None:
        try:
            audio_payload = probe_module.run_ffprobe(source_path, ffprobe=ffprobe.strip())
            metadata = probe_module.parse_ffprobe_payload(audio_payload, fingerprint)
        except ProbeError as exc:
            raise ExportError(f"could not probe source video {source_path}: {exc}.") from exc
        if abs(metadata.duration_seconds - plan.source_duration_seconds) > _DURATION_TOLERANCE_SECONDS:
            raise ExportError(
                "export plan duration does not match the source video; "
                "re-probe the source and rebuild the plan from its metadata."
            )
    else:
        if not isinstance(source, SourceMetadata):
            raise ExportError(
                "invalid source metadata: "
                f"expected SourceMetadata, got {type(source).__name__}."
            )
        if source.fingerprint != fingerprint or source.fingerprint != plan.source_fingerprint:
            raise ExportError(
                "export plan fingerprint does not match the source video; "
                "re-probe the source and rebuild the plan from its metadata."
            )
        if abs(source.duration_seconds - plan.source_duration_seconds) > _DURATION_TOLERANCE_SECONDS:
            raise ExportError(
                "export plan duration does not match the source metadata; "
                "re-probe the source and rebuild the plan from its metadata."
            )
        metadata = source
    _ = metadata  # validated above; ranges carry the authoritative bounds.

    want_audio: bool
    if include_audio is None:
        try:
            if audio_payload is None:
                audio_payload = probe_module.run_ffprobe(source_path, ffprobe=ffprobe.strip())
            want_audio = _audio_present(audio_payload)
        except ProbeError:
            want_audio = False
    else:
        want_audio = bool(include_audio)
    if want_audio:
        _check_encoder("audio_encoder", audio_encoder)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(
            f"could not create output directory {out_dir}: {exc}."
        ) from exc

    destinations = [out_dir / clip_filename(number) for number in range(1, len(plan) + 1)]
    if not overwrite:
        collisions = [path for path in destinations if path.exists()]
        if collisions:
            listing = ", ".join(str(path) for path in collisions)
            raise ExportCollisionError(
                f"output collision: {listing} already exists; "
                "pass overwrite=True to replace explicitly."
            )

    if is_cancelled is not None and is_cancelled():
        raise ExportCancelled("clip export was cancelled before starting.")

    completed: list[Path] = []
    ffmpeg_exe = ffmpeg.strip()
    for number, media_range in enumerate(plan, start=1):
        if is_cancelled is not None and is_cancelled():
            raise ExportCancelled(
                f"clip export was cancelled before clip {number} of {len(plan)}."
            )
        dest = destinations[number - 1]
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(out_dir),
                prefix=dest.name + ".tmp-",
                suffix=".mov",
                delete=False,
            ) as handle:
                tmp_path = handle.name
            args = build_ffmpeg_args(
                source_path,
                media_range.start_seconds,
                media_range.end_seconds,
                tmp_path,
                ffmpeg=ffmpeg_exe,
                video_encoder=encoder,
                video_bitrate=bitrate,
                audio_encoder=audio_encoder,
                include_audio=want_audio,
            )
            try:
                finished = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=float(timeout_seconds),
                )
            except FileNotFoundError as exc:
                raise _missing_tool_error(ffmpeg_exe) from exc
            except subprocess.TimeoutExpired as exc:
                raise ExportError(
                    f"ffmpeg timed out exporting clip {number} of {len(plan)} "
                    f"[{media_range.start_seconds}, {media_range.end_seconds}s) "
                    f"with video encoder {encoder!r} "
                    f"after {timeout_seconds}s.{_hardware_failure_hint(encoder)}"
                ) from exc
            except OSError as exc:
                raise ExportError(
                    f"could not run ffmpeg as {ffmpeg_exe!r}: {exc}."
                ) from exc
            if finished.returncode != 0:
                detail = ((finished.stderr or "").strip() or (finished.stdout or "").strip())
                suffix = f": {detail}" if detail else ": no output"
                raise ExportError(
                    f"ffmpeg failed exporting clip {number} of {len(plan)} "
                    f"[{media_range.start_seconds}, {media_range.end_seconds}s) "
                    f"with video encoder {encoder!r} "
                    f"(exit {finished.returncode}){suffix}."
                    f"{_hardware_failure_hint(encoder)}"
                    f"{_hardware_decode_failure_hint(detail)}"
                )
            if not Path(tmp_path).is_file() or Path(tmp_path).stat().st_size == 0:
                raise ExportError(
                    f"ffmpeg did not produce output for clip {number} of {len(plan)} "
                    f"[{media_range.start_seconds}, {media_range.end_seconds}s) "
                    f"with video encoder {encoder!r}.{_hardware_failure_hint(encoder)}"
                )
            try:
                os.replace(tmp_path, dest)
            except OSError as exc:
                raise ExportError(
                    f"could not move finished clip {number} into place at {dest}: {exc}."
                ) from exc
            tmp_path = None
        except BaseException:
            _remove_quietly(tmp_path)
            raise
        completed.append(dest)
    return completed


def _check_ranges_sequence(ranges: object) -> list:
    from serve_review.domain import MediaRange as _MediaRange

    if isinstance(ranges, _MediaRange) or not isinstance(ranges, (list, tuple)):
        raise ExportError(
            "invalid compilation ranges: expected a non-empty list/tuple "
            f"of MediaRange, got {type(ranges).__name__}."
        )
    items = list(ranges)
    if not items:
        raise ExportError(
            "invalid compilation ranges: at least one range is required; "
            "an empty plan would produce an empty compilation."
        )
    for entry in items:
        if not isinstance(entry, _MediaRange):
            raise ExportError(
                "invalid compilation range: expected MediaRange, "
                f"got {type(entry).__name__}."
            )
    return items


def build_compilation_ffmpeg_args(
    source_video: Path | str,
    ranges: list | tuple,
    temp_output: Path | str,
    *,
    ffmpeg: str = "ffmpeg",
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    audio_encoder: str = DEFAULT_AUDIO_ENCODER,
    include_audio: bool = False,
) -> list[str]:
    """Build the deterministic FFmpeg argument array for one compilation.

    Hardware VideoToolbox decode (``-hwaccel videotoolbox`` as an input
    option before ``-i``) offloads source decode to the Media Engine.
    Each range is trimmed with the ``trim``/``atrim`` filters, timestamps
    are reset with ``setpts``/``asetpts``, and the segments are joined with
    the ``concat`` filter in the given (plan) order. Segments are placed
    back-to-back with no artificial gaps or padding. Stream copy is never
    used; video (and optional audio) are always re-encoded with explicit
    encoders (default VideoToolbox hardware H.264 at the ``video_bitrate``
    review-band target; ``libx264`` remains selectable with preset/CRF).
    The same inputs always produce the same array.
    """
    import math

    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise ExportError(
            f"invalid ffmpeg executable: {ffmpeg!r}; expected a non-blank string."
        )
    encoder = _check_encoder("video_encoder", video_encoder)
    bitrate = _check_video_bitrate("video_bitrate", video_bitrate)
    if include_audio:
        _check_encoder("audio_encoder", audio_encoder)
    items = _check_ranges_sequence(ranges)
    for entry in items:
        for key in ("start_seconds", "end_seconds"):
            value = getattr(entry, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ExportError(
                    f"invalid {key}: {value!r}; expected a finite number of seconds."
                )
        if not 0 <= float(entry.start_seconds) < float(entry.end_seconds):
            raise ExportError(
                f"invalid compilation range: [{entry.start_seconds!r}, "
                f"{entry.end_seconds!r}); expected 0 <= start < end."
            )
    source = str(source_video)
    dest = str(temp_output)
    if not source:
        raise ExportError("invalid source_video: expected a non-blank path.")
    if not dest:
        raise ExportError("invalid temp_output: expected a non-blank path.")
    video_chains: list[str] = []
    audio_chains: list[str] = []
    for index, entry in enumerate(items):
        start = _format_seconds(float(entry.start_seconds))
        end = _format_seconds(float(entry.end_seconds))
        video_chains.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]"
        )
        if include_audio:
            audio_chains.append(
                f"[0:a]atrim=start={start}:end={end},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
    video_inputs = "".join(f"[v{index}]" for index in range(len(items)))
    filter_parts = list(video_chains)
    filter_parts.append(
        f"{video_inputs}concat=n={len(items)}:v=1:a=0[outv]"
    )
    if include_audio:
        audio_inputs = "".join(f"[a{index}]" for index in range(len(items)))
        filter_parts.extend(audio_chains)
        filter_parts.append(
            f"{audio_inputs}concat=n={len(items)}:v=0:a=1[outa]"
        )
    filter_complex = ";".join(filter_parts)
    args = [
        ffmpeg.strip(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-hwaccel",
        "videotoolbox",
        "-i",
        source,
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-c:v",
        encoder,
        *_video_encoder_args(encoder, bitrate),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if include_audio:
        args.extend(
            ["-map", "[outa]", "-c:a", str(audio_encoder).strip()]
        )
    else:
        args.append("-an")
    args.append(dest)
    return args


def _resolve_plan_source(
    source_path: Path,
    plan: ExportPlan,
    source: SourceMetadata | None,
    ffprobe: str,
) -> tuple[SourceMetadata, dict | None]:
    """Validate fingerprint/duration and detect audio for export paths."""
    fingerprint = _fingerprint_or_raise(source_path)
    if fingerprint != plan.source_fingerprint:
        raise ExportError(
            "export plan fingerprint does not match the source video; "
            "re-probe the source and rebuild the plan from its metadata."
        )
    audio_payload: dict | None = None
    if source is None:
        try:
            audio_payload = probe_module.run_ffprobe(source_path, ffprobe=ffprobe.strip())
            metadata = probe_module.parse_ffprobe_payload(audio_payload, fingerprint)
        except ProbeError as exc:
            raise ExportError(
                f"could not probe source video {source_path}: {exc}."
            ) from exc
        if abs(metadata.duration_seconds - plan.source_duration_seconds) > _DURATION_TOLERANCE_SECONDS:
            raise ExportError(
                "export plan duration does not match the source video; "
                "re-probe the source and rebuild the plan from its metadata."
            )
    else:
        if not isinstance(source, SourceMetadata):
            raise ExportError(
                "invalid source metadata: "
                f"expected SourceMetadata, got {type(source).__name__}."
            )
        if source.fingerprint != fingerprint or source.fingerprint != plan.source_fingerprint:
            raise ExportError(
                "export plan fingerprint does not match the source video; "
                "re-probe the source and rebuild the plan from its metadata."
            )
        if abs(source.duration_seconds - plan.source_duration_seconds) > _DURATION_TOLERANCE_SECONDS:
            raise ExportError(
                "export plan duration does not match the source metadata; "
                "re-probe the source and rebuild the plan from its metadata."
            )
        metadata = source
    return metadata, audio_payload


def _decide_audio(
    source_path: Path,
    ffprobe: str,
    audio_payload: dict | None,
    include_audio: bool | None,
    audio_encoder: str,
) -> bool:
    if include_audio is None:
        try:
            if audio_payload is None:
                audio_payload = probe_module.run_ffprobe(source_path, ffprobe=ffprobe.strip())
            return _audio_present(audio_payload)
        except ProbeError:
            return False
    if not isinstance(include_audio, bool):
        raise ExportError(
            f"invalid include_audio: {include_audio!r}; "
            "expected True, False, or None."
        )
    if include_audio:
        _check_encoder("audio_encoder", audio_encoder)
    return bool(include_audio)


def _run_ffmpeg_for_output(
    args: list[str],
    ffmpeg_exe: str,
    tmp_path: str,
    dest: Path,
    label: str,
    timeout_seconds: float,
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
) -> None:
    try:
        finished = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_seconds),
        )
    except FileNotFoundError as exc:
        raise _missing_tool_error(ffmpeg_exe) from exc
    except subprocess.TimeoutExpired as exc:
        raise ExportError(
            f"ffmpeg timed out {label} with video encoder "
            f"{video_encoder!r} after {timeout_seconds}s."
            f"{_hardware_failure_hint(video_encoder)}"
        ) from exc
    except OSError as exc:
        raise ExportError(
            f"could not run ffmpeg as {ffmpeg_exe!r}: {exc}."
        ) from exc
    if finished.returncode != 0:
        detail = ((finished.stderr or "").strip() or (finished.stdout or "").strip())
        suffix = f": {detail}" if detail else ": no output"
        raise ExportError(
            f"ffmpeg failed {label} with video encoder {video_encoder!r} "
            f"(exit {finished.returncode}){suffix}."
            f"{_hardware_failure_hint(video_encoder)}"
            f"{_hardware_decode_failure_hint(detail)}"
        )
    if not Path(tmp_path).is_file() or Path(tmp_path).stat().st_size == 0:
        raise ExportError(
            f"ffmpeg did not produce output {label} with video encoder "
            f"{video_encoder!r}.{_hardware_failure_hint(video_encoder)}"
        )
    try:
        os.replace(tmp_path, dest)
    except OSError as exc:
        raise ExportError(
            f"could not move finished output into place at {dest}: {exc}."
        ) from exc


def _escape_concat_path(path: str) -> str:
    """Escape ``path`` for an FFmpeg concat-demuxer list file."""
    return "'" + path.replace("'", "'\\''") + "'"


def build_concat_copy_args(
    listfile: Path | str,
    temp_output: Path | str,
    *,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the concat-demuxer ``-c copy`` argument array.

    The list file holds one ``file '<path>'`` line per clip in join
    order. Video and audio streams are copied without decoding or
    re-encoding, so the join runs at disk speed with no quality loss.
    The same inputs always produce the same array.
    """
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise ExportError(
            f"invalid ffmpeg executable: {ffmpeg!r}; expected a non-blank string."
        )
    list_path = str(listfile)
    dest = str(temp_output)
    if not list_path:
        raise ExportError("invalid listfile: expected a non-blank path.")
    if not dest:
        raise ExportError("invalid temp_output: expected a non-blank path.")
    return [
        ffmpeg.strip(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        dest,
    ]


def _clip_stream_signatures(payload: dict) -> tuple[tuple, tuple]:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ExportError("clip probe payload has no streams list.")
    video = next(
        (
            entry
            for entry in streams
            if isinstance(entry, dict) and entry.get("codec_type") == "video"
        ),
        None,
    )
    if video is None:
        raise ExportError("clip has no video stream; cannot stream-copy join.")
    video_sig = (
        video.get("codec_name"),
        video.get("width"),
        video.get("height"),
        video.get("avg_frame_rate"),
        video.get("r_frame_rate"),
        video.get("time_base"),
        video.get("pix_fmt"),
    )
    audio_entries = [
        entry
        for entry in streams
        if isinstance(entry, dict) and entry.get("codec_type") == "audio"
    ]
    if not audio_entries:
        audio_sig: tuple = (False,)
    else:
        first = audio_entries[0]
        audio_sig = (
            True,
            first.get("codec_name"),
            first.get("sample_rate"),
            first.get("channels"),
            first.get("time_base"),
        )
    return (video_sig, audio_sig)


def concat_join_readiness(
    clip_paths: list[Path] | tuple[Path, ...],
    *,
    ffprobe: str = "ffprobe",
) -> str | None:
    """Check whether ``clip_paths`` can be joined with stream copy.

    Returns ``None`` when the join is valid; otherwise returns an
    explicit reason string so the caller fails closed to the filter
    re-encode path instead of attempting a corrupt join. Checks, in
    order: every input is present and non-empty, every input probes,
    and every input shares the first clip's video codec, dimensions,
    frame-rate/timebase, pixel format, and audio layout.
    """
    items = list(clip_paths)
    if not items:
        return "no clip files were produced; falling back to filter re-encode."
    for path in items:
        candidate = Path(path)
        if not candidate.is_file():
            return (
                f"clip file is missing: {candidate}; "
                "falling back to filter re-encode."
            )
        try:
            if candidate.stat().st_size == 0:
                return (
                    f"clip file is empty: {candidate}; "
                    "falling back to filter re-encode."
                )
        except OSError as exc:
            return (
                f"could not stat clip file {candidate} ({exc}); "
                "falling back to filter re-encode."
            )
    if not isinstance(ffprobe, str) or not ffprobe.strip():
        return (
            f"invalid ffprobe executable: {ffprobe!r}; "
            "falling back to filter re-encode."
        )
    signatures: list[tuple[tuple, tuple]] = []
    for path in items:
        try:
            payload = probe_module.run_ffprobe(Path(path), ffprobe=ffprobe.strip())
        except ProbeError as exc:
            return (
                f"could not probe clip {path} ({exc}); "
                "falling back to filter re-encode."
            )
        try:
            signatures.append(_clip_stream_signatures(payload))
        except ExportError as exc:
            return f"{exc} Falling back to filter re-encode."
    reference = signatures[0]
    for index, signature in enumerate(signatures[1:], start=2):
        if signature != reference:
            return (
                f"clip {index} of {len(items)} has differing "
                f"codec/timebase/resolution/audio {signature!r} vs "
                f"first clip {reference!r}; falling back to filter "
                "re-encode rather than a corrupt join."
            )
    return None


def _join_clips_with_copy(
    clip_paths: list[Path],
    dest: Path,
    *,
    ffmpeg: str = "ffmpeg",
    is_cancelled: Callable[[], bool] | None = None,
    timeout_seconds: float = 300.0,
) -> Path:
    """Join ``clip_paths`` in order into ``dest`` via ``-c copy``.

    Writes to a temporary file beside ``dest`` plus a temporary concat
    list file, then atomically renames. Temporary files are removed on
    failure, timeout, cancellation, or keyboard interrupt.
    """
    if is_cancelled is not None and is_cancelled():
        raise ExportCancelled("compilation join was cancelled before starting.")
    _ensure_ffmpeg(ffmpeg.strip() if isinstance(ffmpeg, str) else ffmpeg)  # type: ignore[arg-type]
    ffmpeg_exe = ffmpeg.strip()
    parent = dest.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExportError(
                f"could not create output directory {parent}: {exc}."
            ) from exc
    tmp_path: str | None = None
    list_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=dest.name + ".tmp-concat-",
            suffix=".txt",
            delete=False,
        ) as list_handle:
            list_path = list_handle.name
            for clip in clip_paths:
                list_handle.write(f"file {_escape_concat_path(str(clip))}\n")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=dest.name + ".tmp-",
            suffix=".mov",
            delete=False,
        ) as handle:
            tmp_path = handle.name
        args = build_concat_copy_args(list_path, tmp_path, ffmpeg=ffmpeg_exe)
        try:
            finished = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
                timeout=float(timeout_seconds),
            )
        except FileNotFoundError as exc:
            raise _missing_tool_error(ffmpeg_exe) from exc
        except subprocess.TimeoutExpired as exc:
            raise ExportError(
                f"ffmpeg timed out joining {len(clip_paths)} clip(s) "
                f"with stream copy after {timeout_seconds}s."
            ) from exc
        except OSError as exc:
            raise ExportError(
                f"could not run ffmpeg as {ffmpeg_exe!r}: {exc}."
            ) from exc
        if finished.returncode != 0:
            detail = ((finished.stderr or "").strip() or (finished.stdout or "").strip())
            suffix = f": {detail}" if detail else ": no output"
            raise ExportError(
                f"ffmpeg failed joining {len(clip_paths)} clip(s) "
                f"with stream copy (exit {finished.returncode}){suffix}."
            )
        if not Path(tmp_path).is_file() or Path(tmp_path).stat().st_size == 0:
            raise ExportError(
                f"ffmpeg did not produce output joining {len(clip_paths)} "
                "clip(s) with stream copy."
            )
        try:
            os.replace(tmp_path, dest)
        except OSError as exc:
            raise ExportError(
                f"could not move joined compilation into place at {dest}: {exc}."
            ) from exc
        tmp_path = None
    except BaseException:
        _remove_quietly(tmp_path)
        _remove_quietly(list_path)
        raise
    _remove_quietly(list_path)
    return dest


def export_compilation(
    source_video: Path | str,
    plan: ExportPlan,
    dest: Path | str,
    *,
    source: SourceMetadata | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    audio_encoder: str = DEFAULT_AUDIO_ENCODER,
    include_audio: bool | None = None,
    overwrite: bool = False,
    is_cancelled: Callable[[], bool] | None = None,
    timeout_seconds: float = 300.0,
) -> Path:
    """Concatenate plan ranges into one gapless compilation, in plan order.

    Segments are trimmed from the original source samples with exact
    filter-based trims and joined back-to-back with the ``concat`` filter:
    no dead-time gaps are inserted. The output is always re-encoded with
    an explicit video encoder (default VideoToolbox hardware H.264 at the
    ``video_bitrate`` review-band target; ``libx264`` selects software);
    stream copy is never used.

    Args:
        source_video: Original source video; read only, never modified.
        plan: Validated :class:`ExportPlan` in source order. Overlap and
            empty plans are rejected by the domain plan itself; this
            function never reorders or merges ranges.
        dest: Final ``serves.mov``-style output file path.
        source: Optional already-probed metadata; fingerprint/duration
            must agree with ``plan`` and the file.
        include_audio: ``True`` forces trimmed audio, ``False`` forces
            video-only output, ``None`` (default) auto-detects from ffprobe.
        overwrite: When false (default), a pre-existing ``dest`` raises
            :class:`ExportCollisionError` before any FFmpeg work.
        is_cancelled: Optional hook; a true return raises
            :class:`ExportCancelled`.
        timeout_seconds: FFmpeg timeout for the compilation job.

    Returns:
        The final compilation path.
    """
    import math

    if not isinstance(plan, ExportPlan):
        raise ExportError(
            "invalid export plan: "
            f"expected ExportPlan, got {type(plan).__name__}."
        )
    source_path = Path(source_video).expanduser()
    if not source_path.is_file():
        raise ExportError(f"input video does not exist: {source_path}.")
    destination = Path(dest).expanduser()
    encoder = _check_encoder("video_encoder", video_encoder)
    bitrate = _check_video_bitrate("video_bitrate", video_bitrate)
    if include_audio is True:
        _check_encoder("audio_encoder", audio_encoder)
    elif include_audio is not None and not isinstance(include_audio, bool):
        raise ExportError(
            f"invalid include_audio: {include_audio!r}; "
            "expected True, False, or None."
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
    ):
        raise ExportError(
            f"invalid timeout_seconds: {timeout_seconds!r}; "
            "expected a finite number greater than zero."
        )
    if is_cancelled is not None and not callable(is_cancelled):
        raise ExportError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    _ensure_ffmpeg(ffmpeg.strip() if isinstance(ffmpeg, str) else ffmpeg)  # type: ignore[arg-type]
    if not isinstance(ffprobe, str) or not ffprobe.strip():
        raise ExportError(
            f"invalid ffprobe executable: {ffprobe!r}; expected a non-blank string."
        )
    _, audio_payload = _resolve_plan_source(source_path, plan, source, ffprobe.strip())
    want_audio = _decide_audio(
        source_path, ffprobe.strip(), audio_payload, include_audio, audio_encoder
    )
    # NOTE: _resolve_plan_source already validated fingerprint/duration;
    if want_audio:
        _check_encoder("audio_encoder", audio_encoder)
    parent = destination.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExportError(
                f"could not create output directory {parent}: {exc}."
            ) from exc
    if not overwrite and destination.exists():
        raise ExportCollisionError(
            f"output collision: {destination} already exists; "
            "pass overwrite=True to replace explicitly."
        )
    if is_cancelled is not None and is_cancelled():
        raise ExportCancelled("compilation export was cancelled before starting.")
    ffmpeg_exe = ffmpeg.strip()
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=destination.name + ".tmp-",
            suffix=".mov",
            delete=False,
        ) as handle:
            tmp_path = handle.name
        args = build_compilation_ffmpeg_args(
            source_path,
            list(plan.ranges),
            tmp_path,
            ffmpeg=ffmpeg_exe,
            video_encoder=encoder,
            video_bitrate=bitrate,
            audio_encoder=audio_encoder,
            include_audio=want_audio,
        )
        label = (
            f"exporting compilation of {len(plan)} range(s) "
            f"[{plan.ranges[0].start_seconds}, ..., {plan.ranges[-1].end_seconds}s)"
        )
        _run_ffmpeg_for_output(args, ffmpeg_exe, tmp_path, destination, label, float(timeout_seconds), encoder)
        tmp_path = None
    except BaseException:
        _remove_quietly(tmp_path)
        raise
    return destination


def export_outputs(
    source_video: Path | str,
    plan: ExportPlan,
    output_dir: Path | str,
    *,
    mode: str = "compilation",
    source: SourceMetadata | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    audio_encoder: str = DEFAULT_AUDIO_ENCODER,
    include_audio: bool | None = None,
    overwrite: bool = False,
    is_cancelled: Callable[[], bool] | None = None,
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Export ``mode`` (``compilation``/``clips``/``both``) for ``plan``.

    Layout inside ``output_dir``::

        serves.mov            # compilation/both
        clips/serve-001.mov   # clips/both

    Ranges keep plan (source) order; the compilation concatenates them
    back-to-back with no artificial gaps. All destinations are checked
    for collisions before any FFmpeg work runs. Temporary files (plus the
    concat list file) are written beside each final result and atomically
    renamed; they are removed on failure, timeout, or cancellation.

    In ``both`` mode the clips are encoded once from the original source
    samples and the compilation is then joined losslessly from those
    just-encoded clips with the concat demuxer plus stream copy
    (``-c copy``: no re-encode, no quality loss, disk-speed join). When
    the join inputs fail closed validation (missing inputs or differing
    codec/timebase/resolution) or the copy itself fails, the compilation
    falls back to the current filter re-encode path from the source.
    Standalone ``compilation`` mode always uses the filter re-encode
    path unchanged.

    Frame accuracy lives in the clips; the join preserves clip order and
    introduces no dead-time gaps.

    Returns:
        ``{"compilation": Path | None, "clips": list[Path]}``.
    """
    if not isinstance(plan, ExportPlan):
        raise ExportError(
            "invalid export plan: "
            f"expected ExportPlan, got {type(plan).__name__}."
        )
    if mode not in EXPORT_MODES:
        raise ExportError(
            f"invalid output mode: {mode!r}; expected one of {list(EXPORT_MODES)}."
        )
    source_path = Path(source_video).expanduser()
    if not source_path.is_file():
        raise ExportError(f"input video does not exist: {source_path}.")
    _check_encoder("video_encoder", video_encoder)
    _check_video_bitrate("video_bitrate", video_bitrate)
    out_dir = Path(output_dir).expanduser()
    compilation_dest = out_dir / COMPILATION_FILENAME
    clips_dir = out_dir / CLIPS_SUBDIR
    want_compilation = mode in ("compilation", "both")
    want_clips = mode in ("clips", "both")
    # Collision pre-check before any FFmpeg work.
    if not overwrite:
        collisions: list[Path] = []
        if want_compilation and compilation_dest.exists():
            collisions.append(compilation_dest)
        if want_clips:
            for number in range(1, len(plan) + 1):
                candidate = clips_dir / clip_filename(number)
                if candidate.exists():
                    collisions.append(candidate)
        if collisions:
            listing = ", ".join(str(path) for path in collisions)
            raise ExportCollisionError(
                f"output collision: {listing} already exists; "
                "pass overwrite=True to replace explicitly."
            )
    if is_cancelled is not None and not callable(is_cancelled):
        raise ExportError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    if is_cancelled is not None and is_cancelled():
        raise ExportCancelled("export was cancelled before starting.")
    compilation: Path | None = None
    clips: list[Path] = []
    if mode == "both":
        # Single lossy generation: encode each range once as a clip,
        # then join serves.mov losslessly from those clips. Any join
        # problem fails closed to the filter re-encode path below.
        clips = export_clips(
            source_path,
            plan,
            clips_dir,
            source=source,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            video_encoder=video_encoder,
            video_bitrate=video_bitrate,
            audio_encoder=audio_encoder,
            include_audio=include_audio,
            overwrite=overwrite,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
        fallback_reason = concat_join_readiness(clips, ffprobe=ffprobe)
        if fallback_reason is None:
            try:
                compilation = _join_clips_with_copy(
                    clips,
                    compilation_dest,
                    ffmpeg=ffmpeg,
                    is_cancelled=is_cancelled,
                    timeout_seconds=timeout_seconds,
                )
            except ExportCancelled:
                raise
            except KeyboardInterrupt:
                raise
            except ExportError as exc:
                fallback_reason = (
                    "concat stream-copy join failed "
                    f"({exc}); falling back to filter re-encode."
                )
        if fallback_reason is not None:
            compilation = export_compilation(
                source_path,
                plan,
                compilation_dest,
                source=source,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                video_encoder=video_encoder,
                video_bitrate=video_bitrate,
                audio_encoder=audio_encoder,
                include_audio=include_audio,
                overwrite=overwrite,
                is_cancelled=is_cancelled,
                timeout_seconds=timeout_seconds,
            )
        return {"compilation": compilation, "clips": clips}
    if want_compilation:
        compilation = export_compilation(
            source_path,
            plan,
            compilation_dest,
            source=source,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            video_encoder=video_encoder,
            video_bitrate=video_bitrate,
            audio_encoder=audio_encoder,
            include_audio=include_audio,
            overwrite=overwrite,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
    if want_clips:
        clips = export_clips(
            source_path,
            plan,
            clips_dir,
            source=source,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            video_encoder=video_encoder,
            video_bitrate=video_bitrate,
            audio_encoder=audio_encoder,
            include_audio=include_audio,
            overwrite=overwrite,
            is_cancelled=is_cancelled,
            timeout_seconds=timeout_seconds,
        )
    return {"compilation": compilation, "clips": clips}
