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
    "DEFAULT_AUDIO_ENCODER",
    "DEFAULT_VIDEO_ENCODER",
    "ExportCancelled",
    "ExportCollisionError",
    "ExportError",
    "build_ffmpeg_args",
    "clip_filename",
    "export_clips",
]

#: Explicit default video encoder. Always re-encoded; never stream-copied.
DEFAULT_VIDEO_ENCODER = "libx264"
#: Explicit audio encoder used only when the source has an audio stream.
DEFAULT_AUDIO_ENCODER = "aac"

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


def build_ffmpeg_args(
    source_video: Path | str,
    start_seconds: float,
    end_seconds: float,
    temp_output: Path | str,
    *,
    ffmpeg: str = "ffmpeg",
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
    audio_encoder: str = DEFAULT_AUDIO_ENCODER,
    include_audio: bool = False,
) -> list[str]:
    """Build the deterministic FFmpeg argument array for one exact trim.

    Video is trimmed with the ``trim`` filter and re-encoded with
    ``video_encoder``. When ``include_audio`` is true, audio is trimmed
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
        "-i",
        source,
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-c:v",
        encoder,
        "-preset",
        "veryfast",
        "-crf",
        "18",
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
        video_encoder: Explicit video encoder (never ``"copy"``).
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
                    f"after {timeout_seconds}s."
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
                    f"(exit {finished.returncode}){suffix}."
                )
            if not Path(tmp_path).is_file() or Path(tmp_path).stat().st_size == 0:
                raise ExportError(
                    f"ffmpeg did not produce output for clip {number} of {len(plan)} "
                    f"[{media_range.start_seconds}, {media_range.end_seconds}s)."
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
