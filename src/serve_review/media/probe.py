"""ffprobe adapter that normalizes source-video metadata.

This module owns all ffprobe interaction and normalization. It runs ffprobe
with an argument array, parses the JSON document, preserves rational
frame-rate/time-base values exactly, normalizes MOV orientation, and returns
the M1.1 :class:`SourceMetadata` domain object. File output is written
atomically. CLI code must call these helpers instead of parsing ffprobe
output itself.

Rotation precedence (deterministic, first present wins):

1. top-level ``stream["rotation"]`` when present and not null;
2. ``stream["tags"]["rotate"]`` (also accepts ``"rotation"``, case
   insensitive) when present;
3. the first ``side_data_list`` entry whose ``side_data_type`` is
   ``"Display Matrix"`` (case insensitive, ``"displaymatrix"`` accepted)
   carrying a ``"rotation"`` value;
4. otherwise ``0`` (no rotation metadata).

Every present rotation value is normalized with modulo 360 (so ``-90``
becomes ``270`` and ``360`` becomes ``0``) and must then be one of
``0, 90, 180, 270``; malformed or unsupported degrees raise
:class:`ProbeError`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from serve_review.domain import DomainError, SourceMetadata

__all__ = [
    "ProbeError",
    "SUPPORTED_ROTATIONS",
    "build_ffprobe_args",
    "extract_rotation",
    "fingerprint_for_path",
    "normalize_rotation_value",
    "parse_ffprobe_payload",
    "probe_source",
    "run_ffprobe",
    "write_source_json",
]

SUPPORTED_ROTATIONS = (0, 90, 180, 270)


class ProbeError(Exception):
    """Actionable failure to probe a source video with ffprobe."""


def build_ffprobe_args(video: Path | str, *, ffprobe: str = "ffprobe") -> list[str]:
    """Build the ffprobe argument array for ``video``."""
    return [
        ffprobe,
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(video),
    ]


def fingerprint_for_path(path: Path | str) -> str:
    """Return the ``sha256:<hex>`` fingerprint of the file at ``path``."""
    resolved = Path(path)
    try:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProbeError(f"could not read input video {resolved}: {exc}.") from exc
    return f"sha256:{digest.hexdigest()}"


def normalize_rotation_value(raw: object, source: str) -> int:
    """Normalize one raw rotation value to ``0/90/180/270``.

    Accepts ints, integral floats, and numeric strings. Applies modulo 360
    so negative values (``-90``) and full turns (``360``) normalize
    deterministically. Raises :class:`ProbeError` for malformed values
    (non-numeric, non-integral, non-finite) and for unsupported degrees
    (for example ``45``).
    """
    if isinstance(raw, bool):
        raise ProbeError(
            f"malformed rotation for {source}: {raw!r}; "
            "expected a numeric degree value in {0, 90, 180, 270} (modulo 360)."
        )
    degrees: int | None = None
    if isinstance(raw, int):
        degrees = raw
    elif isinstance(raw, float):
        if not math.isfinite(raw) or not raw.is_integer():
            raise ProbeError(
                f"malformed rotation for {source}: {raw!r}; "
                "expected a numeric degree value in {0, 90, 180, 270} (modulo 360)."
            )
        degrees = int(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ProbeError(
                f"malformed rotation for {source}: {raw!r}; "
                "expected a numeric degree value in {0, 90, 180, 270} (modulo 360)."
            )
        try:
            number = float(text)
        except ValueError as exc:
            raise ProbeError(
                f"malformed rotation for {source}: {raw!r}; "
                "expected a numeric degree value in {0, 90, 180, 270} (modulo 360)."
            ) from exc
        if not math.isfinite(number) or not float(number).is_integer():
            raise ProbeError(
                f"malformed rotation for {source}: {raw!r}; "
                "expected a numeric degree value in {0, 90, 180, 270} (modulo 360)."
            )
        degrees = int(number)
    else:
        raise ProbeError(
            f"malformed rotation for {source}: {raw!r}; "
            "expected a numeric degree value in {0, 90, 180, 270} (modulo 360)."
        )
    normalized = degrees % 360
    if normalized not in SUPPORTED_ROTATIONS:
        raise ProbeError(
            f"unsupported rotation for {source}: {raw!r} "
            f"(normalized to {normalized}); "
            "supported degrees are 0, 90, 180, 270 (modulo 360)."
        )
    return normalized


def _is_display_matrix_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    side_type = entry.get("side_data_type")
    if not isinstance(side_type, str):
        return False
    normalized = side_type.strip().lower()
    return normalized in ("display matrix", "displaymatrix")


def extract_rotation(stream: dict) -> int:
    """Extract normalized rotation from one ffprobe video stream dict.

    Implements the module-level precedence: direct ``rotation`` field, then
    ``tags`` rotate/rotation, then ``side_data_list`` Display Matrix, then 0.
    """
    if not isinstance(stream, dict):
        raise ProbeError(
            f"malformed ffprobe output: video stream must be an object, "
            f"got {type(stream).__name__}."
        )
    if "rotation" in stream and stream["rotation"] is not None:
        return normalize_rotation_value(stream["rotation"], "stream rotation")
    tags = stream.get("tags")
    if isinstance(tags, dict):
        for key, value in tags.items():
            if isinstance(key, str) and key.strip().lower() in ("rotate", "rotation"):
                return normalize_rotation_value(value, f"stream tags {key!r}")
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for entry in side_data:
            if not _is_display_matrix_entry(entry):
                continue
            assert isinstance(entry, dict)
            if "rotation" not in entry or entry["rotation"] is None:
                continue
            return normalize_rotation_value(
                entry["rotation"], "side_data_list Display Matrix rotation"
            )
    return 0


def _parse_rational(raw: object, field: str) -> tuple[int, int]:
    if isinstance(raw, bool):
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a rational "
            f"'num/den' string, got {raw!r}."
        )
    if isinstance(raw, (int, float)):
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a rational "
            f"'num/den' string, got {raw!r}."
        )
    if not isinstance(raw, str):
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a rational "
            f"'num/den' string, got {raw!r}."
        )
    text = raw.strip()
    separator: str | None = None
    if "/" in text:
        separator = "/"
    elif ":" in text:
        separator = ":"
    if separator is None:
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a rational "
            f"'num/den' string, got {raw!r}."
        )
    parts = text.split(separator)
    if len(parts) != 2:
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a rational "
            f"'num/den' string, got {raw!r}."
        )
    try:
        num = int(parts[0].strip())
        den = int(parts[1].strip())
    except ValueError as exc:
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a rational "
            f"'num/den' string, got {raw!r}."
        ) from exc
    return (num, den)


def _parse_frame_rate(stream: dict) -> tuple[int, int]:
    avg_raw = stream.get("avg_frame_rate")
    r_raw = stream.get("r_frame_rate")

    def _valid(pair: tuple[int, int]) -> bool:
        num, den = pair
        return num > 0 and den > 0

    if isinstance(avg_raw, str) and avg_raw.strip():
        pair = _parse_rational(avg_raw, "avg_frame_rate")
        if _valid(pair):
            return pair
        if pair == (0, 0):
            pass  # ffprobe sentinel for unknown; fall through to r_frame_rate
        else:
            raise ProbeError(
                f"malformed ffprobe output: avg_frame_rate {avg_raw!r} "
                "is not a positive rational."
            )
    elif avg_raw is not None and not (isinstance(avg_raw, str) and not avg_raw.strip()):
        # Present but not a usable string (e.g. numeric) -> malformed.
        if avg_raw is not None and not (isinstance(avg_raw, str) and avg_raw.strip() == ""):
            if avg_raw is not None and "avg_frame_rate" in stream and not isinstance(avg_raw, str):
                raise ProbeError(
                    f"malformed ffprobe output: avg_frame_rate must be a rational "
                    f"'num/den' string, got {avg_raw!r}."
                )
    if isinstance(r_raw, str) and r_raw.strip():
        pair = _parse_rational(r_raw, "r_frame_rate")
        if _valid(pair):
            return pair
        raise ProbeError(
            f"malformed ffprobe output: r_frame_rate {r_raw!r} "
            "is not a positive rational."
        )
    if r_raw is not None and "r_frame_rate" in stream and not isinstance(r_raw, str):
        raise ProbeError(
            f"malformed ffprobe output: r_frame_rate must be a rational "
            f"'num/den' string, got {r_raw!r}."
        )
    raise ProbeError(
        "malformed ffprobe output: no usable video frame rate "
        "(avg_frame_rate/r_frame_rate are missing or '0/0')."
    )


def _parse_time_base(stream: dict) -> tuple[int | None, int | None]:
    if "time_base" not in stream or stream["time_base"] is None:
        return (None, None)
    raw = stream["time_base"]
    if isinstance(raw, str) and not raw.strip():
        raise ProbeError(
            f"malformed ffprobe output: time_base must be a rational "
            f"'num/den' string, got {raw!r}."
        )
    pair = _parse_rational(raw, "time_base")
    num, den = pair
    if num <= 0 or den <= 0:
        raise ProbeError(
            f"malformed ffprobe output: time_base {raw!r} "
            "must be a positive rational."
        )
    return (num, den)


def _parse_duration(raw: object, field: str) -> float:
    if isinstance(raw, bool):
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a positive "
            f"number of seconds, got {raw!r}."
        )
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ProbeError(
                f"malformed ffprobe output: {field} must be a positive "
                f"number of seconds, got {raw!r}."
            )
        try:
            value = float(text)
        except ValueError as exc:
            raise ProbeError(
                f"malformed ffprobe output: {field} must be a positive "
                f"number of seconds, got {raw!r}."
            ) from exc
    else:
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a positive "
            f"number of seconds, got {raw!r}."
        )
    if not math.isfinite(value) or value <= 0:
        raise ProbeError(
            f"malformed ffprobe output: {field} must be a positive "
            f"finite number of seconds, got {raw!r}."
        )
    return value


def parse_ffprobe_payload(payload: dict, fingerprint: str) -> SourceMetadata:
    """Normalize a decoded ffprobe JSON document to :class:`SourceMetadata`."""
    if not isinstance(payload, dict):
        raise ProbeError(
            "malformed ffprobe output: JSON object is required, "
            f"got {type(payload).__name__}."
        )
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ProbeError("malformed probe input: fingerprint must be non-blank.")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ProbeError(
            "malformed ffprobe output: 'streams' is missing or empty; "
            "no video stream found."
        )
    video_streams = [
        entry
        for entry in streams
        if isinstance(entry, dict) and entry.get("codec_type") == "video"
    ]
    if not video_streams:
        raise ProbeError(
            "unsupported media: ffprobe found no video stream; "
            "a video stream is required."
        )
    stream = video_streams[0]

    width = stream.get("width")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ProbeError(
            f"malformed ffprobe output: video width must be a positive "
            f"integer, got {width!r}."
        )
    height = stream.get("height")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ProbeError(
            f"malformed ffprobe output: video height must be a positive "
            f"integer, got {height!r}."
        )
    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec.strip():
        raise ProbeError(
            f"malformed ffprobe output: video codec_name must be a "
            f"non-blank string, got {codec!r}."
        )
    frame_num, frame_den = _parse_frame_rate(stream)
    time_num, time_den = _parse_time_base(stream)
    rotation = extract_rotation(stream)

    duration: float | None = None
    if "duration" in stream and stream["duration"] is not None:
        duration = _parse_duration(stream["duration"], "stream duration")
    else:
        fmt = payload.get("format")
        fmt_duration: object = None
        if isinstance(fmt, dict):
            fmt_duration = fmt.get("duration")
        if fmt_duration is None:
            raise ProbeError(
                "malformed ffprobe output: video duration is missing "
                "from both stream and format entries."
            )
        duration = _parse_duration(fmt_duration, "format duration")

    try:
        return SourceMetadata(
            fingerprint=fingerprint,
            duration_seconds=duration,
            width=width,
            height=height,
            frame_rate_num=frame_num,
            frame_rate_den=frame_den,
            video_codec=codec.strip(),
            rotation_degrees=rotation,
            time_base_num=time_num,
            time_base_den=time_den,
        )
    except DomainError as exc:
        raise ProbeError(f"malformed ffprobe output: normalized metadata is invalid: {exc}.") from exc


def _missing_tool_error(ffprobe: str) -> ProbeError:
    return ProbeError(
        f"ffprobe was not found as {ffprobe!r}; "
        "install FFmpeg (which provides ffprobe) and ensure it is on PATH."
    )


def run_ffprobe(video: Path | str, *, ffprobe: str = "ffprobe") -> dict:
    """Run ffprobe with an argument array and return the decoded JSON."""
    args = build_ffprobe_args(video, ffprobe=ffprobe)
    if "/" in ffprobe or "\\" in ffprobe:
        if not Path(ffprobe).is_file():
            raise _missing_tool_error(ffprobe)
    elif shutil.which(ffprobe) is None:
        raise _missing_tool_error(ffprobe)
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise _missing_tool_error(ffprobe) from exc
    except OSError as exc:
        raise ProbeError(
            f"could not run ffprobe as {ffprobe!r}: {exc}."
        ) from exc
    if completed.returncode != 0:
        detail = ((completed.stderr or "").strip() or (completed.stdout or "").strip())
        suffix = f": {detail}" if detail else ": no output"
        raise ProbeError(
            f"ffprobe failed for {video} (exit {completed.returncode}){suffix}. "
            "Verify the file is a supported MOV/MP4 video."
        )
    try:
        payload = json.loads(completed.stdout or "")
    except json.JSONDecodeError as exc:
        raise ProbeError(
            f"ffprobe returned malformed JSON for {video}: {exc}."
        ) from exc
    if not isinstance(payload, dict):
        raise ProbeError(
            f"ffprobe returned malformed output for {video}: "
            "JSON object is required."
        )
    return payload


def probe_source(video: Path | str, *, ffprobe: str = "ffprobe") -> SourceMetadata:
    """Probe ``video`` and return normalized :class:`SourceMetadata`."""
    path = Path(video).expanduser()
    if not path.is_file():
        raise ProbeError(f"input video does not exist: {path}.")
    fingerprint = fingerprint_for_path(path)
    payload = run_ffprobe(path, ffprobe=ffprobe)
    return parse_ffprobe_payload(payload, fingerprint)


def write_source_json(metadata: SourceMetadata, dest: Path | str) -> Path:
    """Write ``metadata`` to ``dest`` atomically via temp file plus rename."""
    if not isinstance(metadata, SourceMetadata):
        raise ProbeError(
            "malformed probe input: metadata must be SourceMetadata, "
            f"got {type(metadata).__name__}."
        )
    target = Path(dest).expanduser()
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProbeError(
                f"could not create output directory {parent}: {exc}."
            ) from exc
    payload = metadata.to_json()
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=target.name + ".tmp-",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(payload)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
    except OSError as exc:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise ProbeError(
            f"could not write source metadata to {target}: {exc}."
        ) from exc
    return target
