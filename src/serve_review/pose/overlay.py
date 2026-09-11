"""Diagnostic pose overlay renderer and MP4 writer.

Renders the upright RGB frames actually sampled for inference (the M2.1
sampler output at the extraction sample rate) annotated with the 33
MediaPipe-pose joints as dots and the pose skeleton as lines. Normalized
``[0, 1]`` joint coordinates (top-left origin, ``x`` right, ``y`` down,
never mirrored) are mapped to pixel centers; frames with no detected
person -- or with only missing joints -- are returned unannotated. No
pose is ever fabricated.

Connection geometry: :func:`pose_connections` prefers the MediaPipe
``solutions`` ``POSE_CONNECTIONS`` table when the pinned runtime still
exposes it, and otherwise falls back to the bundled
:data:`BUNDLED_POSE_CONNECTIONS` table (the same 35-segment BlazePose
topology). Rasterization is always pure NumPy so this module adds no new
packages: MediaPipe is only ever consulted for the index-pair table,
never for drawing.

MP4 encoding uses the system FFmpeg through argument arrays only (the
established pipeline pattern): raw ``rgb24`` bytes stream over stdin and
are encoded with H.264 (``libx264``) at exactly the sampled frame
dimensions. Output is written to a temporary file beside the destination
and atomically renamed with :func:`os.replace` after success; the
temporary file is removed on FFmpeg failure, timeout, cancellation, or
keyboard interrupt. A failed overlay never leaves a partial final file.

This module is deterministic: the same ``(image, observation,
connections)`` inputs always produce identical bytes. It performs no
inference, no thresholding, and no media decoding.
"""

from __future__ import annotations

import importlib
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from serve_review.pose.schema import NUM_KEYPOINTS

__all__ = [
    "BUNDLED_POSE_CONNECTIONS",
    "DOT_COLOR",
    "LINE_COLOR",
    "DEFAULT_VIDEO_ENCODER",
    "OverlayError",
    "OverlayCancelled",
    "pose_connections",
    "normalized_to_pixel",
    "render_frame",
    "build_overlay_ffmpeg_args",
    "write_overlay_video",
]

#: Bundled MediaPipe pose skeleton: 35 undirected segments over the 33
#: BlazePose joints in schema order (0..32). Used whenever the pinned
#: mediapipe runtime no longer exposes ``solutions`` connection tables.
#: Every joint index appears in at least one segment.
BUNDLED_POSE_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    sorted(
        [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            (9, 10),
            (11, 12),
            (11, 13),
            (13, 15),
            (15, 17),
            (15, 19),
            (15, 21),
            (17, 19),
            (12, 14),
            (14, 16),
            (16, 18),
            (16, 20),
            (16, 22),
            (18, 20),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (24, 26),
            (25, 27),
            (26, 28),
            (27, 29),
            (28, 30),
            (29, 31),
            (30, 32),
            (27, 31),
            (28, 32),
        ]
    )
)

#: RGB color of joint dots (drawn last, so always on top of lines).
DOT_COLOR: tuple[int, int, int] = (255, 0, 0)
#: RGB color of skeleton segments.
LINE_COLOR: tuple[int, int, int] = (0, 255, 0)
#: Explicit H.264 video encoder for overlay output (never stream copy).
DEFAULT_VIDEO_ENCODER = "libx264"


class OverlayError(Exception):
    """Actionable failure to render or encode a pose overlay."""


class OverlayCancelled(OverlayError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


def _normalize_connections(raw: Any) -> tuple[tuple[int, int], ...]:
    """Validate an ``(a, b)`` index-pair table over the 33 joints."""
    try:
        pairs = list(raw)
    except TypeError as exc:
        raise OverlayError(
            f"invalid pose connection table {raw!r}: expected an iterable "
            "of index pairs."
        ) from exc
    cleaned: list[tuple[int, int]] = []
    for entry in pairs:
        try:
            first, second = entry
        except (TypeError, ValueError) as exc:
            raise OverlayError(
                f"invalid pose connection {entry!r}: expected a pair of "
                "joint indices."
            ) from exc
        for value in (first, second):
            if isinstance(value, bool) or not isinstance(value, int):
                raise OverlayError(
                    f"invalid pose connection {entry!r}: joint indices must "
                    "be integers."
                )
            if not 0 <= value < NUM_KEYPOINTS:
                raise OverlayError(
                    f"invalid pose connection {entry!r}: joint indices must "
                    f"lie in [0, {NUM_KEYPOINTS})."
                )
        if first == second:
            raise OverlayError(
                f"invalid pose connection {entry!r}: self-loops are not "
                "drawn."
            )
        cleaned.append((int(first), int(second)))
    # Undirected, deterministic order.
    return tuple(sorted({(min(a, b), max(a, b)) for a, b in cleaned}))


def pose_connections() -> tuple[tuple[int, int], ...]:
    """Return the MediaPipe pose connection table.

    Prefers the ``POSE_CONNECTIONS`` table shipped with the pinned
    mediapipe runtime (either ``mediapipe.python.solutions.pose`` or
    ``mediapipe.solutions.pose``) when it is importable and valid, and
    otherwise returns the bundled :data:`BUNDLED_POSE_CONNECTIONS`
    table. Rendering is pure NumPy in both cases, so no new packages
    are ever required.
    """
    for module_name in (
        "mediapipe.python.solutions.pose",
        "mediapipe.solutions.pose",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        raw = getattr(module, "POSE_CONNECTIONS", None)
        if raw is None:
            continue
        try:
            return _normalize_connections(raw)
        except OverlayError:
            continue
    return BUNDLED_POSE_CONNECTIONS


def _check_dimensions(width: object, height: object) -> tuple[int, int]:
    for key, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OverlayError(
                f"invalid {key}: {value!r}; expected an integer > 0."
            )
    assert isinstance(width, int) and isinstance(height, int)
    return width, height


def normalized_to_pixel(
    x: object, y: object, width: int, height: int
) -> tuple[int, int]:
    """Map an upright normalized ``(x, y)`` joint to pixel ``(col, row)``.

    The mapping is ``round(x * (width - 1))`` / ``round(y * (height -
    1))``, so ``(0, 0)`` is the top-left pixel center and ``(1, 1)`` is
    the bottom-right pixel center. Inputs are clamped to ``[0, 1]``;
    non-finite or non-numeric inputs raise :class:`OverlayError`.
    """
    w, h = _check_dimensions(width, height)
    mapped: list[int] = []
    for key, value, size in (("x", x, w), ("y", y, h)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OverlayError(
                f"invalid normalized {key}: {value!r}; expected a finite "
                "number."
            )
        number = float(value)
        if not math.isfinite(number):
            raise OverlayError(
                f"invalid normalized {key}: {value!r}; expected a finite "
                "number."
            )
        clamped = min(1.0, max(0.0, number))
        pixel = int(round(clamped * (size - 1)))
        mapped.append(min(size - 1, max(0, pixel)))
    return mapped[0], mapped[1]


def dot_radius_for(width: int, height: int) -> int:
    """Return the deterministic joint-dot radius for a frame size."""
    w, h = _check_dimensions(width, height)
    return max(2, min(w, h) // 240)


def line_thickness_for(width: int, height: int) -> int:
    """Return the deterministic skeleton-line thickness for a frame size."""
    return max(1, dot_radius_for(width, height) // 2)


def _check_image(image: Any) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise OverlayError(
            "invalid overlay frame: expected a numpy uint8 RGB array, "
            f"got {type(image).__name__}."
        )
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise OverlayError(
            "invalid overlay frame: expected a uint8 array with shape "
            f"(height, width, 3), got dtype={image.dtype} "
            f"shape={image.shape}."
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise OverlayError(
            f"invalid overlay frame shape {image.shape!r}: dimensions must "
            "be positive."
        )
    return image


def _paint_dot(
    canvas: np.ndarray,
    col: int,
    row: int,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    height, width, _ = canvas.shape
    paint = np.array(color, dtype=np.uint8)
    for yy in range(max(0, row - radius), min(height - 1, row + radius) + 1):
        for xx in range(max(0, col - radius), min(width - 1, col + radius) + 1):
            if (xx - col) * (xx - col) + (yy - row) * (yy - row) <= radius * radius:
                canvas[yy, xx] = paint


def _paint_line(
    canvas: np.ndarray,
    col0: int,
    row0: int,
    col1: int,
    row1: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    height, width, _ = canvas.shape
    paint = np.array(color, dtype=np.uint8)
    half = thickness // 2
    # Bresenham's line over pixel centers; a square brush of ``thickness``
    # paints each stepped pixel so 1-px lines stay exact.
    dcol = abs(col1 - col0)
    drow = -abs(row1 - row0)
    step_col = 1 if col0 < col1 else -1
    step_row = 1 if row0 < row1 else -1
    error = dcol + drow
    col, row = col0, row0
    while True:
        for yy in range(max(0, row - half), min(height - 1, row + half) + 1):
            for xx in range(max(0, col - half), min(width - 1, col + half) + 1):
                canvas[yy, xx] = paint
        if col == col1 and row == row1:
            break
        doubled = 2 * error
        if doubled >= drow:
            error += drow
            col += step_col
        if doubled <= dcol:
            error += dcol
            row += step_row


def render_frame(
    image: np.ndarray,
    observation: Any = None,
    *,
    connections: Iterable[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` annotated with pose landmarks/skeleton.

    Every present joint of every detected person is drawn as a dot and
    every connection whose endpoints are both present is drawn as a
    line, in normalized coordinates mapped through
    :func:`normalized_to_pixel`. A ``None`` observation, an observation
    with no persons, or a frame whose joints are all missing is
    returned as an unmodified copy -- missing data is never fabricated.
    The input array is never mutated. Rendering is deterministic: the
    same inputs always produce identical bytes.
    """
    frame = _check_image(image)
    canvas = np.ascontiguousarray(frame.copy(), dtype=np.uint8)
    height, width, _ = canvas.shape
    if connections is None:
        table = BUNDLED_POSE_CONNECTIONS
    else:
        table = _normalize_connections(connections)
    persons: Any = ()
    if observation is not None:
        persons = getattr(observation, "persons", None)
        if persons is None:
            raise OverlayError(
                "invalid pose observation: expected an object with a "
                "'persons' sequence or None."
            )
        try:
            persons = tuple(persons)
        except TypeError as exc:
            raise OverlayError(
                "invalid pose observation: 'persons' must be a sequence."
            ) from exc
    radius = dot_radius_for(width, height)
    thickness = line_thickness_for(width, height)
    dot = np.array(DOT_COLOR, dtype=np.uint8)
    _ = dot  # dots paint via _paint_dot; kept explicit for clarity.
    for person in persons:
        joints = getattr(person, "keypoints", None)
        if joints is None:
            raise OverlayError(
                "invalid person observation: expected a 'keypoints' "
                "sequence."
            )
        try:
            joints = tuple(joints)
        except TypeError as exc:
            raise OverlayError(
                "invalid person observation: 'keypoints' must be a sequence."
            ) from exc
        if len(joints) != NUM_KEYPOINTS:
            raise OverlayError(
                f"invalid person observation: expected {NUM_KEYPOINTS} "
                f"joints in schema order, got {len(joints)}."
            )
        pixels: list[tuple[int, int] | None] = []
        for joint in joints:
            if joint is None:
                pixels.append(None)
                continue
            raw_x = getattr(joint, "x", None)
            raw_y = getattr(joint, "y", None)
            if raw_x is None or raw_y is None:
                pixels.append(None)
                continue
            try:
                pixels.append(normalized_to_pixel(raw_x, raw_y, width, height))
            except OverlayError:
                # Out-of-range/non-finite joints are missing data, never
                # fabricated: skip them instead of failing the frame.
                pixels.append(None)
        for first, second in table:
            start = pixels[first]
            end = pixels[second]
            if start is None or end is None:
                continue
            _paint_line(
                canvas, start[0], start[1], end[0], end[1], LINE_COLOR, thickness
            )
        for point in pixels:
            if point is None:
                continue
            _paint_dot(canvas, point[0], point[1], radius, DOT_COLOR)
    return canvas


def _check_ffmpeg_name(ffmpeg: object) -> str:
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise OverlayError(
            f"invalid ffmpeg executable: {ffmpeg!r}; expected a non-blank "
            "string."
        )
    return ffmpeg.strip()


def _ensure_ffmpeg(ffmpeg: str) -> None:
    if "/" in ffmpeg or "\\" in ffmpeg:
        if not Path(ffmpeg).is_file():
            raise OverlayError(
                f"ffmpeg was not found as {ffmpeg!r}; install FFmpeg and "
                "ensure it is on PATH."
            )
    elif shutil.which(ffmpeg) is None:
        raise OverlayError(
            f"ffmpeg was not found as {ffmpeg!r}; install FFmpeg and "
            "ensure it is on PATH."
        )


def _format_fps(fps: float) -> str:
    text = f"{float(fps):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def build_overlay_ffmpeg_args(
    output_path: Path | str,
    *,
    width: int,
    height: int,
    fps: float,
    ffmpeg: str = "ffmpeg",
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
) -> list[str]:
    """Build the deterministic FFmpeg argument array for an overlay MP4.

    Raw ``rgb24`` frames of exactly ``width`` x ``height`` stream over
    stdin (``-i -``) at ``fps`` and are encoded with H.264
    (``libx264``); dimensions are preserved with no scaling. The same
    inputs always produce the same array; no shell interpolation is
    used. Stream copy is never used.
    """
    executable = _check_ffmpeg_name(ffmpeg)
    w, h = _check_dimensions(width, height)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise OverlayError(
            f"invalid fps: {fps!r}; expected a finite number greater than "
            "zero."
        )
    if not isinstance(video_encoder, str) or not video_encoder.strip():
        raise OverlayError(
            f"invalid video encoder: {video_encoder!r}; expected a "
            "non-blank encoder name."
        )
    if video_encoder.strip().lower() == "copy":
        raise OverlayError(
            "invalid video encoder: 'copy' would be a stream copy, which "
            "cannot encode raw annotated frames; use an H.264 encoder such "
            f"as {DEFAULT_VIDEO_ENCODER!r}."
        )
    dest = str(output_path)
    if not dest:
        raise OverlayError("invalid overlay output: expected a non-blank path.")
    return [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-framerate",
        _format_fps(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        video_encoder.strip(),
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        dest,
    ]


def _check_overlay_path(value: object) -> Path:
    if isinstance(value, Path):
        candidate = value.expanduser()
    elif isinstance(value, str):
        if not value.strip():
            raise OverlayError(
                f"invalid overlay path: {value!r}; expected a non-blank path."
            )
        candidate = Path(value.strip()).expanduser()
    else:
        raise OverlayError(
            f"invalid overlay path: {value!r}; expected a path."
        )
    if not str(candidate).strip():
        raise OverlayError("invalid overlay path: expected a non-blank path.")
    if candidate.exists() and candidate.is_dir():
        raise OverlayError(
            f"invalid overlay path {candidate}: destination is a directory."
        )
    if candidate.suffix.lower() != ".mp4":
        raise OverlayError(
            f"invalid overlay path {candidate}: expected an '.mp4' "
            "destination."
        )
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OverlayError(
            f"could not write overlay to {candidate}: parent directory "
            f"could not be created: {exc}."
        ) from exc
    return candidate


def _remove_quietly(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _terminate(proc: Any) -> None:
    try:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    except OSError:
        pass


def _check_cancelled(is_cancelled: Callable[[], bool] | None, message: str) -> None:
    if is_cancelled is None:
        return
    if not callable(is_cancelled):
        raise OverlayError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable "
            "or None."
        )
    if is_cancelled():
        raise OverlayCancelled(message)


def _read_stderr(proc: Any) -> str:
    try:
        assert proc.stderr is not None
        detail = proc.stderr.read() or b""
    except (OSError, AssertionError, ValueError):
        return ""
    if isinstance(detail, bytes):
        return detail.decode("utf-8", "replace").strip()
    return str(detail).strip()


def write_overlay_video(
    frames: Iterable[np.ndarray],
    output_path: Path | str,
    *,
    width: int,
    height: int,
    fps: float,
    ffmpeg: str = "ffmpeg",
    total_frames: int | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Encode annotated RGB ``frames`` as an H.264 MP4 overlay.

    Args:
        frames: Iterable of ``uint8`` RGB arrays, each exactly
            ``(height, width, 3)``; every frame is validated and any
            mismatch raises instead of encoding mixed dimensions. An
            empty sequence raises (an empty overlay is never written).
        output_path: Destination ``.mp4`` file. Parent directories are
            created; a directory destination or non-``.mp4`` suffix is
            rejected explicitly.
        width/height: Sampled frame dimensions (identical to the frames
            used for inference); the output preserves them exactly.
        fps: Output frame rate (the extraction sample rate).
        ffmpeg: System FFmpeg executable (argument arrays only).
        total_frames: Optional total for ``progress_callback`` reporting;
            informational only and never used to pad or truncate.
        progress_callback: Optional ``(done, total_frames)`` hook
            invoked after each encoded frame.
        is_cancelled: Optional hook polled before starting and before
            each frame; a true return raises :class:`OverlayCancelled`.

    Returns:
        The final overlay path after atomic rename.

    Raises:
        OverlayError: On invalid inputs, unwritable destinations,
            per-frame mismatches, or FFmpeg failures (no partial final
            file remains).
        OverlayCancelled: On cancellation (no partial final file
            remains).
    """
    executable = _check_ffmpeg_name(ffmpeg)
    w, h = _check_dimensions(width, height)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(float(fps))
        or float(fps) <= 0
    ):
        raise OverlayError(
            f"invalid fps: {fps!r}; expected a finite number greater than "
            "zero."
        )
    dest = _check_overlay_path(output_path)
    if total_frames is not None and (
        isinstance(total_frames, bool)
        or not isinstance(total_frames, int)
        or total_frames < 0
    ):
        raise OverlayError(
            f"invalid total_frames: {total_frames!r}; expected an integer "
            ">= 0 or None."
        )
    if progress_callback is not None and not callable(progress_callback):
        raise OverlayError(
            f"invalid progress_callback: {progress_callback!r}; expected a "
            "callable or None."
        )
    if is_cancelled is not None and not callable(is_cancelled):
        raise OverlayError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable "
            "or None."
        )
    if isinstance(frames, (bytes, bytearray, str)) or not isinstance(
        frames, Iterable
    ):
        raise OverlayError(
            "invalid frames: expected an iterable of uint8 RGB arrays, "
            f"got {type(frames).__name__}."
        )
    try:
        iterator: Iterator[Any] = iter(frames)  # type: ignore[arg-type]
    except TypeError as exc:
        raise OverlayError(
            f"invalid frames: object is not iterable: {exc}."
        ) from exc
    try:
        first = next(iterator)
    except StopIteration:
        raise OverlayError(
            "no overlay frames were provided; refusing to write an empty "
            "overlay."
        ) from None
    _check_frame_shape(first, w, h, 0)
    _ensure_ffmpeg(executable)

    tmp_path: str | None = None
    proc: Any = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(dest.parent),
                prefix=dest.name + ".tmp-",
                suffix=".mp4",
                delete=False,
            ) as handle:
                tmp_path = handle.name
        except OSError as exc:
            raise OverlayError(
                f"could not write overlay to {dest}: {exc}."
            ) from exc
        args = build_overlay_ffmpeg_args(
            tmp_path, width=w, height=h, fps=float(fps), ffmpeg=executable
        )
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise OverlayError(
                f"ffmpeg was not found as {executable!r}; install FFmpeg "
                "and ensure it is on PATH."
            ) from exc
        except OSError as exc:
            raise OverlayError(
                f"could not run ffmpeg as {executable!r}: {exc}."
            ) from exc
        assert proc.stdin is not None
        _check_cancelled(is_cancelled, "overlay encoding was cancelled before starting.")
        done = 0
        pending: list[Any] = [first]
        index = 0
        while True:
            if not pending:
                try:
                    pending.append(next(iterator))
                except StopIteration:
                    break
            image = pending.pop(0)
            _check_frame_shape(image, w, h, index)
            _check_cancelled(
                is_cancelled,
                f"overlay encoding was cancelled after {done} frame(s).",
            )
            try:
                proc.stdin.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())
            except BrokenPipeError:
                detail = _read_stderr(proc)
                try:
                    proc.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                code = proc.poll()
                suffix = f": {detail}" if detail else ""
                raise OverlayError(
                    f"ffmpeg failed encoding overlay frame {index} "
                    f"(exit {code}){suffix}."
                ) from None
            except OSError as exc:
                raise OverlayError(
                    f"could not stream overlay frame {index} to ffmpeg: "
                    f"{exc}."
                ) from exc
            done += 1
            if progress_callback is not None:
                try:
                    progress_callback(done, total_frames)
                except Exception:
                    pass
            index += 1
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        detail = _read_stderr(proc)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise OverlayError(
                "ffmpeg timed out finalizing the overlay; no overlay was "
                "written."
            ) from exc
        if code != 0:
            suffix = f": {detail}" if detail else ": no output"
            raise OverlayError(
                f"ffmpeg failed encoding overlay (exit {code}){suffix}."
            )
        tmp_file = Path(tmp_path)
        if not tmp_file.is_file() or tmp_file.stat().st_size == 0:
            raise OverlayError(
                "ffmpeg did not produce overlay output; no overlay was "
                "written."
            )
        try:
            os.replace(tmp_path, dest)
        except OSError as exc:
            raise OverlayError(
                f"could not move finished overlay into place at {dest}: "
                f"{exc}."
            ) from exc
        tmp_path = None
    except BaseException:
        if proc is not None:
            _terminate(proc)
            for stream in (
                getattr(proc, "stdin", None),
                getattr(proc, "stderr", None),
            ):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass
        _remove_quietly(tmp_path)
        raise
    return dest


def _check_frame_shape(image: Any, width: int, height: int, index: int) -> None:
    frame = _check_image(image)
    if frame.shape[0] != height or frame.shape[1] != width:
        raise OverlayError(
            f"invalid overlay frame {index}: shape "
            f"({frame.shape[1]}x{frame.shape[0]}) does not match the "
            f"declared dimensions ({width}x{height}); refusing to encode "
            "mixed dimensions."
        )
