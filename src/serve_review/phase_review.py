"""Local phase-review renderer (human visual review of M4 keyframes).

Samples the original upright source at each available/partial stage's
selected keyframe time, pairs it with the exact/nearest cached pose
observation within an explicit bounded tolerance, renders the existing
skeleton overlay with pure NumPy primitives, and burns a deterministic
readable raster caption directly into a JPEG. Unavailable (or
unsupported) phases produce no fabricated image but appear in the
deterministic review index.

Pure-domain helpers (caption layout, manifest building) are separated
from file I/O, FFmpeg subprocesses, and pose-cache access. No pose is
ever inferred here: only complete, identity-compatible caches are
reused. Contact is labelled with provenance and source time explicitly
and is never claimed as exact visual observation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from serve_review.domain import STAGE_ORDER, PhaseDocument, SourceMetadata
from serve_review.media import frames as frames_module
from serve_review.media import probe as probe_module
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose import overlay as overlay_module
from serve_review.pose.mediapipe import MODEL_NAME as _POSE_MODEL_NAME
from serve_review.pose.mediapipe import MODEL_VERSION as _POSE_MODEL_VERSION
from serve_review.pose.schema import CacheIdentity, FrameObservation

__all__ = [
    "PHASE_REVIEW_VERSION",
    "PHASE_REVIEW_SCHEMA_VERSION",
    "REVIEW_DIRNAME",
    "REVIEW_JSON_FILENAME",
    "INDEX_HTML_FILENAME",
    "MANIFEST_TXT_FILENAME",
    "POSE_SUPPORT_TOLERANCE_SECONDS",
    "ReviewError",
    "ReviewCancelled",
    "ReviewResult",
    "default_checkpoints_path_for",
    "default_review_dir_for",
    "find_nearest_pose",
    "build_caption_lines",
    "render_caption",
    "run_review",
]

#: Review pipeline identity recorded in the manifest.
PHASE_REVIEW_VERSION = "phase-review-v1"
#: Schema version of the emitted review.json index.
PHASE_REVIEW_SCHEMA_VERSION = 1
#: Default review directory name under the session directory.
REVIEW_DIRNAME = "review-phases"
#: Machine-readable deterministic manifest filename.
REVIEW_JSON_FILENAME = "review.json"
#: Human-readable deterministic index filename.
INDEX_HTML_FILENAME = "index.html"
#: Plain-text deterministic manifest filename.
MANIFEST_TXT_FILENAME = "MANIFEST.txt"
#: Explicit bounded timing tolerance for cached pose support.
#: A selected keyframe renders only when the nearest cached pose
#: observation lies within this window; otherwise the phase is
#: recorded as unsupported with no fabricated image. The delta is
#: always labelled on rendered images.
POSE_SUPPORT_TOLERANCE_SECONDS = 0.05

#: Tolerance for checkpoint-duration versus probed-duration agreement
#: (same standard as analyze).
_DURATION_TOLERANCE_SECONDS = 1e-6


class ReviewError(Exception):
    """Actionable phase-review failure at one stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}: {self.message}"


class ReviewCancelled(ReviewError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """Outcome of :func:`run_review`."""

    video: Path
    session_dir: Path
    review_dir: Path
    checkpoints_path: Path
    cache_path: Path
    review_json: Path
    index_html: Path
    manifest_txt: Path
    image_count: int
    entry_count: int
    empty: bool


def default_checkpoints_path_for(
    video: Path | str, output_dir: Path | str = "output"
) -> Path:
    """Return the deterministic checkpoints path for ``video``."""
    stem = Path(str(video)).stem
    if not stem:
        raise ReviewError("validate", f"invalid video path {video!r}: no file stem.")
    return Path(output_dir).expanduser() / stem / "checkpoints.json"


def default_review_dir_for(
    video: Path | str, output_dir: Path | str = "output"
) -> Path:
    """Return the deterministic review directory for ``video``."""
    stem = Path(str(video)).stem
    if not stem:
        raise ReviewError("validate", f"invalid video path {video!r}: no file stem.")
    return Path(output_dir).expanduser() / stem / REVIEW_DIRNAME


def _safe_component(name: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)
    cleaned = cleaned.strip("_")
    return cleaned or "item"


# --- Minimal bundled bitmap caption renderer (pure NumPy, no PIL) ---------

# 3x5 font: each glyph is 5 rows of 3 cells ('#' ink, '.' paper).
_FONT_3X5: dict[str, tuple[str, ...]] = {
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": (".##", "#..", "#.#", "#.#", ".##"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("#.#", "###", "###", "###", "#.#"),
    "O": (".#.", "#.#", "#.#", "#.#", ".#."),
    "P": ("##.", "#.#", "##.", "#..", "#.."),
    "Q": (".#.", "#.#", "#.#", "##.", ".##"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "S": (".##", "#..", ".#.", "..#", "##."),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", "#.#", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", "###", "#..", "###"),
    "3": ("###", "..#", ".##", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", ".#.", ".#.", ".#."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
    " ": ("...", "...", "...", "...", "..."),
    "-": ("...", "...", "###", "...", "..."),
    ".": ("...", "...", "...", "...", ".#."),
    ":": ("...", ".#.", "...", ".#.", "..."),
    "/": ("..#", "..#", ".#.", "#..", "#.."),
    "_": ("...", "...", "...", "...", "###"),
    "(": (".##", "#..", "#..", "#..", ".##"),
    ")": ("##.", "..#", "..#", "..#", "##."),
    "+": ("...", ".#.", "###", ".#.", "..."),
    "%": ("#.#", "..#", ".#.", "#..", "#.#"),
    "=": ("...", "###", "...", "###", "..."),
    ",": ("...", "...", "...", ".#.", "#.."),
    "?": ("###", "..#", ".#.", "...", ".#."),
    "|": (".#.", ".#.", ".#.", ".#.", ".#."),
    ">": ("#..", ".#.", "..#", ".#.", "#.."),
    "<": ("..#", ".#.", "#..", ".#.", "..#"),
}

_CAPTION_FG: tuple[int, int, int] = (255, 255, 255)
_CAPTION_BG: tuple[int, int, int] = (0, 0, 0)


def _glyph_for(ch: str) -> tuple[str, ...]:
    upper = ch.upper()
    return _FONT_3X5.get(upper, _FONT_3X5["?"])


def _wrap_line(line: str, max_chars: int) -> list[str]:
    text = str(line).upper()
    if max_chars <= 0:
        return [text]
    chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    return chunks or [""]


def build_caption_lines(
    attempt_id: str,
    stage_key: str,
    *,
    requested_time: float,
    actual_time: float,
    availability: str,
    provenance: str,
    confidence: float,
    support_time: float,
    support_delta: float,
    anomalies: Sequence[str] = (),
) -> list[str]:
    """Build deterministic caption lines burned into the JPEG.

    Every rendered image labels attempt ID, canonical stage, keyframe
    source time (requested and actual sampled frame time),
    confidence/provenance, availability, cache support time/delta, and
    nonblocking anomalies. Contact stages carry an explicit estimate
    disclaimer and never claim exact visual observation.
    """
    lines = [
        f"{attempt_id} {stage_key} t={requested_time:.3f}s (frame t={actual_time:.3f}s)",
        f"avail={availability} prov={provenance} conf={confidence:.2f}",
        f"pose t={support_time:.3f}s d={support_delta:+.3f}s",
        f"anomalies: {','.join(anomalies) if anomalies else 'none'}",
    ]
    if stage_key == "contact":
        lines.append("contact estimate; not visual observation")
    return lines


def render_caption(
    image: np.ndarray,
    lines: Sequence[str],
    *,
    scale: int = 2,
) -> np.ndarray:
    """Return a copy of ``image`` with a top caption banner burned in.

    Pure NumPy rasterization using the bundled 3x5 bitmap font. The
    banner is opaque black with white glyphs; long lines wrap to fit
    the frame width so tiny frames still carry the full label.
    Deterministic: same inputs always produce identical bytes.
    """
    if not isinstance(image, np.ndarray):
        raise ReviewError("review", "invalid frame image for caption rendering.")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ReviewError("review", "invalid frame image for caption rendering.")
    if not isinstance(scale, int) or scale < 1 or scale > 4:
        raise ReviewError("review", f"invalid caption scale {scale!r}.")
    height, width, _ = image.shape
    if height <= 0 or width <= 0:
        raise ReviewError("review", "invalid frame dimensions for caption.")
    char_w = 3 * scale + 1
    max_chars = max(8, width // char_w)
    physical: list[str] = []
    for line in lines:
        physical.extend(_wrap_line(str(line), max_chars))
    line_h = 5 * scale + 2
    pad = 2 * scale
    banner_h = min(height, pad * 2 + line_h * len(physical))
    # Number of physical rows that fit in the (possibly clipped) banner.
    rows_fit = max(0, (banner_h - pad * 2 + 2) // line_h) if banner_h > pad * 2 else 0
    rows_fit = min(rows_fit, len(physical))
    canvas = np.ascontiguousarray(image.copy(), dtype=np.uint8)
    canvas[0:banner_h, 0:width] = np.array(_CAPTION_BG, dtype=np.uint8)
    fg = np.array(_CAPTION_FG, dtype=np.uint8)
    for row in range(rows_fit):
        text = physical[row]
        y0 = pad + row * line_h
        for col, ch in enumerate(text):
            glyph = _glyph_for(ch)
            x0 = pad + col * char_w
            if x0 >= width:
                break
            for gr in range(5):
                for gc in range(3):
                    if glyph[gr][gc] != "#":
                        continue
                    ys = y0 + gr * scale
                    xs = x0 + gc * scale
                    ye = min(banner_h, ys + scale)
                    xe = min(width, xs + scale)
                    ys_c = max(0, ys)
                    xs_c = max(0, xs)
                    if ye > ys_c and xe > xs_c:
                        canvas[ys_c:ye, xs_c:xe] = fg
    return canvas


def find_nearest_pose(
    observations: Sequence[FrameObservation],
    target_seconds: float,
) -> tuple[FrameObservation, float]:
    """Return the nearest cached observation to ``target`` and its delta.

    Delta is ``support_time - target`` in seconds (signed). Raises
    :class:`ReviewError` when no observations exist.
    """
    if not observations:
        raise ReviewError("pose", "pose cache holds no observations.")
    try:
        target = float(target_seconds)
    except (TypeError, ValueError) as exc:
        raise ReviewError("review", f"invalid keyframe time {target_seconds!r}.") from exc
    if not math.isfinite(target) or target < 0:
        raise ReviewError("review", f"invalid keyframe time {target_seconds!r}.")
    best: FrameObservation | None = None
    best_abs = math.inf
    for obs in observations:
        delta = abs(float(obs.time_seconds) - target)
        if delta < best_abs:
            best_abs = delta
            best = obs
    assert best is not None
    return best, float(best.time_seconds) - target


def _check_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ReviewError("validate", f"invalid {name}: {value!r}; expected True or False.")
    return value


def _remove_staging(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _encode_jpeg_ffmpeg(image: np.ndarray, dest: Path, ffmpeg: str) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ReviewError("review", "invalid annotated image for JPEG encoding.")
    height, width, _ = image.shape
    if height <= 0 or width <= 0:
        raise ReviewError("review", "invalid annotated image dimensions.")
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise ReviewError("validate", f"invalid ffmpeg executable: {ffmpeg!r}.")
    exe = ffmpeg.strip()
    args = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(dest),
    ]
    try:
        completed = subprocess.run(args, input=np.ascontiguousarray(image, dtype=np.uint8).tobytes(), capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise ReviewError("review", f"ffmpeg was not found as {exe!r}; install FFmpeg and ensure it is on PATH.") from exc
    except OSError as exc:
        raise ReviewError("review", f"could not run ffmpeg as {exe!r}: {exc}.") from exc
    if completed.returncode != 0:
        detail = ((completed.stderr or b"").decode("utf-8", "replace").strip() if isinstance(completed.stderr, bytes) else str(completed.stderr or "").strip())
        suffix = f": {detail}" if detail else ""
        raise ReviewError("review", f"ffmpeg failed encoding JPEG for {dest.name} (exit {completed.returncode}){suffix}.")
    try:
        if not dest.is_file() or dest.stat().st_size == 0:
            raise ReviewError("review", f"ffmpeg did not produce JPEG output at {dest}.")
    except OSError as exc:
        raise ReviewError("review", f"could not verify JPEG output at {dest}: {exc}.") from exc


def _write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        raise ReviewError("review", f"could not write review file {path}: {exc}.") from exc


def _build_html(entries: list[dict[str, Any]], fingerprint: str) -> str:
    rows: list[str] = []
    for entry in entries:
        img = entry.get("image")
        if img is None:
            cell = "<em>no image</em>"
        else:
            cell = f'<a href="{img}">{img}</a>'
        rows.append(
            "<tr>"
            f"<td>{entry.get('attempt_id')}</td>"
            f"<td>{entry.get('stage')}</td>"
            f"<td>{entry.get('availability')}</td>"
            f"<td>{entry.get('provenance')}</td>"
            f"<td>{entry.get('requested_source_time')}</td>"
            f"<td>{entry.get('actual_source_time')}</td>"
            f"<td>{entry.get('cache_time')}</td>"
            f"<td>{entry.get('cache_delta_seconds')}</td>"
            f"<td>{cell}</td>"
            f"<td>{entry.get('status')}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<title>Phase review</title></head><body>\n"
        f"<h1>Phase review ({fingerprint})</h1>\n"
        "<p>Keyframes are body/audio estimates at canonical source times; "
        "contact is never claimed as exact visual observation.</p>\n"
        "<table border=\"1\">\n"
        "<tr><th>attempt</th><th>stage</th><th>availability</th>"
        "<th>provenance</th><th>requested s</th><th>frame s</th>"
        "<th>pose s</th><th>delta s</th><th>image</th><th>status</th></tr>\n"
        f"{body}\n</table>\n</body></html>\n"
    )


def _build_manifest_txt(entries: list[dict[str, Any]]) -> str:
    lines = ["phase-review-v1 manifest", ""]
    for entry in entries:
        lines.append(
            f"{entry['attempt_id']} {entry['stage']} "
            f"availability={entry['availability']} "
            f"requested={entry['requested_source_time']} "
            f"actual={entry['actual_source_time']} "
            f"cache={entry['cache_time']} "
            f"delta={entry['cache_delta_seconds']} "
            f"image={entry['image'] if entry['image'] is not None else 'none'} "
            f"status={entry['status']}"
        )
    return "\n".join(lines) + "\n"


def run_review(
    video: Path | str,
    *,
    output_dir: Path | str = "output",
    checkpoints_path: Path | str | None = None,
    output: Path | str | None = None,
    cache_path: Path | str | None = None,
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    probe_fn: Callable[[Path], SourceMetadata] | None = None,
    load_cache_fn: Callable[[Path], Any] | None = None,
    sample_frame_fn: Callable[..., Any] | None = None,
    encode_jpeg_fn: Callable[[np.ndarray, Path], None] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ReviewResult:
    """Render one JPEG per available/partial phase keyframe for review.

    Args:
        video: Source video path; read only, never modified.
        output_dir: Generated output base directory.
        checkpoints_path: Explicit checkpoints JSON; defaults to
            ``<output_dir>/<source-stem>/checkpoints.json``.
        output: Explicit final review directory; defaults to
            ``<output_dir>/<source-stem>/review-phases``.
        cache_path: Explicit pose cache; defaults to
            ``<output_dir>/<source-stem>/cache/pose-v1.jsonl``.
        overwrite: Replace an existing review directory explicitly.
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        probe_fn/load_cache_fn/sample_frame_fn/encode_jpeg_fn:
            Injectable stage functions for tests.
        progress_callback: Optional ``(stage_message)`` hook.
        is_cancelled: Optional hook; true raises
            :class:`ReviewCancelled` with no partial final directory.

    Returns:
        A :class:`ReviewResult`.

    Raises:
        ReviewError: Stage-specific failure (``.stage`` names it).
        ReviewCancelled: On cancellation.
    """
    def _progress(message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                pass

    def _cancelled() -> bool:
        if is_cancelled is None:
            return False
        if not callable(is_cancelled):
            raise ReviewError("validate", f"invalid is_cancelled: {is_cancelled!r}.")
        return bool(is_cancelled())

    video_path = Path(video).expanduser()
    if not str(video_path):
        raise ReviewError("validate", "invalid video: expected a non-blank path.")
    if not video_path.is_file():
        raise ReviewError("validate", f"input video does not exist: {video_path}.")
    overwrite_flag = _check_bool(overwrite, "overwrite")
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise ReviewError("validate", f"invalid ffmpeg: {ffmpeg!r}.")
    if not isinstance(ffprobe, str) or not ffprobe.strip():
        raise ReviewError("validate", f"invalid ffprobe: {ffprobe!r}.")
    for name, fn in (
        ("probe_fn", probe_fn),
        ("load_cache_fn", load_cache_fn),
        ("sample_frame_fn", sample_frame_fn),
        ("encode_jpeg_fn", encode_jpeg_fn),
    ):
        if fn is not None and not callable(fn):
            raise ReviewError("validate", f"invalid {name}: {fn!r}.")

    out_base = Path(output_dir).expanduser()
    session_dir = out_base / video_path.stem
    resolved_checkpoints = (
        Path(checkpoints_path).expanduser()
        if checkpoints_path is not None
        else default_checkpoints_path_for(video_path, out_base)
    )
    resolved_cache = (
        Path(cache_path).expanduser()
        if cache_path is not None
        else extract_module.default_cache_path_for(video_path, out_base)
    )
    review_dir = Path(output).expanduser() if output is not None else default_review_dir_for(video_path, out_base)

    def _fail(stage: str, message: str) -> ReviewError:
        return ReviewError(stage, message)

    # --- probe ---
    _progress(f"review: probing {video_path}")
    if _cancelled():
        raise ReviewCancelled("validate", "phase review was cancelled before probing.")
    resolve_probe = probe_fn if probe_fn is not None else (lambda p: probe_module.probe_source(p, ffprobe=ffprobe))
    try:
        metadata = resolve_probe(video_path)
    except ReviewCancelled:
        raise
    except ReviewError:
        raise
    except Exception as exc:
        raise _fail("probe", f"probe failed for {video_path}: {exc}.") from exc
    if not isinstance(metadata, SourceMetadata):
        raise _fail("probe", f"probe returned {type(metadata).__name__}; expected SourceMetadata.")

    # --- checkpoints ---
    _progress("review: loading checkpoints")
    if _cancelled():
        raise ReviewCancelled("checkpoints", "phase review was cancelled before loading checkpoints.")
    if not resolved_checkpoints.is_file():
        raise _fail("checkpoints", f"checkpoints file does not exist: {resolved_checkpoints}. Run analyze for {video_path} first.")
    try:
        raw_text = resolved_checkpoints.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail("checkpoints", f"could not read checkpoints file {resolved_checkpoints}: {exc}.") from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise _fail("checkpoints", f"invalid checkpoints JSON in {resolved_checkpoints}: {exc}.") from exc
    try:
        document = PhaseDocument.from_dict(payload)
    except Exception as exc:
        raise _fail("checkpoints", f"invalid phase document in {resolved_checkpoints}: {exc}.") from exc
    if document.source_fingerprint != metadata.fingerprint:
        raise _fail(
            "checkpoints",
            f"checkpoints source fingerprint does not match the source video (checkpoints={document.source_fingerprint!r} != source={metadata.fingerprint!r}); re-analyze the matching source.",
        )
    if abs(document.source_duration_seconds - metadata.duration_seconds) > _DURATION_TOLERANCE_SECONDS:
        raise _fail(
            "checkpoints",
            f"checkpoints source duration does not match the source video (checkpoints={document.source_duration_seconds!r} != source={metadata.duration_seconds!r}); re-analyze the matching source.",
        )

    # --- pose cache (reuse only; never infer) ---
    _progress("review: loading cached poses")
    if _cancelled():
        raise ReviewCancelled("pose", "phase review was cancelled before loading the pose cache.")
    try:
        snapshot = load_cache_fn(resolved_cache) if load_cache_fn is not None else cache_module.load_cache(resolved_cache)
    except ReviewCancelled:
        raise
    except ReviewError:
        raise
    except cache_module.CacheCorruptError as exc:
        raise _fail("pose", f"pose cache is corrupt: {exc} Quarantine and re-extract rather than reusing it ({resolved_cache}).") from exc
    except cache_module.CacheStaleError as exc:
        raise _fail("pose", f"pose cache is stale: {exc} Re-extract rather than reusing cached rows.") from exc
    except cache_module.CacheError as exc:
        raise _fail("pose", f"pose cache is unavailable: {exc} Run cut or extract-poses for {video_path} first ({resolved_cache}).") from exc
    except Exception as exc:
        raise _fail("pose", f"could not read pose cache {resolved_cache}: {exc}.") from exc
    if not bool(getattr(snapshot, "complete", False)):
        raise _fail("pose", f"pose cache is incomplete (no complete footer): {resolved_cache}. Re-run extraction to a complete cache.")
    try:
        expected_identity = CacheIdentity(
            source_fingerprint=metadata.fingerprint,
            model_name=_POSE_MODEL_NAME,
            model_version=_POSE_MODEL_VERSION,
            sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
            sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
        )
    except Exception as exc:
        raise _fail("pose", f"invalid cache identity contract: {exc}.") from exc
    try:
        stored_identity = getattr(snapshot, "identity")
        if load_cache_fn is not None and not isinstance(stored_identity, CacheIdentity):
            observations_injected = tuple(getattr(snapshot, "frames"))
            for entry in observations_injected:
                if not isinstance(entry, FrameObservation):
                    raise _fail("pose", f"pose cache returned {type(entry).__name__}; expected FrameObservation rows.")
            stored_identity = expected_identity
            observations = observations_injected
        else:
            cache_module.require_matching_identity(stored_identity, expected_identity)
            observations = tuple(snapshot.frames)
    except ReviewError:
        raise
    except cache_module.CacheStaleError as exc:
        raise _fail("pose", f"pose cache is stale: {exc} Re-extract rather than reusing cached rows.") from exc
    except cache_module.CacheError as exc:
        raise _fail("pose", f"pose cache identity is invalid: {exc}.") from exc
    except Exception as exc:
        raise _fail("pose", f"could not validate pose cache {resolved_cache}: {exc}.") from exc

    # --- collision check before staging ---
    if review_dir.exists() and not overwrite_flag:
        raise _fail("review", f"output collision: {review_dir} already exists. Pass --overwrite to replace the review directory.")
    if _cancelled():
        raise ReviewCancelled("review", "phase review was cancelled before rendering.")

    # --- stage into a temp sibling, then atomically rename ---
    parent = review_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _fail("review", f"could not create output directory {parent}: {exc}.") from exc
    try:
        staging = Path(tempfile.mkdtemp(prefix=review_dir.name + ".tmp-", dir=str(parent)))
    except OSError as exc:
        raise _fail("review", f"could not stage review output: {exc}.") from exc

    entries: list[dict[str, Any]] = []
    image_count = 0
    try:
        for attempt_phase in document.attempts:
            if _cancelled():
                raise ReviewCancelled("review", "phase review was cancelled during rendering.")
            attempt_id = attempt_phase.attempt_id
            attempt_anomalies = list(attempt_phase.anomalies)
            safe_attempt = _safe_component(attempt_id)
            for stage_key in STAGE_ORDER:
                if _cancelled():
                    raise ReviewCancelled("review", "phase review was cancelled during rendering.")
                stage = attempt_phase.stages[stage_key]
                availability = stage.availability
                provenance = stage.provenance
                confidence = float(stage.confidence)
                evidence = list(stage.evidence)
                limitations = list(stage.limitations)
                keyframe = stage.keyframe_seconds
                interval = stage.interval
                base: dict[str, Any] = {
                    "attempt_id": attempt_id,
                    "stage": stage_key,
                    "availability": availability,
                    "provenance": provenance,
                    "confidence": confidence,
                    "evidence": evidence,
                    "limitations": limitations,
                    "attempt_anomalies": attempt_anomalies,
                    "interval": interval.to_dict() if interval is not None else None,
                    "requested_source_time": keyframe,
                    "actual_source_time": None,
                    "cache_time": None,
                    "cache_delta_seconds": None,
                    "image": None,
                    "status": "unavailable",
                    "reason": None,
                }
                if availability == "unavailable":
                    base["status"] = "unavailable"
                    base["reason"] = "stage_unavailable"
                    entries.append(base)
                    continue
                if keyframe is None:
                    base["status"] = "unavailable"
                    base["reason"] = "missing_keyframe"
                    entries.append(base)
                    continue
                if not math.isfinite(float(keyframe)) or float(keyframe) < 0 or float(keyframe) >= metadata.duration_seconds:
                    base["status"] = "unavailable"
                    base["reason"] = "keyframe_out_of_bounds"
                    entries.append(base)
                    continue
                try:
                    support, delta = find_nearest_pose(observations, float(keyframe))
                except ReviewError as exc:
                    raise _fail("pose", exc.message) from exc
                if abs(delta) > POSE_SUPPORT_TOLERANCE_SECONDS:
                    base["cache_time"] = float(support.time_seconds)
                    base["cache_delta_seconds"] = delta
                    base["status"] = "unavailable"
                    base["reason"] = "pose_support_beyond_tolerance"
                    entries.append(base)
                    continue
                # Sample the original upright source at the keyframe time.
                _progress(f"review: rendering {attempt_id} {stage_key}")
                try:
                    if sample_frame_fn is not None:
                        sampled = sample_frame_fn(video_path, float(keyframe), metadata)
                    else:
                        iterator = frames_module.iter_sampled_frames(
                            video_path, [float(keyframe)], source=metadata, ffmpeg=ffmpeg, ffprobe=ffprobe
                        )
                        sampled = next(iter(iterator))
                except ReviewCancelled:
                    raise
                except Exception as exc:
                    raise _fail("review", f"could not sample source frame at t={float(keyframe):.3f}s for {attempt_id} {stage_key}: {exc}.") from exc
                try:
                    actual_time = float(sampled.time_seconds)
                    frame_image = np.ascontiguousarray(sampled.image, dtype=np.uint8)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise _fail("review", f"sampled frame for {attempt_id} {stage_key} is malformed: {exc}.") from exc
                try:
                    annotated = overlay_module.render_frame(frame_image, support)
                except overlay_module.OverlayError as exc:
                    raise _fail("review", f"could not render skeleton overlay for {attempt_id} {stage_key}: {exc}.") from exc
                caption = build_caption_lines(
                    attempt_id,
                    stage_key,
                    requested_time=float(keyframe),
                    actual_time=actual_time,
                    availability=availability,
                    provenance=provenance,
                    confidence=confidence,
                    support_time=float(support.time_seconds),
                    support_delta=delta,
                    anomalies=attempt_anomalies,
                )
                try:
                    final = render_caption(annotated, caption)
                except ReviewError as exc:
                    raise _fail("review", f"could not render caption for {attempt_id} {stage_key}: {exc.message}.") from exc
                rel = f"{safe_attempt}/{_safe_component(stage_key)}.jpg"
                dest = staging / rel
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    raise _fail("review", f"could not create review subdirectory {dest.parent}: {exc}.") from exc
                try:
                    if encode_jpeg_fn is not None:
                        encode_jpeg_fn(final, dest)
                    else:
                        _encode_jpeg_ffmpeg(final, dest, ffmpeg)
                except ReviewCancelled:
                    raise
                except ReviewError:
                    raise
                except Exception as exc:
                    raise _fail("review", f"could not write review image {dest}: {exc}.") from exc
                base["actual_source_time"] = actual_time
                base["cache_time"] = float(support.time_seconds)
                base["cache_delta_seconds"] = delta
                base["image"] = rel
                base["status"] = "rendered"
                base["reason"] = None
                entries.append(base)
                image_count += 1

        # Deterministic ordering: attempt id then canonical stage order.
        order_index = {key: idx for idx, key in enumerate(STAGE_ORDER)}
        entries.sort(key=lambda e: (e["attempt_id"], order_index.get(e["stage"], 99)))

        manifest = {
            "attempts": None,
            "cache_path": str(resolved_cache),
            "checkpoints_path": str(resolved_checkpoints),
            "phase_review_version": PHASE_REVIEW_VERSION,
            "pose_support_tolerance_seconds": POSE_SUPPORT_TOLERANCE_SECONDS,
            "schema_version": PHASE_REVIEW_SCHEMA_VERSION,
            "source_duration_seconds": metadata.duration_seconds,
            "source_fingerprint": metadata.fingerprint,
            "entries": entries,
        }
        review_text = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        _write_text(staging / REVIEW_JSON_FILENAME, review_text)
        _write_text(staging / INDEX_HTML_FILENAME, _build_html(entries, metadata.fingerprint))
        _write_text(staging / MANIFEST_TXT_FILENAME, _build_manifest_txt(entries))
        _progress("review: publishing")

        if review_dir.exists():
            if not overwrite_flag:
                raise _fail("review", f"output collision: {review_dir} already exists. Pass --overwrite to replace the review directory.")
            try:
                shutil.rmtree(review_dir)
            except OSError as exc:
                raise _fail("review", f"could not replace review directory {review_dir}: {exc}.") from exc
        try:
            os.replace(staging, review_dir)
        except OSError as exc:
            raise _fail("review", f"could not publish review directory {review_dir}: {exc}.") from exc
    except ReviewCancelled:
        _remove_staging(staging)
        raise
    except ReviewError:
        _remove_staging(staging)
        raise
    except KeyboardInterrupt as exc:
        _remove_staging(staging)
        raise ReviewCancelled("review", "phase review was interrupted.") from exc
    except Exception as exc:
        _remove_staging(staging)
        raise _fail("review", f"phase review failed: {exc}.") from exc

    return ReviewResult(
        video=video_path,
        session_dir=session_dir,
        review_dir=review_dir,
        checkpoints_path=resolved_checkpoints,
        cache_path=resolved_cache,
        review_json=review_dir / REVIEW_JSON_FILENAME,
        index_html=review_dir / INDEX_HTML_FILENAME,
        manifest_txt=review_dir / MANIFEST_TXT_FILENAME,
        image_count=image_count,
        entry_count=len(entries),
        empty=len(document) == 0,
    )
