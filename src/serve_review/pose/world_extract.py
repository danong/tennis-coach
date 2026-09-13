"""Dense native-frame world extraction for one accepted attempt (M4.7).

Runs the approved Heavy Pose Landmarker over every native decoded source
frame inside exactly one accepted unpadded attempt range
(``Attempt.detected_range``, half-open ``[start, end)``) and persists the
paired :class:`~serve_review.pose.world.WorldFrameObservation` rows
(primary ``pose_world_landmarks`` meters plus the synchronized normalized
2D companion) into the accepted ``dense-world-v1`` JSONL cache.

Behavior:

- Takes an explicit source video, an explicit attempts JSON file, one
  attempt id, and one explicit cache output path. Nothing is inferred
  from filenames and no default attempt is ever guessed.
- Validates attempt/source linkage against the probed source
  (fingerprint plus duration tolerance) and uses only the unpadded
  ``detected_range`` -- never the padded ``effective_range``.
- Preserves the source video and the attempts file (both read only);
  only the explicit cache path is written, atomically (temporary
  sibling plus ``os.replace`` via the world-cache writers).
- Native coverage: the expected grid is the exact ffprobe canonical
  PTS set filtered to ``[start, end)`` -- every actual source frame in
  range, including variable-frame-rate sources. No uniform assumed-FPS
  schedule and no rate-based downsampling are involved.
- Exactly one pose backend is opened and serialized (MediaPipe
  ``VIDEO`` mode); integer milliseconds stay strictly increasing and
  every observation preserves its canonical float exactly.
- Missing data stays honest: empty ``persons`` plus all-missing world
  joints record no-person frames; per-joint ``None`` stays per-joint.
  World coordinates come only from ``pose_world_landmarks`` via
  :func:`result_to_world_frame_observation` -- a 2D ``z`` is never
  substituted (the mapping raises instead of fabricating).
- Bounded memory: at most one decoded frame plus one observation is
  held at a time; frames stream through the FFmpeg pipe.
- Resumable and collision-safe: a complete cache whose identity and
  native frame times exactly match is a hit (no inference); an
  incomplete cache whose identity matches and whose stored times are
  an exact prefix of the expected native grid resumes after its last
  stored time; corrupt/stale-identity caches are quarantined aside
  and extraction restarts fresh. A complete cache covering different
  frames, or a partial cache when ``no_resume`` is set, raises
  :class:`WorldExtractionCollisionError` instead of silently
  replacing rows. ``overwrite=True`` replaces any existing cache.
- Cancellation (``is_cancelled``) and inference failures preserve a
  resumable partial cache whenever at least one new frame was
  inferred, never a bogus complete footer.

This module owns no filtering, no waveform metrics, no checkpoint
evidence/DP, no audio, and no sparse-cache behavior (the frozen M2.2
sparse extractor is untouched). All subprocesses use argument arrays
via the underlying media/pose adapters.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from serve_review.domain import Attempt, AttemptDocument, SourceMetadata
from serve_review.media import frames as frames_module
from serve_review.media import probe as probe_module
from serve_review.pose import mediapipe as mediapipe_module
from serve_review.pose import world as world_module
from serve_review.pose.world import WorldFrameObservation

__all__ = [
    "WORLD_EXTRACT_VERSION",
    "CACHE_FLUSH_INTERVAL_FRAMES",
    "WorldExtractionError",
    "WorldExtractionInputError",
    "WorldExtractionCollisionError",
    "WorldExtractionCancelled",
    "WorldExtractionResult",
    "extract_attempt_world",
]

#: Extractor identity (orchestration only; caches record model identity).
WORLD_EXTRACT_VERSION = "dense-world-extract-v1"
#: How often to atomically flush a resumable partial cache during extraction.
CACHE_FLUSH_INTERVAL_FRAMES = 30
#: Tolerance for attempts-duration versus probed-duration agreement.
_DURATION_TOLERANCE_SECONDS = 1e-6


class WorldExtractionError(Exception):
    """Actionable failure to extract dense-world observations."""


class WorldExtractionInputError(WorldExtractionError):
    """Invalid/missing input or linkage (CLI code 2)."""


class WorldExtractionCollisionError(WorldExtractionError):
    """Output collision without --overwrite (CLI code 1)."""


class WorldExtractionCancelled(WorldExtractionError):
    """Raised on cancellation; a partial cache may remain."""


@dataclass(frozen=True, slots=True)
class WorldExtractionResult:
    """Outcome of :func:`extract_attempt_world`."""

    video: Path
    cache_path: Path
    attempt_id: str
    attempt_start_seconds: float
    attempt_end_seconds: float
    source_fingerprint: str
    model_name: str
    model_version: str
    frame_count: int
    cached_frames: int
    inferred_frames: int
    cache_hit: bool
    complete: bool


def _fail_input(message: str) -> WorldExtractionInputError:
    return WorldExtractionInputError(message)


def _check_attempt_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _fail_input(
            f"invalid attempt id: {value!r}; expected 'serve-NNN'."
        )
    prefix = "serve-"
    body = value[len(prefix):] if value.startswith(prefix) else None
    if body is None or len(body) < 3 or not body.isdigit():
        raise _fail_input(
            f"invalid attempt id: {value!r}; expected 'serve-NNN' "
            "with at least three digits."
        )
    return value


def _check_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise _fail_input(f"invalid {name}: {value!r}; expected True or False.")
    return value


def _check_callable_or_none(name: str, value: object) -> Any:
    if value is not None and not callable(value):
        raise _fail_input(f"invalid {name}: {value!r}; expected a callable or None.")
    return value


def _poll_cancelled(is_cancelled: Callable[[], bool] | None, message: str) -> None:
    if is_cancelled is None:
        return
    if is_cancelled():
        raise WorldExtractionCancelled(message)


def _quarantine_quietly(path: Path) -> Path | None:
    try:
        return world_module.quarantine_world_cache(path)
    except world_module.CacheError:
        return None


def _load_attempt(
    attempts_file: Path, wanted_id: str
) -> tuple[AttemptDocument, Attempt]:
    try:
        raw = attempts_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail_input(
            f"could not read attempts file {attempts_file}: {exc}."
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail_input(
            f"invalid attempts JSON in {attempts_file}: {exc}."
        ) from exc
    try:
        document = AttemptDocument.from_dict(payload)
    except Exception as exc:
        raise _fail_input(
            f"invalid attempt document in {attempts_file}: {exc}."
        ) from exc
    matches = [entry for entry in document.attempts if entry.attempt_id == wanted_id]
    if not matches:
        raise _fail_input(
            f"unknown attempt_id {wanted_id!r} for this attempts file; "
            "refusing to guess across attempts."
        )
    if len(matches) != 1:
        raise _fail_input(
            f"ambiguous attempt_id {wanted_id!r}: {len(matches)} matches; "
            "refusing to guess."
        )
    return document, matches[0]


def _default_native_stream(
    video_path: Path,
    start: float,
    end: float,
    metadata: SourceMetadata,
    *,
    ffmpeg: str,
    ffprobe: str,
    is_cancelled: Callable[[], bool] | None,
) -> Iterator[Any]:
    try:
        return frames_module.iter_native_frames(
            video_path,
            start,
            end,
            source=metadata,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            is_cancelled=is_cancelled,
        )
    except frames_module.FrameCancelled as exc:
        raise WorldExtractionCancelled(
            f"dense-world extraction was cancelled: {exc}."
        ) from exc
    except frames_module.FrameError as exc:
        raise WorldExtractionError(f"could not decode native frames: {exc}.") from exc


def extract_attempt_world(
    video: Path | str,
    *,
    attempts_path: Path | str,
    attempt_id: str,
    cache_path: Path | str,
    model_path: Path | str = mediapipe_module.DEFAULT_MODEL_PATH,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
    no_resume: bool = False,
    probe_fn: Callable[[Path], Any] | None = None,
    native_frame_factory: Callable[[tuple[float, ...], Any], Iterator[Any]] | None = None,
    backend_factory: Callable[[], Any] | None = None,
    progress_callback: Callable[[int, int, WorldFrameObservation], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> WorldExtractionResult:
    """Extract dense-world observations for one accepted attempt.

    Args:
        video: Source video path; read only, never modified.
        attempts_path: Explicit attempts JSON file; read only.
        attempt_id: ``'serve-NNN'`` selecting exactly one attempt; the
            unpadded ``detected_range`` is the extraction range.
        cache_path: Explicit destination dense-world JSONL cache file.
        model_path: Approved ``.task`` artifact (used only when
            ``backend_factory`` is None; no download is attempted).
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        overwrite: When True, replace any existing cache instead of
            hitting/resuming it.
        no_resume: When True, fail on an existing incomplete cache
            instead of resuming it.
        probe_fn: Injectable ``(video_path) -> SourceMetadata``.
        native_frame_factory: Injectable ``(remaining_native_times,
            metadata) -> Iterator[SampledFrame]`` covering exactly the
            remaining expected native PTS values in order; defaults to
            streaming :func:`iter_native_frames` for the remaining
            range.
        backend_factory: Injectable ``() -> backend`` exposing
            ``infer_world(image, time_seconds)``,
            ``model_name``/``model_version``, and ``close()``; defaults
            to opening the approved Heavy landmarker with
            ``num_poses=1``. Exactly one backend is opened per call.
        progress_callback: Optional ``(done, total, observation)`` hook
            after each inferred frame.
        is_cancelled: Optional hook polled before each frame/inference.

    Returns:
        A :class:`WorldExtractionResult`; ``cache_hit`` is True only
        when a complete cache with matching identity and exact native
        times was reused without inference.
    """
    video_path = Path(video).expanduser()
    if not str(video_path):
        raise _fail_input("invalid video: expected a non-blank path.")
    attempts_file = Path(attempts_path).expanduser()
    if not str(attempts_file):
        raise _fail_input("invalid attempts path: expected a non-blank path.")
    cache_target = Path(cache_path).expanduser()
    if not str(cache_target):
        raise _fail_input("invalid cache path: expected a non-blank path.")
    overwrite_flag = _check_bool("overwrite", overwrite)
    no_resume_flag = _check_bool("no_resume", no_resume)
    if overwrite_flag and no_resume_flag:
        raise _fail_input(
            "invalid options: --overwrite and --no-resume are mutually "
            "exclusive; overwriting already discards resumable state."
        )
    wanted_id = _check_attempt_id(attempt_id)
    if not video_path.is_file():
        raise _fail_input(f"input video does not exist: {video_path}.")
    if not attempts_file.is_file():
        raise _fail_input(f"attempts file does not exist: {attempts_file}.")
    if isinstance(model_path, Path):
        model_target: Path | str = model_path.expanduser()
    elif isinstance(model_path, str):
        if not model_path.strip():
            raise _fail_input("invalid model path: expected a non-blank path.")
        model_target = Path(model_path.strip()).expanduser()
    else:
        raise _fail_input(
            f"invalid model path: {model_path!r}; expected a path."
        )
    if not isinstance(ffmpeg, str) or not ffmpeg.strip():
        raise _fail_input(f"invalid ffmpeg: {ffmpeg!r}; expected a non-blank name.")
    if not isinstance(ffprobe, str) or not ffprobe.strip():
        raise _fail_input(f"invalid ffprobe: {ffprobe!r}; expected a non-blank name.")
    _check_callable_or_none("probe_fn", probe_fn)
    _check_callable_or_none("native_frame_factory", native_frame_factory)
    _check_callable_or_none("backend_factory", backend_factory)
    _check_callable_or_none("progress_callback", progress_callback)
    _check_callable_or_none("is_cancelled", is_cancelled)
    if cache_target.exists() and cache_target.is_dir():
        raise _fail_input(
            f"invalid cache path {cache_target}: destination is a directory."
        )

    # Guard: the cache must never collide with the read-only inputs.
    try:
        resolved_cache = cache_target.resolve()
        for label, candidate in (
            ("attempts", attempts_file),
            ("video", video_path),
        ):
            try:
                if candidate.resolve() == resolved_cache:
                    raise _fail_input(
                        f"cache path collides with {label} input {candidate}; "
                        "choose a distinct cache file (source/M3 files are "
                        "never modified)."
                    )
            except OSError:
                pass
    except WorldExtractionInputError:
        raise

    # 1. Probe (fingerprint + duration) without modifying the source.
    resolve_probe = probe_fn if probe_fn is not None else (
        lambda path: probe_module.probe_source(path, ffprobe=ffprobe)
    )
    try:
        metadata = resolve_probe(video_path)
    except (WorldExtractionError, WorldExtractionCancelled):
        raise
    except Exception as exc:
        raise WorldExtractionError(
            f"could not probe source video {video_path}: {exc}."
        ) from exc
    if not isinstance(metadata, SourceMetadata):
        raise _fail_input(
            f"probe returned {type(metadata).__name__}; expected SourceMetadata."
        )
    try:
        duration = float(metadata.duration_seconds)
        fingerprint = str(metadata.fingerprint)
    except (AttributeError, TypeError, ValueError) as exc:
        raise _fail_input(f"invalid source metadata for {video_path}: {exc}.") from exc
    if not fingerprint.strip():
        raise _fail_input(f"invalid source metadata for {video_path}: blank fingerprint.")
    if not math.isfinite(duration) or duration <= 0:
        raise _fail_input(
            f"invalid source metadata for {video_path}: bad duration {duration!r}."
        )

    # 2. Attempt linkage: fingerprint + duration, then the single attempt.
    document, attempt = _load_attempt(attempts_file, wanted_id)
    if document.source_fingerprint != metadata.fingerprint:
        raise _fail_input(
            "attempts source fingerprint does not match the source video "
            f"(attempts={document.source_fingerprint!r} != "
            f"source={metadata.fingerprint!r})."
        )
    if (
        abs(document.source_duration_seconds - metadata.duration_seconds)
        > _DURATION_TOLERANCE_SECONDS
    ):
        raise _fail_input(
            "attempts source duration does not match the source video "
            f"(attempts={document.source_duration_seconds!r} != "
            f"source={metadata.duration_seconds!r})."
        )
    # Accepted unpadded range only: detected_range, never effective_range.
    try:
        start = float(attempt.detected_range.start_seconds)
        end = float(attempt.detected_range.end_seconds)
    except (AttributeError, TypeError, ValueError) as exc:
        raise _fail_input(f"invalid detected range for {wanted_id}: {exc}.") from exc
    if not math.isfinite(start) or not math.isfinite(end) or not end > start:
        raise _fail_input(
            f"invalid detected range [{start!r}, {end!r}) for {wanted_id}."
        )
    if start < 0 or end > duration + frames_module.FRAME_MATCH_TOLERANCE_SECONDS:
        raise _fail_input(
            f"detected range [{start!r}, {end!r}) for {wanted_id} lies outside "
            f"the source timeline [0, {duration})."
        )
    end = min(end, duration)

    # 3. Expected native grid: exact ffprobe PTS in [start, end).
    try:
        expected = frames_module.native_frame_times(
            video_path, start, end, source=metadata, ffprobe=ffprobe
        )
    except frames_module.FrameError as exc:
        raise WorldExtractionError(f"could not list native frames: {exc}.") from exc
    total = len(expected)
    if total == 0:  # pragma: no cover - native_frame_times already raises
        raise WorldExtractionError(
            f"native range [{start!r}, {end!r}) holds no decodable source frame."
        )

    # 4. Open exactly one backend (serialized VIDEO inference).
    if backend_factory is not None:
        try:
            backend = backend_factory()
        except (WorldExtractionError, WorldExtractionCancelled):
            raise
        except Exception as exc:
            raise WorldExtractionError(
                f"could not open pose backend: {exc}."
            ) from exc
    else:
        try:
            backend = mediapipe_module.open_mediapipe_backend(
                model_target, num_poses=1
            )
        except Exception as exc:
            raise WorldExtractionError(f"{exc}") from exc
    try:
        infer_world = getattr(backend, "infer_world", None)
        if not callable(infer_world):
            raise WorldExtractionError(
                "pose backend exposes no callable 'infer_world'; refusing to "
                "substitute sparse 2D inference for dense-world rows."
            )
        try:
            model_name = str(backend.model_name)
            model_version = str(backend.model_version)
        except Exception as exc:
            raise WorldExtractionError(
                f"pose backend exposes invalid model identity: {exc}."
            ) from exc
        if not model_name.strip() or not model_version.strip():
            raise WorldExtractionError(
                "pose backend exposes a blank model name/version; refusing "
                "to write an unidentified cache."
            )
        try:
            identity = world_module.WorldCacheIdentity(
                source_fingerprint=fingerprint,
                model_name=model_name,
                model_version=model_version,
            )
        except Exception as exc:
            raise WorldExtractionError(f"invalid cache identity: {exc}.") from exc

        # 5. Existing cache: hit, resume, quarantine-and-restart, or collide.
        cached: list[WorldFrameObservation] = []
        if cache_target.is_file() and not overwrite_flag:
            try:
                snapshot = world_module.load_world_cache(cache_target)
            except world_module.CacheStaleError as exc:
                _quarantine_quietly(cache_target)
                snapshot = None
            except world_module.CacheCorruptError as exc:
                _quarantine_quietly(cache_target)
                snapshot = None
            except world_module.CacheError as exc:
                raise WorldExtractionError(
                    f"could not read dense-world cache {cache_target}: {exc}."
                ) from exc
            if snapshot is not None:
                try:
                    world_module.require_matching_world_identity(
                        snapshot.identity, identity
                    )
                except world_module.CacheStaleError:
                    _quarantine_quietly(cache_target)
                    snapshot = None
                else:
                    stored_times = tuple(
                        frame.time_seconds for frame in snapshot.frames
                    )
                    if snapshot.complete:
                        if tuple(stored_times) == tuple(expected):
                            return WorldExtractionResult(
                                video=video_path,
                                cache_path=cache_target,
                                attempt_id=wanted_id,
                                attempt_start_seconds=start,
                                attempt_end_seconds=end,
                                source_fingerprint=fingerprint,
                                model_name=model_name,
                                model_version=model_version,
                                frame_count=len(snapshot.frames),
                                cached_frames=len(snapshot.frames),
                                inferred_frames=0,
                                cache_hit=True,
                                complete=True,
                            )
                        raise WorldExtractionCollisionError(
                            f"dense-world cache {cache_target} is complete but "
                            f"covers {len(stored_times)} frame(s) that do not "
                            f"exactly match the {total} native frame(s) in "
                            f"[{start!r}, {end!r}) for attempt {wanted_id!r}; "
                            "pass --overwrite to replace it or choose a "
                            "distinct --cache path."
                        )
                    # Incomplete: stored times must be an exact prefix.
                    prefix = tuple(expected[: len(stored_times)])
                    if tuple(stored_times) != prefix:
                        raise WorldExtractionCollisionError(
                            f"dense-world cache {cache_target} holds "
                            f"{len(stored_times)} resumable frame(s) that do "
                            "not exactly match the native PTS prefix for "
                            f"attempt {wanted_id!r} in [{start!r}, {end!r}); "
                            "pass --overwrite to restart or choose a distinct "
                            "--cache path."
                        )
                    if no_resume_flag and stored_times:
                        raise WorldExtractionCollisionError(
                            f"dense-world cache {cache_target} holds "
                            f"{len(stored_times)} resumable frame(s) and "
                            "--no-resume was requested; refusing to resume."
                        )
                    cached = list(snapshot.frames)
        elif cache_target.is_file() and overwrite_flag:
            try:
                world_module.load_world_cache(cache_target)
            except (
                world_module.CacheCorruptError,
                world_module.CacheStaleError,
            ):
                _quarantine_quietly(cache_target)
            except world_module.CacheError:
                try:
                    cache_target.unlink()
                except OSError as exc:
                    raise WorldExtractionError(
                        f"could not replace dense-world cache {cache_target}: {exc}."
                    ) from exc
            else:
                try:
                    cache_target.unlink()
                except OSError as exc:
                    raise WorldExtractionError(
                        f"could not replace dense-world cache {cache_target}: {exc}."
                    ) from exc
            cached = []

        if cached:
            if len(cached) > total:
                raise WorldExtractionError(
                    f"dense-world cache {cache_target} holds more frames "
                    f"({len(cached)}) than the {total} native frame(s) in "
                    f"[{start!r}, {end!r}); refusing to reuse."
                )
            remaining = tuple(expected[len(cached):])
        else:
            remaining = expected
        if not remaining:
            if cached:
                try:
                    world_module.write_complete_world_cache(
                        cache_target, identity, cached
                    )
                except world_module.CacheError as exc:
                    raise WorldExtractionError(
                        f"could not write dense-world cache to {cache_target}: {exc}."
                    ) from exc
                return WorldExtractionResult(
                    video=video_path,
                    cache_path=cache_target,
                    attempt_id=wanted_id,
                    attempt_start_seconds=start,
                    attempt_end_seconds=end,
                    source_fingerprint=fingerprint,
                    model_name=model_name,
                    model_version=model_version,
                    frame_count=len(cached),
                    cached_frames=len(cached),
                    inferred_frames=0,
                    cache_hit=False,
                    complete=True,
                )
            raise WorldExtractionError(  # pragma: no cover - guarded above
                f"native range [{start!r}, {end!r}) holds no decodable source frame."
            )

        # 6. Stream remaining native frames (bounded: one at a time).
        if native_frame_factory is not None:
            try:
                frame_iter = native_frame_factory(remaining, metadata)
            except (WorldExtractionError, WorldExtractionCancelled):
                raise
            except Exception as exc:
                raise WorldExtractionError(
                    f"could not stream native frames: {exc}."
                ) from exc
        else:
            resume_start = float(remaining[0])
            frame_iter = _default_native_stream(
                video_path,
                resume_start,
                end,
                metadata,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                is_cancelled=is_cancelled,
            )
        try:
            iterator = iter(frame_iter)
        except TypeError as exc:
            raise WorldExtractionError(
                f"native frame source is not iterable: {exc}."
            ) from exc

        done_base = len(cached)
        all_obs: list[WorldFrameObservation] = list(cached)
        inferred = 0
        last_obs_time: float | None = (
            float(cached[-1].time_seconds) if cached else None
        )

        def _flush_partial() -> None:
            try:
                world_module.write_partial_world_cache(
                    cache_target, identity, all_obs
                )
            except world_module.CacheError as exc:
                raise WorldExtractionError(
                    f"could not write dense-world cache to {cache_target}: {exc}."
                ) from exc

        try:
            _poll_cancelled(
                is_cancelled, "dense-world extraction was cancelled before starting."
            )
            position = 0
            while True:
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                except WorldExtractionCancelled:
                    raise
                except WorldExtractionError:
                    raise
                except frames_module.FrameCancelled as exc:
                    if inferred:
                        _flush_partial()
                    raise WorldExtractionCancelled(
                        f"dense-world extraction was cancelled: {exc}."
                    ) from exc
                except Exception as exc:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise WorldExtractionError(
                        f"could not decode native frames from {video_path}: {exc}."
                    ) from exc
                _poll_cancelled(
                    is_cancelled,
                    f"dense-world extraction was cancelled after {inferred} "
                    f"of {total} frames.",
                )
                try:
                    frame_time = float(frame.time_seconds)  # type: ignore[union-attr]
                    frame_image = frame.image  # type: ignore[union-attr]
                except (AttributeError, TypeError, ValueError) as exc:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise WorldExtractionError(
                        f"native frame source yielded a malformed frame: {exc}."
                    ) from exc
                if position >= len(remaining) or frame_time != remaining[position]:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    want = remaining[position] if position < len(remaining) else None
                    raise WorldExtractionError(
                        "native frame stream diverged from the expected PTS grid: "
                        f"got t={frame_time!r}, expected t={want!r}; refusing to "
                        "emit misaligned dense-world rows."
                    )
                try:
                    observation = infer_world(frame_image, frame_time)
                except WorldExtractionCancelled:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise
                except Exception as exc:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise WorldExtractionError(
                        f"pose inference failed at t={frame_time}s: {exc}."
                    ) from exc
                if not isinstance(observation, WorldFrameObservation):
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise WorldExtractionError(
                        "pose backend returned "
                        f"{type(observation).__name__}; expected "
                        "WorldFrameObservation (sparse 2D rows are never valid "
                        "dense-world rows)."
                    )
                try:
                    obs_time = float(observation.time_seconds)
                except (AttributeError, TypeError, ValueError) as exc:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise WorldExtractionError(
                        f"pose backend returned a malformed observation: {exc}."
                    ) from exc
                if obs_time != frame_time:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise WorldExtractionError(
                        f"pose backend altered canonical time: frame t={frame_time!r} "
                        f"produced observation t={obs_time!r}; canonical source "
                        "time must be preserved exactly."
                    )
                if last_obs_time is not None and not obs_time > last_obs_time:
                    if inferred:
                        try:
                            _flush_partial()
                        except WorldExtractionError:
                            pass
                    raise WorldExtractionError(
                        "dense-world observations must be strictly increasing: "
                        f"last {last_obs_time!r}, new {obs_time!r}."
                    )
                last_obs_time = obs_time
                all_obs.append(observation)
                inferred += 1
                position += 1
                if progress_callback is not None:
                    try:
                        progress_callback(done_base + inferred, total, observation)
                    except Exception:
                        pass
                if inferred % CACHE_FLUSH_INTERVAL_FRAMES == 0:
                    _flush_partial()
            if position != len(remaining):
                if inferred:
                    try:
                        _flush_partial()
                    except WorldExtractionError:
                        pass
                raise WorldExtractionError(
                    f"native frame stream ended after {position} of {len(remaining)} "
                    "expected frame(s); refusing to emit a truncated cache."
                )
        except WorldExtractionCancelled:
            if inferred:
                try:
                    _flush_partial()
                except WorldExtractionError:
                    pass
            raise
        except WorldExtractionError:
            if inferred and not cache_target.is_file():
                try:
                    _flush_partial()
                except WorldExtractionError:
                    pass
            raise

        # 7. Atomically publish the complete cache.
        if tuple(frame.time_seconds for frame in all_obs) != tuple(expected):
            raise WorldExtractionError(
                "dense-world frame times do not exactly match the native PTS grid; "
                "refusing to publish a misaligned cache."
            )
        try:
            world_module.write_complete_world_cache(cache_target, identity, all_obs)
        except world_module.CacheError as exc:
            raise WorldExtractionError(
                f"could not write dense-world cache to {cache_target}: {exc}."
            ) from exc
        return WorldExtractionResult(
            video=video_path,
            cache_path=cache_target,
            attempt_id=wanted_id,
            attempt_start_seconds=start,
            attempt_end_seconds=end,
            source_fingerprint=fingerprint,
            model_name=model_name,
            model_version=model_version,
            frame_count=len(all_obs),
            cached_frames=len(cached),
            inferred_frames=inferred,
            cache_hit=False,
            complete=True,
        )
    finally:
        closer = getattr(backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
