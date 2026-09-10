"""Pose extraction coordinator (M2.5).

Connects the M2.1 timestamped frame sampler, the M2.4 MediaPipe Heavy
adapter, and the M2.2 resumable JSONL cache behind one diagnostic
operation. This module owns no serve rules, no visualization, no model
downloads, and no network access.

Pipeline for :func:`extract_poses`:

1. Probe the source (fingerprint, duration) without modifying it.
2. Open exactly one pose backend (serialized ``VIDEO`` inference).
3. Build the M2.2 cache identity from source fingerprint, backend
   model name/version, and the uniform sampling schedule
   (``[sampling_start, duration)`` at ``sample_rate_hz``, independent
   of nominal FPS).
4. On an existing cache: a complete cache with matching identity is a
   hit (no inference); an incomplete cache with matching identity
   resumes after its last stored time; a corrupt or stale cache is
   quarantined aside and extraction restarts fresh.
5. Stream sampled frames one at a time (bounded memory: at most one
   decoded frame plus one yielded frame), run serialized ``VIDEO``
   inference per frame, and preserve canonical ``time_seconds``
   exactly. Empty ``persons`` lists record explicit no-person frames;
   nothing is fabricated.
6. Publish progress, honor cancellation, and write the cache
   atomically: periodic resumable partial writes during extraction
   plus a complete footer only on success. Cancellation or inference
   failure leaves a resumable partial cache, never a bogus complete
   one. The source video is only read, never modified.

All external processes use argument arrays; canonical source time
never derives from a frame index or an assumed FPS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from serve_review.media import frames as frames_module
from serve_review.media import probe as probe_module
from serve_review.pose import cache as cache_module
from serve_review.pose import mediapipe as mediapipe_module
from serve_review.pose.schema import CacheIdentity

__all__ = [
    "DEFAULT_SAMPLE_RATE_HZ",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SAMPLING_START_SECONDS",
    "CACHE_FLUSH_INTERVAL_FRAMES",
    "MAX_SAMPLE_RATE_HZ",
    "ExtractionError",
    "ExtractionCancelled",
    "ExtractionResult",
    "default_cache_path_for",
    "extract_poses",
]

#: Default uniform sampling rate (Hz); independent of nominal FPS.
DEFAULT_SAMPLE_RATE_HZ = 30.0
#: Upper bound for the requested sampling rate (mirrors M2.1).
MAX_SAMPLE_RATE_HZ = frames_module.MAX_SAMPLE_RATE_HZ
#: Default approved Heavy artifact location (no download is attempted).
DEFAULT_MODEL_PATH = mediapipe_module.DEFAULT_MODEL_PATH
#: Default start of the uniform schedule (source timeline origin).
DEFAULT_SAMPLING_START_SECONDS = 0.0
#: How often to atomically flush a resumable partial cache during extraction.
CACHE_FLUSH_INTERVAL_FRAMES = 30


class ExtractionError(Exception):
    """Actionable failure to extract pose observations."""


class ExtractionCancelled(ExtractionError):
    """Raised when extraction was cancelled; a partial cache may remain."""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Outcome of :func:`extract_poses`."""

    video: Path
    cache_path: Path
    model_name: str
    model_version: str
    sampling_rate_hz: float
    sampling_start_seconds: float
    frame_count: int
    cached_frames: int
    inferred_frames: int
    cache_hit: bool
    complete: bool


def default_cache_path_for(video: Path | str, output_dir: Path | str = "output") -> Path:
    """Return the default cache path for ``video``.

    The layout follows the design default:
    ``<output_dir>/<source-stem>/cache/pose-v1.jsonl``.
    """
    stem = Path(str(video)).stem
    if not stem:
        raise ExtractionError(f"invalid video path {video!r}: no file stem.")
    return Path(output_dir).expanduser() / stem / "cache" / "pose-v1.jsonl"


def _check_rate(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= MAX_SAMPLE_RATE_HZ
    ):
        raise ExtractionError(
            f"invalid sample rate {value!r}: expected a finite rate in "
            f"(0, {MAX_SAMPLE_RATE_HZ}]."
        )
    return float(value)


def _check_start(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ExtractionError(
            f"invalid sampling start {value!r}: expected a finite number >= 0."
        )
    return float(value)


def _check_cancelled(is_cancelled: Callable[[], bool] | None, message: str) -> None:
    if is_cancelled is None:
        return
    if not callable(is_cancelled):
        raise ExtractionError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    if is_cancelled():
        raise ExtractionCancelled(message)


def _quarantine_quietly(path: Path) -> Path | None:
    try:
        return cache_module.quarantine_corrupt(path)
    except cache_module.CacheError:
        return None


def extract_poses(
    video: Path | str,
    cache_path: Path | str,
    *,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    sampling_start_seconds: float = DEFAULT_SAMPLING_START_SECONDS,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    overwrite: bool = False,
    probe_fn: Callable[[Path], Any] | None = None,
    frame_factory: Callable[[tuple[float, ...], Any], Iterator[Any]] | None = None,
    backend_factory: Callable[[], Any] | None = None,
    progress_callback: Callable[[int, int, Any], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> ExtractionResult:
    """Extract pose observations for ``video`` into ``cache_path``.

    Args:
        video: Source video path; read only, never modified.
        cache_path: Destination JSONL cache file (atomic writes).
        sample_rate_hz: Uniform schedule rate in ``(0, 120]``.
        sampling_start_seconds: Schedule start in ``[0, duration)``.
        model_path: Approved ``.task`` artifact (used only when
            ``backend_factory`` is None; no download is attempted).
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        overwrite: When True, ignore and replace any existing cache.
        probe_fn: Injectable ``(video_path) -> SourceMetadata``; defaults
            to probing the file.
        frame_factory: Injectable ``(remaining_schedule, metadata) ->
            Iterator[SampledFrame]``; defaults to streaming
            :func:`iter_sampled_frames` for the remaining schedule.
        backend_factory: Injectable ``() -> PoseBackend``; defaults to
            opening the approved Heavy landmarker at ``model_path``.
            Exactly one backend is opened and serialized per extraction.
        progress_callback: Optional ``(done, total, observation)``
            hook invoked after each inferred frame.
        is_cancelled: Optional hook polled before each frame and each
            inference; a true return raises :class:`ExtractionCancelled`
            and leaves a resumable partial cache.

    Returns:
        An :class:`ExtractionResult`; ``cache_hit`` is True only when a
        complete cache with matching identity was reused without
        inference.

    Raises:
        ExtractionError: On invalid inputs, probe/decode/inference
            failures, or stale-schedule problems (partial progress is
            preserved as a resumable cache where applicable).
        ExtractionCancelled: On cancellation (a resumable partial cache
            is preserved when any new frame was inferred).
    """
    video_path = Path(video).expanduser()
    if not str(video_path):
        raise ExtractionError("invalid video: expected a non-blank path.")
    if not video_path.is_file():
        raise ExtractionError(f"input video does not exist: {video_path}.")
    cache_target = Path(cache_path).expanduser()
    if not str(cache_target):
        raise ExtractionError("invalid cache path: expected a non-blank path.")
    rate = _check_rate(sample_rate_hz)
    start = _check_start(sampling_start_seconds)
    if not isinstance(overwrite, bool):
        raise ExtractionError(
            f"invalid overwrite: {overwrite!r}; expected True or False."
        )
    if progress_callback is not None and not callable(progress_callback):
        raise ExtractionError(
            f"invalid progress_callback: {progress_callback!r}; "
            "expected a callable or None."
        )
    if is_cancelled is not None and not callable(is_cancelled):
        raise ExtractionError(
            f"invalid is_cancelled: {is_cancelled!r}; expected a callable or None."
        )
    if probe_fn is not None and not callable(probe_fn):
        raise ExtractionError(
            f"invalid probe_fn: {probe_fn!r}; expected a callable or None."
        )
    if frame_factory is not None and not callable(frame_factory):
        raise ExtractionError(
            f"invalid frame_factory: {frame_factory!r}; expected a callable or None."
        )
    if backend_factory is not None and not callable(backend_factory):
        raise ExtractionError(
            f"invalid backend_factory: {backend_factory!r}; expected a callable or None."
        )

    # 1. Probe (fingerprint + duration) without modifying the source.
    resolve_probe = probe_fn if probe_fn is not None else (
        lambda path: probe_module.probe_source(path, ffprobe=ffprobe)
    )
    try:
        metadata = resolve_probe(video_path)
    except ExtractionError:
        raise
    except ExtractionCancelled:
        raise
    except Exception as exc:
        raise ExtractionError(f"could not probe source video {video_path}: {exc}.") from exc
    try:
        duration = float(metadata.duration_seconds)  # type: ignore[union-attr]
        fingerprint = str(metadata.fingerprint)  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExtractionError(
            f"invalid source metadata for {video_path}: {exc}."
        ) from exc
    if not fingerprint.strip():
        raise ExtractionError(
            f"invalid source metadata for {video_path}: blank fingerprint."
        )
    if not math.isfinite(duration) or duration <= 0:
        raise ExtractionError(
            f"invalid source metadata for {video_path}: bad duration {duration!r}."
        )
    if not start < duration:
        raise ExtractionError(
            f"sampling start {start!r} lies outside the source timeline "
            f"[0, {duration}); times must satisfy 0 <= start < duration."
        )

    # 2. Full uniform schedule from (start, duration, rate) only.
    try:
        full_schedule = frames_module.build_uniform_schedule(
            duration, rate, start_seconds=start
        )
    except Exception as exc:
        raise ExtractionError(f"could not build sampling schedule: {exc}.") from exc

    # 3. Open exactly one backend (serialized VIDEO inference).
    if backend_factory is not None:
        try:
            backend = backend_factory()
        except ExtractionError:
            raise
        except ExtractionCancelled:
            raise
        except Exception as exc:
            raise ExtractionError(
                f"could not open pose backend: {exc}."
            ) from exc
    else:
        try:
            backend = mediapipe_module.open_mediapipe_backend(model_path)
        except Exception as exc:
            raise ExtractionError(f"{exc}") from exc
    try:
        try:
            model_name = str(backend.model_name)
            model_version = str(backend.model_version)
        except Exception as exc:
            raise ExtractionError(
                f"pose backend exposes invalid model identity: {exc}."
            ) from exc
        if not model_name.strip() or not model_version.strip():
            raise ExtractionError(
                "pose backend exposes a blank model name/version; "
                "refusing to write an unidentified cache."
            )
        try:
            identity = CacheIdentity(
                source_fingerprint=fingerprint,
                model_name=model_name,
                model_version=model_version,
                sampling_rate_hz=rate,
                sampling_start_seconds=start,
            )
        except Exception as exc:
            raise ExtractionError(f"invalid cache identity: {exc}.") from exc

        # 4. Existing cache: hit, resume, or quarantine-and-restart.
        cached: list[Any] = []
        if cache_target.is_file() and not overwrite:
            try:
                snapshot = cache_module.load_cache(cache_target)
            except cache_module.CacheCorruptError as exc:
                _quarantine_quietly(cache_target)
                snapshot = None
                _note_quarantine = exc
            except cache_module.CacheError as exc:
                raise ExtractionError(
                    f"could not read pose cache {cache_target}: {exc}."
                ) from exc
            if snapshot is not None:
                try:
                    cache_module.require_matching_identity(snapshot.identity, identity)
                except cache_module.CacheStaleError:
                    _quarantine_quietly(cache_target)
                    snapshot = None
                else:
                    if snapshot.complete:
                        return ExtractionResult(
                            video=video_path,
                            cache_path=cache_target,
                            model_name=model_name,
                            model_version=model_version,
                            sampling_rate_hz=rate,
                            sampling_start_seconds=start,
                            frame_count=len(snapshot.frames),
                            cached_frames=len(snapshot.frames),
                            inferred_frames=0,
                            cache_hit=True,
                            complete=True,
                        )
                    cached = list(snapshot.frames)
        elif cache_target.is_file() and overwrite:
            # Fresh start: quarantine corrupt files, otherwise remove.
            try:
                snapshot = cache_module.load_cache(cache_target)
            except cache_module.CacheCorruptError:
                _quarantine_quietly(cache_target)
            except cache_module.CacheError:
                try:
                    cache_target.unlink()
                except OSError as exc:
                    raise ExtractionError(
                        f"could not replace pose cache {cache_target}: {exc}."
                    ) from exc
            else:
                try:
                    cache_target.unlink()
                except OSError as exc:
                    raise ExtractionError(
                        f"could not replace pose cache {cache_target}: {exc}."
                    ) from exc
            cached = []

        # 5. Remaining schedule after the last cached canonical time.
        if cached:
            last_time = float(cached[-1].time_seconds)
            remaining = tuple(
                t for t in full_schedule if t > last_time + 1e-12
            )
        else:
            remaining = full_schedule

        if not remaining:
            # Schedule exhausted but no complete footer: finalize.
            if cached:
                try:
                    cache_module.write_complete_cache(cache_target, identity, cached)
                except cache_module.CacheError as exc:
                    raise ExtractionError(
                        f"could not write pose cache to {cache_target}: {exc}."
                    ) from exc
                return ExtractionResult(
                    video=video_path,
                    cache_path=cache_target,
                    model_name=model_name,
                    model_version=model_version,
                    sampling_rate_hz=rate,
                    sampling_start_seconds=start,
                    frame_count=len(cached),
                    cached_frames=len(cached),
                    inferred_frames=0,
                    cache_hit=False,
                    complete=True,
                )
            raise ExtractionError(
                "empty frame schedule: no uniform sample falls inside "
                f"[start={start}, duration={duration}) at {rate} Hz."
            )

        # 6. Stream frames (bounded: one at a time) with serialized inference.
        if frame_factory is not None:
            frame_iter = frame_factory(remaining, metadata)
        else:
            frame_iter = frames_module.iter_sampled_frames(
                video_path,
                remaining,
                source=metadata,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                is_cancelled=is_cancelled,
            )

        total = len(full_schedule)
        done_base = len(cached)
        all_obs: list[Any] = list(cached)
        inferred = 0
        last_obs_time: float | None = (
            float(cached[-1].time_seconds) if cached else None
        )

        def _flush_partial() -> None:
            try:
                cache_module.write_partial_cache(cache_target, identity, all_obs)
            except cache_module.CacheError as exc:
                raise ExtractionError(
                    f"could not write pose cache to {cache_target}: {exc}."
                ) from exc

        try:
            _check_cancelled(is_cancelled, "pose extraction was cancelled before starting.")
            iterator = iter(frame_iter)
            while True:
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                except ExtractionCancelled:
                    raise
                except ExtractionError:
                    raise
                except frames_module.FrameCancelled as exc:
                    if inferred:
                        _flush_partial()
                    raise ExtractionCancelled(
                        f"pose extraction was cancelled: {exc}."
                    ) from exc
                except Exception as exc:
                    # Distinguish cancellation surfaced as FrameCancelled subclass
                    # already handled; other frame errors are fatal but preserve
                    # partial progress when available.
                    if inferred:
                        try:
                            _flush_partial()
                        except ExtractionError:
                            pass
                    raise ExtractionError(
                        f"could not sample frames from {video_path}: {exc}."
                    ) from exc
                _check_cancelled(
                    is_cancelled,
                    f"pose extraction was cancelled after {inferred} "
                    f"of ~{total} frames.",
                )
                try:
                    frame_time = float(frame.time_seconds)  # type: ignore[union-attr]
                    frame_image = frame.image  # type: ignore[union-attr]
                except (AttributeError, TypeError, ValueError) as exc:
                    if inferred:
                        try:
                            _flush_partial()
                        except ExtractionError:
                            pass
                    raise ExtractionError(
                        f"frame source yielded a malformed frame: {exc}."
                    ) from exc
                try:
                    observation = backend.infer(frame_image, frame_time)
                except ExtractionCancelled:
                    if inferred:
                        try:
                            _flush_partial()
                        except ExtractionError:
                            pass
                    raise
                except Exception as exc:
                    # Preserve resumable progress, then report honestly.
                    # Map backend cancellation-like errors is not possible
                    # generically; every failure here is an error.
                    if inferred:
                        try:
                            _flush_partial()
                        except ExtractionError:
                            pass
                    # Backend error types already carry actionable messages.
                    raise ExtractionError(
                        f"pose inference failed at t={frame_time}s: {exc}."
                    ) from exc
                try:
                    obs_time = float(observation.time_seconds)  # type: ignore[union-attr]
                except (AttributeError, TypeError, ValueError) as exc:
                    if inferred:
                        try:
                            _flush_partial()
                        except ExtractionError:
                            pass
                    raise ExtractionError(
                        f"pose backend returned a malformed observation: {exc}."
                    ) from exc
                if obs_time != frame_time:
                    if inferred:
                        try:
                            _flush_partial()
                        except ExtractionError:
                            pass
                    raise ExtractionError(
                        f"pose backend altered canonical time: frame "
                        f"t={frame_time!r} produced observation "
                        f"t={obs_time!r}; canonical source time must be "
                        "preserved exactly."
                    )
                if last_obs_time is not None and not obs_time > last_obs_time:
                    if inferred:
                        try:
                            _flush_partial()
                        except ExtractionError:
                            pass
                    raise ExtractionError(
                        f"pose observations must be strictly increasing: last "
                        f"{last_obs_time!r}, new {obs_time!r}."
                    )
                last_obs_time = obs_time
                all_obs.append(observation)
                inferred += 1
                if progress_callback is not None:
                    try:
                        progress_callback(done_base + inferred, total, observation)
                    except Exception:
                        pass
                if inferred % CACHE_FLUSH_INTERVAL_FRAMES == 0:
                    _flush_partial()
        except ExtractionCancelled:
            if inferred:
                try:
                    _flush_partial()
                except ExtractionError:
                    pass
            raise
        except ExtractionError:
            if inferred and not cache_target.is_file():
                try:
                    _flush_partial()
                except ExtractionError:
                    pass
            raise

        # 7. Atomically publish the complete cache.
        try:
            cache_module.write_complete_cache(cache_target, identity, all_obs)
        except cache_module.CacheError as exc:
            raise ExtractionError(
                f"could not write pose cache to {cache_target}: {exc}."
            ) from exc
        return ExtractionResult(
            video=video_path,
            cache_path=cache_target,
            model_name=model_name,
            model_version=model_version,
            sampling_rate_hz=rate,
            sampling_start_seconds=start,
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
