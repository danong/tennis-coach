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
from serve_review.pose import overlay as overlay_module
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
    overlay_path: Path | None = None


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


def _render_overlay_for_observations(
    video_path: Path,
    overlay_target: Path,
    full_schedule: tuple[float, ...],
    metadata: Any,
    observations: list[Any],
    *,
    rate_hz: float,
    ffmpeg: str,
    ffprobe: str,
    frame_factory: Callable[[tuple[float, ...], Any], Iterator[Any]] | None,
    progress_callback: Callable[[int, int | None], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> Path:
    """Render the diagnostic overlay MP4 for the full observation set.

    Streams the upright frames for ``full_schedule`` (the extraction
    sample rate), pairs each frame with its observation by exact
    canonical time, renders dots/skeleton lines via
    :mod:`serve_review.pose.overlay`, and encodes an H.264 MP4 with the
    same dimensions as the sampled frames. Requires
    ``len(full_schedule) == len(observations)`` and exact per-frame
    time equality; anything else raises instead of emitting a
    misaligned overlay. The complete pose cache is already published
    when this runs, so overlay failures never invalidate it.
    """
    total = len(observations)
    if total == 0:
        raise ExtractionError(
            "cannot write pose overlay: no observations are available; "
            "refusing to emit an empty overlay."
        )
    if len(full_schedule) != total:
        raise ExtractionError(
            "cannot write pose overlay: the sampling schedule holds "
            f"{len(full_schedule)} frames but {total} observations are "
            "available; refusing to misalign the overlay."
        )
    if frame_factory is not None:
        try:
            frame_source = frame_factory(full_schedule, metadata)
        except Exception as exc:
            raise ExtractionError(
                f"could not sample frames for pose overlay: {exc}."
            ) from exc
    else:
        try:
            frame_source = frames_module.iter_sampled_frames(
                video_path,
                full_schedule,
                source=metadata,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                is_cancelled=is_cancelled,
            )
        except Exception as exc:
            raise ExtractionError(
                f"could not sample frames for pose overlay: {exc}."
            ) from exc
    try:
        iterator = iter(frame_source)
    except TypeError as exc:
        raise ExtractionError(
            f"overlay frame source is not iterable: {exc}."
        ) from exc

    try:
        first = next(iterator)
    except StopIteration:
        raise ExtractionError(
            "overlay frame source produced no frames; refusing to emit "
            "an empty overlay."
        ) from None
    except ExtractionError:
        raise
    except frames_module.FrameCancelled as exc:
        raise ExtractionCancelled(
            f"pose overlay was cancelled: {exc}."
        ) from exc
    except Exception as exc:
        raise ExtractionError(
            f"could not sample frames for pose overlay: {exc}."
        ) from exc
    try:
        first_time = float(first.time_seconds)  # type: ignore[union-attr]
        first_width = int(first.width)  # type: ignore[union-attr]
        first_height = int(first.height)  # type: ignore[union-attr]
        first_image = first.image  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExtractionError(
            f"overlay frame source yielded a malformed frame: {exc}."
        ) from exc
    try:
        first_obs_time = float(observations[0].time_seconds)  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ExtractionError(
            f"pose overlay encountered a malformed observation: {exc}."
        ) from exc
    if first_time != first_obs_time:
        raise ExtractionError(
            f"cannot write pose overlay: frame t={first_time!r} does not "
            f"match observation t={first_obs_time!r}; refusing to misalign "
            "the overlay."
        )

    def _annotated() -> Iterator[Any]:
        try:
            yield overlay_module.render_frame(first_image, observations[0])
        except overlay_module.OverlayError as exc:
            raise ExtractionError(
                f"could not render pose overlay: {exc}."
            ) from exc
        for index in range(1, total):
            try:
                frame = next(iterator)
            except StopIteration:
                raise ExtractionError(
                    f"overlay frame source ended after {index} of {total} "
                    "frames; refusing to emit a truncated overlay."
                ) from None
            except ExtractionError:
                raise
            except frames_module.FrameCancelled as exc:
                raise ExtractionCancelled(
                    f"pose overlay was cancelled: {exc}."
                ) from exc
            except Exception as exc:
                raise ExtractionError(
                    f"could not sample frames for pose overlay: {exc}."
                ) from exc
            try:
                moment = float(frame.time_seconds)  # type: ignore[union-attr]
                width = int(frame.width)  # type: ignore[union-attr]
                height = int(frame.height)  # type: ignore[union-attr]
                image = frame.image  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError) as exc:
                raise ExtractionError(
                    f"overlay frame source yielded a malformed frame: {exc}."
                ) from exc
            if width != first_width or height != first_height:
                raise ExtractionError(
                    f"cannot write pose overlay: frame {index} dimensions "
                    f"({width}x{height}) differ from the first frame "
                    f"({first_width}x{first_height}); refusing to encode "
                    "mixed dimensions."
                )
            try:
                obs_time = float(observations[index].time_seconds)  # type: ignore[union-attr]
            except (AttributeError, TypeError, ValueError) as exc:
                raise ExtractionError(
                    f"pose overlay encountered a malformed observation: {exc}."
                ) from exc
            if moment != obs_time:
                raise ExtractionError(
                    f"cannot write pose overlay: frame t={moment!r} does "
                    f"not match observation t={obs_time!r}; refusing to "
                    "misalign the overlay."
                )
            try:
                yield overlay_module.render_frame(image, observations[index])
            except overlay_module.OverlayError as exc:
                raise ExtractionError(
                    f"could not render pose overlay: {exc}."
                ) from exc
        try:
            _extra = next(iterator)
        except StopIteration:
            return
        except ExtractionCancelled:
            raise
        except ExtractionError:
            raise
        except frames_module.FrameCancelled as exc:
            raise ExtractionCancelled(
                f"pose overlay was cancelled: {exc}."
            ) from exc
        except Exception as exc:
            raise ExtractionError(
                f"could not sample frames for pose overlay: {exc}."
            ) from exc
        raise ExtractionError(
            f"overlay frame source produced extra frames beyond {total} "
            "observations; refusing to emit a misaligned overlay."
        )

    try:
        return overlay_module.write_overlay_video(
            _annotated(),
            overlay_target,
            width=first_width,
            height=first_height,
            fps=rate_hz,
            ffmpeg=ffmpeg,
            total_frames=total,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
    except overlay_module.OverlayCancelled as exc:
        raise ExtractionCancelled(
            f"pose overlay was cancelled: {exc}."
        ) from exc
    except overlay_module.OverlayError as exc:
        raise ExtractionError(
            f"could not write pose overlay to {overlay_target}: {exc}."
        ) from exc


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
    overlay_path: Path | str | None = None,
    overlay_progress_callback: Callable[[int, int | None], None] | None = None,
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
        overlay_path: Optional diagnostic ``.mp4`` destination. When set,
            the upright sampled frames (extraction sample rate) are
            rendered with pose dots/skeleton lines and encoded as an
            H.264 MP4 via the system FFmpeg after the complete cache is
            published. The overlay always covers the full final frame
            set, including cache-hit runs (frames are re-streamed, never
            re-inferred). Defaults to None (off).
        overlay_progress_callback: Optional ``(done, total)`` hook for
            overlay encoding progress (``total`` is the overlay frame
            count).

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
    if overlay_path is None:
        overlay_target: Path | None = None
    elif isinstance(overlay_path, Path):
        overlay_target = overlay_path.expanduser()
    elif isinstance(overlay_path, str):
        if not overlay_path.strip():
            raise ExtractionError(
                "invalid overlay path: expected a non-blank path."
            )
        overlay_target = Path(overlay_path.strip()).expanduser()
    else:
        raise ExtractionError(
            f"invalid overlay path: {overlay_path!r}; expected a path or None."
        )
    if overlay_target is not None:
        if not str(overlay_target).strip():
            raise ExtractionError(
                "invalid overlay path: expected a non-blank path."
            )
        if overlay_target.exists() and overlay_target.is_dir():
            raise ExtractionError(
                f"invalid overlay path {overlay_target}: destination is "
                "a directory."
            )
        if overlay_target.suffix.lower() != ".mp4":
            raise ExtractionError(
                f"invalid overlay path {overlay_target}: expected an "
                "'.mp4' destination."
            )
        try:
            overlay_target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExtractionError(
                f"could not write pose overlay to {overlay_target}: {exc}."
            ) from exc
    if overlay_progress_callback is not None and not callable(
        overlay_progress_callback
    ):
        raise ExtractionError(
            f"invalid overlay_progress_callback: {overlay_progress_callback!r}; "
            "expected a callable or None."
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
                        overlay_dest: Path | None = None
                        if overlay_target is not None:
                            overlay_dest = _render_overlay_for_observations(
                                video_path,
                                overlay_target,
                                full_schedule,
                                metadata,
                                list(snapshot.frames),
                                rate_hz=rate,
                                ffmpeg=ffmpeg,
                                ffprobe=ffprobe,
                                frame_factory=frame_factory,
                                progress_callback=overlay_progress_callback,
                                is_cancelled=is_cancelled,
                            )
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
                            overlay_path=overlay_dest,
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
        overlay_dest = None
        if overlay_target is not None:
            overlay_dest = _render_overlay_for_observations(
                video_path,
                overlay_target,
                full_schedule,
                metadata,
                all_obs,
                rate_hz=rate,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                frame_factory=frame_factory,
                progress_callback=overlay_progress_callback,
                is_cancelled=is_cancelled,
            )
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
            overlay_path=overlay_dest,
        )
    finally:
        closer = getattr(backend, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
