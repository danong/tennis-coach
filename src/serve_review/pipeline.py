"""End-to-end audio-visual serve-cutting pipeline (M3.5 + AV decoder).

Composes probe, cached pose extraction, v2 features, dense audio
sampling, max-pooled audio alignment, scale-invariant transient
decoding, versioned attempts JSON, an inspectable shadow side log,
and requested media export.

Frozen detector configuration: :class:`FeatureConfig`,
:class:`DecoderConfig` (transient ratio/floor, audio window, dropout
hysteresis), and :class:`PlanConfig` (padding from the caller)
defaults supplied by the orchestrator. This module never tunes
thresholds. The legacy ``range_config``/``candidates_fn`` pose-only
path remains only as an injected test bypass; the default path is
the audio-visual decoder.

Behavior:

- ``probe -> cached pose extraction -> features + dense audio ->
  max-pool align -> decode -> planned attempts -> attempts.json +
  shadows.json -> requested export``.
- Valid decoded ranges flow to the existing planner/export;
  ``aborted``/``shadow`` records flow to ``shadows.json``, an
  inspectable side log that never alters the ``attempts.json`` schema.
- Empty detection yields an honest empty attempts document and no
  media output (never substitutes the full video).
- JSON documents and media outputs are written atomically; temporary
  files are removed on failure.
- The pose cache is reused (cache hit returns without inference).
- The source video is only read, never modified.
- Stage failures raise :class:`CutError` carrying the failing stage
  name; partial media outputs created during the run are removed.

All subprocesses use argument arrays via the underlying media/pose
adapters. Canonical source time comes from probe metadata, never from
a frame index or assumed FPS.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from serve_review import __version__ as _PACKAGE_VERSION
from serve_review.detection import decoder as decoder_module
from serve_review.detection import features as features_module
from serve_review.detection import plan as plan_module
from serve_review.detection import ranges as ranges_module
from serve_review.detection.decoder import DECODER_SCHEMA_VERSION, DecoderConfig
from serve_review.detection.features import FeatureConfig
from serve_review.detection.plan import DEFAULT_METHOD_VERSION, PlanConfig
from serve_review.detection.ranges import RangeConfig
from serve_review.domain import AttemptDocument, SourceMetadata
from serve_review.media import audio as audio_module
from serve_review.media import export as export_module
from serve_review.media import probe as probe_module
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module

__all__ = [
    "PIPELINE_VERSION",
    "RUN_SCHEMA_VERSION",
    "SHADOWS_FILENAME",
    "CutError",
    "CutCancelled",
    "CutResult",
    "run_cut",
]

#: Pipeline identity recorded in run.json.
PIPELINE_VERSION = "cut-v1"
#: Schema version of the run.json document written by this module.
RUN_SCHEMA_VERSION = 1
#: Inspectable side-log filename for non-exported shadow/abort records.
#: ``shadows.json`` never alters the ``attempts.json`` schema.
SHADOWS_FILENAME = "shadows.json"


class CutError(Exception):
    """Actionable cut-pipeline failure at one stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}: {self.message}"


class CutCancelled(CutError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


@dataclass(frozen=True, slots=True)
class CutResult:
    """Outcome of :func:`run_cut`."""

    video: Path
    session_dir: Path
    source_metadata: SourceMetadata
    attempts_document: AttemptDocument
    attempts_path: Path
    source_path: Path
    run_path: Path
    cache_path: Path
    cache_hit: bool
    mode: str
    compilation: Path | None = None
    clips: tuple[Path, ...] = ()
    empty: bool = False
    shadows: tuple[Any, ...] = ()
    shadows_path: Path | None = None


def _check_padding(value: object) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CutError(
            "validate",
            f"invalid padding {value!r}; expected a number >= 0.",
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise CutError(
            "validate",
            f"invalid padding {value!r}; expected a finite number >= 0.",
        )
    return number


def _check_mode(mode: object) -> str:
    if not isinstance(mode, str) or mode not in export_module.EXPORT_MODES:
        raise CutError(
            "validate",
            f"invalid output mode {mode!r}; "
            f"expected one of {list(export_module.EXPORT_MODES)}.",
        )
    return mode


def _write_text_atomic(target: Path, text: str) -> Path:
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CutError(
                "attempts", f"could not create output directory {parent}: {exc}."
            ) from exc
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
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
    except CutError:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    except OSError as exc:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise CutError(
            "attempts", f"could not write output file {target}: {exc}."
        ) from exc
    return target


def _remove_quietly(path: Path | None) -> None:
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _remove_tmp_siblings(directory: Path, stems: tuple[str, ...]) -> None:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        name = entry.name
        for stem in stems:
            if name.startswith(stem + ".tmp-"):
                try:
                    entry.unlink(missing_ok=True)
                except OSError:
                    pass
                break


def run_cut(
    video: Path | str,
    *,
    output_dir: Path | str = "output",
    padding_seconds: float = 1.0,
    mode: str = "compilation",
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    model_path: Path | str | None = None,
    sample_rate_hz: float = extract_module.DEFAULT_SAMPLE_RATE_HZ,
    feature_config: FeatureConfig | None = None,
    range_config: RangeConfig | None = None,
    decoder_config: DecoderConfig | None = None,
    audio_step_seconds: float = audio_module.DENSE_AUDIO_STEP_SECONDS,
    probe_fn: Callable[[Path], SourceMetadata] | None = None,
    extract_fn: Callable[..., Any] | None = None,
    load_cache_fn: Callable[[Path], Any] | None = None,
    features_fn: Callable[..., Any] | None = None,
    candidates_fn: Callable[..., Any] | None = None,
    audio_fn: Callable[..., Any] | None = None,
    align_fn: Callable[..., Any] | None = None,
    decode_fn: Callable[..., Any] | None = None,
    plan_fn: Callable[..., AttemptDocument] | None = None,
    export_fn: Callable[..., dict[str, Any]] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> CutResult:
    """Run probe, cached pose extraction, AV detection, JSON, and export.

    The default detection path is audio-visual: v2 pose features plus
    dense band-limited RMS audio, max-pool aligned onto pose-frame
    source times (``F_t``), decoded with the scale-invariant transient
    validator. Valid ranges flow to the planner/export; ``aborted`` /
    ``shadow`` records flow to ``shadows.json`` (inspectable side log,
    never part of ``attempts.json``).

    Args:
        video: Source video path; read only, never modified.
        output_dir: Generated output base directory.
        padding_seconds: Symmetric context padding in seconds (>= 0).
        mode: ``compilation``/``clips``/``both``.
        overwrite: Replace existing media outputs explicitly.
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        model_path: Pose model artifact (default approved Heavy path).
        sample_rate_hz: Pose sampling rate in ``(0, 120]``.
        feature_config/range_config/decoder_config: Frozen detector
            configuration (defaults ``FeatureConfig()`` /
            ``RangeConfig()`` / ``DecoderConfig()``). ``range_config``
            is consulted only by the legacy ``candidates_fn`` bypass.
        audio_step_seconds: Dense audio grid step in seconds (default
            5 ms); must be finite and > 0.
        probe_fn/extract_fn/load_cache_fn/features_fn/candidates_fn/
        audio_fn/align_fn/decode_fn/plan_fn/export_fn: Injectable stage
            functions for tests. When None, the real adapters run.
            ``candidates_fn``, when given, bypasses the audio/decoder
            stages with pose-only candidates (legacy test path) and
            yields an empty shadow log.
        progress_callback: Optional ``(stage_message)`` hook.
        is_cancelled: Optional hook; a true return raises
            :class:`CutCancelled` and removes partial media outputs.

    Returns:
        A :class:`CutResult`. When detection is empty, ``empty`` is
        True, no export runs, and ``compilation``/``clips`` are empty.

    Raises:
        CutError: Stage-specific failure (``.stage`` names the stage).
        CutCancelled: On cancellation.
    """
    started = time.monotonic()
    stage_timings: dict[str, float] = {}

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
            raise CutError(
                "validate", f"invalid is_cancelled: {is_cancelled!r}."
            )
        return bool(is_cancelled())

    video_path = Path(video).expanduser()
    if not str(video_path):
        raise CutError("validate", "invalid video: expected a non-blank path.")
    if not video_path.is_file():
        raise CutError("validate", f"input video does not exist: {video_path}.")
    padding = _check_padding(padding_seconds)
    mode_name = _check_mode(mode)
    if not isinstance(overwrite, bool):
        raise CutError(
            "validate", f"invalid overwrite: {overwrite!r}; expected True or False."
        )
    out_base = Path(output_dir).expanduser()
    session_dir = out_base / video_path.stem
    session_label = f"output/{video_path.stem}"

    if feature_config is None:
        feature_config = FeatureConfig()
    if not isinstance(feature_config, FeatureConfig):
        raise CutError("validate", "invalid feature_config: expected FeatureConfig.")
    if range_config is None:
        range_config = RangeConfig()
    if not isinstance(range_config, RangeConfig):
        raise CutError("validate", "invalid range_config: expected RangeConfig.")
    if decoder_config is None:
        decoder_config = DecoderConfig()
    if not isinstance(decoder_config, DecoderConfig):
        raise CutError("validate", "invalid decoder_config: expected DecoderConfig.")
    import math as _math

    if (
        isinstance(audio_step_seconds, bool)
        or not isinstance(audio_step_seconds, (int, float))
        or not _math.isfinite(float(audio_step_seconds))
        or float(audio_step_seconds) <= 0.0
    ):
        raise CutError(
            "validate",
            f"invalid audio_step_seconds: {audio_step_seconds!r}; "
            "expected a finite number > 0.",
        )
    audio_step = float(audio_step_seconds)
    plan_config = PlanConfig(padding_seconds=padding)

    created_media: list[Path] = []
    compilation_path = session_dir / export_module.COMPILATION_FILENAME
    clips_dir = session_dir / export_module.CLIPS_SUBDIR
    pre_existing: set[Path] = set()
    if compilation_path.exists():
        pre_existing.add(compilation_path)
    try:
        if clips_dir.is_dir():
            for entry in clips_dir.iterdir():
                if entry.is_file():
                    pre_existing.add(entry)
    except OSError:
        pass

    def _cleanup_created_media() -> None:
        for path in created_media:
            if path not in pre_existing:
                _remove_quietly(path)
        # Export may fail after creating some outputs but before
        # returning them; remove any new media files on disk too.
        if compilation_path.exists() and compilation_path not in pre_existing:
            _remove_quietly(compilation_path)
        try:
            if clips_dir.is_dir():
                for entry in clips_dir.iterdir():
                    if entry.is_file() and entry not in pre_existing:
                        _remove_quietly(entry)
        except OSError:
            pass
        _remove_tmp_siblings(session_dir, (export_module.COMPILATION_FILENAME,))
        _remove_tmp_siblings(clips_dir, ("serve-",))

    def _fail(stage: str, message: str) -> CutError:
        _cleanup_created_media()
        return CutError(stage, message)

    error_text: str | None = None
    error_stage: str | None = None
    status = "ok"
    metadata: SourceMetadata | None = None
    document: AttemptDocument | None = None
    cache_path = extract_module.default_cache_path_for(video_path, out_base)
    cache_hit = False
    compilation: Path | None = None
    clips: list[Path] = []
    shadows: tuple[Any, ...] = ()
    source_out = session_dir / "source.json"
    attempts_out = session_dir / "attempts.json"
    shadows_out = session_dir / SHADOWS_FILENAME
    run_out = session_dir / "run.json"

    def _write_run(status_value: str) -> Path:
        elapsed = time.monotonic() - started
        payload = {
            "attempt_count": len(document) if document is not None else 0,
            "audio_step_seconds": audio_step,
            "cache_hit": cache_hit,
            "cache_path": str(cache_path),
            "compilation": str(compilation) if compilation is not None else None,
            "clips": [str(path) for path in clips],
            "decoder_config": decoder_config.to_dict(),
            "decoder_schema_version": DECODER_SCHEMA_VERSION,
            "error": error_text,
            "error_stage": error_stage,
            "feature_config": feature_config.to_dict(),
            "method_version": DEFAULT_METHOD_VERSION,
            "mode": mode_name,
            "output_dir": str(session_dir),
            "padding_seconds": padding,
            "pipeline_version": PIPELINE_VERSION,
            "run_seconds": elapsed,
            "schema_version": RUN_SCHEMA_VERSION,
            "serve_review_version": _PACKAGE_VERSION,
            "shadow_count": len(shadows),
            "shadows_path": str(shadows_out),
            "source_fingerprint": metadata.fingerprint if metadata is not None else None,
            "stage_timings_seconds": dict(stage_timings),
            "status": status_value,
        }
        text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        try:
            return _write_text_atomic(run_out, text)
        except CutError as exc:
            raise CutError("run", str(exc.message)) from exc

    try:
        # --- probe ---
        _progress(f"cut: probing {video_path}")
        if _cancelled():
            raise CutCancelled("validate", "cut was cancelled before probing.")
        probe_start = time.monotonic()
        resolve_probe = (
            probe_fn
            if probe_fn is not None
            else (lambda path: probe_module.probe_source(path, ffprobe=ffprobe))
        )
        try:
            metadata = resolve_probe(video_path)
        except CutCancelled:
            raise
        except CutError:
            raise
        except Exception as exc:
            raise _fail("probe", f"probe failed for {video_path}: {exc}.") from exc
        if not isinstance(metadata, SourceMetadata):
            raise _fail(
                "probe",
                f"probe returned {type(metadata).__name__}; "
                "expected SourceMetadata.",
            )
        stage_timings["probe"] = time.monotonic() - probe_start
        try:
            probe_module.write_source_json(metadata, source_out)
        except Exception as exc:
            raise _fail("probe", f"could not write source.json: {exc}.") from exc

        # --- cached pose extraction ---
        _progress("cut: extracting poses (cached)")
        if _cancelled():
            raise CutCancelled("pose", "cut was cancelled before pose extraction.")
        pose_start = time.monotonic()
        resolved_model = (
            Path(model_path).expanduser()
            if model_path is not None
            else Path(extract_module.DEFAULT_MODEL_PATH)
        )
        try:
            if extract_fn is not None:
                extract_result = extract_fn(
                    video_path,
                    cache_path,
                    sample_rate_hz=sample_rate_hz,
                    model_path=resolved_model,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
                cache_hit = bool(getattr(extract_result, "cache_hit", False))
            else:
                def _pose_progress(done: int, total: int, _obs: object) -> None:
                    _progress(f"cut: pose {done}/{total} frames")

                extract_result = extract_module.extract_poses(
                    video_path,
                    cache_path,
                    sample_rate_hz=sample_rate_hz,
                    model_path=resolved_model,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    overwrite=False,
                    progress_callback=_pose_progress,
                    is_cancelled=is_cancelled,
                )
                cache_hit = bool(extract_result.cache_hit)
        except CutCancelled:
            raise
        except CutError:
            raise
        except extract_module.ExtractionCancelled as exc:
            raise CutCancelled("pose", f"pose extraction was cancelled: {exc}.") from exc
        except Exception as exc:
            raise _fail("pose", f"pose extraction failed: {exc}.") from exc
        stage_timings["pose"] = time.monotonic() - pose_start

        # --- load cached observations ---
        try:
            if load_cache_fn is not None:
                snapshot = load_cache_fn(cache_path)
                observations = tuple(snapshot.frames)
            else:
                snapshot = cache_module.load_cache(cache_path)
                observations = tuple(snapshot.frames)
        except CutCancelled:
            raise
        except CutError:
            raise
        except Exception as exc:
            raise _fail("pose", f"could not read pose cache {cache_path}: {exc}.") from exc

        # --- features ---
        _progress("cut: extracting features")
        if _cancelled():
            raise CutCancelled("features", "cut was cancelled before features.")
        features_start = time.monotonic()
        try:
            if features_fn is not None:
                feature_frames = tuple(features_fn(observations, feature_config))
            else:
                feature_frames = tuple(
                    features_module.extract_features(observations, feature_config)
                )
        except CutCancelled:
            raise
        except CutError:
            raise
        except Exception as exc:
            raise _fail("detect", f"feature extraction failed: {exc}.") from exc
        stage_timings["features"] = time.monotonic() - features_start

        # --- audio-visual detection: dense audio -> max-pool -> decode ---
        # Legacy injected ``candidates_fn`` bypasses the audio/decoder
        # stages with pose-only candidates (existing unit-test path)
        # and yields an empty shadow log.
        _progress("cut: detecting serve candidates (audio-visual)")
        if _cancelled():
            raise CutCancelled("detect", "cut was cancelled before detection.")
        detect_start = time.monotonic()
        if candidates_fn is not None:
            try:
                candidates = tuple(candidates_fn(feature_frames, range_config))
            except CutCancelled:
                raise
            except CutError:
                raise
            except Exception as exc:
                raise _fail("detect", f"candidate detection failed: {exc}.") from exc
            shadows = ()
            stage_timings["detect"] = time.monotonic() - detect_start
        else:
            frame_times = [frame.time_seconds for frame in feature_frames]
            try:
                if not frame_times:
                    decode_result = decoder_module.decode_sequence((), ())
                else:
                    if _cancelled():
                        raise CutCancelled(
                            "audio", "cut was cancelled before audio sampling."
                        )
                    audio_start = time.monotonic()
                    dense_schedule = audio_module.dense_audio_schedule(
                        metadata.duration_seconds, audio_step
                    )
                    try:
                        if audio_fn is not None:
                            dense_audio = tuple(
                                audio_fn(
                                    video_path,
                                    dense_schedule,
                                    ffmpeg=ffmpeg,
                                    ffprobe=ffprobe,
                                    is_cancelled=is_cancelled,
                                )
                            )
                        else:
                            dense_audio = tuple(
                                audio_module.iter_audio_energy(
                                    video_path,
                                    dense_schedule,
                                    ffmpeg=ffmpeg,
                                    ffprobe=ffprobe,
                                    is_cancelled=is_cancelled,
                                )
                            )
                    except CutCancelled:
                        raise
                    except CutError:
                        raise
                    except audio_module.AudioCancelled as exc:
                        raise CutCancelled(
                            "audio", f"audio sampling was cancelled: {exc}."
                        ) from exc
                    except audio_module.AudioError as exc:
                        raise _fail("audio", f"audio sampling failed: {exc}.") from exc
                    except Exception as exc:
                        raise _fail("audio", f"audio sampling failed: {exc}.") from exc
                    stage_timings["audio"] = time.monotonic() - audio_start
                    if _cancelled():
                        raise CutCancelled(
                            "detect", "cut was cancelled before decoding."
                        )
                    try:
                        if align_fn is not None:
                            aligned_audio = tuple(
                                align_fn(dense_audio, tuple(frame_times))
                            )
                        else:
                            aligned_audio = audio_module.align_audio_maxpool(
                                dense_audio, tuple(frame_times)
                            )
                    except CutCancelled:
                        raise
                    except CutError:
                        raise
                    except Exception as exc:
                        raise _fail(
                            "detect", f"audio alignment failed: {exc}."
                        ) from exc
                    try:
                        if decode_fn is not None:
                            decode_result = decode_fn(
                                feature_frames, aligned_audio, decoder_config
                            )
                        else:
                            decode_result = decoder_module.decode_sequence(
                                feature_frames, aligned_audio, decoder_config
                            )
                    except CutCancelled:
                        raise
                    except CutError:
                        raise
                    except Exception as exc:
                        raise _fail("detect", f"decoding failed: {exc}.") from exc
                    if not isinstance(
                        decode_result, decoder_module.DecodeResult
                    ):
                        raise _fail(
                            "detect",
                            f"decoder returned {type(decode_result).__name__}; "
                            "expected DecodeResult.",
                        )
            except CutCancelled:
                raise
            except CutError:
                raise
            candidates = tuple(decode_result.ranges)
            shadows = tuple(decode_result.shadows)
            stage_timings["detect"] = time.monotonic() - detect_start

        # --- padding plan ---
        _progress("cut: planning attempts")
        try:
            if plan_fn is not None:
                document = plan_fn(candidates, metadata, plan_config)
            else:
                document = plan_module.plan_attempts(
                    candidates, metadata, plan_config
                )
        except CutCancelled:
            raise
        except CutError:
            raise
        except Exception as exc:
            raise _fail("plan", f"attempt planning failed: {exc}.") from exc
        if not isinstance(document, AttemptDocument):
            raise _fail(
                "plan",
                f"planner returned {type(document).__name__}; "
                "expected AttemptDocument.",
            )

        # --- attempts.json (atomic; schema unchanged, never carries shadows) ---
        try:
            _write_text_atomic(attempts_out, document.to_json())
        except CutError as exc:
            raise _fail("attempts", str(exc.message)) from exc
        except Exception as exc:
            raise _fail("attempts", f"could not write attempts.json: {exc}.") from exc

        # --- shadows.json (atomic inspectable side log; never alters
        # attempts.json) ---
        try:
            shadow_payload = {
                "decoder_schema_version": DECODER_SCHEMA_VERSION,
                "schema_version": DECODER_SCHEMA_VERSION,
                "shadows": [item.to_dict() for item in shadows],
                "source_fingerprint": metadata.fingerprint,
            }
            _write_text_atomic(
                shadows_out,
                json.dumps(shadow_payload, sort_keys=True, indent=2) + "\n",
            )
        except CutError as exc:
            raise _fail("shadows", str(exc.message)) from exc
        except Exception as exc:
            raise _fail("shadows", f"could not write shadows.json: {exc}.") from exc

        # --- empty detection: honest empty, no media ---
        if len(document) == 0 or not document.export_ranges:
            _progress("cut: no serves detected; writing empty attempts only")
            status = "empty"
            _write_run("empty")
            return CutResult(
                video=video_path,
                session_dir=session_dir,
                source_metadata=metadata,
                attempts_document=document,
                attempts_path=attempts_out,
                source_path=source_out,
                run_path=run_out,
                cache_path=cache_path,
                cache_hit=cache_hit,
                mode=mode_name,
                compilation=None,
                clips=(),
                empty=True,
                shadows=shadows,
                shadows_path=shadows_out,
            )

        # --- export ---
        _progress(f"cut: exporting {mode_name}")
        if _cancelled():
            raise CutCancelled("export", "cut was cancelled before export.")
        export_start = time.monotonic()
        try:
            export_plan = plan_module.to_export_plan(document)
        except Exception as exc:
            raise _fail("plan", f"could not build export plan: {exc}.") from exc
        try:
            if export_fn is not None:
                results = export_fn(
                    video_path,
                    export_plan,
                    session_dir,
                    mode=mode_name,
                    source=metadata,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    overwrite=overwrite,
                )
            else:
                results = export_module.export_outputs(
                    video_path,
                    export_plan,
                    session_dir,
                    mode=mode_name,
                    source=metadata,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                    overwrite=overwrite,
                    is_cancelled=is_cancelled,
                )
        except CutCancelled:
            raise
        except CutError:
            raise
        except export_module.ExportCollisionError as exc:
            raise _fail(
                "export",
                f"output collision: {exc} "
                f"Pass --overwrite to replace outputs in {session_label}.",
            ) from exc
        except export_module.ExportCancelled as exc:
            raise CutCancelled("export", f"export was cancelled: {exc}.") from exc
        except Exception as exc:
            raise _fail("export", f"export failed: {exc}.") from exc
        stage_timings["export"] = time.monotonic() - export_start

        try:
            raw_compilation = results.get("compilation")
            raw_clips = results.get("clips", [])
            compilation = Path(raw_compilation) if raw_compilation is not None else None
            clips = [Path(path) for path in list(raw_clips or [])]
        except Exception as exc:
            raise _fail("export", f"export returned malformed results: {exc}.") from exc
        created_media = ([compilation] if compilation is not None else []) + list(clips)

        status = "ok"
        _progress("cut: done")
        _write_run("ok")
        return CutResult(
            video=video_path,
            session_dir=session_dir,
            source_metadata=metadata,
            attempts_document=document,
            attempts_path=attempts_out,
            source_path=source_out,
            run_path=run_out,
            cache_path=cache_path,
            cache_hit=cache_hit,
            mode=mode_name,
            compilation=compilation,
            clips=tuple(clips),
            empty=False,
            shadows=shadows,
            shadows_path=shadows_out,
        )
    except CutCancelled as exc:
        error_stage = exc.stage
        error_text = exc.message
        try:
            _write_run("cancelled")
        except CutError:
            pass
        raise
    except CutError as exc:
        error_stage = exc.stage
        error_text = exc.message
        if document is not None or metadata is not None:
            try:
                _write_run("error")
            except CutError:
                pass
        raise
