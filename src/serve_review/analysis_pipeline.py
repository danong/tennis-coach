"""Atomic, non-blocking phase analysis orchestration (M4.5).

Loads accepted M3 attempt ranges, reuses the compatible cached pose
observations, demuxes raw-source audio once per run on the dense
existing schedule, derives contact candidate times with the existing
AV decoder's scale-invariant session-relative transient policy, builds
one M4.2 attempt grid plus M4.3 evidence plus one M4.4 phase result per
accepted attempt, and atomically writes the source-bound M4.1
``checkpoints.json`` (:class:`PhaseDocument`).

Behavior:

- Accept an explicit attempts JSON path or locate the prior ``cut``
  output deterministically at
  ``<output-dir>/<source-stem>/attempts.json``.
- Validate the attempts source fingerprint/duration against the probed
  source video; analyze only the unpadded detected ranges.
- Reuse only a complete pose cache whose identity matches the
  source/model/sampling contract; missing, incomplete, stale, or
  corrupt caches fail safely and actionably instead of being silently
  reused.
- Demux raw-source audio exactly once per run on the dense existing
  schedule (:func:`dense_audio_schedule`), max-pool align it onto the
  cached pose observation times, derive session-relative contact
  candidates with the decoder's ``audio_transient_ratio`` /
  ``audio_transient_floor`` policy (never a new absolute threshold),
  and pass the aligned audio plus candidates to M4.2.
- Run M4.2/M4.3/M4.4 for each accepted attempt in detected-range
  order; an empty attempt document yields an honest empty
  phase document; per-attempt sparse/partial outputs are valid.
- Per-attempt phase failures are isolated into honest unavailable
  attempt phases; global stages (probe/attempts/pose/audio/write)
  raise :class:`AnalyzeError` carrying the failing stage name.
- Never modifies the source video, ``attempts.json``, clips,
  compilation, or macro ranges. ``checkpoints.json`` is written to a
  temporary sibling and atomically renamed; failure/cancellation
  writes no partial final file and removes temporary siblings.
- No dense re-inference happens here: only existing cached pose rows
  are consumed.

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
from typing import Any, Callable, Sequence

from serve_review import __version__ as _PACKAGE_VERSION
from serve_review.checkpoints import evidence as evidence_module
from serve_review.checkpoints import phase_features as features_module
from serve_review.checkpoints import phase_solver as solver_module
from serve_review.checkpoints.evidence import PhaseEvidenceConfig
from serve_review.checkpoints.phase_features import PhaseFeaturesConfig
from serve_review.checkpoints.phase_solver import (
    PHASE_SOLVER_DEFAULT_CONFIG_ID,
    PHASE_SOLVER_METHOD_VERSION,
    PhaseSolverConfig,
)
from serve_review.detection import decoder as decoder_module
from serve_review.detection.decoder import DecoderConfig
from serve_review.domain import (
    STAGE_ORDER,
    Attempt,
    AttemptDocument,
    AttemptPhase,
    MediaRange,
    PhaseDocument,
    SourceMetadata,
    StagePhase,
)
from serve_review.media import audio as audio_module
from serve_review.media import probe as probe_module
from serve_review.media.audio import AudioEnergy
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose.mediapipe import MODEL_NAME as _POSE_MODEL_NAME
from serve_review.pose.mediapipe import MODEL_VERSION as _POSE_MODEL_VERSION
from serve_review.pose.schema import CacheIdentity, FrameObservation

__all__ = [
    "ANALYSIS_VERSION",
    "ANALYSIS_RUN_SCHEMA_VERSION",
    "CHECKPOINTS_FILENAME",
    "ATTEMPTS_FILENAME",
    "AnalyzeError",
    "AnalyzeCancelled",
    "AttemptFailure",
    "AnalyzeResult",
    "derive_contact_candidate_times",
    "default_attempts_path_for",
    "run_analyze",
]

#: Pipeline identity recorded for analysis runs.
ANALYSIS_VERSION = "analyze-v1"
#: Schema version of the analysis timing/error context carried on results.
ANALYSIS_RUN_SCHEMA_VERSION = 1
#: Final phase-report filename written atomically beside cut outputs.
CHECKPOINTS_FILENAME = "checkpoints.json"
#: Prior cut-output filename located deterministically when no explicit
#: attempts path is supplied.
ATTEMPTS_FILENAME = "attempts.json"

#: Tolerance for attempts-duration versus probed-duration agreement.
_DURATION_TOLERANCE_SECONDS = 1e-6


class AnalyzeError(Exception):
    """Actionable analysis-pipeline failure at one stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}: {self.message}"


class AnalyzeCancelled(AnalyzeError):
    """Raised when the ``is_cancelled`` hook reports cancellation."""


@dataclass(frozen=True, slots=True)
class AttemptFailure:
    """Isolated per-attempt phase failure (never aborts the run)."""

    attempt_id: str
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class AnalyzeResult:
    """Outcome of :func:`run_analyze`."""

    video: Path
    session_dir: Path
    source_metadata: SourceMetadata
    attempts_document: AttemptDocument
    phase_document: PhaseDocument
    attempts_path: Path
    checkpoints_path: Path
    cache_path: Path
    cache_hit: bool
    frame_count: int
    audio_sample_count: int
    candidate_count: int
    attempt_failures: tuple[AttemptFailure, ...] = ()
    stage_timings_seconds: dict | None = None
    empty: bool = False


def default_attempts_path_for(
    video: Path | str, output_dir: Path | str = "output"
) -> Path:
    """Return the deterministic prior-cut attempts path for ``video``.

    The layout follows the cut default:
    ``<output_dir>/<source-stem>/attempts.json``.
    """
    stem = Path(str(video)).stem
    if not stem:
        raise AnalyzeError(
            "validate", f"invalid video path {video!r}: no file stem."
        )
    return Path(output_dir).expanduser() / stem / ATTEMPTS_FILENAME


def _check_overwrite(value: object) -> bool:
    if not isinstance(value, bool):
        raise AnalyzeError(
            "validate",
            f"invalid overwrite: {value!r}; expected True or False.",
        )
    return value


def _check_audio_step(value: object) -> float:
    import math as _math

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise AnalyzeError(
            "validate",
            f"invalid audio_step_seconds: {value!r}; "
            "expected a finite number > 0.",
        )
    return float(value)


def _check_rate(value: object, name: str) -> float:
    import math as _math

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise AnalyzeError(
            "validate",
            f"invalid {name}: {value!r}; expected a finite number > 0.",
        )
    return float(value)


def _check_start(value: object) -> float:
    import math as _math

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not _math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise AnalyzeError(
            "validate",
            f"invalid sampling_start_seconds: {value!r}; "
            "expected a finite number >= 0.",
        )
    return float(value)


def _source_frame_rate_hz(metadata: SourceMetadata) -> float:
    try:
        rate = float(metadata.frames_per_second)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return 120.0
    import math as _math

    if not _math.isfinite(rate) or rate <= 0.0:
        return 120.0
    return rate


def derive_contact_candidate_times(
    aligned_audio: Sequence[AudioEnergy],
    config: DecoderConfig | None = None,
) -> tuple[float, ...]:
    """Derive session-relative contact candidate times from audio.

    This reuses the existing AV decoder's scale-invariant transient
    policy: the session-median baseline (via the decoder's own
    ``_session_baseline`` helper, robust to a narrow impact spike) and
    the decoder configuration's ``audio_transient_ratio`` plus the
    small ``audio_transient_floor`` against near-silence. Every
    aligned sample whose energy clears both the floor and
    ``ratio * baseline`` becomes a candidate time. No new absolute
    threshold is introduced: scaling every energy by a constant factor
    leaves the candidate set unchanged (up to the near-silence floor).

    Args:
        aligned_audio: Session-level aligned :class:`AudioEnergy`
            values in strictly increasing source-time order.
        config: Decoder configuration owning the transient
            ratio/floor (defaults to ``DecoderConfig()``).

    Returns:
        Candidate source times in strictly increasing order, possibly
        empty (honestly quiet sessions yield no candidates).
    """
    cfg = config if config is not None else DecoderConfig()
    if not isinstance(cfg, DecoderConfig):
        raise AnalyzeError(
            "audio",
            f"invalid decoder_config: {type(cfg).__name__}; "
            "expected DecoderConfig.",
        )
    items = list(aligned_audio)
    for entry in items:
        if not isinstance(entry, AudioEnergy):
            raise AnalyzeError(
                "audio",
                "invalid aligned audio: every entry must be an "
                f"AudioEnergy, got {type(entry).__name__}.",
            )
    for earlier, later in zip(items, items[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise AnalyzeError(
                "audio",
                "invalid aligned audio: times must be strictly "
                f"increasing, got {earlier.time_seconds!r} followed by "
                f"{later.time_seconds!r}.",
            )
    if not items:
        return ()
    # Session-relative baseline owned by the decoder module so the
    # transient policy cannot drift from the M3 cutting contract.
    baseline = float(decoder_module._session_baseline(items))  # noqa: SLF001
    threshold = max(float(cfg.audio_transient_floor), baseline * float(cfg.audio_transient_ratio))
    return tuple(
        float(sample.time_seconds)
        for sample in items
        if float(sample.energy) >= threshold
    )


def _unavailable_attempt_phase(
    attempt: Attempt,
    solver_config: PhaseSolverConfig,
    token: str = "phase_error",
) -> AttemptPhase:
    """Build an honest unavailable phase for an isolated attempt failure."""
    stages: dict[str, StagePhase] = {}
    for key in STAGE_ORDER:
        provenance = "audio_transient" if key == "contact" else "body_pose"
        stages[key] = StagePhase(
            availability="unavailable",
            provenance=provenance,
            confidence=0.0,
            interval=None,
            keyframe_seconds=None,
            temporal_uncertainty_seconds=None,
            evidence=(),
            limitations=(token,),
        )
    return AttemptPhase(
        attempt_id=attempt.attempt_id,
        attempt_range=MediaRange(
            start_seconds=attempt.detected_range.start_seconds,
            end_seconds=attempt.detected_range.end_seconds,
        ),
        method_version=PHASE_SOLVER_METHOD_VERSION,
        config_id=solver_config.config_id,
        stages=stages,
        structural_status="unavailable",
        anomalies=(token,),
    )


def _write_text_atomic(target: Path, text: str) -> Path:
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AnalyzeError(
                "checkpoints",
                f"could not create output directory {parent}: {exc}.",
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
    except AnalyzeError:
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
        raise AnalyzeError(
            "checkpoints", f"could not write output file {target}: {exc}."
        ) from exc
    return target


def _remove_quietly(path: Path | None) -> None:
    if path is None:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _remove_tmp_siblings(directory: Path, stem: str) -> None:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith(stem + ".tmp-"):
            try:
                entry.unlink(missing_ok=True)
            except OSError:
                pass


def run_analyze(
    video: Path | str,
    *,
    output_dir: Path | str = "output",
    attempts_path: Path | str | None = None,
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    model_name: str = _POSE_MODEL_NAME,
    model_version: str = _POSE_MODEL_VERSION,
    sample_rate_hz: float = extract_module.DEFAULT_SAMPLE_RATE_HZ,
    sampling_start_seconds: float = extract_module.DEFAULT_SAMPLING_START_SECONDS,
    audio_step_seconds: float = audio_module.DENSE_AUDIO_STEP_SECONDS,
    features_config: PhaseFeaturesConfig | None = None,
    evidence_config: PhaseEvidenceConfig | None = None,
    solver_config: PhaseSolverConfig | None = None,
    decoder_config: DecoderConfig | None = None,
    probe_fn: Callable[[Path], SourceMetadata] | None = None,
    load_cache_fn: Callable[[Path], Any] | None = None,
    audio_fn: Callable[..., Any] | None = None,
    align_fn: Callable[..., Any] | None = None,
    candidates_fn: Callable[..., Any] | None = None,
    grid_fn: Callable[..., Any] | None = None,
    evidence_fn: Callable[..., Any] | None = None,
    solver_fn: Callable[..., Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> AnalyzeResult:
    """Run attempts + cached pose/audio through M4.2/M4.3/M4.4 phases.

    Args:
        video: Source video path; read only, never modified.
        output_dir: Generated output base directory.
        attempts_path: Explicit attempts JSON; when None, the prior
            cut output is located deterministically at
            ``<output_dir>/<source-stem>/attempts.json``.
        overwrite: Replace an existing ``checkpoints.json`` explicitly.
        ffmpeg/ffprobe: Tool executables (argument arrays only).
        model_name/model_version: Pose cache identity contract.
        sample_rate_hz/sampling_start_seconds: Pose cache sampling
            contract (uniform ``[start, duration)`` grid).
        audio_step_seconds: Dense audio grid step in seconds (default
            5 ms); must be finite and > 0.
        features_config/evidence_config/solver_config: Frozen phase
            configuration (defaults are the modules' defaults).
        decoder_config: Existing transient ratio/floor policy used for
            contact candidate derivation (defaults to
            ``DecoderConfig()``); never a new absolute threshold.
        probe_fn/load_cache_fn/audio_fn/align_fn/candidates_fn/
        grid_fn/evidence_fn/solver_fn: Injectable stage functions for
            tests. When None, the real adapters run.
        progress_callback: Optional ``(stage_message)`` hook.
        is_cancelled: Optional hook; a true return raises
            :class:`AnalyzeCancelled` and writes no final file.

    Returns:
        An :class:`AnalyzeResult`. An empty attempt document yields an
        honest empty phase document (``empty`` True).

    Raises:
        AnalyzeError: Stage-specific failure (``.stage`` names the
            stage). Per-attempt M4.2/M4.3/M4.4 failures are isolated
            into unavailable attempt phases instead.
        AnalyzeCancelled: On cancellation; no final file is written.
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
            raise AnalyzeError(
                "validate", f"invalid is_cancelled: {is_cancelled!r}."
            )
        return bool(is_cancelled())

    video_path = Path(video).expanduser()
    if not str(video_path):
        raise AnalyzeError("validate", "invalid video: expected a non-blank path.")
    if not video_path.is_file():
        raise AnalyzeError(
            "validate", f"input video does not exist: {video_path}."
        )
    overwrite_flag = _check_overwrite(overwrite)
    audio_step = _check_audio_step(audio_step_seconds)
    rate = _check_rate(sample_rate_hz, "sample_rate_hz")
    start_offset = _check_start(sampling_start_seconds)
    if not isinstance(model_name, str) or not model_name.strip():
        raise AnalyzeError(
            "validate",
            f"invalid model_name: {model_name!r}; expected a non-blank string.",
        )
    if not isinstance(model_version, str) or not model_version.strip():
        raise AnalyzeError(
            "validate",
            f"invalid model_version: {model_version!r}; expected a non-blank string.",
        )
    cfg_features = features_config if features_config is not None else PhaseFeaturesConfig()
    if not isinstance(cfg_features, PhaseFeaturesConfig):
        raise AnalyzeError(
            "validate",
            f"invalid features_config: {type(cfg_features).__name__}; "
            "expected PhaseFeaturesConfig.",
        )
    cfg_evidence = evidence_config if evidence_config is not None else PhaseEvidenceConfig()
    if not isinstance(cfg_evidence, PhaseEvidenceConfig):
        raise AnalyzeError(
            "validate",
            f"invalid evidence_config: {type(cfg_evidence).__name__}; "
            "expected PhaseEvidenceConfig.",
        )
    cfg_solver = solver_config if solver_config is not None else PhaseSolverConfig()
    if not isinstance(cfg_solver, PhaseSolverConfig):
        raise AnalyzeError(
            "validate",
            f"invalid solver_config: {type(cfg_solver).__name__}; "
            "expected PhaseSolverConfig.",
        )
    cfg_decoder = decoder_config if decoder_config is not None else DecoderConfig()
    if not isinstance(cfg_decoder, DecoderConfig):
        raise AnalyzeError(
            "validate",
            f"invalid decoder_config: {type(cfg_decoder).__name__}; "
            "expected DecoderConfig.",
        )

    out_base = Path(output_dir).expanduser()
    session_dir = out_base / video_path.stem
    session_label = f"output/{video_path.stem}"
    resolved_attempts = (
        Path(attempts_path).expanduser()
        if attempts_path is not None
        else default_attempts_path_for(video_path, out_base)
    )
    checkpoints_out = session_dir / CHECKPOINTS_FILENAME
    cache_path = extract_module.default_cache_path_for(video_path, out_base)

    def _fail(stage: str, message: str) -> AnalyzeError:
        _remove_tmp_siblings(session_dir, CHECKPOINTS_FILENAME)
        return AnalyzeError(stage, message)

    metadata: SourceMetadata | None = None
    document: AttemptDocument | None = None
    phase_document: PhaseDocument | None = None

    try:
        # --- probe ---
        _progress(f"analyze: probing {video_path}")
        if _cancelled():
            raise AnalyzeCancelled("validate", "analyze was cancelled before probing.")
        probe_start = time.monotonic()
        resolve_probe = (
            probe_fn
            if probe_fn is not None
            else (lambda path: probe_module.probe_source(path, ffprobe=ffprobe))
        )
        try:
            metadata = resolve_probe(video_path)
        except AnalyzeCancelled:
            raise
        except AnalyzeError:
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

        # --- attempts ---
        _progress("analyze: loading attempts")
        if _cancelled():
            raise AnalyzeCancelled(
                "attempts", "analyze was cancelled before loading attempts."
            )
        attempts_start = time.monotonic()
        if not resolved_attempts.is_file():
            if attempts_path is not None:
                raise _fail(
                    "attempts",
                    f"attempts file does not exist: {resolved_attempts}.",
                )
            raise _fail(
                "attempts",
                f"no prior cut output at {resolved_attempts}; "
                f"run cut for {video_path} first or pass --attempts PATH.",
            )
        try:
            raw_text = resolved_attempts.read_text(encoding="utf-8")
        except OSError as exc:
            raise _fail(
                "attempts",
                f"could not read attempts file {resolved_attempts}: {exc}.",
            ) from exc
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise _fail(
                "attempts",
                f"invalid attempts JSON in {resolved_attempts}: {exc}.",
            ) from exc
        try:
            document = AttemptDocument.from_dict(payload)
        except Exception as exc:
            raise _fail(
                "attempts",
                f"invalid attempt document in {resolved_attempts}: {exc}.",
            ) from exc
        if document.source_fingerprint != metadata.fingerprint:
            raise _fail(
                "attempts",
                "attempts source fingerprint does not match the source "
                f"video (attempts={document.source_fingerprint!r} != "
                f"source={metadata.fingerprint!r}); re-cut the source and "
                "analyze the matching attempts.",
            )
        if (
            abs(
                document.source_duration_seconds - metadata.duration_seconds
            )
            > _DURATION_TOLERANCE_SECONDS
        ):
            raise _fail(
                "attempts",
                "attempts source duration does not match the source video "
                f"(attempts={document.source_duration_seconds!r} != "
                f"source={metadata.duration_seconds!r}); re-cut the source "
                "and analyze the matching attempts.",
            )
        stage_timings["attempts"] = time.monotonic() - attempts_start

        # --- cached pose observations (reuse only) ---
        _progress("analyze: loading cached poses")
        if _cancelled():
            raise AnalyzeCancelled(
                "pose", "analyze was cancelled before loading the pose cache."
            )
        pose_start = time.monotonic()
        try:
            if load_cache_fn is not None:
                snapshot = load_cache_fn(cache_path)
            else:
                snapshot = cache_module.load_cache(cache_path)
        except AnalyzeCancelled:
            raise
        except AnalyzeError:
            raise
        except cache_module.CacheCorruptError as exc:
            raise _fail(
                "pose",
                f"pose cache is corrupt: {exc} Quarantine and re-extract "
                f"rather than reusing it ({cache_path}).",
            ) from exc
        except cache_module.CacheError as exc:
            raise _fail(
                "pose",
                f"pose cache is unavailable: {exc} Run cut or "
                f"extract-poses for {video_path} first ({cache_path}).",
            ) from exc
        except Exception as exc:
            raise _fail(
                "pose", f"could not read pose cache {cache_path}: {exc}."
            ) from exc
        complete = bool(getattr(snapshot, "complete", False))
        if not complete:
            raise _fail(
                "pose",
                f"pose cache is incomplete (no complete footer): {cache_path}. "
                "Re-run extraction to a complete cache rather than "
                "analyzing partial rows.",
            )
        try:
            expected_identity = CacheIdentity(
                source_fingerprint=metadata.fingerprint,
                model_name=model_name,
                model_version=model_version,
                sampling_rate_hz=rate,
                sampling_start_seconds=start_offset,
            )
        except Exception as exc:
            raise _fail("pose", f"invalid cache identity contract: {exc}.") from exc
        try:
            stored_identity = getattr(snapshot, "identity")
            if load_cache_fn is not None and not isinstance(
                stored_identity, CacheIdentity
            ):
                # Injected fakes may return bare frame tuples; accept a
                # SimpleNamespace-like snapshot carrying frames.
                observations = tuple(getattr(snapshot, "frames"))
                for entry in observations:
                    if not isinstance(entry, FrameObservation):
                        raise _fail(
                            "pose",
                            "pose cache returned "
                            f"{type(entry).__name__}; expected "
                            "FrameObservation rows.",
                        )
                stored_identity = expected_identity
            else:
                cache_module.require_matching_identity(
                    stored_identity, expected_identity
                )
                observations = tuple(snapshot.frames)
        except AnalyzeError:
            raise
        except cache_module.CacheStaleError as exc:
            raise _fail(
                "pose",
                f"pose cache is stale: {exc} Re-extract rather than "
                "reusing cached rows.",
            ) from exc
        except cache_module.CacheError as exc:
            raise _fail("pose", f"pose cache identity is invalid: {exc}.") from exc
        except Exception as exc:
            raise _fail(
                "pose", f"could not validate pose cache {cache_path}: {exc}."
            ) from exc
        stage_timings["pose"] = time.monotonic() - pose_start

        # --- raw-source audio, demuxed once on the dense schedule ---
        _progress("analyze: demuxing source audio")
        if _cancelled():
            raise AnalyzeCancelled(
                "audio", "analyze was cancelled before audio sampling."
            )
        audio_start = time.monotonic()
        try:
            schedule = audio_module.dense_audio_schedule(
                metadata.duration_seconds, audio_step
            )
        except Exception as exc:
            raise _fail("audio", f"could not build audio schedule: {exc}.") from exc
        try:
            if audio_fn is not None:
                dense_audio = tuple(
                    audio_fn(
                        video_path,
                        schedule,
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        is_cancelled=is_cancelled,
                    )
                )
            else:
                dense_audio = tuple(
                    audio_module.iter_audio_energy(
                        video_path,
                        schedule,
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        is_cancelled=is_cancelled,
                    )
                )
        except AnalyzeCancelled:
            raise
        except AnalyzeError:
            raise
        except audio_module.AudioCancelled as exc:
            raise AnalyzeCancelled(
                "audio", f"audio sampling was cancelled: {exc}."
            ) from exc
        except audio_module.AudioError as exc:
            raise _fail("audio", f"audio sampling failed: {exc}.") from exc
        except Exception as exc:
            raise _fail("audio", f"audio sampling failed: {exc}.") from exc
        for entry in dense_audio:
            if not isinstance(entry, AudioEnergy):
                raise _fail(
                    "audio",
                    "audio sampling returned "
                    f"{type(entry).__name__}; expected AudioEnergy rows.",
                )
        stage_timings["audio"] = time.monotonic() - audio_start

        # --- max-pool alignment onto cached pose times + candidates ---
        _progress("analyze: aligning audio")
        if _cancelled():
            raise AnalyzeCancelled(
                "audio", "analyze was cancelled before audio alignment."
            )
        align_start = time.monotonic()
        frame_times = [frame.time_seconds for frame in observations]
        try:
            if not frame_times or not dense_audio:
                aligned_audio: tuple[AudioEnergy, ...] = ()
            elif align_fn is not None:
                aligned_audio = tuple(
                    align_fn(tuple(dense_audio), tuple(frame_times))
                )
            else:
                aligned_audio = tuple(
                    audio_module.align_audio_maxpool(
                        tuple(dense_audio), tuple(frame_times)
                    )
                )
        except AnalyzeCancelled:
            raise
        except AnalyzeError:
            raise
        except Exception as exc:
            raise _fail("audio", f"audio alignment failed: {exc}.") from exc
        for entry in aligned_audio:
            if not isinstance(entry, AudioEnergy):
                raise _fail(
                    "audio",
                    "audio alignment returned "
                    f"{type(entry).__name__}; expected AudioEnergy rows.",
                )
        try:
            if candidates_fn is not None:
                candidate_times = tuple(
                    float(value)
                    for value in candidates_fn(
                        tuple(aligned_audio), cfg_decoder
                    )
                )
            else:
                candidate_times = derive_contact_candidate_times(
                    tuple(aligned_audio), cfg_decoder
                )
        except AnalyzeCancelled:
            raise
        except AnalyzeError:
            raise
        except Exception as exc:
            raise _fail(
                "audio", f"contact candidate derivation failed: {exc}."
            ) from exc
        stage_timings["audio_align"] = time.monotonic() - align_start

        # --- M4.2 / M4.3 / M4.4 per accepted attempt (un-padded only) ---
        _progress(f"analyze: solving phases for {len(document)} attempt(s)")
        phase_start = time.monotonic()
        source_rate = _source_frame_rate_hz(metadata)
        attempt_phases: list[AttemptPhase] = []
        failures: list[AttemptFailure] = []
        for attempt in document.attempts:
            if _cancelled():
                raise AnalyzeCancelled(
                    "phase",
                    "analyze was cancelled during phase solving.",
                )
            unpadded = attempt.detected_range
            try:
                if grid_fn is not None:
                    grid = grid_fn(
                        tuple(observations),
                        unpadded,
                        audio_energies=tuple(aligned_audio),
                        audio_candidate_times=tuple(candidate_times),
                        config=cfg_features,
                        source_frame_rate_hz=source_rate,
                    )
                else:
                    grid = features_module.build_phase_feature_grid(
                        tuple(observations),
                        unpadded,
                        audio_energies=tuple(aligned_audio),
                        audio_candidate_times=tuple(candidate_times),
                        config=cfg_features,
                        source_frame_rate_hz=source_rate,
                    )
                if evidence_fn is not None:
                    evidence = evidence_fn(grid, cfg_evidence)
                else:
                    evidence = evidence_module.build_phase_evidence(
                        grid, cfg_evidence
                    )
                if solver_fn is not None:
                    attempt_phase = solver_fn(
                        attempt, grid, evidence, cfg_solver
                    )
                else:
                    attempt_phase = solver_module.solve_attempt_phase(
                        attempt, grid, evidence, cfg_solver
                    )
                if not isinstance(attempt_phase, AttemptPhase):
                    raise AnalyzeError(
                        "phase",
                        "solver returned "
                        f"{type(attempt_phase).__name__}; expected "
                        "AttemptPhase.",
                    )
                attempt_phases.append(attempt_phase)
            except AnalyzeCancelled:
                raise
            except AnalyzeError as exc:
                failures.append(
                    AttemptFailure(
                        attempt_id=attempt.attempt_id,
                        stage=exc.stage,
                        message=exc.message,
                    )
                )
                attempt_phases.append(
                    _unavailable_attempt_phase(attempt, cfg_solver)
                )
            except Exception as exc:
                failures.append(
                    AttemptFailure(
                        attempt_id=attempt.attempt_id,
                        stage="phase",
                        message=(
                            f"phase solving failed for {attempt.attempt_id}: "
                            f"{exc}."
                        ),
                    )
                )
                attempt_phases.append(
                    _unavailable_attempt_phase(attempt, cfg_solver)
                )
        stage_timings["phase"] = time.monotonic() - phase_start

        try:
            phase_document = PhaseDocument(
                source_fingerprint=metadata.fingerprint,
                source_duration_seconds=metadata.duration_seconds,
                attempts=tuple(attempt_phases),
            )
        except Exception as exc:
            raise _fail(
                "phase", f"could not build phase document: {exc}."
            ) from exc

        # --- atomic checkpoints.json (no partial final file) ---
        _progress("analyze: writing checkpoints")
        if _cancelled():
            raise AnalyzeCancelled(
                "checkpoints",
                "analyze was cancelled before writing checkpoints.",
            )
        write_start = time.monotonic()
        if checkpoints_out.exists() and not overwrite_flag:
            raise _fail(
                "checkpoints",
                f"output collision: {checkpoints_out} already exists. "
                f"Pass --overwrite to replace outputs in {session_label}.",
            )
        try:
            _write_text_atomic(checkpoints_out, phase_document.to_json())
        except AnalyzeError as exc:
            raise _fail("checkpoints", str(exc.message)) from exc
        except Exception as exc:
            raise _fail(
                "checkpoints",
                f"could not write checkpoints.json: {exc}.",
            ) from exc
        stage_timings["write"] = time.monotonic() - write_start
        stage_timings["total"] = time.monotonic() - started

        _progress("analyze: done")
        return AnalyzeResult(
            video=video_path,
            session_dir=session_dir,
            source_metadata=metadata,
            attempts_document=document,
            phase_document=phase_document,
            attempts_path=resolved_attempts,
            checkpoints_path=checkpoints_out,
            cache_path=cache_path,
            cache_hit=True,
            frame_count=len(observations),
            audio_sample_count=len(dense_audio),
            candidate_count=len(candidate_times),
            attempt_failures=tuple(failures),
            stage_timings_seconds=dict(stage_timings),
            empty=len(document) == 0,
        )
    except AnalyzeCancelled as exc:
        _remove_tmp_siblings(session_dir, CHECKPOINTS_FILENAME)
        raise
    except AnalyzeError:
        _remove_tmp_siblings(session_dir, CHECKPOINTS_FILENAME)
        raise


#: Backwards-compatible alias for the phase method identity recorded on
#: every emitted attempt phase.
ANALYZE_METHOD_VERSION = PHASE_SOLVER_METHOD_VERSION
#: Default solver configuration identity used when none is supplied.
ANALYZE_DEFAULT_CONFIG_ID = PHASE_SOLVER_DEFAULT_CONFIG_ID
