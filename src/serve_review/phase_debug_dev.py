"""Private dev-only phase-debug bridge (M4 remediation diagnostic).

Runs the exact current sparse M4 analysis path for one frozen
development attempt and renders the accepted phase-debug inspection
report, without modifying M3 outputs, checkpoints, caches, clips,
sources, annotations, or model artifacts.

Behavior:

- Takes explicit source video, attempts JSON, complete compatible pose
  cache, manual phase-annotation manifest, one attempt id, and one
  explicit output directory (a new private/ignored directory).
- Validates source/attempts/cache linkage against the probed source
  (fingerprint and duration) and reuses only a complete pose cache
  whose identity matches the source/model/sampling contract.
- Rejects non-``dev`` manifests, unknown attempt ids, manifest range
  linkage mismatches, and ``ambiguous`` manual rows rather than
  guessing.
- Extracts only the requested attempt's manual keyframe mapping
  through the existing annotation codecs (available rows contribute
  their manual keyframe, unavailable/missing rows read as null).
- Reuses the exact current sparse M4 path with default configs:
  raw-source audio demux on the dense schedule, max-pool alignment
  onto cached pose times, decoder-policy contact candidates,
  :func:`build_phase_feature_grid`,
  :func:`build_phase_evidence`, and :func:`solve_with_diagnostics`.
- Serializes grid/evidence/solver-result/solver-config as
  deterministic diagnostic provenance inputs inside the output
  directory, then invokes the existing phase-debug reporting to write
  the deterministic debug JSON/HTML.
- Collision-safe and atomic: existing outputs fail without
  ``overwrite``; writes use temporary siblings plus ``os.replace``;
  failure removes temporary siblings and writes no partial final
  files.
- No dense re-inference, no solver/evidence/config tuning, no
  held-out support, no media/frame rendering, no M3 mutation.

All subprocesses use argument arrays via the underlying media/pose
adapters. Canonical source time comes from probe metadata, never from
a frame index or assumed FPS.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from serve_review.analysis_pipeline import derive_contact_candidate_times
from serve_review.checkpoint_evaluation import (
    SPLIT_DEV,
    STATUS_AMBIGUOUS,
    STATUS_AVAILABLE,
    CheckpointEvaluationError,
    PhaseAnnotationManifest,
)
from serve_review.checkpoints import evidence as evidence_module
from serve_review.checkpoints import phase_features as features_module
from serve_review.checkpoints import phase_solver as solver_module
from serve_review.checkpoints.evidence import PhaseEvidenceConfig
from serve_review.checkpoints.phase_features import PhaseFeaturesConfig
from serve_review.checkpoints.phase_solver import PhaseSolverConfig
from serve_review.detection.decoder import DecoderConfig
from serve_review.domain import STAGE_ORDER, Attempt, AttemptDocument, SourceMetadata
from serve_review.media import audio as audio_module
from serve_review.media import probe as probe_module
from serve_review.media.audio import AudioEnergy
from serve_review.phase_debug_report import run_phase_debug_report
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose.mediapipe import MODEL_NAME as _POSE_MODEL_NAME
from serve_review.pose.mediapipe import MODEL_VERSION as _POSE_MODEL_VERSION
from serve_review.pose.schema import CacheIdentity, FrameObservation

__all__ = [
    "PHASE_DEBUG_DEV_VERSION",
    "GRID_FILENAME",
    "EVIDENCE_FILENAME",
    "SOLVER_RESULT_FILENAME",
    "SOLVER_CONFIG_FILENAME",
    "DEBUG_JSON_FILENAME",
    "DEBUG_HTML_FILENAME",
    "PhaseDebugDevError",
    "PhaseDebugDevInputError",
    "PhaseDebugDevCollisionError",
    "PhaseDebugDevResult",
    "manual_keyframes_for_attempt",
    "run_phase_debug_dev",
]

#: Bridge identity (orchestration only; artifacts record their own methods).
PHASE_DEBUG_DEV_VERSION = "phase-debug-dev-v1"

#: Provenance + report filenames written inside the explicit output directory.
GRID_FILENAME = "grid.json"
EVIDENCE_FILENAME = "evidence.json"
SOLVER_RESULT_FILENAME = "solver-result.json"
SOLVER_CONFIG_FILENAME = "solver-config.json"
DEBUG_JSON_FILENAME = "phase-debug.json"
DEBUG_HTML_FILENAME = "phase-debug.html"

#: Tolerance for attempts-duration versus probed-duration agreement.
_DURATION_TOLERANCE_SECONDS = 1e-6

_CANONICAL_STAGES: tuple[str, ...] = STAGE_ORDER


class PhaseDebugDevError(Exception):
    """Actionable phase-debug-dev failure at one stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}: {self.message}"


class PhaseDebugDevInputError(PhaseDebugDevError):
    """Invalid/missing dev input (CLI code 2)."""


class PhaseDebugDevCollisionError(PhaseDebugDevError):
    """Output collision without --overwrite (CLI code 1)."""


@dataclass(frozen=True, slots=True)
class PhaseDebugDevResult:
    """Outcome of :func:`run_phase_debug_dev`."""

    video: Path
    attempt_id: str
    output_dir: Path
    output_json: Path
    output_html: Path
    grid_path: Path
    evidence_path: Path
    solver_result_path: Path
    solver_config_path: Path
    total_objective: float | None


def _fail_input(stage: str, message: str) -> PhaseDebugDevInputError:
    return PhaseDebugDevInputError(stage, message)


def _check_overwrite(value: object) -> bool:
    if not isinstance(value, bool):
        raise _fail_input(
            "validate", f"invalid overwrite: {value!r}; expected True or False."
        )
    return value


def _source_frame_rate_hz(metadata: SourceMetadata) -> float:
    try:
        rate = float(metadata.frames_per_second)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return 120.0
    import math as _math

    if not _math.isfinite(rate) or rate <= 0.0:
        return 120.0
    return rate


def _check_attempt_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise _fail_input(
            "attempt", f"invalid attempt id: {value!r}; expected 'serve-NNN'."
        )
    prefix = "serve-"
    body = value[len(prefix):] if value.startswith(prefix) else None
    if body is None or len(body) < 3 or not body.isdigit():
        raise _fail_input(
            "attempt",
            f"invalid attempt id: {value!r}; expected 'serve-NNN' "
            "with at least three digits.",
        )
    return value


def manual_keyframes_for_attempt(
    manifest: PhaseAnnotationManifest,
    attempt_id: str,
    attempt: Attempt,
) -> dict[str, float | None]:
    """Extract the manual keyframe mapping for one attempt.

    Pure; no I/O. Validates through the existing annotation codecs'
    types: the manifest must carry the ``dev`` split, must contain rows
    for ``attempt_id``, every row's attempt range linkage must exactly
    equal the attempts-document unpadded ``detected_range``, and no row
    may be ``ambiguous``. Available rows contribute their
    ``manual_keyframe_seconds`` (possibly null); unavailable or missing
    stages read as null.
    """
    if not isinstance(manifest, PhaseAnnotationManifest):
        raise _fail_input(
            "annotations",
            f"invalid manifest: expected PhaseAnnotationManifest, "
            f"got {type(manifest).__name__}.",
        )
    if manifest.dataset_split != SPLIT_DEV:
        raise _fail_input(
            "annotations",
            f"unsupported dataset_split {manifest.dataset_split!r}; "
            "phase-debug-dev supports only dev manifests (held-out "
            "annotations are never rendered here).",
        )
    if not isinstance(attempt, Attempt):
        raise _fail_input(
            "attempt",
            f"invalid attempt: expected Attempt, got {type(attempt).__name__}.",
        )
    rows = [entry for entry in manifest.annotations if entry.attempt_id == attempt_id]
    if not rows:
        raise _fail_input(
            "annotations",
            f"unknown attempt_id {attempt_id!r} for this manifest; "
            "no annotation rows match (refusing to guess across attempts).",
        )
    expected_start = float(attempt.detected_range.start_seconds)
    expected_end = float(attempt.detected_range.end_seconds)
    for entry in rows:
        if float(entry.attempt_start_seconds) != expected_start or float(
            entry.attempt_end_seconds
        ) != expected_end:
            raise _fail_input(
                "annotations",
                f"manifest attempt range "
                f"[{entry.attempt_start_seconds!r}, "
                f"{entry.attempt_end_seconds!r}) does not match attempts "
                f"range [{expected_start!r}, {expected_end!r}) for attempt "
                f"{attempt_id!r}; refusing to pair across ranges.",
            )
        if entry.status == STATUS_AMBIGUOUS:
            raise _fail_input(
                "annotations",
                f"ambiguous manual row for attempt {attempt_id!r} stage "
                f"{entry.stage!r}; refusing to guess an exact keyframe.",
            )
    by_stage = {entry.stage: entry for entry in rows}
    mapping: dict[str, float | None] = {}
    for stage in _CANONICAL_STAGES:
        entry = by_stage.get(stage)
        if entry is None or entry.status != STATUS_AVAILABLE:
            mapping[stage] = None
            continue
        keyframe = entry.manual_keyframe_seconds
        mapping[stage] = None if keyframe is None else float(keyframe)
    return mapping


def _write_text_atomic(target: Path, text: str) -> None:
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PhaseDebugDevError(
                "report", f"could not create output directory {parent}: {exc}."
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
    except PhaseDebugDevError:
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
        raise PhaseDebugDevError(
            "report", f"could not write output file {target}: {exc}."
        ) from exc


def _remove_tmp_siblings(directory: Path, stems: tuple[str, ...]) -> None:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        for stem in stems:
            if entry.name.startswith(stem + ".tmp-"):
                try:
                    entry.unlink(missing_ok=True)
                except OSError:
                    pass
                break


def run_phase_debug_dev(
    video: Path | str,
    *,
    attempts_path: Path | str,
    cache_path: Path | str,
    annotations_path: Path | str,
    attempt_id: str,
    output_dir: Path | str,
    overwrite: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    model_name: str = _POSE_MODEL_NAME,
    model_version: str = _POSE_MODEL_VERSION,
    sample_rate_hz: float = extract_module.DEFAULT_SAMPLE_RATE_HZ,
    sampling_start_seconds: float = extract_module.DEFAULT_SAMPLING_START_SECONDS,
    audio_step_seconds: float = audio_module.DENSE_AUDIO_STEP_SECONDS,
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
) -> PhaseDebugDevResult:
    """Run the sparse M4 path for one dev attempt and write the debug report.

    Never modifies the source video, attempts JSON, checkpoints, pose
    cache, annotations manifest, clips, compilation, or model
    artifacts; all outputs are written atomically inside ``output_dir``
    only. No dense re-inference and no solver/evidence/config tuning:
    default configs are used unless injectable stage functions are
    supplied by tests.
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
            raise _fail_input("validate", f"invalid is_cancelled: {is_cancelled!r}.")
        return bool(is_cancelled())

    video_path = Path(video).expanduser()
    if not str(video_path):
        raise _fail_input("validate", "invalid video: expected a non-blank path.")
    if not video_path.is_file():
        raise _fail_input("validate", f"input video does not exist: {video_path}.")
    attempts_file = Path(attempts_path).expanduser()
    cache_file = Path(cache_path).expanduser()
    annotations_file = Path(annotations_path).expanduser()
    out_dir = Path(output_dir).expanduser()
    if not str(attempts_file):
        raise _fail_input("validate", "invalid attempts path: expected a non-blank path.")
    if not str(cache_file):
        raise _fail_input("validate", "invalid cache path: expected a non-blank path.")
    if not str(annotations_file):
        raise _fail_input(
            "validate", "invalid annotations path: expected a non-blank path."
        )
    if not str(out_dir):
        raise _fail_input("validate", "invalid output directory: expected a non-blank path.")
    overwrite_flag = _check_overwrite(overwrite)
    wanted_id = _check_attempt_id(attempt_id)
    if not isinstance(model_name, str) or not model_name.strip():
        raise _fail_input(
            "validate",
            f"invalid model_name: {model_name!r}; expected a non-blank string.",
        )
    if not isinstance(model_version, str) or not model_version.strip():
        raise _fail_input(
            "validate",
            f"invalid model_version: {model_version!r}; expected a non-blank string.",
        )
    import math as _math

    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, float))
        or not _math.isfinite(float(sample_rate_hz))
        or float(sample_rate_hz) <= 0.0
    ):
        raise _fail_input(
            "validate",
            f"invalid sample_rate_hz: {sample_rate_hz!r}; "
            "expected a finite number > 0.",
        )
    if (
        isinstance(sampling_start_seconds, bool)
        or not isinstance(sampling_start_seconds, (int, float))
        or not _math.isfinite(float(sampling_start_seconds))
        or float(sampling_start_seconds) < 0.0
    ):
        raise _fail_input(
            "validate",
            f"invalid sampling_start_seconds: {sampling_start_seconds!r}; "
            "expected a finite number >= 0.",
        )
    if (
        isinstance(audio_step_seconds, bool)
        or not isinstance(audio_step_seconds, (int, float))
        or not _math.isfinite(float(audio_step_seconds))
        or float(audio_step_seconds) <= 0.0
    ):
        raise _fail_input(
            "validate",
            f"invalid audio_step_seconds: {audio_step_seconds!r}; "
            "expected a finite number > 0.",
        )
    rate = float(sample_rate_hz)
    start_offset = float(sampling_start_seconds)
    audio_step = float(audio_step_seconds)

    grid_out = out_dir / GRID_FILENAME
    evidence_out = out_dir / EVIDENCE_FILENAME
    result_out = out_dir / SOLVER_RESULT_FILENAME
    config_out = out_dir / SOLVER_CONFIG_FILENAME
    debug_json = out_dir / DEBUG_JSON_FILENAME
    debug_html = out_dir / DEBUG_HTML_FILENAME
    final_files = (
        grid_out,
        evidence_out,
        result_out,
        config_out,
        debug_json,
        debug_html,
    )
    stems = (
        GRID_FILENAME,
        EVIDENCE_FILENAME,
        SOLVER_RESULT_FILENAME,
        SOLVER_CONFIG_FILENAME,
        DEBUG_JSON_FILENAME,
        DEBUG_HTML_FILENAME,
    )

    def _cleanup_tmps() -> None:
        _remove_tmp_siblings(out_dir, stems)

    # Guard: never treat an input file as an output file.
    try:
        resolved_outputs = {path.resolve() for path in final_files}
        for label, candidate in (
            ("attempts", attempts_file),
            ("cache", cache_file),
            ("annotations", annotations_file),
            ("video", video_path),
        ):
            try:
                if candidate.resolve() in resolved_outputs:
                    raise _fail_input(
                        "validate",
                        f"output directory collides with {label} input "
                        f"{candidate}; choose a distinct private directory.",
                    )
            except OSError:
                pass
    except PhaseDebugDevInputError:
        raise

    try:
        if _cancelled():
            raise PhaseDebugDevError("validate", "phase-debug-dev was cancelled.")
        # --- probe ---
        _progress("phase-debug-dev: probing source")
        resolve_probe = (
            probe_fn
            if probe_fn is not None
            else (lambda path: probe_module.probe_source(path, ffprobe=ffprobe))
        )
        try:
            metadata = resolve_probe(video_path)
        except PhaseDebugDevError:
            raise
        except Exception as exc:
            raise _fail_input("probe", f"probe failed for {video_path}: {exc}.") from exc
        if not isinstance(metadata, SourceMetadata):
            raise _fail_input(
                "probe",
                f"probe returned {type(metadata).__name__}; expected SourceMetadata.",
            )

        # --- attempts + exact attempt linkage ---
        _progress("phase-debug-dev: loading attempts")
        if _cancelled():
            raise PhaseDebugDevError("attempts", "phase-debug-dev was cancelled.")
        if not attempts_file.is_file():
            raise _fail_input(
                "attempts", f"attempts file does not exist: {attempts_file}."
            )
        try:
            raw_attempts = attempts_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise _fail_input(
                "attempts", f"could not read attempts file {attempts_file}: {exc}."
            ) from exc
        try:
            attempts_payload = json.loads(raw_attempts)
        except json.JSONDecodeError as exc:
            raise _fail_input(
                "attempts", f"invalid attempts JSON in {attempts_file}: {exc}."
            ) from exc
        try:
            document = AttemptDocument.from_dict(attempts_payload)
        except Exception as exc:
            raise _fail_input(
                "attempts", f"invalid attempt document in {attempts_file}: {exc}."
            ) from exc
        if document.source_fingerprint != metadata.fingerprint:
            raise _fail_input(
                "attempts",
                "attempts source fingerprint does not match the source video "
                f"(attempts={document.source_fingerprint!r} != "
                f"source={metadata.fingerprint!r}).",
            )
        if (
            abs(document.source_duration_seconds - metadata.duration_seconds)
            > _DURATION_TOLERANCE_SECONDS
        ):
            raise _fail_input(
                "attempts",
                "attempts source duration does not match the source video "
                f"(attempts={document.source_duration_seconds!r} != "
                f"source={metadata.duration_seconds!r}).",
            )
        matches = [
            entry for entry in document.attempts if entry.attempt_id == wanted_id
        ]
        if len(matches) != 1:
            if not matches:
                raise _fail_input(
                    "attempt",
                    f"unknown attempt_id {wanted_id!r} for this attempts file; "
                    "refusing to guess across attempts.",
                )
            raise _fail_input(
                "attempt",
                f"ambiguous attempt_id {wanted_id!r}: "
                f"{len(matches)} matches; refusing to guess.",
            )
        attempt = matches[0]
        unpadded = attempt.detected_range

        # --- pose cache reuse only (never re-infer) ---
        _progress("phase-debug-dev: loading cached poses")
        if _cancelled():
            raise PhaseDebugDevError("pose", "phase-debug-dev was cancelled.")
        try:
            if load_cache_fn is not None:
                snapshot = load_cache_fn(cache_file)
            else:
                snapshot = cache_module.load_cache(cache_file)
        except PhaseDebugDevError:
            raise
        except cache_module.CacheCorruptError as exc:
            raise _fail_input(
                "pose", f"pose cache is corrupt: {exc} ({cache_file})."
            ) from exc
        except cache_module.CacheError as exc:
            raise _fail_input(
                "pose", f"pose cache is unavailable: {exc} ({cache_file})."
            ) from exc
        except Exception as exc:
            raise _fail_input(
                "pose", f"could not read pose cache {cache_file}: {exc}."
            ) from exc
        if not bool(getattr(snapshot, "complete", False)):
            raise _fail_input(
                "pose",
                f"pose cache is incomplete (no complete footer): {cache_file}. "
                "Re-run extraction to a complete cache.",
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
            raise _fail_input("pose", f"invalid cache identity contract: {exc}.") from exc
        try:
            stored_identity = getattr(snapshot, "identity")
            if load_cache_fn is not None and not isinstance(
                stored_identity, CacheIdentity
            ):
                observations = tuple(getattr(snapshot, "frames"))
                for entry in observations:
                    if not isinstance(entry, FrameObservation):
                        raise _fail_input(
                            "pose",
                            "pose cache returned "
                            f"{type(entry).__name__}; expected FrameObservation rows.",
                        )
                stored_identity = expected_identity
            else:
                cache_module.require_matching_identity(
                    stored_identity, expected_identity
                )
                observations = tuple(snapshot.frames)
        except PhaseDebugDevError:
            raise
        except cache_module.CacheStaleError as exc:
            raise _fail_input(
                "pose", f"pose cache is stale: {exc} Re-extract rather than reusing."
            ) from exc
        except cache_module.CacheError as exc:
            raise _fail_input(
                "pose", f"pose cache identity is invalid: {exc}."
            ) from exc
        except Exception as exc:
            raise _fail_input(
                "pose", f"could not validate pose cache {cache_file}: {exc}."
            ) from exc

        # --- manual manifest (dev only, exact linkage) ---
        _progress("phase-debug-dev: loading manual annotations")
        if _cancelled():
            raise PhaseDebugDevError("annotations", "phase-debug-dev was cancelled.")
        if not annotations_file.is_file():
            raise _fail_input(
                "annotations",
                f"annotations file does not exist: {annotations_file}.",
            )
        try:
            raw_manifest = annotations_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise _fail_input(
                "annotations",
                f"could not read annotations file {annotations_file}: {exc}.",
            ) from exc
        try:
            manifest = PhaseAnnotationManifest.from_json(raw_manifest)
        except CheckpointEvaluationError as exc:
            raise _fail_input("annotations", f"invalid annotation manifest: {exc}.") from exc
        except Exception as exc:
            raise _fail_input("annotations", f"invalid annotation manifest: {exc}.") from exc
        manual = manual_keyframes_for_attempt(manifest, wanted_id, attempt)

        # --- raw-source audio demuxed once on the dense schedule ---
        _progress("phase-debug-dev: demuxing source audio")
        if _cancelled():
            raise PhaseDebugDevError("audio", "phase-debug-dev was cancelled.")
        try:
            schedule = audio_module.dense_audio_schedule(
                metadata.duration_seconds, audio_step
            )
        except Exception as exc:
            raise _fail_input("audio", f"could not build audio schedule: {exc}.") from exc
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
        except PhaseDebugDevError:
            raise
        except audio_module.AudioCancelled as exc:
            raise PhaseDebugDevError(
                "audio", f"audio sampling was cancelled: {exc}."
            ) from exc
        except audio_module.AudioError as exc:
            raise _fail_input("audio", f"audio sampling failed: {exc}.") from exc
        except Exception as exc:
            raise _fail_input("audio", f"audio sampling failed: {exc}.") from exc
        for entry in dense_audio:
            if not isinstance(entry, AudioEnergy):
                raise _fail_input(
                    "audio",
                    "audio sampling returned "
                    f"{type(entry).__name__}; expected AudioEnergy rows.",
                )

        # --- max-pool alignment + decoder-policy candidates ---
        _progress("phase-debug-dev: aligning audio")
        if _cancelled():
            raise PhaseDebugDevError("audio", "phase-debug-dev was cancelled.")
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
        except PhaseDebugDevError:
            raise
        except Exception as exc:
            raise _fail_input("audio", f"audio alignment failed: {exc}.") from exc
        for entry in aligned_audio:
            if not isinstance(entry, AudioEnergy):
                raise _fail_input(
                    "audio",
                    "audio alignment returned "
                    f"{type(entry).__name__}; expected AudioEnergy rows.",
                )
        try:
            decoder_config = DecoderConfig()
            if candidates_fn is not None:
                candidate_times = tuple(
                    float(value)
                    for value in candidates_fn(tuple(aligned_audio), decoder_config)
                )
            else:
                candidate_times = derive_contact_candidate_times(
                    tuple(aligned_audio), decoder_config
                )
        except PhaseDebugDevError:
            raise
        except Exception as exc:
            raise _fail_input(
                "audio", f"contact candidate derivation failed: {exc}."
            ) from exc

        # --- exact sparse M4 path with default configs ---
        _progress("phase-debug-dev: solving phases")
        if _cancelled():
            raise PhaseDebugDevError("phase", "phase-debug-dev was cancelled.")
        source_rate = _source_frame_rate_hz(metadata)
        cfg_features = PhaseFeaturesConfig()
        cfg_evidence = PhaseEvidenceConfig()
        cfg_solver = PhaseSolverConfig()
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
        except PhaseDebugDevError:
            raise
        except Exception as exc:
            raise PhaseDebugDevError(
                "phase", f"phase feature grid failed for {wanted_id}: {exc}."
            ) from exc
        try:
            if evidence_fn is not None:
                evidence = evidence_fn(grid, cfg_evidence)
            else:
                evidence = evidence_module.build_phase_evidence(grid, cfg_evidence)
        except PhaseDebugDevError:
            raise
        except Exception as exc:
            raise PhaseDebugDevError(
                "phase", f"phase evidence failed for {wanted_id}: {exc}."
            ) from exc
        try:
            if solver_fn is not None:
                solver_result = solver_fn(attempt, grid, evidence, cfg_solver)
                from serve_review.checkpoints.phase_solver import PhaseSolverResult as _R

                if not isinstance(solver_result, _R):
                    raise PhaseDebugDevError(
                        "phase",
                        f"solver returned {type(solver_result).__name__}; "
                        "expected PhaseSolverResult.",
                    )
            else:
                solver_result = solver_module.solve_with_diagnostics(
                    attempt, grid, evidence, cfg_solver
                )
        except PhaseDebugDevError:
            raise
        except Exception as exc:
            raise PhaseDebugDevError(
                "phase", f"phase solver failed for {wanted_id}: {exc}."
            ) from exc

        # --- deterministic provenance inputs + existing report ---
        _progress("phase-debug-dev: writing report")
        if _cancelled():
            raise PhaseDebugDevError("report", "phase-debug-dev was cancelled.")
        grid_text = json.dumps(grid.to_dict(), sort_keys=True, indent=2) + "\n"
        evidence_text = evidence.to_json()
        result_text = solver_result.to_json()
        config_text = cfg_solver.to_json()

        if not overwrite_flag:
            for path in final_files:
                if path.exists():
                    _cleanup_tmps()
                    raise PhaseDebugDevCollisionError(
                        "report",
                        f"output collision: {path} already exists. "
                        "Pass --overwrite to replace outputs.",
                    )

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _cleanup_tmps()
            raise PhaseDebugDevError(
                "report", f"could not create output directory {out_dir}: {exc}."
            ) from exc

        # Write provenance inputs atomically first so the report consumes
        # exactly the serialized inputs preserved beside it.
        try:
            _write_text_atomic(grid_out, grid_text)
            _write_text_atomic(evidence_out, evidence_text)
            _write_text_atomic(result_out, result_text)
            _write_text_atomic(config_out, config_text)
        except PhaseDebugDevError as exc:
            _cleanup_tmps()
            raise

        from serve_review.phase_debug_report import (
            PhaseDebugReportCollisionError,
            PhaseDebugReportError,
            PhaseDebugReportInputError,
        )

        try:
            report = run_phase_debug_report(
                grid_out,
                evidence_out,
                result_out,
                config_out,
                debug_json,
                debug_html,
                manual_times=manual,
                overwrite=True,
            )
        except PhaseDebugReportCollisionError as exc:
            _cleanup_tmps()
            raise PhaseDebugDevCollisionError(exc.stage, exc.message) from exc
        except PhaseDebugReportInputError as exc:
            raise _fail_input(exc.stage, exc.message) from exc
        except PhaseDebugReportError as exc:
            raise PhaseDebugDevError(exc.stage, exc.message) from exc

        _progress("phase-debug-dev: done")
        return PhaseDebugDevResult(
            video=video_path,
            attempt_id=wanted_id,
            output_dir=out_dir,
            output_json=report.output_json,
            output_html=report.output_html,
            grid_path=grid_out,
            evidence_path=evidence_out,
            solver_result_path=result_out,
            solver_config_path=config_out,
            total_objective=report.total_objective,
        )
    except (PhaseDebugDevCollisionError, PhaseDebugDevInputError, PhaseDebugDevError):
        _remove_tmp_siblings(out_dir, stems)
        raise
