"""Stage evidence and candidate generation (M4.3).

Pure, deterministic, in-memory generation of bounded candidate
frames/intervals and unary evidence scores ``S_i(t)`` for the eight
Kovacs stage keys from one M4.2 :class:`PhaseFeatureGrid`.

This module is not a solver: it never enforces global chronology,
never chooses one final phase sequence, and never emits
``AttemptPhase`` / ``StagePhase`` / ``PhaseDocument``. Each stage is
scored independently inside its own broad, configurable search window.
Candidate sets are bounded, ordered deterministically by
``(-score, keyframe_seconds)``, and may be empty. Missing evidence is
never fabricated to complete a serve.

Rubric honesty:

- ``release`` is a body-pose estimate from an early upward
  wrist-trajectory/velocity candidate, never ball observation; side is
  ambiguous unless evidence can state it.
- ``cocking`` is a body-pose proxy combining wrist/torso geometry and
  elbow-flexion evidence, never racket drop/head orientation.
- ``contact`` uses only existing ``audio_candidate`` rows with
  ``audio_transient`` provenance; audio strength is normalized
  scale-invariantly within the grid/search window (never divided by an
  arbitrary absolute amplitude).
- Camera-relative quantities stay named camera-relative; no image-x to
  anatomical forward/posterior mapping, no handedness/viewpoint
  inference, no racket/ball observation claims.
- Interpolated samples cannot alone yield high-confidence visual
  candidates; low derivative confidence or missing inputs lower the
  score or yield none. Audio-only contact never manufactures body
  candidates.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from serve_review.domain import (
    STAGE_ORDER,
    STAGE_PROVENANCES,
    MediaRange,
)
from serve_review.checkpoints.phase_features import PhaseFeatureGrid

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_METHOD_VERSION",
    "EVIDENCE_DEFAULT_CONFIG_ID",
    "PhaseEvidenceError",
    "PhaseEvidenceConfig",
    "StageCandidate",
    "PhaseEvidence",
    "compute_stage_scores",
    "generate_stage_candidates",
    "build_phase_evidence",
]

#: Version of the evidence configuration/candidate schemas.
EVIDENCE_SCHEMA_VERSION = 1
#: Method identity recorded on every candidate set.
EVIDENCE_METHOD_VERSION = "phase-evidence-v1"
#: Default configuration identity.
EVIDENCE_DEFAULT_CONFIG_ID = "phase-evidence-default-v1"

_CANONICAL_STAGES: tuple[str, ...] = STAGE_ORDER

_CONTACT_STAGE = "contact"
_VISUAL_STAGES: tuple[str, ...] = tuple(k for k in STAGE_ORDER if k != _CONTACT_STAGE)


class PhaseEvidenceError(ValueError):
    """Raised when evidence input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEvidenceError(
            f"phase_evidence_config: {name!r} must be a number in [0, 1], "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseEvidenceError(
            f"phase_evidence_config: {name!r} must lie in [0, 1], got {value!r}."
        )
    return number


def _check_positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEvidenceError(
            f"phase_evidence_config: {name!r} must be a number > 0, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise PhaseEvidenceError(
            f"phase_evidence_config: {name!r} must be finite and > 0, "
            f"got {value!r}."
        )
    return number


def _check_unit(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEvidenceError(
            f"phase_evidence_config: {name!r} must be a number in [0, 1], "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseEvidenceError(
            f"phase_evidence_config: {name!r} must lie in [0, 1], got {value!r}."
        )
    return number


def _check_count(name: str, value: Any) -> int:
    if not _is_int(value) or value < 1 or value > 16:
        raise PhaseEvidenceError(
            f"phase_evidence_config: {name!r} must be an integer in [1, 16], "
            f"got {value!r}."
        )
    return value


def _check_weight(name: str, value: Any) -> float:
    return _check_unit(name, value)


@dataclass(frozen=True, slots=True)
class PhaseEvidenceConfig:
    """Versioned, immutable configuration for stage evidence.

    Search windows are broad fractions of the attempt duration
    ``[start_frac, end_frac]`` per stage, deliberately wide so no
    narrow biomechanical threshold is hard-coded. Caps, floors, and
    cue weights are centralized here.
    """

    config_id: str = EVIDENCE_DEFAULT_CONFIG_ID
    max_candidates_per_stage: int = 3
    candidate_half_window_seconds: float = 0.06
    candidate_floor: float = 0.15
    interpolated_score_cap: float = 0.5
    contact_audio_weight: float = 0.75
    # Broad per-stage search windows as fractions of attempt duration.
    start_search_start_frac: float = 0.0
    start_search_end_frac: float = 0.30
    release_search_start_frac: float = 0.0
    release_search_end_frac: float = 0.35
    loading_search_start_frac: float = 0.15
    loading_search_end_frac: float = 0.60
    cocking_search_start_frac: float = 0.30
    cocking_search_end_frac: float = 0.75
    acceleration_search_start_frac: float = 0.40
    acceleration_search_end_frac: float = 0.90
    contact_search_start_frac: float = 0.20
    contact_search_end_frac: float = 1.0
    deceleration_search_start_frac: float = 0.50
    deceleration_search_end_frac: float = 1.0
    finish_search_start_frac: float = 0.65
    finish_search_end_frac: float = 1.0
    schema_version: int = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_evidence_config"
        if not _is_int(self.schema_version):
            raise PhaseEvidenceError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise PhaseEvidenceError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {EVIDENCE_SCHEMA_VERSION}."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise PhaseEvidenceError(
                f"{name}: 'config_id' must be a non-blank string, "
                f"got {self.config_id!r}."
            )
        object.__setattr__(
            self, "max_candidates_per_stage",
            _check_count("'max_candidates_per_stage'", self.max_candidates_per_stage),
        )
        object.__setattr__(
            self, "candidate_half_window_seconds",
            _check_positive("'candidate_half_window_seconds'", self.candidate_half_window_seconds),
        )
        object.__setattr__(
            self, "candidate_floor",
            _check_unit("'candidate_floor'", self.candidate_floor),
        )
        object.__setattr__(
            self, "interpolated_score_cap",
            _check_unit("'interpolated_score_cap'", self.interpolated_score_cap),
        )
        object.__setattr__(
            self, "contact_audio_weight",
            _check_weight("'contact_audio_weight'", self.contact_audio_weight),
        )
        for stage in _CANONICAL_STAGES:
            lo = getattr(self, f"{stage}_search_start_frac")
            hi = getattr(self, f"{stage}_search_end_frac")
            lo_f = _check_fraction(f"'{stage}_search_start_frac'", lo)
            hi_f = _check_fraction(f"'{stage}_search_end_frac'", hi)
            if not lo_f < hi_f:
                raise PhaseEvidenceError(
                    f"{name}: stage {stage!r} search window must satisfy "
                    f"start_frac < end_frac, got ({lo_f!r}, {hi_f!r})."
                )
            object.__setattr__(self, f"{stage}_search_start_frac", lo_f)
            object.__setattr__(self, f"{stage}_search_end_frac", hi_f)

    def search_window_fractions(self, stage: str) -> tuple[float, float]:
        """Return the configured broad ``(start_frac, end_frac)`` window."""
        if stage not in _CANONICAL_STAGES:
            raise PhaseEvidenceError(
                f"phase_evidence_config: unknown stage key {stage!r}."
            )
        return (
            float(getattr(self, f"{stage}_search_start_frac")),
            float(getattr(self, f"{stage}_search_end_frac")),
        )

    def search_interval(self, attempt_range: MediaRange, stage: str) -> MediaRange:
        """Return the configured broad search bounds in source seconds."""
        if not isinstance(attempt_range, MediaRange):
            raise PhaseEvidenceError(
                "phase_evidence_config: 'attempt_range' must be a MediaRange, "
                f"got {type(attempt_range).__name__}."
            )
        lo_frac, hi_frac = self.search_window_fractions(stage)
        duration = attempt_range.end_seconds - attempt_range.start_seconds
        lo = attempt_range.start_seconds + lo_frac * duration
        hi = attempt_range.start_seconds + hi_frac * duration
        if hi <= lo:
            hi = min(attempt_range.end_seconds, lo + 1e-6)
        hi = min(hi, attempt_range.end_seconds)
        lo = max(lo, attempt_range.start_seconds)
        if not hi > lo:
            # Degenerate attempt; fall back to the full attempt range copy.
            return MediaRange(
                start_seconds=attempt_range.start_seconds,
                end_seconds=attempt_range.end_seconds,
            )
        return MediaRange(start_seconds=lo, end_seconds=hi)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_id": self.config_id,
            "schema_version": self.schema_version,
            "max_candidates_per_stage": self.max_candidates_per_stage,
            "candidate_half_window_seconds": self.candidate_half_window_seconds,
            "candidate_floor": self.candidate_floor,
            "interpolated_score_cap": self.interpolated_score_cap,
            "contact_audio_weight": self.contact_audio_weight,
        }
        for stage in _CANONICAL_STAGES:
            payload[f"{stage}_search_start_frac"] = getattr(
                self, f"{stage}_search_start_frac"
            )
            payload[f"{stage}_search_end_frac"] = getattr(
                self, f"{stage}_search_end_frac"
            )
        return payload

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseEvidenceConfig:
        if not isinstance(values, dict):
            raise PhaseEvidenceError(
                "phase_evidence_config: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "config_id",
            "schema_version",
            "max_candidates_per_stage",
            "candidate_half_window_seconds",
            "candidate_floor",
            "interpolated_score_cap",
            "contact_audio_weight",
        }
        for stage in _CANONICAL_STAGES:
            known.add(f"{stage}_search_start_frac")
            known.add(f"{stage}_search_end_frac")
        missing = sorted(known - set(values))
        if missing:
            raise PhaseEvidenceError(
                f"phase_evidence_config: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseEvidenceError(
                f"phase_evidence_config: unknown keys {unknown!r}."
            )
        kwargs: dict[str, Any] = {key: values[key] for key in known}
        return cls(**kwargs)  # type: ignore[arg-type]


def _check_stage_key(name: str, value: Any) -> str:
    if value not in _CANONICAL_STAGES:
        raise PhaseEvidenceError(
            f"{name}: 'stage' must be one of {list(_CANONICAL_STAGES)}, "
            f"got {value!r}."
        )
    return value  # type: ignore[return-value]


def _check_score(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEvidenceError(
            f"{name}: 'score' must be a number in [0, 1], got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseEvidenceError(
            f"{name}: 'score' must lie in [0, 1], got {value!r}."
        )
    return number


def _check_quality(name: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEvidenceError(
            f"{name}: {key!r} must be a number in [0, 1], got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseEvidenceError(
            f"{name}: {key!r} must lie in [0, 1], got {value!r}."
        )
    return number


def _check_keyframe(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEvidenceError(
            f"{name}: 'keyframe_seconds' must be a finite number >= 0, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseEvidenceError(
            f"{name}: 'keyframe_seconds' must be finite and >= 0, "
            f"got {value!r}."
        )
    return number


def _check_uncertainty(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseEvidenceError(
            f"{name}: 'temporal_uncertainty_seconds' must be finite and >= 0, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseEvidenceError(
            f"{name}: 'temporal_uncertainty_seconds' must be finite and >= 0, "
            f"got {value!r}."
        )
    return number


def _check_tokens(name: str, key: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PhaseEvidenceError(
            f"{name}: {key!r} must be a list or tuple of non-blank strings, "
            f"got {type(value).__name__}."
        )
    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise PhaseEvidenceError(
                f"{name}: {key!r} entries must be non-blank strings, "
                f"got {entry!r}."
            )
        lowered = entry.lower()
        # Never claim direct racket/ball observation or anatomical axes.
        # Honest "no_*" absence limitations are allowed.
        is_absence = lowered.startswith("no_")
        for forbidden in (
            "racket_visual",
            "racket_drop_observed",
            "forward_posterior",
            "anatomical_forward",
            "anatomical_posterior",
            "handedness_",
            "viewpoint_known",
            "racket_orientation",
            "head_orientation",
            "ball_racket",
        ):
            if forbidden in lowered:
                raise PhaseEvidenceError(
                    f"{name}: {key!r} entry {entry!r} makes a forbidden claim."
                )
        if not is_absence and "ball_observation" in lowered:
            raise PhaseEvidenceError(
                f"{name}: {key!r} entry {entry!r} makes a forbidden claim."
            )
        if entry in seen:
            raise PhaseEvidenceError(
                f"{name}: {key!r} must not contain duplicates, got {entry!r}."
            )
        seen.add(entry)
        cleaned.append(entry)
    return tuple(cleaned)


@dataclass(frozen=True, slots=True)
class StageCandidate:
    """One immutable bounded candidate for a single stage.

    ``keyframe_seconds`` is the canonical representative source time and
    always lies in ``interval``. ``interval`` is a half-open
    ``[start, end)`` range inside the owning attempt range.
    ``observation_quality`` / ``derivative_quality`` echo the supporting
    grid sample's support channels. ``evidence`` / ``limitations`` are
    machine-readable identifier collections using camera-relative,
    side-uncertain vocabulary. ``provenance`` is limited to the M4.1
    vocabulary; visual stages use ``body_pose`` and contact uses
    ``audio_transient``.
    """

    stage: str = "start"
    keyframe_seconds: float = 0.0
    interval: MediaRange = None  # type: ignore[assignment]
    score: float = 0.0
    observation_quality: float = 0.0
    derivative_quality: float = 0.0
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    provenance: str = "body_pose"
    temporal_uncertainty_seconds: float = 0.0
    method_version: str = EVIDENCE_METHOD_VERSION
    config_id: str = EVIDENCE_DEFAULT_CONFIG_ID
    schema_version: int = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "stage_candidate"
        if not _is_int(self.schema_version):
            raise PhaseEvidenceError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise PhaseEvidenceError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {EVIDENCE_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "stage", _check_stage_key(name, self.stage))
        object.__setattr__(self, "keyframe_seconds", _check_keyframe(name, self.keyframe_seconds))
        if not isinstance(self.interval, MediaRange):
            raise PhaseEvidenceError(
                f"{name}: 'interval' must be a MediaRange, "
                f"got {type(self.interval).__name__}."
            )
        object.__setattr__(self, "score", _check_score(name, self.score))
        object.__setattr__(
            self, "observation_quality",
            _check_quality(name, "'observation_quality'", self.observation_quality),
        )
        object.__setattr__(
            self, "derivative_quality",
            _check_quality(name, "'derivative_quality'", self.derivative_quality),
        )
        if not self.interval.contains(self.keyframe_seconds):
            # Allow exact start-boundary keyframes explicitly.
            if not self.keyframe_seconds == self.interval.start_seconds:
                raise PhaseEvidenceError(
                    f"{name}: 'keyframe_seconds' ({self.keyframe_seconds!r}) must "
                    f"lie in interval [{self.interval.start_seconds!r}, "
                    f"{self.interval.end_seconds!r})."
                )
        object.__setattr__(
            self, "evidence", _check_tokens(name, "'evidence'", self.evidence)
        )
        object.__setattr__(
            self, "limitations", _check_tokens(name, "'limitations'", self.limitations)
        )
        if self.provenance not in STAGE_PROVENANCES:
            raise PhaseEvidenceError(
                f"{name}: 'provenance' must be one of "
                f"{list(STAGE_PROVENANCES)}, got {self.provenance!r}."
            )
        if self.stage == _CONTACT_STAGE and self.provenance != "audio_transient":
            raise PhaseEvidenceError(
                f"{name}: contact candidates must use 'audio_transient' "
                f"provenance, got {self.provenance!r}."
            )
        if self.stage != _CONTACT_STAGE and self.provenance != "body_pose":
            raise PhaseEvidenceError(
                f"{name}: visual stage {self.stage!r} candidates must use "
                f"'body_pose' provenance, got {self.provenance!r}."
            )
        object.__setattr__(
            self, "temporal_uncertainty_seconds",
            _check_uncertainty(name, self.temporal_uncertainty_seconds),
        )
        if not isinstance(self.method_version, str) or not self.method_version.strip():
            raise PhaseEvidenceError(
                f"{name}: 'method_version' must be a non-blank string."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise PhaseEvidenceError(
                f"{name}: 'config_id' must be a non-blank string."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "derivative_quality": self.derivative_quality,
            "evidence": list(self.evidence),
            "interval": self.interval.to_dict(),
            "keyframe_seconds": self.keyframe_seconds,
            "limitations": list(self.limitations),
            "method_version": self.method_version,
            "observation_quality": self.observation_quality,
            "provenance": self.provenance,
            "schema_version": self.schema_version,
            "score": self.score,
            "stage": self.stage,
            "temporal_uncertainty_seconds": self.temporal_uncertainty_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StageCandidate:
        name = "stage_candidate"
        if not isinstance(values, dict):
            raise PhaseEvidenceError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "config_id",
            "derivative_quality",
            "evidence",
            "interval",
            "keyframe_seconds",
            "limitations",
            "method_version",
            "observation_quality",
            "provenance",
            "schema_version",
            "score",
            "stage",
            "temporal_uncertainty_seconds",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseEvidenceError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseEvidenceError(f"{name}: unknown keys {unknown!r}.")
        raw_interval = values["interval"]
        if not isinstance(raw_interval, dict):
            raise PhaseEvidenceError(
                f"{name}: 'interval' must decode from a mapping, "
                f"got {type(raw_interval).__name__}."
            )
        try:
            interval = MediaRange.from_dict(raw_interval)
        except Exception as exc:
            raise PhaseEvidenceError(f"{name}: invalid 'interval': {exc}.") from exc
        return cls(
            stage=values["stage"],
            keyframe_seconds=values["keyframe_seconds"],
            interval=interval,
            score=values["score"],
            observation_quality=values["observation_quality"],
            derivative_quality=values["derivative_quality"],
            evidence=tuple(values["evidence"]),
            limitations=tuple(values["limitations"]),
            provenance=values["provenance"],
            temporal_uncertainty_seconds=values["temporal_uncertainty_seconds"],
            method_version=values["method_version"],
            config_id=values["config_id"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> StageCandidate:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PhaseEvidenceError(
                    "stage_candidate: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise PhaseEvidenceError(
                "stage_candidate: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PhaseEvidenceError(
                f"stage_candidate: invalid JSON: {exc}."
            ) from exc
        if not isinstance(decoded, dict):
            raise PhaseEvidenceError(
                "stage_candidate: JSON object is required, "
                f"got {type(decoded).__name__}."
            )
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class PhaseEvidence:
    """Immutable bounded candidate collection for one feature grid."""

    attempt_range: MediaRange = None  # type: ignore[assignment]
    method_version: str = EVIDENCE_METHOD_VERSION
    config_id: str = EVIDENCE_DEFAULT_CONFIG_ID
    candidates: tuple[StageCandidate, ...] = ()
    schema_version: int = EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_evidence"
        if not _is_int(self.schema_version):
            raise PhaseEvidenceError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != EVIDENCE_SCHEMA_VERSION:
            raise PhaseEvidenceError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {EVIDENCE_SCHEMA_VERSION}."
            )
        if not isinstance(self.attempt_range, MediaRange):
            raise PhaseEvidenceError(
                f"{name}: 'attempt_range' must be a MediaRange, "
                f"got {type(self.attempt_range).__name__}."
            )
        if not isinstance(self.method_version, str) or not self.method_version.strip():
            raise PhaseEvidenceError(f"{name}: 'method_version' must be non-blank.")
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise PhaseEvidenceError(f"{name}: 'config_id' must be non-blank.")
        raw = self.candidates
        if not isinstance(raw, (list, tuple)):
            raise PhaseEvidenceError(
                f"{name}: 'candidates' must be a list or tuple of StageCandidate."
            )
        normalized = tuple(raw)
        for entry in normalized:
            if not isinstance(entry, StageCandidate):
                raise PhaseEvidenceError(
                    f"{name}: every candidate must be a StageCandidate, "
                    f"got {type(entry).__name__}."
                )
            if (
                entry.interval.start_seconds < self.attempt_range.start_seconds
                or entry.interval.end_seconds > self.attempt_range.end_seconds
            ):
                raise PhaseEvidenceError(
                    f"{name}: candidate interval {entry.interval.to_dict()!r} "
                    f"must lie inside attempt range "
                    f"{self.attempt_range.to_dict()!r}."
                )
            if entry.method_version != self.method_version:
                raise PhaseEvidenceError(
                    f"{name}: candidate method_version {entry.method_version!r} "
                    f"must match set method_version {self.method_version!r}."
                )
            if entry.config_id != self.config_id:
                raise PhaseEvidenceError(
                    f"{name}: candidate config_id {entry.config_id!r} must match "
                    f"set config_id {self.config_id!r}."
                )
        object.__setattr__(self, "candidates", normalized)

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.candidates)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.candidates[index]

    def for_stage(self, stage: str) -> tuple[StageCandidate, ...]:
        """Return candidates for ``stage`` in stored (ranked) order."""
        if stage not in _CANONICAL_STAGES:
            raise PhaseEvidenceError(
                f"phase_evidence: unknown stage key {stage!r}."
            )
        return tuple(entry for entry in self.candidates if entry.stage == stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_range": self.attempt_range.to_dict(),
            "candidates": [entry.to_dict() for entry in self.candidates],
            "config_id": self.config_id,
            "method_version": self.method_version,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseEvidence:
        name = "phase_evidence"
        if not isinstance(values, dict):
            raise PhaseEvidenceError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempt_range",
            "candidates",
            "config_id",
            "method_version",
            "schema_version",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseEvidenceError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseEvidenceError(f"{name}: unknown keys {unknown!r}.")
        raw_range = values["attempt_range"]
        if not isinstance(raw_range, dict):
            raise PhaseEvidenceError(
                f"{name}: 'attempt_range' must decode from a mapping."
            )
        try:
            attempt_range = MediaRange.from_dict(raw_range)
        except Exception as exc:
            raise PhaseEvidenceError(
                f"{name}: invalid 'attempt_range': {exc}."
            ) from exc
        raw_candidates = values["candidates"]
        if not isinstance(raw_candidates, list):
            raise PhaseEvidenceError(
                f"{name}: 'candidates' must be a JSON list."
            )
        try:
            candidates = tuple(
                StageCandidate.from_dict(entry) for entry in raw_candidates
            )
        except PhaseEvidenceError:
            raise
        except Exception as exc:
            raise PhaseEvidenceError(
                f"{name}: invalid candidate: {exc}."
            ) from exc
        return cls(
            attempt_range=attempt_range,
            method_version=values["method_version"],
            config_id=values["config_id"],
            candidates=candidates,
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PhaseEvidence:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PhaseEvidenceError(
                    "phase_evidence: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise PhaseEvidenceError(
                "phase_evidence: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PhaseEvidenceError(f"phase_evidence: invalid JSON: {exc}.") from exc
        if not isinstance(decoded, dict):
            raise PhaseEvidenceError(
                "phase_evidence: JSON object is required, "
                f"got {type(decoded).__name__}."
            )
        return cls.from_dict(decoded)


# --- Internal cue helpers ----------------------------------------------------


def _wrist_speed(sample: Any) -> float | None:
    """Best available camera-relative wrist speed magnitude."""
    speeds: list[float] = []
    pairs = (
        (sample.wrist_left_vx, sample.wrist_left_vy),
        (sample.wrist_right_vx, sample.wrist_right_vy),
    )
    for vx, vy in pairs:
        if vx is None or vy is None:
            continue
        value = math.hypot(float(vx), float(vy))
        if math.isfinite(value):
            speeds.append(value)
    if not speeds:
        return None
    return max(speeds)


def _wrist_upward(sample: Any) -> float | None:
    """Best available upward wrist velocity (camera y grows downward)."""
    values: list[float] = []
    for vy in (sample.wrist_left_vy, sample.wrist_right_vy):
        if vy is None:
            continue
        upward = -float(vy)
        if math.isfinite(upward):
            values.append(upward)
    if not values:
        return None
    return max(values)


def _wrist_elevation_raw(sample: Any) -> float | None:
    """Best available wrist elevation proxy (higher on screen is larger)."""
    values: list[float] = []
    for key in ("wrist_left_y", "wrist_right_y"):
        raw = getattr(sample, key)
        if raw is None:
            continue
        value = -float(raw)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return max(values)


def _knee_angle_best(sample: Any) -> float | None:
    """Deepest available knee angle (smaller means more flexion)."""
    values: list[float] = []
    for key in ("knee_angle_left", "knee_angle_right"):
        raw = getattr(sample, key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return min(values)


def _elbow_flexion_best(sample: Any) -> float | None:
    """Strongest available elbow flexion in [0, 1] (flexed is larger)."""
    values: list[float] = []
    for key in ("elbow_angle_left", "elbow_angle_right"):
        raw = getattr(sample, key)
        if raw is None:
            continue
        value = (180.0 - float(raw)) / 180.0
        if math.isfinite(value):
            values.append(max(0.0, min(1.0, value)))
    if not values:
        return None
    return max(values)


def _knee_extension_vel(sample: Any) -> float | None:
    """Best available knee-extension velocity (positive means extending)."""
    values: list[float] = []
    for key in ("knee_angle_left_vel", "knee_angle_right_vel"):
        raw = getattr(sample, key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return None
    return max(values)


def _torso_disp(sample: Any) -> float | None:
    raw = sample.torso_displacement_camera
    if raw is None:
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _deceleration_raw(sample: Any) -> float | None:
    """Best available camera-relative slowing proxy (positive = slowing)."""
    values: list[float] = []
    for side in ("left", "right"):
        vx = getattr(sample, f"wrist_{side}_vx")
        vy = getattr(sample, f"wrist_{side}_vy")
        ax = getattr(sample, f"wrist_{side}_ax")
        ay = getattr(sample, f"wrist_{side}_ay")
        if vx is None or vy is None or ax is None or ay is None:
            continue
        speed = math.hypot(float(vx), float(vy))
        if not math.isfinite(speed) or speed <= 1e-12:
            continue
        slowing = -(float(vx) * float(ax) + float(vy) * float(ay)) / speed
        if math.isfinite(slowing):
            values.append(slowing)
    if not values:
        return None
    return max(values)


def _normalize_01(raw: Sequence[float | None]) -> list[float]:
    """Scale-invariant min-max normalization of one cue over a window.

    Spans at or below floating-point noise (<= 1e-9) carry no
    discriminative evidence and normalize to all zeros rather than
    amplifying numerical noise into full-scale scores."""
    finite = [float(v) for v in raw if v is not None and math.isfinite(float(v))]
    if not finite:
        return [0.0] * len(raw)
    lo = min(finite)
    hi = max(finite)
    span = hi - lo
    if not math.isfinite(span) or span <= 1e-9:
        return [0.0] * len(raw)
    out: list[float] = []
    for value in raw:
        if value is None or not math.isfinite(float(value)):
            out.append(0.0)
        else:
            out.append(max(0.0, min(1.0, (float(value) - lo) / span)))
    return out


def _local_minimum_strength(values: Sequence[float | None], radius: int = 2) -> list[float]:
    """Graded local-minimum strength in [0, 1] for a cue where small wins."""
    count = len(values)
    out = [0.0] * count
    for index in range(count):
        center = values[index]
        if center is None or not math.isfinite(float(center)):
            continue
        neighborhood: list[float] = []
        for j in range(max(0, index - radius), min(count, index + radius + 1)):
            candidate = values[j]
            if candidate is not None and math.isfinite(float(candidate)):
                neighborhood.append(float(candidate))
        if len(neighborhood) < 2:
            continue
        peak = max(neighborhood)
        trough = min(neighborhood)
        span = peak - trough
        if span <= 1e-9 or not math.isfinite(span):
            continue
        if float(center) > trough + 1e-12:
            continue
        out[index] = max(0.0, min(1.0, (peak - float(center)) / span))
    return out


def _window_indices(
    grid: PhaseFeatureGrid, config: PhaseEvidenceConfig, stage: str
) -> list[int]:
    bounds = config.search_interval(grid.attempt_range, stage)
    indices: list[int] = []
    for position, sample in enumerate(grid.samples):
        if bounds.contains(sample.time_seconds):
            indices.append(position)
    return indices


# --- Unary scores S_i(t) -----------------------------------------------------


def compute_stage_scores(
    grid: PhaseFeatureGrid,
    config: PhaseEvidenceConfig | None = None,
) -> Mapping[str, tuple[float, ...]]:
    """Compute deterministic unary evidence scores ``S_i(t)`` per stage.

    Returns a read-only mapping from each canonical stage key to a tuple
    of per-grid-sample scores in ``[0, 1]`` aligned with
    ``grid.samples``. Samples outside a stage's broad search window
    score ``0``. Visual scores are quality-weighted and capped for
    interpolated support; contact scores are scale-invariant
    normalizations over flagged transient energies only.
    """
    cfg = config if config is not None else PhaseEvidenceConfig()
    if not isinstance(cfg, PhaseEvidenceConfig):
        raise PhaseEvidenceError(
            "compute_stage_scores: 'config' must be a PhaseEvidenceConfig, "
            f"got {type(cfg).__name__}."
        )
    if not isinstance(grid, PhaseFeatureGrid):
        raise PhaseEvidenceError(
            "compute_stage_scores: 'grid' must be a PhaseFeatureGrid, "
            f"got {type(grid).__name__}."
        )
    count = len(grid.samples)
    scores: dict[str, list[float]] = {stage: [0.0] * count for stage in _CANONICAL_STAGES}
    if count == 0:
        return MappingProxyType({stage: tuple(scores[stage]) for stage in _CANONICAL_STAGES})

    # Per-sample raw cue series (full grid, None where unavailable).
    speed = [_wrist_speed(s) for s in grid.samples]
    upward = [_wrist_upward(s) for s in grid.samples]
    elevation = [_wrist_elevation_raw(s) for s in grid.samples]
    knee = [_knee_angle_best(s) for s in grid.samples]
    elbow_flex = [_elbow_flexion_best(s) for s in grid.samples]
    knee_ext = [_knee_extension_vel(s) for s in grid.samples]
    torso = [_torso_disp(s) for s in grid.samples]
    decel = [_deceleration_raw(s) for s in grid.samples]

    norm_speed = _normalize_01(speed)
    norm_upward = _normalize_01(upward)
    norm_elevation = _normalize_01(elevation)
    norm_knee_depth = _normalize_01(
        [None if v is None else -v for v in knee],
    )
    knee_min_strength = _local_minimum_strength(knee)
    norm_elbow = _normalize_01(elbow_flex)
    norm_knee_ext = _normalize_01(knee_ext)
    norm_torso = _normalize_01(torso)
    norm_decel = _normalize_01(decel)
    stillness = [1.0 - v for v in norm_speed]

    for stage in _CANONICAL_STAGES:
        window = _window_indices(grid, cfg, stage)
        if not window:
            continue
        if stage == _CONTACT_STAGE:
            _fill_contact_scores(grid, cfg, scores[stage], window)
            continue
        # Window-local normalization keeps cues scale-invariant and
        # availability-qualified: missing inputs contribute 0.
        get = _stage_cue_blend(
            stage,
            window,
            norm_speed=norm_speed,
            norm_upward=norm_upward,
            norm_elevation=norm_elevation,
            norm_knee_depth=norm_knee_depth,
            knee_min_strength=knee_min_strength,
            norm_elbow=norm_elbow,
            norm_knee_ext=norm_knee_ext,
            norm_torso=norm_torso,
            norm_decel=norm_decel,
            stillness=stillness,
        )
        for position in window:
            sample = grid.samples[position]
            raw = get[position]
            quality = _quality_weight(stage, sample)
            final = raw * quality
            if not sample.observed:
                final = min(final, cfg.interpolated_score_cap)
            scores[stage][position] = max(0.0, min(1.0, final))

    frozen = {stage: tuple(scores[stage]) for stage in _CANONICAL_STAGES}
    return MappingProxyType(frozen)


def _stage_cue_blend(
    stage: str,
    window: list[int],
    *,
    norm_speed: list[float],
    norm_upward: list[float],
    norm_elevation: list[float],
    norm_knee_depth: list[float],
    knee_min_strength: list[float],
    norm_elbow: list[float],
    norm_knee_ext: list[float],
    norm_torso: list[float],
    norm_decel: list[float],
    stillness: list[float],
) -> dict[int, float]:
    """Combine multiple window cues for one visual stage (no extremum)."""
    out: dict[int, float] = {}
    for position in window:
        if stage == "start":
            # Early sustained body motion / torso displacement proxy.
            out[position] = 0.55 * norm_speed[position] + 0.45 * norm_torso[position]
        elif stage == "release":
            # Early upward wrist trajectory/velocity (side-ambiguous).
            out[position] = 0.60 * norm_upward[position] + 0.40 * norm_speed[position]
        elif stage == "loading":
            # Knee-flexion depth plus local-minimum support.
            out[position] = 0.55 * norm_knee_depth[position] + 0.45 * knee_min_strength[position]
        elif stage == "cocking":
            # Wrist/torso geometry plus elbow-flexion evidence.
            out[position] = (
                0.40 * norm_elbow[position]
                + 0.35 * norm_elevation[position]
                + 0.25 * norm_torso[position]
            )
        elif stage == "acceleration":
            # Rising wrist/arm derivative and knee-extension proxy.
            out[position] = (
                0.45 * norm_upward[position]
                + 0.35 * norm_speed[position]
                + 0.20 * norm_knee_ext[position]
            )
        elif stage == "deceleration":
            # Post-contact camera-relative slowing proxy.
            out[position] = 0.60 * norm_decel[position] + 0.40 * norm_speed[position]
        elif stage == "finish":
            # Late recovery/stillness and torso-motion reduction.
            out[position] = 0.65 * stillness[position] + 0.35 * (1.0 - norm_torso[position])
        else:
            out[position] = 0.0
    return out


def _quality_weight(stage: str, sample: Any) -> float:
    """Availability-qualified, quality-weighted support factor in [0, 1]."""
    obs = float(sample.observation_quality)
    deriv_conf = float(sample.derivative_confidence)
    if not math.isfinite(obs) or obs < 0.0:
        obs = 0.0
    if not math.isfinite(deriv_conf) or deriv_conf < 0.0:
        deriv_conf = 0.0
    obs = max(0.0, min(1.0, obs))
    deriv_conf = max(0.0, min(1.0, deriv_conf))
    if stage in ("release", "acceleration", "deceleration", "start"):
        # Derivative-dependent cues need both observation and derivative support.
        return obs * (0.30 + 0.70 * deriv_conf)
    if stage == "finish":
        return obs * (0.50 + 0.50 * deriv_conf)
    return obs


def _fill_contact_scores(
    grid: PhaseFeatureGrid,
    config: PhaseEvidenceConfig,
    series: list[float],
    window: list[int],
) -> None:
    """Scale-invariant audio strength over flagged transients only."""
    flagged = [i for i in window if grid.samples[i].audio_candidate]
    if not flagged:
        return
    energies: dict[int, float | None] = {}
    for position in flagged:
        energy = grid.samples[position].audio_energy
        if energy is None:
            energies[position] = None
        else:
            try:
                value = float(energy)
            except (TypeError, ValueError):
                energies[position] = None
                continue
            energies[position] = value if math.isfinite(value) and value >= 0.0 else None
    valid = [v for v in energies.values() if v is not None]
    if not valid:
        return
    peak = max(valid)
    if not math.isfinite(peak) or peak <= 0.0:
        # Safe zero handling: flagged but silent energies yield no evidence.
        return
    weight = float(config.contact_audio_weight)
    for position in flagged:
        energy = energies[position]
        if energy is None:
            continue
        strength = max(0.0, min(1.0, energy / peak))
        series[position] = max(0.0, min(1.0, weight * strength + (1.0 - weight)))


# --- Candidate generation ----------------------------------------------------


_VISUAL_EVIDENCE: dict[str, tuple[str, ...]] = {
    "start": ("wrist_speed_camera", "torso_displacement_camera"),
    "release": ("wrist_upward_velocity_camera", "wrist_trajectory_camera"),
    "loading": ("knee_flexion_minimum_camera", "knee_angle_camera"),
    "cocking": ("elbow_flexion_camera", "wrist_elevation_camera", "torso_geometry_camera"),
    "acceleration": (
        "wrist_upward_velocity_camera",
        "wrist_speed_camera",
        "knee_extension_velocity_camera",
    ),
    "deceleration": ("wrist_slowing_camera", "wrist_speed_camera"),
    "finish": ("stillness_camera", "torso_motion_reduction_camera"),
}
_CONTACT_EVIDENCE: tuple[str, ...] = ("audio_transient_energy", "audio_transient_flag")


def _candidate_tokens(stage: str, sample: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return honest machine-readable (evidence, limitations) identifiers."""
    if stage == _CONTACT_STAGE:
        return (
            _CONTACT_EVIDENCE,
            ("audio_only_anchor", "camera_relative_only"),
        )
    evidence = _VISUAL_EVIDENCE.get(stage, ("body_motion_camera",))
    limitations: list[str] = ["body_pose_estimate", "camera_relative_only"]
    if stage in ("release", "cocking", "acceleration"):
        limitations.append("side_ambiguous")
    if stage == "release":
        limitations.append("no_ball_observation")
    if stage == "cocking":
        limitations.append("no_racket_observation")
    if not sample.observed:
        limitations.append("interpolated_support")
    if float(sample.derivative_confidence) < 0.5:
        limitations.append("low_derivative_confidence")
    if float(sample.observation_quality) < 0.5:
        limitations.append("low_observation_quality")
    return (evidence, tuple(limitations))


def generate_stage_candidates(
    grid: PhaseFeatureGrid,
    config: PhaseEvidenceConfig | None = None,
) -> PhaseEvidence:
    """Generate bounded, ordered, deterministic candidates per stage."""
    cfg = config if config is not None else PhaseEvidenceConfig()
    if not isinstance(cfg, PhaseEvidenceConfig):
        raise PhaseEvidenceError(
            "generate_stage_candidates: 'config' must be a PhaseEvidenceConfig, "
            f"got {type(cfg).__name__}."
        )
    if not isinstance(grid, PhaseFeatureGrid):
        raise PhaseEvidenceError(
            "generate_stage_candidates: 'grid' must be a PhaseFeatureGrid, "
            f"got {type(grid).__name__}."
        )
    scores = compute_stage_scores(grid, cfg)
    step = (1.0 / float(grid.grid_rate_hz)) if float(grid.grid_rate_hz) > 0 else 0.033
    half = float(cfg.candidate_half_window_seconds)
    floor = float(cfg.candidate_floor)
    cap = int(cfg.max_candidates_per_stage)
    attempt = grid.attempt_range

    ranked: list[StageCandidate] = []
    for stage in _CANONICAL_STAGES:
        series = scores[stage]
        # Eligible samples inside the search window at/above the floor.
        window = set(_window_indices(grid, cfg, stage))
        eligible: list[tuple[float, float, int]] = []
        for position, sample in enumerate(grid.samples):
            if position not in window:
                continue
            value = float(series[position])
            if not math.isfinite(value) or value < floor:
                continue
            if stage == _CONTACT_STAGE and not sample.audio_candidate:
                continue
            eligible.append((-value, float(sample.time_seconds), position))
        # Deterministic order: strongest first, ties earliest first,
        # then lowest grid index for exact time ties.
        eligible.sort(key=lambda entry: (entry[0], entry[1]))
        # Greedy non-maximum suppression keeps candidates bounded and
        # distinct without enforcing cross-stage chronology.
        chosen: list[int] = []
        for _, _, position in eligible:
            moment = float(grid.samples[position].time_seconds)
            suppressed = False
            for other in chosen:
                other_moment = float(grid.samples[other].time_seconds)
                if abs(moment - other_moment) < half + 1e-12:
                    suppressed = True
                    break
            if suppressed:
                continue
            chosen.append(position)
            if len(chosen) >= cap:
                break
        for position in chosen:
            sample = grid.samples[position]
            keyframe = float(sample.time_seconds)
            if stage == _CONTACT_STAGE:
                uncertainty = max(float(sample.temporal_uncertainty_seconds), step / 2.0)
                radius = uncertainty
            else:
                uncertainty = max(float(sample.temporal_uncertainty_seconds), step / 2.0)
                radius = half
            lo = max(attempt.start_seconds, keyframe - radius)
            hi = min(attempt.end_seconds, keyframe + radius)
            if not hi > lo:
                # Clamp collapse at attempt edges: nudge a minimal width.
                if keyframe >= attempt.end_seconds:
                    hi = attempt.end_seconds
                    lo = max(attempt.start_seconds, hi - step)
                else:
                    lo = max(attempt.start_seconds, min(keyframe, attempt.end_seconds - step))
                    hi = min(attempt.end_seconds, lo + step)
                if not hi > lo:
                    continue
            try:
                interval = MediaRange(start_seconds=lo, end_seconds=hi)
            except Exception:
                continue
            evidence, limitations = _candidate_tokens(stage, sample)
            ranked.append(
                StageCandidate(
                    stage=stage,
                    keyframe_seconds=keyframe,
                    interval=interval,
                    score=max(0.0, min(1.0, float(series[position]))),
                    observation_quality=max(
                        0.0, min(1.0, float(sample.observation_quality))
                    ),
                    derivative_quality=max(
                        0.0, min(1.0, float(sample.derivative_quality))
                    ),
                    evidence=evidence,
                    limitations=limitations,
                    provenance=(
                        "audio_transient" if stage == _CONTACT_STAGE else "body_pose"
                    ),
                    temporal_uncertainty_seconds=uncertainty,
                    method_version=EVIDENCE_METHOD_VERSION,
                    config_id=cfg.config_id,
                )
            )
    # Global stored order is deterministic: stage canonical order first,
    # then strongest-first with earliest-time tie-breaks within a stage.
    order = {stage: index for index, stage in enumerate(_CANONICAL_STAGES)}
    ranked.sort(key=lambda c: (order[c.stage], -c.score, c.keyframe_seconds))
    return PhaseEvidence(
        attempt_range=attempt,
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=cfg.config_id,
        candidates=tuple(ranked),
    )


def build_phase_evidence(
    grid: PhaseFeatureGrid,
    config: PhaseEvidenceConfig | None = None,
) -> PhaseEvidence:
    """Alias for :func:`generate_stage_candidates` (pure generation)."""
    return generate_stage_candidates(grid, config)
