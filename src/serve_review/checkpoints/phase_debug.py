"""Phase-debug diagnostic artifact schemas (M4 remediation, representation only).

Pure, deterministic, immutable versioned JSON value objects describing one
attempt's phase-debug diagnostic: exactly the eight canonical Kovacs stages,
method/config identities, manual and selected stage values, explicit
unavailable reasons/anomalies, reserved score/rank/objective fields, support
snapshots, candidate summaries, and bounded feature traces.

This module performs no inference, no file I/O, no CLI/pipeline/rendering,
and no production phase behavior or configuration. All absent information
remains null/empty; nested containers are immutable; codecs are
deterministic; validation is strict (finite/range/half-open/keyframe/order/
unknown-key). A single fixed v1 channel allowlist governs both support
feature values and trace samples; arbitrary channel names are rejected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from serve_review.domain import STAGE_ORDER, STAGE_PROVENANCES, MediaRange

__all__ = [
    "PHASE_DEBUG_SCHEMA_VERSION",
    "PHASE_DEBUG_METHOD_VERSION",
    "PHASE_DEBUG_DEFAULT_CONFIG_ID",
    "TRACE_CHANNEL_ALLOWLIST_V1",
    "SUPPORT_FEATURE_ALLOWLIST_V1",
    "TRACE_SAMPLING_METHOD_V1",
    "MAX_TRACE_SAMPLES",
    "MAX_WINDOWS_PER_STAGE",
    "PhaseDebugError",
    "SupportSnapshot",
    "CandidateSummary",
    "TraceSample",
    "TraceWindow",
    "StageTrace",
    "StageDebug",
    "StagePhaseDebug",
    "PhaseDebugArtifact",
    "PhaseDebug",
]

#: Version of every phase-debug schema in this module.
#: v2 adds durable ``manual_score``/``manual_rank`` fields on every
#: stage record (v1 payloads without them are rejected as incomplete).
PHASE_DEBUG_SCHEMA_VERSION = 2
#: Method identity recorded on every artifact/stage record.
PHASE_DEBUG_METHOD_VERSION = "phase-debug-v1"
#: Default configuration identity.
PHASE_DEBUG_DEFAULT_CONFIG_ID = "phase-debug-default-v1"

#: Single fixed v1 diagnostic channel allowlist. Planned channels so a later
#: builder never needs a schema change. Exactly these names are allowed;
#: arbitrary names are rejected.
TRACE_CHANNEL_ALLOWLIST_V1: tuple[str, ...] = (
    "left_wrist_x_camera",
    "left_wrist_y_camera",
    "left_wrist_upward_velocity_camera",
    "left_wrist_speed_camera",
    "face_height_threshold_camera",
    "combined_knee_flexion",
    "shoulder_line_tilt_camera_deg",
    "hip_line_tilt_camera_deg",
    "shoulder_hip_angle_difference_camera_deg",
    "right_elbow_flexion",
    "right_wrist_x_camera",
    "right_wrist_y_camera",
    "right_wrist_elevation_camera",
    "right_wrist_upward_velocity_camera",
    "right_wrist_speed_camera",
    "right_wrist_arc_length",
    "torso_rotation_proxy_camera",
    "audio_transient_energy",
    "audio_candidate",
)

#: The same fixed allowlist governs support feature values.
SUPPORT_FEATURE_ALLOWLIST_V1: tuple[str, ...] = TRACE_CHANNEL_ALLOWLIST_V1

_ALLOWED_SET = frozenset(TRACE_CHANNEL_ALLOWLIST_V1)

#: Sampling/truncation method identity for bounded traces.
TRACE_SAMPLING_METHOD_V1 = "phase-debug-trace-sampling-v1"

#: Maximum strictly increasing canonical source-time samples per stage trace.
MAX_TRACE_SAMPLES = 241
#: Maximum manual/selected windows per stage trace.
MAX_WINDOWS_PER_STAGE = 2

_CANONICAL_STAGES: tuple[str, ...] = STAGE_ORDER
_WINDOW_KINDS: tuple[str, ...] = ("manual", "selected")

_OFFSET_TOLERANCE_SECONDS = 1e-9
_UNCERTAINTY_SLACK_SECONDS = 1e-12


class PhaseDebugError(ValueError):
    """Raised when phase-debug input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_time(name: str, key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseDebugError(
            f"{name}: {key!r} must be a finite number >= 0, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseDebugError(
            f"{name}: {key!r} must be finite and >= 0, got {value!r}."
        )
    return number


def _check_optional_time(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    return _check_time(name, key, value)


def _check_optional_offset(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseDebugError(
            f"{name}: {key!r} must be a finite number or null, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise PhaseDebugError(
            f"{name}: {key!r} must be finite or null, got {value!r}."
        )
    return number


def _check_optional_unit(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseDebugError(
            f"{name}: {key!r} must be a number in [0, 1] or null, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseDebugError(
            f"{name}: {key!r} must lie in [0, 1] or be null, got {value!r}."
        )
    return number


def _check_optional_nonnegative(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseDebugError(
            f"{name}: {key!r} must be a finite number >= 0 or null, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseDebugError(
            f"{name}: {key!r} must be finite and >= 0 or null, "
            f"got {value!r}."
        )
    return number


def _check_optional_finite(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseDebugError(
            f"{name}: {key!r} must be a finite number or null, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise PhaseDebugError(
            f"{name}: {key!r} must be finite or null, got {value!r}."
        )
    return number


def _check_tokens(name: str, key: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PhaseDebugError(
            f"{name}: {key!r} must be a list or tuple of non-blank strings, "
            f"got {type(value).__name__}."
        )
    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise PhaseDebugError(
                f"{name}: {key!r} entries must be non-blank strings, "
                f"got {entry!r}."
            )
        if entry in seen:
            raise PhaseDebugError(
                f"{name}: {key!r} must not contain duplicates, got {entry!r}."
            )
        seen.add(entry)
        cleaned.append(entry)
    return tuple(cleaned)


def _check_attempt_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise PhaseDebugError(
            f"{name}: 'attempt_id' must be a non-empty string, got {value!r}."
        )
    prefix = "serve-"
    if not value.startswith(prefix):
        raise PhaseDebugError(
            f"{name}: 'attempt_id' must look like 'serve-NNN' with at least "
            f"three digits, got {value!r}."
        )
    body = value[len(prefix):]
    if len(body) < 3 or not body.isdigit():
        raise PhaseDebugError(
            f"{name}: 'attempt_id' must look like 'serve-NNN' with at least "
            f"three digits, got {value!r}."
        )
    return value


def _check_non_blank(name: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseDebugError(
            f"{name}: {key!r} must be a non-blank string, got {value!r}."
        )
    return value


def _check_feature_values(
    name: str, value: Any
) -> Mapping[str, float | bool | None]:
    if not isinstance(value, dict):
        raise PhaseDebugError(
            f"{name}: 'feature_values' must decode from a mapping, "
            f"got {type(value).__name__}."
        )
    cleaned: dict[str, float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise PhaseDebugError(
                f"{name}: 'feature_values' keys must be non-blank strings, "
                f"got {key!r}."
            )
        if key not in _ALLOWED_SET:
            raise PhaseDebugError(
                f"{name}: unknown diagnostic channel {key!r}; allowed "
                f"channels are fixed in TRACE_CHANNEL_ALLOWLIST_V1."
            )
        if item is None:
            cleaned[key] = None
            continue
        if key == "audio_candidate":
            if isinstance(item, bool):
                cleaned[key] = item
                continue
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                number = float(item)
                if not math.isfinite(number):
                    raise PhaseDebugError(
                        f"{name}: 'feature_values[{key}]' must be finite, "
                        f"a bool, or null, got {item!r}."
                    )
                cleaned[key] = number
                continue
            raise PhaseDebugError(
                f"{name}: 'feature_values[{key}]' must be a bool, finite "
                f"number, or null, got {item!r}."
            )
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PhaseDebugError(
                f"{name}: 'feature_values[{key}]' must be a finite number "
                f"or null, got {item!r}."
            )
        number = float(item)
        if not math.isfinite(number):
            raise PhaseDebugError(
                f"{name}: 'feature_values[{key}]' must be finite or null, "
                f"got {item!r}."
            )
        cleaned[key] = number
    return MappingProxyType(cleaned)


def _feature_values_to_dict(
    values: Mapping[str, float | bool | None],
) -> dict[str, Any]:
    return {key: values[key] for key in sorted(values)}


def _dumps_deterministic(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _loads_object(name: str, data: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhaseDebugError(
                f"{name}: invalid UTF-8 JSON payload."
            ) from exc
    if not isinstance(data, str):
        raise PhaseDebugError(
            f"{name}: JSON payload must be str or bytes, "
            f"got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PhaseDebugError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise PhaseDebugError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


def _require_keys(name: str, values: dict[str, Any], known: set[str]) -> None:
    missing = sorted(known - set(values))
    if missing:
        raise PhaseDebugError(f"{name}: missing required keys {missing!r}.")
    unknown = sorted(set(values) - known)
    if unknown:
        raise PhaseDebugError(f"{name}: unknown keys {unknown!r}.")


# --- Support snapshots -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportSnapshot:
    """Immutable support snapshot at one queried canonical source time.

    ``queried_time_seconds`` is the canonical query time;
    ``support_time_seconds`` is the nearest support canonical time or null
    when no support exists; ``support_offset_seconds`` is the exact signed
    ``support-minus-query`` offset or null when no support exists.
    ``observed`` is True only for a direct observation. Quality/span/
    uncertainty/confidence fields are null when support is absent.
    ``feature_values`` is an immutable mapping of available values keyed by
    the single fixed v1 allowlist; absent channels are simply missing
    (empty mapping means no available values).
    """

    queried_time_seconds: float = 0.0
    support_time_seconds: float | None = None
    support_offset_seconds: float | None = None
    observed: bool = False
    observation_quality: float | None = None
    interpolation_span_seconds: float | None = None
    temporal_uncertainty_seconds: float | None = None
    derivative_confidence: float | None = None
    feature_values: Mapping[str, float | bool | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    schema_version: int = PHASE_DEBUG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "support_snapshot"
        if not _is_int(self.schema_version):
            raise PhaseDebugError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_DEBUG_SCHEMA_VERSION:
            raise PhaseDebugError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_DEBUG_SCHEMA_VERSION}."
            )
        object.__setattr__(
            self, "queried_time_seconds",
            _check_time(name, "'queried_time_seconds'", self.queried_time_seconds),
        )
        object.__setattr__(
            self, "support_time_seconds",
            _check_optional_time(name, "'support_time_seconds'", self.support_time_seconds),
        )
        object.__setattr__(
            self, "support_offset_seconds",
            _check_optional_offset(name, "'support_offset_seconds'", self.support_offset_seconds),
        )
        if not isinstance(self.observed, bool):
            raise PhaseDebugError(
                f"{name}: 'observed' must be a bool, got {self.observed!r}."
            )
        object.__setattr__(
            self, "observation_quality",
            _check_optional_unit(name, "'observation_quality'", self.observation_quality),
        )
        object.__setattr__(
            self, "interpolation_span_seconds",
            _check_optional_nonnegative(
                name, "'interpolation_span_seconds'", self.interpolation_span_seconds
            ),
        )
        object.__setattr__(
            self, "temporal_uncertainty_seconds",
            _check_optional_nonnegative(
                name, "'temporal_uncertainty_seconds'",
                self.temporal_uncertainty_seconds,
            ),
        )
        object.__setattr__(
            self, "derivative_confidence",
            _check_optional_unit(
                name, "'derivative_confidence'", self.derivative_confidence
            ),
        )
        raw_features = self.feature_values
        if isinstance(raw_features, Mapping):
            as_dict = dict(raw_features)
        elif isinstance(raw_features, dict):
            as_dict = dict(raw_features)
        else:
            raise PhaseDebugError(
                f"{name}: 'feature_values' must be a mapping, "
                f"got {type(raw_features).__name__}."
            )
        object.__setattr__(
            self, "feature_values", _check_feature_values(name, as_dict)
        )
        queried = float(self.queried_time_seconds)
        support = self.support_time_seconds
        offset = self.support_offset_seconds
        features = self.feature_values
        assert isinstance(features, Mapping)
        if support is None:
            if offset is not None:
                raise PhaseDebugError(
                    f"{name}: 'support_offset_seconds' must be null when "
                    f"'support_time_seconds' is null, got {offset!r}."
                )
            if self.observed:
                raise PhaseDebugError(
                    f"{name}: 'observed' must be false when "
                    f"'support_time_seconds' is null."
                )
            if (
                self.observation_quality is not None
                or self.interpolation_span_seconds is not None
                or self.temporal_uncertainty_seconds is not None
                or self.derivative_confidence is not None
            ):
                raise PhaseDebugError(
                    f"{name}: quality/span/uncertainty/confidence must all be "
                    f"null when 'support_time_seconds' is null."
                )
            if len(features) != 0:
                raise PhaseDebugError(
                    f"{name}: 'feature_values' must be empty when "
                    f"'support_time_seconds' is null."
                )
            return
        expected = float(support) - queried
        if offset is None:
            raise PhaseDebugError(
                f"{name}: 'support_offset_seconds' must be present when "
                f"'support_time_seconds' is present."
            )
        if abs(float(offset) - expected) > _OFFSET_TOLERANCE_SECONDS:
            raise PhaseDebugError(
                f"{name}: 'support_offset_seconds' ({offset!r}) must equal "
                f"support-minus-query ({expected!r})."
            )
        span = self.interpolation_span_seconds
        uncertainty = self.temporal_uncertainty_seconds
        if span is None or uncertainty is None:
            raise PhaseDebugError(
                f"{name}: 'interpolation_span_seconds' and "
                f"'temporal_uncertainty_seconds' must be present when support "
                f"is present."
            )
        if self.observation_quality is None or self.derivative_confidence is None:
            raise PhaseDebugError(
                f"{name}: 'observation_quality' and 'derivative_confidence' "
                f"must be present when support is present."
            )
        if uncertainty + _UNCERTAINTY_SLACK_SECONDS < abs(float(offset)):
            raise PhaseDebugError(
                f"{name}: 'temporal_uncertainty_seconds' must be >= "
                f"abs(support_offset_seconds)."
            )
        if uncertainty + _UNCERTAINTY_SLACK_SECONDS < float(span) / 2.0:
            raise PhaseDebugError(
                f"{name}: 'temporal_uncertainty_seconds' must be >= "
                f"interpolation_span_seconds / 2."
            )
        if self.observed:
            if float(span) != 0.0:
                raise PhaseDebugError(
                    f"{name}: observed snapshots must carry "
                    f"interpolation_span_seconds == 0.0, got {span!r}."
                )
        else:
            if (
                self.observation_quality is not None
                and float(self.observation_quality) == 1.0
            ):
                raise PhaseDebugError(
                    f"{name}: non-observed snapshots must not claim "
                    f"observation_quality == 1.0."
                )

    def to_dict(self) -> dict[str, Any]:
        features = self.feature_values
        assert isinstance(features, Mapping)
        return {
            "derivative_confidence": self.derivative_confidence,
            "feature_values": _feature_values_to_dict(features),
            "interpolation_span_seconds": self.interpolation_span_seconds,
            "observation_quality": self.observation_quality,
            "observed": self.observed,
            "queried_time_seconds": self.queried_time_seconds,
            "schema_version": self.schema_version,
            "support_offset_seconds": self.support_offset_seconds,
            "support_time_seconds": self.support_time_seconds,
            "temporal_uncertainty_seconds": self.temporal_uncertainty_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SupportSnapshot:
        name = "support_snapshot"
        if not isinstance(values, dict):
            raise PhaseDebugError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "derivative_confidence",
            "feature_values",
            "interpolation_span_seconds",
            "observation_quality",
            "observed",
            "queried_time_seconds",
            "schema_version",
            "support_offset_seconds",
            "support_time_seconds",
            "temporal_uncertainty_seconds",
        }
        _require_keys(name, values, known)
        raw_features = values["feature_values"]
        if not isinstance(raw_features, dict):
            raise PhaseDebugError(
                f"{name}: 'feature_values' must decode from a mapping, "
                f"got {type(raw_features).__name__}."
            )
        return cls(
            queried_time_seconds=values["queried_time_seconds"],
            support_time_seconds=values["support_time_seconds"],
            support_offset_seconds=values["support_offset_seconds"],
            observed=values["observed"],
            observation_quality=values["observation_quality"],
            interpolation_span_seconds=values["interpolation_span_seconds"],
            temporal_uncertainty_seconds=values["temporal_uncertainty_seconds"],
            derivative_confidence=values["derivative_confidence"],
            feature_values=raw_features,
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> SupportSnapshot:
        return cls.from_dict(_loads_object("support_snapshot", data))


# --- Candidate summaries -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    """Immutable summary of one stage candidate (or honest absence).

    ``interval`` is a half-open ``[start, end)`` range or null when no
    candidate exists; ``keyframe_seconds`` lies inside ``interval``.
    ``unary_score`` is the reserved unary evidence score in [0, 1] or null;
    ``rank`` is the deterministic 1-based rank (1 is best) or null;
    ``observation_quality``/``derivative_quality`` echo support quality;
    ``provenance`` is a canonical provenance or null;
    ``temporal_uncertainty_seconds`` is an uncertainty width or null;
    ``evidence_ids``/``limitation_ids`` are identifier collections.
    """

    stage: str = "start"
    interval: MediaRange | None = None
    keyframe_seconds: float | None = None
    unary_score: float | None = None
    rank: int | None = None
    observation_quality: float | None = None
    derivative_quality: float | None = None
    provenance: str | None = None
    temporal_uncertainty_seconds: float | None = None
    evidence_ids: tuple[str, ...] = ()
    limitation_ids: tuple[str, ...] = ()
    schema_version: int = PHASE_DEBUG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "candidate_summary"
        if not _is_int(self.schema_version):
            raise PhaseDebugError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_DEBUG_SCHEMA_VERSION:
            raise PhaseDebugError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_DEBUG_SCHEMA_VERSION}."
            )
        if self.stage not in _CANONICAL_STAGES:
            raise PhaseDebugError(
                f"{name}: 'stage' must be one of {list(_CANONICAL_STAGES)}, "
                f"got {self.stage!r}."
            )
        interval = self.interval
        if interval is not None and not isinstance(interval, MediaRange):
            raise PhaseDebugError(
                f"{name}: 'interval' must be a MediaRange or null, "
                f"got {type(interval).__name__}."
            )
        object.__setattr__(
            self, "keyframe_seconds",
            _check_optional_time(name, "'keyframe_seconds'", self.keyframe_seconds),
        )
        object.__setattr__(
            self, "unary_score",
            _check_optional_unit(name, "'unary_score'", self.unary_score),
        )
        rank = self.rank
        if rank is not None:
            if not _is_int(rank) or rank < 1:
                raise PhaseDebugError(
                    f"{name}: 'rank' must be an integer >= 1 or null, "
                    f"got {rank!r}."
                )
            object.__setattr__(self, "rank", int(rank))
        object.__setattr__(
            self, "observation_quality",
            _check_optional_unit(
                name, "'observation_quality'", self.observation_quality
            ),
        )
        object.__setattr__(
            self, "derivative_quality",
            _check_optional_unit(
                name, "'derivative_quality'", self.derivative_quality
            ),
        )
        provenance = self.provenance
        if provenance is not None and provenance not in STAGE_PROVENANCES:
            raise PhaseDebugError(
                f"{name}: 'provenance' must be one of "
                f"{list(STAGE_PROVENANCES)} or null, got {provenance!r}."
            )
        object.__setattr__(
            self, "temporal_uncertainty_seconds",
            _check_optional_nonnegative(
                name, "'temporal_uncertainty_seconds'",
                self.temporal_uncertainty_seconds,
            ),
        )
        object.__setattr__(
            self, "evidence_ids",
            _check_tokens(name, "'evidence_ids'", self.evidence_ids),
        )
        object.__setattr__(
            self, "limitation_ids",
            _check_tokens(name, "'limitation_ids'", self.limitation_ids),
        )
        if interval is None:
            if self.keyframe_seconds is not None:
                raise PhaseDebugError(
                    f"{name}: 'keyframe_seconds' must be null when 'interval' "
                    f"is null."
                )
            if self.unary_score is not None:
                raise PhaseDebugError(
                    f"{name}: 'unary_score' must be null when 'interval' is null."
                )
            if self.rank is not None:
                raise PhaseDebugError(
                    f"{name}: 'rank' must be null when 'interval' is null."
                )
            if (
                self.observation_quality is not None
                or self.derivative_quality is not None
                or self.provenance is not None
                or self.temporal_uncertainty_seconds is not None
            ):
                raise PhaseDebugError(
                    f"{name}: quality/provenance/uncertainty must all be null "
                    f"when 'interval' is null."
                )
            if len(self.evidence_ids) != 0:
                raise PhaseDebugError(
                    f"{name}: 'evidence_ids' must be empty when 'interval' "
                    f"is null."
                )
            return
        keyframe = self.keyframe_seconds
        if keyframe is None:
            raise PhaseDebugError(
                f"{name}: 'keyframe_seconds' must be present when 'interval' "
                f"is present."
            )
        if not interval.contains(float(keyframe)) and not (
            float(keyframe) == interval.start_seconds
        ):
            raise PhaseDebugError(
                f"{name}: 'keyframe_seconds' ({keyframe!r}) must lie in "
                f"interval [{interval.start_seconds!r}, "
                f"{interval.end_seconds!r})."
            )
        if not (interval.start_seconds <= float(keyframe) < interval.end_seconds):
            raise PhaseDebugError(
                f"{name}: 'keyframe_seconds' ({keyframe!r}) must lie in "
                f"interval [{interval.start_seconds!r}, "
                f"{interval.end_seconds!r})."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "derivative_quality": self.derivative_quality,
            "evidence_ids": list(self.evidence_ids),
            "interval": None
            if self.interval is None
            else self.interval.to_dict(),
            "keyframe_seconds": self.keyframe_seconds,
            "limitation_ids": list(self.limitation_ids),
            "observation_quality": self.observation_quality,
            "provenance": self.provenance,
            "rank": self.rank,
            "schema_version": self.schema_version,
            "stage": self.stage,
            "temporal_uncertainty_seconds": self.temporal_uncertainty_seconds,
            "unary_score": self.unary_score,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CandidateSummary:
        name = "candidate_summary"
        if not isinstance(values, dict):
            raise PhaseDebugError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "derivative_quality",
            "evidence_ids",
            "interval",
            "keyframe_seconds",
            "limitation_ids",
            "observation_quality",
            "provenance",
            "rank",
            "schema_version",
            "stage",
            "temporal_uncertainty_seconds",
            "unary_score",
        }
        _require_keys(name, values, known)
        raw_interval = values["interval"]
        interval: MediaRange | None
        if raw_interval is None:
            interval = None
        elif isinstance(raw_interval, dict):
            try:
                interval = MediaRange.from_dict(raw_interval)
            except Exception as exc:
                raise PhaseDebugError(
                    f"{name}: invalid 'interval': {exc}."
                ) from exc
        else:
            raise PhaseDebugError(
                f"{name}: 'interval' must decode from a mapping or null, "
                f"got {type(raw_interval).__name__}."
            )
        return cls(
            stage=values["stage"],
            interval=interval,
            keyframe_seconds=values["keyframe_seconds"],
            unary_score=values["unary_score"],
            rank=values["rank"],
            observation_quality=values["observation_quality"],
            derivative_quality=values["derivative_quality"],
            provenance=values["provenance"],
            temporal_uncertainty_seconds=values["temporal_uncertainty_seconds"],
            evidence_ids=tuple(values["evidence_ids"]),
            limitation_ids=tuple(values["limitation_ids"]),
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> CandidateSummary:
        return cls.from_dict(_loads_object("candidate_summary", data))


# --- Bounded traces ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceSample:
    """One immutable canonical source-time trace sample.

    ``time_seconds`` is canonical source time; ``values`` maps available
    diagnostic channels (fixed v1 allowlist subset) to finite numbers
    (bool allowed only for ``audio_candidate``) or null. Absent channels
    are simply missing.
    """

    time_seconds: float = 0.0
    values: Mapping[str, float | bool | None] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        name = "trace_sample"
        object.__setattr__(
            self, "time_seconds",
            _check_time(name, "'time_seconds'", self.time_seconds),
        )
        raw = self.values
        if isinstance(raw, Mapping):
            as_dict = dict(raw)
        elif isinstance(raw, dict):
            as_dict = dict(raw)
        else:
            raise PhaseDebugError(
                f"{name}: 'values' must be a mapping, "
                f"got {type(raw).__name__}."
            )
        object.__setattr__(self, "values", _check_feature_values(name, as_dict))

    def to_dict(self) -> dict[str, Any]:
        values = self.values
        assert isinstance(values, Mapping)
        return {
            "time_seconds": self.time_seconds,
            "values": _feature_values_to_dict(values),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TraceSample:
        name = "trace_sample"
        if not isinstance(values, dict):
            raise PhaseDebugError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"time_seconds", "values"}
        _require_keys(name, values, known)
        raw_values = values["values"]
        if not isinstance(raw_values, dict):
            raise PhaseDebugError(
                f"{name}: 'values' must decode from a mapping, "
                f"got {type(raw_values).__name__}."
            )
        return cls(time_seconds=values["time_seconds"], values=raw_values)


@dataclass(frozen=True, slots=True)
class TraceWindow:
    """One immutable manual/selected trace window (half-open interval)."""

    kind: str = "manual"
    interval: MediaRange | None = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        name = "trace_window"
        if self.kind not in _WINDOW_KINDS:
            raise PhaseDebugError(
                f"{name}: 'kind' must be one of {list(_WINDOW_KINDS)}, "
                f"got {self.kind!r}."
            )
        if not isinstance(self.interval, MediaRange):
            raise PhaseDebugError(
                f"{name}: 'interval' must be a MediaRange, "
                f"got {type(self.interval).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        interval = self.interval
        assert isinstance(interval, MediaRange)
        return {"interval": interval.to_dict(), "kind": self.kind}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TraceWindow:
        name = "trace_window"
        if not isinstance(values, dict):
            raise PhaseDebugError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"interval", "kind"}
        _require_keys(name, values, known)
        raw_interval = values["interval"]
        if not isinstance(raw_interval, dict):
            raise PhaseDebugError(
                f"{name}: 'interval' must decode from a mapping, "
                f"got {type(raw_interval).__name__}."
            )
        try:
            interval = MediaRange.from_dict(raw_interval)
        except Exception as exc:
            raise PhaseDebugError(
                f"{name}: invalid 'interval': {exc}."
            ) from exc
        return cls(kind=values["kind"], interval=interval)


@dataclass(frozen=True, slots=True)
class StageTrace:
    """Immutable bounded trace for one canonical stage.

    At most two windows (one manual, one selected), at most
    ``MAX_TRACE_SAMPLES`` strictly increasing canonical source-time
    samples, the pre-truncation ``source_sample_count``, the
    sampling/truncation method identity, and the truncation flag.
    Sample values use the single fixed v1 allowlist.
    """

    stage: str = "start"
    windows: tuple[TraceWindow, ...] = ()
    samples: tuple[TraceSample, ...] = ()
    source_sample_count: int = 0
    sampling_method: str = TRACE_SAMPLING_METHOD_V1
    truncated: bool = False
    schema_version: int = PHASE_DEBUG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "stage_trace"
        if not _is_int(self.schema_version):
            raise PhaseDebugError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_DEBUG_SCHEMA_VERSION:
            raise PhaseDebugError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_DEBUG_SCHEMA_VERSION}."
            )
        if self.stage not in _CANONICAL_STAGES:
            raise PhaseDebugError(
                f"{name}: 'stage' must be one of {list(_CANONICAL_STAGES)}, "
                f"got {self.stage!r}."
            )
        raw_windows = self.windows
        if not isinstance(raw_windows, (list, tuple)):
            raise PhaseDebugError(
                f"{name}: 'windows' must be a list or tuple of TraceWindow."
            )
        windows = tuple(raw_windows)
        for entry in windows:
            if not isinstance(entry, TraceWindow):
                raise PhaseDebugError(
                    f"{name}: every window must be a TraceWindow, "
                    f"got {type(entry).__name__}."
                )
        if len(windows) > MAX_WINDOWS_PER_STAGE:
            raise PhaseDebugError(
                f"{name}: at most {MAX_WINDOWS_PER_STAGE} windows per stage, "
                f"got {len(windows)}."
            )
        kinds = [entry.kind for entry in windows]
        if len(set(kinds)) != len(kinds):
            raise PhaseDebugError(
                f"{name}: window kinds must be unique (at most one manual "
                f"and one selected), got {kinds!r}."
            )
        object.__setattr__(self, "windows", windows)
        raw_samples = self.samples
        if not isinstance(raw_samples, (list, tuple)):
            raise PhaseDebugError(
                f"{name}: 'samples' must be a list or tuple of TraceSample."
            )
        samples = tuple(raw_samples)
        for entry in samples:
            if not isinstance(entry, TraceSample):
                raise PhaseDebugError(
                    f"{name}: every sample must be a TraceSample, "
                    f"got {type(entry).__name__}."
                )
        if len(samples) > MAX_TRACE_SAMPLES:
            raise PhaseDebugError(
                f"{name}: at most {MAX_TRACE_SAMPLES} samples per stage, "
                f"got {len(samples)}."
            )
        for index in range(1, len(samples)):
            if not samples[index].time_seconds > samples[index - 1].time_seconds:
                raise PhaseDebugError(
                    f"{name}: sample times must be strictly increasing."
                )
        object.__setattr__(self, "samples", samples)
        if not _is_int(self.source_sample_count) or self.source_sample_count < 0:
            raise PhaseDebugError(
                f"{name}: 'source_sample_count' must be an integer >= 0, "
                f"got {self.source_sample_count!r}."
            )
        if self.source_sample_count < len(samples):
            raise PhaseDebugError(
                f"{name}: 'source_sample_count' ({self.source_sample_count!r}) "
                f"must be >= len(samples) ({len(samples)!r})."
            )
        if not isinstance(self.truncated, bool):
            raise PhaseDebugError(
                f"{name}: 'truncated' must be a bool, "
                f"got {self.truncated!r}."
            )
        if self.truncated:
            if not self.source_sample_count > len(samples):
                raise PhaseDebugError(
                    f"{name}: truncated traces must carry "
                    f"'source_sample_count' > len(samples)."
                )
        else:
            if not self.source_sample_count == len(samples):
                raise PhaseDebugError(
                    f"{name}: non-truncated traces must carry "
                    f"'source_sample_count' == len(samples), got "
                    f"{self.source_sample_count!r} vs {len(samples)!r}."
                )
        object.__setattr__(
            self, "sampling_method",
            _check_non_blank(name, "'sampling_method'", self.sampling_method),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [entry.to_dict() for entry in self.samples],
            "sampling_method": self.sampling_method,
            "schema_version": self.schema_version,
            "source_sample_count": self.source_sample_count,
            "stage": self.stage,
            "truncated": self.truncated,
            "windows": [entry.to_dict() for entry in self.windows],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StageTrace:
        name = "stage_trace"
        if not isinstance(values, dict):
            raise PhaseDebugError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "samples",
            "sampling_method",
            "schema_version",
            "source_sample_count",
            "stage",
            "truncated",
            "windows",
        }
        _require_keys(name, values, known)
        raw_windows = values["windows"]
        if not isinstance(raw_windows, list):
            raise PhaseDebugError(
                f"{name}: 'windows' must be a JSON list."
            )
        try:
            windows = tuple(TraceWindow.from_dict(entry) for entry in raw_windows)
        except PhaseDebugError:
            raise
        except Exception as exc:
            raise PhaseDebugError(
                f"{name}: invalid window: {exc}."
            ) from exc
        raw_samples = values["samples"]
        if not isinstance(raw_samples, list):
            raise PhaseDebugError(
                f"{name}: 'samples' must be a JSON list."
            )
        try:
            samples = tuple(TraceSample.from_dict(entry) for entry in raw_samples)
        except PhaseDebugError:
            raise
        except Exception as exc:
            raise PhaseDebugError(
                f"{name}: invalid sample: {exc}."
            ) from exc
        return cls(
            stage=values["stage"],
            windows=windows,
            samples=samples,
            source_sample_count=values["source_sample_count"],
            sampling_method=values["sampling_method"],
            truncated=values["truncated"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> StageTrace:
        return cls.from_dict(_loads_object("stage_trace", data))


# --- Per-stage debug ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageDebug:
    """Immutable debug record for one canonical stage.

    ``manual_time_seconds``/``selected_time_seconds`` are the manual and
    selected canonical values (null when absent); ``manual_support``/
    ``selected_support`` are the corresponding support snapshots;
    ``candidates`` holds candidate summaries for this stage;
    ``manual_score``/``manual_rank`` are the durable manual score/rank
    fields (null when absent; an exact candidate-keyframe match at the
    manual time yields that candidate's unary score/rank, otherwise
    null/null -- never interpolated); ``selected_score``/``selected_rank``
    are the corresponding selected fields;
    ``objective_contribution`` is the reserved objective field (null when
    absent); ``unavailable_reason`` is explicit when nothing was selected;
    ``anomalies`` carries stable anomaly identifiers.
    """

    stage: str = "start"
    manual_time_seconds: float | None = None
    selected_time_seconds: float | None = None
    manual_support: SupportSnapshot | None = None
    selected_support: SupportSnapshot | None = None
    candidates: tuple[CandidateSummary, ...] = ()
    manual_score: float | None = None
    manual_rank: int | None = None
    selected_score: float | None = None
    selected_rank: int | None = None
    objective_contribution: float | None = None
    unavailable_reason: str | None = None
    anomalies: tuple[str, ...] = ()
    schema_version: int = PHASE_DEBUG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "stage_debug"
        if not _is_int(self.schema_version):
            raise PhaseDebugError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_DEBUG_SCHEMA_VERSION:
            raise PhaseDebugError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_DEBUG_SCHEMA_VERSION}."
            )
        if self.stage not in _CANONICAL_STAGES:
            raise PhaseDebugError(
                f"{name}: 'stage' must be one of {list(_CANONICAL_STAGES)}, "
                f"got {self.stage!r}."
            )
        object.__setattr__(
            self, "manual_time_seconds",
            _check_optional_time(
                name, "'manual_time_seconds'", self.manual_time_seconds
            ),
        )
        object.__setattr__(
            self, "selected_time_seconds",
            _check_optional_time(
                name, "'selected_time_seconds'", self.selected_time_seconds
            ),
        )
        for key in ("manual_support", "selected_support"):
            entry = getattr(self, key)
            if entry is not None and not isinstance(entry, SupportSnapshot):
                raise PhaseDebugError(
                    f"{name}: {key!r} must be a SupportSnapshot or null, "
                    f"got {type(entry).__name__}."
                )
        manual_support = self.manual_support
        selected_support = self.selected_support
        manual_time = self.manual_time_seconds
        selected_time = self.selected_time_seconds
        if manual_support is not None:
            if manual_time is None:
                raise PhaseDebugError(
                    f"{name}: 'manual_support' requires 'manual_time_seconds'."
                )
            if abs(float(manual_support.queried_time_seconds) - float(manual_time)) > (
                _OFFSET_TOLERANCE_SECONDS
            ):
                raise PhaseDebugError(
                    f"{name}: manual support query "
                    f"({manual_support.queried_time_seconds!r}) must match "
                    f"'manual_time_seconds' ({manual_time!r})."
                )
        if selected_support is not None:
            if selected_time is None:
                raise PhaseDebugError(
                    f"{name}: 'selected_support' requires "
                    f"'selected_time_seconds'."
                )
            if abs(
                float(selected_support.queried_time_seconds) - float(selected_time)
            ) > _OFFSET_TOLERANCE_SECONDS:
                raise PhaseDebugError(
                    f"{name}: selected support query "
                    f"({selected_support.queried_time_seconds!r}) must match "
                    f"'selected_time_seconds' ({selected_time!r})."
                )
        raw_candidates = self.candidates
        if not isinstance(raw_candidates, (list, tuple)):
            raise PhaseDebugError(
                f"{name}: 'candidates' must be a list or tuple of "
                f"CandidateSummary."
            )
        candidates = tuple(raw_candidates)
        for entry in candidates:
            if not isinstance(entry, CandidateSummary):
                raise PhaseDebugError(
                    f"{name}: every candidate must be a CandidateSummary, "
                    f"got {type(entry).__name__}."
                )
            if entry.stage != self.stage:
                raise PhaseDebugError(
                    f"{name}: candidate stage {entry.stage!r} must match "
                    f"parent stage {self.stage!r}."
                )
        ranks = [entry.rank for entry in candidates if entry.rank is not None]
        if len(set(ranks)) != len(ranks):
            raise PhaseDebugError(
                f"{name}: candidate ranks must be unique, got {ranks!r}."
            )
        if ranks:
            ordered = sorted(ranks)
            stored = [entry.rank for entry in candidates if entry.rank is not None]
            # Deterministic rank order: stored order must follow rank order
            # over the ranked subset.
            ranked_only = [entry for entry in candidates if entry.rank is not None]
            if [entry.rank for entry in ranked_only] != ordered:
                raise PhaseDebugError(
                    f"{name}: candidates must be stored in deterministic rank "
                    f"order, got {[e.rank for e in ranked_only]!r}."
                )
            _ = stored
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self, "manual_score",
            _check_optional_unit(name, "'manual_score'", self.manual_score),
        )
        manual_rank = self.manual_rank
        if manual_rank is not None:
            if not _is_int(manual_rank) or manual_rank < 1:
                raise PhaseDebugError(
                    f"{name}: 'manual_rank' must be an integer >= 1 or "
                    f"null, got {manual_rank!r}."
                )
            object.__setattr__(self, "manual_rank", int(manual_rank))
        object.__setattr__(
            self, "selected_score",
            _check_optional_unit(name, "'selected_score'", self.selected_score),
        )
        rank = self.selected_rank
        if rank is not None:
            if not _is_int(rank) or rank < 1:
                raise PhaseDebugError(
                    f"{name}: 'selected_rank' must be an integer >= 1 or "
                    f"null, got {rank!r}."
                )
            object.__setattr__(self, "selected_rank", int(rank))
        object.__setattr__(
            self, "objective_contribution",
            _check_optional_finite(
                name, "'objective_contribution'", self.objective_contribution
            ),
        )
        reason = self.unavailable_reason
        if reason is not None:
            if not isinstance(reason, str) or not reason.strip():
                raise PhaseDebugError(
                    f"{name}: 'unavailable_reason' must be a non-blank string "
                    f"or null, got {reason!r}."
                )
        has_selected = (
            selected_time is not None
            or selected_support is not None
            or any(entry.interval is not None for entry in candidates)
            or self.selected_score is not None
            or self.selected_rank is not None
        )
        if has_selected:
            if reason is not None:
                raise PhaseDebugError(
                    f"{name}: 'unavailable_reason' must be null when selected "
                    f"information is present."
                )
        else:
            if reason is None:
                raise PhaseDebugError(
                    f"{name}: 'unavailable_reason' is required when no "
                    f"selected information is present."
                )
        object.__setattr__(
            self, "anomalies", _check_tokens(name, "'anomalies'", self.anomalies)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomalies": list(self.anomalies),
            "candidates": [entry.to_dict() for entry in self.candidates],
            "manual_score": self.manual_score,
            "manual_rank": self.manual_rank,
            "manual_support": None
            if self.manual_support is None
            else self.manual_support.to_dict(),
            "manual_time_seconds": self.manual_time_seconds,
            "objective_contribution": self.objective_contribution,
            "schema_version": self.schema_version,
            "selected_rank": self.selected_rank,
            "selected_score": self.selected_score,
            "selected_support": None
            if self.selected_support is None
            else self.selected_support.to_dict(),
            "selected_time_seconds": self.selected_time_seconds,
            "stage": self.stage,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StageDebug:
        name = "stage_debug"
        if not isinstance(values, dict):
            raise PhaseDebugError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "anomalies",
            "candidates",
            "manual_rank",
            "manual_score",
            "manual_support",
            "manual_time_seconds",
            "objective_contribution",
            "schema_version",
            "selected_rank",
            "selected_score",
            "selected_support",
            "selected_time_seconds",
            "stage",
            "unavailable_reason",
        }
        _require_keys(name, values, known)
        raw_manual = values["manual_support"]
        manual_support: SupportSnapshot | None
        if raw_manual is None:
            manual_support = None
        elif isinstance(raw_manual, dict):
            try:
                manual_support = SupportSnapshot.from_dict(raw_manual)
            except PhaseDebugError:
                raise
            except Exception as exc:
                raise PhaseDebugError(
                    f"{name}: invalid 'manual_support': {exc}."
                ) from exc
        else:
            raise PhaseDebugError(
                f"{name}: 'manual_support' must decode from a mapping or "
                f"null, got {type(raw_manual).__name__}."
            )
        raw_selected = values["selected_support"]
        selected_support: SupportSnapshot | None
        if raw_selected is None:
            selected_support = None
        elif isinstance(raw_selected, dict):
            try:
                selected_support = SupportSnapshot.from_dict(raw_selected)
            except PhaseDebugError:
                raise
            except Exception as exc:
                raise PhaseDebugError(
                    f"{name}: invalid 'selected_support': {exc}."
                ) from exc
        else:
            raise PhaseDebugError(
                f"{name}: 'selected_support' must decode from a mapping or "
                f"null, got {type(raw_selected).__name__}."
            )
        raw_candidates = values["candidates"]
        if not isinstance(raw_candidates, list):
            raise PhaseDebugError(
                f"{name}: 'candidates' must be a JSON list."
            )
        try:
            candidates = tuple(
                CandidateSummary.from_dict(entry) for entry in raw_candidates
            )
        except PhaseDebugError:
            raise
        except Exception as exc:
            raise PhaseDebugError(
                f"{name}: invalid candidate: {exc}."
            ) from exc
        return cls(
            stage=values["stage"],
            manual_time_seconds=values["manual_time_seconds"],
            selected_time_seconds=values["selected_time_seconds"],
            manual_support=manual_support,
            selected_support=selected_support,
            candidates=candidates,
            manual_score=values["manual_score"],
            manual_rank=values["manual_rank"],
            selected_score=values["selected_score"],
            selected_rank=values["selected_rank"],
            objective_contribution=values["objective_contribution"],
            unavailable_reason=values["unavailable_reason"],
            anomalies=tuple(values["anomalies"]),
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> StageDebug:
        return cls.from_dict(_loads_object("stage_debug", data))


#: Backwards/forwards-compatible alias for :class:`StageDebug`.
StagePhaseDebug = StageDebug


# --- One-attempt artifact ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhaseDebugArtifact:
    """Immutable versioned debug artifact for exactly one attempt.

    ``stages`` and ``traces`` each contain exactly the eight canonical
    stage keys; ``stages`` values are :class:`StageDebug` records and
    ``traces`` values are :class:`StageTrace` records. ``total_objective``
    is the reserved artifact-level objective field (null when absent).
    """

    attempt_id: str = "serve-001"
    attempt_range: MediaRange | None = None  # type: ignore[assignment]
    method_version: str = PHASE_DEBUG_METHOD_VERSION
    config_id: str = PHASE_DEBUG_DEFAULT_CONFIG_ID
    stages: Mapping[str, StageDebug] = field(
        default_factory=lambda: MappingProxyType({})
    )
    traces: Mapping[str, StageTrace] = field(
        default_factory=lambda: MappingProxyType({})
    )
    anomalies: tuple[str, ...] = ()
    total_objective: float | None = None
    schema_version: int = PHASE_DEBUG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_debug"
        if not _is_int(self.schema_version):
            raise PhaseDebugError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_DEBUG_SCHEMA_VERSION:
            raise PhaseDebugError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_DEBUG_SCHEMA_VERSION}."
            )
        object.__setattr__(
            self, "attempt_id", _check_attempt_id(name, self.attempt_id)
        )
        if not isinstance(self.attempt_range, MediaRange):
            raise PhaseDebugError(
                f"{name}: 'attempt_range' must be a MediaRange, "
                f"got {type(self.attempt_range).__name__}."
            )
        object.__setattr__(
            self, "method_version",
            _check_non_blank(name, "'method_version'", self.method_version),
        )
        object.__setattr__(
            self, "config_id",
            _check_non_blank(name, "'config_id'", self.config_id),
        )
        raw_stages = self.stages
        if not isinstance(raw_stages, Mapping):
            raise PhaseDebugError(
                f"{name}: 'stages' must be a mapping of stage key to "
                f"StageDebug, got {type(raw_stages).__name__}."
            )
        if set(raw_stages.keys()) != set(_CANONICAL_STAGES):
            raise PhaseDebugError(
                f"{name}: 'stages' must contain exactly the eight stage keys "
                f"{list(_CANONICAL_STAGES)}, got {sorted(raw_stages.keys())!r}."
            )
        for key in _CANONICAL_STAGES:
            entry = raw_stages[key]
            if not isinstance(entry, StageDebug):
                raise PhaseDebugError(
                    f"{name}: stage {key!r} must be a StageDebug, "
                    f"got {type(entry).__name__}."
                )
            if entry.stage != key:
                raise PhaseDebugError(
                    f"{name}: stage key {key!r} must match record stage "
                    f"{entry.stage!r}."
                )
        object.__setattr__(
            self,
            "stages",
            MappingProxyType({key: raw_stages[key] for key in _CANONICAL_STAGES}),
        )
        raw_traces = self.traces
        if not isinstance(raw_traces, Mapping):
            raise PhaseDebugError(
                f"{name}: 'traces' must be a mapping of stage key to "
                f"StageTrace, got {type(raw_traces).__name__}."
            )
        if set(raw_traces.keys()) != set(_CANONICAL_STAGES):
            raise PhaseDebugError(
                f"{name}: 'traces' must contain exactly the eight stage keys "
                f"{list(_CANONICAL_STAGES)}, got {sorted(raw_traces.keys())!r}."
            )
        for key in _CANONICAL_STAGES:
            entry = raw_traces[key]
            if not isinstance(entry, StageTrace):
                raise PhaseDebugError(
                    f"{name}: trace {key!r} must be a StageTrace, "
                    f"got {type(entry).__name__}."
                )
            if entry.stage != key:
                raise PhaseDebugError(
                    f"{name}: trace key {key!r} must match record stage "
                    f"{entry.stage!r}."
                )
        object.__setattr__(
            self,
            "traces",
            MappingProxyType({key: raw_traces[key] for key in _CANONICAL_STAGES}),
        )
        object.__setattr__(
            self, "anomalies", _check_tokens(name, "'anomalies'", self.anomalies)
        )
        object.__setattr__(
            self, "total_objective",
            _check_optional_finite(name, "'total_objective'", self.total_objective),
        )
        stages = self.stages
        assert isinstance(stages, Mapping)
        attempt_range = self.attempt_range
        assert isinstance(attempt_range, MediaRange)
        for key in _CANONICAL_STAGES:
            record = stages[key]
            assert isinstance(record, StageDebug)
            for label in ("manual_time_seconds", "selected_time_seconds"):
                moment = getattr(record, label)
                if moment is not None and not attempt_range.contains(float(moment)):
                    if not float(moment) == attempt_range.start_seconds:
                        raise PhaseDebugError(
                            f"{name}: stage {key!r} {label} ({moment!r}) must "
                            f"lie in attempt range "
                            f"{attempt_range.to_dict()!r}."
                        )
            for support_label in ("manual_support", "selected_support"):
                snapshot = getattr(record, support_label)
                if snapshot is not None and snapshot.support_time_seconds is not None:
                    moment = float(snapshot.support_time_seconds)
                    if not attempt_range.contains(moment):
                        if not moment == attempt_range.start_seconds:
                            raise PhaseDebugError(
                                f"{name}: stage {key!r} {support_label} support "
                                f"time ({moment!r}) must lie in attempt range "
                                f"{attempt_range.to_dict()!r}."
                            )
            for candidate in record.candidates:
                if candidate.interval is not None:
                    interval = candidate.interval
                    assert isinstance(interval, MediaRange)
                    if (
                        interval.start_seconds < attempt_range.start_seconds
                        or interval.end_seconds > attempt_range.end_seconds
                    ):
                        raise PhaseDebugError(
                            f"{name}: stage {key!r} candidate interval "
                            f"{interval.to_dict()!r} must lie inside attempt "
                            f"range {attempt_range.to_dict()!r}."
                        )
        # Chronological order across canonical stages for present values.
        for label in ("manual_time_seconds", "selected_time_seconds"):
            previous: float | None = None
            for key in _CANONICAL_STAGES:
                moment = getattr(stages[key], label)
                if moment is None:
                    continue
                if previous is not None and not float(moment) > float(previous):
                    raise PhaseDebugError(
                        f"{name}: {label} values must be strictly increasing "
                        f"in canonical stage order."
                    )
                previous = float(moment)

    def to_dict(self) -> dict[str, Any]:
        stages = self.stages
        traces = self.traces
        assert isinstance(stages, Mapping)
        assert isinstance(traces, Mapping)
        attempt_range = self.attempt_range
        assert isinstance(attempt_range, MediaRange)
        return {
            "anomalies": list(self.anomalies),
            "attempt_id": self.attempt_id,
            "attempt_range": attempt_range.to_dict(),
            "config_id": self.config_id,
            "method_version": self.method_version,
            "schema_version": self.schema_version,
            "stages": {
                key: stages[key].to_dict() for key in _CANONICAL_STAGES
            },
            "total_objective": self.total_objective,
            "traces": {key: traces[key].to_dict() for key in _CANONICAL_STAGES},
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseDebugArtifact:
        name = "phase_debug"
        if not isinstance(values, dict):
            raise PhaseDebugError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "anomalies",
            "attempt_id",
            "attempt_range",
            "config_id",
            "method_version",
            "schema_version",
            "stages",
            "total_objective",
            "traces",
        }
        _require_keys(name, values, known)
        raw_range = values["attempt_range"]
        if not isinstance(raw_range, dict):
            raise PhaseDebugError(
                f"{name}: 'attempt_range' must decode from a mapping, "
                f"got {type(raw_range).__name__}."
            )
        try:
            attempt_range = MediaRange.from_dict(raw_range)
        except Exception as exc:
            raise PhaseDebugError(
                f"{name}: invalid 'attempt_range': {exc}."
            ) from exc
        raw_stages = values["stages"]
        if not isinstance(raw_stages, dict):
            raise PhaseDebugError(
                f"{name}: 'stages' must decode from a mapping, "
                f"got {type(raw_stages).__name__}."
            )
        try:
            stages = {
                key: StageDebug.from_dict(raw_stages[key])
                for key in list(raw_stages.keys())
            }
        except PhaseDebugError:
            raise
        except Exception as exc:
            raise PhaseDebugError(
                f"{name}: invalid stage: {exc}."
            ) from exc
        raw_traces = values["traces"]
        if not isinstance(raw_traces, dict):
            raise PhaseDebugError(
                f"{name}: 'traces' must decode from a mapping, "
                f"got {type(raw_traces).__name__}."
            )
        try:
            traces = {
                key: StageTrace.from_dict(raw_traces[key])
                for key in list(raw_traces.keys())
            }
        except PhaseDebugError:
            raise
        except Exception as exc:
            raise PhaseDebugError(
                f"{name}: invalid trace: {exc}."
            ) from exc
        return cls(
            attempt_id=values["attempt_id"],
            attempt_range=attempt_range,
            method_version=values["method_version"],
            config_id=values["config_id"],
            stages=stages,  # type: ignore[arg-type]
            traces=traces,  # type: ignore[arg-type]
            anomalies=tuple(values["anomalies"]),
            total_objective=values["total_objective"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PhaseDebugArtifact:
        return cls.from_dict(_loads_object("phase_debug", data))


#: Alias for :class:`PhaseDebugArtifact`.
PhaseDebug = PhaseDebugArtifact
