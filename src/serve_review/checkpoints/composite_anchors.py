"""Pure composite six-anchor candidate generation (M4.10a).

Pure, deterministic, in-memory generation of dense composite anchor
candidates from one M4.9
:class:`~serve_review.checkpoints.kinematic_waveforms.KinematicWaveformTrack`.

Non-goals (never done here):

- No phase-solver/DP, no chronology enforcement, no stage selection,
  no keyframe picking, no non-maximum suppression. Every waveform
  sample yields exactly one candidate per anchor stage (dense grid);
  downstream M4.10b maps these dense candidates into solver evidence.
- No sparse M4.2 phase features, no CLI, no media decoding, no audio
  extraction, no pose cache I/O, no private footage inspection.
- No weight tuning: :class:`CompositeAnchorConfig` pins one set of
  initial generic weights verbatim (documented as untuned).

Scoring contract:

- Only availability-qualified local temporal waveform evidence is
  scored. A cue contributes at a frame only when every waveform
  channel it requires is ``available`` (non-``None``) at every stencil
  member it touches; otherwise the cue value is honestly ``None``
  (missing), never zero-filled, never bridged.
- Within-attempt robust quantile normalization maps each cue's finite
  raw series to ``[0, 1]`` via the 5th/95th percentiles of the
  available raw values in this attempt. Explicit finite guards: fewer
  than two finite raw values, non-finite quantiles, or a quantile span
  ``<= 1e-9`` yields ``0.0`` for every available raw value (no
  discriminative evidence) rather than amplifying numerical noise.
  Exception: the contact ``audio_transient`` cue never normalizes an
  explicit binary flag (a rare ``[0, ..., 1, ..., 0]`` spike would
  otherwise collapse to all zeros); explicit ``0.0``/``1.0`` flags pass
  through verbatim and only flag-missing frames fall back to the
  quantile-normalized continuous energy.
- Coverage is the mean available cue weight
  (``sum(available weights) / sum(all weights)``) in ``[0, 1]``.
- The unary score is the availability-weighted mean of the available
  normalized cue values
  (``sum(w_i * n_i) / sum(available w_i)``); zero available cues give
  score ``0.0`` with coverage ``0.0``. Contradictory evidence (low
  normalized cue values) honestly lowers the score; missing cues are
  excluded from the numerator and denominator deterministically.
- Candidate records hold the anchor stage, the exact waveform PTS
  (``time_seconds``/``timestamp_ms`` verbatim), the unary score,
  coverage, per-cue normalized values (``None`` when honestly
  missing), provenance, and deterministic JSON codecs. Ordering is
  deterministic: canonical stage order, then strictly increasing PTS.

Cue inventory (all local to the waveform grid; larger raw means more
evidence after the documented orientation):

- start: ``left_arm_elevation_rise`` (local PTS slope of
  ``left_arm_elevation``), ``left_elbow_extension``
  (``180 - elbow_flexion_left``), ``stillness``
  (``-whole_body_settling_energy``), ``left_arm_low``
  (``-left_arm_elevation``).
- release: ``left_arm_elevation``, ``left_arm_elevation_rise``,
  ``left_arm_extension``, ``preparation``
  (``-whole_body_settling_energy`` calm-preparation proxy).
- loading: ``knee_flexion`` (mean of available knee flexions),
  ``shoulder_hip_separation``
  (``shoulder_hip_separation_transverse_deg``), ``tilt`` (mean of
  available ``|shoulder_tilt_deg|``/``|hip_tilt_deg|``),
  ``left_arm`` (``left_arm_elevation``), ``right_elbow_flexion``
  (``elbow_flexion_right``), ``stillness``.
- cocking: ``right_wrist_elevation_trough``
  (``-right_wrist_rel_shoulder_dy`` with ``+y``-down upward-positive dy), ``right_wrist_acceleration``,
  ``torso_rise``, ``knee_unload`` (``-mean knee flexion``),
  ``loading_unwind`` (``-shoulder_hip_separation_transverse_deg``).
- contact: ``right_wrist_elevation_apex``
  (``right_wrist_rel_shoulder_dy``, upward-positive with ``+y`` down),
  ``right_wrist_speed_peak`` (``right_wrist_speed_peak`` flag: strict local
  speed maximum only; trough valleys never score), ``right_wrist_acceleration_peak``
  (``right_wrist_accel_peak`` flag: strict local acceleration maximum only),
  ``torso_rise``, ``right_arm_extension`` (directional
  ``right_wrist_rel_shoulder_distance`` gated by positive
  ``right_wrist_rel_shoulder_dy``: raw is ``distance`` when the wrist is
  above the shoulder and ``0.0`` when at/below, so a straight arm hanging
  down contributes zero; missing support stays ``None``),
  ``audio_transient`` (explicit binary ``audio_transient_flag`` preserved
  directly as ``0``/``1`` per frame, never p5/p95-normalized; continuous
  ``audio_transient_energy`` is quantile-normalized only on frames without
  an explicit flag; honestly ``None`` without audio, never zero-filled).
- finish: ``whole_body_settling``
  (``-whole_body_settling_energy``), ``right_wrist_settling``
  (``-right_wrist_speed``), ``torso_settling``
  (``-right_wrist_acceleration`` calm proxy; the M4.9 matrix carries no
  torso-velocity channel, so wrist-acceleration calm is the documented
  honest proxy).

Dense preservation: one candidate per waveform sample per stage, in
PTS order. No local-extrema pruning is applied; the dense grid is the
documented preservation policy, so no frame evidence is discarded
before M4.10b.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from serve_review.checkpoints.kinematic_waveforms import (
    CHANNEL_INDEX,
    KinematicWaveformTrack,
)

__all__ = [
    "COMPOSITE_ANCHORS_SCHEMA_VERSION",
    "COMPOSITE_ANCHORS_METHOD_VERSION",
    "COMPOSITE_ANCHORS_DEFAULT_CONFIG_ID",
    "COMPOSITE_ANCHOR_STAGES",
    "COMPOSITE_CUE_NAMES",
    "COMPOSITE_DEFAULT_WEIGHTS",
    "COMPOSITE_ANCHOR_PROVENANCE",
    "QUANTILE_LOW",
    "QUANTILE_HIGH",
    "QUANTILE_MIN_SPAN",
    "CompositeAnchorsError",
    "CompositeAnchorConfig",
    "CompositeAnchorCandidate",
    "CompositeAnchorSet",
    "quantile_normalize_series",
    "compute_composite_anchor_scores",
    "build_composite_anchor_set",
    "generate_composite_anchor_candidates",
]

#: Version of the composite-anchor schemas in this module.
COMPOSITE_ANCHORS_SCHEMA_VERSION = 2
#: Method identity recorded on every candidate and set.
COMPOSITE_ANCHORS_METHOD_VERSION = "composite-anchors-v2"
#: Default configuration identity (untuned generic weights).
COMPOSITE_ANCHORS_DEFAULT_CONFIG_ID = "composite-anchors-default-v2"
#: Provenance recorded on every candidate (pure waveform evidence only).
COMPOSITE_ANCHOR_PROVENANCE = "kinematic_waveform"

#: Canonical six-anchor order (subset of Kovacs keys; no reordering).
COMPOSITE_ANCHOR_STAGES: tuple[str, ...] = (
    "start",
    "release",
    "loading",
    "cocking",
    "contact",
    "finish",
)

#: Cue names per anchor stage (ordered deterministically).
COMPOSITE_CUE_NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "start": (
            "left_arm_elevation_rise",
            "left_elbow_extension",
            "stillness",
            "left_arm_low",
        ),
        "release": (
            "left_arm_elevation",
            "left_arm_elevation_rise",
            "left_arm_extension",
            "preparation",
        ),
        "loading": (
            "knee_flexion",
            "shoulder_hip_separation",
            "tilt",
            "left_arm",
            "right_elbow_flexion",
            "stillness",
        ),
        "cocking": (
            "right_wrist_elevation_trough",
            "right_wrist_acceleration",
            "torso_rise",
            "knee_unload",
            "loading_unwind",
        ),
        "contact": (
            "right_wrist_elevation_apex",
            "right_wrist_speed_peak",
            "right_wrist_acceleration_peak",
            "torso_rise",
            "right_arm_extension",
            "audio_transient",
        ),
        "finish": (
            "whole_body_settling",
            "right_wrist_settling",
            "torso_settling",
        ),
    }
)

#: Exact initial generic weights (untuned; each stage sums to 1.0).
COMPOSITE_DEFAULT_WEIGHTS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "start": MappingProxyType(
            {
                "left_arm_elevation_rise": 0.35,
                "left_elbow_extension": 0.25,
                "stillness": 0.25,
                "left_arm_low": 0.15,
            }
        ),
        "release": MappingProxyType(
            {
                "left_arm_elevation": 0.30,
                "left_arm_elevation_rise": 0.30,
                "left_arm_extension": 0.25,
                "preparation": 0.15,
            }
        ),
        "loading": MappingProxyType(
            {
                "knee_flexion": 0.20,
                "shoulder_hip_separation": 0.20,
                "tilt": 0.15,
                "left_arm": 0.20,
                "right_elbow_flexion": 0.15,
                "stillness": 0.10,
            }
        ),
        "cocking": MappingProxyType(
            {
                "right_wrist_elevation_trough": 0.25,
                "right_wrist_acceleration": 0.20,
                "torso_rise": 0.15,
                "knee_unload": 0.15,
                "loading_unwind": 0.25,
            }
        ),
        "contact": MappingProxyType(
            {
                "right_wrist_elevation_apex": 0.25,
                "right_wrist_speed_peak": 0.20,
                "right_wrist_acceleration_peak": 0.15,
                "torso_rise": 0.10,
                "right_arm_extension": 0.10,
                "audio_transient": 0.20,
            }
        ),
        "finish": MappingProxyType(
            {
                "whole_body_settling": 0.45,
                "right_wrist_settling": 0.35,
                "torso_settling": 0.20,
            }
        ),
    }
)

#: Robust normalization quantiles (within-attempt, deterministic).
QUANTILE_LOW = 0.05
QUANTILE_HIGH = 0.95
#: Minimum quantile span carrying discriminative evidence.
QUANTILE_MIN_SPAN = 1e-9


class CompositeAnchorsError(ValueError):
    """Raised when composite-anchor input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _dumps_deterministic(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _loads_object(name: str, data: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CompositeAnchorsError(f"{name}: invalid UTF-8 JSON payload.") from exc
    if not isinstance(data, str):
        raise CompositeAnchorsError(
            f"{name}: JSON payload must be str or bytes, got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CompositeAnchorsError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise CompositeAnchorsError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


def _check_schema_version(name: str, values: Mapping[str, Any]) -> None:
    if "schema_version" not in values:
        raise CompositeAnchorsError(f"{name}: missing required key 'schema_version'.")
    version = values["schema_version"]
    if not _is_int(version):
        raise CompositeAnchorsError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != COMPOSITE_ANCHORS_SCHEMA_VERSION:
        raise CompositeAnchorsError(
            f"{name}: unsupported schema_version {version!r}; expected "
            f"{COMPOSITE_ANCHORS_SCHEMA_VERSION}."
        )


def _check_weights(name: str, stage: str, mapping: Any) -> Mapping[str, float]:
    expected_cues = COMPOSITE_CUE_NAMES[stage]
    if not isinstance(mapping, Mapping):
        raise CompositeAnchorsError(
            f"{name}: weights for stage {stage!r} must decode from a mapping, "
            f"got {type(mapping).__name__}."
        )
    if tuple(sorted(mapping.keys())) != tuple(sorted(expected_cues)):
        raise CompositeAnchorsError(
            f"{name}: weights for stage {stage!r} must contain exactly cues "
            f"{list(expected_cues)!r}, got {sorted(mapping.keys())!r}."
        )
    cleaned: dict[str, float] = {}
    for cue in expected_cues:
        value = mapping[cue]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompositeAnchorsError(
                f"{name}: weight {stage}.{cue} must be a number in [0, 1], "
                f"got {value!r}."
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise CompositeAnchorsError(
                f"{name}: weight {stage}.{cue} must lie in [0, 1], got {value!r}."
            )
        cleaned[cue] = number
    total = sum(cleaned.values())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
        raise CompositeAnchorsError(
            f"{name}: weights for stage {stage!r} must sum to 1.0, got {total!r}."
        )
    return MappingProxyType(cleaned)


# --- Configuration -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompositeAnchorConfig:
    """Immutable versioned configuration pinning the untuned weights.

    Each stage mapping carries exactly the documented cue keys with
    weights in ``[0, 1]`` summing to ``1.0``. The defaults equal
    :data:`COMPOSITE_DEFAULT_WEIGHTS` verbatim and are recorded as
    initial generic (untuned) values.
    """

    config_id: str = COMPOSITE_ANCHORS_DEFAULT_CONFIG_ID
    start: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPOSITE_DEFAULT_WEIGHTS["start"])
    )
    release: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPOSITE_DEFAULT_WEIGHTS["release"])
    )
    loading: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPOSITE_DEFAULT_WEIGHTS["loading"])
    )
    cocking: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPOSITE_DEFAULT_WEIGHTS["cocking"])
    )
    contact: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPOSITE_DEFAULT_WEIGHTS["contact"])
    )
    finish: Mapping[str, float] = field(
        default_factory=lambda: dict(COMPOSITE_DEFAULT_WEIGHTS["finish"])
    )
    method_version: str = COMPOSITE_ANCHORS_METHOD_VERSION
    schema_version: int = COMPOSITE_ANCHORS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "composite_anchor_config"
        if not _is_int(self.schema_version):
            raise CompositeAnchorsError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != COMPOSITE_ANCHORS_SCHEMA_VERSION:
            raise CompositeAnchorsError(
                f"{name}: unsupported schema_version {self.schema_version!r}; expected "
                f"{COMPOSITE_ANCHORS_SCHEMA_VERSION}."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise CompositeAnchorsError(
                f"{name}: 'config_id' must be a non-blank string, got {self.config_id!r}."
            )
        if not isinstance(self.method_version, str) or self.method_version != (
            COMPOSITE_ANCHORS_METHOD_VERSION
        ):
            raise CompositeAnchorsError(
                f"{name}: 'method_version' must equal "
                f"{COMPOSITE_ANCHORS_METHOD_VERSION!r}, got {self.method_version!r}."
            )
        for stage in COMPOSITE_ANCHOR_STAGES:
            raw = getattr(self, stage)
            # Accept plain dicts on construction; freeze deterministically.
            as_dict = dict(raw) if isinstance(raw, Mapping) else raw
            cleaned = _check_weights(name, stage, as_dict)
            object.__setattr__(self, stage, cleaned)

    def weights(self) -> Mapping[str, Mapping[str, float]]:
        """Return the per-stage weight mappings in canonical stage order."""
        return MappingProxyType(
            {stage: getattr(self, stage) for stage in COMPOSITE_ANCHOR_STAGES}
        )

    def weight_for(self, stage: str) -> Mapping[str, float]:
        """Return the weight mapping for ``stage``."""
        if stage not in COMPOSITE_ANCHOR_STAGES:
            raise CompositeAnchorsError(
                f"composite_anchor_config: unknown stage {stage!r}; expected one of "
                f"{list(COMPOSITE_ANCHOR_STAGES)!r}."
            )
        value = getattr(self, stage)
        assert isinstance(value, Mapping)
        return value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "config_id": self.config_id,
            "method_version": self.method_version,
            "schema_version": self.schema_version,
            "weights": {
                stage: dict(getattr(self, stage)) for stage in COMPOSITE_ANCHOR_STAGES
            },
        }
        # Mirror per-stage keys top-level for direct readability; both
        # spellings carry the same exact weights.
        for stage in COMPOSITE_ANCHOR_STAGES:
            payload[stage] = dict(getattr(self, stage))
        return payload

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CompositeAnchorConfig:
        name = "composite_anchor_config"
        if not isinstance(values, dict):
            raise CompositeAnchorsError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "config_id",
            "method_version",
            "schema_version",
            "weights",
            *COMPOSITE_ANCHOR_STAGES,
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise CompositeAnchorsError(f"{name}: unknown keys {unknown!r}.")
        for required in ("config_id", "method_version", "schema_version"):
            if required not in values:
                raise CompositeAnchorsError(
                    f"{name}: missing required key {required!r}."
                )
        _check_schema_version(name, values)
        # Prefer the nested "weights" spelling when present; otherwise
        # read the mirrored per-stage top-level keys.
        if "weights" in values:
            raw_weights = values["weights"]
            if not isinstance(raw_weights, dict):
                raise CompositeAnchorsError(
                    f"{name}: 'weights' must decode from a mapping."
                )
            if sorted(raw_weights.keys()) != sorted(COMPOSITE_ANCHOR_STAGES):
                raise CompositeAnchorsError(
                    f"{name}: 'weights' must contain exactly stages "
                    f"{list(COMPOSITE_ANCHOR_STAGES)!r}."
                )
            staged = {stage: raw_weights[stage] for stage in COMPOSITE_ANCHOR_STAGES}
        else:
            missing = [s for s in COMPOSITE_ANCHOR_STAGES if s not in values]
            if missing:
                raise CompositeAnchorsError(
                    f"{name}: missing required keys {missing!r} (or 'weights')."
                )
            staged = {stage: values[stage] for stage in COMPOSITE_ANCHOR_STAGES}
        # When both spellings are present they must agree exactly.
        for stage in COMPOSITE_ANCHOR_STAGES:
            if "weights" in values and stage in values:
                if dict(values[stage]) != dict(values["weights"][stage]):
                    raise CompositeAnchorsError(
                        f"{name}: mirrored weights for stage {stage!r} disagree."
                    )
        try:
            return cls(
                config_id=values["config_id"],
                method_version=values["method_version"],
                schema_version=values["schema_version"],
                **staged,  # type: ignore[arg-type]
            )
        except CompositeAnchorsError:
            raise
        except Exception as exc:
            raise CompositeAnchorsError(
                f"{name}: invalid configuration: {exc}."
            ) from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> CompositeAnchorConfig:
        return cls.from_dict(_loads_object("composite_anchor_config", data))


# --- Quantile normalization --------------------------------------------------


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Deterministic linear-interpolation quantile over sorted values."""
    count = len(sorted_values)
    assert count >= 1
    position = probability * (count - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def quantile_normalize_series(
    raw: Sequence[float | None],
) -> tuple[float | None, ...]:
    """Robust within-attempt quantile normalization with finite guards.

    Finite raw values map through the attempt's 5th/95th percentiles to
    ``[0, 1]`` (clipped); ``None``/non-finite raw stays ``None``
    (honestly missing). Guards: fewer than two finite values,
    non-finite quantiles, or a quantile span ``<= 1e-9`` yields ``0.0``
    for every available raw value (no discriminative evidence) instead
    of amplifying noise. Deterministic in the input order and values.
    """
    normalized: list[float | None] = []
    finite = sorted(
        float(v) for v in raw if v is not None and math.isfinite(float(v))
    )
    if len(finite) < 2:
        for value in raw:
            if value is None or not math.isfinite(float(value)):
                normalized.append(None)
            else:
                normalized.append(0.0)
        return tuple(normalized)
    low = _quantile(finite, QUANTILE_LOW)
    high = _quantile(finite, QUANTILE_HIGH)
    if not (math.isfinite(low) and math.isfinite(high)):
        for value in raw:
            if value is None or not math.isfinite(float(value)):
                normalized.append(None)
            else:
                normalized.append(0.0)
        return tuple(normalized)
    span = high - low
    if not math.isfinite(span) or span <= QUANTILE_MIN_SPAN:
        for value in raw:
            if value is None or not math.isfinite(float(value)):
                normalized.append(None)
            else:
                normalized.append(0.0)
        return tuple(normalized)
    for value in raw:
        if value is None or not math.isfinite(float(value)):
            normalized.append(None)
            continue
        scaled = (float(value) - low) / span
        normalized.append(max(0.0, min(1.0, scaled)))
    return tuple(normalized)


# --- Raw cue extraction (availability-qualified, local) ----------------------


def _series(track: KinematicWaveformTrack, channel: str) -> tuple[float | None, ...]:
    return track.channel_series(channel)


def _local_rise(
    values: Sequence[float | None], times: Sequence[float]
) -> list[float | None]:
    """Local PTS slope with qualified stencil only (centered/one-sided).

    Interior frames use ``(v[i+1]-v[i-1])/(t[i+1]-t[i-1])``; edge frames
    use the single qualified neighbor difference; isolated frames and
    non-positive/non-finite steps yield ``None``. Deterministic; never
    bridges missing support.
    """
    count = len(values)
    out: list[float | None] = [None] * count
    for i in range(count):
        center = values[i]
        if center is None or not math.isfinite(float(center)):
            continue
        if 0 < i < count - 1:
            prev = values[i - 1]
            nxt = values[i + 1]
            if (
                prev is not None
                and nxt is not None
                and math.isfinite(float(prev))
                and math.isfinite(float(nxt))
            ):
                denom = float(times[i + 1]) - float(times[i - 1])
                if math.isfinite(denom) and denom > 0.0:
                    rate = (float(nxt) - float(prev)) / denom
                    out[i] = rate if math.isfinite(rate) else None
            continue
        if i == 0 and count > 1:
            nxt = values[1]
            if nxt is not None and math.isfinite(float(nxt)):
                denom = float(times[1]) - float(times[0])
                if math.isfinite(denom) and denom > 0.0:
                    rate = (float(nxt) - float(center)) / denom
                    out[i] = rate if math.isfinite(rate) else None
        elif i == count - 1 and count > 1:
            prev = values[i - 1]
            if prev is not None and math.isfinite(float(prev)):
                denom = float(times[i]) - float(times[i - 1])
                if math.isfinite(denom) and denom > 0.0:
                    rate = (float(center) - float(prev)) / denom
                    out[i] = rate if math.isfinite(rate) else None
    return out


def _negate(values: Sequence[float | None]) -> list[float | None]:
    out: list[float | None] = []
    for value in values:
        if value is None or not math.isfinite(float(value)):
            out.append(None)
        else:
            negated = -float(value)
            out.append(negated if math.isfinite(negated) else None)
    return out


def _mean_pair(
    first: Sequence[float | None], second: Sequence[float | None]
) -> list[float | None]:
    out: list[float | None] = []
    for a, b in zip(first, second):
        pair = [
            float(v)
            for v in (a, b)
            if v is not None and math.isfinite(float(v))
        ]
        if not pair:
            out.append(None)
        else:
            mean = sum(pair) / len(pair)
            out.append(mean if math.isfinite(mean) else None)
    return out


def _mean_abs_tilt(
    shoulder: Sequence[float | None], hip: Sequence[float | None]
) -> list[float | None]:
    out: list[float | None] = []
    for a, b in zip(shoulder, hip):
        vals = [
            abs(float(v))
            for v in (a, b)
            if v is not None and math.isfinite(float(v))
        ]
        if not vals:
            out.append(None)
        else:
            mean = sum(vals) / len(vals)
            out.append(mean if math.isfinite(mean) else None)
    return out


def _extension_from_flexion(
    flexion: Sequence[float | None],
) -> list[float | None]:
    out: list[float | None] = []
    for value in flexion:
        if value is None or not math.isfinite(float(value)):
            out.append(None)
        else:
            ext = 180.0 - float(value)
            out.append(ext if math.isfinite(ext) else None)
    return out


def _optional_series(track: KinematicWaveformTrack, channel: str) -> list[float | None]:
    """Return a channel series, or all-``None`` when the track predates it.

    Legacy 36-channel tracks carry no audio channels; the audio contact cue
    is then honestly unavailable at every frame instead of raising.
    """
    try:
        return list(track.channel_series(channel))
    except Exception:
        return [None] * len(track.samples)


def _audio_transient_cue(
    flags: Sequence[float | None], energies: Sequence[float | None]
) -> list[float | None]:
    """Derive the contact audio cue: explicit flag when available, else energy.

    Per-frame preference is the binary ``audio_transient_flag`` (``1.0`` =
    transient, ``0.0`` = no transient); frames where the flag is missing but
    the continuous ``audio_transient_energy`` is available fall back to the
    energy value. Frames with neither are honestly ``None`` (unavailable
    without audio, never zero-filled). Non-finite entries are treated as
    missing.
    """
    out: list[float | None] = []
    for flag, energy in zip(flags, energies):
        if flag is not None and math.isfinite(float(flag)):
            out.append(float(flag))
        elif energy is not None and math.isfinite(float(energy)):
            out.append(float(energy))
        else:
            out.append(None)
    return out


def _directional_arm_extension(
    wrist_dy: Sequence[float | None], wrist_dist: Sequence[float | None]
) -> list[float | None]:
    """Directional right-arm extension: reach gated by upward elevation.

    Raw support is ``distance`` only when the corrected upward-positive
    ``right_wrist_rel_shoulder_dy`` is strictly ``> 0`` (wrist above the
    shoulder with ``+y`` down); when the wrist is at or below the shoulder
    the raw is ``0.0`` so a straight arm hanging down contributes zero
    after normalization; when either input is missing/non-finite the raw
    is honestly ``None``.
    """
    out: list[float | None] = []
    for dy, dist in zip(wrist_dy, wrist_dist):
        if (
            dy is None
            or dist is None
            or not math.isfinite(float(dy))
            or not math.isfinite(float(dist))
        ):
            out.append(None)
        elif float(dy) > 0.0:
            out.append(float(dist))
        else:
            out.append(0.0)
    return out


def _normalize_audio_transient_cue(
    flags: Sequence[float | None], energies: Sequence[float | None]
) -> tuple[float | None, ...]:
    """Normalize the contact audio cue without destroying rare binary flags.

    Explicit binary flags (``0.0``/``1.0``) pass through verbatim per frame,
    so a rare ``[0, ..., 1, ..., 0]`` spike retains ``1`` instead of being
    p5/p95-collapsed to all zeros. Frames without an explicit flag fall back
    to the quantile-normalized continuous energy (``None`` where neither
    exists). Non-finite entries count as missing.
    """
    clean_flags: list[float | None] = []
    for flag in flags:
        if flag is not None and math.isfinite(float(flag)):
            clean_flags.append(float(flag))
        else:
            clean_flags.append(None)
    normalized_energy = quantile_normalize_series(
        [
            (float(e) if (e is not None and math.isfinite(float(e))) else None)
            for e in energies
        ]
    )
    out: list[float | None] = []
    for flag, norm_e, energy in zip(clean_flags, normalized_energy, energies):
        if flag is not None:
            out.append(flag)
        elif norm_e is not None:
            out.append(norm_e)
        elif energy is not None and math.isfinite(float(energy)):
            out.append(norm_e)
        else:
            out.append(None)
    return tuple(out)


def _extract_raw_cues(
    track: KinematicWaveformTrack,
) -> dict[str, dict[str, list[float | None]]]:
    """Extract availability-qualified raw cue series per stage."""
    count = len(track.samples)
    times = [float(s.time_seconds) for s in track.samples]
    left_elev = list(_series(track, "left_arm_elevation"))
    left_ext = list(_series(track, "left_arm_extension"))
    elbow_left = list(_series(track, "elbow_flexion_left"))
    elbow_right = list(_series(track, "elbow_flexion_right"))
    knee_left = list(_series(track, "knee_flexion_left"))
    knee_right = list(_series(track, "knee_flexion_right"))
    separation = list(_series(track, "shoulder_hip_separation_transverse_deg"))
    shoulder_tilt = list(_series(track, "shoulder_tilt_deg"))
    hip_tilt = list(_series(track, "hip_tilt_deg"))
    torso = list(_series(track, "torso_rise"))
    wrist_dy = list(_series(track, "right_wrist_rel_shoulder_dy"))
    wrist_dist = list(_series(track, "right_wrist_rel_shoulder_distance"))
    wrist_acc = list(_series(track, "right_wrist_acceleration"))
    wrist_speed = list(_series(track, "right_wrist_speed"))
    speed_peak = list(_series(track, "right_wrist_speed_peak"))
    accel_peak = list(_series(track, "right_wrist_accel_peak"))
    settling = list(_series(track, "whole_body_settling_energy"))
    audio_flags = _optional_series(track, "audio_transient_flag")
    audio_energies = _optional_series(track, "audio_transient_energy")
    audio_cue = _audio_transient_cue(audio_flags, audio_energies)

    left_elev_rise = _local_rise(left_elev, times)
    left_elbow_ext = _extension_from_flexion(elbow_left)
    stillness_raw = _negate(settling)
    left_low = _negate(left_elev)
    knee_mean = _mean_pair(knee_left, knee_right)
    knee_unload = _negate(knee_mean)
    tilt_mean = _mean_abs_tilt(shoulder_tilt, hip_tilt)
    trough = _negate(wrist_dy)
    unwind = _negate(separation)
    wrist_settle = _negate(wrist_speed)
    torso_settle = _negate(wrist_acc)

    raw: dict[str, dict[str, list[float | None]]] = {
        "start": {
            "left_arm_elevation_rise": left_elev_rise,
            "left_elbow_extension": left_elbow_ext,
            "stillness": list(stillness_raw),
            "left_arm_low": list(left_low),
        },
        "release": {
            "left_arm_elevation": list(left_elev),
            "left_arm_elevation_rise": list(left_elev_rise),
            "left_arm_extension": list(left_ext),
            "preparation": list(stillness_raw),
        },
        "loading": {
            "knee_flexion": list(knee_mean),
            "shoulder_hip_separation": list(separation),
            "tilt": list(tilt_mean),
            "left_arm": list(left_elev),
            "right_elbow_flexion": list(elbow_right),
            "stillness": list(stillness_raw),
        },
        "cocking": {
            "right_wrist_elevation_trough": list(trough),
            "right_wrist_acceleration": list(wrist_acc),
            "torso_rise": list(torso),
            "knee_unload": list(knee_unload),
            "loading_unwind": list(unwind),
        },
        "contact": {
            "right_wrist_elevation_apex": list(wrist_dy),
            "right_wrist_speed_peak": list(speed_peak),
            "right_wrist_acceleration_peak": list(accel_peak),
            "torso_rise": list(torso),
            "right_arm_extension": _directional_arm_extension(wrist_dy, wrist_dist),
            "audio_transient": list(audio_cue),
        },
        "finish": {
            "whole_body_settling": list(stillness_raw),
            "right_wrist_settling": list(wrist_settle),
            "torso_settling": list(torso_settle),
        },
    }
    assert all(len(series) == count for staged in raw.values() for series in staged.values())
    return raw


# --- Candidate records -------------------------------------------------------


def _check_stage(name: str, value: Any) -> str:
    if value not in COMPOSITE_ANCHOR_STAGES:
        raise CompositeAnchorsError(
            f"{name}: 'stage' must be one of {list(COMPOSITE_ANCHOR_STAGES)}, "
            f"got {value!r}."
        )
    return value  # type: ignore[return-value]


def _check_score(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompositeAnchorsError(
            f"{name}: 'score' must be a number in [0, 1], got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise CompositeAnchorsError(
            f"{name}: 'score' must lie in [0, 1], got {value!r}."
        )
    return number


def _check_cue_values(
    name: str, stage: str, values: Any
) -> Mapping[str, float | None]:
    expected = COMPOSITE_CUE_NAMES[stage]
    if not isinstance(values, Mapping):
        raise CompositeAnchorsError(
            f"{name}: 'cue_values' must decode from a mapping, "
            f"got {type(values).__name__}."
        )
    if tuple(sorted(values.keys())) != tuple(sorted(expected)):
        raise CompositeAnchorsError(
            f"{name}: 'cue_values' for stage {stage!r} must contain exactly cues "
            f"{list(expected)!r}, got {sorted(values.keys())!r}."
        )
    cleaned: dict[str, float | None] = {}
    for cue in expected:
        entry = values[cue]
        if entry is None:
            cleaned[cue] = None
            continue
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise CompositeAnchorsError(
                f"{name}: cue {cue!r} must be a number in [0, 1] or null, "
                f"got {entry!r}."
            )
        number = float(entry)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise CompositeAnchorsError(
                f"{name}: cue {cue!r} must lie in [0, 1] or be null, got {entry!r}."
            )
        cleaned[cue] = number
    return MappingProxyType(cleaned)


@dataclass(frozen=True, slots=True)
class CompositeAnchorCandidate:
    """One immutable dense anchor candidate at an exact waveform PTS.

    ``cue_values`` holds per-cue quantile-normalized values in
    ``[0, 1]`` (``None`` when the cue lacked qualified waveform support
    at this frame). ``score`` is the availability-weighted mean over the
    available cues; ``coverage`` is the mean available cue weight.
    ``provenance`` is always ``"kinematic_waveform"`` (pure waveform
    evidence; no audio, no manual labels).
    ``temporal_uncertainty_seconds`` is the maximum per-channel waveform
    uncertainty at this PTS (honest bound, never narrower than support).
    """

    stage: str = "start"
    time_seconds: float = 0.0
    timestamp_ms: int = 0
    score: float = 0.0
    coverage: float = 0.0
    cue_values: Mapping[str, float | None] = field(default_factory=dict)  # type: ignore[assignment]
    provenance: str = COMPOSITE_ANCHOR_PROVENANCE
    temporal_uncertainty_seconds: float = 0.0
    method_version: str = COMPOSITE_ANCHORS_METHOD_VERSION
    config_id: str = COMPOSITE_ANCHORS_DEFAULT_CONFIG_ID
    schema_version: int = COMPOSITE_ANCHORS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "composite_anchor_candidate"
        if not _is_int(self.schema_version):
            raise CompositeAnchorsError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != COMPOSITE_ANCHORS_SCHEMA_VERSION:
            raise CompositeAnchorsError(
                f"{name}: unsupported schema_version {self.schema_version!r}; expected "
                f"{COMPOSITE_ANCHORS_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "stage", _check_stage(name, self.stage))
        stage = self.stage
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise CompositeAnchorsError(
                f"{name}: 'time_seconds' must be a finite number >= 0, "
                f"got {self.time_seconds!r}."
            )
        object.__setattr__(self, "time_seconds", float(self.time_seconds))
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise CompositeAnchorsError(
                f"{name}: 'timestamp_ms' must be an integer >= 0, "
                f"got {self.timestamp_ms!r}."
            )
        if self.timestamp_ms != int(round(float(self.time_seconds) * 1000)):
            raise CompositeAnchorsError(
                f"{name}: 'timestamp_ms' ({self.timestamp_ms!r}) must equal "
                f"round(time_seconds * 1000) ({int(round(float(self.time_seconds) * 1000))!r})."
            )
        object.__setattr__(self, "score", _check_score(name, self.score))
        coverage = self.coverage
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
            raise CompositeAnchorsError(
                f"{name}: 'coverage' must be a number in [0, 1], got {coverage!r}."
            )
        coverage_number = float(coverage)
        if not math.isfinite(coverage_number) or coverage_number < 0.0 or coverage_number > 1.0:
            raise CompositeAnchorsError(
                f"{name}: 'coverage' must lie in [0, 1], got {coverage!r}."
            )
        object.__setattr__(self, "coverage", coverage_number)
        cleaned = _check_cue_values(name, stage, self.cue_values)
        object.__setattr__(self, "cue_values", cleaned)
        available = sum(1 for v in cleaned.values() if v is not None)
        if available == 0 and (self.score != 0.0 or self.coverage != 0.0):
            raise CompositeAnchorsError(
                f"{name}: zero available cues must carry score == 0.0 and "
                f"coverage == 0.0, got score={self.score!r} coverage={self.coverage!r}."
            )
        if available > 0 and self.coverage <= 0.0:
            raise CompositeAnchorsError(
                f"{name}: positive available cues must carry coverage > 0, "
                f"got {self.coverage!r}."
            )
        if not isinstance(self.provenance, str) or self.provenance != (
            COMPOSITE_ANCHOR_PROVENANCE
        ):
            raise CompositeAnchorsError(
                f"{name}: 'provenance' must equal {COMPOSITE_ANCHOR_PROVENANCE!r}, "
                f"got {self.provenance!r}."
            )
        uncertainty = self.temporal_uncertainty_seconds
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
            raise CompositeAnchorsError(
                f"{name}: 'temporal_uncertainty_seconds' must be finite and >= 0, "
                f"got {uncertainty!r}."
            )
        uncertainty_number = float(uncertainty)
        if not math.isfinite(uncertainty_number) or uncertainty_number < 0.0:
            raise CompositeAnchorsError(
                f"{name}: 'temporal_uncertainty_seconds' must be finite and >= 0, "
                f"got {uncertainty!r}."
            )
        object.__setattr__(self, "temporal_uncertainty_seconds", uncertainty_number)
        if not isinstance(self.method_version, str) or self.method_version != (
            COMPOSITE_ANCHORS_METHOD_VERSION
        ):
            raise CompositeAnchorsError(
                f"{name}: 'method_version' must equal "
                f"{COMPOSITE_ANCHORS_METHOD_VERSION!r}, got {self.method_version!r}."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise CompositeAnchorsError(
                f"{name}: 'config_id' must be a non-blank string."
            )

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.cue_values, Mapping)
        return {
            "config_id": self.config_id,
            "coverage": self.coverage,
            "cue_values": {cue: self.cue_values[cue] for cue in COMPOSITE_CUE_NAMES[self.stage]},
            "method_version": self.method_version,
            "provenance": self.provenance,
            "schema_version": self.schema_version,
            "score": self.score,
            "stage": self.stage,
            "temporal_uncertainty_seconds": self.temporal_uncertainty_seconds,
            "time_seconds": self.time_seconds,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CompositeAnchorCandidate:
        name = "composite_anchor_candidate"
        if not isinstance(values, dict):
            raise CompositeAnchorsError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "config_id",
            "coverage",
            "cue_values",
            "method_version",
            "provenance",
            "schema_version",
            "score",
            "stage",
            "temporal_uncertainty_seconds",
            "time_seconds",
            "timestamp_ms",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CompositeAnchorsError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise CompositeAnchorsError(f"{name}: unknown keys {unknown!r}.")
        _check_schema_version(name, values)
        raw_cues = values["cue_values"]
        if not isinstance(raw_cues, dict):
            raise CompositeAnchorsError(f"{name}: 'cue_values' must decode from a mapping.")
        return cls(
            stage=values["stage"],
            time_seconds=values["time_seconds"],
            timestamp_ms=values["timestamp_ms"],
            score=values["score"],
            coverage=values["coverage"],
            cue_values=dict(raw_cues),
            provenance=values["provenance"],
            temporal_uncertainty_seconds=values["temporal_uncertainty_seconds"],
            method_version=values["method_version"],
            config_id=values["config_id"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> CompositeAnchorCandidate:
        return cls.from_dict(_loads_object("composite_anchor_candidate", data))


@dataclass(frozen=True, slots=True)
class CompositeAnchorSet:
    """Immutable dense candidate collection for one waveform track.

    Holds exactly ``len(track)`` candidates per anchor stage in
    deterministic order (canonical stage order, then increasing PTS).
    No chronology is enforced and no stage is selected here; the dense
    grid preserves every frame for M4.10b mapping.
    """

    config_id: str = COMPOSITE_ANCHORS_DEFAULT_CONFIG_ID
    method_version: str = COMPOSITE_ANCHORS_METHOD_VERSION
    candidates: tuple[CompositeAnchorCandidate, ...] = ()
    schema_version: int = COMPOSITE_ANCHORS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "composite_anchor_set"
        if not _is_int(self.schema_version):
            raise CompositeAnchorsError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != COMPOSITE_ANCHORS_SCHEMA_VERSION:
            raise CompositeAnchorsError(
                f"{name}: unsupported schema_version {self.schema_version!r}; expected "
                f"{COMPOSITE_ANCHORS_SCHEMA_VERSION}."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise CompositeAnchorsError(f"{name}: 'config_id' must be non-blank.")
        if not isinstance(self.method_version, str) or self.method_version != (
            COMPOSITE_ANCHORS_METHOD_VERSION
        ):
            raise CompositeAnchorsError(
                f"{name}: 'method_version' must equal "
                f"{COMPOSITE_ANCHORS_METHOD_VERSION!r}."
            )
        raw = self.candidates
        if not isinstance(raw, (list, tuple)) or not raw:
            raise CompositeAnchorsError(
                f"{name}: 'candidates' must be a non-empty sequence of "
                "CompositeAnchorCandidate."
            )
        normalized = tuple(raw)
        for entry in normalized:
            if not isinstance(entry, CompositeAnchorCandidate):
                raise CompositeAnchorsError(
                    f"{name}: every candidate must be a CompositeAnchorCandidate, "
                    f"got {type(entry).__name__}."
                )
            if entry.method_version != self.method_version:
                raise CompositeAnchorsError(
                    f"{name}: candidate method_version {entry.method_version!r} must match "
                    f"set method_version {self.method_version!r}."
                )
            if entry.config_id != self.config_id:
                raise CompositeAnchorsError(
                    f"{name}: candidate config_id {entry.config_id!r} must match set "
                    f"config_id {self.config_id!r}."
                )
        object.__setattr__(self, "candidates", normalized)
        # Deterministic ordering: stage order, then increasing time.
        order = {stage: i for i, stage in enumerate(COMPOSITE_ANCHOR_STAGES)}
        keys = [(order[c.stage], c.time_seconds, c.timestamp_ms) for c in normalized]
        if keys != sorted(keys):
            raise CompositeAnchorsError(
                f"{name}: candidates must be ordered by (stage order, time_seconds)."
            )
        # Dense preservation: every stage covers the same PTS grid.
        grids = {
            stage: tuple(c.time_seconds for c in normalized if c.stage == stage)
            for stage in COMPOSITE_ANCHOR_STAGES
        }
        reference = grids[COMPOSITE_ANCHOR_STAGES[0]]
        if not reference:
            raise CompositeAnchorsError(
                f"{name}: each stage must hold at least one candidate."
            )
        for stage in COMPOSITE_ANCHOR_STAGES[1:]:
            if grids[stage] != reference:
                raise CompositeAnchorsError(
                    f"{name}: every stage must cover the identical PTS grid "
                    f"(dense preservation); stage {stage!r} diverges."
                )
        for earlier, later in zip(reference, reference[1:]):
            if not later > earlier:
                raise CompositeAnchorsError(
                    f"{name}: PTS grid must be strictly increasing."
                )

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.candidates)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.candidates[index]

    def for_stage(self, stage: str) -> tuple[CompositeAnchorCandidate, ...]:
        """Return dense candidates for ``stage`` in PTS order."""
        if stage not in COMPOSITE_ANCHOR_STAGES:
            raise CompositeAnchorsError(
                f"composite_anchor_set: unknown stage {stage!r}."
            )
        return tuple(entry for entry in self.candidates if entry.stage == stage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [entry.to_dict() for entry in self.candidates],
            "config_id": self.config_id,
            "method_version": self.method_version,
            "schema_version": self.schema_version,
            "stages": list(COMPOSITE_ANCHOR_STAGES),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CompositeAnchorSet:
        name = "composite_anchor_set"
        if not isinstance(values, dict):
            raise CompositeAnchorsError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"candidates", "config_id", "method_version", "schema_version", "stages"}
        missing = sorted(known - set(values))
        if missing:
            raise CompositeAnchorsError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise CompositeAnchorsError(f"{name}: unknown keys {unknown!r}.")
        _check_schema_version(name, values)
        if list(values["stages"]) != list(COMPOSITE_ANCHOR_STAGES):
            raise CompositeAnchorsError(
                f"{name}: 'stages' must equal {list(COMPOSITE_ANCHOR_STAGES)!r}."
            )
        raw_candidates = values["candidates"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise CompositeAnchorsError(
                f"{name}: 'candidates' must be a non-empty JSON list."
            )
        try:
            candidates = tuple(
                CompositeAnchorCandidate.from_dict(entry) for entry in raw_candidates
            )
        except CompositeAnchorsError:
            raise
        except Exception as exc:
            raise CompositeAnchorsError(f"{name}: invalid candidate: {exc}.") from exc
        return cls(
            config_id=values["config_id"],
            method_version=values["method_version"],
            candidates=candidates,
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> CompositeAnchorSet:
        return cls.from_dict(_loads_object("composite_anchor_set", data))


# --- Builder (dense, no chronology, no selection) -----------------------------


def _validate_track(name: str, track: Any) -> KinematicWaveformTrack:
    if not isinstance(track, KinematicWaveformTrack):
        raise CompositeAnchorsError(
            f"{name}: 'track' must be a KinematicWaveformTrack, "
            f"got {type(track).__name__}."
        )
    if len(track.samples) == 0:
        raise CompositeAnchorsError(
            f"{name}: waveform track holds no samples; refusing to emit candidates."
        )
    for channel in (
        "left_arm_elevation",
        "left_arm_extension",
        "elbow_flexion_left",
        "elbow_flexion_right",
        "knee_flexion_left",
        "knee_flexion_right",
        "shoulder_hip_separation_transverse_deg",
        "shoulder_tilt_deg",
        "hip_tilt_deg",
        "torso_rise",
        "right_wrist_rel_shoulder_dy",
        "right_wrist_rel_shoulder_distance",
        "right_wrist_speed",
        "right_wrist_acceleration",
        "right_wrist_speed_peak",
        "right_wrist_speed_trough",
        "right_wrist_accel_peak",
        "right_wrist_accel_trough",
        "whole_body_settling_energy",
        "audio_transient_energy",
        "audio_transient_flag",
    ):
        if channel not in CHANNEL_INDEX:
            raise CompositeAnchorsError(
                f"{name}: waveform track is missing required channel {channel!r}."
            )
    return track


def compute_composite_anchor_scores(
    track: KinematicWaveformTrack,
    config: CompositeAnchorConfig | None = None,
) -> Mapping[str, Mapping[str, tuple[float | None, ...]]]:
    """Compute normalized cue series per stage (diagnostic helper).

    Returns ``{stage: {cue: tuple[normalized|None ...]}}`` aligned with
    ``track.samples``. Pure and deterministic;same quantile
    normalization the builder uses.
    """
    cfg = config if config is not None else CompositeAnchorConfig()
    if not isinstance(cfg, CompositeAnchorConfig):
        raise CompositeAnchorsError(
            "compute_composite_anchor_scores: 'config' must be a "
            f"CompositeAnchorConfig, got {type(config).__name__}."
        )
    _validate_track("compute_composite_anchor_scores", track)
    raw = _extract_raw_cues(track)
    normalized: dict[str, dict[str, tuple[float | None, ...]]] = {}
    for stage in COMPOSITE_ANCHOR_STAGES:
        staged: dict[str, tuple[float | None, ...]] = {}
        for cue in COMPOSITE_CUE_NAMES[stage]:
            staged[cue] = quantile_normalize_series(raw[stage][cue])
        normalized[stage] = staged
    # Preserve explicit binary audio flags verbatim (never p5/p95-normalize).
    normalized["contact"]["audio_transient"] = _normalize_audio_transient_cue(
        _optional_series(track, "audio_transient_flag"),
        _optional_series(track, "audio_transient_energy"),
    )
    outer: dict[str, Mapping[str, tuple[float | None, ...]]] = {
        stage: MappingProxyType(dict(staged)) for stage, staged in normalized.items()
    }
    return MappingProxyType(outer)


def build_composite_anchor_set(
    track: KinematicWaveformTrack,
    config: CompositeAnchorConfig | None = None,
) -> CompositeAnchorSet:
    """Build the dense six-anchor candidate set for one waveform track.

    Exactly one candidate per waveform sample per stage, in deterministic
    ``(stage order, PTS)`` order with verbatim PTS. No chronology is
    enforced and no stage is selected here.
    """
    cfg = config if config is not None else CompositeAnchorConfig()
    if not isinstance(cfg, CompositeAnchorConfig):
        raise CompositeAnchorsError(
            "build_composite_anchor_set: 'config' must be a CompositeAnchorConfig, "
            f"got {type(config).__name__}."
        )
    _validate_track("build_composite_anchor_set", track)
    raw = _extract_raw_cues(track)
    normalized: dict[str, dict[str, tuple[float | None, ...]]] = {}
    for stage in COMPOSITE_ANCHOR_STAGES:
        staged: dict[str, tuple[float | None, ...]] = {}
        for cue in COMPOSITE_CUE_NAMES[stage]:
            staged[cue] = quantile_normalize_series(raw[stage][cue])
        normalized[stage] = staged
    # Preserve explicit binary audio flags verbatim (never p5/p95-normalize).
    normalized["contact"]["audio_transient"] = _normalize_audio_transient_cue(
        _optional_series(track, "audio_transient_flag"),
        _optional_series(track, "audio_transient_energy"),
    )

    candidates: list[CompositeAnchorCandidate] = []
    for stage in COMPOSITE_ANCHOR_STAGES:
        weights = cfg.weight_for(stage)
        cues = COMPOSITE_CUE_NAMES[stage]
        total_weight = sum(float(weights[cue]) for cue in cues)
        for index, sample in enumerate(track.samples):
            cue_values: dict[str, float | None] = {
                cue: normalized[stage][cue][index] for cue in cues
            }
            available_weight = sum(
                float(weights[cue])
                for cue in cues
                if cue_values[cue] is not None
            )
            coverage = (
                available_weight / total_weight
                if total_weight > 0.0
                else 0.0
            )
            if available_weight > 0.0:
                weighted = sum(
                    float(weights[cue]) * float(cue_values[cue])  # type: ignore[arg-type]
                    for cue in cues
                    if cue_values[cue] is not None
                )
                score = weighted / available_weight
            else:
                score = 0.0
            score = max(0.0, min(1.0, float(score)))
            coverage = max(0.0, min(1.0, float(coverage)))
            uncertainty = max(
                (float(u) for u in sample.channel_uncertainty_seconds),
                default=0.0,
            )
            if not math.isfinite(uncertainty) or uncertainty < 0.0:
                uncertainty = 0.0
            candidates.append(
                CompositeAnchorCandidate(
                    stage=stage,
                    time_seconds=float(sample.time_seconds),
                    timestamp_ms=int(sample.timestamp_ms),
                    score=score,
                    coverage=coverage,
                    cue_values=cue_values,
                    provenance=COMPOSITE_ANCHOR_PROVENANCE,
                    temporal_uncertainty_seconds=float(uncertainty),
                    method_version=COMPOSITE_ANCHORS_METHOD_VERSION,
                    config_id=cfg.config_id,
                    schema_version=COMPOSITE_ANCHORS_SCHEMA_VERSION,
                )
            )
    return CompositeAnchorSet(
        config_id=cfg.config_id,
        method_version=COMPOSITE_ANCHORS_METHOD_VERSION,
        candidates=tuple(candidates),
        schema_version=COMPOSITE_ANCHORS_SCHEMA_VERSION,
    )


def generate_composite_anchor_candidates(
    track: KinematicWaveformTrack,
    config: CompositeAnchorConfig | None = None,
) -> CompositeAnchorSet:
    """Alias for :func:`build_composite_anchor_set` (dense, no selection)."""
    return build_composite_anchor_set(track, config)
