"""Segment-safe low-pass Butterworth filtering of dense-world tracks (M4.8).

Pure, deterministic filtering of ``dense-world-v1``
:class:`~serve_review.pose.world.WorldFrameObservation` rows into an
immutable filtered-world track. No media decoding, no inference, no
cache writes, no waveform metrics, no checkpoint logic, no CLI.

Approved filter design (fixed defaults, versioned below):

- SciPy second-order-sections Butterworth low-pass, order 4;
- cutoff 12.0 Hz against the track sample rate estimated from the
  supplied exact-PTS grid (median step);
- zero-phase offline application via :func:`scipy.signal.sosfiltfilt`
  (odd extension, SciPy default padding);
- minimum continuous qualified segment of 21 samples.

Grid and gap contract:

- The canonical native-time analysis grid is exactly the supplied
  strictly increasing ``time_seconds`` sequence. No uniform
  assumed-FPS resampling is performed; each output sample carries its
  input PTS verbatim, so irregular (VFR-style) spacing is preserved in
  the output time metadata.
- The filter operates on the sample-index sequence inside each
  segment using the median-step sample rate for the analog design.
  Irregular spacing is therefore retained honestly rather than
  resampled away; per-sample temporal uncertainty records the local
  interpolation support.
- A joint sample is *qualified* when its world joint is present
  (``WorldLandmark``, never a zero fill). Missing joints stay missing;
  no-person frames are unqualified for every joint.
- Linear-in-time interpolation is allowed only across short interior
  gaps: a missing run bounded on both sides by qualified samples whose
  bounding span is ``<= max_interpolation_gap_seconds``, and only when
  every consecutive qualified step inside the segment also satisfies
  that bound. Any longer time gap (including a long PTS jump between
  consecutive frames) splits the track into independent segments that
  are never filtered across.
- Interpolated samples receive filtered values but keep
  ``observed == False`` / ``interpolated == True`` with reduced
  observation quality and widened temporal uncertainty. Samples in
  segments shorter than ``min_segment_samples`` (or no longer than the
  SciPy padding length) keep ``available == False`` and
  ``filtered == None`` honestly: the raw input is never substituted as
  a filtered value.
- Samples within the SciPy odd-extension padding length of a segment
  boundary carry ``edge == True`` with reduced filter confidence;
  interior observed samples carry full confidence. Unavailable samples
  carry zero filter confidence.

All 33 joints and all x/y/z axes are filtered independently with the
same section coefficients. Input rows are never mutated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import signal

from serve_review.pose.schema import FrameObservation
from serve_review.pose.world import (
    NUM_WORLD_LANDMARKS,
    WorldFrameObservation,
    WorldLandmark,
)

try:
    from serve_review.pose.schema import JOINT_NAMES as _JOINT_NAMES
except Exception:  # pragma: no cover - schema always provides joint names
    _JOINT_NAMES = tuple(f"joint-{index:02d}" for index in range(NUM_WORLD_LANDMARKS))

__all__ = [
    "WORLD_FILTER_SCHEMA_VERSION",
    "WORLD_FILTER_METHOD_VERSION",
    "WORLD_FILTER_DEFAULT_ORDER",
    "WORLD_FILTER_DEFAULT_CUTOFF_HZ",
    "WORLD_FILTER_DEFAULT_MIN_SEGMENT_SAMPLES",
    "WORLD_FILTER_DEFAULT_MAX_INTERPOLATION_GAP_SECONDS",
    "WorldFilterError",
    "WorldFilterConfig",
    "FilterSegment",
    "FilteredWorldSample",
    "FilteredWorldTrack",
    "default_sos_padlen",
    "design_filter_sos",
    "estimate_sample_rate_hz",
    "build_filtered_world_track",
]

#: Version of every filtered-world schema in this module.
WORLD_FILTER_SCHEMA_VERSION = 1
#: Method identity recorded on every filtered track.
WORLD_FILTER_METHOD_VERSION = "butterworth-sos-zerophase-v1"
#: Approved default: Butterworth order.
WORLD_FILTER_DEFAULT_ORDER = 4
#: Approved default: low-pass cutoff in Hz.
WORLD_FILTER_DEFAULT_CUTOFF_HZ = 12.0
#: Approved default: minimum continuous qualified segment in samples.
WORLD_FILTER_DEFAULT_MIN_SEGMENT_SAMPLES = 21
#: Default short-gap interpolation bound in seconds.
WORLD_FILTER_DEFAULT_MAX_INTERPOLATION_GAP_SECONDS = 0.10

#: Full filter confidence for interior observed samples.
_FULL_CONFIDENCE = 1.0
#: Reduced filter confidence for edge/padding samples.
_EDGE_CONFIDENCE = 0.5
#: Reduced confidence/quality for short-gap interpolated samples.
_INTERPOLATED_CONFIDENCE = 0.5
_INTERPOLATED_QUALITY = 0.5
#: Confidence/quality when no filtered value is available.
_UNAVAILABLE_CONFIDENCE = 0.0
_OBSERVED_QUALITY = 1.0
_MISSING_QUALITY = 0.0


class WorldFilterError(ValueError):
    """Raised when world-track filter input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_filter_version(name: str, values: dict[str, Any]) -> None:
    if "schema_version" not in values:
        raise WorldFilterError(f"{name}: missing required key 'schema_version'.")
    version = values["schema_version"]
    if not _is_int(version):
        raise WorldFilterError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != WORLD_FILTER_SCHEMA_VERSION:
        if version > WORLD_FILTER_SCHEMA_VERSION:
            raise WorldFilterError(
                f"{name}: unsupported newer schema_version {version!r}; "
                f"this build supports version {WORLD_FILTER_SCHEMA_VERSION}."
            )
        raise WorldFilterError(
            f"{name}: unsupported schema_version {version!r}; "
            f"expected version {WORLD_FILTER_SCHEMA_VERSION}."
        )


def _dumps_deterministic(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _loads_object(name: str, data: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorldFilterError(f"{name}: invalid UTF-8 JSON payload.") from exc
    if not isinstance(data, str):
        raise WorldFilterError(
            f"{name}: JSON payload must be str or bytes, "
            f"got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise WorldFilterError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise WorldFilterError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


@dataclass(frozen=True, slots=True)
class WorldFilterConfig:
    """Immutable versioned configuration for world-track filtering.

    Defaults are the approved M4.8 design: SOS Butterworth order 4,
    12.0 Hz low-pass, zero-phase :func:`scipy.signal.sosfiltfilt`, and
    a 21-sample minimum qualified segment. Other positive values are
    accepted (and validated) so the design stays explicit and
    versioned, but callers must use the defaults unless a later leaf
    re-approves them.
    """

    filter_order: int = WORLD_FILTER_DEFAULT_ORDER
    cutoff_hz: float = WORLD_FILTER_DEFAULT_CUTOFF_HZ
    min_segment_samples: int = WORLD_FILTER_DEFAULT_MIN_SEGMENT_SAMPLES
    max_interpolation_gap_seconds: float = (
        WORLD_FILTER_DEFAULT_MAX_INTERPOLATION_GAP_SECONDS
    )
    config_id: str = "world-butterworth-default-v1"
    schema_version: int = WORLD_FILTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "world_filter_config"
        if not _is_int(self.schema_version):
            raise WorldFilterError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != WORLD_FILTER_SCHEMA_VERSION:
            raise WorldFilterError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {WORLD_FILTER_SCHEMA_VERSION}."
            )
        if not _is_int(self.filter_order) or self.filter_order < 1:
            raise WorldFilterError(
                f"{name}: 'filter_order' must be an integer >= 1, "
                f"got {self.filter_order!r}."
            )
        if (
            isinstance(self.cutoff_hz, bool)
            or not isinstance(self.cutoff_hz, (int, float))
            or not math.isfinite(float(self.cutoff_hz))
            or float(self.cutoff_hz) <= 0.0
        ):
            raise WorldFilterError(
                f"{name}: 'cutoff_hz' must be a finite number > 0, "
                f"got {self.cutoff_hz!r}."
            )
        if not _is_int(self.min_segment_samples) or self.min_segment_samples < 1:
            raise WorldFilterError(
                f"{name}: 'min_segment_samples' must be an integer >= 1, "
                f"got {self.min_segment_samples!r}."
            )
        if (
            isinstance(self.max_interpolation_gap_seconds, bool)
            or not isinstance(self.max_interpolation_gap_seconds, (int, float))
            or not math.isfinite(float(self.max_interpolation_gap_seconds))
            or float(self.max_interpolation_gap_seconds) <= 0.0
        ):
            raise WorldFilterError(
                f"{name}: 'max_interpolation_gap_seconds' must be a finite "
                f"number > 0, got {self.max_interpolation_gap_seconds!r}."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise WorldFilterError(
                f"{name}: 'config_id' must be a non-blank string, "
                f"got {self.config_id!r}."
            )
        object.__setattr__(self, "filter_order", int(self.filter_order))
        object.__setattr__(self, "cutoff_hz", float(self.cutoff_hz))
        object.__setattr__(
            self, "min_segment_samples", int(self.min_segment_samples)
        )
        object.__setattr__(
            self,
            "max_interpolation_gap_seconds",
            float(self.max_interpolation_gap_seconds),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "cutoff_hz": self.cutoff_hz,
            "filter_order": self.filter_order,
            "max_interpolation_gap_seconds": self.max_interpolation_gap_seconds,
            "min_segment_samples": self.min_segment_samples,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> WorldFilterConfig:
        name = "world_filter_config"
        if not isinstance(values, dict):
            raise WorldFilterError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "config_id",
            "cutoff_hz",
            "filter_order",
            "max_interpolation_gap_seconds",
            "min_segment_samples",
            "schema_version",
        }
        missing = sorted(known - set(values))
        if missing:
            raise WorldFilterError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise WorldFilterError(f"{name}: unknown keys {unknown!r}.")
        try:
            _check_filter_version(name, values)
        except WorldFilterError:
            raise
        try:
            return cls(
                filter_order=values["filter_order"],
                cutoff_hz=values["cutoff_hz"],
                min_segment_samples=values["min_segment_samples"],
                max_interpolation_gap_seconds=values[
                    "max_interpolation_gap_seconds"
                ],
                config_id=values["config_id"],
                schema_version=values["schema_version"],
            )
        except WorldFilterError:
            raise
        except (TypeError, ValueError) as exc:
            raise WorldFilterError(f"{name}: invalid configuration: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> WorldFilterConfig:
        return cls.from_dict(_loads_object("world_filter_config", data))


def estimate_sample_rate_hz(times: Sequence[float]) -> float:
    """Estimate the track sample rate from exact-PTS steps.

    The rate is ``1 / median(positive step)`` over the supplied strictly
    increasing canonical times. Deterministic; raises
    :class:`WorldFilterError` on fewer than two times or on
    non-finite/non-positive steps.
    """
    steps: list[float] = []
    ordered = [float(t) for t in list(times)]
    if len(ordered) < 2:
        raise WorldFilterError(
            "estimate_sample_rate_hz: at least two sample times are required, "
            f"got {len(ordered)}."
        )
    for earlier, later in zip(ordered, ordered[1:]):
        step = later - earlier
        if not math.isfinite(step) or step <= 0.0:
            raise WorldFilterError(
                "estimate_sample_rate_hz: sample times must be strictly "
                f"increasing and finite, got {earlier!r} followed by {later!r}."
            )
        steps.append(step)
    ordered_steps = sorted(steps)
    mid = len(ordered_steps) // 2
    if len(ordered_steps) % 2 == 1:
        median = ordered_steps[mid]
    else:
        median = (ordered_steps[mid - 1] + ordered_steps[mid]) / 2.0
    if not math.isfinite(median) or median <= 0.0:
        raise WorldFilterError(
            f"estimate_sample_rate_hz: invalid median step {median!r}."
        )
    return float(1.0 / median)


def design_filter_sos(config: WorldFilterConfig, sample_rate_hz: float) -> np.ndarray:
    """Design the versioned SOS Butterworth low-pass for ``config``.

    Normalized cutoff is ``cutoff_hz / (sample_rate_hz / 2)``. Raises
    :class:`WorldFilterError` when the rate is not finite/positive or
    when the cutoff is at or above Nyquist (an honest failure rather
    than an aliased design).
    """
    if not isinstance(config, WorldFilterConfig):
        raise WorldFilterError(
            "design_filter_sos: 'config' must be a WorldFilterConfig, "
            f"got {type(config).__name__}."
        )
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, float))
        or not math.isfinite(float(sample_rate_hz))
        or float(sample_rate_hz) <= 0.0
    ):
        raise WorldFilterError(
            "design_filter_sos: 'sample_rate_hz' must be a finite number > 0, "
            f"got {sample_rate_hz!r}."
        )
    rate = float(sample_rate_hz)
    nyquist = rate / 2.0
    if not float(config.cutoff_hz) < nyquist:
        raise WorldFilterError(
            "design_filter_sos: cutoff_hz "
            f"({float(config.cutoff_hz)!r}) must lie strictly below Nyquist "
            f"({nyquist!r}) for sample_rate_hz={rate!r}; refusing to design "
            "an aliased filter."
        )
    try:
        sos = signal.butter(
            int(config.filter_order),
            float(config.cutoff_hz) / nyquist,
            btype="low",
            output="sos",
        )
    except Exception as exc:
        raise WorldFilterError(f"design_filter_sos: SciPy design failed: {exc}.") from exc
    return np.asarray(sos, dtype=np.float64)


def default_sos_padlen(sos: np.ndarray) -> int:
    """Return SciPy's default odd-extension padding length for ``sos``.

    Mirrors the ``sosfiltfilt`` default (``3 * (2 * n_sections + 1)``);
    output samples within this distance of a segment boundary are marked
    as edge/padding with reduced filter confidence.
    """
    sections = int(np.asarray(sos).shape[0])
    if sections < 1:
        raise WorldFilterError(
            f"default_sos_padlen: invalid SOS shape {np.asarray(sos).shape!r}."
        )
    return int(3 * (2 * sections + 1))


def _check_bool_list(
    name: str, key: str, values: Any, length: int
) -> tuple[bool, ...]:
    if not isinstance(values, list) or len(values) != length:
        raise WorldFilterError(
            f"{name}: {key!r} must be a JSON list of length {length}, "
            f"got {type(values).__name__} with "
            f"{len(values) if isinstance(values, list) else '?'} entries."
        )
    normalized: list[bool] = []
    for entry in values:
        if not isinstance(entry, bool):
            raise WorldFilterError(
                f"{name}: {key!r} entries must be booleans, got {entry!r}."
            )
        normalized.append(entry)
    return tuple(normalized)


def _check_quality_list(
    name: str, key: str, values: Any, length: int
) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != length:
        raise WorldFilterError(
            f"{name}: {key!r} must be a JSON list of length {length}, "
            f"got {type(values).__name__}."
        )
    normalized: list[float] = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise WorldFilterError(
                f"{name}: {key!r} entries must be numbers in [0, 1], "
                f"got {entry!r}."
            )
        number = float(entry)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise WorldFilterError(
                f"{name}: {key!r} entries must lie in [0, 1], got {entry!r}."
            )
        normalized.append(number)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class FilterSegment:
    """One per-joint continuous qualified segment over the grid.

    ``[start_index, end_index]`` are inclusive grid positions covering
    ``[start_time_seconds, end_time_seconds]``. ``sample_count`` counts
    every grid point covered (including short-gap interpolated points);
    ``qualified_count`` counts only directly observed joints.
    ``filtered`` is True only when zero-phase filtering was applied;
    otherwise ``reason`` explains the honest short/padding outcome.
    """

    joint_index: int = 0
    joint_name: str = ""
    start_index: int = 0
    end_index: int = 0
    start_time_seconds: float = 0.0
    end_time_seconds: float = 0.0
    sample_count: int = 0
    qualified_count: int = 0
    filtered: bool = False
    reason: str = ""
    schema_version: int = WORLD_FILTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "filter_segment"
        if not _is_int(self.schema_version):
            raise WorldFilterError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != WORLD_FILTER_SCHEMA_VERSION:
            raise WorldFilterError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {WORLD_FILTER_SCHEMA_VERSION}."
            )
        if not _is_int(self.joint_index) or not (
            0 <= self.joint_index < NUM_WORLD_LANDMARKS
        ):
            raise WorldFilterError(
                f"{name}: 'joint_index' must be an integer in "
                f"[0, {NUM_WORLD_LANDMARKS}), got {self.joint_index!r}."
            )
        if not isinstance(self.joint_name, str) or not self.joint_name.strip():
            raise WorldFilterError(
                f"{name}: 'joint_name' must be a non-blank string, "
                f"got {self.joint_name!r}."
            )
        for key in ("start_index", "end_index", "sample_count", "qualified_count"):
            value = getattr(self, key)
            if not _is_int(value) or value < 0:
                raise WorldFilterError(
                    f"{name}: {key!r} must be an integer >= 0, got {value!r}."
                )
        if self.end_index < self.start_index:
            raise WorldFilterError(
                f"{name}: 'end_index' ({self.end_index!r}) must be >= "
                f"'start_index' ({self.start_index!r})."
            )
        if self.sample_count != self.end_index - self.start_index + 1:
            raise WorldFilterError(
                f"{name}: 'sample_count' ({self.sample_count!r}) must equal "
                f"end_index - start_index + 1 "
                f"({self.end_index - self.start_index + 1!r})."
            )
        if not (0 < self.qualified_count <= self.sample_count):
            raise WorldFilterError(
                f"{name}: 'qualified_count' must lie in (0, sample_count], "
                f"got {self.qualified_count!r} with "
                f"sample_count={self.sample_count!r}."
            )
        for key in ("start_time_seconds", "end_time_seconds"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise WorldFilterError(
                    f"{name}: {key!r} must be a finite number >= 0, "
                    f"got {value!r}."
                )
        object.__setattr__(self, "start_time_seconds", float(self.start_time_seconds))
        object.__setattr__(self, "end_time_seconds", float(self.end_time_seconds))
        if self.end_time_seconds < self.start_time_seconds:
            raise WorldFilterError(
                f"{name}: 'end_time_seconds' must be >= 'start_time_seconds'."
            )
        if not isinstance(self.filtered, bool):
            raise WorldFilterError(
                f"{name}: 'filtered' must be a bool, got {self.filtered!r}."
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise WorldFilterError(
                f"{name}: 'reason' must be a non-blank string, "
                f"got {self.reason!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_index": self.end_index,
            "end_time_seconds": self.end_time_seconds,
            "filtered": self.filtered,
            "joint_index": self.joint_index,
            "joint_name": self.joint_name,
            "qualified_count": self.qualified_count,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "schema_version": self.schema_version,
            "start_index": self.start_index,
            "start_time_seconds": self.start_time_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FilterSegment:
        name = "filter_segment"
        if not isinstance(values, dict):
            raise WorldFilterError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "end_index",
            "end_time_seconds",
            "filtered",
            "joint_index",
            "joint_name",
            "qualified_count",
            "reason",
            "sample_count",
            "schema_version",
            "start_index",
            "start_time_seconds",
        }
        missing = sorted(known - set(values))
        if missing:
            raise WorldFilterError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise WorldFilterError(f"{name}: unknown keys {unknown!r}.")
        try:
            _check_filter_version(name, values)
        except WorldFilterError:
            raise
        try:
            return cls(**{key: values[key] for key in known})
        except WorldFilterError:
            raise
        except (TypeError, ValueError) as exc:
            raise WorldFilterError(f"{name}: invalid segment: {exc}.") from exc


@dataclass(frozen=True, slots=True)
class FilteredWorldSample:
    """One immutable filtered-world sample on the canonical PTS grid.

    ``time_seconds``/``timestamp_ms`` reproduce the input PTS verbatim.
    ``frame_2d`` is the original synchronized 2D companion (linkage for
    visibility and review; never filtered). ``filtered`` holds the
    primary zero-phase filtered 3D coordinates (``None`` per joint when
    honestly unavailable). ``observed`` marks direct world support;
    ``interpolated`` marks short-gap interpolated support;
    ``available`` marks joints carrying a filtered value; ``edge`` marks
    padding-proximal output. ``filter_confidence`` and
    ``observation_quality`` lie in ``[0, 1]``.
    ``interpolation_span_seconds`` is the widest bounding qualified gap
    supporting any interpolated joint at this sample (0 when none);
    ``temporal_uncertainty_seconds`` is never narrower than half that
    span.
    """

    time_seconds: float = 0.0
    timestamp_ms: int = 0
    frame_2d: FrameObservation | None = None
    filtered: tuple[WorldLandmark | None, ...] = ()
    observed: tuple[bool, ...] = ()
    interpolated: tuple[bool, ...] = ()
    available: tuple[bool, ...] = ()
    edge: tuple[bool, ...] = ()
    filter_confidence: tuple[float, ...] = ()
    observation_quality: tuple[float, ...] = ()
    interpolation_span_seconds: float = 0.0
    temporal_uncertainty_seconds: float = 0.0
    schema_version: int = WORLD_FILTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "filtered_world_sample"
        if not _is_int(self.schema_version):
            raise WorldFilterError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != WORLD_FILTER_SCHEMA_VERSION:
            raise WorldFilterError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {WORLD_FILTER_SCHEMA_VERSION}."
            )
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise WorldFilterError(
                f"{name}: 'time_seconds' must be a finite number >= 0, "
                f"got {self.time_seconds!r}."
            )
        object.__setattr__(self, "time_seconds", float(self.time_seconds))
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise WorldFilterError(
                f"{name}: 'timestamp_ms' must be an integer >= 0, "
                f"got {self.timestamp_ms!r}."
            )
        if self.timestamp_ms != int(round(float(self.time_seconds) * 1000)):
            raise WorldFilterError(
                f"{name}: 'timestamp_ms' ({self.timestamp_ms!r}) must equal "
                f"round(time_seconds * 1000) "
                f"({int(round(float(self.time_seconds) * 1000))!r})."
            )
        if not isinstance(self.frame_2d, FrameObservation):
            raise WorldFilterError(
                f"{name}: 'frame_2d' must be a FrameObservation companion, "
                f"got {type(self.frame_2d).__name__}."
            )
        raw = self.filtered
        if not isinstance(raw, (list, tuple)) or len(tuple(raw)) != NUM_WORLD_LANDMARKS:
            length = len(tuple(raw)) if isinstance(raw, (list, tuple)) else "?"
            raise WorldFilterError(
                f"{name}: 'filtered' must hold exactly {NUM_WORLD_LANDMARKS} "
                f"entries, got {length!r}."
            )
        joints = tuple(raw)
        for index, joint in enumerate(joints):
            if joint is not None and not isinstance(joint, WorldLandmark):
                raise WorldFilterError(
                    f"{name}: filtered joint {index} must be WorldLandmark "
                    f"or None, got {type(joint).__name__}."
                )
        object.__setattr__(self, "filtered", joints)
        for key in ("observed", "interpolated", "available", "edge"):
            object.__setattr__(
                self,
                key,
                _check_bool_list(
                    name, f"'{key}'", list(getattr(self, key)), NUM_WORLD_LANDMARKS
                ),
            )
        for key in ("filter_confidence", "observation_quality"):
            object.__setattr__(
                self,
                key,
                _check_quality_list(
                    name, f"'{key}'", list(getattr(self, key)), NUM_WORLD_LANDMARKS
                ),
            )
        for key in ("interpolation_span_seconds", "temporal_uncertainty_seconds"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise WorldFilterError(
                    f"{name}: {key!r} must be a finite number >= 0, "
                    f"got {value!r}."
                )
            object.__setattr__(self, key, float(value))
        if (
            self.temporal_uncertainty_seconds + 1e-12
            < self.interpolation_span_seconds / 2.0
        ):
            raise WorldFilterError(
                f"{name}: 'temporal_uncertainty_seconds' must be >= "
                f"interpolation_span_seconds / 2."
            )
        for index in range(NUM_WORLD_LANDMARKS):
            is_available = self.available[index]
            joint = self.filtered[index]
            if is_available and joint is None:
                raise WorldFilterError(
                    f"{name}: joint {index} is marked available but carries "
                    "no filtered value."
                )
            if not is_available and joint is not None:
                raise WorldFilterError(
                    f"{name}: joint {index} is marked unavailable but carries "
                    "a filtered value; missing support must stay missing."
                )
            if self.interpolated[index] and self.observed[index]:
                raise WorldFilterError(
                    f"{name}: joint {index} cannot be both observed and "
                    "interpolated."
                )
            if self.observed[index] and self.observation_quality[index] == 0.0:
                raise WorldFilterError(
                    f"{name}: joint {index} is observed but carries zero "
                    "observation quality."
                )
            if not self.observed[index] and not self.interpolated[index]:
                if self.available[index]:
                    raise WorldFilterError(
                        f"{name}: joint {index} has no support "
                        "(neither observed nor interpolated) but is marked "
                        "available; long/missing spans must never be filtered."
                    )
            if not is_available and self.filter_confidence[index] != 0.0:
                raise WorldFilterError(
                    f"{name}: joint {index} is unavailable but carries "
                    "nonzero filter confidence."
                )
        if any(self.interpolated) and self.interpolation_span_seconds <= 0.0:
            raise WorldFilterError(
                f"{name}: interpolated joints require a positive "
                "'interpolation_span_seconds'."
            )
        if not any(self.interpolated) and self.interpolation_span_seconds != 0.0:
            raise WorldFilterError(
                f"{name}: non-interpolated samples must carry "
                "'interpolation_span_seconds' == 0.0."
            )

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.frame_2d, FrameObservation)
        return {
            "available": list(self.available),
            "edge": list(self.edge),
            "filter_confidence": list(self.filter_confidence),
            "filtered": [
                joint.to_dict() if joint is not None else None
                for joint in self.filtered
            ],
            "frame_2d": self.frame_2d.to_dict(),
            "interpolated": list(self.interpolated),
            "interpolation_span_seconds": self.interpolation_span_seconds,
            "observation_quality": list(self.observation_quality),
            "observed": list(self.observed),
            "schema_version": self.schema_version,
            "temporal_uncertainty_seconds": self.temporal_uncertainty_seconds,
            "time_seconds": self.time_seconds,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FilteredWorldSample:
        name = "filtered_world_sample"
        if not isinstance(values, dict):
            raise WorldFilterError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "available",
            "edge",
            "filter_confidence",
            "filtered",
            "frame_2d",
            "interpolated",
            "interpolation_span_seconds",
            "observation_quality",
            "observed",
            "schema_version",
            "temporal_uncertainty_seconds",
            "time_seconds",
            "timestamp_ms",
        }
        missing = sorted(known - set(values))
        if missing:
            raise WorldFilterError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise WorldFilterError(f"{name}: unknown keys {unknown!r}.")
        try:
            _check_filter_version(name, values)
        except WorldFilterError:
            raise
        raw_joints = values["filtered"]
        if not isinstance(raw_joints, list) or len(raw_joints) != NUM_WORLD_LANDMARKS:
            raise WorldFilterError(
                f"{name}: 'filtered' must be a JSON list of length "
                f"{NUM_WORLD_LANDMARKS}."
            )
        joints: list[WorldLandmark | None] = []
        for index, entry in enumerate(raw_joints):
            if entry is None:
                joints.append(None)
                continue
            if not isinstance(entry, dict):
                raise WorldFilterError(
                    f"{name}: filtered joint {index} must be an object or null."
                )
            try:
                joints.append(WorldLandmark.from_dict(entry))
            except Exception as exc:
                raise WorldFilterError(
                    f"{name}: invalid filtered joint {index}: {exc}."
                ) from exc
        try:
            companion = FrameObservation.from_dict(values["frame_2d"])
        except WorldFilterError:
            raise
        except Exception as exc:
            raise WorldFilterError(
                f"{name}: invalid 2D companion: {exc}."
            ) from exc
        try:
            return cls(
                time_seconds=values["time_seconds"],
                timestamp_ms=values["timestamp_ms"],
                frame_2d=companion,
                filtered=tuple(joints),
                observed=tuple(values["observed"]),
                interpolated=tuple(values["interpolated"]),
                available=tuple(values["available"]),
                edge=tuple(values["edge"]),
                filter_confidence=tuple(values["filter_confidence"]),
                observation_quality=tuple(values["observation_quality"]),
                interpolation_span_seconds=values["interpolation_span_seconds"],
                temporal_uncertainty_seconds=values[
                    "temporal_uncertainty_seconds"
                ],
                schema_version=values["schema_version"],
            )
        except WorldFilterError:
            raise
        except (TypeError, ValueError) as exc:
            raise WorldFilterError(f"{name}: invalid sample: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> FilteredWorldSample:
        return cls.from_dict(_loads_object("filtered_world_sample", data))


@dataclass(frozen=True, slots=True)
class FilteredWorldTrack:
    """Immutable deterministic filtered-world track.

    ``samples`` follow the canonical input PTS grid one-to-one (same
    length, same times, same order). ``segments`` inventories every
    per-joint continuous qualified segment. ``sample_rate_hz`` is the
    median-step estimate used for the analog design;
    ``sos_coefficients`` records the exact designed sections.
    """

    config: WorldFilterConfig = None  # type: ignore[assignment]
    method_version: str = WORLD_FILTER_METHOD_VERSION
    sample_rate_hz: float = 0.0
    sos_coefficients: tuple[tuple[float, ...], ...] = ()
    samples: tuple[FilteredWorldSample, ...] = ()
    segments: tuple[FilterSegment, ...] = ()
    schema_version: int = WORLD_FILTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "filtered_world_track"
        if not _is_int(self.schema_version):
            raise WorldFilterError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != WORLD_FILTER_SCHEMA_VERSION:
            raise WorldFilterError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {WORLD_FILTER_SCHEMA_VERSION}."
            )
        if not isinstance(self.config, WorldFilterConfig):
            raise WorldFilterError(
                f"{name}: 'config' must be a WorldFilterConfig, "
                f"got {type(self.config).__name__}."
            )
        if not isinstance(self.method_version, str) or (
            self.method_version != WORLD_FILTER_METHOD_VERSION
        ):
            raise WorldFilterError(
                f"{name}: 'method_version' must equal "
                f"{WORLD_FILTER_METHOD_VERSION!r}, "
                f"got {self.method_version!r}."
            )
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, (int, float))
            or not math.isfinite(float(self.sample_rate_hz))
            or float(self.sample_rate_hz) < 0
        ):
            raise WorldFilterError(
                f"{name}: 'sample_rate_hz' must be a finite number >= 0, "
                f"got {self.sample_rate_hz!r}."
            )
        object.__setattr__(self, "sample_rate_hz", float(self.sample_rate_hz))
        sos_rows = tuple(tuple(float(v) for v in row) for row in self.sos_coefficients)
        if not sos_rows:
            raise WorldFilterError(
                f"{name}: 'sos_coefficients' must hold at least one section."
            )
        for row in sos_rows:
            if len(row) != 6 or not all(math.isfinite(v) for v in row):
                raise WorldFilterError(
                    f"{name}: every SOS row must hold 6 finite coefficients, "
                    f"got {row!r}."
                )
        object.__setattr__(self, "sos_coefficients", sos_rows)
        raw_samples = self.samples
        if not isinstance(raw_samples, (list, tuple)) or not raw_samples:
            raise WorldFilterError(
                f"{name}: 'samples' must be a non-empty sequence of "
                "FilteredWorldSample."
            )
        normalized_samples = tuple(raw_samples)
        for entry in normalized_samples:
            if not isinstance(entry, FilteredWorldSample):
                raise WorldFilterError(
                    f"{name}: every sample must be a FilteredWorldSample, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "samples", normalized_samples)
        for earlier, later in zip(normalized_samples, normalized_samples[1:]):
            if not later.time_seconds > earlier.time_seconds:
                raise WorldFilterError(
                    f"{name}: sample times must be strictly increasing, got "
                    f"{earlier.time_seconds!r} followed by "
                    f"{later.time_seconds!r}."
                )
        raw_segments = self.segments
        if not isinstance(raw_segments, (list, tuple)):
            raise WorldFilterError(
                f"{name}: 'segments' must be a sequence of FilterSegment."
            )
        normalized_segments = tuple(raw_segments)
        for entry in normalized_segments:
            if not isinstance(entry, FilterSegment):
                raise WorldFilterError(
                    f"{name}: every segment must be a FilterSegment, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "segments", normalized_segments)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.samples)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.samples[index]

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.config, WorldFilterConfig)
        return {
            "config": self.config.to_dict(),
            "method_version": self.method_version,
            "sample_rate_hz": self.sample_rate_hz,
            "samples": [sample.to_dict() for sample in self.samples],
            "schema_version": self.schema_version,
            "segments": [segment.to_dict() for segment in self.segments],
            "sos_coefficients": [list(row) for row in self.sos_coefficients],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FilteredWorldTrack:
        name = "filtered_world_track"
        if not isinstance(values, dict):
            raise WorldFilterError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "config",
            "method_version",
            "sample_rate_hz",
            "samples",
            "schema_version",
            "segments",
            "sos_coefficients",
        }
        missing = sorted(known - set(values))
        if missing:
            raise WorldFilterError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise WorldFilterError(f"{name}: unknown keys {unknown!r}.")
        try:
            _check_filter_version(name, values)
        except WorldFilterError:
            raise
        try:
            config = WorldFilterConfig.from_dict(values["config"])
        except WorldFilterError as exc:
            raise WorldFilterError(
                f"{name}: invalid 'config': {exc}."
            ) from exc
        raw_sos = values["sos_coefficients"]
        if not isinstance(raw_sos, list) or not raw_sos:
            raise WorldFilterError(
                f"{name}: 'sos_coefficients' must be a non-empty JSON list."
            )
        sos_rows: list[tuple[float, ...]] = []
        for row in raw_sos:
            if not isinstance(row, list) or len(row) != 6:
                raise WorldFilterError(
                    f"{name}: every SOS row must be a JSON list of 6 numbers."
                )
            sos_rows.append(tuple(row))
        raw_samples = values["samples"]
        if not isinstance(raw_samples, list) or not raw_samples:
            raise WorldFilterError(
                f"{name}: 'samples' must be a non-empty JSON list."
            )
        try:
            samples = tuple(
                FilteredWorldSample.from_dict(entry) for entry in raw_samples
            )
        except WorldFilterError:
            raise
        except Exception as exc:
            raise WorldFilterError(
                f"{name}: invalid sample: {exc}."
            ) from exc
        raw_segments = values["segments"]
        if not isinstance(raw_segments, list):
            raise WorldFilterError(
                f"{name}: 'segments' must be a JSON list."
            )
        try:
            segments = tuple(
                FilterSegment.from_dict(entry) for entry in raw_segments
            )
        except WorldFilterError:
            raise
        except Exception as exc:
            raise WorldFilterError(
                f"{name}: invalid segment: {exc}."
            ) from exc
        try:
            return cls(
                config=config,
                method_version=values["method_version"],
                sample_rate_hz=values["sample_rate_hz"],
                sos_coefficients=tuple(sos_rows),
                samples=samples,
                segments=segments,
                schema_version=values["schema_version"],
            )
        except WorldFilterError:
            raise
        except (TypeError, ValueError) as exc:
            raise WorldFilterError(f"{name}: invalid track: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> FilteredWorldTrack:
        return cls.from_dict(_loads_object("filtered_world_track", data))


def _validate_frames(
    frames: Sequence[WorldFrameObservation],
) -> tuple[WorldFrameObservation, ...]:
    if isinstance(frames, (str, bytes, bytearray)):
        raise WorldFilterError(
            "build_filtered_world_track: 'frames' must be a sequence of "
            f"WorldFrameObservation, got {type(frames).__name__}."
        )
    try:
        items = tuple(frames)  # type: ignore[arg-type]
    except TypeError as exc:
        raise WorldFilterError(
            "build_filtered_world_track: 'frames' must be a sequence of "
            f"WorldFrameObservation: {exc}."
        ) from exc
    if not items:
        raise WorldFilterError(
            "build_filtered_world_track: at least one frame is required; "
            "refusing to emit an empty filtered track."
        )
    for index, entry in enumerate(items):
        if not isinstance(entry, WorldFrameObservation):
            raise WorldFilterError(
                "build_filtered_world_track: entry "
                f"{index} must be a WorldFrameObservation, "
                f"got {type(entry).__name__}."
            )
    for earlier, later in zip(items, items[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise WorldFilterError(
                "build_filtered_world_track: frame times must be strictly "
                f"increasing, got {earlier.time_seconds!r} followed by "
                f"{later.time_seconds!r}; the canonical grid is built only "
                "from the supplied PTS order."
            )
    if len({entry.time_seconds for entry in items}) != len(items):
        raise WorldFilterError(
            "build_filtered_world_track: duplicate sample times are not allowed."
        )
    return items


def _raw_joint_series(
    frames: tuple[WorldFrameObservation, ...], joint_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, qualified)`` for one joint.

    ``values`` is ``(N, 3)`` float64 with NaN rows where the joint is
    missing; ``qualified`` is the boolean observed mask.
    """
    count = len(frames)
    values = np.full((count, 3), np.nan, dtype=np.float64)
    qualified = np.zeros((count,), dtype=bool)
    for row, frame in enumerate(frames):
        joint = frame.world_landmarks[joint_index]
        if joint is None:
            continue
        values[row, 0] = float(joint.x)
        values[row, 1] = float(joint.y)
        values[row, 2] = float(joint.z)
        qualified[row] = True
    return values, qualified


def _split_joint_segments(
    times: list[float],
    qualified: np.ndarray,
    max_gap_seconds: float,
) -> list[tuple[int, int, list[int]]]:
    """Split one joint track into continuous qualified segments.

    Each segment is ``(start_grid, end_grid, qualified_positions)`` where
    the grid span ``[start_grid, end_grid]`` is inclusive and every
    consecutive qualified pair inside satisfies the gap bound. Interior
    missing runs bounded on both sides with a bounding span
    ``<= max_gap_seconds`` are absorbed (interpolated); any longer span
    splits the segment. Returns segments in grid order.
    """
    positions = [int(i) for i in np.flatnonzero(qualified).tolist()]
    segments: list[tuple[int, int, list[int]]] = []
    if not positions:
        return segments
    run: list[int] = [positions[0]]
    for previous, current in zip(positions, positions[1:]):
        span = times[current] - times[previous]
        if span <= max_gap_seconds:
            run.append(current)
        else:
            segments.append((run[0], run[-1], list(run)))
            run = [current]
    segments.append((run[0], run[-1], list(run)))
    return segments


def build_filtered_world_track(
    frames: Sequence[WorldFrameObservation],
    config: WorldFilterConfig | None = None,
) -> FilteredWorldTrack:
    """Filter dense-world rows into a deterministic filtered-world track.

    Args:
        frames: Non-empty strictly increasing ``WorldFrameObservation``
            rows. The canonical grid is exactly these PTS values, in
            this order; the rows are only read, never mutated.
        config: Immutable filter configuration (defaults are the
            approved M4.8 design).

    Returns:
        An immutable :class:`FilteredWorldTrack` with one sample per
        input PTS, per-joint support/quality metadata, and the designed
        SOS coefficients.
    """
    active = config if config is not None else WorldFilterConfig()
    if not isinstance(active, WorldFilterConfig):
        raise WorldFilterError(
            "build_filtered_world_track: 'config' must be a "
            f"WorldFilterConfig, got {type(config).__name__}."
        )
    items = _validate_frames(frames)
    count = len(items)
    times = [float(entry.time_seconds) for entry in items]
    stamps = [int(entry.timestamp_ms) for entry in items]

    if count >= 2:
        sample_rate_hz = estimate_sample_rate_hz(times)
    else:
        sample_rate_hz = 0.0

    sos: np.ndarray | None = None
    padlen = 0
    if count >= 2:
        sos = design_filter_sos(active, sample_rate_hz)
        padlen = default_sos_padlen(sos)
    else:
        # Single-sample tracks cannot estimate a rate, so no filtering is
        # ever applied to them. Record digital-identity sections (one per
        # ceil(order / 2)) purely so the versioned schema stays complete;
        # every joint stays honestly unavailable below.
        n_sections = max(1, (int(active.filter_order) + 1) // 2)
        identity = np.zeros((n_sections, 6), dtype=np.float64)
        identity[:, 0] = 1.0
        identity[:, 3] = 1.0
        sos = identity

    max_gap = float(active.max_interpolation_gap_seconds)
    min_samples = int(active.min_segment_samples)

    # Per-joint outputs indexed [joint][grid] then transposed per sample.
    filtered_xyz: list[list[WorldLandmark | None]] = [
        [None] * count for _ in range(NUM_WORLD_LANDMARKS)
    ]
    observed_mask = np.zeros((NUM_WORLD_LANDMARKS, count), dtype=bool)
    interpolated_mask = np.zeros((NUM_WORLD_LANDMARKS, count), dtype=bool)
    available_mask = np.zeros((NUM_WORLD_LANDMARKS, count), dtype=bool)
    edge_mask = np.zeros((NUM_WORLD_LANDMARKS, count), dtype=bool)
    confidence = np.zeros((NUM_WORLD_LANDMARKS, count), dtype=np.float64)
    quality = np.zeros((NUM_WORLD_LANDMARKS, count), dtype=np.float64)
    span_of = np.zeros((NUM_WORLD_LANDMARKS, count), dtype=np.float64)
    segments: list[FilterSegment] = []

    time_array = np.asarray(times, dtype=np.float64)

    for joint_index in range(NUM_WORLD_LANDMARKS):
        joint_name = (
            _JOINT_NAMES[joint_index]
            if joint_index < len(_JOINT_NAMES)
            else f"joint-{joint_index:02d}"
        )
        values, qualified = _raw_joint_series(items, joint_index)
        observed_mask[joint_index, :] = qualified
        quality[joint_index, qualified] = _OBSERVED_QUALITY
        joint_segments = _split_joint_segments(times, qualified, max_gap)
        for start_grid, end_grid, qualified_positions in joint_segments:
            qualified_count = len(qualified_positions)
            sample_count = end_grid - start_grid + 1
            if qualified_count < min_samples or (
                sos is not None and sample_count <= padlen
            ):
                if qualified_count < min_samples:
                    reason = (
                        f"too_short: {qualified_count} qualified sample(s) "
                        f"< minimum {min_samples}; raw support retained, "
                        "no filtered value emitted."
                    )
                else:
                    reason = (
                        f"padding_dominated: {sample_count} covered sample(s) "
                        f"<= SciPy padding length {padlen}; raw support "
                        "retained, no filtered value emitted."
                    )
                segments.append(
                    FilterSegment(
                        joint_index=joint_index,
                        joint_name=str(joint_name),
                        start_index=start_grid,
                        end_index=end_grid,
                        start_time_seconds=times[start_grid],
                        end_time_seconds=times[end_grid],
                        sample_count=sample_count,
                        qualified_count=qualified_count,
                        filtered=False,
                        reason=reason,
                    )
                )
                continue
            # Interpolate interior missing grid points linearly in time.
            assert sos is not None
            grid_positions = np.arange(start_grid, end_grid + 1, dtype=int)
            support_times = time_array[qualified_positions]
            work = np.empty((sample_count, 3), dtype=np.float64)
            is_gap_point = np.zeros((sample_count,), dtype=bool)
            for axis in range(3):
                support_values = values[qualified_positions, axis]
                work[:, axis] = np.interp(
                    time_array[grid_positions], support_times, support_values
                )
            qualified_set = set(qualified_positions)
            for offset, grid in enumerate(grid_positions.tolist()):
                if int(grid) not in qualified_set:
                    is_gap_point[offset] = True
                    interpolated_mask[joint_index, int(grid)] = True
                    quality[joint_index, int(grid)] = _INTERPOLATED_QUALITY
                    # Bounding qualified span supporting this gap point.
                    before = max(q for q in qualified_positions if q < int(grid))
                    after = min(q for q in qualified_positions if q > int(grid))
                    span_of[joint_index, int(grid)] = times[after] - times[before]
            try:
                axes_filtered = np.empty_like(work)
                for axis in range(3):
                    axes_filtered[:, axis] = signal.sosfiltfilt(
                        sos, work[:, axis]
                    )
            except ValueError as exc:
                segments.append(
                    FilterSegment(
                        joint_index=joint_index,
                        joint_name=str(joint_name),
                        start_index=start_grid,
                        end_index=end_grid,
                        start_time_seconds=times[start_grid],
                        end_time_seconds=times[end_grid],
                        sample_count=sample_count,
                        qualified_count=qualified_count,
                        filtered=False,
                        reason=(
                            "filter_rejected: SciPy sosfiltfilt refused the "
                            f"segment ({exc}); raw support retained, no "
                            "filtered value emitted."
                        ),
                    )
                )
                interpolated_mask[joint_index, grid_positions] = False
                quality[joint_index, grid_positions] = np.where(
                    qualified[grid_positions],
                    _OBSERVED_QUALITY,
                    _MISSING_QUALITY,
                )
                span_of[joint_index, grid_positions] = 0.0
                continue
            margin = min(padlen, sample_count - 1)
            for offset, grid in enumerate(grid_positions.tolist()):
                gi = int(grid)
                is_edge = offset < margin or (sample_count - 1 - offset) < margin
                edge_mask[joint_index, gi] = bool(is_edge)
                available_mask[joint_index, gi] = True
                filtered_xyz[joint_index][gi] = WorldLandmark(
                    x=float(axes_filtered[offset, 0]),
                    y=float(axes_filtered[offset, 1]),
                    z=float(axes_filtered[offset, 2]),
                )
                if is_gap_point[offset]:
                    confidence[joint_index, gi] = _INTERPOLATED_CONFIDENCE
                elif is_edge:
                    confidence[joint_index, gi] = _EDGE_CONFIDENCE
                else:
                    confidence[joint_index, gi] = _FULL_CONFIDENCE
                # Interpolated edge samples honestly carry the lower of the
                # two reductions.
                if is_gap_point[offset] and is_edge:
                    confidence[joint_index, gi] = min(
                        _INTERPOLATED_CONFIDENCE, _EDGE_CONFIDENCE
                    )
            segments.append(
                FilterSegment(
                    joint_index=joint_index,
                    joint_name=str(joint_name),
                    start_index=start_grid,
                    end_index=end_grid,
                    start_time_seconds=times[start_grid],
                    end_time_seconds=times[end_grid],
                    sample_count=sample_count,
                    qualified_count=qualified_count,
                    filtered=True,
                    reason=(
                        f"filtered: zero-phase SOS order "
                        f"{int(active.filter_order)} cutoff "
                        f"{float(active.cutoff_hz)} Hz over "
                        f"{sample_count} sample(s), padding margin {margin}."
                    ),
                )
            )

    # Joints with no qualified sample at all leave no segment record; the
    # per-sample unavailable encoding below is their honest record.

    samples: list[FilteredWorldSample] = []
    for row in range(count):
        row_spans = span_of[:, row]
        row_span = float(np.max(row_spans)) if count else 0.0
        row_uncertainty = float(row_span / 2.0)
        samples.append(
            FilteredWorldSample(
                time_seconds=times[row],
                timestamp_ms=stamps[row],
                frame_2d=items[row].frame_2d,
                filtered=tuple(filtered_xyz[joint][row] for joint in range(NUM_WORLD_LANDMARKS)),
                observed=tuple(bool(observed_mask[joint, row]) for joint in range(NUM_WORLD_LANDMARKS)),
                interpolated=tuple(
                    bool(interpolated_mask[joint, row]) for joint in range(NUM_WORLD_LANDMARKS)
                ),
                available=tuple(bool(available_mask[joint, row]) for joint in range(NUM_WORLD_LANDMARKS)),
                edge=tuple(bool(edge_mask[joint, row]) for joint in range(NUM_WORLD_LANDMARKS)),
                filter_confidence=tuple(
                    float(confidence[joint, row]) for joint in range(NUM_WORLD_LANDMARKS)
                ),
                observation_quality=tuple(
                    float(quality[joint, row]) for joint in range(NUM_WORLD_LANDMARKS)
                ),
                interpolation_span_seconds=row_span,
                temporal_uncertainty_seconds=row_uncertainty,
            )
        )

    ordered_segments = tuple(
        sorted(segments, key=lambda seg: (seg.joint_index, seg.start_index))
    )
    assert sos is not None
    return FilteredWorldTrack(
        config=active,
        method_version=WORLD_FILTER_METHOD_VERSION,
        sample_rate_hz=float(sample_rate_hz),
        sos_coefficients=tuple(tuple(float(v) for v in row) for row in np.asarray(sos).tolist()),
        samples=tuple(samples),
        segments=ordered_segments,
    )
