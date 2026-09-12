"""Attempt-local uniform phase feature grid (M4.2).

Pure, deterministic conversion of existing timestamped pose/audio
observations inside one unpadded accepted attempt into a uniform,
confidence-aware source-time feature grid. No stage selection, no DP
solver, no CLI, no detector/media/pose changes.

Contract:

- Input is existing ordered :class:`FrameObservation` rows plus aligned
  :class:`AudioEnergy` values / candidate transient times and one
  unpadded :class:`MediaRange`. Whole-session re-inference is never
  performed here.
- Output is a uniform attempt-local grid over the half-open attempt
  range ``[start, end)`` at the configured analysis rate, with canonical
  metadata (source frame rate, pose observation rate, analysis grid
  rate) and per-sample ``observed`` / ``observation_quality`` /
  ``interpolation_span_seconds`` / ``derivative_confidence`` /
  ``derivative_quality`` channels.
- Only short, visibility-qualified spans are resampled onto the grid;
  configured long gaps, missing poses, and visibility failures are
  never bridged. Interpolated samples support smoothing/derivatives
  with reduced quality and never claim observed visual evidence.
- Smoothing windows are physical seconds converted deterministically
  to legal odd sample windows for the actual grid cadence.
- Positions/angles use timestamp-aware local-quadratic
  (Savitzky-Golay-equivalent) smoothing; first/second derivatives come
  from the same local fits without assuming nominal FPS.
  Boundaries and gaps explicitly reduce derivative confidence.
- Only supportable signals are emitted: visibility-gated normalized
  wrist/elbow/shoulder/hip/knee/ankle trajectories, elbow/knee angles,
  shoulder/hip/torso axes, camera-relative torso displacement/rotation
  proxies, and aligned audio transient values/candidates. Camera-axis
  quantities stay named camera-relative; no anatomical
  forward/posterior labels, handedness, or viewpoint inference.
- Missingness is preserved honestly; structures are immutable with no
  aliasing; JSON codecs are deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Sequence

from serve_review.domain import MediaRange
from serve_review.media.audio import AudioEnergy
from serve_review.pose.schema import JOINT_INDEX, FrameObservation

__all__ = [
    "PHASE_FEATURES_SCHEMA_VERSION",
    "PHASE_FEATURES_METHOD_VERSION",
    "TRACKED_JOINT_NAMES",
    "PhaseFeaturesError",
    "PhaseFeaturesConfig",
    "PhaseFeatureSample",
    "PhaseFeatureGrid",
    "window_samples_for_seconds",
    "resolve_direct_observation_tolerance_seconds",
    "build_phase_feature_grid",
]

#: Version of the phase-feature configuration and grid schemas.
PHASE_FEATURES_SCHEMA_VERSION = 2
#: Method identity recorded on every grid.
PHASE_FEATURES_METHOD_VERSION = "phase-features-v2"

#: Visibility-gated trajectory joints (left/right pairs).
TRACKED_JOINT_NAMES: tuple[str, ...] = (
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_TRACKED_INDICES: tuple[int, ...] = tuple(JOINT_INDEX[name] for name in TRACKED_JOINT_NAMES)

_LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
_RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]
_LEFT_ELBOW = JOINT_INDEX["left_elbow"]
_RIGHT_ELBOW = JOINT_INDEX["right_elbow"]
_LEFT_WRIST = JOINT_INDEX["left_wrist"]
_RIGHT_WRIST = JOINT_INDEX["right_wrist"]
_LEFT_HIP = JOINT_INDEX["left_hip"]
_RIGHT_HIP = JOINT_INDEX["right_hip"]
_LEFT_KNEE = JOINT_INDEX["left_knee"]
_RIGHT_KNEE = JOINT_INDEX["right_knee"]
_LEFT_ANKLE = JOINT_INDEX["left_ankle"]
_RIGHT_ANKLE = JOINT_INDEX["right_ankle"]

_EXACT_TOLERANCE_SECONDS = 1e-9

#: Fraction of the limiting cadence step bounding direct alignment.
#: Effective tolerance stays strictly below half a step so one source
#: observation can align with at most one grid point and vice versa.
_DIRECT_ALIGNMENT_FRACTION = 0.45
#: Numerical slack for inclusive tolerance comparison.
_DIRECT_TOLERANCE_EPSILON_SECONDS = 1e-12

#: Channels smoothed with the position window (raw interpolated values).
_POSITION_CHANNELS: tuple[str, ...] = (
    "wrist_left_x",
    "wrist_left_y",
    "wrist_right_x",
    "wrist_right_y",
    "elbow_left_x",
    "elbow_left_y",
    "elbow_right_x",
    "elbow_right_y",
    "shoulder_left_x",
    "shoulder_left_y",
    "shoulder_right_x",
    "shoulder_right_y",
    "hip_left_x",
    "hip_left_y",
    "hip_right_x",
    "hip_right_y",
    "knee_left_x",
    "knee_left_y",
    "knee_right_x",
    "knee_right_y",
    "ankle_left_x",
    "ankle_left_y",
    "ankle_right_x",
    "ankle_right_y",
    "elbow_angle_left",
    "elbow_angle_right",
    "knee_angle_left",
    "knee_angle_right",
    "torso_mid_camera_x",
    "torso_mid_camera_y",
    "shoulder_axis_camera_dx",
    "shoulder_axis_camera_dy",
    "hip_axis_camera_dx",
    "hip_axis_camera_dy",
    "torso_axis_camera_dx",
    "torso_axis_camera_dy",
)

#: Channels that additionally yield velocity/acceleration estimates.
_DERIVATIVE_POSITION_KEYS: tuple[str, ...] = (
    "wrist_left_x",
    "wrist_left_y",
    "wrist_right_x",
    "wrist_right_y",
)
_DERIVATIVE_ANGLE_KEYS: tuple[str, ...] = (
    "elbow_angle_left",
    "elbow_angle_right",
    "knee_angle_left",
    "knee_angle_right",
)


class PhaseFeaturesError(ValueError):
    """Raised when phase-feature input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def window_samples_for_seconds(window_seconds: float, grid_rate_hz: float) -> int:
    """Convert a physical smoothing window to a legal odd sample window.

    ``round(window_seconds * grid_rate_hz)`` samples (minimum 1) is
    rounded up by one when even so every Savitzky-Golay-equivalent fit
    stays centered. Deterministic; raises :class:`PhaseFeaturesError`
    on non-finite/non-positive inputs.
    """
    for name, value in (("window_seconds", window_seconds), ("grid_rate_hz", grid_rate_hz)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PhaseFeaturesError(
                f"phase_features: {name!r} must be a number, got {value!r}."
            )
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise PhaseFeaturesError(
                f"phase_features: {name!r} must be finite and > 0, got {value!r}."
            )
    count = max(1, int(round(float(window_seconds) * float(grid_rate_hz))))
    if count % 2 == 0:
        count += 1
    return count


def resolve_direct_observation_tolerance_seconds(
    config: PhaseFeaturesConfig,
    *,
    grid_rate_hz: float | None = None,
    qualified_source_times: Sequence[float] = (),
) -> float:
    """Resolve the effective bounded direct-observation tolerance.

    The effective tolerance is the configured
    ``direct_observation_tolerance_seconds`` capped deterministically by
    the physical cadence so it can never bridge distinct frames or gaps:
    ``_DIRECT_ALIGNMENT_FRACTION`` (0.45) times the uniform grid step,
    times 0.45 of the configured ``max_interpolation_gap_seconds``, and
    times 0.45 of the minimum positive qualified source gap when at
    least two qualified source times are supplied. The result is finite
    and strictly positive; deterministic in its inputs.
    """
    if not isinstance(config, PhaseFeaturesConfig):
        raise PhaseFeaturesError(
            "resolve_direct_observation_tolerance: 'config' must be a "
            f"PhaseFeaturesConfig, got {type(config).__name__}."
        )
    rate = float(config.grid_rate_hz) if grid_rate_hz is None else float(grid_rate_hz)
    if grid_rate_hz is not None:
        if isinstance(grid_rate_hz, bool) or not isinstance(grid_rate_hz, (int, float)):
            raise PhaseFeaturesError(
                "resolve_direct_observation_tolerance: 'grid_rate_hz' must be a "
                f"number > 0, got {grid_rate_hz!r}."
            )
    if not math.isfinite(rate) or rate <= 0.0:
        raise PhaseFeaturesError(
            "resolve_direct_observation_tolerance: 'grid_rate_hz' must be finite "
            f"and > 0, got {grid_rate_hz!r}."
        )
    grid_step = 1.0 / rate
    cap = min(
        float(config.direct_observation_tolerance_seconds),
        _DIRECT_ALIGNMENT_FRACTION * grid_step,
        _DIRECT_ALIGNMENT_FRACTION * float(config.max_interpolation_gap_seconds),
    )
    times = [float(t) for t in list(qualified_source_times)]
    if len(times) >= 2:
        ordered = sorted(times)
        min_gap: float | None = None
        for earlier, later in zip(ordered, ordered[1:]):
            gap = later - earlier
            if math.isfinite(gap) and gap > 0.0:
                min_gap = gap if min_gap is None else min(min_gap, gap)
        if min_gap is not None and min_gap > 0.0:
            cap = min(cap, _DIRECT_ALIGNMENT_FRACTION * min_gap)
    if not math.isfinite(cap) or cap <= 0.0:
        raise PhaseFeaturesError(
            "resolve_direct_observation_tolerance: effective tolerance must be "
            f"finite and > 0, got {cap!r}."
        )
    return float(cap)


def _check_rate(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseFeaturesError(
            f"phase_features: {name!r} must be a number, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise PhaseFeaturesError(
            f"phase_features: {name!r} must be finite and > 0, got {value!r}."
        )
    return number


def _check_gap(name: str, value: Any) -> float:
    return _check_rate(name, value)


def _check_floor(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseFeaturesError(
            f"phase_features: {name!r} must be a number in [0, 1], got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseFeaturesError(
            f"phase_features: {name!r} must lie in [0, 1], got {value!r}."
        )
    return number


@dataclass(frozen=True, slots=True)
class PhaseFeaturesConfig:
    """Versioned configuration for attempt-local phase features."""

    grid_rate_hz: float = 30.0
    max_interpolation_gap_seconds: float = 0.15
    visibility_floor: float = 0.4
    position_smoothing_seconds: float = 0.12
    derivative_window_seconds: float = 0.12
    direct_observation_tolerance_seconds: float = 0.015
    config_id: str = "phase-features-default-v2"
    schema_version: int = PHASE_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise PhaseFeaturesError(
                "phase_features_config: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_FEATURES_SCHEMA_VERSION:
            raise PhaseFeaturesError(
                f"phase_features_config: unsupported schema_version "
                f"{self.schema_version!r}; expected {PHASE_FEATURES_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "grid_rate_hz", _check_rate("'grid_rate_hz'", self.grid_rate_hz))
        object.__setattr__(
            self, "max_interpolation_gap_seconds",
            _check_gap("'max_interpolation_gap_seconds'", self.max_interpolation_gap_seconds),
        )
        object.__setattr__(
            self, "visibility_floor", _check_floor("'visibility_floor'", self.visibility_floor)
        )
        object.__setattr__(
            self, "position_smoothing_seconds",
            _check_rate("'position_smoothing_seconds'", self.position_smoothing_seconds),
        )
        object.__setattr__(
            self, "derivative_window_seconds",
            _check_rate("'derivative_window_seconds'", self.derivative_window_seconds),
        )
        object.__setattr__(
            self, "direct_observation_tolerance_seconds",
            _check_rate(
                "'direct_observation_tolerance_seconds'",
                self.direct_observation_tolerance_seconds,
            ),
        )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise PhaseFeaturesError(
                "phase_features_config: 'config_id' must be a non-blank string, "
                f"got {self.config_id!r}."
            )

    def position_window_samples(self) -> int:
        """Return the legal odd position window for the grid cadence."""
        return window_samples_for_seconds(self.position_smoothing_seconds, self.grid_rate_hz)

    def derivative_window_samples(self) -> int:
        """Return the legal odd derivative window for the grid cadence."""
        return window_samples_for_seconds(self.derivative_window_seconds, self.grid_rate_hz)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "derivative_window_seconds": self.derivative_window_seconds,
            "direct_observation_tolerance_seconds": self.direct_observation_tolerance_seconds,
            "grid_rate_hz": self.grid_rate_hz,
            "max_interpolation_gap_seconds": self.max_interpolation_gap_seconds,
            "position_smoothing_seconds": self.position_smoothing_seconds,
            "schema_version": self.schema_version,
            "visibility_floor": self.visibility_floor,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseFeaturesConfig:
        if not isinstance(values, dict):
            raise PhaseFeaturesError(
                "phase_features_config: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "config_id",
            "derivative_window_seconds",
            "direct_observation_tolerance_seconds",
            "grid_rate_hz",
            "max_interpolation_gap_seconds",
            "position_smoothing_seconds",
            "schema_version",
            "visibility_floor",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseFeaturesError(
                f"phase_features_config: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseFeaturesError(
                f"phase_features_config: unknown keys {unknown!r}."
            )
        return cls(
            grid_rate_hz=values["grid_rate_hz"],
            max_interpolation_gap_seconds=values["max_interpolation_gap_seconds"],
            visibility_floor=values["visibility_floor"],
            position_smoothing_seconds=values["position_smoothing_seconds"],
            derivative_window_seconds=values["derivative_window_seconds"],
            direct_observation_tolerance_seconds=values["direct_observation_tolerance_seconds"],
            config_id=values["config_id"],
            schema_version=values["schema_version"],
        )


def _optional_rate(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseFeaturesError(
            f"phase_feature_sample: {name!r} must be a number or None, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number):
        raise PhaseFeaturesError(
            f"phase_feature_sample: {name!r} must be finite or None, got {value!r}."
        )
    return number


def _optional_unit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = _optional_rate(value, name)
    assert number is not None
    if number < 0.0 or number > 1.0:
        raise PhaseFeaturesError(
            f"phase_feature_sample: {name!r} must lie in [0, 1] or be None, "
            f"got {value!r}."
        )
    return number


def _optional_angle(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = _optional_rate(value, name)
    assert number is not None
    if number < -1e-9 or number > 180.0 + 1e-9:
        raise PhaseFeaturesError(
            f"phase_feature_sample: {name!r} must lie in [0, 180] or be None, "
            f"got {value!r}."
        )
    return max(0.0, min(180.0, number))


def _optional_axis(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = _optional_rate(value, name)
    assert number is not None
    if number < -1.0 - 1e-9 or number > 1.0 + 1e-9:
        raise PhaseFeaturesError(
            f"phase_feature_sample: {name!r} must lie in [-1, 1] or be None, "
            f"got {value!r}."
        )
    return max(-1.0, min(1.0, number))


@dataclass(frozen=True, slots=True)
class PhaseFeatureSample:
    """One immutable uniform-grid sample with confidence channels.

    Trajectory/axis positions are scale-normalized camera-relative
    coordinates (translation/scale invariant); angles are interior
    joint angles in degrees; torso displacement/rotation proxies stay
    explicitly camera-relative; audio carries aligned transient energy
    plus a candidate flag. ``observed`` is True only when one gated pose
    observation lies within the bounded direct-observation tolerance of
    the canonical grid time (at most one source per grid point and vice
    versa; ties deterministic). ``source_time_offset_seconds`` records
    the signed source-minus-grid offset (0.0 when not observed) and
    ``temporal_uncertainty_seconds`` is never narrower than
    ``abs(offset)`` or half the supporting span, so canonical grid times
    never claim precision finer than the direct support. ``observation_quality``
    in [0, 1] is reduced for interpolated samples and zero for
    missing samples. ``derivative_confidence``/``derivative_quality``
    in [0, 1] degrade at boundaries and gaps.
    """

    time_seconds: float = 0.0
    observed: bool = False
    observation_quality: float = 0.0
    interpolation_span_seconds: float = 0.0
    source_time_offset_seconds: float = 0.0
    temporal_uncertainty_seconds: float = 0.0
    derivative_confidence: float = 0.0
    derivative_quality: float = 0.0
    # Normalized trajectories (camera-relative, scale-normalized).
    wrist_left_x: float | None = None
    wrist_left_y: float | None = None
    wrist_right_x: float | None = None
    wrist_right_y: float | None = None
    elbow_left_x: float | None = None
    elbow_left_y: float | None = None
    elbow_right_x: float | None = None
    elbow_right_y: float | None = None
    shoulder_left_x: float | None = None
    shoulder_left_y: float | None = None
    shoulder_right_x: float | None = None
    shoulder_right_y: float | None = None
    hip_left_x: float | None = None
    hip_left_y: float | None = None
    hip_right_x: float | None = None
    hip_right_y: float | None = None
    knee_left_x: float | None = None
    knee_left_y: float | None = None
    knee_right_x: float | None = None
    knee_right_y: float | None = None
    ankle_left_x: float | None = None
    ankle_left_y: float | None = None
    ankle_right_x: float | None = None
    ankle_right_y: float | None = None
    # Joint angles (degrees).
    elbow_angle_left: float | None = None
    elbow_angle_right: float | None = None
    knee_angle_left: float | None = None
    knee_angle_right: float | None = None
    # Body axes (unit vectors, camera-relative).
    shoulder_axis_camera_dx: float | None = None
    shoulder_axis_camera_dy: float | None = None
    hip_axis_camera_dx: float | None = None
    hip_axis_camera_dy: float | None = None
    torso_axis_camera_dx: float | None = None
    torso_axis_camera_dy: float | None = None
    # Torso camera-relative proxies.
    torso_mid_camera_x: float | None = None
    torso_mid_camera_y: float | None = None
    torso_displacement_camera: float | None = None
    torso_rotation_camera_deg: float | None = None
    # Aligned audio.
    audio_energy: float | None = None
    audio_candidate: bool = False
    # First derivatives (timestamp-aware, per source second).
    wrist_left_vx: float | None = None
    wrist_left_vy: float | None = None
    wrist_right_vx: float | None = None
    wrist_right_vy: float | None = None
    elbow_angle_left_vel: float | None = None
    elbow_angle_right_vel: float | None = None
    knee_angle_left_vel: float | None = None
    knee_angle_right_vel: float | None = None
    # Second derivatives.
    wrist_left_ax: float | None = None
    wrist_left_ay: float | None = None
    wrist_right_ax: float | None = None
    wrist_right_ay: float | None = None
    elbow_angle_left_acc: float | None = None
    elbow_angle_right_acc: float | None = None
    knee_angle_left_acc: float | None = None
    knee_angle_right_acc: float | None = None
    schema_version: int = PHASE_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_feature_sample"
        if not _is_int(self.schema_version):
            raise PhaseFeaturesError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_FEATURES_SCHEMA_VERSION:
            raise PhaseFeaturesError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_FEATURES_SCHEMA_VERSION}."
            )
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise PhaseFeaturesError(
                f"{name}: 'time_seconds' must be a finite number >= 0, "
                f"got {self.time_seconds!r}."
            )
        object.__setattr__(self, "time_seconds", float(self.time_seconds))
        if not isinstance(self.observed, bool):
            raise PhaseFeaturesError(
                f"{name}: 'observed' must be a bool, got {self.observed!r}."
            )
        for key in (
            "observation_quality",
            "derivative_confidence",
            "derivative_quality",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PhaseFeaturesError(
                    f"{name}: {key!r} must be a number in [0, 1], got {value!r}."
                )
            number = float(value)
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                raise PhaseFeaturesError(
                    f"{name}: {key!r} must lie in [0, 1], got {value!r}."
                )
            object.__setattr__(self, key, number)
        span = self.interpolation_span_seconds
        if isinstance(span, bool) or not isinstance(span, (int, float)):
            raise PhaseFeaturesError(
                f"{name}: 'interpolation_span_seconds' must be a number >= 0, "
                f"got {span!r}."
            )
        span_number = float(span)
        if not math.isfinite(span_number) or span_number < 0.0:
            raise PhaseFeaturesError(
                f"{name}: 'interpolation_span_seconds' must be finite and >= 0, "
                f"got {span!r}."
            )
        object.__setattr__(self, "interpolation_span_seconds", span_number)
        offset = self.source_time_offset_seconds
        if isinstance(offset, bool) or not isinstance(offset, (int, float)):
            raise PhaseFeaturesError(
                f"{name}: 'source_time_offset_seconds' must be a finite number, "
                f"got {offset!r}."
            )
        offset_number = float(offset)
        if not math.isfinite(offset_number):
            raise PhaseFeaturesError(
                f"{name}: 'source_time_offset_seconds' must be finite, got {offset!r}."
            )
        object.__setattr__(self, "source_time_offset_seconds", offset_number)
        uncertainty = self.temporal_uncertainty_seconds
        if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
            raise PhaseFeaturesError(
                f"{name}: 'temporal_uncertainty_seconds' must be a number >= 0, "
                f"got {uncertainty!r}."
            )
        uncertainty_number = float(uncertainty)
        if not math.isfinite(uncertainty_number) or uncertainty_number < 0.0:
            raise PhaseFeaturesError(
                f"{name}: 'temporal_uncertainty_seconds' must be finite and >= 0, "
                f"got {uncertainty!r}."
            )
        object.__setattr__(self, "temporal_uncertainty_seconds", uncertainty_number)
        if not self.observed and offset_number != 0.0:
            raise PhaseFeaturesError(
                f"{name}: non-observed samples must carry source_time_offset_seconds "
                f"== 0.0, got {offset_number!r}."
            )
        if uncertainty_number + 1e-12 < abs(offset_number):
            raise PhaseFeaturesError(
                f"{name}: 'temporal_uncertainty_seconds' must be >= "
                f"abs(source_time_offset_seconds), got {uncertainty_number!r} vs "
                f"{offset_number!r}."
            )
        if uncertainty_number + 1e-12 < span_number / 2.0:
            raise PhaseFeaturesError(
                f"{name}: 'temporal_uncertainty_seconds' must be >= "
                f"interpolation_span_seconds / 2, got {uncertainty_number!r} vs "
                f"{span_number!r}."
            )
        if self.observed and self.interpolation_span_seconds != 0.0:
            raise PhaseFeaturesError(
                f"{name}: observed samples must carry interpolation_span_seconds "
                f"== 0.0, got {self.interpolation_span_seconds!r}."
            )
        if not self.observed and self.observation_quality == 1.0:
            # Interpolated/missing samples must never claim full observed
            # precision; exact 1.0 is reserved for gated observations.
            # (Observed samples may also sit below 1.0.)
            raise PhaseFeaturesError(
                f"{name}: non-observed samples must not claim "
                "observation_quality == 1.0."
            )
        if not isinstance(self.audio_candidate, bool):
            raise PhaseFeaturesError(
                f"{name}: 'audio_candidate' must be a bool, got {self.audio_candidate!r}."
            )
        for key in (
            "wrist_left_x", "wrist_left_y", "wrist_right_x", "wrist_right_y",
            "elbow_left_x", "elbow_left_y", "elbow_right_x", "elbow_right_y",
            "shoulder_left_x", "shoulder_left_y",
            "shoulder_right_x", "shoulder_right_y",
            "hip_left_x", "hip_left_y", "hip_right_x", "hip_right_y",
            "knee_left_x", "knee_left_y", "knee_right_x", "knee_right_y",
            "ankle_left_x", "ankle_left_y", "ankle_right_x", "ankle_right_y",
            "torso_mid_camera_x", "torso_mid_camera_y",
            "torso_displacement_camera",
            "audio_energy",
            "wrist_left_vx", "wrist_left_vy",
            "wrist_right_vx", "wrist_right_vy",
            "elbow_angle_left_vel", "elbow_angle_right_vel",
            "knee_angle_left_vel", "knee_angle_right_vel",
            "wrist_left_ax", "wrist_left_ay",
            "wrist_right_ax", "wrist_right_ay",
            "elbow_angle_left_acc", "elbow_angle_right_acc",
            "knee_angle_left_acc", "knee_angle_right_acc",
        ):
            object.__setattr__(self, key, _optional_rate(getattr(self, key), f"'{key}'"))
        if self.audio_energy is not None and self.audio_energy < 0.0:
            raise PhaseFeaturesError(
                f"{name}: 'audio_energy' must be >= 0 or None, got {self.audio_energy!r}."
            )
        for key in (
            "elbow_angle_left", "elbow_angle_right",
            "knee_angle_left", "knee_angle_right",
        ):
            object.__setattr__(self, key, _optional_angle(getattr(self, key), f"'{key}'"))
        for key in (
            "shoulder_axis_camera_dx", "shoulder_axis_camera_dy",
            "hip_axis_camera_dx", "hip_axis_camera_dy",
            "torso_axis_camera_dx", "torso_axis_camera_dy",
        ):
            object.__setattr__(self, key, _optional_axis(getattr(self, key), f"'{key}'"))
        rotation = _optional_rate(self.torso_rotation_camera_deg, "'torso_rotation_camera_deg'")
        if rotation is not None and (rotation < -180.0 or rotation > 180.0):
            raise PhaseFeaturesError(
                f"{name}: 'torso_rotation_camera_deg' must lie in [-180, 180] "
                f"or be None, got {self.torso_rotation_camera_deg!r}."
            )
        object.__setattr__(self, "torso_rotation_camera_deg", rotation)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_name in (
            "time_seconds", "observed", "observation_quality",
            "interpolation_span_seconds", "source_time_offset_seconds",
            "temporal_uncertainty_seconds", "derivative_confidence",
            "derivative_quality",
            "wrist_left_x", "wrist_left_y", "wrist_right_x", "wrist_right_y",
            "elbow_left_x", "elbow_left_y", "elbow_right_x", "elbow_right_y",
            "shoulder_left_x", "shoulder_left_y",
            "shoulder_right_x", "shoulder_right_y",
            "hip_left_x", "hip_left_y", "hip_right_x", "hip_right_y",
            "knee_left_x", "knee_left_y", "knee_right_x", "knee_right_y",
            "ankle_left_x", "ankle_left_y", "ankle_right_x", "ankle_right_y",
            "elbow_angle_left", "elbow_angle_right",
            "knee_angle_left", "knee_angle_right",
            "shoulder_axis_camera_dx", "shoulder_axis_camera_dy",
            "hip_axis_camera_dx", "hip_axis_camera_dy",
            "torso_axis_camera_dx", "torso_axis_camera_dy",
            "torso_mid_camera_x", "torso_mid_camera_y",
            "torso_displacement_camera", "torso_rotation_camera_deg",
            "audio_energy", "audio_candidate",
            "wrist_left_vx", "wrist_left_vy",
            "wrist_right_vx", "wrist_right_vy",
            "elbow_angle_left_vel", "elbow_angle_right_vel",
            "knee_angle_left_vel", "knee_angle_right_vel",
            "wrist_left_ax", "wrist_left_ay",
            "wrist_right_ax", "wrist_right_ay",
            "elbow_angle_left_acc", "elbow_angle_right_acc",
            "knee_angle_left_acc", "knee_angle_right_acc",
            "schema_version",
        ):
            payload[field_name] = getattr(self, field_name)
        return payload

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseFeatureSample:
        if not isinstance(values, dict):
            raise PhaseFeaturesError(
                "phase_feature_sample: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "time_seconds", "observed", "observation_quality",
            "interpolation_span_seconds", "source_time_offset_seconds",
            "temporal_uncertainty_seconds", "derivative_confidence",
            "derivative_quality",
            "wrist_left_x", "wrist_left_y", "wrist_right_x", "wrist_right_y",
            "elbow_left_x", "elbow_left_y", "elbow_right_x", "elbow_right_y",
            "shoulder_left_x", "shoulder_left_y",
            "shoulder_right_x", "shoulder_right_y",
            "hip_left_x", "hip_left_y", "hip_right_x", "hip_right_y",
            "knee_left_x", "knee_left_y", "knee_right_x", "knee_right_y",
            "ankle_left_x", "ankle_left_y", "ankle_right_x", "ankle_right_y",
            "elbow_angle_left", "elbow_angle_right",
            "knee_angle_left", "knee_angle_right",
            "shoulder_axis_camera_dx", "shoulder_axis_camera_dy",
            "hip_axis_camera_dx", "hip_axis_camera_dy",
            "torso_axis_camera_dx", "torso_axis_camera_dy",
            "torso_mid_camera_x", "torso_mid_camera_y",
            "torso_displacement_camera", "torso_rotation_camera_deg",
            "audio_energy", "audio_candidate",
            "wrist_left_vx", "wrist_left_vy",
            "wrist_right_vx", "wrist_right_vy",
            "elbow_angle_left_vel", "elbow_angle_right_vel",
            "knee_angle_left_vel", "knee_angle_right_vel",
            "wrist_left_ax", "wrist_left_ay",
            "wrist_right_ax", "wrist_right_ay",
            "elbow_angle_left_acc", "elbow_angle_right_acc",
            "knee_angle_left_acc", "knee_angle_right_acc",
            "schema_version",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseFeaturesError(
                f"phase_feature_sample: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseFeaturesError(
                f"phase_feature_sample: unknown keys {unknown!r}."
            )
        return cls(**{key: values[key] for key in known})


@dataclass(frozen=True, slots=True)
class PhaseFeatureGrid:
    """Immutable uniform attempt-local feature grid."""

    attempt_range: MediaRange = None  # type: ignore[assignment]
    method_version: str = PHASE_FEATURES_METHOD_VERSION
    config_id: str = "phase-features-default-v2"
    source_frame_rate_hz: float = 0.0
    pose_observation_rate_hz: float = 0.0
    grid_rate_hz: float = 0.0
    position_window_samples: int = 1
    derivative_window_samples: int = 1
    direct_observation_tolerance_seconds: float = 0.015
    samples: tuple[PhaseFeatureSample, ...] = ()
    schema_version: int = PHASE_FEATURES_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_feature_grid"
        if not _is_int(self.schema_version):
            raise PhaseFeaturesError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_FEATURES_SCHEMA_VERSION:
            raise PhaseFeaturesError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_FEATURES_SCHEMA_VERSION}."
            )
        if not isinstance(self.attempt_range, MediaRange):
            raise PhaseFeaturesError(
                f"{name}: 'attempt_range' must be a MediaRange, "
                f"got {type(self.attempt_range).__name__}."
            )
        if not isinstance(self.method_version, str) or not self.method_version.strip():
            raise PhaseFeaturesError(
                f"{name}: 'method_version' must be a non-blank string."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise PhaseFeaturesError(
                f"{name}: 'config_id' must be a non-blank string."
            )
        for key in (
            "source_frame_rate_hz",
            "grid_rate_hz",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PhaseFeaturesError(
                    f"{name}: {key!r} must be a number > 0, got {value!r}."
                )
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise PhaseFeaturesError(
                    f"{name}: {key!r} must be finite and > 0, got {value!r}."
                )
            object.__setattr__(self, key, number)
        pose_rate = self.pose_observation_rate_hz
        if isinstance(pose_rate, bool) or not isinstance(pose_rate, (int, float)):
            raise PhaseFeaturesError(
                f"{name}: 'pose_observation_rate_hz' must be a number >= 0."
            )
        pose_number = float(pose_rate)
        if not math.isfinite(pose_number) or pose_number < 0.0:
            raise PhaseFeaturesError(
                f"{name}: 'pose_observation_rate_hz' must be finite and >= 0."
            )
        object.__setattr__(self, "pose_observation_rate_hz", pose_number)
        tolerance = self.direct_observation_tolerance_seconds
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise PhaseFeaturesError(
                f"{name}: 'direct_observation_tolerance_seconds' must be a number > 0, "
                f"got {tolerance!r}."
            )
        tolerance_number = float(tolerance)
        if not math.isfinite(tolerance_number) or tolerance_number <= 0.0:
            raise PhaseFeaturesError(
                f"{name}: 'direct_observation_tolerance_seconds' must be finite and > 0, "
                f"got {tolerance!r}."
            )
        object.__setattr__(self, "direct_observation_tolerance_seconds", tolerance_number)
        for key in ("position_window_samples", "derivative_window_samples"):
            value = getattr(self, key)
            if not _is_int(value) or value < 1 or value % 2 == 0:
                raise PhaseFeaturesError(
                    f"{name}: {key!r} must be an odd integer >= 1, got {value!r}."
                )
        raw = self.samples
        if not isinstance(raw, (list, tuple)):
            raise PhaseFeaturesError(
                f"{name}: 'samples' must be a list or tuple of PhaseFeatureSample."
            )
        normalized = tuple(raw)
        for entry in normalized:
            if not isinstance(entry, PhaseFeatureSample):
                raise PhaseFeaturesError(
                    f"{name}: every sample must be a PhaseFeatureSample, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "samples", normalized)
        for index in range(1, len(normalized)):
            if not normalized[index].time_seconds > normalized[index - 1].time_seconds:
                raise PhaseFeaturesError(
                    f"{name}: sample times must be strictly increasing."
                )
        for sample in normalized:
            if not self.attempt_range.contains(sample.time_seconds):
                raise PhaseFeaturesError(
                    f"{name}: sample time {sample.time_seconds!r} lies outside "
                    f"attempt range {self.attempt_range.to_dict()!r}."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.samples)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.samples[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_range": self.attempt_range.to_dict(),
            "config_id": self.config_id,
            "derivative_window_samples": self.derivative_window_samples,
            "direct_observation_tolerance_seconds": self.direct_observation_tolerance_seconds,
            "grid_rate_hz": self.grid_rate_hz,
            "method_version": self.method_version,
            "pose_observation_rate_hz": self.pose_observation_rate_hz,
            "position_window_samples": self.position_window_samples,
            "samples": [sample.to_dict() for sample in self.samples],
            "schema_version": self.schema_version,
            "source_frame_rate_hz": self.source_frame_rate_hz,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseFeatureGrid:
        if not isinstance(values, dict):
            raise PhaseFeaturesError(
                "phase_feature_grid: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "attempt_range",
            "config_id",
            "derivative_window_samples",
            "direct_observation_tolerance_seconds",
            "grid_rate_hz",
            "method_version",
            "pose_observation_rate_hz",
            "position_window_samples",
            "samples",
            "schema_version",
            "source_frame_rate_hz",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseFeaturesError(
                f"phase_feature_grid: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseFeaturesError(
                f"phase_feature_grid: unknown keys {unknown!r}."
            )
        try:
            attempt_range = MediaRange.from_dict(values["attempt_range"])
        except Exception as exc:
            raise PhaseFeaturesError(
                f"phase_feature_grid: invalid 'attempt_range': {exc}."
            ) from exc
        raw_samples = values["samples"]
        if not isinstance(raw_samples, list):
            raise PhaseFeaturesError(
                "phase_feature_grid: 'samples' must be a JSON list."
            )
        try:
            samples = tuple(PhaseFeatureSample.from_dict(entry) for entry in raw_samples)
        except PhaseFeaturesError:
            raise
        except Exception as exc:
            raise PhaseFeaturesError(
                f"phase_feature_grid: invalid sample: {exc}."
            ) from exc
        return cls(
            attempt_range=attempt_range,
            method_version=values["method_version"],
            config_id=values["config_id"],
            source_frame_rate_hz=values["source_frame_rate_hz"],
            pose_observation_rate_hz=values["pose_observation_rate_hz"],
            grid_rate_hz=values["grid_rate_hz"],
            position_window_samples=values["position_window_samples"],
            derivative_window_samples=values["derivative_window_samples"],
            direct_observation_tolerance_seconds=values["direct_observation_tolerance_seconds"],
            samples=samples,
            schema_version=values["schema_version"],
        )


# --- Internal geometry -------------------------------------------------------


def _gated_joint(frame_joints: Any, index: int, floor: float) -> Any | None:
    try:
        joint = frame_joints[index]
    except (IndexError, TypeError):
        return None
    if joint is None:
        return None
    try:
        visibility = float(joint.visibility)
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(visibility) or visibility < floor:
        return None
    return joint


def _midpoint_coords(
    first: Any | None, second: Any | None
) -> tuple[float, float] | None:
    if first is None or second is None:
        return None
    return ((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)


def _joint_angle_deg(vertex: Any | None, first: Any | None, second: Any | None) -> float | None:
    if vertex is None or first is None or second is None:
        return None
    ax = first.x - vertex.x
    ay = first.y - vertex.y
    bx = second.x - vertex.x
    by = second.y - vertex.y
    norm_a = math.hypot(ax, ay)
    norm_b = math.hypot(bx, by)
    if not math.isfinite(norm_a) or not math.isfinite(norm_b):
        return None
    if norm_a <= 0.0 or norm_b <= 0.0:
        return None
    cosine = (ax * bx + ay * by) / (norm_a * norm_b)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _unit_vector(dx: float, dy: float) -> tuple[float, float] | None:
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return (dx / norm, dy / norm)


def _extract_observation_channels(
    frame: FrameObservation, floor: float
) -> tuple[dict[str, float | None], float]:
    """Extract normalized channels for one observation.

    Returns ``(channels, quality)`` where ``channels`` maps every entry
    of :data:`_POSITION_CHANNELS` to a value or ``None`` and ``quality``
    is the gated tracked-joint fraction in [0, 1] (0 for no-person).
    """
    channels: dict[str, float | None] = {key: None for key in _POSITION_CHANNELS}
    if not frame.has_person:
        return channels, 0.0
    person = frame.persons[0]
    joints = person.keypoints
    gated = [_gated_joint(joints, index, floor) for index in range(len(joints))]

    shoulder_mid = _midpoint_coords(gated[_LEFT_SHOULDER], gated[_RIGHT_SHOULDER])
    hip_mid = _midpoint_coords(gated[_LEFT_HIP], gated[_RIGHT_HIP])
    origin = shoulder_mid if shoulder_mid is not None else hip_mid
    scale: float | None = None
    if shoulder_mid is not None and hip_mid is not None:
        candidate = math.hypot(
            shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]
        )
        if math.isfinite(candidate) and candidate > 0.0:
            scale = candidate
    if scale is None:
        try:
            box_height = person.box.y_max - person.box.y_min
        except (AttributeError, TypeError):
            box_height = None
        if box_height is not None and math.isfinite(box_height) and box_height > 0.0:
            scale = float(box_height)

    present = 0
    for joint_name in TRACKED_JOINT_NAMES:
        index = JOINT_INDEX[joint_name]
        if gated[index] is not None:
            present += 1
    quality = present / len(TRACKED_JOINT_NAMES)

    if origin is None or scale is None or scale <= 0.0:
        # Angles/axes that do not need normalization can still compute.
        channels["elbow_angle_left"] = _joint_angle_deg(
            gated[_LEFT_ELBOW], gated[_LEFT_SHOULDER], gated[_LEFT_WRIST]
        )
        channels["elbow_angle_right"] = _joint_angle_deg(
            gated[_RIGHT_ELBOW], gated[_RIGHT_SHOULDER], gated[_RIGHT_WRIST]
        )
        channels["knee_angle_left"] = _joint_angle_deg(
            gated[_LEFT_KNEE], gated[_LEFT_HIP], gated[_LEFT_ANKLE]
        )
        channels["knee_angle_right"] = _joint_angle_deg(
            gated[_RIGHT_KNEE], gated[_RIGHT_HIP], gated[_RIGHT_ANKLE]
        )
        _fill_axes(channels, gated)
        return channels, quality

    mapping = {
        "wrist_left": _LEFT_WRIST,
        "wrist_right": _RIGHT_WRIST,
        "elbow_left": _LEFT_ELBOW,
        "elbow_right": _RIGHT_ELBOW,
        "shoulder_left": _LEFT_SHOULDER,
        "shoulder_right": _RIGHT_SHOULDER,
        "hip_left": _LEFT_HIP,
        "hip_right": _RIGHT_HIP,
        "knee_left": _LEFT_KNEE,
        "knee_right": _RIGHT_KNEE,
        "ankle_left": _LEFT_ANKLE,
        "ankle_right": _RIGHT_ANKLE,
    }
    for prefix, index in mapping.items():
        joint = gated[index]
        if joint is None:
            continue
        channels[f"{prefix}_x"] = (joint.x - origin[0]) / scale
        channels[f"{prefix}_y"] = (joint.y - origin[1]) / scale

    channels["elbow_angle_left"] = _joint_angle_deg(
        gated[_LEFT_ELBOW], gated[_LEFT_SHOULDER], gated[_LEFT_WRIST]
    )
    channels["elbow_angle_right"] = _joint_angle_deg(
        gated[_RIGHT_ELBOW], gated[_RIGHT_SHOULDER], gated[_RIGHT_WRIST]
    )
    channels["knee_angle_left"] = _joint_angle_deg(
        gated[_LEFT_KNEE], gated[_LEFT_HIP], gated[_LEFT_ANKLE]
    )
    channels["knee_angle_right"] = _joint_angle_deg(
        gated[_RIGHT_KNEE], gated[_RIGHT_HIP], gated[_RIGHT_ANKLE]
    )
    _fill_axes(channels, gated)

    torso_mid: tuple[float, float] | None = None
    if shoulder_mid is not None and hip_mid is not None:
        torso_mid = (
            (shoulder_mid[0] + hip_mid[0]) / 2.0,
            (shoulder_mid[1] + hip_mid[1]) / 2.0,
        )
    elif shoulder_mid is not None:
        torso_mid = shoulder_mid
    elif hip_mid is not None:
        torso_mid = hip_mid
    if torso_mid is not None:
        channels["torso_mid_camera_x"] = (torso_mid[0] - origin[0]) / scale
        channels["torso_mid_camera_y"] = (torso_mid[1] - origin[1]) / scale
    return channels, quality


def _fill_axes(channels: dict[str, float | None], gated: list[Any | None]) -> None:
    left_shoulder = gated[_LEFT_SHOULDER]
    right_shoulder = gated[_RIGHT_SHOULDER]
    left_hip = gated[_LEFT_HIP]
    right_hip = gated[_RIGHT_HIP]
    if left_shoulder is not None and right_shoulder is not None:
        unit = _unit_vector(
            right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y
        )
        if unit is not None:
            channels["shoulder_axis_camera_dx"] = unit[0]
            channels["shoulder_axis_camera_dy"] = unit[1]
    if left_hip is not None and right_hip is not None:
        unit = _unit_vector(right_hip.x - left_hip.x, right_hip.y - left_hip.y)
        if unit is not None:
            channels["hip_axis_camera_dx"] = unit[0]
            channels["hip_axis_camera_dy"] = unit[1]
    shoulder_mid = _midpoint_coords(left_shoulder, right_shoulder)
    hip_mid = _midpoint_coords(left_hip, right_hip)
    if shoulder_mid is not None and hip_mid is not None:
        unit = _unit_vector(
            shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1]
        )
        if unit is not None:
            channels["torso_axis_camera_dx"] = unit[0]
            channels["torso_axis_camera_dy"] = unit[1]


# --- Local polynomial smoothing/derivatives ----------------------------------


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> tuple[float, float, float] | None:
    a11, a12, a13 = matrix[0]
    a21, a22, a23 = matrix[1]
    a31, a32, a33 = matrix[2]
    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    if not math.isfinite(det) or abs(det) < 1e-18:
        return None
    b1, b2, b3 = vector
    x = (
        b1 * (a22 * a33 - a23 * a32)
        - a12 * (b2 * a33 - a23 * b3)
        + a13 * (b2 * a32 - a22 * b3)
    ) / det
    y = (
        a11 * (b2 * a33 - a23 * b3)
        - b1 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * b3 - b2 * a31)
    ) / det
    z = (
        a11 * (a22 * b3 - b2 * a32)
        - a12 * (a21 * b3 - b2 * a31)
        + b1 * (a21 * a32 - a22 * a31)
    ) / det
    if not all(math.isfinite(value) for value in (x, y, z)):
        return None
    return (x, y, z)


def _quadratic_coefficients(
    dts: list[float], values: list[float]
) -> tuple[float, float, float] | None:
    n = len(dts)
    s0 = float(n)
    s1 = sum(dts)
    s2 = sum(dt * dt for dt in dts)
    s3 = sum(dt * dt * dt for dt in dts)
    s4 = sum(dt * dt * dt * dt for dt in dts)
    t0 = sum(values)
    t1 = sum(dt * value for dt, value in zip(dts, values))
    t2 = sum(dt * dt * value for dt, value in zip(dts, values))
    return _solve_3x3(
        [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]],
        [t0, t1, t2],
    )


def _window_bounds(index: int, count: int, window: int) -> list[int]:
    half = window // 2
    start = max(0, index - half)
    end = min(count, index + half + 1)
    return list(range(start, end))


def _cap_window(window: int, count: int) -> int:
    if count <= 0:
        return 1
    capped = min(window, count)
    if capped % 2 == 0:
        capped = max(1, capped - 1)
    return capped


def _smooth_series(
    values: list[float | None], times: list[float], window: int
) -> tuple[list[float | None], list[int]]:
    count = len(values)
    capped = _cap_window(window, count)
    smoothed: list[float | None] = [None] * count
    counts: list[int] = [0] * count
    for index in range(count):
        if values[index] is None:
            continue
        members = [
            j for j in _window_bounds(index, count, capped) if values[j] is not None
        ]
        counts[index] = len(members)
        if not members:
            smoothed[index] = None
        elif len(members) < 3:
            smoothed[index] = values[index]
        else:
            dts = [times[j] - times[index] for j in members]
            ys = [float(values[j]) for j in members]  # type: ignore[arg-type]
            coefficients = _quadratic_coefficients(dts, ys)
            if coefficients is None:
                smoothed[index] = values[index]
            else:
                smoothed[index] = coefficients[0]
    return smoothed, counts


def _derivative_series(
    values: list[float | None], times: list[float], window: int
) -> tuple[list[float | None], list[float | None], list[int]]:
    count = len(values)
    capped = _cap_window(window, count)
    velocities: list[float | None] = [None] * count
    accelerations: list[float | None] = [None] * count
    counts: list[int] = [0] * count
    for index in range(count):
        if values[index] is None:
            continue
        members = [
            j for j in _window_bounds(index, count, capped) if values[j] is not None
        ]
        counts[index] = len(members)
        if len(members) < 2:
            continue
        dts = [times[j] - times[index] for j in members]
        ys = [float(values[j]) for j in members]  # type: ignore[arg-type]
        if len(members) == 2:
            denom = dts[1] - dts[0]
            if math.isfinite(denom) and denom != 0.0:
                velocities[index] = (ys[1] - ys[0]) / denom
            continue
        coefficients = _quadratic_coefficients(dts, ys)
        if coefficients is None:
            # Fall back to a local secant across the valid span.
            denom = dts[-1] - dts[0]
            if math.isfinite(denom) and denom != 0.0:
                velocities[index] = (ys[-1] - ys[0]) / denom
            continue
        velocities[index] = coefficients[1]
        accelerations[index] = 2.0 * coefficients[2]
    return velocities, accelerations, counts


# --- Grid construction ---------------------------------------------------------


def build_phase_feature_grid(
    observations: Sequence[FrameObservation],
    attempt_range: MediaRange,
    *,
    audio_energies: Sequence[AudioEnergy] = (),
    audio_candidate_times: Sequence[float] = (),
    config: PhaseFeaturesConfig | None = None,
    source_frame_rate_hz: float = 120.0,
) -> PhaseFeatureGrid:
    """Build the uniform attempt-local phase feature grid.

    ``observations`` must carry strictly increasing canonical source
    times (existing sparse/dense pose output; never re-inferred here).
    Only rows with ``start <= t < end`` are used. ``audio_energies``
    must be strictly increasing when non-empty; ``audio_candidate_times``
    must be strictly increasing finite times ``>= 0`` when non-empty.
    """
    cfg = config if config is not None else PhaseFeaturesConfig()
    if not isinstance(cfg, PhaseFeaturesConfig):
        raise PhaseFeaturesError(
            "build_phase_feature_grid: 'config' must be a PhaseFeaturesConfig, "
            f"got {type(cfg).__name__}."
        )
    if not isinstance(attempt_range, MediaRange):
        raise PhaseFeaturesError(
            "build_phase_feature_grid: 'attempt_range' must be a MediaRange, "
            f"got {type(attempt_range).__name__}."
        )
    source_rate = _check_rate("'source_frame_rate_hz'", source_frame_rate_hz)

    items = list(observations)
    for entry in items:
        if not isinstance(entry, FrameObservation):
            raise PhaseFeaturesError(
                "build_phase_feature_grid: every observation must be a "
                f"FrameObservation, got {type(entry).__name__}."
            )
    for earlier, later in zip(items, items[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise PhaseFeaturesError(
                "build_phase_feature_grid: observation times must be strictly "
                f"increasing, got {earlier.time_seconds!r} followed by "
                f"{later.time_seconds!r}."
            )

    audio_list = list(audio_energies)
    for entry in audio_list:
        if not isinstance(entry, AudioEnergy):
            raise PhaseFeaturesError(
                "build_phase_feature_grid: every audio energy must be an "
                f"AudioEnergy, got {type(entry).__name__}."
            )
    for earlier, later in zip(audio_list, audio_list[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise PhaseFeaturesError(
                "build_phase_feature_grid: audio times must be strictly "
                "increasing."
            )
    candidates = list(audio_candidate_times)
    for entry in candidates:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise PhaseFeaturesError(
                "build_phase_feature_grid: every audio candidate time must be "
                f"a finite number >= 0, got {entry!r}."
            )
        if not math.isfinite(float(entry)) or float(entry) < 0.0:
            raise PhaseFeaturesError(
                "build_phase_feature_grid: every audio candidate time must be "
                f"a finite number >= 0, got {entry!r}."
            )
    candidate_times = tuple(float(entry) for entry in candidates)
    for earlier, later in zip(candidate_times, candidate_times[1:]):
        if not later > earlier:
            raise PhaseFeaturesError(
                "build_phase_feature_grid: audio candidate times must be "
                "strictly increasing."
            )

    sliced = [frame for frame in items if attempt_range.contains(frame.time_seconds)]
    sliced_channels: list[dict[str, float | None]] = []
    sliced_quality: list[float] = []
    sliced_has_person: list[bool] = []
    for frame in sliced:
        channels, quality = _extract_observation_channels(frame, cfg.visibility_floor)
        sliced_channels.append(channels)
        sliced_quality.append(quality)
        sliced_has_person.append(frame.has_person)
    sliced_times = [frame.time_seconds for frame in sliced]
    if len(sliced_times) >= 2:
        span = sliced_times[-1] - sliced_times[0]
        pose_rate = (len(sliced_times) - 1) / span if span > 0 else 0.0
    else:
        pose_rate = 0.0

    grid_rate = cfg.grid_rate_hz
    step = 1.0 / grid_rate
    start = attempt_range.start_seconds
    end = attempt_range.end_seconds
    grid_times: list[float] = []
    index = 0
    while True:
        moment = start + index * step
        if not moment < end - 1e-9:
            # Exclude the half-open end within tolerance while keeping
            # deterministic cadence from the attempt start.
            if moment < end and (end - moment) > 1e-9:
                grid_times.append(float(moment))
            break
        grid_times.append(float(moment))
        index += 1
        if index > 100000:
            raise PhaseFeaturesError(
                "build_phase_feature_grid: attempt range yields an excessive grid."
            )
    if not grid_times:
        grid_times = [float(start)]

    raw_values: dict[str, list[float | None]] = {
        key: [None] * len(grid_times) for key in _POSITION_CHANNELS
    }
    raw_audio: list[float | None] = [None] * len(grid_times)
    observed_flags: list[bool] = [False] * len(grid_times)
    qualities: list[float] = [0.0] * len(grid_times)
    spans: list[float] = [0.0] * len(grid_times)
    offsets: list[float] = [0.0] * len(grid_times)
    uncertainties: list[float] = [step / 2.0] * len(grid_times)
    candidate_flags: list[bool] = [False] * len(grid_times)

    audio_times = [entry.time_seconds for entry in audio_list]
    audio_values = [entry.energy for entry in audio_list]

    qualified_indices = [
        idx
        for idx, (has_person, quality) in enumerate(
            zip(sliced_has_person, sliced_quality)
        )
        if has_person and quality > 0.0
    ]
    qualified_times = tuple(sliced_times[idx] for idx in qualified_indices)
    effective_tolerance = resolve_direct_observation_tolerance_seconds(
        cfg, grid_rate_hz=grid_rate, qualified_source_times=qualified_times
    )
    # Deterministic bounded direct alignment: at most one qualified source
    # per grid point and at most one grid point per source. Grid order is
    # canonical; candidates sort by (|offset|, source time, source index).
    # The effective tolerance stays below half a step so conflicts are
    # structurally impossible, but consumption is still enforced.
    assigned_source: list[int] = [-1] * len(grid_times)
    consumed_sources: set[int] = set()
    for position, moment in enumerate(grid_times):
        best: list[tuple[float, float, int, float]] = []
        for qualified_pos, source_idx in enumerate(qualified_indices):
            if source_idx in consumed_sources:
                continue
            offset = sliced_times[source_idx] - moment
            distance = abs(offset)
            if distance <= effective_tolerance + _DIRECT_TOLERANCE_EPSILON_SECONDS:
                best.append((distance, sliced_times[source_idx], source_idx, offset))
        if best:
            best.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
            chosen = best[0]
            assigned_source[position] = chosen[2]
            consumed_sources.add(chosen[2])
    qualified_rank = {source_idx: rank for rank, source_idx in enumerate(qualified_indices)}

    for position, moment in enumerate(grid_times):
        source_idx = assigned_source[position]
        if source_idx >= 0:
            offset_value = sliced_times[source_idx] - moment
            observed_flags[position] = True
            qualities[position] = sliced_quality[source_idx]
            spans[position] = 0.0
            offsets[position] = offset_value
            rank = qualified_rank[source_idx]
            neighbor_gaps: list[float] = []
            if rank > 0:
                neighbor_gaps.append(
                    sliced_times[source_idx] - sliced_times[qualified_indices[rank - 1]]
                )
            if rank + 1 < len(qualified_indices):
                neighbor_gaps.append(
                    sliced_times[qualified_indices[rank + 1]] - sliced_times[source_idx]
                )
            local_half = step / 2.0
            for gap_value in neighbor_gaps:
                if math.isfinite(gap_value) and gap_value > 0.0:
                    local_half = max(local_half, gap_value / 2.0)
            uncertainties[position] = max(abs(offset_value), step / 2.0, local_half)
            for key in _POSITION_CHANNELS:
                raw_values[key][position] = sliced_channels[source_idx][key]
        else:
            prev_index = -1
            next_index = -1
            for candidate_index, obs_time in enumerate(sliced_times):
                if obs_time < moment - _EXACT_TOLERANCE_SECONDS:
                    prev_index = candidate_index
                elif obs_time > moment + _EXACT_TOLERANCE_SECONDS:
                    next_index = candidate_index
                    break
            if prev_index >= 0 and next_index >= 0:
                prev_time = sliced_times[prev_index]
                next_time = sliced_times[next_index]
                gap = next_time - prev_time
                if (
                    gap <= cfg.max_interpolation_gap_seconds + 1e-12
                    and sliced_has_person[prev_index]
                    and sliced_has_person[next_index]
                ):
                    alpha = (moment - prev_time) / gap if gap > 0 else 0.0
                    interpolated_any = False
                    for key in _POSITION_CHANNELS:
                        before = sliced_channels[prev_index][key]
                        after = sliced_channels[next_index][key]
                        if before is None or after is None:
                            continue
                        raw_values[key][position] = before + alpha * (after - before)
                        interpolated_any = True
                    if interpolated_any:
                        mean_quality = (
                            sliced_quality[prev_index] + sliced_quality[next_index]
                        ) / 2.0
                        # Interpolated samples never claim observed precision:
                        # strictly below both endpoints and below 1.0.
                        qualities[position] = min(
                            0.5 * mean_quality,
                            sliced_quality[prev_index] - 1e-9
                            if sliced_quality[prev_index] > 0 else 0.0,
                            sliced_quality[next_index] - 1e-9
                            if sliced_quality[next_index] > 0 else 0.0,
                        )
                        if qualities[position] < 0.0:
                            qualities[position] = 0.0
                        if qualities[position] >= 1.0:
                            qualities[position] = 0.999
                        spans[position] = gap
                        uncertainties[position] = max(gap / 2.0, step / 2.0)
                    observed_flags[position] = False
                    offsets[position] = 0.0
                else:
                    observed_flags[position] = False
                    qualities[position] = 0.0
                    spans[position] = 0.0
                    offsets[position] = 0.0
                    uncertainties[position] = step / 2.0
            else:
                observed_flags[position] = False
                qualities[position] = 0.0
                spans[position] = 0.0
                offsets[position] = 0.0
                uncertainties[position] = step / 2.0

        # Aligned audio energy: exact echo else short-span linear blend.
        if audio_list:
            audio_exact: float | None = None
            for audio_index, audio_time in enumerate(audio_times):
                if abs(audio_time - moment) <= _EXACT_TOLERANCE_SECONDS:
                    audio_exact = audio_values[audio_index]
                    break
            if audio_exact is not None:
                raw_audio[position] = audio_exact
            else:
                prev_audio = -1
                next_audio = -1
                for audio_index, audio_time in enumerate(audio_times):
                    if audio_time < moment - _EXACT_TOLERANCE_SECONDS:
                        prev_audio = audio_index
                    elif audio_time > moment + _EXACT_TOLERANCE_SECONDS:
                        next_audio = audio_index
                        break
                if prev_audio >= 0 and next_audio >= 0:
                    audio_gap = audio_times[next_audio] - audio_times[prev_audio]
                    if audio_gap <= cfg.max_interpolation_gap_seconds + 1e-12 and audio_gap > 0:
                        beta = (moment - audio_times[prev_audio]) / audio_gap
                        raw_audio[position] = audio_values[prev_audio] + beta * (
                            audio_values[next_audio] - audio_values[prev_audio]
                        )
        # Candidate flag: a transient candidate within half a grid step.
        for candidate_time in candidate_times:
            if abs(candidate_time - moment) <= step / 2.0 + 1e-9:
                candidate_flags[position] = True
                break

    position_window = cfg.position_window_samples()
    derivative_window = cfg.derivative_window_samples()

    smoothed_values: dict[str, list[float | None]] = {}
    for key in _POSITION_CHANNELS:
        smoothed, _ = _smooth_series(raw_values[key], grid_times, position_window)
        # Renormalize axis unit vectors after independent smoothing.
        smoothed_values[key] = smoothed
    for pair in (
        ("shoulder_axis_camera_dx", "shoulder_axis_camera_dy"),
        ("hip_axis_camera_dx", "hip_axis_camera_dy"),
        ("torso_axis_camera_dx", "torso_axis_camera_dy"),
    ):
        for position in range(len(grid_times)):
            dx = smoothed_values[pair[0]][position]
            dy = smoothed_values[pair[1]][position]
            if dx is None or dy is None:
                smoothed_values[pair[0]][position] = None
                smoothed_values[pair[1]][position] = None
                continue
            unit = _unit_vector(dx, dy)
            if unit is None:
                smoothed_values[pair[0]][position] = None
                smoothed_values[pair[1]][position] = None
            else:
                smoothed_values[pair[0]][position] = unit[0]
                smoothed_values[pair[1]][position] = unit[1]

    derivative_vel: dict[str, list[float | None]] = {}
    derivative_acc: dict[str, list[float | None]] = {}
    derivative_counts: dict[str, list[int]] = {}
    for key in _DERIVATIVE_POSITION_KEYS + _DERIVATIVE_ANGLE_KEYS:
        vel, acc, counts = _derivative_series(raw_values[key], grid_times, derivative_window)
        derivative_vel[key] = vel
        derivative_acc[key] = acc
        derivative_counts[key] = counts

    # Per-sample derivative confidence from window fill, averaged over
    # derivative channels; quality folds in observation quality.
    derivative_window_capped = _cap_window(derivative_window, len(grid_times))
    confidences: list[float] = []
    for position in range(len(grid_times)):
        total = sum(derivative_counts[key][position] for key in derivative_counts)
        average = total / len(derivative_counts) if derivative_counts else 0.0
        confidence = average / derivative_window_capped if derivative_window_capped else 0.0
        confidence = max(0.0, min(1.0, confidence))
        if raw_values["wrist_left_x"][position] is None and all(
            raw_values[key][position] is None for key in _DERIVATIVE_ANGLE_KEYS
        ):
            # No differenti able support at all: no derivative confidence.
            has_any = any(
                derivative_vel[key][position] is not None for key in derivative_vel
            )
            if not has_any:
                confidence = 0.0
        confidences.append(confidence)

    # Torso displacement/rotation camera proxies derived from smoothed
    # torso midpoint/axis so they track the denoised shape honestly.
    anchor: tuple[float, float] | None = None
    for position in range(len(grid_times)):
        mid_x = smoothed_values["torso_mid_camera_x"][position]
        mid_y = smoothed_values["torso_mid_camera_y"][position]
        if mid_x is not None and mid_y is not None:
            anchor = (mid_x, mid_y)
            break
    torso_displacement: list[float | None] = [None] * len(grid_times)
    torso_rotation: list[float | None] = [None] * len(grid_times)
    for position in range(len(grid_times)):
        mid_x = smoothed_values["torso_mid_camera_x"][position]
        mid_y = smoothed_values["torso_mid_camera_y"][position]
        if mid_x is not None and mid_y is not None and anchor is not None:
            torso_displacement[position] = math.hypot(mid_x - anchor[0], mid_y - anchor[1])
        axis_dx = smoothed_values["torso_axis_camera_dx"][position]
        axis_dy = smoothed_values["torso_axis_camera_dy"][position]
        if axis_dx is not None and axis_dy is not None:
            torso_rotation[position] = math.degrees(math.atan2(axis_dx, -axis_dy))

    samples: list[PhaseFeatureSample] = []
    for position, moment in enumerate(grid_times):
        confidence = confidences[position]
        quality = qualities[position]
        derivative_quality = confidence * quality
        # Missing samples carry zero derivative confidence/quality.
        if quality == 0.0 and not observed_flags[position]:
            all_missing = all(
                smoothed_values[key][position] is None for key in _POSITION_CHANNELS
            )
            if all_missing:
                confidence = 0.0
                derivative_quality = 0.0
        # Interpolated samples keep smoothing/derivatives but quality is
        # already reduced above; observed precision is never exceeded.
        samples.append(
            PhaseFeatureSample(
                time_seconds=moment,
                observed=observed_flags[position],
                observation_quality=min(0.999, quality)
                if not observed_flags[position] and quality >= 1.0
                else quality,
                interpolation_span_seconds=spans[position],
                source_time_offset_seconds=offsets[position],
                temporal_uncertainty_seconds=uncertainties[position],
                derivative_confidence=confidence,
                derivative_quality=derivative_quality,
                wrist_left_x=smoothed_values["wrist_left_x"][position],
                wrist_left_y=smoothed_values["wrist_left_y"][position],
                wrist_right_x=smoothed_values["wrist_right_x"][position],
                wrist_right_y=smoothed_values["wrist_right_y"][position],
                elbow_left_x=smoothed_values["elbow_left_x"][position],
                elbow_left_y=smoothed_values["elbow_left_y"][position],
                elbow_right_x=smoothed_values["elbow_right_x"][position],
                elbow_right_y=smoothed_values["elbow_right_y"][position],
                shoulder_left_x=smoothed_values["shoulder_left_x"][position],
                shoulder_left_y=smoothed_values["shoulder_left_y"][position],
                shoulder_right_x=smoothed_values["shoulder_right_x"][position],
                shoulder_right_y=smoothed_values["shoulder_right_y"][position],
                hip_left_x=smoothed_values["hip_left_x"][position],
                hip_left_y=smoothed_values["hip_left_y"][position],
                hip_right_x=smoothed_values["hip_right_x"][position],
                hip_right_y=smoothed_values["hip_right_y"][position],
                knee_left_x=smoothed_values["knee_left_x"][position],
                knee_left_y=smoothed_values["knee_left_y"][position],
                knee_right_x=smoothed_values["knee_right_x"][position],
                knee_right_y=smoothed_values["knee_right_y"][position],
                ankle_left_x=smoothed_values["ankle_left_x"][position],
                ankle_left_y=smoothed_values["ankle_left_y"][position],
                ankle_right_x=smoothed_values["ankle_right_x"][position],
                ankle_right_y=smoothed_values["ankle_right_y"][position],
                elbow_angle_left=smoothed_values["elbow_angle_left"][position],
                elbow_angle_right=smoothed_values["elbow_angle_right"][position],
                knee_angle_left=smoothed_values["knee_angle_left"][position],
                knee_angle_right=smoothed_values["knee_angle_right"][position],
                shoulder_axis_camera_dx=smoothed_values["shoulder_axis_camera_dx"][position],
                shoulder_axis_camera_dy=smoothed_values["shoulder_axis_camera_dy"][position],
                hip_axis_camera_dx=smoothed_values["hip_axis_camera_dx"][position],
                hip_axis_camera_dy=smoothed_values["hip_axis_camera_dy"][position],
                torso_axis_camera_dx=smoothed_values["torso_axis_camera_dx"][position],
                torso_axis_camera_dy=smoothed_values["torso_axis_camera_dy"][position],
                torso_mid_camera_x=smoothed_values["torso_mid_camera_x"][position],
                torso_mid_camera_y=smoothed_values["torso_mid_camera_y"][position],
                torso_displacement_camera=torso_displacement[position],
                torso_rotation_camera_deg=torso_rotation[position],
                audio_energy=raw_audio[position],
                audio_candidate=candidate_flags[position],
                wrist_left_vx=derivative_vel["wrist_left_x"][position],
                wrist_left_vy=derivative_vel["wrist_left_y"][position],
                wrist_right_vx=derivative_vel["wrist_right_x"][position],
                wrist_right_vy=derivative_vel["wrist_right_y"][position],
                elbow_angle_left_vel=derivative_vel["elbow_angle_left"][position],
                elbow_angle_right_vel=derivative_vel["elbow_angle_right"][position],
                knee_angle_left_vel=derivative_vel["knee_angle_left"][position],
                knee_angle_right_vel=derivative_vel["knee_angle_right"][position],
                wrist_left_ax=derivative_acc["wrist_left_x"][position],
                wrist_left_ay=derivative_acc["wrist_left_y"][position],
                wrist_right_ax=derivative_acc["wrist_right_x"][position],
                wrist_right_ay=derivative_acc["wrist_right_y"][position],
                elbow_angle_left_acc=derivative_acc["elbow_angle_left"][position],
                elbow_angle_right_acc=derivative_acc["elbow_angle_right"][position],
                knee_angle_left_acc=derivative_acc["knee_angle_left"][position],
                knee_angle_right_acc=derivative_acc["knee_angle_right"][position],
            )
        )

    # Freeze samples without aliasing caller structures.
    frozen_samples = tuple(samples)
    return PhaseFeatureGrid(
        attempt_range=attempt_range,
        method_version=PHASE_FEATURES_METHOD_VERSION,
        config_id=cfg.config_id,
        source_frame_rate_hz=source_rate,
        pose_observation_rate_hz=pose_rate,
        grid_rate_hz=grid_rate,
        position_window_samples=_cap_window(position_window, len(grid_times)),
        derivative_window_samples=derivative_window_capped,
        direct_observation_tolerance_seconds=effective_tolerance,
        samples=frozen_samples,
    )
