"""Deterministic 3D kinematic waveform matrix (M4.9).

Pure, deterministic conversion of an M4.8
:class:`~serve_review.checkpoints.world_filter.FilteredWorldTrack` into an
immutable, versioned, timestamped :class:`KinematicWaveformTrack`. No
media decoding, no cache I/O, no audio extraction, no candidate scoring,
no solver/DP, no CLI.

Input contract:

- Exactly one M4.8 ``FilteredWorldTrack``. Only its primary filtered 3D
  coordinates (``sample.filtered``) plus its per-joint
  availability/filter-confidence/observed-support metadata are
  consumed. Raw unfiltered rows and 2D companions are never read here.
  Optional aligned audio (``audio_energies``/``audio``) is the sole audio
  input: exactly one entry per filtered-track PTS in order, never decoded
  or demuxed here. ``None`` (or no audio at all) leaves both audio
  channels honestly unavailable, never zero-filled.
- A filtered joint is usable only when ``sample.available[j]`` is True
  and ``sample.filtered[j]`` is not ``None``. Anything else is missing
  and stays missing; no zero-fill, no bridging, no fabrication.

Versioned conventions (pinned in :class:`KinematicWaveformsConfig`):

- Coordinate convention
  (``COORDINATE_CONVENTION_VERSION =
  "mediapipe-world-hip-centered-v1"``): input axes are MediaPipe
  ``pose_world_landmarks`` meters as stored by M4.7/M4.8
  (hip-centered, finite ``x``/``y``/``z``). Axes are preserved verbatim
  with no mirroring, rotation, or re-centering. ``+y`` is documented as
  superior (up); ``+x``/``+z`` keep their stored MediaPipe sense and
  only enter documented differences, distances, and transverse (``x/z``)
  projections, so no handedness or viewpoint inference is required.
- Normalization (``NORMALIZATION_VERSION =
  "torso-length-normalization-v1"``): every scalar distance, relative
  component, speed, acceleration, rise, elevation, and extension
  channel is dimensionless body lengths. The per-sample reference is
  ``body_length_m = |shoulder_mid - hip_mid|`` in meters (3D Euclidean
  distance between the shoulder midpoint and the hip/pelvis midpoint).
  Samples whose reference is missing, non-finite, or smaller than
  ``min_body_length_m`` leave every normalized channel honestly
  unavailable (``None``); no fallback length and no cross-sample
  borrowing are ever used. Angles, angular velocities, and binary
  turning-point indicators are not length-normalized (degrees,
  degrees/second, and unitless flags respectively).
- Derivative method (``DERIVATIVE_METHOD_VERSION =
  "centered-nonuniform-pts-v1"``): first derivatives use exact-PTS
  finite differences on the qualified series only. Interior samples of
  a contiguous qualified run use the centered nonuniform form
  ``(v[i+1] - v[i-1]) / (t[i+1] - t[i-1])`` (component-wise for
  vectors). Run-boundary samples (including overall track boundaries)
  use a one-sided difference to the single qualified neighbor,
  ``(v[i+1] - v[i]) / (t[i+1] - t[i])`` or
  ``(v[i] - v[i-1]) / (t[i] - t[i-1])``, only when that neighbor is
  qualified. Interior samples with any stencil neighbor missing, and
  isolated single-sample runs, are honestly unavailable. No uniform-FPS
  assumption is made; irregular (VFR-style) PTS spacing flows through
  the exact denominators. Acceleration is the same operator applied to
  the velocity series (two-stage finite differences), so it requires a
  wider qualified neighborhood.

Angles (numerically safe, documented):

- Interior joint angles use ``acos(clamp(dot, -1, 1))`` in degrees, so
  the output always lies in ``[0, 180]`` for valid input. Zero-length
  limbs (norm ``<= 0`` or non-finite) yield missing, never ``NaN``.
- Flexion channels are ``flexion_deg = 180 - interior_deg``: ``0``
  means full extension (straight limb) and larger values mean deeper
  flexion. A right angle reads ``90`` under both conventions.
- Tilt channels are ``asin(clamp(dy / length, -1, 1))`` in degrees over
  ``[-90, 90]``: ``0`` is level, positive means the left joint is
  higher (``+y`` superior) than the right joint.
- Transverse shoulder-hip separation is the unsigned angle in degrees
  over ``[0, 180]`` between the shoulder-axis and hip-axis projections
  onto the transverse (``x/z``) plane via ``acos(clamp(cos, -1, 1))``.
  Degenerate projections (norm ``<= 0`` or non-finite) yield missing.

Missingness and quality propagation:

- A base channel is available only when every anatomically required
  filtered joint (plus the body-length reference, when normalized) is
  available at that sample. Otherwise its value is ``None`` with
  ``available == False`` and ``quality == 0.0``.
- Base channel quality is ``min_j min(observation_quality_j,
  filter_confidence_j)`` over required joints (``1.0`` interior
  observed, ``0.5`` edge/interpolated, ``0.0`` unavailable), so any
  weak support honestly caps the derived channel.
- Base channel temporal uncertainty is ``max_j`` of the supporting
  filtered samples' ``temporal_uncertainty_seconds``.
- Derivative/turning/settling channels combine the stencil members the
  same way (quality ``min``, uncertainty ``max``) and are available
  only when the full stencil is qualified.

Channel inventory (see :data:`CHANNEL_NAMES`, 38 channels):

- Bilateral knee/elbow flexion angles (deg) plus timestamp-aware first
  derivatives (deg/s).
- Shoulder-line and hip-line tilt (deg); transverse shoulder-hip
  separation (deg); torso and hip rise (body lengths, body-relative
  vertical extents, not absolute court height, because MediaPipe world
  is hip-centered per frame).
- Left-arm elevation (body lengths, wrist-minus-shoulder vertical) and
  extension (body lengths, shoulder-wrist reach).
- Right wrist relative to right shoulder, right elbow, torso midpoint,
  and pelvis midpoint: normalized ``dx``/``dy``/``dz`` components plus
  normalized distance (body lengths each).
- Right wrist speed (body lengths/s), acceleration (body lengths/s^2),
  and velocity/acceleration turning-point indicators (strict local
  extremum flags ``1.0``/``0.0`` over a qualified triple, ``None``
  when the triple is not fully qualified).
- Whole-body post-contact settling proxy (velocity energy):
  mean squared normalized speed over 12 major joints
  (shoulders/elbows/wrists/hips/knees/ankles bilaterally),
  available only when all 12 velocities are qualified.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from serve_review.pose.schema import JOINT_INDEX
from serve_review.checkpoints.world_filter import FilteredWorldTrack

__all__ = [
    "KINEMATIC_WAVEFORMS_SCHEMA_VERSION",
    "KINEMATIC_WAVEFORMS_METHOD_VERSION",
    "COORDINATE_CONVENTION_VERSION",
    "NORMALIZATION_VERSION",
    "DERIVATIVE_METHOD_VERSION",
    "ANGLE_UNIT",
    "DEFAULT_MIN_BODY_LENGTH_M",
    "CHANNEL_NAMES",
    "CHANNEL_UNITS",
    "CHANNEL_INDEX",
    "NUM_CHANNELS",
    "SETTLING_JOINT_NAMES",
    "KinematicWaveformsError",
    "KinematicWaveformsConfig",
    "KinematicWaveformSample",
    "KinematicWaveformTrack",
    "build_kinematic_waveform_track",
]

#: Version of every kinematic-waveform schema in this module.
KINEMATIC_WAVEFORMS_SCHEMA_VERSION = 1
#: Method identity recorded on every waveform track.
KINEMATIC_WAVEFORMS_METHOD_VERSION = "kinematic-waveforms-v1"
#: Versioned coordinate-frame convention (axes preserved verbatim).
COORDINATE_CONVENTION_VERSION = "mediapipe-world-hip-centered-v1"
#: Versioned body-length normalization policy.
NORMALIZATION_VERSION = "torso-length-normalization-v1"
#: Versioned derivative policy (centered nonuniform PTS differences).
DERIVATIVE_METHOD_VERSION = "centered-nonuniform-pts-v1"
#: Unit of every angle channel.
ANGLE_UNIT = "degrees"
#: Minimum plausible torso reference length in meters.
DEFAULT_MIN_BODY_LENGTH_M = 1e-6

#: Ordered channel inventory (38 named scalar channels; 36 body-kinematic
#: plus 2 optional per-PTS audio-transient channels appended at the end).
CHANNEL_NAMES: tuple[str, ...] = (
    "knee_flexion_left",
    "knee_flexion_right",
    "knee_flexion_velocity_left",
    "knee_flexion_velocity_right",
    "elbow_flexion_left",
    "elbow_flexion_right",
    "elbow_flexion_velocity_left",
    "elbow_flexion_velocity_right",
    "shoulder_tilt_deg",
    "hip_tilt_deg",
    "shoulder_hip_separation_transverse_deg",
    "torso_rise",
    "hip_rise",
    "left_arm_elevation",
    "left_arm_extension",
    "right_wrist_rel_shoulder_dx",
    "right_wrist_rel_shoulder_dy",
    "right_wrist_rel_shoulder_dz",
    "right_wrist_rel_shoulder_distance",
    "right_wrist_rel_elbow_dx",
    "right_wrist_rel_elbow_dy",
    "right_wrist_rel_elbow_dz",
    "right_wrist_rel_elbow_distance",
    "right_wrist_rel_torso_mid_dx",
    "right_wrist_rel_torso_mid_dy",
    "right_wrist_rel_torso_mid_dz",
    "right_wrist_rel_torso_mid_distance",
    "right_wrist_rel_pelvis_mid_dx",
    "right_wrist_rel_pelvis_mid_dy",
    "right_wrist_rel_pelvis_mid_dz",
    "right_wrist_rel_pelvis_mid_distance",
    "right_wrist_speed",
    "right_wrist_acceleration",
    "right_wrist_speed_turning",
    "right_wrist_accel_turning",
    "whole_body_settling_energy",
    "audio_transient_energy",
    "audio_transient_flag",
)

#: Channel name to position lookup.
CHANNEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(CHANNEL_NAMES)}

#: Number of waveform channels.
NUM_CHANNELS = len(CHANNEL_NAMES)

_DEG_CHANNELS = frozenset(
    {
        "knee_flexion_left",
        "knee_flexion_right",
        "elbow_flexion_left",
        "elbow_flexion_right",
        "shoulder_hip_separation_transverse_deg",
    }
)
_TILT_CHANNELS = frozenset({"shoulder_tilt_deg", "hip_tilt_deg"})
_DEG_PER_SEC_CHANNELS = frozenset(
    {
        "knee_flexion_velocity_left",
        "knee_flexion_velocity_right",
        "elbow_flexion_velocity_left",
        "elbow_flexion_velocity_right",
    }
)
_BODY_LENGTH_CHANNELS = frozenset(
    {
        "torso_rise",
        "hip_rise",
        "left_arm_elevation",
        "left_arm_extension",
        "right_wrist_rel_shoulder_dx",
        "right_wrist_rel_shoulder_dy",
        "right_wrist_rel_shoulder_dz",
        "right_wrist_rel_shoulder_distance",
        "right_wrist_rel_elbow_dx",
        "right_wrist_rel_elbow_dy",
        "right_wrist_rel_elbow_dz",
        "right_wrist_rel_elbow_distance",
        "right_wrist_rel_torso_mid_dx",
        "right_wrist_rel_torso_mid_dy",
        "right_wrist_rel_torso_mid_dz",
        "right_wrist_rel_torso_mid_distance",
        "right_wrist_rel_pelvis_mid_dx",
        "right_wrist_rel_pelvis_mid_dy",
        "right_wrist_rel_pelvis_mid_dz",
        "right_wrist_rel_pelvis_mid_distance",
    }
)
_BODY_LENGTH_PER_SEC_CHANNELS = frozenset({"right_wrist_speed"})
_BODY_LENGTH_PER_SEC2_CHANNELS = frozenset({"right_wrist_acceleration"})
_ENERGY_CHANNELS = frozenset({"whole_body_settling_energy"})
_FLAG_CHANNELS = frozenset({"right_wrist_speed_turning", "right_wrist_accel_turning", "audio_transient_flag"})
#: Band-limited RMS audio transient energy channels (non-negative, full-scale
#: RMS units; never body-length normalized). Honestly unavailable without audio.
_AUDIO_ENERGY_CHANNELS = frozenset({"audio_transient_energy"})

#: Human-readable units per channel (documented, versioned via config).
CHANNEL_UNITS: dict[str, str] = {
    name: (
        "degrees"
        if name in _DEG_CHANNELS
        else (
            "degrees/second"
            if name in _DEG_PER_SEC_CHANNELS
            else (
                "body_lengths"
                if name in _BODY_LENGTH_CHANNELS
                else (
                    "body_lengths/second"
                    if name in _BODY_LENGTH_PER_SEC_CHANNELS
                    else (
                        "body_lengths/second^2"
                        if name in _BODY_LENGTH_PER_SEC2_CHANNELS
                        else (
                            "(body_lengths/second)^2"
                            if name in _ENERGY_CHANNELS
                            else (
                                "rms_energy"
                                if name in _AUDIO_ENERGY_CHANNELS
                                else "flag(0/1)"
                            )
                        )
                    )
                )
            )
        )
    )
    for name in CHANNEL_NAMES
}

#: Joints feeding the whole-body settling proxy (bilateral major joints).
SETTLING_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

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

_SETTLING_INDICES: tuple[int, ...] = tuple(
    JOINT_INDEX[name] for name in SETTLING_JOINT_NAMES
)


class KinematicWaveformsError(ValueError):
    """Raised when waveform input, configuration, or codec is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _dumps_deterministic(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _loads_object(name: str, data: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KinematicWaveformsError(f"{name}: invalid UTF-8 JSON payload.") from exc
    if not isinstance(data, str):
        raise KinematicWaveformsError(
            f"{name}: JSON payload must be str or bytes, got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise KinematicWaveformsError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise KinematicWaveformsError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


def _check_schema_version(name: str, values: dict[str, Any]) -> None:
    if "schema_version" not in values:
        raise KinematicWaveformsError(f"{name}: missing required key 'schema_version'.")
    version = values["schema_version"]
    if not _is_int(version):
        raise KinematicWaveformsError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != KINEMATIC_WAVEFORMS_SCHEMA_VERSION:
        if version > KINEMATIC_WAVEFORMS_SCHEMA_VERSION:
            raise KinematicWaveformsError(
                f"{name}: unsupported newer schema_version {version!r}; this build "
                f"supports version {KINEMATIC_WAVEFORMS_SCHEMA_VERSION}."
            )
        raise KinematicWaveformsError(
            f"{name}: unsupported schema_version {version!r}; expected version "
            f"{KINEMATIC_WAVEFORMS_SCHEMA_VERSION}."
        )


@dataclass(frozen=True, slots=True)
class KinematicWaveformsConfig:
    """Immutable versioned configuration for the waveform matrix.

    The convention/normalization/derivative/angle fields must equal the
    module constants; any deviation is rejected so later leaves cannot
    silently reinterpret channels. ``min_body_length_m`` guards the
    torso-length reference against degenerate geometry.
    """

    config_id: str = "kinematic-waveforms-default-v1"
    coordinate_convention: str = COORDINATE_CONVENTION_VERSION
    normalization: str = NORMALIZATION_VERSION
    derivative_method: str = DERIVATIVE_METHOD_VERSION
    angle_unit: str = ANGLE_UNIT
    min_body_length_m: float = DEFAULT_MIN_BODY_LENGTH_M
    schema_version: int = KINEMATIC_WAVEFORMS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "kinematic_waveforms_config"
        if not _is_int(self.schema_version):
            raise KinematicWaveformsError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != KINEMATIC_WAVEFORMS_SCHEMA_VERSION:
            raise KinematicWaveformsError(
                f"{name}: unsupported schema_version {self.schema_version!r}; expected "
                f"{KINEMATIC_WAVEFORMS_SCHEMA_VERSION}."
            )
        for key, expected in (
            ("coordinate_convention", COORDINATE_CONVENTION_VERSION),
            ("normalization", NORMALIZATION_VERSION),
            ("derivative_method", DERIVATIVE_METHOD_VERSION),
            ("angle_unit", ANGLE_UNIT),
        ):
            value = getattr(self, key)
            if not isinstance(value, str) or value != expected:
                raise KinematicWaveformsError(
                    f"{name}: {key!r} must equal {expected!r}, got {value!r}."
                )
        length = self.min_body_length_m
        if (
            isinstance(length, bool)
            or not isinstance(length, (int, float))
            or not math.isfinite(float(length))
            or float(length) <= 0.0
        ):
            raise KinematicWaveformsError(
                f"{name}: 'min_body_length_m' must be a finite number > 0, got {length!r}."
            )
        object.__setattr__(self, "min_body_length_m", float(length))
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise KinematicWaveformsError(
                f"{name}: 'config_id' must be a non-blank string, got {self.config_id!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_unit": self.angle_unit,
            "config_id": self.config_id,
            "coordinate_convention": self.coordinate_convention,
            "derivative_method": self.derivative_method,
            "min_body_length_m": self.min_body_length_m,
            "normalization": self.normalization,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> KinematicWaveformsConfig:
        name = "kinematic_waveforms_config"
        if not isinstance(values, dict):
            raise KinematicWaveformsError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "angle_unit",
            "config_id",
            "coordinate_convention",
            "derivative_method",
            "min_body_length_m",
            "normalization",
            "schema_version",
        }
        missing = sorted(known - set(values))
        if missing:
            raise KinematicWaveformsError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise KinematicWaveformsError(f"{name}: unknown keys {unknown!r}.")
        try:
            _check_schema_version(name, values)
        except KinematicWaveformsError:
            raise
        try:
            return cls(
                config_id=values["config_id"],
                coordinate_convention=values["coordinate_convention"],
                normalization=values["normalization"],
                derivative_method=values["derivative_method"],
                angle_unit=values["angle_unit"],
                min_body_length_m=values["min_body_length_m"],
                schema_version=values["schema_version"],
            )
        except KinematicWaveformsError:
            raise
        except (TypeError, ValueError) as exc:
            raise KinematicWaveformsError(f"{name}: invalid configuration: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> KinematicWaveformsConfig:
        return cls.from_dict(_loads_object("kinematic_waveforms_config", data))


def _check_channel_values(name: str, values: Any) -> tuple[float | None, ...]:
    if not isinstance(values, list) or len(values) != NUM_CHANNELS:
        length = len(values) if isinstance(values, list) else "?"
        raise KinematicWaveformsError(
            f"{name}: 'channel_values' must be a JSON list of length {NUM_CHANNELS}, "
            f"got {length!r}."
        )
    normalized: list[float | None] = []
    for position, entry in enumerate(values):
        channel = CHANNEL_NAMES[position]
        if entry is None:
            normalized.append(None)
            continue
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise KinematicWaveformsError(
                f"{name}: channel {channel!r} must be a finite number or null, "
                f"got {entry!r}."
            )
        number = float(entry)
        if not math.isfinite(number):
            raise KinematicWaveformsError(
                f"{name}: channel {channel!r} must be finite or null, got {entry!r}."
            )
        if channel in _FLAG_CHANNELS and number not in (0.0, 1.0):
            raise KinematicWaveformsError(
                f"{name}: channel {channel!r} must be 0.0, 1.0, or null, "
                f"got {entry!r}."
            )
        if channel in _DEG_CHANNELS and (
            number < -1e-9 or number > 180.0 + 1e-9
        ):
            raise KinematicWaveformsError(
                f"{name}: channel {channel!r} must lie in [0, 180] or be null, "
                f"got {entry!r}."
            )
        if channel in _TILT_CHANNELS and (
            number < -90.0 - 1e-9 or number > 90.0 + 1e-9
        ):
            raise KinematicWaveformsError(
                f"{name}: channel {channel!r} must lie in [-90, 90] or be null, "
                f"got {entry!r}."
            )
        if channel in _AUDIO_ENERGY_CHANNELS and number < 0.0:
            raise KinematicWaveformsError(
                f"{name}: channel {channel!r} must be >= 0 or null, "
                f"got {entry!r}."
            )
        normalized.append(number)
    return tuple(normalized)


def _check_bool_channels(name: str, key: str, values: Any) -> tuple[bool, ...]:
    if not isinstance(values, list) or len(values) != NUM_CHANNELS:
        raise KinematicWaveformsError(
            f"{name}: {key!r} must be a JSON list of length {NUM_CHANNELS}."
        )
    normalized: list[bool] = []
    for entry in values:
        if not isinstance(entry, bool):
            raise KinematicWaveformsError(
                f"{name}: {key!r} entries must be booleans, got {entry!r}."
            )
        normalized.append(entry)
    return tuple(normalized)


def _check_quality_channels(name: str, key: str, values: Any) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != NUM_CHANNELS:
        raise KinematicWaveformsError(
            f"{name}: {key!r} must be a JSON list of length {NUM_CHANNELS}."
        )
    normalized: list[float] = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise KinematicWaveformsError(
                f"{name}: {key!r} entries must be numbers in [0, 1], got {entry!r}."
            )
        number = float(entry)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise KinematicWaveformsError(
                f"{name}: {key!r} entries must lie in [0, 1], got {entry!r}."
            )
        normalized.append(number)
    return tuple(normalized)


def _check_uncertainty_channels(name: str, key: str, values: Any) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != NUM_CHANNELS:
        raise KinematicWaveformsError(
            f"{name}: {key!r} must be a JSON list of length {NUM_CHANNELS}."
        )
    normalized: list[float] = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise KinematicWaveformsError(
                f"{name}: {key!r} entries must be finite numbers >= 0, got {entry!r}."
            )
        number = float(entry)
        if not math.isfinite(number) or number < 0.0:
            raise KinematicWaveformsError(
                f"{name}: {key!r} entries must be finite and >= 0, got {entry!r}."
            )
        normalized.append(number)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class KinematicWaveformSample:
    """One immutable timestamped waveform sample.

    ``time_seconds``/``timestamp_ms`` reproduce the M4.8 grid PTS
    verbatim (``timestamp_ms == round(time_seconds * 1000)``).
    ``channel_values`` holds one entry per :data:`CHANNEL_NAMES` in
    order (``None`` when honestly unavailable, never a zero fill);
    ``channel_available`` mirrors non-``None`` values exactly;
    ``channel_quality`` lies in ``[0, 1]`` (``0.0`` whenever
    unavailable); ``channel_uncertainty_seconds`` carries the
    per-channel temporal uncertainty (``max`` over the supporting
    filtered samples and derivative stencil members).
    """

    time_seconds: float = 0.0
    timestamp_ms: int = 0
    channel_values: tuple[float | None, ...] = ()
    channel_available: tuple[bool, ...] = ()
    channel_quality: tuple[float, ...] = ()
    channel_uncertainty_seconds: tuple[float, ...] = ()
    schema_version: int = KINEMATIC_WAVEFORMS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "kinematic_waveform_sample"
        if not _is_int(self.schema_version):
            raise KinematicWaveformsError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != KINEMATIC_WAVEFORMS_SCHEMA_VERSION:
            raise KinematicWaveformsError(
                f"{name}: unsupported schema_version {self.schema_version!r}; expected "
                f"{KINEMATIC_WAVEFORMS_SCHEMA_VERSION}."
            )
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise KinematicWaveformsError(
                f"{name}: 'time_seconds' must be a finite number >= 0, "
                f"got {self.time_seconds!r}."
            )
        object.__setattr__(self, "time_seconds", float(self.time_seconds))
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise KinematicWaveformsError(
                f"{name}: 'timestamp_ms' must be an integer >= 0, "
                f"got {self.timestamp_ms!r}."
            )
        if self.timestamp_ms != int(round(float(self.time_seconds) * 1000)):
            raise KinematicWaveformsError(
                f"{name}: 'timestamp_ms' ({self.timestamp_ms!r}) must equal "
                f"round(time_seconds * 1000) ({int(round(float(self.time_seconds) * 1000))!r})."
            )
        raw_values = self.channel_values
        if not isinstance(raw_values, (list, tuple)) or len(tuple(raw_values)) != NUM_CHANNELS:
            raise KinematicWaveformsError(
                f"{name}: 'channel_values' must hold exactly {NUM_CHANNELS} entries."
            )
        values = _check_channel_values(name, list(raw_values))
        object.__setattr__(self, "channel_values", values)
        available = _check_bool_channels(name, "'channel_available'", list(self.channel_available))
        object.__setattr__(self, "channel_available", available)
        quality = _check_quality_channels(
            name, "'channel_quality'", list(self.channel_quality)
        )
        object.__setattr__(self, "channel_quality", quality)
        uncertainty = _check_uncertainty_channels(
            name, "'channel_uncertainty_seconds'", list(self.channel_uncertainty_seconds)
        )
        object.__setattr__(self, "channel_uncertainty_seconds", uncertainty)
        for position, channel in enumerate(CHANNEL_NAMES):
            value = values[position]
            is_available = available[position]
            if is_available and value is None:
                raise KinematicWaveformsError(
                    f"{name}: channel {channel!r} is marked available but carries null; "
                    "missing support must stay missing."
                )
            if not is_available and value is not None:
                raise KinematicWaveformsError(
                    f"{name}: channel {channel!r} is marked unavailable but carries "
                    f"{value!r}; never fabricate missing values."
                )
            if not is_available and quality[position] != 0.0:
                raise KinematicWaveformsError(
                    f"{name}: channel {channel!r} is unavailable but carries nonzero "
                    "quality."
                )

    def value(self, channel: str) -> float | None:
        """Return the value for ``channel`` (``None`` when unavailable)."""
        try:
            return self.channel_values[CHANNEL_INDEX[channel]]
        except KeyError as exc:
            raise KinematicWaveformsError(
                f"kinematic_waveform_sample: unknown channel {channel!r}; expected one of "
                f"{list(CHANNEL_NAMES)!r}."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_available": list(self.channel_available),
            "channel_quality": list(self.channel_quality),
            "channel_uncertainty_seconds": list(self.channel_uncertainty_seconds),
            "channel_values": list(self.channel_values),
            "schema_version": self.schema_version,
            "time_seconds": self.time_seconds,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> KinematicWaveformSample:
        name = "kinematic_waveform_sample"
        if not isinstance(values, dict):
            raise KinematicWaveformsError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "channel_available",
            "channel_quality",
            "channel_uncertainty_seconds",
            "channel_values",
            "schema_version",
            "time_seconds",
            "timestamp_ms",
        }
        missing = sorted(known - set(values))
        if missing:
            raise KinematicWaveformsError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise KinematicWaveformsError(f"{name}: unknown keys {unknown!r}.")
        try:
            _check_schema_version(name, values)
        except KinematicWaveformsError:
            raise
        try:
            return cls(
                time_seconds=values["time_seconds"],
                timestamp_ms=values["timestamp_ms"],
                channel_values=tuple(values["channel_values"]),
                channel_available=tuple(values["channel_available"]),
                channel_quality=tuple(values["channel_quality"]),
                channel_uncertainty_seconds=tuple(
                    values["channel_uncertainty_seconds"]
                ),
                schema_version=values["schema_version"],
            )
        except KinematicWaveformsError:
            raise
        except (TypeError, ValueError) as exc:
            raise KinematicWaveformsError(f"{name}: invalid sample: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> KinematicWaveformSample:
        return cls.from_dict(_loads_object("kinematic_waveform_sample", data))


@dataclass(frozen=True, slots=True)
class KinematicWaveformTrack:
    """Immutable deterministic kinematic waveform track.

    ``samples`` follow the M4.8 PTS grid one-to-one (same length, same
    times, same order). ``config`` pins the versioned conventions;
    ``method_version``/``coordinate_convention``/``normalization``/
    ``derivative_method`` repeat the version strings for provenance.
    """

    config: KinematicWaveformsConfig = None  # type: ignore[assignment]
    method_version: str = KINEMATIC_WAVEFORMS_METHOD_VERSION
    coordinate_convention: str = COORDINATE_CONVENTION_VERSION
    normalization: str = NORMALIZATION_VERSION
    derivative_method: str = DERIVATIVE_METHOD_VERSION
    samples: tuple[KinematicWaveformSample, ...] = ()
    schema_version: int = KINEMATIC_WAVEFORMS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "kinematic_waveform_track"
        if not _is_int(self.schema_version):
            raise KinematicWaveformsError(
                f"{name}: 'schema_version' must be an integer, got {self.schema_version!r}."
            )
        if self.schema_version != KINEMATIC_WAVEFORMS_SCHEMA_VERSION:
            raise KinematicWaveformsError(
                f"{name}: unsupported schema_version {self.schema_version!r}; expected "
                f"{KINEMATIC_WAVEFORMS_SCHEMA_VERSION}."
            )
        if not isinstance(self.config, KinematicWaveformsConfig):
            raise KinematicWaveformsError(
                f"{name}: 'config' must be a KinematicWaveformsConfig, "
                f"got {type(self.config).__name__}."
            )
        for key, expected in (
            ("method_version", KINEMATIC_WAVEFORMS_METHOD_VERSION),
            ("coordinate_convention", COORDINATE_CONVENTION_VERSION),
            ("normalization", NORMALIZATION_VERSION),
            ("derivative_method", DERIVATIVE_METHOD_VERSION),
        ):
            value = getattr(self, key)
            if not isinstance(value, str) or value != expected:
                raise KinematicWaveformsError(
                    f"{name}: {key!r} must equal {expected!r}, got {value!r}."
                )
        raw = self.samples
        if not isinstance(raw, (list, tuple)) or not raw:
            raise KinematicWaveformsError(
                f"{name}: 'samples' must be a non-empty sequence of "
                "KinematicWaveformSample."
            )
        normalized = tuple(raw)
        for entry in normalized:
            if not isinstance(entry, KinematicWaveformSample):
                raise KinematicWaveformsError(
                    f"{name}: every sample must be a KinematicWaveformSample, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "samples", normalized)
        for earlier, later in zip(normalized, normalized[1:]):
            if not later.time_seconds > earlier.time_seconds:
                raise KinematicWaveformsError(
                    f"{name}: sample times must be strictly increasing, got "
                    f"{earlier.time_seconds!r} followed by {later.time_seconds!r}."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.samples)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.samples[index]

    def channel_series(self, channel: str) -> tuple[float | None, ...]:
        """Return the value series for ``channel`` in sample order."""
        if channel not in CHANNEL_INDEX:
            raise KinematicWaveformsError(
                f"kinematic_waveform_track: unknown channel {channel!r}; expected one of "
                f"{list(CHANNEL_NAMES)!r}."
            )
        position = CHANNEL_INDEX[channel]
        return tuple(sample.channel_values[position] for sample in self.samples)

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.config, KinematicWaveformsConfig)
        return {
            "channel_names": list(CHANNEL_NAMES),
            "config": self.config.to_dict(),
            "coordinate_convention": self.coordinate_convention,
            "derivative_method": self.derivative_method,
            "method_version": self.method_version,
            "normalization": self.normalization,
            "samples": [sample.to_dict() for sample in self.samples],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> KinematicWaveformTrack:
        name = "kinematic_waveform_track"
        if not isinstance(values, dict):
            raise KinematicWaveformsError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "channel_names",
            "config",
            "coordinate_convention",
            "derivative_method",
            "method_version",
            "normalization",
            "samples",
            "schema_version",
        }
        missing = sorted(known - set(values))
        if missing:
            raise KinematicWaveformsError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise KinematicWaveformsError(f"{name}: unknown keys {unknown!r}.")
        try:
            _check_schema_version(name, values)
        except KinematicWaveformsError:
            raise
        if list(values["channel_names"]) != list(CHANNEL_NAMES):
            raise KinematicWaveformsError(
                f"{name}: 'channel_names' must equal the versioned inventory "
                f"{list(CHANNEL_NAMES)!r}."
            )
        try:
            config = KinematicWaveformsConfig.from_dict(values["config"])
        except KinematicWaveformsError as exc:
            raise KinematicWaveformsError(f"{name}: invalid 'config': {exc}.") from exc
        raw_samples = values["samples"]
        if not isinstance(raw_samples, list) or not raw_samples:
            raise KinematicWaveformsError(
                f"{name}: 'samples' must be a non-empty JSON list."
            )
        try:
            samples = tuple(
                KinematicWaveformSample.from_dict(entry) for entry in raw_samples
            )
        except KinematicWaveformsError:
            raise
        except Exception as exc:
            raise KinematicWaveformsError(f"{name}: invalid sample: {exc}.") from exc
        try:
            return cls(
                config=config,
                method_version=values["method_version"],
                coordinate_convention=values["coordinate_convention"],
                normalization=values["normalization"],
                derivative_method=values["derivative_method"],
                samples=samples,
                schema_version=values["schema_version"],
            )
        except KinematicWaveformsError:
            raise
        except (TypeError, ValueError) as exc:
            raise KinematicWaveformsError(f"{name}: invalid track: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> KinematicWaveformTrack:
        return cls.from_dict(_loads_object("kinematic_waveform_track", data))


# --- Internal geometry -------------------------------------------------------


def _xyz_of(filtered: Any) -> tuple[float, float, float] | None:
    if filtered is None:
        return None
    try:
        values = (float(filtered.x), float(filtered.y), float(filtered.z))
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return values


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _interior_angle_deg(
    vertex: tuple[float, float, float],
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float | None:
    """Numerically safe interior angle in degrees over ``[0, 180]``.

    Returns ``None`` for degenerate (zero-length or non-finite) limbs
    instead of ``NaN``; the cosine is clamped to ``[-1, 1]`` before
    ``acos`` so near-collinear geometry cannot escape the range.
    """
    ax = (first[0] - vertex[0], first[1] - vertex[1], first[2] - vertex[2])
    bx = (second[0] - vertex[0], second[1] - vertex[1], second[2] - vertex[2])
    na = _norm(ax)
    nb = _norm(bx)
    if not (math.isfinite(na) and math.isfinite(nb)) or na <= 0.0 or nb <= 0.0:
        return None
    cosine = _clamp(_dot(ax, bx) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _tilt_deg(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float | None:
    """Line tilt in degrees over ``[-90, 90]`` (positive = left higher)."""
    dx = right[0] - left[0]
    dy = left[1] - right[1]
    dz = right[2] - left[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not math.isfinite(length) or length <= 0.0:
        return None
    if not math.isfinite(dy):
        return None
    return math.degrees(math.asin(_clamp(dy / length, -1.0, 1.0)))


def _transverse_separation_deg(
    shoulder_left: tuple[float, float, float],
    shoulder_right: tuple[float, float, float],
    hip_left: tuple[float, float, float],
    hip_right: tuple[float, float, float],
) -> float | None:
    """Unsigned transverse (x/z) shoulder-hip separation in degrees."""
    s = (shoulder_right[0] - shoulder_left[0], shoulder_right[2] - shoulder_left[2])
    h = (hip_right[0] - hip_left[0], hip_right[2] - hip_left[2])
    ns = math.hypot(s[0], s[1])
    nh = math.hypot(h[0], h[1])
    if not (math.isfinite(ns) and math.isfinite(nh)) or ns <= 0.0 or nh <= 0.0:
        return None
    cosine = _clamp((s[0] * h[0] + s[1] * h[1]) / (ns * nh), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _midpoint(
    a: tuple[float, float, float] | None, b: tuple[float, float, float] | None
) -> tuple[float, float, float] | None:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _support_quality(
    track: FilteredWorldTrack, row: int, joints: Sequence[int]
) -> tuple[bool, float, float]:
    """Return ``(usable, quality, uncertainty)`` for ``joints`` at ``row``.

    ``usable`` requires every joint to carry a filtered value
    (``available`` and non-``None``). Quality is the minimum over
    ``min(observation_quality, filter_confidence)``; uncertainty is the
    maximum ``temporal_uncertainty_seconds`` of this row (per-sample in
    M4.8, shared across joints, so the row value is the honest bound).
    """
    sample = track.samples[row]
    for joint in joints:
        if not sample.available[joint] or sample.filtered[joint] is None:
            return (False, 0.0, float(sample.temporal_uncertainty_seconds))
    quality = min(
        min(float(sample.observation_quality[joint]), float(sample.filter_confidence[joint]))
        for joint in joints
    )
    return (True, float(quality), float(sample.temporal_uncertainty_seconds))


def _scalar_derivative_series(
    values: list[float | None], times: list[float]
) -> list[float | None]:
    """Centered nonuniform PTS first differences with qualified edges.

    Interior samples of each contiguous qualified run use
    ``(v[i+1] - v[i-1]) / (t[i+1] - t[i-1])``; run boundaries use the
    one-sided neighbor difference; isolated samples and interior gaps
    yield ``None``. Non-positive or non-finite time steps yield ``None``
    for that stencil rather than an unphysical rate.
    """
    count = len(values)
    out: list[float | None] = [None] * count
    if count == 0:
        return out
    run_start: int | None = None
    for i in range(count + 1):
        is_qualified = i < count and values[i] is not None
        if is_qualified and run_start is None:
            run_start = i
        if (not is_qualified or i == count) and run_start is not None:
            run_end = i - 1
            if run_end == run_start:
                out[run_start] = None
            elif run_end == run_start + 1:
                dt = times[run_end] - times[run_start]
                if (
                    math.isfinite(dt)
                    and dt > 0.0
                    and values[run_start] is not None
                    and values[run_end] is not None
                ):
                    assert values[run_start] is not None and values[run_end] is not None
                    rate = (float(values[run_end]) - float(values[run_start])) / dt
                    out[run_start] = rate if math.isfinite(rate) else None
                    out[run_end] = rate if math.isfinite(rate) else None
                else:
                    out[run_start] = None
                    out[run_end] = None
            else:
                # First sample: forward difference.
                dt_first = times[run_start + 1] - times[run_start]
                if (
                    math.isfinite(dt_first)
                    and dt_first > 0.0
                    and values[run_start] is not None
                    and values[run_start + 1] is not None
                ):
                    assert values[run_start] is not None and values[run_start + 1] is not None
                    rate = (float(values[run_start + 1]) - float(values[run_start])) / dt_first
                    out[run_start] = rate if math.isfinite(rate) else None
                # Interior: centered differences.
                for k in range(run_start + 1, run_end):
                    assert values[k - 1] is not None and values[k + 1] is not None
                    dt = times[k + 1] - times[k - 1]
                    if (
                        math.isfinite(dt)
                        and dt > 0.0
                        and values[k - 1] is not None
                        and values[k + 1] is not None
                    ):
                        rate = (float(values[k + 1]) - float(values[k - 1])) / dt
                        out[k] = rate if math.isfinite(rate) else None
                    else:
                        out[k] = None
                # Last sample: backward difference.
                dt_last = times[run_end] - times[run_end - 1]
                if (
                    math.isfinite(dt_last)
                    and dt_last > 0.0
                    and values[run_end] is not None
                    and values[run_end - 1] is not None
                ):
                    assert values[run_end] is not None and values[run_end - 1] is not None
                    rate = (float(values[run_end]) - float(values[run_end - 1])) / dt_last
                    out[run_end] = rate if math.isfinite(rate) else None
            run_start = None
    return out


def _vector_derivative_series(
    values: list[tuple[float, float, float] | None], times: list[float]
) -> list[tuple[float, float, float] | None]:
    """Component-wise :func:`_scalar_derivative_series` for 3D vectors."""
    series: list[list[float | None]] = [[None] * len(values) for _ in range(3)]
    for axis in range(3):
        component = [v[axis] if v is not None else None for v in values]
        derived = _scalar_derivative_series(component, times)
        series[axis] = derived
    out: list[tuple[float, float, float] | None] = []
    for i in range(len(values)):
        triple = (series[0][i], series[1][i], series[2][i])
        if triple[0] is None or triple[1] is None or triple[2] is None:
            out.append(None)
        else:
            assert triple[0] is not None and triple[1] is not None and triple[2] is not None
            candidate = (float(triple[0]), float(triple[1]), float(triple[2]))
            out.append(candidate if all(math.isfinite(v) for v in candidate) else None)
    return out


_AUDIO_UNSET: Any = object()


def _audio_energy_of(entry: Any) -> float | None:
    """Extract a finite non-negative energy from a duck-typed audio entry."""
    if isinstance(entry, bool):
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: audio energy entries must be finite "
            f"numbers >= 0 or null, got {entry!r}."
        )
    if isinstance(entry, (int, float)):
        number = float(entry)
        if not math.isfinite(number) or number < 0.0:
            raise KinematicWaveformsError(
                "build_kinematic_waveform_track: audio energy entries must be finite "
                f"numbers >= 0 or null, got {entry!r}."
            )
        return number
    energy = getattr(entry, "energy", None)
    if isinstance(energy, bool) or not isinstance(energy, (int, float)):
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: audio entries must be floats, null, "
            f"AudioEnergy-like, mappings, or (energy, flag) pairs; got {entry!r}."
        )
    number = float(energy)
    if not math.isfinite(number) or number < 0.0:
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: audio energy entries must be finite "
            f"numbers >= 0 or null, got {entry!r}."
        )
    return number


def _audio_flag_of(entry: Any) -> float | None:
    """Extract an explicit 0.0/1.0 flag from a duck-typed audio entry."""
    if isinstance(entry, bool):
        return 1.0 if entry else 0.0
    if isinstance(entry, (int, float)):
        number = float(entry)
        if number not in (0.0, 1.0):
            raise KinematicWaveformsError(
                "build_kinematic_waveform_track: audio flag entries must be 0.0, 1.0, "
                f"or null, got {entry!r}."
            )
        return number
    return None


def _normalize_audio_inputs(
    audio: Any, count: int, times: list[float]
) -> tuple[list[float | None], list[Any]]:
    """Normalize the optional aligned audio sequence to per-row energies/flags.

    Returns ``(energies, explicit_flags)`` where ``explicit_flags[row]`` is
    ``_AUDIO_UNSET`` when the caller supplied energy only (flag is then
    derived as a strict local energy maximum), otherwise ``None``/``0.0``/``1.0``.
    """
    energies: list[float | None] = [None] * count
    explicit: list[Any] = [_AUDIO_UNSET] * count
    if audio is None:
        return (energies, explicit)
    if isinstance(audio, (str, bytes, bytearray)) or not isinstance(audio, (list, tuple)):
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: 'audio_energies' must be None or a "
            f"list/tuple with one entry per filtered-track sample ({count}), "
            f"got {type(audio).__name__}."
        )
    rows = list(audio)
    if len(rows) != count:
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: 'audio_energies' must hold exactly one "
            f"entry per filtered-track sample ({count}), got {len(rows)}."
        )
    for row, entry in enumerate(rows):
        if entry is None:
            energies[row] = None
            explicit[row] = _AUDIO_UNSET
            continue
        if isinstance(entry, (list, tuple)) and len(entry) == 2 and not isinstance(
            entry, (str, bytes, bytearray)
        ):
            raw_energy, raw_flag = entry[0], entry[1]
            if raw_energy is None:
                energies[row] = None
            else:
                energies[row] = _audio_energy_of(raw_energy)
            if raw_flag is None:
                explicit[row] = None
            else:
                explicit[row] = _audio_flag_of(raw_flag)
            continue
        if isinstance(entry, dict):
            if "time_seconds" in entry:
                moment = entry["time_seconds"]
                if isinstance(moment, bool) or not isinstance(moment, (int, float)):
                    raise KinematicWaveformsError(
                        "build_kinematic_waveform_track: audio mapping 'time_seconds' "
                        f"must be a finite number, got {moment!r}."
                    )
                if not math.isfinite(float(moment)) or abs(float(moment) - times[row]) > 1e-9:
                    raise KinematicWaveformsError(
                        "build_kinematic_waveform_track: audio mapping time "
                        f"{moment!r} does not match filtered-track PTS {times[row]!r} "
                        f"at row {row}; audio must align one-for-one in order."
                    )
            raw_energy = None
            for key in ("energy", "audio_energy", "value"):
                if key in entry:
                    raw_energy = entry[key]
                    break
            else:
                raise KinematicWaveformsError(
                    "build_kinematic_waveform_track: audio mappings must carry an "
                    f"'energy' key, got {sorted(entry.keys())!r}."
                )
            if raw_energy is None:
                energies[row] = None
            else:
                energies[row] = _audio_energy_of(raw_energy)
            raw_flag = None
            flag_found = False
            for key in ("flag", "audio_flag", "audio_transient_flag", "candidate"):
                if key in entry:
                    raw_flag = entry[key]
                    flag_found = True
                    break
            if not flag_found:
                explicit[row] = _AUDIO_UNSET
            elif raw_flag is None:
                explicit[row] = None
            else:
                explicit[row] = _audio_flag_of(raw_flag)
            continue
        moment = getattr(entry, "time_seconds", None)
        if moment is not None:
            if isinstance(moment, bool) or not isinstance(moment, (int, float)):
                raise KinematicWaveformsError(
                    "build_kinematic_waveform_track: audio 'time_seconds' must be a "
                    f"finite number, got {moment!r}."
                )
            if not math.isfinite(float(moment)) or abs(float(moment) - times[row]) > 1e-9:
                raise KinematicWaveformsError(
                    "build_kinematic_waveform_track: audio time "
                    f"{moment!r} does not match filtered-track PTS {times[row]!r} "
                    f"at row {row}; audio must align one-for-one in order."
                )
            if hasattr(entry, "energy"):
                energies[row] = _audio_energy_of(entry)
                flag_attr = getattr(entry, "flag", getattr(entry, "audio_flag", _AUDIO_UNSET))
                if flag_attr is _AUDIO_UNSET:
                    explicit[row] = _AUDIO_UNSET
                elif flag_attr is None:
                    explicit[row] = None
                else:
                    explicit[row] = _audio_flag_of(flag_attr)
                continue
        if hasattr(entry, "energy"):
            energies[row] = _audio_energy_of(entry)
            explicit[row] = _AUDIO_UNSET
            continue
        energies[row] = _audio_energy_of(entry)
        explicit[row] = _AUDIO_UNSET
    return (energies, explicit)


def _derive_audio_flags(energies: list[float | None]) -> list[float | None]:
    """Derive transient flags as strict local energy maxima (peak-picking).

    Interior rows with a fully qualified triple carry ``1.0`` for a strict
    local maximum (louder than both neighbors) and ``0.0`` otherwise;
    edges, gaps, and isolated rows are honestly ``None`` (mirrors the
    wrist turning-point stencil policy without fabricating support).
    """
    count = len(energies)
    out: list[float | None] = [None] * count
    for row in range(count):
        center = energies[row]
        if center is None:
            continue
        if row == 0 or row == count - 1:
            continue
        prev = energies[row - 1]
        nxt = energies[row + 1]
        if prev is None or nxt is None:
            continue
        out[row] = 1.0 if (float(center) > float(prev) and float(center) > float(nxt)) else 0.0
    return out


def build_kinematic_waveform_track(
    track: FilteredWorldTrack,
    config: KinematicWaveformsConfig | None = None,
    audio_energies: Sequence[Any] | None = None,
    audio: Sequence[Any] | None = None,
) -> KinematicWaveformTrack:
    """Build a deterministic waveform track from an M4.8 filtered track.

    Args:
        track: Non-empty M4.8 :class:`FilteredWorldTrack`. Read-only;
            never mutated.
        config: Immutable waveform configuration (defaults pin the
            versioned coordinate/normalization/derivative conventions).
        audio_energies: Optional aligned audio sequence with exactly one
            entry per filtered-track sample in PTS order (``None`` when
            absent). Each entry is either ``None`` (honestly quiet/missing
            at that PTS), a finite non-negative energy ``float``, an
            ``AudioEnergy``-like object (``.time_seconds`` must match the
            filtered-track PTS within ``1e-9`` and ``.energy`` holds the
            RMS value), a mapping with ``energy`` (plus optional
            ``flag`` and optional ``time_seconds`` checked the same way),
            or an ``(energy, flag)`` pair carrying an explicit ``0.0``/``1.0``
            transient flag. ``audio`` is an accepted alias; supplying both
            with differing payloads raises.

    Returns:
        An immutable :class:`KinematicWaveformTrack` with one sample
        per M4.8 PTS, in the same order with verbatim times. The trailing
        ``audio_transient_energy`` channel stores the aligned RMS energy
        (``None`` without audio, never zero-filled) and
        ``audio_transient_flag`` stores the explicit flag when supplied,
        else the strict-local-maximum peak flag derived from the energy
        series (``None`` without audio or without a qualified triple).
        Both channels carry availability/quality/uncertainty like every
        body channel (quality ``1.0`` when available, ``0.0`` when missing;
        uncertainty is the honest M4.8 temporal bound, widened over the
        flag stencil when derived).

    Raises:
        KinematicWaveformsError: On wrong input/config types, an
            empty track, a misaligned audio sequence, or invalid audio
            values. Missing joints never raise; they propagate as
            honestly unavailable channels.
    """
    active = config if config is not None else KinematicWaveformsConfig()
    if not isinstance(active, KinematicWaveformsConfig):
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: 'config' must be a "
            f"KinematicWaveformsConfig, got {type(config).__name__}."
        )
    if not isinstance(track, FilteredWorldTrack):
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: 'track' must be a FilteredWorldTrack, "
            f"got {type(track).__name__}."
        )
    count = len(track.samples)
    if count == 0:
        raise KinematicWaveformsError(
            "build_kinematic_waveform_track: filtered track holds no samples; refusing "
            "to emit an empty waveform track."
        )
    times = [float(sample.time_seconds) for sample in track.samples]
    stamps = [int(sample.timestamp_ms) for sample in track.samples]
    if audio_energies is not None and audio is not None:
        if list(audio_energies) != list(audio):  # type: ignore[arg-type]
            raise KinematicWaveformsError(
                "build_kinematic_waveform_track: 'audio_energies' and alias 'audio' "
                "disagree; supply exactly one aligned audio sequence."
            )
        resolved_audio = audio_energies
    elif audio is not None:
        resolved_audio = audio
    else:
        resolved_audio = audio_energies
    audio_energy_series, audio_explicit_flags = _normalize_audio_inputs(
        resolved_audio, count, times
    )
    audio_derived_flags = _derive_audio_flags(audio_energy_series)

    min_length = float(active.min_body_length_m)

    # --- Per-sample joint positions and base channels -------------------------
    joints_xyz: list[dict[int, tuple[float, float, float]]] = []
    for row in range(count):
        sample = track.samples[row]
        mapping: dict[int, tuple[float, float, float]] = {}
        for joint in (
            _LEFT_SHOULDER,
            _RIGHT_SHOULDER,
            _LEFT_ELBOW,
            _RIGHT_ELBOW,
            _LEFT_WRIST,
            _RIGHT_WRIST,
            _LEFT_HIP,
            _RIGHT_HIP,
            _LEFT_KNEE,
            _RIGHT_KNEE,
            _LEFT_ANKLE,
            _RIGHT_ANKLE,
        ):
            if sample.available[joint] and sample.filtered[joint] is not None:
                point = _xyz_of(sample.filtered[joint])
                if point is not None:
                    mapping[joint] = point
        joints_xyz.append(mapping)

    def _get(row: int, joint: int) -> tuple[float, float, float] | None:
        return joints_xyz[row].get(joint)

    body_length: list[float | None] = []
    body_ok: list[bool] = []
    body_quality: list[float] = []
    body_uncertainty: list[float] = []
    for row in range(count):
        usable, quality, uncertainty = _support_quality(
            track, row, (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
        )
        shoulder_mid = _midpoint(_get(row, _LEFT_SHOULDER), _get(row, _RIGHT_SHOULDER))
        hip_mid = _midpoint(_get(row, _LEFT_HIP), _get(row, _RIGHT_HIP))
        length: float | None = None
        if usable and shoulder_mid is not None and hip_mid is not None:
            candidate = _norm(_sub(shoulder_mid, hip_mid))
            if math.isfinite(candidate) and candidate >= min_length:
                length = candidate
        body_length.append(length)
        body_ok.append(length is not None)
        body_quality.append(quality if length is not None else 0.0)
        body_uncertainty.append(uncertainty)

    base_values: dict[str, list[float | None]] = {
        name: [None] * count for name in CHANNEL_NAMES
    }
    base_quality: dict[str, list[float]] = {
        name: [0.0] * count for name in CHANNEL_NAMES
    }
    base_uncertainty: dict[str, list[float]] = {
        name: [0.0] * count for name in CHANNEL_NAMES
    }

    def _set_base(
        channel: str,
        row: int,
        value: float | None,
        joints: Sequence[int],
        *,
        needs_body: bool = False,
    ) -> None:
        if value is None or not math.isfinite(float(value)):
            base_values[channel][row] = None
            base_quality[channel][row] = 0.0
            base_uncertainty[channel][row] = float(track.samples[row].temporal_uncertainty_seconds)
            return
        usable, quality, uncertainty = _support_quality(track, row, joints)
        if not usable:
            base_values[channel][row] = None
            base_quality[channel][row] = 0.0
            base_uncertainty[channel][row] = float(
                track.samples[row].temporal_uncertainty_seconds
            )
            return
        if needs_body:
            if not body_ok[row]:
                base_values[channel][row] = None
                base_quality[channel][row] = 0.0
                base_uncertainty[channel][row] = float(
                    track.samples[row].temporal_uncertainty_seconds
                )
                return
            quality = min(quality, body_quality[row])
            uncertainty = max(uncertainty, body_uncertainty[row])
        base_values[channel][row] = float(value)
        base_quality[channel][row] = float(quality)
        base_uncertainty[channel][row] = float(uncertainty)

    for row in range(count):
        length = body_length[row]
        scale = length if length is not None else float("nan")

        # Knee flexion (vertex knee, limbs to hip and ankle).
        for side, knee, hip, ankle, chan in (
            ("left", _LEFT_KNEE, _LEFT_HIP, _LEFT_ANKLE, "knee_flexion_left"),
            ("right", _RIGHT_KNEE, _RIGHT_HIP, _RIGHT_ANKLE, "knee_flexion_right"),
        ):
            interior: float | None = None
            joints = (hip, knee, ankle)
            if all(_get(row, j) is not None for j in joints):
                assert _get(row, knee) is not None
                interior = _interior_angle_deg(
                    _get(row, knee) or (0.0, 0.0, 0.0),
                    _get(row, hip) or (0.0, 0.0, 0.0),
                    _get(row, ankle) or (0.0, 0.0, 0.0),
                )
            flexion = 180.0 - interior if interior is not None else None
            _set_base(chan, row, flexion, joints)

        # Elbow flexion (vertex elbow, limbs to shoulder and wrist).
        for elbow, shoulder, wrist, chan in (
            (_LEFT_ELBOW, _LEFT_SHOULDER, _LEFT_WRIST, "elbow_flexion_left"),
            (_RIGHT_ELBOW, _RIGHT_SHOULDER, _RIGHT_WRIST, "elbow_flexion_right"),
        ):
            joints = (shoulder, elbow, wrist)
            interior = None
            if all(_get(row, j) is not None for j in joints):
                interior = _interior_angle_deg(
                    _get(row, elbow) or (0.0, 0.0, 0.0),
                    _get(row, shoulder) or (0.0, 0.0, 0.0),
                    _get(row, wrist) or (0.0, 0.0, 0.0),
                )
            flexion = 180.0 - interior if interior is not None else None
            _set_base(chan, row, flexion, joints)

        # Tilts.
        shoulder_tilt: float | None = None
        if _get(row, _LEFT_SHOULDER) is not None and _get(row, _RIGHT_SHOULDER) is not None:
            shoulder_tilt = _tilt_deg(
                _get(row, _LEFT_SHOULDER) or (0.0, 0.0, 0.0),
                _get(row, _RIGHT_SHOULDER) or (0.0, 0.0, 0.0),
            )
        _set_base(
            "shoulder_tilt_deg", row, shoulder_tilt, (_LEFT_SHOULDER, _RIGHT_SHOULDER)
        )
        hip_tilt: float | None = None
        if _get(row, _LEFT_HIP) is not None and _get(row, _RIGHT_HIP) is not None:
            hip_tilt = _tilt_deg(
                _get(row, _LEFT_HIP) or (0.0, 0.0, 0.0),
                _get(row, _RIGHT_HIP) or (0.0, 0.0, 0.0),
            )
        _set_base("hip_tilt_deg", row, hip_tilt, (_LEFT_HIP, _RIGHT_HIP))

        # Transverse separation.
        separation: float | None = None
        if all(
            _get(row, j) is not None
            for j in (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)
        ):
            separation = _transverse_separation_deg(
                _get(row, _LEFT_SHOULDER) or (0.0, 0.0, 0.0),
                _get(row, _RIGHT_SHOULDER) or (0.0, 0.0, 0.0),
                _get(row, _LEFT_HIP) or (0.0, 0.0, 0.0),
                _get(row, _RIGHT_HIP) or (0.0, 0.0, 0.0),
            )
        _set_base(
            "shoulder_hip_separation_transverse_deg",
            row,
            separation,
            (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP),
        )

        # Torso / hip rise (body-relative vertical extents).
        shoulder_mid = _midpoint(_get(row, _LEFT_SHOULDER), _get(row, _RIGHT_SHOULDER))
        hip_mid = _midpoint(_get(row, _LEFT_HIP), _get(row, _RIGHT_HIP))
        ankle_mid = _midpoint(_get(row, _LEFT_ANKLE), _get(row, _RIGHT_ANKLE))
        torso_rise: float | None = None
        if shoulder_mid is not None and hip_mid is not None and length is not None:
            torso_rise = (shoulder_mid[1] - hip_mid[1]) / scale
        _set_base(
            "torso_rise",
            row,
            torso_rise,
            (_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP),
            needs_body=True,
        )
        hip_rise: float | None = None
        if hip_mid is not None and ankle_mid is not None and length is not None:
            hip_rise = (hip_mid[1] - ankle_mid[1]) / scale
        _set_base(
            "hip_rise",
            row,
            hip_rise,
            (_LEFT_HIP, _RIGHT_HIP, _LEFT_ANKLE, _RIGHT_ANKLE),
            needs_body=True,
        )

        # Left arm elevation / extension.
        elevation: float | None = None
        extension: float | None = None
        if (
            _get(row, _LEFT_WRIST) is not None
            and _get(row, _LEFT_SHOULDER) is not None
            and length is not None
        ):
            wrist = _get(row, _LEFT_WRIST) or (0.0, 0.0, 0.0)
            shoulder = _get(row, _LEFT_SHOULDER) or (0.0, 0.0, 0.0)
            elevation = (wrist[1] - shoulder[1]) / scale
            extension = _norm(_sub(wrist, shoulder)) / scale
        _set_base(
            "left_arm_elevation", row, elevation, (_LEFT_SHOULDER, _LEFT_WRIST),
            needs_body=True,
        )
        _set_base(
            "left_arm_extension", row, extension, (_LEFT_SHOULDER, _LEFT_WRIST),
            needs_body=True,
        )

        # Right-wrist relative geometry (normalized components + distance).
        wrist = _get(row, _RIGHT_WRIST)
        torso_mid = _midpoint(shoulder_mid, hip_mid)
        references: tuple[tuple[str, tuple[float, float, float] | None, tuple[int, ...]], ...] = (
            ("shoulder", _get(row, _RIGHT_SHOULDER), (_RIGHT_WRIST, _RIGHT_SHOULDER)),
            ("elbow", _get(row, _RIGHT_ELBOW), (_RIGHT_WRIST, _RIGHT_ELBOW)),
            ("torso_mid", torso_mid, (_RIGHT_WRIST, _LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP)),
            ("pelvis_mid", hip_mid, (_RIGHT_WRIST, _LEFT_HIP, _RIGHT_HIP)),
        )
        for label, ref, joints in references:
            vector: tuple[float, float, float] | None = None
            distance: float | None = None
            if wrist is not None and ref is not None and length is not None:
                diff = _sub(wrist, ref)
                vector = (diff[0] / scale, diff[1] / scale, diff[2] / scale)
                distance = _norm(diff) / scale
            for axis, suffix in ((0, "dx"), (1, "dy"), (2, "dz")):
                component = vector[axis] if vector is not None else None
                _set_base(
                    f"right_wrist_rel_{label}_{suffix}", row, component, joints,
                    needs_body=True,
                )
            _set_base(
                f"right_wrist_rel_{label}_distance", row, distance, joints,
                needs_body=True,
            )

    # --- Normalized wrist trajectory, velocity, acceleration -------------------
    wrist_norm: list[tuple[float, float, float] | None] = []
    wrist_norm_quality: list[float] = [0.0] * count
    wrist_norm_uncertainty: list[float] = [0.0] * count
    for row in range(count):
        usable, quality, uncertainty = _support_quality(track, row, (_RIGHT_WRIST,))
        point = _get(row, _RIGHT_WRIST)
        if usable and point is not None and body_ok[row]:
            assert body_length[row] is not None
            scale = float(body_length[row] or float("nan"))
            wrist_norm.append((point[0] / scale, point[1] / scale, point[2] / scale))
            wrist_norm_quality[row] = min(quality, body_quality[row])
            wrist_norm_uncertainty[row] = max(uncertainty, body_uncertainty[row])
        else:
            wrist_norm.append(None)
            wrist_norm_quality[row] = 0.0
            wrist_norm_uncertainty[row] = float(
                track.samples[row].temporal_uncertainty_seconds
            )

    wrist_vel = _vector_derivative_series(wrist_norm, times)
    wrist_vel_quality: list[float] = [0.0] * count
    wrist_vel_uncertainty: list[float] = [0.0] * count
    for i in range(count):
        if wrist_vel[i] is None:
            wrist_vel_quality[i] = 0.0
            wrist_vel_uncertainty[i] = float(
                track.samples[i].temporal_uncertainty_seconds
            )
            continue
        # Stencil actually used: centered triple when interior, pair at edges.
        if i > 0 and i < count - 1 and wrist_norm[i - 1] is not None and wrist_norm[i + 1] is not None:
            members = (i - 1, i, i + 1)
        elif i == 0 or wrist_norm[i - 1] is None:
            members = tuple(k for k in (i, i + 1) if 0 <= k < count and wrist_norm[k] is not None)
        else:
            members = tuple(k for k in (i - 1, i) if 0 <= k < count and wrist_norm[k] is not None)
        qualities = [wrist_norm_quality[k] for k in members]
        uncertainties = [
            max(wrist_norm_uncertainty[k], float(track.samples[k].temporal_uncertainty_seconds))
            for k in members
        ]
        wrist_vel_quality[i] = min(qualities) if qualities else 0.0
        wrist_vel_uncertainty[i] = max(uncertainties) if uncertainties else 0.0

    wrist_acc = _vector_derivative_series(wrist_vel, times)
    wrist_acc_quality: list[float] = [0.0] * count
    wrist_acc_uncertainty: list[float] = [0.0] * count
    for i in range(count):
        if wrist_acc[i] is None:
            wrist_acc_quality[i] = 0.0
            wrist_acc_uncertainty[i] = float(track.samples[i].temporal_uncertainty_seconds)
            continue
        if i > 0 and i < count - 1 and wrist_vel[i - 1] is not None and wrist_vel[i + 1] is not None:
            members = (i - 1, i, i + 1)
        elif i == 0 or wrist_vel[i - 1] is None:
            members = tuple(k for k in (i, i + 1) if 0 <= k < count and wrist_vel[k] is not None)
        else:
            members = tuple(k for k in (i - 1, i) if 0 <= k < count and wrist_vel[k] is not None)
        qualities = [wrist_vel_quality[k] for k in members]
        uncertainties = [wrist_vel_uncertainty[k] for k in members]
        wrist_acc_quality[i] = min(qualities) if qualities else 0.0
        wrist_acc_uncertainty[i] = max(uncertainties) if uncertainties else 0.0

    speed: list[float | None] = [None] * count
    for i in range(count):
        speed[i] = _norm(wrist_vel[i]) if wrist_vel[i] is not None else None
    accel: list[float | None] = [None] * count
    for i in range(count):
        accel[i] = _norm(wrist_acc[i]) if wrist_acc[i] is not None else None

    for row in range(count):
        if speed[row] is not None:
            base_values["right_wrist_speed"][row] = float(speed[row])
            base_quality["right_wrist_speed"][row] = float(wrist_vel_quality[row])
            base_uncertainty["right_wrist_speed"][row] = float(wrist_vel_uncertainty[row])
        if accel[row] is not None:
            base_values["right_wrist_acceleration"][row] = float(accel[row])
            base_quality["right_wrist_acceleration"][row] = float(wrist_acc_quality[row])
            base_uncertainty["right_wrist_acceleration"][row] = float(
                wrist_acc_uncertainty[row]
            )

    # Turning-point indicators: strict local extrema over qualified triples.
    for row in range(count):
        for series, vel_quality, vel_uncertainty, channel in (
            (speed, wrist_vel_quality, wrist_vel_uncertainty, "right_wrist_speed_turning"),
            (accel, wrist_acc_quality, wrist_acc_uncertainty, "right_wrist_accel_turning"),
        ):
            center = series[row]
            if (
                center is None
                or row == 0
                or row == count - 1
                or series[row - 1] is None
                or series[row + 1] is None
            ):
                continue
            prev = float(series[row - 1] or 0.0)
            curr = float(center)
            nxt = float(series[row + 1] or 0.0)
            is_peak = curr > prev and curr > nxt
            is_valley = curr < prev and curr < nxt
            base_values[channel][row] = 1.0 if (is_peak or is_valley) else 0.0
            base_quality[channel][row] = min(
                vel_quality[row - 1], vel_quality[row], vel_quality[row + 1]
            )
            base_uncertainty[channel][row] = max(
                vel_uncertainty[row - 1], vel_uncertainty[row], vel_uncertainty[row + 1]
            )

    # --- Angle velocities (same nonuniform operator on flexion series) ---------
    for flexion_channel, velocity_channel in (
        ("knee_flexion_left", "knee_flexion_velocity_left"),
        ("knee_flexion_right", "knee_flexion_velocity_right"),
        ("elbow_flexion_left", "elbow_flexion_velocity_left"),
        ("elbow_flexion_right", "elbow_flexion_velocity_right"),
    ):
        derived = _scalar_derivative_series(base_values[flexion_channel], times)
        for row in range(count):
            if derived[row] is None:
                continue
            if row > 0 and row < count - 1 and (
                base_values[flexion_channel][row - 1] is not None
                and base_values[flexion_channel][row + 1] is not None
            ):
                members = (row - 1, row, row + 1)
            elif row == 0 or base_values[flexion_channel][row - 1] is None:
                members = tuple(
                    k
                    for k in (row, row + 1)
                    if 0 <= k < count and base_values[flexion_channel][k] is not None
                )
            else:
                members = tuple(
                    k
                    for k in (row - 1, row)
                    if 0 <= k < count and base_values[flexion_channel][k] is not None
                )
            base_values[velocity_channel][row] = float(derived[row] or 0.0)
            base_quality[velocity_channel][row] = min(
                base_quality[flexion_channel][k] for k in members
            )
            base_uncertainty[velocity_channel][row] = max(
                base_uncertainty[flexion_channel][k] for k in members
            )

    # --- Whole-body settling proxy --------------------------------------------
    settling_norm: dict[int, list[tuple[float, float, float] | None]] = {}
    settling_quality: dict[int, list[float]] = {}
    settling_uncertainty: dict[int, list[float]] = {}
    for joint in _SETTLING_INDICES:
        series: list[tuple[float, float, float] | None] = []
        qualities: list[float] = []
        uncertainties: list[float] = []
        for row in range(count):
            usable, quality, uncertainty = _support_quality(track, row, (joint,))
            point = _get(row, joint)
            if usable and point is not None and body_ok[row]:
                assert body_length[row] is not None
                scale = float(body_length[row] or float("nan"))
                series.append((point[0] / scale, point[1] / scale, point[2] / scale))
                qualities.append(min(quality, body_quality[row]))
                uncertainties.append(max(uncertainty, body_uncertainty[row]))
            else:
                series.append(None)
                qualities.append(0.0)
                uncertainties.append(
                    float(track.samples[row].temporal_uncertainty_seconds)
                )
        settling_norm[joint] = series
        settling_quality[joint] = qualities
        settling_uncertainty[joint] = uncertainties

    settling_vel: dict[int, list[tuple[float, float, float] | None]] = {}
    settling_vel_quality: dict[int, list[float]] = {}
    settling_vel_uncertainty: dict[int, list[float]] = {}
    for joint in _SETTLING_INDICES:
        derived = _vector_derivative_series(settling_norm[joint], times)
        settling_vel[joint] = derived
        qualities: list[float] = [0.0] * count
        uncertainties: list[float] = [0.0] * count
        for i in range(count):
            if derived[i] is None:
                uncertainties[i] = float(track.samples[i].temporal_uncertainty_seconds)
                continue
            series = settling_norm[joint]
            if i > 0 and i < count - 1 and series[i - 1] is not None and series[i + 1] is not None:
                members = (i - 1, i, i + 1)
            elif i == 0 or series[i - 1] is None:
                members = tuple(
                    k for k in (i, i + 1) if 0 <= k < count and series[k] is not None
                )
            else:
                members = tuple(
                    k for k in (i - 1, i) if 0 <= k < count and series[k] is not None
                )
            qualities[i] = (
                min(settling_quality[joint][k] for k in members) if members else 0.0
            )
            uncertainties[i] = (
                max(
                    max(settling_uncertainty[joint][k], float(track.samples[k].temporal_uncertainty_seconds))
                    for k in members
                )
                if members
                else float(track.samples[i].temporal_uncertainty_seconds)
            )
        settling_vel_quality[joint] = qualities
        settling_vel_uncertainty[joint] = uncertainties

    for row in range(count):
        if all(settling_vel[joint][row] is not None for joint in _SETTLING_INDICES):
            energy = sum(
                _norm(settling_vel[joint][row] or (0.0, 0.0, 0.0)) ** 2
                for joint in _SETTLING_INDICES
            ) / len(_SETTLING_INDICES)
            if math.isfinite(energy):
                base_values["whole_body_settling_energy"][row] = float(energy)
                base_quality["whole_body_settling_energy"][row] = min(
                    settling_vel_quality[joint][row] for joint in _SETTLING_INDICES
                )
                base_uncertainty["whole_body_settling_energy"][row] = max(
                    settling_vel_uncertainty[joint][row] for joint in _SETTLING_INDICES
                )

    # --- Optional aligned audio-transient channels (never zero-filled) ----------
    for row in range(count):
        row_uncertainty = float(track.samples[row].temporal_uncertainty_seconds)
        energy = audio_energy_series[row]
        if energy is not None:
            base_values["audio_transient_energy"][row] = float(energy)
            base_quality["audio_transient_energy"][row] = 1.0
            base_uncertainty["audio_transient_energy"][row] = row_uncertainty
        else:
            base_values["audio_transient_energy"][row] = None
            base_quality["audio_transient_energy"][row] = 0.0
            base_uncertainty["audio_transient_energy"][row] = row_uncertainty
        marker = audio_explicit_flags[row]
        if marker is _AUDIO_UNSET:
            flag = audio_derived_flags[row]
            if flag is not None:
                base_values["audio_transient_flag"][row] = float(flag)
                base_quality["audio_transient_flag"][row] = 1.0
                base_uncertainty["audio_transient_flag"][row] = max(
                    float(track.samples[k].temporal_uncertainty_seconds)
                    for k in (row - 1, row, row + 1)
                    if 0 <= k < count
                )
            else:
                base_values["audio_transient_flag"][row] = None
                base_quality["audio_transient_flag"][row] = 0.0
                base_uncertainty["audio_transient_flag"][row] = row_uncertainty
        elif marker is None:
            base_values["audio_transient_flag"][row] = None
            base_quality["audio_transient_flag"][row] = 0.0
            base_uncertainty["audio_transient_flag"][row] = row_uncertainty
        else:
            base_values["audio_transient_flag"][row] = float(marker)
            base_quality["audio_transient_flag"][row] = 1.0
            base_uncertainty["audio_transient_flag"][row] = row_uncertainty

    # --- Assemble immutable samples --------------------------------------------
    samples: list[KinematicWaveformSample] = []
    for row in range(count):
        values = tuple(base_values[name][row] for name in CHANNEL_NAMES)
        available = tuple(value is not None for value in values)
        quality = tuple(
            float(base_quality[name][row]) if values[i] is not None else 0.0
            for i, name in enumerate(CHANNEL_NAMES)
        )
        uncertainty = tuple(float(base_uncertainty[name][row]) for name in CHANNEL_NAMES)
        samples.append(
            KinematicWaveformSample(
                time_seconds=times[row],
                timestamp_ms=stamps[row],
                channel_values=values,
                channel_available=available,
                channel_quality=quality,
                channel_uncertainty_seconds=uncertainty,
            )
        )

    return KinematicWaveformTrack(
        config=active,
        method_version=KINEMATIC_WAVEFORMS_METHOD_VERSION,
        coordinate_convention=COORDINATE_CONVENTION_VERSION,
        normalization=NORMALIZATION_VERSION,
        derivative_method=DERIVATIVE_METHOD_VERSION,
        samples=tuple(samples),
        schema_version=KINEMATIC_WAVEFORMS_SCHEMA_VERSION,
    )
