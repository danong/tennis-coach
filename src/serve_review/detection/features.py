"""Temporal feature extraction over portable pose observations (M3.1).

Converts an ordered sequence of :class:`FrameObservation` (M2 portable
pose observations) into a versioned, deterministic sequence of
per-frame body-kinematic features for serve-range detection.

Output per frame (:class:`FeatureFrame`):

- ``time_seconds``: canonical source time, preserved exactly.
- ``has_person``: True when the input frame carried >= 1 person.
- ``visible_fraction``: smoothed joint-coverage fraction in [0, 1].
  Raw coverage is ``present_joints / 33`` for the primary person
  (first entry) and ``0.0`` for no-person frames.
- ``torso_scale`` / ``player_scale``: normalized-coordinate sizes used
  for scale normalization (shoulder-hip length and fallback box
  height). ``None`` when uncomputable or for no-person frames.
- ``wrist_speed`` / ``elbow_speed``: scale-normalized joint speeds
  (normalized units per source second) from source-time derivatives of
  translation-normalized positions. ``None`` when uncomputable or for
  no-person frames. Each is the max over available left/right sides.
- ``body_motion``: mean scale-normalized joint speed over joints that
  can be differentiated. ``None`` when uncomputable or no-person.
- ``overhead_evidence``: ``1.0`` when any available wrist sits above
  its same-side shoulder (image ``y`` decreases upward), ``0.0`` when
  computable but no wrist is overhead, ``None`` when the required
  joints are missing or for no-person frames. Body-kinematic only;
  never a claim about rackets, balls, or contact.
- ``rest_evidence`` / ``motion_evidence``: complementary pair in
  [0, 1] derived from smoothed ``body_motion`` via
  ``motion = min(1, body_motion / motion_reference)`` and
  ``rest = 1 - motion``. ``None`` when ``body_motion`` is ``None`` or
  for no-person frames.

Version-2 proximal geometry (strictly additive; every field above
keeps its exact version-1 semantics and default thresholds):

- ``elbow_flexion_left`` / ``elbow_flexion_right``: interior angle
  in degrees [0, 180] at the elbow between shoulder-elbow-wrist
  (180 = fully extended/straight limb, smaller = more bent).
  ``knee_flexion_left`` / ``knee_flexion_right``: likewise at the
  knee between hip-knee-ankle. ``None`` when any required joint is
  missing, gated out, or degenerate.
- ``shoulder_tilt``: signed lateral shoulder tilt in degrees
  [-90, 90] measured *relative to the torso/spine axis*, never to
  screen-space horizontal: with ``s`` the left-to-right shoulder
  vector and ``u`` the hip-midpoint-to-shoulder-midpoint spine unit
  vector, ``tilt = asin(clamp(dot(s_hat, u)))`` in degrees. ``0``
  means the shoulder line is perpendicular to the spine (level
  relative to the torso); positive means the right shoulder sits
  higher (toward the head end of the spine axis) than the left.
  Rigid rotations/translations of the whole skeleton leave it
  unchanged, so frame orientation and camera tilt cannot corrupt
  it. ``None`` when any of the four torso joints is missing/gated
  or the torso is degenerate.
- ``torso_displacement``: scale-normalized spatial displacement
  (normalized units, >= 0) of the torso midpoint over the trailing
  ``TORSO_DISPLACEMENT_WINDOW_SECONDS`` (1.0 s): the distance
  between the current torso midpoint and the most recent prior
  has-person torso midpoint at least 1.0 s earlier, divided by the
  current frame's scale (gated torso length else box height). The
  torso midpoint is the mean of the shoulder midpoint and the hip
  midpoint (falling back to whichever midpoint is computable). It
  is the stillness baseline for loading/rest and never uses limb
  joints. ``None`` when uncomputable (no 1.0 s history, gated or
  missing torso joints, no scale) or for no-person frames.
- Confidence gating (exclusion, documented choice): keypoints with
  ``visibility`` below ``FeatureConfig.visibility_floor`` (default
  0.4) are treated as missing *for the version-2 geometry channels
  only*. No interpolation is performed: inventing positions would
  fabricate joint angles. The version-1 auxiliary signals use raw
  joint presence exactly as before and ignore the floor.

Smoothing applies to every version-2 channel with the same explicit
contract as the version-1 smoothed fields: truncated centered
window mean over non-``None`` values from has-person frames only;
no-person frames always yield ``None`` and are excluded from
neighbors' averages.

Normalization:

- Translation is normalized out by expressing every joint relative to
  a per-frame body origin (shoulder midpoint, falling back to hip
  midpoint). Uniform image translations therefore leave speeds,
  overhead, and rest/motion evidence unchanged.
- Scale is normalized out by dividing by a per-frame scale
  (``torso_scale`` when computable, else box height). Uniform image
  scalings leave speeds and evidences unchanged; raw ``torso_scale`` /
  ``player_scale`` themselves scale proportionally by construction.
- Derivatives use canonical ``time_seconds`` differences
  (``distance / dt``), so unequal time steps yield correct physical
  rates. Non-positive ``dt`` (unordered/duplicate times) is rejected;
  a zero or missing scale yields ``None`` rather than a fabricated 0.

Smoothing (centered moving average with explicit boundaries):

- ``smoothing_window`` (>= 1) selects a truncated centered window
  ``[i - window//2, i + window//2]`` intersected with the sequence.
- ``visible_fraction`` for a no-person frame is exactly ``0.0``
  regardless of ``smoothing_window``: smoothing never bleeds neighbor
  visibility into a no-person frame. Conversely, no-person frames are
  excluded from the averages of neighboring has-person frames, so a
  gap never drags its neighbors down either. At sequence boundaries
  the window truncates (fewer samples, same mean semantics).
- Scales and speeds are smoothed the same way: truncated-window mean
  over non-``None`` values from has-person frames only; ``None`` when
  no sample qualifies; no-person frames always yield ``None`` (scales,
  speeds) / ``0.0`` (visibility).
- ``overhead_evidence`` is per-frame (unsmoothed) to avoid inventing
  overhead where joints were missing; ``rest``/``motion`` evidence is
  derived from smoothed ``body_motion``.

Missing data is propagated honestly: any feature that cannot be
computed from present joints/scales is ``None``. No-person frames
yield ``visible_fraction == 0.0``, ``torso_scale/player_scale``,
``wrist/elbow/body`` speeds, and all evidences ``None``.

Determinism: pure Python/std-lib only, no randomness, no wall clock,
no global state; iteration order is input order; JSON codecs sort
keys. Extracting twice from equal inputs yields equal outputs.

Primary-person rule: only ``persons[0]`` is used; extra persons are
ignored deterministically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from serve_review.pose.schema import (
    JOINT_INDEX,
    NUM_KEYPOINTS,
    FrameObservation,
)

__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "TORSO_DISPLACEMENT_WINDOW_SECONDS",
    "FeatureError",
    "FeatureConfig",
    "FeatureFrame",
    "extract_features",
]

#: Version of the feature configuration and frame schemas.
#:
#: Version 2 is strictly additive over version 1: every version-1 field
#: keeps its name, units, and default-threshold semantics, and five new
#: proximal-geometry channels plus a versioned visibility floor are
#: added. Version-1 payloads are rejected (missing keys / version
#: mismatch) rather than silently reinterpreted.
FEATURE_SCHEMA_VERSION = 2

#: Trailing window (seconds) for the torso stillness displacement.
TORSO_DISPLACEMENT_WINDOW_SECONDS = 1.0

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


class FeatureError(ValueError):
    """Raised when feature extraction input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_finite_positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureError(f"feature_config: {name!r} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise FeatureError(
            f"feature_config: {name!r} must be finite and > 0, got {value!r}."
        )
    return number


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Versioned configuration for temporal feature extraction.

    ``smoothing_window`` is the centered moving-average width in frames
    (>= 1; 1 disables smoothing). Even widths are allowed and use
    ``[i - window//2, i + window//2]`` truncated at boundaries.
    ``motion_reference`` is the scale-normalized speed (units/second)
    that maps to ``motion_evidence == 1.0``.
    ``visibility_floor`` (version 2, default 0.4) is the minimum
    keypoint ``visibility`` admitted into the proximal-geometry
    channels (joint angles, shoulder tilt, torso displacement):
    keypoints below the floor are treated as missing for those
    channels (exclusion, never interpolation), so low-confidence
    jitter never corrupts geometry. The version-1 auxiliary signals
    (``wrist_speed``, ``elbow_speed``, ``body_motion``, scales,
    visibility, overhead/rest/motion evidence) ignore the floor and
    keep their exact version-1 semantics.
    """

    smoothing_window: int = 3
    motion_reference: float = 2.0
    visibility_floor: float = 0.4
    schema_version: int = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise FeatureError(
                "feature_config: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise FeatureError(
                f"feature_config: unsupported schema_version {self.schema_version!r}; "
                f"expected {FEATURE_SCHEMA_VERSION}."
            )
        if not _is_int(self.smoothing_window) or self.smoothing_window < 1:
            raise FeatureError(
                "feature_config: 'smoothing_window' must be an integer >= 1, "
                f"got {self.smoothing_window!r}."
            )
        try:
            motion_reference = _check_finite_positive(
                "'motion_reference'", self.motion_reference
            )
        except FeatureError:
            raise
        object.__setattr__(self, "motion_reference", motion_reference)
        floor = self.visibility_floor
        if (
            isinstance(floor, bool)
            or not isinstance(floor, (int, float))
            or not math.isfinite(float(floor))
            or float(floor) < 0.0
            or float(floor) > 1.0
        ):
            raise FeatureError(
                "feature_config: 'visibility_floor' must lie in [0, 1], "
                f"got {self.visibility_floor!r}."
            )
        object.__setattr__(self, "visibility_floor", float(floor))

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_reference": self.motion_reference,
            "schema_version": self.schema_version,
            "smoothing_window": self.smoothing_window,
            "visibility_floor": self.visibility_floor,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FeatureConfig:
        if not isinstance(values, dict):
            raise FeatureError(
                "feature_config: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "motion_reference",
            "schema_version",
            "smoothing_window",
            "visibility_floor",
        }
        missing = sorted(known - set(values))
        if missing:
            raise FeatureError(
                f"feature_config: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise FeatureError(f"feature_config: unknown keys {unknown!r}.")
        return cls(
            smoothing_window=values["smoothing_window"],
            motion_reference=values["motion_reference"],
            visibility_floor=values["visibility_floor"],
            schema_version=values["schema_version"],
        )


def _optional_unit(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise FeatureError(
            f"feature_frame: {name!r} must lie in [0, 1] or be None, "
            f"got {value!r}."
        )
    return number


def _optional_joint_angle(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureError(
            f"feature_frame: {name!r} must be a number in [0, 180] or None, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 180.0:
        raise FeatureError(
            f"feature_frame: {name!r} must lie in [0, 180] or be None, "
            f"got {value!r}."
        )
    return number


def _optional_tilt(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureError(
            f"feature_frame: {name!r} must be a number in [-90, 90] or None, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < -90.0 or number > 90.0:
        raise FeatureError(
            f"feature_frame: {name!r} must lie in [-90, 90] or be None, "
            f"got {value!r}."
        )
    return number


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeatureError(
            f"feature_frame: {name!r} must be a number or None, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise FeatureError(
            f"feature_frame: {name!r} must be finite and >= 0 or None, "
            f"got {value!r}."
        )
    return number


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """One frame of normalized temporal features (versioned, immutable)."""

    time_seconds: float = 0.0
    has_person: bool = False
    visible_fraction: float = 0.0
    torso_scale: float | None = None
    player_scale: float | None = None
    wrist_speed: float | None = None
    elbow_speed: float | None = None
    body_motion: float | None = None
    overhead_evidence: float | None = None
    rest_evidence: float | None = None
    motion_evidence: float | None = None
    elbow_flexion_left: float | None = None
    elbow_flexion_right: float | None = None
    knee_flexion_left: float | None = None
    knee_flexion_right: float | None = None
    shoulder_tilt: float | None = None
    torso_displacement: float | None = None
    schema_version: int = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise FeatureError(
                "feature_frame: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise FeatureError(
                f"feature_frame: unsupported schema_version {self.schema_version!r}; "
                f"expected {FEATURE_SCHEMA_VERSION}."
            )
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise FeatureError(
                "feature_frame: 'time_seconds' must be a finite number >= 0, "
                f"got {self.time_seconds!r}."
            )
        object.__setattr__(self, "time_seconds", float(self.time_seconds))
        if not isinstance(self.has_person, bool):
            raise FeatureError(
                "feature_frame: 'has_person' must be a bool, "
                f"got {self.has_person!r}."
            )
        try:
            visible = float(self.visible_fraction)
        except (TypeError, ValueError) as exc:
            raise FeatureError(
                "feature_frame: 'visible_fraction' must be a number in "
                f"[0, 1], got {self.visible_fraction!r}."
            ) from exc
        if (
            isinstance(self.visible_fraction, bool)
            or not math.isfinite(visible)
            or visible < 0.0
            or visible > 1.0
        ):
            raise FeatureError(
                "feature_frame: 'visible_fraction' must lie in [0, 1], "
                f"got {self.visible_fraction!r}."
            )
        object.__setattr__(self, "visible_fraction", visible)
        # No-person invariant: visibility is exactly 0 and every scale /
        # speed / evidence field is None (never smoothed or fabricated).
        if not self.has_person:
            if visible != 0.0:
                raise FeatureError(
                    "feature_frame: 'visible_fraction' must be 0.0 when "
                    f"'has_person' is False, got {visible!r}."
                )
            for key in (
                "torso_scale",
                "player_scale",
                "wrist_speed",
                "elbow_speed",
                "body_motion",
                "overhead_evidence",
                "rest_evidence",
                "motion_evidence",
                "elbow_flexion_left",
                "elbow_flexion_right",
                "knee_flexion_left",
                "knee_flexion_right",
                "shoulder_tilt",
                "torso_displacement",
            ):
                if getattr(self, key) is not None:
                    raise FeatureError(
                        f"feature_frame: {key!r} must be None when "
                        "'has_person' is False."
                    )
        object.__setattr__(
            self, "torso_scale", _optional_nonnegative(self.torso_scale, "'torso_scale'")
        )
        object.__setattr__(
            self,
            "player_scale",
            _optional_nonnegative(self.player_scale, "'player_scale'"),
        )
        object.__setattr__(
            self, "wrist_speed", _optional_nonnegative(self.wrist_speed, "'wrist_speed'")
        )
        object.__setattr__(
            self, "elbow_speed", _optional_nonnegative(self.elbow_speed, "'elbow_speed'")
        )
        object.__setattr__(
            self, "body_motion", _optional_nonnegative(self.body_motion, "'body_motion'")
        )
        object.__setattr__(
            self,
            "overhead_evidence",
            _optional_unit(self.overhead_evidence, "'overhead_evidence'"),
        )
        object.__setattr__(
            self, "rest_evidence", _optional_unit(self.rest_evidence, "'rest_evidence'")
        )
        object.__setattr__(
            self,
            "motion_evidence",
            _optional_unit(self.motion_evidence, "'motion_evidence'"),
        )
        if (self.rest_evidence is None) != (self.motion_evidence is None):
            raise FeatureError(
                "feature_frame: 'rest_evidence' and 'motion_evidence' must "
                "both be None or both be set."
            )
        object.__setattr__(
            self,
            "elbow_flexion_left",
            _optional_joint_angle(self.elbow_flexion_left, "'elbow_flexion_left'"),
        )
        object.__setattr__(
            self,
            "elbow_flexion_right",
            _optional_joint_angle(
                self.elbow_flexion_right, "'elbow_flexion_right'"
            ),
        )
        object.__setattr__(
            self,
            "knee_flexion_left",
            _optional_joint_angle(self.knee_flexion_left, "'knee_flexion_left'"),
        )
        object.__setattr__(
            self,
            "knee_flexion_right",
            _optional_joint_angle(self.knee_flexion_right, "'knee_flexion_right'"),
        )
        object.__setattr__(
            self,
            "shoulder_tilt",
            _optional_tilt(self.shoulder_tilt, "'shoulder_tilt'"),
        )
        object.__setattr__(
            self,
            "torso_displacement",
            _optional_nonnegative(self.torso_displacement, "'torso_displacement'"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_motion": self.body_motion,
            "elbow_flexion_left": self.elbow_flexion_left,
            "elbow_flexion_right": self.elbow_flexion_right,
            "elbow_speed": self.elbow_speed,
            "has_person": self.has_person,
            "knee_flexion_left": self.knee_flexion_left,
            "knee_flexion_right": self.knee_flexion_right,
            "motion_evidence": self.motion_evidence,
            "overhead_evidence": self.overhead_evidence,
            "player_scale": self.player_scale,
            "rest_evidence": self.rest_evidence,
            "schema_version": self.schema_version,
            "shoulder_tilt": self.shoulder_tilt,
            "time_seconds": self.time_seconds,
            "torso_displacement": self.torso_displacement,
            "torso_scale": self.torso_scale,
            "visible_fraction": self.visible_fraction,
            "wrist_speed": self.wrist_speed,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FeatureFrame:
        if not isinstance(values, dict):
            raise FeatureError(
                "feature_frame: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "body_motion",
            "elbow_flexion_left",
            "elbow_flexion_right",
            "elbow_speed",
            "has_person",
            "knee_flexion_left",
            "knee_flexion_right",
            "motion_evidence",
            "overhead_evidence",
            "player_scale",
            "rest_evidence",
            "schema_version",
            "shoulder_tilt",
            "time_seconds",
            "torso_displacement",
            "torso_scale",
            "visible_fraction",
            "wrist_speed",
        }
        missing = sorted(known - set(values))
        if missing:
            raise FeatureError(
                f"feature_frame: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise FeatureError(f"feature_frame: unknown keys {unknown!r}.")
        return cls(**{key: values[key] for key in known})


def _midpoint(
    first: Any | None, second: Any | None
) -> tuple[float, float] | None:
    if first is None or second is None:
        return None
    return ((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)


def _distance(
    first: tuple[float, float] | None, second: tuple[float, float] | None
) -> float | None:
    if first is None or second is None:
        return None
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _gated(joint: Any | None, floor: float) -> Any | None:
    """Return ``joint`` unless it is missing or below the visibility floor."""
    if joint is None:
        return None
    try:
        visibility = float(joint.visibility)
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(visibility) or visibility < floor:
        return None
    return joint


def _joint_angle_deg(
    vertex: Any | None, first: Any | None, second: Any | None
) -> float | None:
    """Interior angle in degrees at ``vertex`` between ``first``/``second``.

    Returns 180.0 for a straight limb and ``None`` when any joint is
    missing or the limb is degenerate (zero-length segment).
    """
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


def _shoulder_tilt_deg(
    left_shoulder: Any | None,
    right_shoulder: Any | None,
    left_hip: Any | None,
    right_hip: Any | None,
) -> float | None:
    """Signed shoulder tilt in degrees relative to the spine axis.

    Zero when the shoulder line is perpendicular to the
    hip-midpoint-to-shoulder-midpoint spine vector; positive when the
    right shoulder sits toward the head end of the spine axis. Uses
    only relative vectors, so rigid rotation/translation of the whole
    skeleton leaves the result unchanged. ``None`` when any torso
    joint is missing or the torso is degenerate.
    """
    if (
        left_shoulder is None
        or right_shoulder is None
        or left_hip is None
        or right_hip is None
    ):
        return None
    sx = right_shoulder.x - left_shoulder.x
    sy = right_shoulder.y - left_shoulder.y
    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    if shoulder_mid is None or hip_mid is None:
        return None
    vx = shoulder_mid[0] - hip_mid[0]
    vy = shoulder_mid[1] - hip_mid[1]
    norm_s = math.hypot(sx, sy)
    norm_v = math.hypot(vx, vy)
    if norm_s <= 0.0 or norm_v <= 0.0:
        return None
    if not (
        math.isfinite(norm_s) and math.isfinite(norm_v)
    ):
        return None
    sine = (sx * vx + sy * vy) / (norm_s * norm_v)
    sine = max(-1.0, min(1.0, sine))
    return math.degrees(math.asin(sine))


def _window_indices(index: int, count: int, window: int) -> list[int]:
    half = window // 2
    # Centered window; for even widths the interval is asymmetric by one
    # on the right ([i - w//2, i + w//2]); truncated at both boundaries.
    start = max(0, index - half)
    end = min(count, index + half + 1)
    return list(range(start, end))


def extract_features(
    frames: Sequence[FrameObservation],
    config: FeatureConfig | None = None,
) -> tuple[FeatureFrame, ...]:
    """Convert ordered pose observations into normalized features.

    ``frames`` must carry strictly increasing canonical
    ``time_seconds``; violations raise :class:`FeatureError`. An empty
    sequence yields an empty tuple. ``config`` defaults to
    :class:`FeatureConfig()` (``smoothing_window=3``).
    """
    cfg = config if config is not None else FeatureConfig()
    if not isinstance(cfg, FeatureConfig):
        raise FeatureError(
            "extract_features: 'config' must be a FeatureConfig, "
            f"got {type(cfg).__name__}."
        )
    items = list(frames)
    for entry in items:
        if not isinstance(entry, FrameObservation):
            raise FeatureError(
                "extract_features: every frame must be a FrameObservation, "
                f"got {type(entry).__name__}."
            )
    for earlier, later in zip(items, items[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise FeatureError(
                "extract_features: frame times must be strictly increasing, "
                f"got {earlier.time_seconds!r} followed by "
                f"{later.time_seconds!r}."
            )
    count = len(items)
    if count == 0:
        return ()

    has_person: list[bool] = [frame.has_person for frame in items]
    floor = cfg.visibility_floor
    raw_visible: list[float] = []
    raw_torso: list[float | None] = []
    raw_player: list[float | None] = []
    raw_overhead: list[float | None] = []
    raw_elbow_left: list[float | None] = []
    raw_elbow_right: list[float | None] = []
    raw_knee_left: list[float | None] = []
    raw_knee_right: list[float | None] = []
    raw_tilt: list[float | None] = []
    # Gated torso midpoint/scale per frame for the displacement channel.
    gated_mid: list[tuple[float, float] | None] = []
    gated_scale: list[float | None] = []
    # Normalized joint positions per frame: joint index -> (nx, ny).
    normalized: list[dict[int, tuple[float, float]] | None] = []

    for frame in items:
        if not frame.has_person:
            raw_visible.append(0.0)
            raw_torso.append(None)
            raw_player.append(None)
            raw_overhead.append(None)
            raw_elbow_left.append(None)
            raw_elbow_right.append(None)
            raw_knee_left.append(None)
            raw_knee_right.append(None)
            raw_tilt.append(None)
            gated_mid.append(None)
            gated_scale.append(None)
            normalized.append(None)
            continue
        person = frame.persons[0]
        joints = person.keypoints
        present = sum(1 for joint in joints if joint is not None)
        raw_visible.append(present / NUM_KEYPOINTS)

        shoulder_mid = _midpoint(joints[_LEFT_SHOULDER], joints[_RIGHT_SHOULDER])
        hip_mid = _midpoint(joints[_LEFT_HIP], joints[_RIGHT_HIP])
        torso = _distance(shoulder_mid, hip_mid)
        raw_torso.append(torso)
        if torso is not None and torso > 0.0:
            scale = torso
        else:
            try:
                box_height = person.box.y_max - person.box.y_min
            except (AttributeError, TypeError):
                box_height = None
            scale = (
                box_height
                if box_height is not None and box_height > 0.0
                else None
            )
        raw_player.append(scale if scale is not None and scale > 0.0 else None)

        # Overhead evidence: max over sides with wrist + same-side
        # shoulder present; 1 when the wrist is above the shoulder.
        sides = (
            (joints[_LEFT_WRIST], joints[_LEFT_SHOULDER]),
            (joints[_RIGHT_WRIST], joints[_RIGHT_SHOULDER]),
        )
        available = [
            1.0 if wrist.y < shoulder.y else 0.0
            for wrist, shoulder in sides
            if wrist is not None and shoulder is not None
        ]
        raw_overhead.append(max(available) if available else None)

        # Version-2 proximal geometry uses visibility-gated joints
        # (exclusion only; the version-1 signals above ignore the floor).
        gated = [
            _gated(joint, floor) for joint in joints
        ]
        raw_elbow_left.append(
            _joint_angle_deg(
                gated[_LEFT_ELBOW],
                gated[_LEFT_SHOULDER],
                gated[_LEFT_WRIST],
            )
        )
        raw_elbow_right.append(
            _joint_angle_deg(
                gated[_RIGHT_ELBOW],
                gated[_RIGHT_SHOULDER],
                gated[_RIGHT_WRIST],
            )
        )
        raw_knee_left.append(
            _joint_angle_deg(
                gated[_LEFT_KNEE], gated[_LEFT_HIP], gated[_LEFT_ANKLE]
            )
        )
        raw_knee_right.append(
            _joint_angle_deg(
                gated[_RIGHT_KNEE], gated[_RIGHT_HIP], gated[_RIGHT_ANKLE]
            )
        )
        raw_tilt.append(
            _shoulder_tilt_deg(
                gated[_LEFT_SHOULDER],
                gated[_RIGHT_SHOULDER],
                gated[_LEFT_HIP],
                gated[_RIGHT_HIP],
            )
        )
        gated_shoulder_mid = _midpoint(
            gated[_LEFT_SHOULDER], gated[_RIGHT_SHOULDER]
        )
        gated_hip_mid = _midpoint(gated[_LEFT_HIP], gated[_RIGHT_HIP])
        gated_torso = _distance(gated_shoulder_mid, gated_hip_mid)
        if gated_torso is not None and gated_torso > 0.0:
            gated_frame_scale: float | None = gated_torso
        else:
            try:
                gated_box_height = person.box.y_max - person.box.y_min
            except (AttributeError, TypeError):
                gated_box_height = None
            gated_frame_scale = (
                gated_box_height
                if gated_box_height is not None and gated_box_height > 0.0
                else None
            )
        if gated_shoulder_mid is not None and gated_hip_mid is not None:
            gated_mid.append(
                (
                    (gated_shoulder_mid[0] + gated_hip_mid[0]) / 2.0,
                    (gated_shoulder_mid[1] + gated_hip_mid[1]) / 2.0,
                )
            )
        elif gated_shoulder_mid is not None:
            gated_mid.append(gated_shoulder_mid)
        elif gated_hip_mid is not None:
            gated_mid.append(gated_hip_mid)
        else:
            gated_mid.append(None)
        gated_scale.append(
            gated_frame_scale
            if gated_frame_scale is not None and gated_frame_scale > 0.0
            else None
        )

        origin = shoulder_mid if shoulder_mid is not None else hip_mid
        if origin is None or scale is None or scale <= 0.0:
            normalized.append(None)
            continue
        positions: dict[int, tuple[float, float]] = {}
        for index, joint in enumerate(joints):
            if joint is None:
                continue
            positions[index] = (
                (joint.x - origin[0]) / scale,
                (joint.y - origin[1]) / scale,
            )
        normalized.append(positions)

    # Raw source-time derivatives (unsmoothed speeds).
    raw_wrist: list[float | None] = [None] * count
    raw_elbow: list[float | None] = [None] * count
    raw_body: list[float | None] = [None] * count
    wrist_pairs = ((_LEFT_WRIST,), (_RIGHT_WRIST,))
    elbow_pairs = ((_LEFT_ELBOW,), (_RIGHT_ELBOW,))
    for index in range(1, count):
        if not has_person[index] or not has_person[index - 1]:
            continue
        prev = normalized[index - 1]
        current = normalized[index]
        if prev is None or current is None:
            continue
        dt = items[index].time_seconds - items[index - 1].time_seconds
        if not math.isfinite(dt) or dt <= 0.0:
            continue

        def _side_speed(choices: tuple[tuple[int, ...], ...]) -> float | None:
            speeds: list[float] = []
            for (joint_index,) in choices:
                before = prev.get(joint_index)
                after = current.get(joint_index)
                if before is None or after is None:
                    continue
                speeds.append(
                    math.hypot(after[0] - before[0], after[1] - before[1]) / dt
                )
            return max(speeds) if speeds else None

        raw_wrist[index] = _side_speed(wrist_pairs)
        raw_elbow[index] = _side_speed(elbow_pairs)
        joint_speeds: list[float] = []
        for joint_index, after in current.items():
            before = prev.get(joint_index)
            if before is None:
                continue
            joint_speeds.append(
                math.hypot(after[0] - before[0], after[1] - before[1]) / dt
            )
        raw_body[index] = (
            sum(joint_speeds) / len(joint_speeds) if joint_speeds else None
        )

    # Raw torso stillness displacement: trailing 1.0 s endpoint distance
    # of the gated torso midpoint, normalized by the current frame's
    # gated scale. The reference is the most recent prior has-person
    # frame with a computable midpoint at least 1.0 s earlier
    # (1e-9 tolerance); otherwise the sample is honestly None.
    raw_displacement: list[float | None] = [None] * count
    horizon = TORSO_DISPLACEMENT_WINDOW_SECONDS
    for index in range(count):
        if not has_person[index]:
            continue
        current_mid = gated_mid[index]
        current_scale = gated_scale[index]
        if current_mid is None or current_scale is None:
            continue
        moment = items[index].time_seconds
        reference = -1
        for prior in range(index - 1, -1, -1):
            if not has_person[prior] or gated_mid[prior] is None:
                continue
            if moment - items[prior].time_seconds >= horizon - 1e-9:
                reference = prior
                break
        if reference < 0:
            continue
        raw_displacement[index] = (
            math.hypot(
                current_mid[0] - gated_mid[reference][0],
                current_mid[1] - gated_mid[reference][1],
            )
            / current_scale
        )

    window = cfg.smoothing_window

    def _smooth_fraction(values: list[float]) -> list[float]:
        smoothed: list[float] = []
        for index in range(count):
            if not has_person[index]:
                # Never bleed neighbor visibility into a no-person frame.
                smoothed.append(0.0)
                continue
            members = [
                values[j]
                for j in _window_indices(index, count, window)
                if has_person[j]
            ]
            smoothed.append(sum(members) / len(members) if members else 0.0)
        return smoothed

    def _smooth_optional(values: list[float | None]) -> list[float | None]:
        smoothed: list[float | None] = []
        for index in range(count):
            if not has_person[index]:
                smoothed.append(None)
                continue
            members = [
                values[j]
                for j in _window_indices(index, count, window)
                if has_person[j] and values[j] is not None
            ]
            if not members:
                smoothed.append(None)
            else:
                smoothed.append(sum(members) / len(members))
        return smoothed

    smooth_visible = _smooth_fraction(raw_visible)
    smooth_torso = _smooth_optional(raw_torso)
    smooth_player = _smooth_optional(raw_player)
    smooth_wrist = _smooth_optional(raw_wrist)
    smooth_elbow = _smooth_optional(raw_elbow)
    smooth_body = _smooth_optional(raw_body)
    smooth_elbow_left = _smooth_optional(raw_elbow_left)
    smooth_elbow_right = _smooth_optional(raw_elbow_right)
    smooth_knee_left = _smooth_optional(raw_knee_left)
    smooth_knee_right = _smooth_optional(raw_knee_right)
    smooth_tilt = _smooth_optional(raw_tilt)
    smooth_displacement = _smooth_optional(raw_displacement)

    output: list[FeatureFrame] = []
    for index, frame in enumerate(items):
        moment = frame.time_seconds
        if not has_person[index]:
            output.append(
                FeatureFrame(
                    time_seconds=moment,
                    has_person=False,
                    visible_fraction=0.0,
                    torso_scale=None,
                    player_scale=None,
                    wrist_speed=None,
                    elbow_speed=None,
                    body_motion=None,
                    overhead_evidence=None,
                    rest_evidence=None,
                    motion_evidence=None,
                    elbow_flexion_left=None,
                    elbow_flexion_right=None,
                    knee_flexion_left=None,
                    knee_flexion_right=None,
                    shoulder_tilt=None,
                    torso_displacement=None,
                )
            )
            continue
        body = smooth_body[index]
        if body is None:
            rest: float | None = None
            motion: float | None = None
        else:
            motion = min(1.0, body / cfg.motion_reference)
            rest = 1.0 - motion
        output.append(
            FeatureFrame(
                time_seconds=moment,
                has_person=True,
                visible_fraction=smooth_visible[index],
                torso_scale=smooth_torso[index],
                player_scale=smooth_player[index],
                wrist_speed=smooth_wrist[index],
                elbow_speed=smooth_elbow[index],
                body_motion=body,
                overhead_evidence=raw_overhead[index],
                rest_evidence=rest,
                motion_evidence=motion,
                elbow_flexion_left=smooth_elbow_left[index],
                elbow_flexion_right=smooth_elbow_right[index],
                knee_flexion_left=smooth_knee_left[index],
                knee_flexion_right=smooth_knee_right[index],
                shoulder_tilt=smooth_tilt[index],
                torso_displacement=smooth_displacement[index],
            )
        )
    return tuple(output)
