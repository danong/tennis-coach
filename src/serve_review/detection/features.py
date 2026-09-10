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
    "FeatureError",
    "FeatureConfig",
    "FeatureFrame",
    "extract_features",
]

#: Version of the feature configuration and frame schemas.
FEATURE_SCHEMA_VERSION = 1

_LEFT_SHOULDER = JOINT_INDEX["left_shoulder"]
_RIGHT_SHOULDER = JOINT_INDEX["right_shoulder"]
_LEFT_HIP = JOINT_INDEX["left_hip"]
_RIGHT_HIP = JOINT_INDEX["right_hip"]
_LEFT_ELBOW = JOINT_INDEX["left_elbow"]
_RIGHT_ELBOW = JOINT_INDEX["right_elbow"]
_LEFT_WRIST = JOINT_INDEX["left_wrist"]
_RIGHT_WRIST = JOINT_INDEX["right_wrist"]


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
    """

    smoothing_window: int = 3
    motion_reference: float = 2.0
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_reference": self.motion_reference,
            "schema_version": self.schema_version,
            "smoothing_window": self.smoothing_window,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FeatureConfig:
        if not isinstance(values, dict):
            raise FeatureError(
                "feature_config: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {"motion_reference", "schema_version", "smoothing_window"}
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_motion": self.body_motion,
            "elbow_speed": self.elbow_speed,
            "has_person": self.has_person,
            "motion_evidence": self.motion_evidence,
            "overhead_evidence": self.overhead_evidence,
            "player_scale": self.player_scale,
            "rest_evidence": self.rest_evidence,
            "schema_version": self.schema_version,
            "time_seconds": self.time_seconds,
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
            "elbow_speed",
            "has_person",
            "motion_evidence",
            "overhead_evidence",
            "player_scale",
            "rest_evidence",
            "schema_version",
            "time_seconds",
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
    raw_visible: list[float] = []
    raw_torso: list[float | None] = []
    raw_player: list[float | None] = []
    raw_overhead: list[float | None] = []
    # Normalized joint positions per frame: joint index -> (nx, ny).
    normalized: list[dict[int, tuple[float, float]] | None] = []

    for frame in items:
        if not frame.has_person:
            raw_visible.append(0.0)
            raw_torso.append(None)
            raw_player.append(None)
            raw_overhead.append(None)
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
            )
        )
    return tuple(output)
