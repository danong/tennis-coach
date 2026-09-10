"""Portable pose observation schemas (M2.2).

This module is intentionally pure: no file I/O, no subprocesses, no model
or framework objects. It defines versioned person boxes, 33-keypoint body
observations, visibility, frame observations, and cache identity with
deterministic JSON codecs and strict validation.

Coordinate contract (upright, unmirrored, normalized):

- Coordinates are normalized to the upright display frame produced by the
  M2.1 sampler (rotation already applied). Origin is the top-left pixel,
  ``x`` increases to the right, ``y`` increases downward; the image is
  never mirrored.
- Every coordinate lies in ``[0, 1]``; ``x_min < x_max`` and
  ``y_min < y_max`` for boxes. Out-of-range, non-finite, or boolean
  values are rejected.
- Canonical source time is ``time_seconds`` (decimal seconds derived
  from integer container timestamps). ``timestamp_ms`` is only the
  ``round(time_seconds * 1000)`` convenience for ordered ``VIDEO``
  inference calls and must equal that rounding exactly; canonical logic
  must never derive time from a frame index or an assumed FPS.

Missing data is explicit: each of the 33 body joints is either a
:class:`BodyKeypoint` or ``None`` (``null`` in JSON). An explicit
no-person frame carries an empty ``persons`` list, never a fabricated
pose. A person entry with all 33 joints missing is rejected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "POSE_SCHEMA_VERSION",
    "NUM_KEYPOINTS",
    "JOINT_NAMES",
    "JOINT_INDEX",
    "PoseError",
    "SchemaVersionError",
    "PersonBoxError",
    "KeypointError",
    "PersonObservationError",
    "FrameObservationError",
    "CacheIdentityError",
    "PersonBox",
    "BodyKeypoint",
    "PersonObservation",
    "FrameObservation",
    "CacheIdentity",
]

#: Version of every pose schema in this module.
POSE_SCHEMA_VERSION = 1
#: Number of body joints per person (MediaPipe Pose order).
NUM_KEYPOINTS = 33

#: Canonical joint order (MediaPipe Pose landmark order, 0..32).
JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

#: Joint name to index lookup.
JOINT_INDEX: dict[str, int] = {name: index for index, name in enumerate(JOINT_NAMES)}


class PoseError(ValueError):
    """Base error for pose observation validation failures."""


class SchemaVersionError(PoseError):
    """Raised when a document carries an unsupported schema version."""


class PersonBoxError(PoseError):
    """Raised when a person box is invalid."""


class KeypointError(PoseError):
    """Raised when a body keypoint is invalid."""


class PersonObservationError(PoseError):
    """Raised when a person observation is invalid."""


class FrameObservationError(PoseError):
    """Raised when a frame observation is invalid."""


class CacheIdentityError(PoseError):
    """Raised when a cache identity is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_schema_version(name: str, values: dict[str, Any]) -> int:
    if "schema_version" not in values:
        raise PoseError(f"{name}: missing required key 'schema_version'.")
    version = values["schema_version"]
    if not _is_int(version):
        raise PoseError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != POSE_SCHEMA_VERSION:
        if version > POSE_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{name}: unsupported newer schema_version {version!r}; "
                f"this build supports version {POSE_SCHEMA_VERSION}."
            )
        raise SchemaVersionError(
            f"{name}: unsupported schema_version {version!r}; "
            f"expected version {POSE_SCHEMA_VERSION}."
        )
    return version


def _check_no_unknown_keys(name: str, values: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise PoseError(f"{name}: unknown keys {unknown!r}.")


def _check_normalized(name: str, key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise PoseError(
            f"{name}: {key!r} must be a finite number in [0, 1], "
            f"got {value!r}."
        )
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise PoseError(
            f"{name}: {key!r} must lie in [0, 1] in upright unmirrored "
            f"normalized coordinates, got {value!r}."
        )
    return number


def _check_unit_score(name: str, key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise PoseError(
            f"{name}: {key!r} must be a finite number in [0, 1], "
            f"got {value!r}."
        )
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise PoseError(
            f"{name}: {key!r} must lie in [0, 1], got {value!r}."
        )
    return number


def _check_non_blank(name: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PoseError(
            f"{name}: {key!r} must be a non-blank string, got {value!r}."
        )
    return value


def _dumps_deterministic(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _loads_object(name: str, data: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PoseError(f"{name}: invalid UTF-8 JSON payload.") from exc
    if not isinstance(data, str):
        raise PoseError(
            f"{name}: JSON payload must be str or bytes, "
            f"got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise PoseError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise PoseError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


@dataclass(frozen=True, slots=True)
class PersonBox:
    """Upright unmirrored normalized person box.

    Origin is the top-left of the upright display frame, ``x`` right and
    ``y`` down. Bounds satisfy ``0 <= x_min < x_max <= 1`` and
    ``0 <= y_min < y_max <= 1``. The box carries geometry only;
    detector confidence lives on :class:`PersonObservation`.
    """

    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    schema_version: int = POSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "person_box"
        version = self.schema_version
        if not _is_int(version):
            raise PersonBoxError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != POSE_SCHEMA_VERSION:
            if version > POSE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {POSE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {POSE_SCHEMA_VERSION}."
            )
        try:
            x_min = _check_normalized(name, "'x_min'", self.x_min)
            y_min = _check_normalized(name, "'y_min'", self.y_min)
            x_max = _check_normalized(name, "'x_max'", self.x_max)
            y_max = _check_normalized(name, "'y_max'", self.y_max)
        except PoseError as exc:
            raise PersonBoxError(str(exc)) from exc
        if not x_min < x_max:
            raise PersonBoxError(
                f"{name}: 'x_min' ({x_min!r}) must be less than "
                f"'x_max' ({x_max!r})."
            )
        if not y_min < y_max:
            raise PersonBoxError(
                f"{name}: 'y_min' ({y_min!r}) must be less than "
                f"'y_max' ({y_max!r})."
            )
        object.__setattr__(self, "x_min", x_min)
        object.__setattr__(self, "y_min", y_min)
        object.__setattr__(self, "x_max", x_max)
        object.__setattr__(self, "y_max", y_max)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "x_max": self.x_max,
            "x_min": self.x_min,
            "y_max": self.y_max,
            "y_min": self.y_min,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PersonBox:
        name = "person_box"
        if not isinstance(values, dict):
            raise PersonBoxError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"schema_version", "x_max", "x_min", "y_max", "y_min"}
        missing = sorted(known - set(values))
        if missing:
            raise PersonBoxError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise PersonBoxError(str(exc)) from exc
        try:
            _check_schema_version(name, values)
        except PoseError as exc:
            raise _regrade_box_error(exc) from exc
        try:
            return cls(
                x_min=values["x_min"],
                y_min=values["y_min"],
                x_max=values["x_max"],
                y_max=values["y_max"],
                schema_version=values["schema_version"],
            )
        except (PersonBoxError, SchemaVersionError):
            raise
        except PoseError as exc:
            raise PersonBoxError(str(exc)) from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PersonBox:
        return cls.from_dict(_loads_object("person_box", data))


def _regrade_box_error(exc: PoseError) -> PoseError:
    if isinstance(exc, SchemaVersionError):
        return exc
    return PersonBoxError(str(exc))


@dataclass(frozen=True, slots=True)
class BodyKeypoint:
    """One body joint in upright unmirrored normalized coordinates.

    ``x``/``y`` lie in ``[0, 1]`` with top-left origin; ``visibility``
    lies in ``[0, 1]`` (0 = likely occluded, 1 = likely visible) and
    records estimator uncertainty honestly. Missing joints are
    represented as ``None`` in the enclosing observation, never as a
    zero-filled keypoint. Versioning is owned by the enclosing
    :class:`PersonObservation`.
    """

    x: float = 0.0
    y: float = 0.0
    visibility: float = 0.0

    def __post_init__(self) -> None:
        name = "body_keypoint"
        try:
            x = _check_normalized(name, "'x'", self.x)
            y = _check_normalized(name, "'y'", self.y)
            visibility = _check_unit_score(name, "'visibility'", self.visibility)
        except PoseError as exc:
            raise KeypointError(str(exc)) from exc
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "visibility", visibility)

    def to_dict(self) -> dict[str, Any]:
        return {"visibility": self.visibility, "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BodyKeypoint:
        name = "body_keypoint"
        if not isinstance(values, dict):
            raise KeypointError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"visibility", "x", "y"}
        missing = sorted(known - set(values))
        if missing:
            raise KeypointError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise KeypointError(str(exc)) from exc
        try:
            return cls(
                x=values["x"], y=values["y"], visibility=values["visibility"]
            )
        except (KeypointError, PoseError) as exc:
            raise KeypointError(str(exc)) from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> BodyKeypoint:
        return cls.from_dict(_loads_object("body_keypoint", data))


@dataclass(frozen=True, slots=True)
class PersonObservation:
    """One detected person: box, 33 joints, and detector confidence.

    ``keypoints`` holds exactly ``NUM_KEYPOINTS`` entries in
    :data:`JOINT_NAMES` order; each entry is a :class:`BodyKeypoint` or
    ``None`` for an explicitly missing joint. At least one joint must be
    present. ``score`` is the person-detection confidence in ``[0, 1]``.
    """

    box: PersonBox = None  # type: ignore[assignment]
    keypoints: tuple[BodyKeypoint | None, ...] = ()
    score: float = 0.0
    schema_version: int = POSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "person_observation"
        version = self.schema_version
        if not _is_int(version):
            raise PersonObservationError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != POSE_SCHEMA_VERSION:
            if version > POSE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {POSE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {POSE_SCHEMA_VERSION}."
            )
        if not isinstance(self.box, PersonBox):
            raise PersonObservationError(
                f"{name}: 'box' must be a PersonBox, "
                f"got {type(self.box).__name__}."
            )
        raw = self.keypoints
        if not isinstance(raw, (list, tuple)):
            raise PersonObservationError(
                f"{name}: 'keypoints' must be a list or tuple of length "
                f"{NUM_KEYPOINTS}, got {type(raw).__name__}."
            )
        joints = tuple(raw)
        if len(joints) != NUM_KEYPOINTS:
            raise PersonObservationError(
                f"{name}: 'keypoints' must hold exactly {NUM_KEYPOINTS} "
                f"entries in JOINT_NAMES order, got {len(joints)}."
            )
        for index, joint in enumerate(joints):
            if joint is not None and not isinstance(joint, BodyKeypoint):
                raise PersonObservationError(
                    f"{name}: keypoint {index} ({JOINT_NAMES[index]!r}) must "
                    f"be BodyKeypoint or None, got {type(joint).__name__}."
                )
        if all(joint is None for joint in joints):
            raise PersonObservationError(
                f"{name}: at least one of the {NUM_KEYPOINTS} joints must be "
                "present; a person entry with every joint missing is invalid "
                "(use an empty frame persons list for no-person frames)."
            )
        object.__setattr__(self, "keypoints", joints)
        try:
            score = _check_unit_score(name, "'score'", self.score)
        except PoseError as exc:
            raise PersonObservationError(str(exc)) from exc
        object.__setattr__(self, "score", score)

    @property
    def present_count(self) -> int:
        """Return the number of non-missing joints."""
        return sum(1 for joint in self.keypoints if joint is not None)

    def joint(self, name: str) -> BodyKeypoint | None:
        """Return the joint for ``name`` (``None`` when missing)."""
        try:
            return self.keypoints[JOINT_INDEX[name]]
        except KeyError as exc:
            raise PersonObservationError(
                f"person_observation: unknown joint name {name!r}; "
                f"expected one of {list(JOINT_NAMES)!r}."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "box": self.box.to_dict(),
            "keypoints": [
                joint.to_dict() if joint is not None else None
                for joint in self.keypoints
            ],
            "schema_version": self.schema_version,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PersonObservation:
        name = "person_observation"
        if not isinstance(values, dict):
            raise PersonObservationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"box", "keypoints", "schema_version", "score"}
        missing = sorted(known - set(values))
        if missing:
            raise PersonObservationError(
                f"{name}: missing required keys {missing!r}."
            )
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise PersonObservationError(str(exc)) from exc
        try:
            _check_schema_version(name, values)
        except PoseError as exc:
            if isinstance(exc, SchemaVersionError):
                raise
            raise PersonObservationError(str(exc)) from exc
        try:
            box = PersonBox.from_dict(values["box"])
        except PoseError as exc:
            raise PersonObservationError(
                f"{name}: invalid box: {exc}."
            ) from exc
        raw_joints = values["keypoints"]
        if not isinstance(raw_joints, list):
            raise PersonObservationError(
                f"{name}: 'keypoints' must be a JSON list of length "
                f"{NUM_KEYPOINTS} with objects or nulls, "
                f"got {type(raw_joints).__name__}."
            )
        if len(raw_joints) != NUM_KEYPOINTS:
            raise PersonObservationError(
                f"{name}: 'keypoints' must hold exactly {NUM_KEYPOINTS} "
                f"entries, got {len(raw_joints)}."
            )
        joints: list[BodyKeypoint | None] = []
        for index, entry in enumerate(raw_joints):
            if entry is None:
                joints.append(None)
                continue
            try:
                joints.append(BodyKeypoint.from_dict(entry))
            except PoseError as exc:
                raise PersonObservationError(
                    f"{name}: invalid keypoint {index} "
                    f"({JOINT_NAMES[index]!r}): {exc}."
                ) from exc
        try:
            return cls(
                box=box,
                keypoints=tuple(joints),
                score=values["score"],
                schema_version=values["schema_version"],
            )
        except (PersonObservationError, SchemaVersionError):
            raise
        except PoseError as exc:
            raise PersonObservationError(str(exc)) from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PersonObservation:
        return cls.from_dict(_loads_object("person_observation", data))


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """All person observations at one canonical source time.

    ``time_seconds`` is canonical; ``timestamp_ms`` must equal
    ``round(time_seconds * 1000)``. ``persons`` may be empty, which
    explicitly records a no-person frame. Times across a cache must be
    strictly increasing (checked by the cache layer).
    """

    time_seconds: float = 0.0
    timestamp_ms: int = 0
    persons: tuple[PersonObservation, ...] = ()
    schema_version: int = POSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "frame_observation"
        version = self.schema_version
        if not _is_int(version):
            raise FrameObservationError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != POSE_SCHEMA_VERSION:
            if version > POSE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {POSE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {POSE_SCHEMA_VERSION}."
            )
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise FrameObservationError(
                f"{name}: 'time_seconds' must be a finite number >= 0, "
                f"got {self.time_seconds!r}."
            )
        moment = float(self.time_seconds)
        object.__setattr__(self, "time_seconds", moment)
        if (
            isinstance(self.timestamp_ms, bool)
            or not isinstance(self.timestamp_ms, int)
            or self.timestamp_ms < 0
        ):
            raise FrameObservationError(
                f"{name}: 'timestamp_ms' must be an integer >= 0, "
                f"got {self.timestamp_ms!r}."
            )
        expected_ms = int(round(moment * 1000))
        if self.timestamp_ms != expected_ms:
            raise FrameObservationError(
                f"{name}: 'timestamp_ms' ({self.timestamp_ms!r}) must equal "
                f"round(time_seconds * 1000) ({expected_ms!r}); keep the "
                "canonical float separate from the integer convenience."
            )
        raw = self.persons
        if not isinstance(raw, (list, tuple)):
            raise FrameObservationError(
                f"{name}: 'persons' must be a list or tuple of "
                f"PersonObservation, got {type(raw).__name__}."
            )
        persons = tuple(raw)
        for entry in persons:
            if not isinstance(entry, PersonObservation):
                raise FrameObservationError(
                    f"{name}: every person must be a PersonObservation, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "persons", persons)

    @property
    def has_person(self) -> bool:
        """Return True when at least one person was observed."""
        return len(self.persons) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "persons": [person.to_dict() for person in self.persons],
            "schema_version": self.schema_version,
            "time_seconds": self.time_seconds,
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> FrameObservation:
        name = "frame_observation"
        if not isinstance(values, dict):
            raise FrameObservationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"persons", "schema_version", "time_seconds", "timestamp_ms"}
        missing = sorted(known - set(values))
        if missing:
            raise FrameObservationError(
                f"{name}: missing required keys {missing!r}."
            )
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise FrameObservationError(str(exc)) from exc
        try:
            _check_schema_version(name, values)
        except PoseError as exc:
            if isinstance(exc, SchemaVersionError):
                raise
            raise FrameObservationError(str(exc)) from exc
        raw_persons = values["persons"]
        if not isinstance(raw_persons, list):
            raise FrameObservationError(
                f"{name}: 'persons' must be a JSON list, "
                f"got {type(raw_persons).__name__}."
            )
        persons: list[PersonObservation] = []
        for index, entry in enumerate(raw_persons):
            try:
                persons.append(PersonObservation.from_dict(entry))
            except PoseError as exc:
                raise FrameObservationError(
                    f"{name}: invalid person {index}: {exc}."
                ) from exc
        try:
            return cls(
                time_seconds=values["time_seconds"],
                timestamp_ms=values["timestamp_ms"],
                persons=tuple(persons),
                schema_version=values["schema_version"],
            )
        except (FrameObservationError, SchemaVersionError):
            raise
        except PoseError as exc:
            raise FrameObservationError(str(exc)) from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> FrameObservation:
        return cls.from_dict(_loads_object("frame_observation", data))


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """Identity binding a pose cache to its source, model, and sampling.

    ``source_fingerprint`` is the opaque ``sha256:`` digest of the source
    media; ``model_name``/``model_version`` pin the estimator; and
    ``sampling_rate_hz``/``sampling_start_seconds`` pin the M2.1 schedule
    (uniform ``[start, duration)`` grid at ``rate_hz``, independent of
    nominal FPS). Any mismatch means cached rows are stale and must never
    be treated as valid for the requested extraction.
    """

    source_fingerprint: str = ""
    model_name: str = ""
    model_version: str = ""
    sampling_rate_hz: float = 0.0
    sampling_start_seconds: float = 0.0
    schema_version: int = POSE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "cache_identity"
        version = self.schema_version
        if not _is_int(version):
            raise CacheIdentityError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != POSE_SCHEMA_VERSION:
            if version > POSE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {POSE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {POSE_SCHEMA_VERSION}."
            )
        try:
            _check_non_blank(name, "'source_fingerprint'", self.source_fingerprint)
            _check_non_blank(name, "'model_name'", self.model_name)
            _check_non_blank(name, "'model_version'", self.model_version)
        except PoseError as exc:
            raise CacheIdentityError(str(exc)) from exc
        if (
            isinstance(self.sampling_rate_hz, bool)
            or not isinstance(self.sampling_rate_hz, (int, float))
            or not math.isfinite(float(self.sampling_rate_hz))
            or float(self.sampling_rate_hz) <= 0
        ):
            raise CacheIdentityError(
                f"{name}: 'sampling_rate_hz' must be a finite number > 0, "
                f"got {self.sampling_rate_hz!r}."
            )
        if (
            isinstance(self.sampling_start_seconds, bool)
            or not isinstance(self.sampling_start_seconds, (int, float))
            or not math.isfinite(float(self.sampling_start_seconds))
            or float(self.sampling_start_seconds) < 0
        ):
            raise CacheIdentityError(
                f"{name}: 'sampling_start_seconds' must be a finite number "
                f">= 0, got {self.sampling_start_seconds!r}."
            )
        object.__setattr__(
            self, "sampling_rate_hz", float(self.sampling_rate_hz)
        )
        object.__setattr__(
            self, "sampling_start_seconds", float(self.sampling_start_seconds)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sampling_start_seconds": self.sampling_start_seconds,
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CacheIdentity:
        name = "cache_identity"
        if not isinstance(values, dict):
            raise CacheIdentityError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "model_name",
            "model_version",
            "sampling_rate_hz",
            "sampling_start_seconds",
            "schema_version",
            "source_fingerprint",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CacheIdentityError(
                f"{name}: missing required keys {missing!r}."
            )
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise CacheIdentityError(str(exc)) from exc
        try:
            _check_schema_version(name, values)
        except PoseError as exc:
            if isinstance(exc, SchemaVersionError):
                raise
            raise CacheIdentityError(str(exc)) from exc
        try:
            return cls(
                source_fingerprint=values["source_fingerprint"],
                model_name=values["model_name"],
                model_version=values["model_version"],
                sampling_rate_hz=values["sampling_rate_hz"],
                sampling_start_seconds=values["sampling_start_seconds"],
                schema_version=values["schema_version"],
            )
        except (CacheIdentityError, SchemaVersionError):
            raise
        except PoseError as exc:
            raise CacheIdentityError(str(exc)) from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> CacheIdentity:
        return cls.from_dict(_loads_object("cache_identity", data))
