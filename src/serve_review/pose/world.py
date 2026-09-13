"""Dense native MediaPipe world-landmark observation and cache (M4.7, leaf 1).

This module is the portable representation needed *before* native frame
decoding/extraction exists. It defines a versioned, source-bound
timestamped world-pose observation plus its JSONL cache, and nothing
else. It performs no frame decoding, no model execution, no filtering,
no waveform metrics, no checkpoint logic, no CLI, and no private-media
access.

Relationship to the frozen M2.2 sparse 2D baseline:

- :mod:`serve_review.pose.schema` (``pose-v1``) and
  :mod:`serve_review.pose.cache` are untouched and remain the only
  owners of sparse normalized 2D observations. This module imports the
  existing :class:`~serve_review.pose.schema.FrameObservation` as
  companion data; it never redefines 2D coordinates.
- The primary 3D coordinates are MediaPipe ``pose_world_landmarks``
  (meters, hip-centered, finite ``x``/``y``/``z``). They are *not*
  normalized to ``[0, 1]`` and must never be fabricated from 2D ``z``.
  A missing world joint is ``None`` (``null`` in JSON), never a
  zero-filled landmark.
- One :class:`WorldFrameObservation` holds exactly one timestamped
  single-person world pose (33 optional joints) plus its synchronized
  2D companion :class:`FrameObservation`. Multi-pose selection,
  extraction, and filtering are explicitly deferred.
- Cache identity carries ``representation == "dense-world-v1"`` and the
  header record type ``"dense-world-header"`` so dense-world caches
  never validate as sparse ``pose-v1`` caches (whose header type is
  ``"header"``) and vice versa.

File format (JSONL, one object per line, UTF-8, deterministic key order):

- line 1: ``{"type": "dense-world-header", ...}`` pinning the source
  fingerprint, model, representation, and schema versions;
- lines 2..N+1: ``{"type": "dense-world-frame", "observation": {...}}``
  with strictly increasing canonical ``time_seconds``;
- last line (complete caches only): ``{"type":
  "dense-world-footer", "schema_version": 1, "frame_count": N,
  "complete": true}``.

Completeness mirrors the 2D cache: complete only with a valid footer
whose ``frame_count`` matches and ``complete`` is true. Anything
malformed raises :class:`CacheCorruptError` (never complete); identity
mismatches raise :class:`CacheStaleError`.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve_review.pose.cache import (
    CacheCorruptError,
    CacheError,
    CacheStaleError,
)
from serve_review.pose.schema import (
    POSE_SCHEMA_VERSION,
    FrameObservation,
    NUM_KEYPOINTS,
    PoseError,
    SchemaVersionError,
)

__all__ = [
    "WORLD_SCHEMA_VERSION",
    "WORLD_CACHE_SCHEMA_VERSION",
    "WORLD_REPRESENTATION",
    "NUM_WORLD_LANDMARKS",
    "WorldLandmark",
    "WorldFrameObservation",
    "WorldCacheIdentity",
    "WorldCacheSnapshot",
    "world_header_to_dict",
    "world_header_from_dict",
    "require_matching_world_identity",
    "load_world_cache",
    "write_complete_world_cache",
    "write_partial_world_cache",
    "append_world_frames",
    "finalize_world_cache",
    "quarantine_world_cache",
]

#: Version of every world schema in this module.
WORLD_SCHEMA_VERSION = 1
#: Version of the dense-world JSONL envelope (header/footer framing).
WORLD_CACHE_SCHEMA_VERSION = 1
#: Representation tag distinguishing dense-world caches from sparse 2D pose-v1.
WORLD_REPRESENTATION = "dense-world-v1"
#: World joints per observation (MediaPipe pose_world_landmarks order, 0..32).
NUM_WORLD_LANDMARKS = 33

assert NUM_WORLD_LANDMARKS == NUM_KEYPOINTS


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_world_version(name: str, values: dict[str, Any]) -> int:
    if "schema_version" not in values:
        raise PoseError(f"{name}: missing required key 'schema_version'.")
    version = values["schema_version"]
    if not _is_int(version):
        raise PoseError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != WORLD_SCHEMA_VERSION:
        if version > WORLD_SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{name}: unsupported newer schema_version {version!r}; "
                f"this build supports version {WORLD_SCHEMA_VERSION}."
            )
        raise SchemaVersionError(
            f"{name}: unsupported schema_version {version!r}; "
            f"expected version {WORLD_SCHEMA_VERSION}."
        )
    return version


def _check_no_unknown_keys(name: str, values: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise PoseError(f"{name}: unknown keys {unknown!r}.")


def _check_finite_meters(name: str, key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise PoseError(
            f"{name}: {key!r} must be a finite number in meters, "
            f"got {value!r} (missing joints are null, never zeros)."
        )
    return float(value)


def _check_non_blank(name: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PoseError(
            f"{name}: {key!r} must be a non-blank string, got {value!r}."
        )
    return value


def _dumps_deterministic(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _dumps_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True) + "\n"


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
class WorldLandmark:
    """One MediaPipe ``pose_world_landmarks`` joint in meters.

    Coordinates are hip-centered meters (not normalized ``[0, 1]``); any
    finite float is accepted, including negatives. Missing joints are
    ``None`` in the enclosing observation, never a zero-filled
    landmark. No range clamping is applied.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        name = "world_landmark"
        try:
            x = _check_finite_meters(name, "'x'", self.x)
            y = _check_finite_meters(name, "'y'", self.y)
            z = _check_finite_meters(name, "'z'", self.z)
        except PoseError as exc:
            # Re-raise as PoseError with world context (catchable as PoseError).
            raise PoseError(str(exc)) from exc
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "z", z)

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> WorldLandmark:
        name = "world_landmark"
        if not isinstance(values, dict):
            raise PoseError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"x", "y", "z"}
        missing = sorted(known - set(values))
        if missing:
            raise PoseError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise PoseError(str(exc)) from exc
        try:
            return cls(x=values["x"], y=values["y"], z=values["z"])
        except PoseError:
            raise
        except (TypeError, ValueError) as exc:
            raise PoseError(f"{name}: invalid coordinates: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> WorldLandmark:
        return cls.from_dict(_loads_object("world_landmark", data))


@dataclass(frozen=True, slots=True)
class WorldFrameObservation:
    """One timestamped single-person dense-world observation.

    ``world_landmarks`` holds exactly ``NUM_WORLD_LANDMARKS`` entries in
    MediaPipe order; each entry is a :class:`WorldLandmark` or ``None``
    for an explicitly missing joint. All-``None`` world landmarks are
    allowed and explicitly record a no-world observation (the mapping
    layer only produces that together with an empty 2D companion;
    per-joint missingness otherwise stays per-joint).

    ``frame_2d`` is the synchronized normalized companion
    :class:`FrameObservation` (visibility included). Its canonical
    ``time_seconds``/``timestamp_ms`` must equal this observation's
    exactly; world/2D timestamp divergence is rejected rather than
    coerced.

    ``time_seconds`` is canonical source time; ``timestamp_ms`` must
    equal ``round(time_seconds * 1000)``.
    """

    time_seconds: float = 0.0
    timestamp_ms: int = 0
    world_landmarks: tuple[WorldLandmark | None, ...] = ()
    frame_2d: FrameObservation | None = None
    schema_version: int = WORLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "world_frame_observation"
        version = self.schema_version
        if not _is_int(version):
            raise PoseError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != WORLD_SCHEMA_VERSION:
            if version > WORLD_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {WORLD_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {WORLD_SCHEMA_VERSION}."
            )
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0
        ):
            raise PoseError(
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
            raise PoseError(
                f"{name}: 'timestamp_ms' must be an integer >= 0, "
                f"got {self.timestamp_ms!r}."
            )
        expected_ms = int(round(moment * 1000))
        if self.timestamp_ms != expected_ms:
            raise PoseError(
                f"{name}: 'timestamp_ms' ({self.timestamp_ms!r}) must equal "
                f"round(time_seconds * 1000) ({expected_ms!r})."
            )
        raw = self.world_landmarks
        if not isinstance(raw, (list, tuple)):
            raise PoseError(
                f"{name}: 'world_landmarks' must be a list or tuple of length "
                f"{NUM_WORLD_LANDMARKS}, got {type(raw).__name__}."
            )
        joints = tuple(raw)
        if len(joints) != NUM_WORLD_LANDMARKS:
            raise PoseError(
                f"{name}: 'world_landmarks' must hold exactly "
                f"{NUM_WORLD_LANDMARKS} entries in MediaPipe order, "
                f"got {len(joints)}."
            )
        for index, joint in enumerate(joints):
            if joint is not None and not isinstance(joint, WorldLandmark):
                raise PoseError(
                    f"{name}: world landmark {index} must be WorldLandmark "
                    f"or None, got {type(joint).__name__}."
                )
        object.__setattr__(self, "world_landmarks", joints)
        if not isinstance(self.frame_2d, FrameObservation):
            raise PoseError(
                f"{name}: 'frame_2d' must be a FrameObservation companion, "
                f"got {type(self.frame_2d).__name__}."
            )
        companion: FrameObservation = self.frame_2d
        if companion.time_seconds != moment:
            raise PoseError(
                f"{name}: world/2D timestamp mismatch: world time_seconds "
                f"{moment!r} != companion {companion.time_seconds!r}; "
                "world and 2D streams must share exact source time."
            )
        if companion.timestamp_ms != self.timestamp_ms:
            raise PoseError(
                f"{name}: world/2D timestamp_ms mismatch: world "
                f"{self.timestamp_ms!r} != companion "
                f"{companion.timestamp_ms!r}."
            )

    @property
    def world_present_count(self) -> int:
        """Return the number of non-missing world joints."""
        return sum(1 for joint in self.world_landmarks if joint is not None)

    @property
    def world_missing_count(self) -> int:
        """Return the number of missing world joints."""
        return NUM_WORLD_LANDMARKS - self.world_present_count

    @property
    def has_world(self) -> bool:
        """Return True when at least one world joint is present."""
        return any(joint is not None for joint in self.world_landmarks)

    @property
    def has_person(self) -> bool:
        """Return True when the 2D companion holds at least one person."""
        assert isinstance(self.frame_2d, FrameObservation)
        return self.frame_2d.has_person

    def to_dict(self) -> dict[str, Any]:
        assert isinstance(self.frame_2d, FrameObservation)
        return {
            "frame_2d": self.frame_2d.to_dict(),
            "schema_version": self.schema_version,
            "time_seconds": self.time_seconds,
            "timestamp_ms": self.timestamp_ms,
            "world_landmarks": [
                joint.to_dict() if joint is not None else None
                for joint in self.world_landmarks
            ],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> WorldFrameObservation:
        name = "world_frame_observation"
        if not isinstance(values, dict):
            raise PoseError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "frame_2d",
            "schema_version",
            "time_seconds",
            "timestamp_ms",
            "world_landmarks",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PoseError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise PoseError(str(exc)) from exc
        try:
            _check_world_version(name, values)
        except PoseError:
            raise
        raw_joints = values["world_landmarks"]
        if not isinstance(raw_joints, list):
            raise PoseError(
                f"{name}: 'world_landmarks' must be a JSON list of length "
                f"{NUM_WORLD_LANDMARKS} with objects or nulls, "
                f"got {type(raw_joints).__name__}."
            )
        if len(raw_joints) != NUM_WORLD_LANDMARKS:
            raise PoseError(
                f"{name}: 'world_landmarks' must hold exactly "
                f"{NUM_WORLD_LANDMARKS} entries, got {len(raw_joints)}."
            )
        joints: list[WorldLandmark | None] = []
        for index, entry in enumerate(raw_joints):
            if entry is None:
                joints.append(None)
                continue
            try:
                joints.append(WorldLandmark.from_dict(entry))
            except PoseError as exc:
                raise PoseError(
                    f"{name}: invalid world landmark {index}: {exc}."
                ) from exc
        try:
            companion = FrameObservation.from_dict(values["frame_2d"])
        except PoseError as exc:
            raise PoseError(f"{name}: invalid 2D companion: {exc}.") from exc
        try:
            return cls(
                time_seconds=values["time_seconds"],
                timestamp_ms=values["timestamp_ms"],
                world_landmarks=tuple(joints),
                frame_2d=companion,
                schema_version=values["schema_version"],
            )
        except (PoseError, SchemaVersionError):
            raise
        except (TypeError, ValueError) as exc:
            raise PoseError(f"{name}: invalid observation: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> WorldFrameObservation:
        return cls.from_dict(_loads_object("world_frame_observation", data))


@dataclass(frozen=True, slots=True)
class WorldCacheIdentity:
    """Identity binding a dense-world cache to its source and model.

    ``source_fingerprint`` is the opaque ``sha256:`` digest of the
    source media; ``model_name``/``model_version`` pin the estimator;
    ``representation`` must equal :data:`WORLD_REPRESENTATION`, which is
    what distinguishes dense-world caches from sparse 2D ``pose-v1``
    caches at both the header and the identity layer. Any mismatch
    means cached rows are stale and must never be reused.
    """

    source_fingerprint: str = ""
    model_name: str = ""
    model_version: str = ""
    representation: str = WORLD_REPRESENTATION
    schema_version: int = WORLD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "world_cache_identity"
        version = self.schema_version
        if not _is_int(version):
            raise PoseError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != WORLD_SCHEMA_VERSION:
            if version > WORLD_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {WORLD_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {WORLD_SCHEMA_VERSION}."
            )
        try:
            _check_non_blank(name, "'source_fingerprint'", self.source_fingerprint)
            _check_non_blank(name, "'model_name'", self.model_name)
            _check_non_blank(name, "'model_version'", self.model_version)
        except PoseError as exc:
            raise PoseError(str(exc)) from exc
        if self.representation != WORLD_REPRESENTATION:
            raise PoseError(
                f"{name}: 'representation' must equal "
                f"{WORLD_REPRESENTATION!r}, got {self.representation!r}."
            )
        object.__setattr__(self, "representation", str(self.representation))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "representation": self.representation,
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> WorldCacheIdentity:
        name = "world_cache_identity"
        if not isinstance(values, dict):
            raise PoseError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "model_name",
            "model_version",
            "representation",
            "schema_version",
            "source_fingerprint",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PoseError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except PoseError as exc:
            raise PoseError(str(exc)) from exc
        try:
            _check_world_version(name, values)
        except PoseError:
            raise
        try:
            return cls(
                source_fingerprint=values["source_fingerprint"],
                model_name=values["model_name"],
                model_version=values["model_version"],
                representation=values["representation"],
                schema_version=values["schema_version"],
            )
        except (PoseError, SchemaVersionError):
            raise
        except (TypeError, ValueError) as exc:
            raise PoseError(f"{name}: invalid identity: {exc}.") from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> WorldCacheIdentity:
        return cls.from_dict(_loads_object("world_cache_identity", data))


@dataclass(frozen=True, slots=True)
class WorldCacheSnapshot:
    """Result of :func:`load_world_cache`."""

    identity: WorldCacheIdentity
    frames: tuple[WorldFrameObservation, ...]
    complete: bool


def world_header_to_dict(identity: WorldCacheIdentity) -> dict[str, Any]:
    """Serialize ``identity`` as a dense-world header record."""
    if not isinstance(identity, WorldCacheIdentity):
        raise CacheError(
            "invalid world cache identity: expected WorldCacheIdentity, "
            f"got {type(identity).__name__}."
        )
    return {
        "model_name": identity.model_name,
        "model_version": identity.model_version,
        "pose_schema_version": POSE_SCHEMA_VERSION,
        "representation": WORLD_REPRESENTATION,
        "schema_version": WORLD_CACHE_SCHEMA_VERSION,
        "source_fingerprint": identity.source_fingerprint,
        "type": "dense-world-header",
        "world_schema_version": WORLD_SCHEMA_VERSION,
    }


def world_header_from_dict(values: dict[str, Any]) -> WorldCacheIdentity:
    """Parse and version-check a dense-world header record."""
    name = "world_cache/header"
    if not isinstance(values, dict):
        raise CacheCorruptError(
            f"{name}: header record must be a JSON object, "
            f"got {type(values).__name__}."
        )
    known = {
        "model_name",
        "model_version",
        "pose_schema_version",
        "representation",
        "schema_version",
        "source_fingerprint",
        "type",
        "world_schema_version",
    }
    unknown = sorted(set(values) - known)
    if unknown:
        raise CacheCorruptError(f"{name}: unknown keys {unknown!r}.")
    if values.get("type") != "dense-world-header":
        raise CacheCorruptError(
            f"{name}: first record must be a dense-world header "
            f"({{'type': 'dense-world-header', ...}}), got "
            f"{values.get('type')!r}; sparse 2D pose-v1 caches "
            "(\"header\") are never valid dense-world caches."
        )
    if values.get("schema_version") != WORLD_CACHE_SCHEMA_VERSION:
        raise CacheCorruptError(
            f"{name}: unsupported cache schema_version "
            f"{values.get('schema_version')!r}; this build supports version "
            f"{WORLD_CACHE_SCHEMA_VERSION}."
        )
    if values.get("world_schema_version") != WORLD_SCHEMA_VERSION:
        raise CacheCorruptError(
            f"{name}: unsupported world_schema_version "
            f"{values.get('world_schema_version')!r}; this build supports "
            f"version {WORLD_SCHEMA_VERSION}."
        )
    if values.get("pose_schema_version") != POSE_SCHEMA_VERSION:
        raise CacheCorruptError(
            f"{name}: unsupported pose_schema_version "
            f"{values.get('pose_schema_version')!r}; this build supports "
            f"version {POSE_SCHEMA_VERSION}."
        )
    if values.get("representation") != WORLD_REPRESENTATION:
        raise CacheCorruptError(
            f"{name}: unsupported representation "
            f"{values.get('representation')!r}; expected "
            f"{WORLD_REPRESENTATION!r}."
        )
    try:
        return WorldCacheIdentity(
            source_fingerprint=values["source_fingerprint"],
            model_name=values["model_name"],
            model_version=values["model_version"],
            representation=values["representation"],
        )
    except (PoseError, KeyError, TypeError) as exc:
        raise CacheCorruptError(f"{name}: invalid identity: {exc}.") from exc


def _world_footer_to_dict(frame_count: int) -> dict[str, Any]:
    return {
        "complete": True,
        "frame_count": int(frame_count),
        "schema_version": WORLD_CACHE_SCHEMA_VERSION,
        "type": "dense-world-footer",
    }


def _world_frame_to_dict(observation: WorldFrameObservation) -> dict[str, Any]:
    if not isinstance(observation, WorldFrameObservation):
        raise CacheError(
            "invalid world frame observation: expected "
            f"WorldFrameObservation, got {type(observation).__name__}."
        )
    return {"observation": observation.to_dict(), "type": "dense-world-frame"}


def require_matching_world_identity(
    stored: WorldCacheIdentity, expected: WorldCacheIdentity
) -> None:
    """Raise :class:`CacheStaleError` when any identity field differs."""
    if not isinstance(stored, WorldCacheIdentity):
        raise CacheError(
            "invalid stored identity: expected WorldCacheIdentity, "
            f"got {type(stored).__name__}."
        )
    if not isinstance(expected, WorldCacheIdentity):
        raise CacheError(
            "invalid expected identity: expected WorldCacheIdentity, "
            f"got {type(expected).__name__}."
        )
    fields = (
        "source_fingerprint",
        "model_name",
        "model_version",
        "representation",
    )
    mismatched = [
        field
        for field in fields
        if getattr(stored, field) != getattr(expected, field)
    ]
    if mismatched:
        details = ", ".join(
            f"{field} (cached={getattr(stored, field)!r} "
            f"!= requested={getattr(expected, field)!r})"
            for field in mismatched
        )
        raise CacheStaleError(
            "dense-world cache identity is stale: "
            f"{details}. Re-extract rather than reusing cached rows."
        )


def _check_world_frame_order(
    frames: list[WorldFrameObservation], name: str
) -> None:
    for earlier, later in zip(frames, frames[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise CacheCorruptError(
                f"{name}: frame times must be strictly increasing, got "
                f"{earlier.time_seconds!r} followed by {later.time_seconds!r}."
            )


def load_world_cache(path: Path | str) -> WorldCacheSnapshot:
    """Load and validate the dense-world cache at ``path``."""
    target = Path(path).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CacheError(f"dense-world cache does not exist: {target}.") from exc
    except OSError as exc:
        raise CacheError(
            f"could not read dense-world cache {target}: {exc}."
        ) from exc
    name = f"world_cache/{target.name}"
    if not text.strip():
        raise CacheCorruptError(
            f"{name}: cache file is empty; no dense-world header found."
        )
    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise CacheCorruptError(
                f"{name}: line {lineno} is blank; cache records must be "
                "dense JSONL with no blank lines."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CacheCorruptError(
                f"{name}: line {lineno} is not valid JSON ({exc})."
            ) from exc
        if not isinstance(record, dict):
            raise CacheCorruptError(
                f"{name}: line {lineno} must be a JSON object, "
                f"got {type(record).__name__}."
            )
        records.append(record)
    if not records or records[0].get("type") != "dense-world-header":
        raise CacheCorruptError(
            f"{name}: first record must be a dense-world header "
            "({'type': 'dense-world-header', ...}); sparse 2D pose-v1 "
            "caches are never valid here."
        )
    identity = world_header_from_dict(records[0])
    frames: list[WorldFrameObservation] = []
    footer: dict[str, Any] | None = None
    for lineno, record in enumerate(records[1:], start=2):
        kind = record.get("type")
        if kind == "dense-world-footer":
            if footer is not None:
                raise CacheCorruptError(
                    f"{name}: line {lineno} carries a second footer."
                )
            footer = record
            continue
        if footer is not None:
            raise CacheCorruptError(
                f"{name}: line {lineno} follows the footer; records must "
                "end at the footer."
            )
        if kind != "dense-world-frame":
            raise CacheCorruptError(
                f"{name}: line {lineno} has unknown record type {kind!r}; "
                "expected 'dense-world-frame' or 'dense-world-footer'."
            )
        if set(record) - {"observation", "type"}:
            raise CacheCorruptError(
                f"{name}: line {lineno} has unknown frame keys "
                f"{sorted(set(record) - {'observation', 'type'})!r}."
            )
        if "observation" not in record:
            raise CacheCorruptError(
                f"{name}: line {lineno} frame record is missing 'observation'."
            )
        payload = record["observation"]
        if not isinstance(payload, dict):
            raise CacheCorruptError(
                f"{name}: line {lineno} observation must be a JSON object."
            )
        try:
            frames.append(WorldFrameObservation.from_dict(payload))
        except PoseError as exc:
            raise CacheCorruptError(
                f"{name}: line {lineno} carries an invalid world frame "
                f"observation: {exc}."
            ) from exc
    _check_world_frame_order(frames, name)
    if footer is None:
        return WorldCacheSnapshot(
            identity=identity, frames=tuple(frames), complete=False
        )
    known_footer = {"complete", "frame_count", "schema_version", "type"}
    unknown_footer = sorted(set(footer) - known_footer)
    if unknown_footer:
        raise CacheCorruptError(
            f"{name}: footer has unknown keys {unknown_footer!r}."
        )
    if footer.get("schema_version") != WORLD_CACHE_SCHEMA_VERSION:
        raise CacheCorruptError(
            f"{name}: unsupported footer schema_version "
            f"{footer.get('schema_version')!r}."
        )
    if footer.get("type") != "dense-world-footer":
        raise CacheCorruptError(
            f"{name}: invalid footer type {footer.get('type')!r}."
        )
    frame_count = footer.get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 0
    ):
        raise CacheCorruptError(
            f"{name}: footer 'frame_count' must be an integer >= 0, "
            f"got {frame_count!r}."
        )
    if frame_count != len(frames):
        raise CacheCorruptError(
            f"{name}: footer frame_count ({frame_count!r}) does not match "
            f"the {len(frames)} stored frame(s)."
        )
    if footer.get("complete") is True:
        return WorldCacheSnapshot(
            identity=identity, frames=tuple(frames), complete=True
        )
    if footer.get("complete") is False:
        return WorldCacheSnapshot(
            identity=identity, frames=tuple(frames), complete=False
        )
    raise CacheCorruptError(
        f"{name}: footer 'complete' must be true or false, "
        f"got {footer.get('complete')!r}."
    )


def _write_world_snapshot_atomic(
    target: Path,
    identity: WorldCacheIdentity,
    frames: list[WorldFrameObservation],
    *,
    complete: bool,
) -> None:
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheError(
                f"could not create dense-world cache directory {parent}: {exc}."
            ) from exc
    _check_world_frame_order(list(frames), f"world_cache/{target.name}")
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=target.name + ".tmp-",
            suffix=".jsonl",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(_dumps_line(world_header_to_dict(identity)))
            for observation in frames:
                handle.write(_dumps_line(_world_frame_to_dict(observation)))
            if complete:
                handle.write(_dumps_line(_world_footer_to_dict(len(frames))))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
    except CacheError:
        if tmp_path is not None:
            _remove_quietly(tmp_path)
        raise
    except (OSError, TypeError, ValueError) as exc:
        if tmp_path is not None:
            _remove_quietly(tmp_path)
        raise CacheError(
            f"could not write dense-world cache to {target}: {exc}."
        ) from exc


def _remove_quietly(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def write_complete_world_cache(
    path: Path | str,
    identity: WorldCacheIdentity,
    frames: list[WorldFrameObservation] | tuple[WorldFrameObservation, ...],
) -> Path:
    """Atomically write a complete dense-world cache with footer."""
    target = Path(path).expanduser()
    items = list(frames)
    for entry in items:
        if not isinstance(entry, WorldFrameObservation):
            raise CacheError(
                "invalid frames: expected WorldFrameObservation entries, "
                f"got {type(entry).__name__}."
            )
    _write_world_snapshot_atomic(target, identity, items, complete=True)
    return target


def write_partial_world_cache(
    path: Path | str,
    identity: WorldCacheIdentity,
    frames: list[WorldFrameObservation] | tuple[WorldFrameObservation, ...],
) -> Path:
    """Atomically write an incomplete (resumable, footerless) cache."""
    target = Path(path).expanduser()
    items = list(frames)
    for entry in items:
        if not isinstance(entry, WorldFrameObservation):
            raise CacheError(
                "invalid frames: expected WorldFrameObservation entries, "
                f"got {type(entry).__name__}."
            )
    _write_world_snapshot_atomic(target, identity, items, complete=False)
    return target


def append_world_frames(
    path: Path | str,
    identity: WorldCacheIdentity,
    new_frames: list[WorldFrameObservation] | tuple[WorldFrameObservation, ...],
) -> Path:
    """Resume an incomplete dense-world cache by appending ``new_frames``."""
    target = Path(path).expanduser()
    additions = list(new_frames)
    for entry in additions:
        if not isinstance(entry, WorldFrameObservation):
            raise CacheError(
                "invalid new_frames: expected WorldFrameObservation entries, "
                f"got {type(entry).__name__}."
            )
    snapshot = load_world_cache(target)
    require_matching_world_identity(snapshot.identity, identity)
    if snapshot.complete:
        raise CacheError(
            f"dense-world cache {target} is already complete with "
            f"{len(snapshot.frames)} frame(s); refusing to append."
        )
    combined = list(snapshot.frames) + additions
    _check_world_frame_order(combined, f"world_cache/{target.name}")
    _write_world_snapshot_atomic(target, identity, combined, complete=False)
    return target


def finalize_world_cache(path: Path | str, identity: WorldCacheIdentity) -> Path:
    """Mark a resumable dense-world cache complete without adding frames."""
    target = Path(path).expanduser()
    snapshot = load_world_cache(target)
    require_matching_world_identity(snapshot.identity, identity)
    if snapshot.complete:
        return target
    _write_world_snapshot_atomic(
        target, identity, list(snapshot.frames), complete=True
    )
    return target


def quarantine_world_cache(path: Path | str) -> Path:
    """Move a corrupt/incompatible dense-world cache aside."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise CacheError(f"dense-world cache does not exist: {target}.")
    candidate = target.with_name(target.name + ".corrupt")
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = target.with_name(f"{target.name}.corrupt.{counter}")
    try:
        os.replace(target, candidate)
    except OSError as exc:
        raise CacheError(
            f"could not quarantine corrupt dense-world cache {target}: {exc}."
        ) from exc
    return candidate
