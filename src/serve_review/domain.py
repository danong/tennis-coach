"""Versioned media domain schemas.

This module is intentionally pure: no file I/O, no subprocesses, no framework
objects. It defines immutable source metadata, half-open media ranges, and
export plans with deterministic JSON codecs and strict validation.

Ranges are half-open ``[start_seconds, end_seconds)`` measured in decimal
seconds of the source timeline. Bounds are finite, nonnegative, and satisfy
``start < end``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = [
    "SOURCE_SCHEMA_VERSION",
    "RANGE_SCHEMA_VERSION",
    "EXPORT_PLAN_SCHEMA_VERSION",
    "ATTEMPT_SCHEMA_VERSION",
    "ATTEMPT_DOCUMENT_SCHEMA_VERSION",
    "PHASE_SCHEMA_VERSION",
    "STAGE_ORDER",
    "STAGE_KEYS",
    "STAGE_AVAILABILITIES",
    "STAGE_PROVENANCES",
    "STRUCTURAL_STATUSES",
    "DomainError",
    "SchemaVersionError",
    "SourceMetadataError",
    "RangeError",
    "ExportPlanError",
    "AttemptError",
    "PhaseError",
    "SourceMetadata",
    "MediaRange",
    "ExportPlan",
    "Attempt",
    "AttemptDocument",
    "StagePhase",
    "AttemptPhase",
    "PhaseDocument",
]

SOURCE_SCHEMA_VERSION = 1
RANGE_SCHEMA_VERSION = 1
EXPORT_PLAN_SCHEMA_VERSION = 1
ATTEMPT_SCHEMA_VERSION = 1
ATTEMPT_DOCUMENT_SCHEMA_VERSION = 1

_ROTATIONS = (0, 90, 180, 270)


class DomainError(ValueError):
    """Base error for versioned media domain validation failures."""


class SchemaVersionError(DomainError):
    """Raised when a document carries an unsupported schema version."""


class SourceMetadataError(DomainError):
    """Raised when source metadata is invalid."""


class RangeError(DomainError):
    """Raised when a media range is invalid or cannot be applied."""


class ExportPlanError(DomainError):
    """Raised when an export plan is invalid."""


class AttemptError(DomainError):
    """Raised when an attempt or attempt document is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_schema_version(name: str, values: dict[str, Any], current: int) -> int:
    if "schema_version" not in values:
        raise DomainError(f"{name}: missing required key 'schema_version'.")
    version = values["schema_version"]
    if not _is_int(version):
        raise DomainError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != current:
        if version > current:
            raise SchemaVersionError(
                f"{name}: unsupported newer schema_version {version!r}; "
                f"this build supports version {current}."
            )
        raise SchemaVersionError(
            f"{name}: unsupported schema_version {version!r}; "
            f"expected version {current}."
        )
    return version


def _check_no_unknown_keys(name: str, values: dict[str, Any], known: set[str]) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise DomainError(f"{name}: unknown keys {unknown!r}.")


def _check_finite_positive(name: str, key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value) or value <= 0:
        raise DomainError(
            f"{name}: {key!r} must be a finite number greater than zero, "
            f"got {value!r}."
        )
    return float(value)


def _check_non_blank(name: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DomainError(
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
            raise DomainError(f"{name}: invalid UTF-8 JSON payload.") from exc
    if not isinstance(data, str):
        raise DomainError(
            f"{name}: JSON payload must be str or bytes, got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise DomainError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise DomainError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Immutable normalized description of one source video file.

    The fingerprint is an opaque caller-supplied identity (for example a
    ``sha256:`` digest computed where file I/O is allowed); this module never
    computes it. Frame rate and time base are preserved as rational
    numerator/denominator pairs so canonical source time never depends on an
    assumed frames-per-second value.
    """

    schema_version: int = SOURCE_SCHEMA_VERSION
    fingerprint: str = ""
    duration_seconds: float = 0.0
    width: int = 0
    height: int = 0
    frame_rate_num: int = 0
    frame_rate_den: int = 1
    video_codec: str = ""
    rotation_degrees: int = 0
    time_base_num: int | None = None
    time_base_den: int | None = None

    def __post_init__(self) -> None:
        name = "source_metadata"
        version = self.schema_version
        if not _is_int(version):
            raise SourceMetadataError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != SOURCE_SCHEMA_VERSION:
            if version > SOURCE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {SOURCE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {SOURCE_SCHEMA_VERSION}."
            )
        try:
            _check_non_blank(name, "fingerprint", self.fingerprint)
        except DomainError as exc:
            raise SourceMetadataError(str(exc)) from exc
        object.__setattr__(
            self,
            "duration_seconds",
            _checked_duration(name, self.duration_seconds),
        )
        for key in ("width", "height"):
            value = getattr(self, key)
            if not _is_int(value) or value <= 0:
                raise SourceMetadataError(
                    f"{name}: {key!r} must be a positive integer, got {value!r}."
                )
        for key in ("frame_rate_num", "frame_rate_den"):
            value = getattr(self, key)
            if not _is_int(value) or value <= 0:
                raise SourceMetadataError(
                    f"{name}: {key!r} must be a positive integer, got {value!r}."
                )
        try:
            _check_non_blank(name, "video_codec", self.video_codec)
        except DomainError as exc:
            raise SourceMetadataError(str(exc)) from exc
        if not _is_int(self.rotation_degrees) or self.rotation_degrees not in _ROTATIONS:
            raise SourceMetadataError(
                f"{name}: 'rotation_degrees' must be one of {list(_ROTATIONS)}, "
                f"got {self.rotation_degrees!r}."
            )
        time_fields = (self.time_base_num, self.time_base_den)
        if (time_fields[0] is None) != (time_fields[1] is None):
            raise SourceMetadataError(
                f"{name}: 'time_base_num' and 'time_base_den' must both be "
                f"set or both be null, got {time_fields!r}."
            )
        for key, value in (
            ("time_base_num", self.time_base_num),
            ("time_base_den", self.time_base_den),
        ):
            if value is not None and (not _is_int(value) or value <= 0):
                raise SourceMetadataError(
                    f"{name}: {key!r} must be a positive integer or null, "
                    f"got {value!r}."
                )

    @property
    def frames_per_second(self) -> float:
        """Nominal frame rate as a float; never canonical source time."""
        return self.frame_rate_num / self.frame_rate_den

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "fingerprint": self.fingerprint,
            "frame_rate_den": self.frame_rate_den,
            "frame_rate_num": self.frame_rate_num,
            "height": self.height,
            "rotation_degrees": self.rotation_degrees,
            "schema_version": self.schema_version,
            "time_base_den": self.time_base_den,
            "time_base_num": self.time_base_num,
            "video_codec": self.video_codec,
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SourceMetadata:
        name = "source_metadata"
        if not isinstance(values, dict):
            raise SourceMetadataError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "duration_seconds",
            "fingerprint",
            "frame_rate_den",
            "frame_rate_num",
            "height",
            "rotation_degrees",
            "schema_version",
            "time_base_den",
            "time_base_num",
            "video_codec",
            "width",
        }
        missing = sorted(known - set(values))
        if missing:
            raise SourceMetadataError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except DomainError as exc:
            raise SourceMetadataError(str(exc)) from exc
        try:
            _check_schema_version(name, values, SOURCE_SCHEMA_VERSION)
        except DomainError as exc:
            raise _regrade_source_error(exc) from exc
        try:
            return cls(
                schema_version=values["schema_version"],
                fingerprint=values["fingerprint"],
                duration_seconds=values["duration_seconds"],
                width=values["width"],
                height=values["height"],
                frame_rate_num=values["frame_rate_num"],
                frame_rate_den=values["frame_rate_den"],
                video_codec=values["video_codec"],
                rotation_degrees=values["rotation_degrees"],
                time_base_num=values["time_base_num"],
                time_base_den=values["time_base_den"],
            )
        except (SourceMetadataError, SchemaVersionError):
            raise
        except DomainError as exc:
            raise SourceMetadataError(str(exc)) from exc

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> SourceMetadata:
        return cls.from_dict(_loads_object("source_metadata", data))


def _checked_duration(name: str, value: Any) -> float:
    try:
        return _check_finite_positive(name, "'duration_seconds'", value)
    except DomainError as exc:
        if name == "source_metadata":
            raise SourceMetadataError(str(exc)) from exc
        raise


def _regrade_source_error(exc: DomainError) -> DomainError:
    if isinstance(exc, SchemaVersionError):
        return exc
    return SourceMetadataError(str(exc))


def _check_range_bound(name: str, key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value) or value < 0:
        raise RangeError(
            f"{name}: {key!r} must be a finite number greater than or equal "
            f"to zero, got {value!r}."
        )
    return float(value)


@dataclass(frozen=True, slots=True)
class MediaRange:
    """Half-open ``[start_seconds, end_seconds)`` source-time interval."""

    start_seconds: float = field(default=0.0)
    end_seconds: float = field(default=0.0)
    schema_version: int = field(default=RANGE_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        name = "media_range"
        version = self.schema_version
        if not _is_int(version):
            raise RangeError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != RANGE_SCHEMA_VERSION:
            if version > RANGE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {RANGE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {RANGE_SCHEMA_VERSION}."
            )
        start = _check_range_bound(name, "'start_seconds'", self.start_seconds)
        end = _check_range_bound(name, "'end_seconds'", self.end_seconds)
        if not start < end:
            raise RangeError(
                f"{name}: 'start_seconds' ({start!r}) must be less than "
                f"'end_seconds' ({end!r}) for a half-open range."
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    def _key(self) -> tuple[float, float]:
        return (self.start_seconds, self.end_seconds)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, MediaRange):
            return NotImplemented
        return self._key() < other._key()

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, MediaRange):
            return NotImplemented
        return self._key() <= other._key()

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, MediaRange):
            return NotImplemented
        return self._key() > other._key()

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, MediaRange):
            return NotImplemented
        return self._key() >= other._key()

    def contains(self, moment_seconds: Any) -> bool:
        """Return True when ``moment`` lies in ``[start, end)``."""
        if not _is_number(moment_seconds) or not math.isfinite(moment_seconds):
            return False
        moment = float(moment_seconds)
        return self.start_seconds <= moment < self.end_seconds

    def overlaps(self, other: MediaRange) -> bool:
        """Half-open overlap; merely adjacent ranges do not overlap."""
        if not isinstance(other, MediaRange):
            raise RangeError(
                f"media_range: can only compare against MediaRange, "
                f"got {type(other).__name__}."
            )
        return (
            max(self.start_seconds, other.start_seconds)
            < min(self.end_seconds, other.end_seconds)
        )

    def is_adjacent(self, other: MediaRange) -> bool:
        """Return True when the ranges touch without overlapping."""
        if not isinstance(other, MediaRange):
            raise RangeError(
                f"media_range: can only compare against MediaRange, "
                f"got {type(other).__name__}."
            )
        return (
            self.end_seconds == other.start_seconds
            or other.end_seconds == self.start_seconds
        )

    def union(self, other: MediaRange) -> MediaRange:
        """Merge overlapping or adjacent ranges into one minimal range."""
        if not isinstance(other, MediaRange):
            raise RangeError(
                f"media_range: can only merge with MediaRange, "
                f"got {type(other).__name__}."
            )
        if not self.overlaps(other) and not self.is_adjacent(other):
            raise RangeError(
                f"media_range: cannot merge disjoint ranges "
                f"{self.to_dict()!r} and {other.to_dict()!r}."
            )
        return MediaRange(
            start_seconds=min(self.start_seconds, other.start_seconds),
            end_seconds=max(self.end_seconds, other.end_seconds),
        )

    def clamp(self, source_duration_seconds: Any) -> MediaRange:
        """Clamp this range to ``[0, source_duration]``.

        Raises :class:`RangeError` when the duration is invalid or when no
        part of the range lies inside the source bounds.
        """
        name = "media_range"
        if (
            not _is_number(source_duration_seconds)
            or not math.isfinite(source_duration_seconds)
            or source_duration_seconds <= 0
        ):
            raise RangeError(
                f"{name}: source duration must be a finite number greater "
                f"than zero, got {source_duration_seconds!r}."
            )
        duration = float(source_duration_seconds)
        if self.start_seconds >= duration:
            raise RangeError(
                f"{name}: range start ({self.start_seconds!r}) lies beyond "
                f"source duration ({duration!r})."
            )
        return MediaRange(
            start_seconds=max(0.0, self.start_seconds),
            end_seconds=min(self.end_seconds, duration),
        )

    def with_padding(
        self, pad_seconds: Any, source_duration_seconds: Any
    ) -> MediaRange:
        """Expand symmetrically by ``pad_seconds`` and clamp to the source."""
        name = "media_range"
        if (
            not _is_number(pad_seconds)
            or not math.isfinite(pad_seconds)
            or pad_seconds < 0
        ):
            raise RangeError(
                f"{name}: padding must be a finite number greater than or "
                f"equal to zero, got {pad_seconds!r}."
            )
        if (
            not _is_number(source_duration_seconds)
            or not math.isfinite(source_duration_seconds)
            or source_duration_seconds <= 0
        ):
            raise RangeError(
                f"{name}: source duration must be a finite number greater "
                f"than zero, got {source_duration_seconds!r}."
            )
        pad = float(pad_seconds)
        duration = float(source_duration_seconds)
        return MediaRange(
            start_seconds=max(0.0, self.start_seconds - pad),
            end_seconds=min(duration, self.end_seconds + pad),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_seconds": self.end_seconds,
            "schema_version": self.schema_version,
            "start_seconds": self.start_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MediaRange:
        name = "media_range"
        if not isinstance(values, dict):
            raise RangeError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"end_seconds", "schema_version", "start_seconds"}
        missing = sorted(known - set(values))
        if missing:
            raise RangeError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        _check_schema_version(name, values, RANGE_SCHEMA_VERSION)
        return cls(
            start_seconds=values["start_seconds"],
            end_seconds=values["end_seconds"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> MediaRange:
        return cls.from_dict(_loads_object("media_range", data))


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """Validated, ordered set of ranges to export from one source.

    Ranges must be sorted by ``(start, end)``, pairwise non-overlapping
    (adjacency is allowed), and fully inside ``[0, source_duration]``.
    """

    source_fingerprint: str = ""
    source_duration_seconds: float = 0.0
    ranges: tuple[MediaRange, ...] = ()
    schema_version: int = EXPORT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "export_plan"
        version = self.schema_version
        if not _is_int(version):
            raise ExportPlanError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != EXPORT_PLAN_SCHEMA_VERSION:
            if version > EXPORT_PLAN_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {EXPORT_PLAN_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {EXPORT_PLAN_SCHEMA_VERSION}."
            )
        _check_non_blank(name, "source_fingerprint", self.source_fingerprint)
        try:
            duration = _check_finite_positive(
                name, "'source_duration_seconds'", self.source_duration_seconds
            )
        except DomainError as exc:
            raise ExportPlanError(str(exc)) from exc
        object.__setattr__(self, "source_duration_seconds", duration)

        raw = self.ranges
        if isinstance(raw, MediaRange):
            raise ExportPlanError(
                f"{name}: 'ranges' must be a sequence of MediaRange, "
                f"got a single MediaRange."
            )
        if not isinstance(raw, (list, tuple)):
            raise ExportPlanError(
                f"{name}: 'ranges' must be a list or tuple of MediaRange, "
                f"got {type(raw).__name__}."
            )
        normalized = tuple(raw)
        if not normalized:
            raise ExportPlanError(f"{name}: at least one range is required.")
        for entry in normalized:
            if not isinstance(entry, MediaRange):
                raise ExportPlanError(
                    f"{name}: every range must be a MediaRange, "
                    f"got {type(entry).__name__}."
                )
            if entry.end_seconds > duration:
                raise ExportPlanError(
                    f"{name}: range {entry.to_dict()!r} extends beyond source "
                    f"duration ({duration!r}); clamp it first."
                )
        object.__setattr__(self, "ranges", normalized)
        for first, second in zip(normalized, normalized[1:]):
            if second._key() < first._key():
                raise ExportPlanError(
                    f"{name}: ranges must be sorted by (start, end); "
                    f"{first.to_dict()!r} precedes {second.to_dict()!r}."
                )
            if first.overlaps(second):
                raise ExportPlanError(
                    f"{name}: ranges must not overlap; "
                    f"{first.to_dict()!r} overlaps {second.to_dict()!r}."
                )

    @classmethod
    def for_source(
        cls, source: SourceMetadata, ranges: list[MediaRange] | tuple[MediaRange, ...]
    ) -> ExportPlan:
        """Build a strictly validated plan for ``source``."""
        if not isinstance(source, SourceMetadata):
            raise ExportPlanError(
                "export_plan: 'source' must be SourceMetadata, "
                f"got {type(source).__name__}."
            )
        return cls(
            source_fingerprint=source.fingerprint,
            source_duration_seconds=source.duration_seconds,
            ranges=tuple(ranges),
        )

    @classmethod
    def clamped(
        cls, source: SourceMetadata, ranges: list[MediaRange] | tuple[MediaRange, ...]
    ) -> ExportPlan:
        """Clamp each range to the source duration, then strictly validate."""
        if not isinstance(source, SourceMetadata):
            raise ExportPlanError(
                "export_plan: 'source' must be SourceMetadata, "
                f"got {type(source).__name__}."
            )
        if isinstance(ranges, MediaRange) or not isinstance(ranges, (list, tuple)):
            raise ExportPlanError(
                "export_plan: 'ranges' must be a list or tuple of MediaRange, "
                f"got {type(ranges).__name__}."
            )
        return cls.for_source(
            source, [entry.clamp(source.duration_seconds) for entry in ranges]
        )

    @property
    def total_duration_seconds(self) -> float:
        return sum(entry.duration_seconds for entry in self.ranges)

    def __len__(self) -> int:
        return len(self.ranges)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.ranges)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.ranges[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranges": [entry.to_dict() for entry in self.ranges],
            "schema_version": self.schema_version,
            "source_duration_seconds": self.source_duration_seconds,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ExportPlan:
        name = "export_plan"
        if not isinstance(values, dict):
            raise ExportPlanError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "ranges",
            "schema_version",
            "source_duration_seconds",
            "source_fingerprint",
        }
        missing = sorted(known - set(values))
        if missing:
            raise ExportPlanError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        _check_schema_version(name, values, EXPORT_PLAN_SCHEMA_VERSION)
        raw_ranges = values["ranges"]
        if not isinstance(raw_ranges, (list, tuple)):
            raise ExportPlanError(
                f"{name}: 'ranges' must be a list of range objects, "
                f"got {type(raw_ranges).__name__}."
            )
        try:
            parsed = tuple(MediaRange.from_dict(entry) for entry in raw_ranges)
        except DomainError as exc:
            raise ExportPlanError(f"{name}: invalid range: {exc}.") from exc
        return cls(
            source_fingerprint=values["source_fingerprint"],
            source_duration_seconds=values["source_duration_seconds"],
            ranges=parsed,
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> ExportPlan:
        return cls.from_dict(_loads_object("export_plan", data))


_ATTEMPT_ID_PREFIX = "serve-"


def _check_attempt_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AttemptError(
            f"{name}: 'attempt_id' must be a non-empty string, "
            f"got {value!r}."
        )
    body = value[len(_ATTEMPT_ID_PREFIX):] if value.startswith(_ATTEMPT_ID_PREFIX) else None
    if body is None or len(body) < 3 or not body.isdigit():
        raise AttemptError(
            f"{name}: 'attempt_id' must look like 'serve-NNN' with at least "
            f"three digits, got {value!r}."
        )
    return value


def _check_confidence(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttemptError(
            f"{name}: 'confidence' must be a number in [0, 1] or None, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise AttemptError(
            f"{name}: 'confidence' must lie in [0, 1] or be None, "
            f"got {value!r}."
        )
    return number


def _check_evidence(name: str, value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AttemptError(
            f"{name}: 'evidence' must be a mapping of str to finite "
            f"numbers, got {type(value).__name__}."
        )
    cleaned: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise AttemptError(
                f"{name}: 'evidence' keys must be non-blank strings, "
                f"got {key!r}."
            )
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise AttemptError(
                f"{name}: 'evidence[{key}]' must be a finite number, "
                f"got {item!r}."
            )
        number = float(item)
        if not math.isfinite(number):
            raise AttemptError(
                f"{name}: 'evidence[{key}]' must be finite, got {item!r}."
            )
        cleaned[key] = number
    return cleaned


def _check_padding_value(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttemptError(
            f"{name}: 'padding_seconds' must be a number, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise AttemptError(
            f"{name}: 'padding_seconds' must be finite and >= 0, "
            f"got {value!r}."
        )
    return number


def _union_overlapping(ranges: tuple[MediaRange, ...]) -> tuple[MediaRange, ...]:
    """Merge strictly overlapping ranges; adjacency is preserved."""
    if not ranges:
        return ()
    ordered = sorted(ranges, key=lambda item: (item.start_seconds, item.end_seconds))
    merged: list[MediaRange] = [ordered[0]]
    for item in ordered[1:]:
        tail = merged[-1]
        if item.start_seconds < tail.end_seconds:
            merged[-1] = MediaRange(
                start_seconds=tail.start_seconds,
                end_seconds=max(tail.end_seconds, item.end_seconds),
            )
        else:
            merged.append(item)
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class Attempt:
    """One versioned serve attempt with detected and effective ranges.

    ``detected_range`` is the unpadded detector output, half-open
    ``[start, end)``. ``effective_range`` is the padded, source-clamped
    range used for export; it always contains ``detected_range``.
    ``confidence`` is an optional score in ``[0, 1]`` (``None`` means
    unknown/uncertain). ``evidence`` maps non-blank names to finite
    numbers and round-trips through JSON deterministically.
    """

    attempt_id: str = ""
    detected_range: MediaRange = field(default_factory=lambda: MediaRange(0.0, 0.1))
    effective_range: MediaRange = field(default_factory=lambda: MediaRange(0.0, 0.1))
    confidence: float | None = None
    evidence: dict[str, float] = field(default_factory=dict)
    schema_version: int = ATTEMPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "attempt"
        version = self.schema_version
        if not _is_int(version):
            raise AttemptError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != ATTEMPT_SCHEMA_VERSION:
            if version > ATTEMPT_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {ATTEMPT_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {ATTEMPT_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "attempt_id", _check_attempt_id(name, self.attempt_id))
        detected = self.detected_range
        effective = self.effective_range
        if not isinstance(detected, MediaRange):
            raise AttemptError(
                f"{name}: 'detected_range' must be a MediaRange, "
                f"got {type(detected).__name__}."
            )
        if not isinstance(effective, MediaRange):
            raise AttemptError(
                f"{name}: 'effective_range' must be a MediaRange, "
                f"got {type(effective).__name__}."
            )
        if effective.start_seconds > detected.start_seconds or effective.end_seconds < detected.end_seconds:
            raise AttemptError(
                f"{name}: 'effective_range' {effective.to_dict()!r} must contain "
                f"'detected_range' {detected.to_dict()!r}."
            )
        object.__setattr__(self, "confidence", _check_confidence(name, self.confidence))
        object.__setattr__(self, "evidence", _check_evidence(name, self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "confidence": self.confidence,
            "detected_range": self.detected_range.to_dict(),
            "effective_range": self.effective_range.to_dict(),
            "evidence": {key: self.evidence[key] for key in sorted(self.evidence)},
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Attempt:
        name = "attempt"
        if not isinstance(values, dict):
            raise AttemptError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempt_id",
            "confidence",
            "detected_range",
            "effective_range",
            "evidence",
            "schema_version",
        }
        missing = sorted(known - set(values))
        if missing:
            raise AttemptError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except DomainError as exc:
            raise AttemptError(str(exc)) from exc
        try:
            _check_schema_version(name, values, ATTEMPT_SCHEMA_VERSION)
        except SchemaVersionError:
            raise
        except DomainError as exc:
            raise AttemptError(str(exc)) from exc
        try:
            detected = MediaRange.from_dict(values["detected_range"])
        except DomainError as exc:
            raise AttemptError(f"{name}: invalid 'detected_range': {exc}.") from exc
        try:
            effective = MediaRange.from_dict(values["effective_range"])
        except DomainError as exc:
            raise AttemptError(f"{name}: invalid 'effective_range': {exc}.") from exc
        return cls(
            attempt_id=values["attempt_id"],
            detected_range=detected,
            effective_range=effective,
            confidence=values["confidence"],
            evidence=values["evidence"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> Attempt:
        return cls.from_dict(_loads_object("attempt", data))


@dataclass(frozen=True, slots=True)
class AttemptDocument:
    """Versioned set of attempts for one source plus export ranges.

    ``attempts`` preserve unpadded detector output alongside padded
    effective ranges, ordered by ``(detected.start, detected.end)`` with
    stable ``serve-NNN`` identifiers. ``export_ranges`` are the union of
    strictly overlapping effective ranges (adjacent ranges stay
    separate); detected ranges are never rewritten by the union.
    An empty detection yields empty ``attempts`` and ``export_ranges``.
    """

    source_fingerprint: str = ""
    source_duration_seconds: float = 0.0
    padding_seconds: float = 0.0
    method_version: str = "candidate-ranges-v1+plan-v1"
    attempts: tuple[Attempt, ...] = ()
    export_ranges: tuple[MediaRange, ...] = ()
    schema_version: int = ATTEMPT_DOCUMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "attempt_document"
        version = self.schema_version
        if not _is_int(version):
            raise AttemptError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != ATTEMPT_DOCUMENT_SCHEMA_VERSION:
            if version > ATTEMPT_DOCUMENT_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {ATTEMPT_DOCUMENT_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {ATTEMPT_DOCUMENT_SCHEMA_VERSION}."
            )
        _check_non_blank(name, "source_fingerprint", self.source_fingerprint)
        try:
            duration = _check_finite_positive(
                name, "'source_duration_seconds'", self.source_duration_seconds
            )
        except DomainError as exc:
            raise AttemptError(str(exc)) from exc
        object.__setattr__(self, "source_duration_seconds", duration)
        try:
            padding = _check_padding_value(name, self.padding_seconds)
        except AttemptError:
            raise
        object.__setattr__(self, "padding_seconds", padding)
        try:
            _check_non_blank(name, "method_version", self.method_version)
        except DomainError as exc:
            raise AttemptError(str(exc)) from exc
        raw_attempts = self.attempts
        if isinstance(raw_attempts, Attempt):
            raise AttemptError(
                f"{name}: 'attempts' must be a sequence of Attempt, "
                f"got a single Attempt."
            )
        if not isinstance(raw_attempts, (list, tuple)):
            raise AttemptError(
                f"{name}: 'attempts' must be a list or tuple of Attempt, "
                f"got {type(raw_attempts).__name__}."
            )
        normalized = tuple(raw_attempts)
        for entry in normalized:
            if not isinstance(entry, Attempt):
                raise AttemptError(
                    f"{name}: every attempt must be an Attempt, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "attempts", normalized)
        raw_export = self.export_ranges
        if isinstance(raw_export, MediaRange):
            raise AttemptError(
                f"{name}: 'export_ranges' must be a sequence of MediaRange, "
                f"got a single MediaRange."
            )
        if not isinstance(raw_export, (list, tuple)):
            raise AttemptError(
                f"{name}: 'export_ranges' must be a list or tuple of MediaRange, "
                f"got {type(raw_export).__name__}."
            )
        export = tuple(raw_export)
        for entry in export:
            if not isinstance(entry, MediaRange):
                raise AttemptError(
                    f"{name}: every export range must be a MediaRange, "
                    f"got {type(entry).__name__}."
                )
            if entry.end_seconds > duration:
                raise AttemptError(
                    f"{name}: export range {entry.to_dict()!r} extends beyond "
                    f"source duration ({duration!r})."
                )
        object.__setattr__(self, "export_ranges", export)
        # Attempt-level bounds and ordering.
        for entry in normalized:
            if entry.detected_range.end_seconds > duration:
                raise AttemptError(
                    f"{name}: detected range {entry.detected_range.to_dict()!r} "
                    f"extends beyond source duration ({duration!r})."
                )
            if entry.effective_range.end_seconds > duration:
                raise AttemptError(
                    f"{name}: effective range {entry.effective_range.to_dict()!r} "
                    f"extends beyond source duration ({duration!r})."
                )
        for first, second in zip(normalized, normalized[1:]):
            if (second.detected_range.start_seconds, second.detected_range.end_seconds) < (
                first.detected_range.start_seconds, first.detected_range.end_seconds
            ):
                raise AttemptError(
                    f"{name}: attempts must be sorted by detected range; "
                    f"{first.attempt_id!r} precedes {second.attempt_id!r}."
                )
        seen_ids = [entry.attempt_id for entry in normalized]
        if len(set(seen_ids)) != len(seen_ids):
            raise AttemptError(f"{name}: duplicate attempt_id values {seen_ids!r}.")
        for index, entry in enumerate(normalized, start=1):
            expected = f"serve-{index:03d}"
            if entry.attempt_id != expected:
                raise AttemptError(
                    f"{name}: attempt #{index} must carry id {expected!r}, "
                    f"got {entry.attempt_id!r}."
                )
        # Export ranges must be sorted, non-overlapping, and exactly the
        # union of strictly overlapping effective ranges.
        for first, second in zip(export, export[1:]):
            if (second.start_seconds, second.end_seconds) < (first.start_seconds, first.end_seconds):
                raise AttemptError(
                    f"{name}: export ranges must be sorted; "
                    f"{first.to_dict()!r} precedes {second.to_dict()!r}."
                )
            if first.overlaps(second):
                raise AttemptError(
                    f"{name}: export ranges must not overlap; "
                    f"{first.to_dict()!r} overlaps {second.to_dict()!r}."
                )
        expected_export = _union_overlapping(
            tuple(entry.effective_range for entry in normalized)
        )
        if export != expected_export:
            raise AttemptError(
                f"{name}: 'export_ranges' must equal the union of overlapping "
                f"effective ranges; got {[item.to_dict() for item in export]!r}, "
                f"expected {[item.to_dict() for item in expected_export]!r}."
            )

    def __len__(self) -> int:
        return len(self.attempts)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.attempts)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.attempts[index]

    @property
    def total_export_duration_seconds(self) -> float:
        return sum(item.duration_seconds for item in self.export_ranges)

    def to_export_plan(self) -> ExportPlan:
        """Return an :class:`ExportPlan` over the unioned export ranges."""
        if not self.export_ranges:
            raise AttemptError("attempt_document: no export ranges to export.")
        return ExportPlan(
            source_fingerprint=self.source_fingerprint,
            source_duration_seconds=self.source_duration_seconds,
            ranges=self.export_ranges,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [entry.to_dict() for entry in self.attempts],
            "export_ranges": [entry.to_dict() for entry in self.export_ranges],
            "method_version": self.method_version,
            "padding_seconds": self.padding_seconds,
            "schema_version": self.schema_version,
            "source_duration_seconds": self.source_duration_seconds,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AttemptDocument:
        name = "attempt_document"
        if not isinstance(values, dict):
            raise AttemptError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempts",
            "export_ranges",
            "method_version",
            "padding_seconds",
            "schema_version",
            "source_duration_seconds",
            "source_fingerprint",
        }
        missing = sorted(known - set(values))
        if missing:
            raise AttemptError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except DomainError as exc:
            raise AttemptError(str(exc)) from exc
        try:
            _check_schema_version(name, values, ATTEMPT_DOCUMENT_SCHEMA_VERSION)
        except SchemaVersionError:
            raise
        except DomainError as exc:
            raise AttemptError(str(exc)) from exc
        raw_attempts = values["attempts"]
        raw_export = values["export_ranges"]
        if not isinstance(raw_attempts, (list, tuple)):
            raise AttemptError(
                f"{name}: 'attempts' must be a list of attempt objects, "
                f"got {type(raw_attempts).__name__}."
            )
        if not isinstance(raw_export, (list, tuple)):
            raise AttemptError(
                f"{name}: 'export_ranges' must be a list of range objects, "
                f"got {type(raw_export).__name__}."
            )
        try:
            parsed_attempts = tuple(Attempt.from_dict(entry) for entry in raw_attempts)
        except DomainError as exc:
            raise AttemptError(f"{name}: invalid attempt: {exc}.") from exc
        try:
            parsed_export = tuple(MediaRange.from_dict(entry) for entry in raw_export)
        except DomainError as exc:
            raise AttemptError(f"{name}: invalid export range: {exc}.") from exc
        return cls(
            source_fingerprint=values["source_fingerprint"],
            source_duration_seconds=values["source_duration_seconds"],
            padding_seconds=values["padding_seconds"],
            method_version=values["method_version"],
            attempts=parsed_attempts,
            export_ranges=parsed_export,
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> AttemptDocument:
        return cls.from_dict(_loads_object("attempt_document", data))


# --- Kovacs eight-stage phase schema (M4.1) ---------------------------------
#
# Pure, versioned, immutable representation for serve-phase analysis.
# No inference, feature extraction, solver logic, or file I/O lives here.
#
# Canonical source time (decimal seconds of the source timeline) is the only
# temporal authority. Every stage interval/keyframe lies inside its owning
# unpadded attempt range. Public stage keys are concise; method caveats
# (body-pose estimates, no racket/ball observation, audio-anchored contact)
# belong in provenance/evidence/limitations, never in renamed stage keys.

PHASE_SCHEMA_VERSION = 1

STAGE_ORDER: tuple[str, ...] = (
    "start",
    "release",
    "loading",
    "cocking",
    "acceleration",
    "contact",
    "deceleration",
    "finish",
)
STAGE_KEYS: tuple[str, ...] = STAGE_ORDER

STAGE_AVAILABILITIES: tuple[str, ...] = ("available", "partial", "unavailable")
STAGE_PROVENANCES: tuple[str, ...] = (
    "body_pose",
    "audio_transient",
    "body_pose_audio",
    "manual",
)
STRUCTURAL_STATUSES: tuple[str, ...] = (
    "complete",
    "partial",
    "incomplete",
    "unavailable",
)

_CONTACT_AUDIO_PROVENANCES: tuple[str, ...] = ("audio_transient", "body_pose_audio")
_CONTACT_ALLOWED_PROVENANCES: tuple[str, ...] = (
    "audio_transient",
    "body_pose_audio",
    "manual",
)


class PhaseError(DomainError):
    """Raised when a phase stage, attempt phase, or phase document is invalid."""


def _check_phase_attempt_id(name: str, value: Any) -> str:
    try:
        return _check_attempt_id(name, value)
    except AttemptError as exc:
        raise PhaseError(str(exc)) from exc


def _check_phase_confidence(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseError(
            f"{name}: 'confidence' must be a finite number in [0, 1], "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseError(
            f"{name}: 'confidence' must lie in [0, 1], got {value!r}."
        )
    return number


def _check_phase_keyframe(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseError(
            f"{name}: 'keyframe_seconds' must be a finite number or null, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseError(
            f"{name}: 'keyframe_seconds' must be finite and >= 0, "
            f"got {value!r}."
        )
    return number


def _check_phase_uncertainty(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseError(
            f"{name}: 'temporal_uncertainty_seconds' must be a finite "
            f"number >= 0 or null, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseError(
            f"{name}: 'temporal_uncertainty_seconds' must be finite and "
            f">= 0, got {value!r}."
        )
    return number


def _check_str_collection(name: str, key: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PhaseError(
            f"{name}: {key!r} must be a list or tuple of non-blank strings, "
            f"got {type(value).__name__}."
        )
    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise PhaseError(
                f"{name}: {key!r} entries must be non-blank strings, "
                f"got {entry!r}."
            )
        if entry in seen:
            raise PhaseError(
                f"{name}: {key!r} must not contain duplicates, "
                f"got {entry!r}."
            )
        seen.add(entry)
        cleaned.append(entry)
    return tuple(cleaned)


def _check_phase_schema_version(name: str, values: dict[str, Any]) -> int:
    try:
        return _check_schema_version(name, values, PHASE_SCHEMA_VERSION)
    except SchemaVersionError:
        raise
    except DomainError as exc:
        raise PhaseError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class StagePhase:
    """One immutable estimate for a single Kovacs stage.

    ``interval`` is a half-open ``[start, end)`` source-time range or null.
    ``keyframe_seconds`` is an optional representative source time inside
    ``interval``. ``temporal_uncertainty_seconds`` is an optional finite
    nonnegative uncertainty width. ``evidence``/``limitations`` are
    machine-readable identifier collections (possibly empty).

    An ``unavailable`` stage honestly carries no observation: ``interval``,
    ``keyframe_seconds``, and ``temporal_uncertainty_seconds`` are forbidden
    (must be null). ``confidence``, ``provenance``, ``evidence``, and
    ``limitations`` are still validated normally. ``available``/``partial``
    stages require a non-null ``interval``; keyframe/uncertainty are optional
    except for audio-anchored contact (enforced at the attempt level).
    """

    availability: str = "unavailable"
    provenance: str = "body_pose"
    confidence: float = 0.0
    interval: MediaRange | None = None
    keyframe_seconds: float | None = None
    temporal_uncertainty_seconds: float | None = None
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    schema_version: int = PHASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "stage_phase"
        version = self.schema_version
        if not _is_int(version):
            raise PhaseError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != PHASE_SCHEMA_VERSION:
            if version > PHASE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {PHASE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {PHASE_SCHEMA_VERSION}."
            )
        if self.availability not in STAGE_AVAILABILITIES:
            raise PhaseError(
                f"{name}: 'availability' must be one of "
                f"{list(STAGE_AVAILABILITIES)}, got {self.availability!r}."
            )
        if self.provenance not in STAGE_PROVENANCES:
            raise PhaseError(
                f"{name}: 'provenance' must be one of "
                f"{list(STAGE_PROVENANCES)}, got {self.provenance!r}."
            )
        object.__setattr__(
            self, "confidence", _check_phase_confidence(name, self.confidence)
        )
        interval = self.interval
        if self.availability == "unavailable":
            if interval is not None:
                raise PhaseError(
                    f"{name}: 'interval' must be null for unavailable stages, "
                    f"got {interval!r}."
                )
            if self.keyframe_seconds is not None:
                raise PhaseError(
                    f"{name}: 'keyframe_seconds' must be null for unavailable "
                    f"stages, got {self.keyframe_seconds!r}."
                )
            if self.temporal_uncertainty_seconds is not None:
                raise PhaseError(
                    f"{name}: 'temporal_uncertainty_seconds' must be null for "
                    f"unavailable stages, got "
                    f"{self.temporal_uncertainty_seconds!r}."
                )
        else:
            if not isinstance(interval, MediaRange):
                raise PhaseError(
                    f"{name}: 'interval' must be a MediaRange for "
                    f"{self.availability!r} stages, got {interval!r}."
                )
        object.__setattr__(
            self,
            "keyframe_seconds",
            _check_phase_keyframe(name, self.keyframe_seconds),
        )
        object.__setattr__(
            self,
            "temporal_uncertainty_seconds",
            _check_phase_uncertainty(name, self.temporal_uncertainty_seconds),
        )
        keyframe = self.keyframe_seconds
        if keyframe is not None:
            if not isinstance(interval, MediaRange):
                raise PhaseError(
                    f"{name}: 'keyframe_seconds' requires a non-null interval."
                )
            if not interval.contains(keyframe) and not (
                keyframe == interval.start_seconds
            ):
                # contains() already covers [start, end); keep explicit check
                # readable for boundary diagnostics.
                raise PhaseError(
                    f"{name}: 'keyframe_seconds' ({keyframe!r}) must lie in "
                    f"interval [{interval.start_seconds!r}, "
                    f"{interval.end_seconds!r})."
                )
            if not (interval.start_seconds <= keyframe < interval.end_seconds):
                raise PhaseError(
                    f"{name}: 'keyframe_seconds' ({keyframe!r}) must lie in "
                    f"interval [{interval.start_seconds!r}, "
                    f"{interval.end_seconds!r})."
                )
        object.__setattr__(
            self, "evidence", _check_str_collection(name, "'evidence'", self.evidence)
        )
        object.__setattr__(
            self,
            "limitations",
            _check_str_collection(name, "'limitations'", self.limitations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "availability": self.availability,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "interval": self.interval.to_dict() if self.interval is not None else None,
            "keyframe_seconds": self.keyframe_seconds,
            "limitations": list(self.limitations),
            "provenance": self.provenance,
            "schema_version": self.schema_version,
            "temporal_uncertainty_seconds": self.temporal_uncertainty_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StagePhase:
        name = "stage_phase"
        if not isinstance(values, dict):
            raise PhaseError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "availability",
            "confidence",
            "evidence",
            "interval",
            "keyframe_seconds",
            "limitations",
            "provenance",
            "schema_version",
            "temporal_uncertainty_seconds",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except DomainError as exc:
            raise PhaseError(str(exc)) from exc
        _check_phase_schema_version(name, values)
        raw_interval = values["interval"]
        interval: MediaRange | None
        if raw_interval is None:
            interval = None
        elif isinstance(raw_interval, dict):
            try:
                interval = MediaRange.from_dict(raw_interval)
            except DomainError as exc:
                raise PhaseError(f"{name}: invalid 'interval': {exc}.") from exc
        else:
            raise PhaseError(
                f"{name}: 'interval' must be a range object or null, "
                f"got {type(raw_interval).__name__}."
            )
        return cls(
            availability=values["availability"],
            provenance=values["provenance"],
            confidence=values["confidence"],
            interval=interval,
            keyframe_seconds=values["keyframe_seconds"],
            temporal_uncertainty_seconds=values["temporal_uncertainty_seconds"],
            evidence=values["evidence"],
            limitations=values["limitations"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> StagePhase:
        return cls.from_dict(_loads_object("stage_phase", data))


@dataclass(frozen=True, slots=True)
class AttemptPhase:
    """Phase result for one unpadded attempt range.

    ``attempt_range`` is the owning unpadded half-open source-time range.
    ``stages`` maps each of the eight concise stage keys to a
    :class:`StagePhase`, stored as a read-only mapping in canonical
    :data:`STAGE_ORDER`. ``structural_status`` is ``complete`` iff every
    stage is ``available``; ``unavailable`` iff every stage is
    ``unavailable``; ``partial`` iff at least one stage is ``available``
    without all being ``available``; otherwise ``incomplete`` (some
    ``partial`` evidence but no ``available`` stage). ``anomalies`` are
    stable machine-readable identifiers (possibly empty).

    Available/partial stage intervals must lie inside ``attempt_range`` and
    run chronologically without overlap; keyframes follow the same order.
    An available/partial ``contact`` must use ``audio_transient``,
    ``body_pose_audio``, or ``manual`` provenance (never body-only visual
    contact); audio-anchored contact additionally requires a non-null
    keyframe inside its interval and a finite nonnegative uncertainty.
    """

    attempt_id: str = ""
    attempt_range: MediaRange = field(
        default_factory=lambda: MediaRange(0.0, 0.1)
    )
    method_version: str = ""
    config_id: str = ""
    stages: Mapping[str, StagePhase] = field(default_factory=dict)  # type: ignore[assignment]
    structural_status: str = "unavailable"
    anomalies: tuple[str, ...] = ()
    schema_version: int = PHASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "attempt_phase"
        version = self.schema_version
        if not _is_int(version):
            raise PhaseError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != PHASE_SCHEMA_VERSION:
            if version > PHASE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {PHASE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {PHASE_SCHEMA_VERSION}."
            )
        object.__setattr__(
            self, "attempt_id", _check_phase_attempt_id(name, self.attempt_id)
        )
        attempt_range = self.attempt_range
        if not isinstance(attempt_range, MediaRange):
            raise PhaseError(
                f"{name}: 'attempt_range' must be a MediaRange, "
                f"got {type(attempt_range).__name__}."
            )
        try:
            _check_non_blank(name, "method_version", self.method_version)
        except DomainError as exc:
            raise PhaseError(str(exc)) from exc
        try:
            _check_non_blank(name, "config_id", self.config_id)
        except DomainError as exc:
            raise PhaseError(str(exc)) from exc
        if self.structural_status not in STRUCTURAL_STATUSES:
            raise PhaseError(
                f"{name}: 'structural_status' must be one of "
                f"{list(STRUCTURAL_STATUSES)}, got {self.structural_status!r}."
            )
        object.__setattr__(
            self, "anomalies", _check_str_collection(name, "'anomalies'", self.anomalies)
        )
        raw = self.stages
        if not isinstance(raw, Mapping):
            raise PhaseError(
                f"{name}: 'stages' must be a mapping of stage key to "
                f"StagePhase, got {type(raw).__name__}."
            )
        raw_keys = set(raw.keys())
        expected_keys = set(STAGE_ORDER)
        if raw_keys != expected_keys:
            raise PhaseError(
                f"{name}: 'stages' must contain exactly the eight stage keys "
                f"{list(STAGE_ORDER)}, got {sorted(raw_keys)!r}."
            )
        for key in STAGE_ORDER:
            entry = raw[key]
            if not isinstance(entry, StagePhase):
                raise PhaseError(
                    f"{name}: stage {key!r} must be a StagePhase, "
                    f"got {type(entry).__name__}."
                )
        ordered = MappingProxyType({key: raw[key] for key in STAGE_ORDER})
        object.__setattr__(self, "stages", ordered)
        # Containment inside the unpadded attempt.
        for key in STAGE_ORDER:
            stage = ordered[key]
            if stage.availability in ("available", "partial"):
                interval = stage.interval
                assert isinstance(interval, MediaRange)
                if interval.start_seconds < attempt_range.start_seconds or (
                    interval.end_seconds > attempt_range.end_seconds
                ):
                    raise PhaseError(
                        f"{name}: stage {key!r} interval "
                        f"{interval.to_dict()!r} must lie inside attempt range "
                        f"{attempt_range.to_dict()!r}."
                    )
        # Chronological ordering over available/partial intervals/keyframes.
        previous_end: float | None = None
        previous_keyframe: float | None = None
        previous_key: str | None = None
        for key in STAGE_ORDER:
            stage = ordered[key]
            if stage.availability not in ("available", "partial"):
                continue
            assert isinstance(stage.interval, MediaRange)
            start = stage.interval.start_seconds
            end = stage.interval.end_seconds
            if previous_end is not None and start < previous_end:
                assert previous_key is not None
                raise PhaseError(
                    f"{name}: stage {key!r} interval starts at {start!r} "
                    f"before previous stage {previous_key!r} ends at "
                    f"{previous_end!r}; stages must be chronological."
                )
            previous_end = end
            previous_key = key
            if stage.keyframe_seconds is not None:
                if (
                    previous_keyframe is not None
                    and stage.keyframe_seconds < previous_keyframe
                ):
                    raise PhaseError(
                        f"{name}: stage {key!r} keyframe "
                        f"({stage.keyframe_seconds!r}) precedes earlier keyframe "
                        f"({previous_keyframe!r}); keyframes must be chronological."
                    )
                previous_keyframe = stage.keyframe_seconds
        # Honest contact provenance: never visually observed contact.
        contact = ordered["contact"]
        if contact.availability in ("available", "partial"):
            if contact.provenance not in _CONTACT_ALLOWED_PROVENANCES:
                raise PhaseError(
                    f"{name}: contact with {contact.availability!r} availability "
                    f"must use provenance one of "
                    f"{list(_CONTACT_ALLOWED_PROVENANCES)}, got "
                    f"{contact.provenance!r}; body-only contact must never claim "
                    f"visually observed contact."
                )
            if contact.provenance in _CONTACT_AUDIO_PROVENANCES:
                if contact.keyframe_seconds is None:
                    raise PhaseError(
                        f"{name}: contact with {contact.provenance!r} provenance "
                        f"requires a non-null keyframe_seconds audio anchor."
                    )
                if contact.temporal_uncertainty_seconds is None:
                    raise PhaseError(
                        f"{name}: contact with {contact.provenance!r} provenance "
                        f"requires finite nonnegative "
                        f"temporal_uncertainty_seconds."
                    )
        # Structural status consistency.
        counts = {value: 0 for value in STAGE_AVAILABILITIES}
        for key in STAGE_ORDER:
            counts[ordered[key].availability] += 1
        expected: str
        if counts["available"] == len(STAGE_ORDER):
            expected = "complete"
        elif counts["unavailable"] == len(STAGE_ORDER):
            expected = "unavailable"
        elif counts["available"] >= 1:
            expected = "partial"
        else:
            expected = "incomplete"
        if self.structural_status != expected:
            raise PhaseError(
                f"{name}: 'structural_status' {self.structural_status!r} is "
                f"inconsistent with stage availabilities "
                f"{counts!r}; expected {expected!r}."
            )

    def stage(self, key: str) -> StagePhase:
        """Return the :class:`StagePhase` for canonical stage ``key``."""
        try:
            return self.stages[key]  # type: ignore[index]
        except KeyError as exc:
            raise PhaseError(
                f"attempt_phase: unknown stage key {key!r}."
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        stages = self.stages
        assert isinstance(stages, Mapping)
        return {
            "anomalies": list(self.anomalies),
            "attempt_id": self.attempt_id,
            "attempt_range": self.attempt_range.to_dict(),
            "config_id": self.config_id,
            "method_version": self.method_version,
            "schema_version": self.schema_version,
            "stages": {key: stages[key].to_dict() for key in STAGE_ORDER},
            "structural_status": self.structural_status,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AttemptPhase:
        name = "attempt_phase"
        if not isinstance(values, dict):
            raise PhaseError(
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
            "structural_status",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except DomainError as exc:
            raise PhaseError(str(exc)) from exc
        _check_phase_schema_version(name, values)
        raw_range = values["attempt_range"]
        if not isinstance(raw_range, dict):
            raise PhaseError(
                f"{name}: 'attempt_range' must be a range object, "
                f"got {type(raw_range).__name__}."
            )
        try:
            attempt_range = MediaRange.from_dict(raw_range)
        except DomainError as exc:
            raise PhaseError(f"{name}: invalid 'attempt_range': {exc}.") from exc
        raw_stages = values["stages"]
        if not isinstance(raw_stages, dict):
            raise PhaseError(
                f"{name}: 'stages' must be a mapping of stage key to object, "
                f"got {type(raw_stages).__name__}."
            )
        parsed: dict[str, StagePhase] = {}
        for key, payload in raw_stages.items():
            if key not in STAGE_ORDER:
                raise PhaseError(
                    f"{name}: unknown stage key {key!r}."
                )
            if not isinstance(payload, dict):
                raise PhaseError(
                    f"{name}: stage {key!r} must decode from a mapping, "
                    f"got {type(payload).__name__}."
                )
            try:
                parsed[key] = StagePhase.from_dict(payload)
            except DomainError as exc:
                raise PhaseError(
                    f"{name}: invalid stage {key!r}: {exc}."
                ) from exc
        return cls(
            attempt_id=values["attempt_id"],
            attempt_range=attempt_range,
            method_version=values["method_version"],
            config_id=values["config_id"],
            stages=parsed,
            structural_status=values["structural_status"],
            anomalies=values["anomalies"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> AttemptPhase:
        return cls.from_dict(_loads_object("attempt_phase", data))


@dataclass(frozen=True, slots=True)
class PhaseDocument:
    """Versioned, immutable top-level phase result for ``checkpoints.json``.

    ``attempts`` is an ordered tuple of :class:`AttemptPhase` sorted by
    ``(attempt_range.start, attempt_range.end)`` with unique ``attempt_id``
    values and pairwise non-overlapping (adjacency allowed) unpadded ranges
    fully inside ``[0, source_duration_seconds]``. An empty tuple honestly
    represents no phase results. All times are canonical source times.
    """

    source_fingerprint: str = ""
    source_duration_seconds: float = 0.0
    attempts: tuple[AttemptPhase, ...] = ()
    schema_version: int = PHASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_document"
        version = self.schema_version
        if not _is_int(version):
            raise PhaseError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != PHASE_SCHEMA_VERSION:
            if version > PHASE_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version {PHASE_SCHEMA_VERSION}."
                )
            raise SchemaVersionError(
                f"{name}: unsupported schema_version {version!r}; "
                f"expected version {PHASE_SCHEMA_VERSION}."
            )
        try:
            _check_non_blank(name, "source_fingerprint", self.source_fingerprint)
        except DomainError as exc:
            raise PhaseError(str(exc)) from exc
        try:
            duration = _check_finite_positive(
                name, "'source_duration_seconds'", self.source_duration_seconds
            )
        except DomainError as exc:
            raise PhaseError(str(exc)) from exc
        object.__setattr__(self, "source_duration_seconds", duration)
        raw = self.attempts
        if isinstance(raw, AttemptPhase):
            raise PhaseError(
                f"{name}: 'attempts' must be a sequence of AttemptPhase, "
                f"got a single AttemptPhase."
            )
        if not isinstance(raw, (list, tuple)):
            raise PhaseError(
                f"{name}: 'attempts' must be a list or tuple of AttemptPhase, "
                f"got {type(raw).__name__}."
            )
        normalized = tuple(raw)
        for entry in normalized:
            if not isinstance(entry, AttemptPhase):
                raise PhaseError(
                    f"{name}: every attempt must be an AttemptPhase, "
                    f"got {type(entry).__name__}."
                )
            if entry.attempt_range.end_seconds > duration:
                raise PhaseError(
                    f"{name}: attempt {entry.attempt_id!r} range "
                    f"{entry.attempt_range.to_dict()!r} extends beyond source "
                    f"duration ({duration!r})."
                )
        object.__setattr__(self, "attempts", normalized)
        seen = [entry.attempt_id for entry in normalized]
        if len(set(seen)) != len(seen):
            raise PhaseError(f"{name}: duplicate attempt_id values {seen!r}.")
        for first, second in zip(normalized, normalized[1:]):
            first_key = (
                first.attempt_range.start_seconds,
                first.attempt_range.end_seconds,
            )
            second_key = (
                second.attempt_range.start_seconds,
                second.attempt_range.end_seconds,
            )
            if second_key < first_key:
                raise PhaseError(
                    f"{name}: attempts must be sorted by attempt range; "
                    f"{first.attempt_id!r} precedes {second.attempt_id!r}."
                )
            if first.attempt_range.overlaps(second.attempt_range):
                raise PhaseError(
                    f"{name}: attempt ranges must not overlap; "
                    f"{first.attempt_range.to_dict()!r} overlaps "
                    f"{second.attempt_range.to_dict()!r}."
                )

    def __len__(self) -> int:
        return len(self.attempts)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.attempts)

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.attempts[index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [entry.to_dict() for entry in self.attempts],
            "schema_version": self.schema_version,
            "source_duration_seconds": self.source_duration_seconds,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseDocument:
        name = "phase_document"
        if not isinstance(values, dict):
            raise PhaseError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempts",
            "schema_version",
            "source_duration_seconds",
            "source_fingerprint",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseError(f"{name}: missing required keys {missing!r}.")
        try:
            _check_no_unknown_keys(name, values, known)
        except DomainError as exc:
            raise PhaseError(str(exc)) from exc
        _check_phase_schema_version(name, values)
        raw_attempts = values["attempts"]
        if not isinstance(raw_attempts, (list, tuple)):
            raise PhaseError(
                f"{name}: 'attempts' must be a list of attempt objects, "
                f"got {type(raw_attempts).__name__}."
            )
        try:
            parsed = tuple(AttemptPhase.from_dict(entry) for entry in raw_attempts)
        except DomainError as exc:
            raise PhaseError(f"{name}: invalid attempt: {exc}.") from exc
        return cls(
            source_fingerprint=values["source_fingerprint"],
            source_duration_seconds=values["source_duration_seconds"],
            attempts=parsed,
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PhaseDocument:
        return cls.from_dict(_loads_object("phase_document", data))
