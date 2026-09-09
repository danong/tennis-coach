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
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SOURCE_SCHEMA_VERSION",
    "RANGE_SCHEMA_VERSION",
    "EXPORT_PLAN_SCHEMA_VERSION",
    "DomainError",
    "SchemaVersionError",
    "SourceMetadataError",
    "RangeError",
    "ExportPlanError",
    "SourceMetadata",
    "MediaRange",
    "ExportPlan",
]

SOURCE_SCHEMA_VERSION = 1
RANGE_SCHEMA_VERSION = 1
EXPORT_PLAN_SCHEMA_VERSION = 1

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
