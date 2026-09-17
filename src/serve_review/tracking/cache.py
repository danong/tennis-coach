"""Identity-bound JSONL cache for raw per-attempt RacketVision observations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve_review.media.color import MODEL_INPUT_PREPROCESS_VERSION
from serve_review.media.frames import SAMPLER_ORIENTATION_VERSION
from serve_review.scene import Point2D, Racket2D
from serve_review.tracking.racketvision import RacketVisionFrameObservation

__all__ = [
    "RACKETVISION_CACHE_SCHEMA_VERSION",
    "RACKETVISION_PREPROCESSING_VERSION",
    "RacketVisionCacheCorruptError",
    "RacketVisionCacheError",
    "RacketVisionCacheIdentity",
    "RacketVisionCacheSnapshot",
    "RacketVisionCacheStaleError",
    "fingerprint_frame_times",
    "load_racketvision_cache",
    "write_racketvision_cache",
]

RACKETVISION_CACHE_SCHEMA_VERSION = 1
RACKETVISION_PREPROCESSING_VERSION = MODEL_INPUT_PREPROCESS_VERSION


class RacketVisionCacheError(Exception):
    """A RacketVision cache could not be read or written safely."""


class RacketVisionCacheCorruptError(RacketVisionCacheError):
    """A cache is malformed or incomplete."""


class RacketVisionCacheStaleError(RacketVisionCacheError):
    """A valid cache belongs to different inputs or inference settings."""


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RacketVisionCacheError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise RacketVisionCacheError(f"{name} must be a finite number.")
    return result


def _non_blank(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RacketVisionCacheError(f"{name} must be a non-blank string.")
    return value


def _current_version(name: str, value: object, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RacketVisionCacheError(
            f"{name} must equal the supported version {expected}, got {value!r}."
        )
    return value


def fingerprint_frame_times(frame_times: Sequence[float]) -> str:
    """Return a stable digest of an exact ordered canonical-PTS sequence."""
    digest = hashlib.sha256()
    previous: float | None = None
    for index, raw in enumerate(frame_times):
        moment = _finite(f"frame_times[{index}]", raw)
        if moment < 0.0:
            raise RacketVisionCacheError("frame times must be >= 0.")
        if previous is not None and moment <= previous:
            raise RacketVisionCacheError("frame times must be strictly increasing.")
        digest.update(struct.pack(">d", moment))
        previous = moment
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class RacketVisionCacheIdentity:
    """Inputs that make one per-attempt raw object track reusable."""

    source_fingerprint: str
    attempt_start_seconds: float
    attempt_end_seconds: float
    timeline_fingerprint: str
    tracker_fingerprint: str
    sampler_orientation_version: int = SAMPLER_ORIENTATION_VERSION
    preprocessing_version: int = RACKETVISION_PREPROCESSING_VERSION
    schema_version: int = RACKETVISION_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_fingerprint",
            _non_blank("source_fingerprint", self.source_fingerprint),
        )
        start = _finite("attempt_start_seconds", self.attempt_start_seconds)
        end = _finite("attempt_end_seconds", self.attempt_end_seconds)
        if start < 0.0 or end <= start:
            raise RacketVisionCacheError("attempt range must satisfy 0 <= start < end.")
        object.__setattr__(self, "attempt_start_seconds", start)
        object.__setattr__(self, "attempt_end_seconds", end)
        object.__setattr__(
            self,
            "timeline_fingerprint",
            _non_blank("timeline_fingerprint", self.timeline_fingerprint),
        )
        object.__setattr__(
            self,
            "tracker_fingerprint",
            _non_blank("tracker_fingerprint", self.tracker_fingerprint),
        )
        _current_version(
            "sampler_orientation_version",
            self.sampler_orientation_version,
            SAMPLER_ORIENTATION_VERSION,
        )
        _current_version(
            "preprocessing_version",
            self.preprocessing_version,
            RACKETVISION_PREPROCESSING_VERSION,
        )
        _current_version(
            "schema_version", self.schema_version, RACKETVISION_CACHE_SCHEMA_VERSION
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_end_seconds": self.attempt_end_seconds,
            "attempt_start_seconds": self.attempt_start_seconds,
            "preprocessing_version": self.preprocessing_version,
            "sampler_orientation_version": self.sampler_orientation_version,
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "timeline_fingerprint": self.timeline_fingerprint,
            "tracker_fingerprint": self.tracker_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class RacketVisionCacheSnapshot:
    """One validated complete raw RacketVision track."""

    identity: RacketVisionCacheIdentity
    observations: tuple[RacketVisionFrameObservation, ...]
    complete: bool = True


def _exact_keys(name: str, value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RacketVisionCacheCorruptError(f"{name} must be a JSON object.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RacketVisionCacheCorruptError(
            f"{name} fields do not match schema; missing={missing!r}, "
            f"unknown={unknown!r}."
        )
    return value


def _identity_from_dict(value: object) -> RacketVisionCacheIdentity:
    fields = {
        "attempt_end_seconds",
        "attempt_start_seconds",
        "preprocessing_version",
        "sampler_orientation_version",
        "schema_version",
        "source_fingerprint",
        "timeline_fingerprint",
        "tracker_fingerprint",
    }
    values = _exact_keys("cache identity", value, fields)
    try:
        return RacketVisionCacheIdentity(**values)
    except (RacketVisionCacheError, TypeError) as exc:
        raise RacketVisionCacheCorruptError(f"invalid cache identity: {exc}") from exc


def _point_to_dict(point: Point2D) -> dict[str, float]:
    return {"confidence": point.confidence, "x": point.x, "y": point.y}


def _point_from_dict(value: object) -> Point2D:
    values = _exact_keys("point", value, {"confidence", "x", "y"})
    try:
        return Point2D(x=values["x"], y=values["y"], confidence=values["confidence"])
    except (TypeError, ValueError) as exc:
        raise RacketVisionCacheCorruptError(f"invalid point: {exc}") from exc


def _racket_to_dict(racket: Racket2D) -> dict[str, Any]:
    return {
        "bbox": list(racket.bbox),
        "bbox_confidence": racket.bbox_confidence,
        "keypoints": {
            name: _point_to_dict(point) if point is not None else None
            for name, point in sorted(racket.keypoints.items())
        },
    }


def _racket_from_dict(value: object) -> Racket2D:
    values = _exact_keys("racket", value, {"bbox", "bbox_confidence", "keypoints"})
    keypoints = values["keypoints"]
    if not isinstance(keypoints, dict) or any(
        not isinstance(name, str) or not name for name in keypoints
    ):
        raise RacketVisionCacheCorruptError(
            "racket keypoints must be an object with non-blank string keys."
        )
    try:
        return Racket2D(
            bbox=values["bbox"],
            bbox_confidence=values["bbox_confidence"],
            keypoints={
                name: _point_from_dict(point) if point is not None else None
                for name, point in keypoints.items()
            },
        )
    except (TypeError, ValueError) as exc:
        raise RacketVisionCacheCorruptError(f"invalid racket: {exc}") from exc


def _observation_to_dict(observation: RacketVisionFrameObservation) -> dict[str, Any]:
    if not isinstance(observation, RacketVisionFrameObservation):
        raise RacketVisionCacheError(
            "observations must contain RacketVisionFrameObservation values."
        )
    return {
        "ball_2d": (
            _point_to_dict(observation.ball_2d)
            if observation.ball_2d is not None
            else None
        ),
        "racket_2d": (
            _racket_to_dict(observation.racket_2d)
            if observation.racket_2d is not None
            else None
        ),
        "time_seconds": observation.time_seconds,
    }


def _observation_from_dict(value: object) -> RacketVisionFrameObservation:
    values = _exact_keys(
        "frame observation", value, {"ball_2d", "racket_2d", "time_seconds"}
    )
    try:
        return RacketVisionFrameObservation(
            time_seconds=values["time_seconds"],
            ball_2d=(
                _point_from_dict(values["ball_2d"])
                if values["ball_2d"] is not None
                else None
            ),
            racket_2d=(
                _racket_from_dict(values["racket_2d"])
                if values["racket_2d"] is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise RacketVisionCacheCorruptError(
            f"invalid frame observation: {exc}"
        ) from exc


def _validate_observations(
    observations: Sequence[RacketVisionFrameObservation],
    expected_times: Sequence[float],
) -> tuple[RacketVisionFrameObservation, ...]:
    items = tuple(observations)
    actual_times = tuple(
        item.time_seconds
        for item in items
        if isinstance(item, RacketVisionFrameObservation)
    )
    if len(actual_times) != len(items):
        raise RacketVisionCacheError(
            "observations must contain RacketVisionFrameObservation values."
        )
    expected = tuple(float(value) for value in expected_times)
    fingerprint_frame_times(expected)
    fingerprint_frame_times(actual_times)
    if actual_times != expected:
        raise RacketVisionCacheError(
            "RacketVision observation times do not exactly match canonical PTS."
        )
    return items


def _line(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def write_racketvision_cache(
    path: Path | str,
    identity: RacketVisionCacheIdentity,
    observations: Sequence[RacketVisionFrameObservation],
    *,
    expected_times: Sequence[float],
) -> Path:
    """Atomically publish one complete cache after exact-PTS validation."""
    if not isinstance(identity, RacketVisionCacheIdentity):
        raise RacketVisionCacheError(
            "identity must be a RacketVisionCacheIdentity value."
        )
    items = _validate_observations(observations, expected_times)
    if identity.timeline_fingerprint != fingerprint_frame_times(expected_times):
        raise RacketVisionCacheError(
            "cache identity timeline fingerprint does not match canonical PTS."
        )
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_line({"identity": identity.to_dict(), "type": "header"}))
            for observation in items:
                handle.write(
                    _line(
                        {
                            "observation": _observation_to_dict(observation),
                            "type": "frame",
                        }
                    )
                )
            handle.write(
                _line(
                    {
                        "complete": True,
                        "frame_count": len(items),
                        "schema_version": RACKETVISION_CACHE_SCHEMA_VERSION,
                        "type": "footer",
                    }
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except RacketVisionCacheError:
        raise
    except OSError as exc:
        raise RacketVisionCacheError(
            f"could not write RacketVision cache {target}: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return target


def load_racketvision_cache(
    path: Path | str,
    expected_identity: RacketVisionCacheIdentity,
    *,
    expected_times: Sequence[float],
) -> RacketVisionCacheSnapshot:
    """Load a complete cache and require exact identity and canonical PTS."""
    if not isinstance(expected_identity, RacketVisionCacheIdentity):
        raise RacketVisionCacheError(
            "expected_identity must be a RacketVisionCacheIdentity value."
        )
    target = Path(path).expanduser()
    try:
        raw_lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RacketVisionCacheError(
            f"could not read RacketVision cache {target}: {exc}"
        ) from exc
    if len(raw_lines) < 2 or any(not line.strip() for line in raw_lines):
        raise RacketVisionCacheCorruptError(
            "RacketVision cache is empty, incomplete, or contains blank lines."
        )
    try:
        records = [json.loads(line) for line in raw_lines]
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RacketVisionCacheCorruptError(
            f"RacketVision cache is not valid JSONL: {exc}"
        ) from exc

    header = _exact_keys("cache header", records[0], {"identity", "type"})
    if header["type"] != "header":
        raise RacketVisionCacheCorruptError("first cache record must be a header.")
    stored_identity = _identity_from_dict(header["identity"])
    if stored_identity != expected_identity:
        raise RacketVisionCacheStaleError(
            "RacketVision cache identity does not match requested analysis."
        )

    footer = _exact_keys(
        "cache footer",
        records[-1],
        {"complete", "frame_count", "schema_version", "type"},
    )
    if (
        footer["type"] != "footer"
        or footer["complete"] is not True
        or isinstance(footer["frame_count"], bool)
        or not isinstance(footer["frame_count"], int)
        or footer["frame_count"] != len(records) - 2
        or footer["schema_version"] != RACKETVISION_CACHE_SCHEMA_VERSION
    ):
        raise RacketVisionCacheCorruptError(
            "RacketVision cache footer is invalid or incomplete."
        )

    observations = []
    for record in records[1:-1]:
        frame = _exact_keys("cache frame", record, {"observation", "type"})
        if frame["type"] != "frame":
            raise RacketVisionCacheCorruptError(
                "records between header and footer must be frames."
            )
        observations.append(_observation_from_dict(frame["observation"]))
    try:
        items = _validate_observations(observations, expected_times)
    except RacketVisionCacheError as exc:
        raise RacketVisionCacheCorruptError(str(exc)) from exc
    if stored_identity.timeline_fingerprint != fingerprint_frame_times(expected_times):
        raise RacketVisionCacheStaleError(
            "RacketVision cache timeline fingerprint does not match canonical PTS."
        )
    return RacketVisionCacheSnapshot(stored_identity, items)
