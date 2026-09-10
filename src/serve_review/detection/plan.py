"""Attempt padding planner (M3.3).

Turns unpadded :class:`CandidateRange` values (M3.2) into a versioned
:class:`AttemptDocument` (domain) with stable identifiers, padded
effective ranges clamped to the source bounds, and unioned export
ranges. Pure, deterministic, std-lib only.

Planning rules:

- Candidates are sorted by ``(start, end)`` before identifiers are
  assigned, so unsorted input still yields a deterministic document.
- Identifiers are ``serve-001``, ``serve-002``, ... in sorted order.
  Replanning the same candidates against the same source and padding
  yields identical identifiers (stable for one analysis run).
- Symmetric padding ``pad`` expands each detected range to
  ``[start - pad, end + pad)`` clamped to ``[0, source_duration]``.
  Zero padding preserves detected bounds exactly.
- Detected ranges must lie inside ``[0, source_duration]``; ranges
  extending beyond the source are rejected rather than silently
  clipped, so detector/source mismatches surface as errors.
- ``export_ranges`` are the union of strictly overlapping effective
  ranges. Adjacent ranges (``end == start``) stay separate because an
  :class:`ExportPlan` already permits adjacency. Detected ranges are
  never rewritten by the union.
- Empty input yields an empty document (no attempts, no export
  ranges). Converting an empty document to an :class:`ExportPlan`
  raises :class:`AttemptError`; callers must surface empty detection
  honestly instead of substituting the full video.
- ``confidence`` (``None`` or in ``[0, 1]``) and ``evidence``
  (mapping of names to finite numbers) are carried verbatim and
  serialized deterministically.

All thresholds live in the versioned :class:`PlanConfig` with a JSON
codec that rejects unknown keys and unsupported schema versions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from serve_review.domain import (
    Attempt,
    AttemptDocument,
    AttemptError,
    ExportPlan,
    MediaRange,
    SourceMetadata,
)
from serve_review.detection.ranges import CandidateRange

__all__ = [
    "PLAN_SCHEMA_VERSION",
    "DEFAULT_METHOD_VERSION",
    "PlanError",
    "PlanConfig",
    "union_effective_ranges",
    "plan_attempts",
    "to_export_plan",
]

#: Version of the padding-planner configuration schema.
PLAN_SCHEMA_VERSION = 1

#: Default method identity recorded on planned documents.
DEFAULT_METHOD_VERSION = "candidate-ranges-v1+plan-v1"


class PlanError(ValueError):
    """Raised when padding-planner input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class PlanConfig:
    """Versioned symmetric padding configuration.

    ``padding_seconds`` is applied symmetrically on both sides of every
    detected range and clamped to the source bounds (``>= 0``).
    """

    padding_seconds: float = 1.0
    schema_version: int = PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise PlanError(
                "plan_config: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise PlanError(
                f"plan_config: unsupported schema_version {self.schema_version!r}; "
                f"expected {PLAN_SCHEMA_VERSION}."
            )
        if isinstance(self.padding_seconds, bool) or not isinstance(
            self.padding_seconds, (int, float)
        ):
            raise PlanError(
                "plan_config: 'padding_seconds' must be a number, "
                f"got {self.padding_seconds!r}."
            )
        number = float(self.padding_seconds)
        if not math.isfinite(number) or number < 0.0:
            raise PlanError(
                "plan_config: 'padding_seconds' must be finite and >= 0, "
                f"got {self.padding_seconds!r}."
            )
        object.__setattr__(self, "padding_seconds", number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "padding_seconds": self.padding_seconds,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PlanConfig:
        if not isinstance(values, dict):
            raise PlanError(
                "plan_config: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {"padding_seconds", "schema_version"}
        missing = sorted(known - set(values))
        if missing:
            raise PlanError(f"plan_config: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise PlanError(f"plan_config: unknown keys {unknown!r}.")
        return cls(
            padding_seconds=values["padding_seconds"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PlanConfig:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PlanError("plan_config: invalid UTF-8 JSON payload.") from exc
        if not isinstance(data, str):
            raise PlanError(
                "plan_config: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PlanError(f"plan_config: invalid JSON: {exc}.") from exc
        return cls.from_dict(decoded)


def union_effective_ranges(
    ranges: Sequence[MediaRange],
) -> tuple[MediaRange, ...]:
    """Union strictly overlapping ranges, preserving adjacency.

    Input is sorted internally; output is sorted and pairwise
    non-overlapping. Adjacent ranges (``end == start``) are kept
    separate. An empty sequence yields ``()``.
    """
    items = list(ranges)
    for entry in items:
        if not isinstance(entry, MediaRange):
            raise PlanError(
                "plan: every range must be a MediaRange, "
                f"got {type(entry).__name__}."
            )
    if not items:
        return ()
    ordered = sorted(items, key=lambda item: (item.start_seconds, item.end_seconds))
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


def _check_confidences(
    confidences: Sequence[float | None] | None, count: int
) -> tuple[float | None, ...]:
    if confidences is None:
        return (None,) * count
    items = list(confidences)
    if len(items) != count:
        raise PlanError(
            f"plan: 'confidences' length ({len(items)!r}) must match "
            f"candidate count ({count!r})."
        )
    cleaned: list[float | None] = []
    for value in items:
        if value is None:
            cleaned.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PlanError(
                "plan: every confidence must be in [0, 1] or None, "
                f"got {value!r}."
            )
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 1.0:
            raise PlanError(
                "plan: every confidence must lie in [0, 1] or be None, "
                f"got {value!r}."
            )
        cleaned.append(number)
    return tuple(cleaned)


def _check_evidences(
    evidences: Sequence[Mapping[str, float] | dict[str, float] | None] | None,
    count: int,
) -> tuple[dict[str, float], ...]:
    if evidences is None:
        return ({},) * count
    items = list(evidences)
    if len(items) != count:
        raise PlanError(
            f"plan: 'evidences' length ({len(items)!r}) must match "
            f"candidate count ({count!r})."
        )
    cleaned: list[dict[str, float]] = []
    for value in items:
        if value is None:
            cleaned.append({})
            continue
        if not isinstance(value, Mapping):
            raise PlanError(
                "plan: every evidence must be a mapping or None, "
                f"got {type(value).__name__}."
            )
        entry: dict[str, float] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise PlanError(
                    f"plan: evidence keys must be non-blank strings, got {key!r}."
                )
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise PlanError(
                    f"plan: evidence[{key!r}] must be a finite number, "
                    f"got {item!r}."
                )
            number = float(item)
            if not math.isfinite(number):
                raise PlanError(
                    f"plan: evidence[{key!r}] must be finite, got {item!r}."
                )
            entry[key] = number
        cleaned.append(entry)
    return tuple(cleaned)


def plan_attempts(
    candidates: Sequence[CandidateRange],
    source: SourceMetadata,
    config: PlanConfig | None = None,
    *,
    confidences: Sequence[float | None] | None = None,
    evidences: Sequence[Mapping[str, float] | dict[str, float] | None] | None = None,
    method_version: str = DEFAULT_METHOD_VERSION,
) -> AttemptDocument:
    """Plan versioned attempts from unpadded candidates.

    ``candidates`` are sorted internally and assigned stable
    ``serve-NNN`` identifiers. Each detected range is padded
    symmetrically by ``config.padding_seconds`` and clamped to the
    source duration. Overlapping effective ranges are unioned for
    export without rewriting detected ranges. Empty input yields an
    empty document.
    """
    cfg = config if config is not None else PlanConfig()
    if not isinstance(cfg, PlanConfig):
        raise PlanError(
            "plan: 'config' must be a PlanConfig, "
            f"got {type(cfg).__name__}."
        )
    if not isinstance(source, SourceMetadata):
        raise PlanError(
            "plan: 'source' must be SourceMetadata, "
            f"got {type(source).__name__}."
        )
    if isinstance(candidates, CandidateRange) or not isinstance(
        candidates, (list, tuple)
    ):
        raise PlanError(
            "plan: 'candidates' must be a list or tuple of CandidateRange, "
            f"got {type(candidates).__name__}."
        )
    items = list(candidates)
    for entry in items:
        if not isinstance(entry, CandidateRange):
            raise PlanError(
                "plan: every candidate must be a CandidateRange, "
                f"got {type(entry).__name__}."
            )
    if not isinstance(method_version, str) or not method_version.strip():
        raise PlanError(
            "plan: 'method_version' must be a non-blank string, "
            f"got {method_version!r}."
        )
    duration = source.duration_seconds
    ordered = sorted(items, key=lambda item: (item.start_seconds, item.end_seconds))
    for entry in ordered:
        if entry.end_seconds > duration:
            raise PlanError(
                f"plan: candidate {entry.to_dict()!r} extends beyond source "
                f"duration ({duration!r})."
            )
    clean_conf = _check_confidences(confidences, len(ordered))
    clean_ev = _check_evidences(evidences, len(ordered))
    attempts: list[Attempt] = []
    for index, (entry, confidence, evidence) in enumerate(
        zip(ordered, clean_conf, clean_ev), start=1
    ):
        detected = MediaRange(
            start_seconds=entry.start_seconds, end_seconds=entry.end_seconds
        )
        effective = detected.with_padding(cfg.padding_seconds, duration)
        try:
            attempts.append(
                Attempt(
                    attempt_id=f"serve-{index:03d}",
                    detected_range=detected,
                    effective_range=effective,
                    confidence=confidence,
                    evidence=dict(evidence),
                )
            )
        except (AttemptError, ValueError) as exc:
            raise PlanError(f"plan: invalid attempt #{index}: {exc}.") from exc
    export_ranges = union_effective_ranges(
        [attempt.effective_range for attempt in attempts]
    )
    try:
        return AttemptDocument(
            source_fingerprint=source.fingerprint,
            source_duration_seconds=duration,
            padding_seconds=cfg.padding_seconds,
            method_version=method_version,
            attempts=tuple(attempts),
            export_ranges=export_ranges,
        )
    except (AttemptError, ValueError) as exc:
        raise PlanError(f"plan: invalid attempt document: {exc}.") from exc


def to_export_plan(document: AttemptDocument) -> ExportPlan:
    """Return an :class:`ExportPlan` over a document's export ranges."""
    if not isinstance(document, AttemptDocument):
        raise PlanError(
            "plan: 'document' must be an AttemptDocument, "
            f"got {type(document).__name__}."
        )
    try:
        return document.to_export_plan()
    except (AttemptError, ValueError) as exc:
        raise PlanError(str(exc)) from exc
