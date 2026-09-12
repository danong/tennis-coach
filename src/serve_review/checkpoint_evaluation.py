"""Deterministic manual-gate phase evaluation (M4.6).

Pure module: no file I/O, no subprocesses, no solver/feature thresholds.
It validates session-disjoint manual phase annotation manifests and
evaluates one M4.1 :class:`PhaseDocument` against one validated manifest
without tuning or changing solver behavior.

Manual model
------------

A :class:`PhaseAnnotationManifest` carries one dataset split (``"dev"``
or ``"heldout"``) and a list of :class:`PhaseAnnotation` rows. Each row
identifies a source-relative ``media`` path, a recording ``session_id``,
an attempt linkage (``attempt_id`` plus the unpadded half-open
``[attempt_start_seconds, attempt_end_seconds)`` range), one canonical
stage key from :data:`STAGE_ORDER`, and either a manually accepted
half-open interval (``status == "available"``) or an explicit
``"unavailable"`` / ``"ambiguous"`` mark with no manual times. An
optional representative manual keyframe, an optional annotator
``confidence`` in ``[0, 1]``, and an optional per-attempt structural
expectation (``attempt_label`` in ``{"serve", "aborted"}``) may be
supplied as appropriate.

Manual uncertainty intervals are never merged with machine intervals:
acceptance checks whether the machine keyframe falls inside the manual
accepted interval, and overlap/IoU is reported only where both sides
carry intervals. ``"unavailable"`` / ``"ambiguous"`` manual rows never
require ball/racket ground truth; a missing machine phase against such
a row is reported honestly (``accepted`` / ``iou`` / ``error`` are null)
instead of being counted as a false visual claim.

Matching
--------

Evaluation matches attempts deterministically by identity plus range
linkage: the manifest attempt-id set must exactly equal the
:class:`PhaseDocument` attempt-id set, and each manifest attempt range
must exactly equal the document's unpadded ``attempt_range``. No
cross-attempt or cross-session pairing occurs. Within a matched attempt,
stages pair by canonical key in :data:`STAGE_ORDER`.

Metrics
-------

Per-stage and pooled (``stage == "overall"``) aggregates report:

- annotated availability versus predicted availability counts;
- accepted-keyframe rate: fraction of evaluable pairs (manual
  ``available`` with a machine keyframe) whose machine keyframe lies in
  the manual half-open interval ``[start, end)``;
- interval overlap/IoU where both sides carry intervals
  (``intersection = max(0, min_ends - max_starts)``,
  ``union = dur_manual + dur_machine - intersection``,
  ``iou = intersection / union``);
- signed/absolute keyframe timing error (``pred - manual``) where both
  sides carry keyframes, with mean signed, mean absolute, median
  absolute, and high-percentile (p90) absolute error;
- stage-order violations for machine and manual sequences;
- confidence calibration over accepted-keyframe pairs;
- structural completeness distribution and serve-versus-aborted
  structural separation.

Percentiles use the nearest-rank rule: for ``n`` ascending values,
``p_q`` is the element at ``ceil(q * n) - 1`` (0-based). The median is
the middle value for odd ``n`` and the mean of the two middle values
for even ``n`` (computed with :func:`math.fsum`). All other aggregates
use :func:`math.fsum` over deterministically sorted inputs, so reports
are bit-for-bit reproducible.

Confidence calibration formula (documented here and in
:func:`evaluate_phase_document`): over the ``n`` accepted-keyframe
evaluable pairs, correctness ``y_i`` is ``1.0`` when the machine
keyframe is accepted and ``0.0`` otherwise, with machine confidence
``c_i``. The Brier-style score is
``mean((c_i - y_i)^2)``. Deterministic equal-width bins over ``[0, 1]``
with edges ``(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)`` assign
``bin = min(int(c * 5), 4)``; each bin reports its count, mean
confidence, mean accuracy, and absolute gap ``|mean_conf - mean_acc|``
(null when empty).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from serve_review.domain import (
    STAGE_ORDER,
    AttemptPhase,
    PhaseDocument,
)

__all__ = [
    "CHECKPOINT_ANNOTATION_SCHEMA_VERSION",
    "CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION",
    "SPLIT_DEV",
    "SPLIT_HELDOUT",
    "ALLOWED_SPLITS",
    "ANNOTATION_STATUSES",
    "STATUS_AVAILABLE",
    "STATUS_UNAVAILABLE",
    "STATUS_AMBIGUOUS",
    "ATTEMPT_LABELS",
    "LABEL_SERVE",
    "LABEL_ABORTED",
    "CALIBRATION_BIN_EDGES",
    "HIGH_PERCENTILE_Q",
    "CheckpointEvaluationError",
    "PhaseAnnotation",
    "PhaseAnnotationManifest",
    "StagePairDetail",
    "AttemptEvaluation",
    "StageAggregate",
    "CalibrationBin",
    "OrderViolation",
    "GroupStructuralSummary",
    "StructuralSeparation",
    "CheckpointEvaluationReport",
    "validate_session_disjoint",
    "interval_intersection_seconds",
    "interval_iou",
    "median_of_sorted",
    "nearest_rank_percentile",
    "evaluate_phase_document",
]

#: Version of the manual phase annotation manifest schema.
CHECKPOINT_ANNOTATION_SCHEMA_VERSION = 1

#: Version of the machine-readable checkpoint evaluation report schema.
CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION = 1

#: Development split name.
SPLIT_DEV = "dev"

#: Held-out split name.
SPLIT_HELDOUT = "heldout"

#: Splits accepted on checkpoint manifests.
ALLOWED_SPLITS = frozenset({SPLIT_DEV, SPLIT_HELDOUT})

#: Manual annotation status: accepted interval present.
STATUS_AVAILABLE = "available"

#: Manual annotation status: explicitly no observable stage.
STATUS_UNAVAILABLE = "unavailable"

#: Manual annotation status: explicitly uncertain / body-only ambiguity.
STATUS_AMBIGUOUS = "ambiguous"

#: Statuses accepted on phase annotations.
ANNOTATION_STATUSES = (STATUS_AVAILABLE, STATUS_UNAVAILABLE, STATUS_AMBIGUOUS)

#: Per-attempt structural expectation: a complete serve motion.
LABEL_SERVE = "serve"

#: Per-attempt structural expectation: an aborted/truncated motion.
LABEL_ABORTED = "aborted"

#: Attempt labels accepted on phase annotations (or null when ungraded).
ATTEMPT_LABELS = frozenset({LABEL_SERVE, LABEL_ABORTED})

#: Deterministic equal-width calibration bin edges over [0, 1].
CALIBRATION_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

#: High-percentile level for absolute keyframe timing error.
HIGH_PERCENTILE_Q = 0.9

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_ATTEMPT_ID_PREFIX = "serve-"


class CheckpointEvaluationError(ValueError):
    """Raised when checkpoint evaluation input or a document is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _dumps_deterministic(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _loads_object(name: str, data: str | bytes | bytearray) -> dict[str, Any]:
    if isinstance(data, (bytes, bytearray)):
        try:
            data = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid UTF-8 JSON payload."
            ) from exc
    if not isinstance(data, str):
        raise CheckpointEvaluationError(
            f"{name}: JSON payload must be str or bytes, "
            f"got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise CheckpointEvaluationError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise CheckpointEvaluationError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


def _check_schema_version(name: str, values: Mapping[str, Any], current: int) -> None:
    if "schema_version" not in values:
        raise CheckpointEvaluationError(
            f"{name}: missing required key 'schema_version'."
        )
    version = values["schema_version"]
    if not _is_int(version):
        raise CheckpointEvaluationError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != current:
        if version > current:
            raise CheckpointEvaluationError(
                f"{name}: unsupported newer schema_version {version!r}; "
                f"this build supports version {current}."
            )
        raise CheckpointEvaluationError(
            f"{name}: unsupported schema_version {version!r}; "
            f"expected version {current}."
        )


def _check_no_unknown_keys(
    name: str, values: Mapping[str, Any], known: set[str]
) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise CheckpointEvaluationError(f"{name}: unknown keys {unknown!r}.")


def _check_non_blank(name: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must be a non-blank string, got {value!r}."
        )
    return value


def _check_time_bound(name: str, key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value) or value < 0:
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must be a finite number greater than or equal "
            f"to zero, got {value!r}."
        )
    return float(value)


def _check_relative_media(name: str, value: Any) -> str:
    _check_non_blank(name, "'media'", value)
    text = value
    if (
        os.path.isabs(text)
        or PurePosixPath(text).is_absolute()
        or text.startswith("\\\\")
        or _WINDOWS_DRIVE_RE.match(text) is not None
    ):
        raise CheckpointEvaluationError(
            f"{name}: 'media' must be a relative local path, got {value!r}."
        )
    if ".." in PurePosixPath(text).parts:
        raise CheckpointEvaluationError(
            f"{name}: 'media' must not escape its directory with '..', "
            f"got {value!r}."
        )
    if "\\" in text:
        raise CheckpointEvaluationError(
            f"{name}: 'media' must use POSIX-style relative paths, "
            f"got {value!r}."
        )
    return text


def _check_attempt_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointEvaluationError(
            f"{name}: 'attempt_id' must be a non-empty string, "
            f"got {value!r}."
        )
    body = (
        value[len(_ATTEMPT_ID_PREFIX):]
        if value.startswith(_ATTEMPT_ID_PREFIX)
        else None
    )
    if body is None or len(body) < 3 or not body.isdigit():
        raise CheckpointEvaluationError(
            f"{name}: 'attempt_id' must look like 'serve-NNN' with at least "
            f"three digits, got {value!r}."
        )
    return value


def _check_stage(name: str, value: Any) -> str:
    if value not in STAGE_ORDER:
        raise CheckpointEvaluationError(
            f"{name}: 'stage' must be one of {list(STAGE_ORDER)}, "
            f"got {value!r}."
        )
    return value


def _check_status(name: str, value: Any) -> str:
    if value not in ANNOTATION_STATUSES:
        raise CheckpointEvaluationError(
            f"{name}: 'status' must be one of {list(ANNOTATION_STATUSES)}, "
            f"got {value!r}."
        )
    return value


def _check_optional_bound(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    return _check_time_bound(name, key, value)


def _check_optional_confidence(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(value):
        raise CheckpointEvaluationError(
            f"{name}: 'confidence' must be a finite number in [0, 1] or "
            f"None, got {value!r}."
        )
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise CheckpointEvaluationError(
            f"{name}: 'confidence' must lie in [0, 1] or be None, "
            f"got {value!r}."
        )
    return number


def _check_optional_attempt_label(name: str, value: Any) -> str | None:
    if value is None:
        return None
    if value not in ATTEMPT_LABELS:
        raise CheckpointEvaluationError(
            f"{name}: 'attempt_label' must be one of "
            f"{sorted(ATTEMPT_LABELS)} or None, got {value!r}."
        )
    return value


def interval_intersection_seconds(
    first_start: float, first_end: float, second_start: float, second_end: float
) -> float:
    """Half-open overlap duration in seconds (``0.0`` when disjoint)."""
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def interval_iou(
    first_start: float, first_end: float, second_start: float, second_end: float
) -> float:
    """Intersection over union for two half-open intervals in ``[0, 1]``."""
    inter = interval_intersection_seconds(
        first_start, first_end, second_start, second_end
    )
    union = (first_end - first_start) + (second_end - second_start) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def median_of_sorted(values: Sequence[float]) -> float | None:
    """Median of an ascending sequence (None when empty).

    Odd ``n`` returns the middle value; even ``n`` returns the mean of
    the two middle values via :func:`math.fsum`.
    """
    n = len(values)
    if n == 0:
        return None
    ordered = list(values)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float(math.fsum((ordered[mid - 1], ordered[mid])) / 2.0)


def nearest_rank_percentile(values: Sequence[float], quantile: float) -> float | None:
    """Nearest-rank percentile of an ascending sequence (None when empty).

    Returns the element at ``ceil(q * n) - 1`` (0-based) for
    ``0 < q <= 1``.
    """
    name = "percentile"
    if (
        not _is_number(quantile)
        or not math.isfinite(quantile)
        or not 0.0 < quantile <= 1.0
    ):
        raise CheckpointEvaluationError(
            f"{name}: quantile must lie in (0, 1], got {quantile!r}."
        )
    n = len(values)
    if n == 0:
        return None
    index = int(math.ceil(float(quantile) * n)) - 1
    index = max(0, min(n - 1, index))
    return float(list(values)[index])


@dataclass(frozen=True, slots=True)
class PhaseAnnotation:
    """One manual stage annotation linked to an unpadded attempt range."""

    session_id: str = ""
    media: str = ""
    attempt_id: str = ""
    attempt_start_seconds: float = 0.0
    attempt_end_seconds: float = 0.0
    stage: str = "start"
    status: str = STATUS_AVAILABLE
    interval_start_seconds: float | None = None
    interval_end_seconds: float | None = None
    manual_keyframe_seconds: float | None = None
    confidence: float | None = None
    attempt_label: str | None = None

    def __post_init__(self) -> None:
        name = "phase_annotation"
        object.__setattr__(
            self, "session_id", _check_non_blank(name, "'session_id'", self.session_id)
        )
        object.__setattr__(self, "media", _check_relative_media(name, self.media))
        object.__setattr__(
            self, "attempt_id", _check_attempt_id(name, self.attempt_id)
        )
        start = _check_time_bound(
            name, "'attempt_start_seconds'", self.attempt_start_seconds
        )
        end = _check_time_bound(
            name, "'attempt_end_seconds'", self.attempt_end_seconds
        )
        if not start < end:
            raise CheckpointEvaluationError(
                f"{name}: 'attempt_start_seconds' ({start!r}) must be less "
                f"than 'attempt_end_seconds' ({end!r})."
            )
        object.__setattr__(self, "attempt_start_seconds", start)
        object.__setattr__(self, "attempt_end_seconds", end)
        object.__setattr__(self, "stage", _check_stage(name, self.stage))
        object.__setattr__(self, "status", _check_status(name, self.status))
        interval_start = _check_optional_bound(
            name, "'interval_start_seconds'", self.interval_start_seconds
        )
        interval_end = _check_optional_bound(
            name, "'interval_end_seconds'", self.interval_end_seconds
        )
        keyframe = _check_optional_bound(
            name, "'manual_keyframe_seconds'", self.manual_keyframe_seconds
        )
        if self.status == STATUS_AVAILABLE:
            if interval_start is None or interval_end is None:
                raise CheckpointEvaluationError(
                    f"{name}: 'interval_start_seconds' and "
                    f"'interval_end_seconds' are required when "
                    f"status is 'available', got "
                    f"({self.interval_start_seconds!r}, "
                    f"{self.interval_end_seconds!r})."
                )
            if not interval_start < interval_end:
                raise CheckpointEvaluationError(
                    f"{name}: manual interval start ({interval_start!r}) must "
                    f"be less than end ({interval_end!r})."
                )
            if interval_start < start or interval_end > end:
                raise CheckpointEvaluationError(
                    f"{name}: manual interval "
                    f"[{interval_start!r}, {interval_end!r}) must lie inside "
                    f"attempt range [{start!r}, {end!r})."
                )
            if keyframe is not None and not (
                interval_start <= keyframe < interval_end
            ):
                raise CheckpointEvaluationError(
                    f"{name}: 'manual_keyframe_seconds' ({keyframe!r}) must "
                    f"lie in manual interval [{interval_start!r}, "
                    f"{interval_end!r})."
                )
        else:
            if interval_start is not None or interval_end is not None:
                raise CheckpointEvaluationError(
                    f"{name}: manual interval must be null when status is "
                    f"{self.status!r}, got ({self.interval_start_seconds!r}, "
                    f"{self.interval_end_seconds!r})."
                )
            if keyframe is not None:
                raise CheckpointEvaluationError(
                    f"{name}: 'manual_keyframe_seconds' must be null when "
                    f"status is {self.status!r}, got {keyframe!r}."
                )
            interval_start = None
            interval_end = None
            keyframe = None
        object.__setattr__(self, "interval_start_seconds", interval_start)
        object.__setattr__(self, "interval_end_seconds", interval_end)
        object.__setattr__(self, "manual_keyframe_seconds", keyframe)
        object.__setattr__(
            self, "confidence", _check_optional_confidence(name, self.confidence)
        )
        object.__setattr__(
            self,
            "attempt_label",
            _check_optional_attempt_label(name, self.attempt_label),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_end_seconds": self.attempt_end_seconds,
            "attempt_id": self.attempt_id,
            "attempt_label": self.attempt_label,
            "attempt_start_seconds": self.attempt_start_seconds,
            "confidence": self.confidence,
            "interval_end_seconds": self.interval_end_seconds,
            "interval_start_seconds": self.interval_start_seconds,
            "manual_keyframe_seconds": self.manual_keyframe_seconds,
            "media": self.media,
            "session_id": self.session_id,
            "stage": self.stage,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseAnnotation:
        name = "phase_annotation"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempt_end_seconds",
            "attempt_id",
            "attempt_label",
            "attempt_start_seconds",
            "confidence",
            "interval_end_seconds",
            "interval_start_seconds",
            "manual_keyframe_seconds",
            "media",
            "session_id",
            "stage",
            "status",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        return cls(
            session_id=values["session_id"],
            media=values["media"],
            attempt_id=values["attempt_id"],
            attempt_start_seconds=values["attempt_start_seconds"],
            attempt_end_seconds=values["attempt_end_seconds"],
            stage=values["stage"],
            status=values["status"],
            interval_start_seconds=values["interval_start_seconds"],
            interval_end_seconds=values["interval_end_seconds"],
            manual_keyframe_seconds=values["manual_keyframe_seconds"],
            confidence=values["confidence"],
            attempt_label=values["attempt_label"],
        )


@dataclass(frozen=True, slots=True)
class PhaseAnnotationManifest:
    """Versioned set of manual phase annotations for one dataset split."""

    schema_version: int = CHECKPOINT_ANNOTATION_SCHEMA_VERSION
    dataset_split: str = SPLIT_DEV
    annotations: tuple[PhaseAnnotation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = "checkpoint_annotation_manifest"
        version = self.schema_version
        if not _is_int(version):
            raise CheckpointEvaluationError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != CHECKPOINT_ANNOTATION_SCHEMA_VERSION:
            if version > CHECKPOINT_ANNOTATION_SCHEMA_VERSION:
                raise CheckpointEvaluationError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version "
                    f"{CHECKPOINT_ANNOTATION_SCHEMA_VERSION}."
                )
            raise CheckpointEvaluationError(
                f"{name}: unsupported schema_version {version!r}; expected "
                f"version {CHECKPOINT_ANNOTATION_SCHEMA_VERSION}."
            )
        if self.dataset_split not in ALLOWED_SPLITS:
            raise CheckpointEvaluationError(
                f"{name}: 'dataset_split' must be one of "
                f"{sorted(ALLOWED_SPLITS)}, got {self.dataset_split!r}."
            )
        raw = self.annotations
        if isinstance(raw, PhaseAnnotation) or not isinstance(raw, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'annotations' must be a list or tuple of "
                f"PhaseAnnotation, got {type(raw).__name__}."
            )
        items = tuple(raw)
        for entry in items:
            if not isinstance(entry, PhaseAnnotation):
                raise CheckpointEvaluationError(
                    f"{name}: every annotation must be a PhaseAnnotation, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "annotations", items)
        seen: dict[tuple[str, str], PhaseAnnotation] = {}
        linkage: dict[str, tuple[str, str, float, float, str | None]] = {}
        for entry in items:
            key = (entry.attempt_id, entry.stage)
            if key in seen:
                previous = seen[key]
                if previous == entry:
                    raise CheckpointEvaluationError(
                        f"{name}: duplicate annotation for attempt "
                        f"{entry.attempt_id!r} stage {entry.stage!r}."
                    )
                raise CheckpointEvaluationError(
                    f"{name}: conflicting annotations for attempt "
                    f"{entry.attempt_id!r} stage {entry.stage!r}."
                )
            seen[key] = entry
            link = (
                entry.session_id,
                entry.media,
                entry.attempt_start_seconds,
                entry.attempt_end_seconds,
                entry.attempt_label,
            )
            if entry.attempt_id in linkage and linkage[entry.attempt_id] != link:
                raise CheckpointEvaluationError(
                    f"{name}: conflicting attempt linkage for attempt "
                    f"{entry.attempt_id!r}: rows for one attempt must share "
                    f"session/media/range/label."
                )
            linkage[entry.attempt_id] = link

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.annotations)

    @property
    def sessions(self) -> tuple[str, ...]:
        """Sorted session identifiers referenced by this manifest."""
        return tuple(sorted({entry.session_id for entry in self.annotations}))

    @property
    def attempt_ids(self) -> tuple[str, ...]:
        """Sorted attempt identifiers referenced by this manifest."""
        return tuple(sorted({entry.attempt_id for entry in self.annotations}))

    def rows_for_attempt(self, attempt_id: str) -> tuple[PhaseAnnotation, ...]:
        """Rows for one attempt in canonical :data:`STAGE_ORDER`."""
        order = {key: index for index, key in enumerate(STAGE_ORDER)}
        rows = [entry for entry in self.annotations if entry.attempt_id == attempt_id]
        return tuple(sorted(rows, key=lambda entry: order[entry.stage]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotations": [entry.to_dict() for entry in self.annotations],
            "dataset_split": self.dataset_split,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseAnnotationManifest:
        name = "checkpoint_annotation_manifest"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"annotations", "dataset_split", "schema_version"}
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        _check_schema_version(name, values, CHECKPOINT_ANNOTATION_SCHEMA_VERSION)
        raw = values["annotations"]
        if not isinstance(raw, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'annotations' must be a list of annotation "
                f"objects, got {type(raw).__name__}."
            )
        try:
            parsed = tuple(PhaseAnnotation.from_dict(entry) for entry in raw)
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid annotation: {exc}."
            ) from exc
        split = values["dataset_split"]
        if split not in ALLOWED_SPLITS:
            raise CheckpointEvaluationError(
                f"{name}: 'dataset_split' must be one of "
                f"{sorted(ALLOWED_SPLITS)}, got {split!r}."
            )
        return cls(
            schema_version=values["schema_version"],
            dataset_split=split,
            annotations=parsed,
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PhaseAnnotationManifest:
        return cls.from_dict(
            _loads_object("checkpoint_annotation_manifest", data)
        )


def validate_session_disjoint(
    first: PhaseAnnotationManifest, second: PhaseAnnotationManifest
) -> None:
    """Reject any shared ``session_id`` between two checkpoint manifests."""
    name = "session_disjoint"
    if not isinstance(first, PhaseAnnotationManifest):
        raise CheckpointEvaluationError(
            f"{name}: first manifest must be a PhaseAnnotationManifest, "
            f"got {type(first).__name__}."
        )
    if not isinstance(second, PhaseAnnotationManifest):
        raise CheckpointEvaluationError(
            f"{name}: second manifest must be a PhaseAnnotationManifest, "
            f"got {type(second).__name__}."
        )
    leaked = sorted(set(first.sessions) & set(second.sessions))
    if leaked:
        raise CheckpointEvaluationError(
            f"{name}: session leakage between manifests: {leaked!r}. "
            f"Split labels by recording session, not by nearby clips."
        )


def _check_optional_float_or_none(
    name: str, key: str, value: Any, *, allow_negative: bool = False
) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(value):
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must be a finite number or None, "
            f"got {value!r}."
        )
    number = float(value)
    if not allow_negative and number < 0.0:
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must be >= 0 or None, got {value!r}."
        )
    return number


def _check_ratio_or_none(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(value):
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must be a finite number in [0, 1] or None, "
            f"got {value!r}."
        )
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must lie in [0, 1] or be None, got {value!r}."
        )
    return number


def _check_bool_or_none(name: str, key: str, value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must be a boolean or None, got {value!r}."
        )
    return value


@dataclass(frozen=True, slots=True)
class StagePairDetail:
    """Per-attempt, per-stage manual/machine comparison (nulls stay honest)."""

    stage: str = "start"
    manual_status: str = STATUS_AVAILABLE
    manual_interval_start_seconds: float | None = None
    manual_interval_end_seconds: float | None = None
    manual_keyframe_seconds: float | None = None
    machine_availability: str = "unavailable"
    machine_interval_start_seconds: float | None = None
    machine_interval_end_seconds: float | None = None
    machine_keyframe_seconds: float | None = None
    machine_confidence: float = 0.0
    accepted: bool | None = None
    iou: float | None = None
    intersection_seconds: float | None = None
    signed_error_seconds: float | None = None
    abs_error_seconds: float | None = None

    def __post_init__(self) -> None:
        name = "stage_pair_detail"
        object.__setattr__(self, "stage", _check_stage(name, self.stage))
        object.__setattr__(self, "manual_status", _check_status(name, self.manual_status))
        if self.machine_availability not in (
            "available",
            "partial",
            "unavailable",
        ):
            raise CheckpointEvaluationError(
                f"{name}: 'machine_availability' must be one of "
                f"['available', 'partial', 'unavailable'], "
                f"got {self.machine_availability!r}."
            )
        for key in (
            "manual_interval_start_seconds",
            "manual_interval_end_seconds",
            "manual_keyframe_seconds",
            "machine_interval_start_seconds",
            "machine_interval_end_seconds",
            "machine_keyframe_seconds",
        ):
            object.__setattr__(
                self,
                key,
                _check_optional_float_or_none(name, key, getattr(self, key)),
            )
        conf = self.machine_confidence
        if not _is_number(conf) or not math.isfinite(conf):
            raise CheckpointEvaluationError(
                f"{name}: 'machine_confidence' must be finite, got {conf!r}."
            )
        number = float(conf)
        if number < 0.0 or number > 1.0:
            raise CheckpointEvaluationError(
                f"{name}: 'machine_confidence' must lie in [0, 1], "
                f"got {conf!r}."
            )
        object.__setattr__(self, "machine_confidence", number)
        object.__setattr__(
            self, "accepted", _check_bool_or_none(name, "accepted", self.accepted)
        )
        object.__setattr__(
            self, "iou", _check_ratio_or_none(name, "iou", self.iou)
        )
        object.__setattr__(
            self,
            "intersection_seconds",
            _check_optional_float_or_none(
                name, "intersection_seconds", self.intersection_seconds
            ),
        )
        object.__setattr__(
            self,
            "signed_error_seconds",
            _check_optional_float_or_none(
                name, "signed_error_seconds", self.signed_error_seconds,
                allow_negative=True,
            ),
        )
        object.__setattr__(
            self,
            "abs_error_seconds",
            _check_optional_float_or_none(
                name, "abs_error_seconds", self.abs_error_seconds
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "abs_error_seconds": self.abs_error_seconds,
            "accepted": self.accepted,
            "intersection_seconds": self.intersection_seconds,
            "iou": self.iou,
            "machine_availability": self.machine_availability,
            "machine_confidence": self.machine_confidence,
            "machine_interval_end_seconds": self.machine_interval_end_seconds,
            "machine_interval_start_seconds": self.machine_interval_start_seconds,
            "machine_keyframe_seconds": self.machine_keyframe_seconds,
            "manual_interval_end_seconds": self.manual_interval_end_seconds,
            "manual_interval_start_seconds": self.manual_interval_start_seconds,
            "manual_keyframe_seconds": self.manual_keyframe_seconds,
            "manual_status": self.manual_status,
            "signed_error_seconds": self.signed_error_seconds,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StagePairDetail:
        name = "stage_pair_detail"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "abs_error_seconds",
            "accepted",
            "intersection_seconds",
            "iou",
            "machine_availability",
            "machine_confidence",
            "machine_interval_end_seconds",
            "machine_interval_start_seconds",
            "machine_keyframe_seconds",
            "manual_interval_end_seconds",
            "manual_interval_start_seconds",
            "manual_keyframe_seconds",
            "manual_status",
            "signed_error_seconds",
            "stage",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        return cls(**{key: values[key] for key in known})


@dataclass(frozen=True, slots=True)
class AttemptEvaluation:
    """Per-attempt manual/machine comparison across the eight stages."""

    attempt_id: str = ""
    session_id: str = ""
    media: str = ""
    attempt_start_seconds: float = 0.0
    attempt_end_seconds: float = 0.0
    attempt_label: str | None = None
    structural_status: str = "unavailable"
    stages: tuple[StagePairDetail, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = "attempt_evaluation"
        object.__setattr__(
            self, "attempt_id", _check_attempt_id(name, self.attempt_id)
        )
        object.__setattr__(
            self, "session_id", _check_non_blank(name, "'session_id'", self.session_id)
        )
        object.__setattr__(self, "media", _check_relative_media(name, self.media))
        start = _check_time_bound(name, "'attempt_start_seconds'", self.attempt_start_seconds)
        end = _check_time_bound(name, "'attempt_end_seconds'", self.attempt_end_seconds)
        if not start < end:
            raise CheckpointEvaluationError(
                f"{name}: attempt range must satisfy start < end, got "
                f"({start!r}, {end!r})."
            )
        object.__setattr__(self, "attempt_start_seconds", start)
        object.__setattr__(self, "attempt_end_seconds", end)
        object.__setattr__(
            self,
            "attempt_label",
            _check_optional_attempt_label(name, self.attempt_label),
        )
        if self.structural_status not in (
            "complete",
            "partial",
            "incomplete",
            "unavailable",
        ):
            raise CheckpointEvaluationError(
                f"{name}: 'structural_status' must be one of "
                f"['complete', 'partial', 'incomplete', 'unavailable'], "
                f"got {self.structural_status!r}."
            )
        raw = self.stages
        if not isinstance(raw, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'stages' must be a list or tuple, "
                f"got {type(raw).__name__}."
            )
        items = tuple(raw)
        if len(items) != len(STAGE_ORDER):
            raise CheckpointEvaluationError(
                f"{name}: 'stages' must contain exactly eight entries, "
                f"got {len(items)}."
            )
        for entry in items:
            if not isinstance(entry, StagePairDetail):
                raise CheckpointEvaluationError(
                    f"{name}: every stage must be a StagePairDetail, "
                    f"got {type(entry).__name__}."
                )
        if [entry.stage for entry in items] != list(STAGE_ORDER):
            raise CheckpointEvaluationError(
                f"{name}: 'stages' must follow canonical order "
                f"{list(STAGE_ORDER)}, got {[entry.stage for entry in items]!r}."
            )
        object.__setattr__(self, "stages", items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_end_seconds": self.attempt_end_seconds,
            "attempt_id": self.attempt_id,
            "attempt_label": self.attempt_label,
            "attempt_start_seconds": self.attempt_start_seconds,
            "media": self.media,
            "session_id": self.session_id,
            "stages": [entry.to_dict() for entry in self.stages],
            "structural_status": self.structural_status,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AttemptEvaluation:
        name = "attempt_evaluation"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempt_end_seconds",
            "attempt_id",
            "attempt_label",
            "attempt_start_seconds",
            "media",
            "session_id",
            "stages",
            "structural_status",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        raw = values["stages"]
        if not isinstance(raw, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'stages' must be a list, got {type(raw).__name__}."
            )
        try:
            parsed = tuple(StagePairDetail.from_dict(entry) for entry in raw)
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid stage: {exc}."
            ) from exc
        return cls(
            attempt_id=values["attempt_id"],
            session_id=values["session_id"],
            media=values["media"],
            attempt_start_seconds=values["attempt_start_seconds"],
            attempt_end_seconds=values["attempt_end_seconds"],
            attempt_label=values["attempt_label"],
            structural_status=values["structural_status"],
            stages=parsed,
        )


def _check_count(name: str, key: str, value: Any) -> int:
    if not _is_int(value) or value < 0:
        raise CheckpointEvaluationError(
            f"{name}: {key!r} must be an integer >= 0, got {value!r}."
        )
    return value


@dataclass(frozen=True, slots=True)
class StageAggregate:
    """Counts and pooled error/overlap statistics for one stage (or overall)."""

    stage: str = "overall"
    n_annotated_available: int = 0
    n_annotated_unavailable: int = 0
    n_annotated_ambiguous: int = 0
    n_predicted_available: int = 0
    n_predicted_unavailable: int = 0
    n_both_available: int = 0
    n_machine_missing_manual_available: int = 0
    n_machine_available_manual_not_available: int = 0
    n_keyframe_evaluated: int = 0
    n_keyframe_accepted: int = 0
    accepted_keyframe_rate: float | None = None
    n_overlap_evaluated: int = 0
    mean_iou: float | None = None
    mean_intersection_seconds: float | None = None
    n_timing_evaluated: int = 0
    mean_signed_error_seconds: float | None = None
    mean_abs_error_seconds: float | None = None
    median_abs_error_seconds: float | None = None
    p90_abs_error_seconds: float | None = None

    def __post_init__(self) -> None:
        name = "stage_aggregate"
        if self.stage != "overall" and self.stage not in STAGE_ORDER:
            raise CheckpointEvaluationError(
                f"{name}: 'stage' must be 'overall' or one of "
                f"{list(STAGE_ORDER)}, got {self.stage!r}."
            )
        for key in (
            "n_annotated_available",
            "n_annotated_unavailable",
            "n_annotated_ambiguous",
            "n_predicted_available",
            "n_predicted_unavailable",
            "n_both_available",
            "n_machine_missing_manual_available",
            "n_machine_available_manual_not_available",
            "n_keyframe_evaluated",
            "n_keyframe_accepted",
            "n_overlap_evaluated",
            "n_timing_evaluated",
        ):
            object.__setattr__(
                self, key, _check_count(name, key, getattr(self, key))
            )
        if self.n_keyframe_accepted > self.n_keyframe_evaluated:
            raise CheckpointEvaluationError(
                f"{name}: accepted count exceeds evaluated count."
            )
        object.__setattr__(
            self,
            "accepted_keyframe_rate",
            _check_ratio_or_none(
                name, "accepted_keyframe_rate", self.accepted_keyframe_rate
            ),
        )
        if (self.accepted_keyframe_rate is None) != (
            self.n_keyframe_evaluated == 0
        ):
            raise CheckpointEvaluationError(
                f"{name}: 'accepted_keyframe_rate' must be None exactly when "
                f"no keyframes were evaluated."
            )
        object.__setattr__(
            self, "mean_iou", _check_ratio_or_none(name, "mean_iou", self.mean_iou)
        )
        if (self.mean_iou is None) != (self.n_overlap_evaluated == 0):
            raise CheckpointEvaluationError(
                f"{name}: 'mean_iou' must be None exactly when no overlaps "
                f"were evaluated."
            )
        object.__setattr__(
            self,
            "mean_intersection_seconds",
            _check_optional_float_or_none(
                name,
                "mean_intersection_seconds",
                self.mean_intersection_seconds,
            ),
        )
        if (self.mean_intersection_seconds is None) != (
            self.n_overlap_evaluated == 0
        ):
            raise CheckpointEvaluationError(
                f"{name}: 'mean_intersection_seconds' must be None exactly "
                f"when no overlaps were evaluated."
            )
        for key in (
            "mean_signed_error_seconds",
            "mean_abs_error_seconds",
            "median_abs_error_seconds",
            "p90_abs_error_seconds",
        ):
            allow_negative = key == "mean_signed_error_seconds"
            object.__setattr__(
                self,
                key,
                _check_optional_float_or_none(
                    name, key, getattr(self, key),
                    allow_negative=allow_negative,
                ),
            )
            if (getattr(self, key) is None) != (self.n_timing_evaluated == 0):
                raise CheckpointEvaluationError(
                    f"{name}: {key!r} must be None exactly when no timing "
                    f"pairs were evaluated."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_keyframe_rate": self.accepted_keyframe_rate,
            "mean_abs_error_seconds": self.mean_abs_error_seconds,
            "mean_intersection_seconds": self.mean_intersection_seconds,
            "mean_iou": self.mean_iou,
            "mean_signed_error_seconds": self.mean_signed_error_seconds,
            "median_abs_error_seconds": self.median_abs_error_seconds,
            "n_annotated_ambiguous": self.n_annotated_ambiguous,
            "n_annotated_available": self.n_annotated_available,
            "n_annotated_unavailable": self.n_annotated_unavailable,
            "n_both_available": self.n_both_available,
            "n_keyframe_accepted": self.n_keyframe_accepted,
            "n_keyframe_evaluated": self.n_keyframe_evaluated,
            "n_machine_available_manual_not_available": (
                self.n_machine_available_manual_not_available
            ),
            "n_machine_missing_manual_available": (
                self.n_machine_missing_manual_available
            ),
            "n_overlap_evaluated": self.n_overlap_evaluated,
            "n_predicted_available": self.n_predicted_available,
            "n_predicted_unavailable": self.n_predicted_unavailable,
            "n_timing_evaluated": self.n_timing_evaluated,
            "p90_abs_error_seconds": self.p90_abs_error_seconds,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StageAggregate:
        name = "stage_aggregate"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "accepted_keyframe_rate",
            "mean_abs_error_seconds",
            "mean_intersection_seconds",
            "mean_iou",
            "mean_signed_error_seconds",
            "median_abs_error_seconds",
            "n_annotated_ambiguous",
            "n_annotated_available",
            "n_annotated_unavailable",
            "n_both_available",
            "n_keyframe_accepted",
            "n_keyframe_evaluated",
            "n_machine_available_manual_not_available",
            "n_machine_missing_manual_available",
            "n_overlap_evaluated",
            "n_predicted_available",
            "n_predicted_unavailable",
            "n_timing_evaluated",
            "p90_abs_error_seconds",
            "stage",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        return cls(**{key: values[key] for key in known})


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One deterministic confidence bin over accepted-keyframe pairs."""

    bin_index: int = 0
    lo: float = 0.0
    hi: float = 0.2
    count: int = 0
    mean_confidence: float | None = None
    mean_accuracy: float | None = None
    abs_gap: float | None = None

    def __post_init__(self) -> None:
        name = "calibration_bin"
        if not _is_int(self.bin_index) or self.bin_index < 0:
            raise CheckpointEvaluationError(
                f"{name}: 'bin_index' must be an integer >= 0, "
                f"got {self.bin_index!r}."
            )
        for key in ("lo", "hi"):
            value = getattr(self, key)
            if not _is_number(value) or not math.isfinite(value):
                raise CheckpointEvaluationError(
                    f"{name}: {key!r} must be finite, got {value!r}."
                )
            object.__setattr__(self, key, float(value))
        object.__setattr__(self, "count", _check_count(name, "count", self.count))
        for key in ("mean_confidence", "mean_accuracy", "abs_gap"):
            object.__setattr__(
                self, key, _check_ratio_or_none(name, key, getattr(self, key))
            )
            if (getattr(self, key) is None) != (self.count == 0):
                raise CheckpointEvaluationError(
                    f"{name}: {key!r} must be None exactly when empty."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "abs_gap": self.abs_gap,
            "bin_index": self.bin_index,
            "count": self.count,
            "hi": self.hi,
            "lo": self.lo,
            "mean_accuracy": self.mean_accuracy,
            "mean_confidence": self.mean_confidence,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CalibrationBin:
        name = "calibration_bin"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "abs_gap",
            "bin_index",
            "count",
            "hi",
            "lo",
            "mean_accuracy",
            "mean_confidence",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        return cls(**{key: values[key] for key in known})


@dataclass(frozen=True, slots=True)
class OrderViolation:
    """One chronological violation on the machine or manual side."""

    attempt_id: str = ""
    side: str = "machine"
    detail: str = ""

    def __post_init__(self) -> None:
        name = "order_violation"
        object.__setattr__(
            self, "attempt_id", _check_attempt_id(name, self.attempt_id)
        )
        if self.side not in ("machine", "manual"):
            raise CheckpointEvaluationError(
                f"{name}: 'side' must be 'machine' or 'manual', "
                f"got {self.side!r}."
            )
        object.__setattr__(
            self, "detail", _check_non_blank(name, "'detail'", self.detail)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "detail": self.detail,
            "side": self.side,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> OrderViolation:
        name = "order_violation"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"attempt_id", "detail", "side"}
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        return cls(**{key: values[key] for key in known})


@dataclass(frozen=True, slots=True)
class GroupStructuralSummary:
    """Structural-status distribution for one attempt-label group."""

    attempt_label: str = LABEL_SERVE
    count: int = 0
    complete: int = 0
    partial: int = 0
    incomplete: int = 0
    unavailable: int = 0
    complete_rate: float | None = None

    def __post_init__(self) -> None:
        name = "group_structural_summary"
        if self.attempt_label not in ATTEMPT_LABELS:
            raise CheckpointEvaluationError(
                f"{name}: 'attempt_label' must be one of "
                f"{sorted(ATTEMPT_LABELS)}, got {self.attempt_label!r}."
            )
        for key in ("count", "complete", "partial", "incomplete", "unavailable"):
            object.__setattr__(
                self, key, _check_count(name, key, getattr(self, key))
            )
        if (
            self.complete + self.partial + self.incomplete + self.unavailable
        ) != self.count:
            raise CheckpointEvaluationError(
                f"{name}: status counts must sum to 'count'."
            )
        object.__setattr__(
            self,
            "complete_rate",
            _check_ratio_or_none(name, "complete_rate", self.complete_rate),
        )
        if (self.complete_rate is None) != (self.count == 0):
            raise CheckpointEvaluationError(
                f"{name}: 'complete_rate' must be None exactly when empty."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_label": self.attempt_label,
            "complete": self.complete,
            "complete_rate": self.complete_rate,
            "count": self.count,
            "incomplete": self.incomplete,
            "partial": self.partial,
            "unavailable": self.unavailable,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GroupStructuralSummary:
        name = "group_structural_summary"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempt_label",
            "complete",
            "complete_rate",
            "count",
            "incomplete",
            "partial",
            "unavailable",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        return cls(**{key: values[key] for key in known})


@dataclass(frozen=True, slots=True)
class StructuralSeparation:
    """Serve-versus-aborted structural separation (null when ungraded)."""

    serve: GroupStructuralSummary = field(
        default_factory=lambda: GroupStructuralSummary(
            attempt_label=LABEL_SERVE,
            count=0,
            complete=0,
            partial=0,
            incomplete=0,
            unavailable=0,
            complete_rate=None,
        )
    )
    aborted: GroupStructuralSummary = field(
        default_factory=lambda: GroupStructuralSummary(
            attempt_label=LABEL_ABORTED,
            count=0,
            complete=0,
            partial=0,
            incomplete=0,
            unavailable=0,
            complete_rate=None,
        )
    )

    def __post_init__(self) -> None:
        name = "structural_separation"
        if not isinstance(self.serve, GroupStructuralSummary):
            raise CheckpointEvaluationError(
                f"{name}: 'serve' must be a GroupStructuralSummary."
            )
        if not isinstance(self.aborted, GroupStructuralSummary):
            raise CheckpointEvaluationError(
                f"{name}: 'aborted' must be a GroupStructuralSummary."
            )
        if self.serve.attempt_label != LABEL_SERVE:
            raise CheckpointEvaluationError(
                f"{name}: 'serve' group must carry label 'serve'."
            )
        if self.aborted.attempt_label != LABEL_ABORTED:
            raise CheckpointEvaluationError(
                f"{name}: 'aborted' group must carry label 'aborted'."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "aborted": self.aborted.to_dict(),
            "serve": self.serve.to_dict(),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StructuralSeparation:
        name = "structural_separation"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"aborted", "serve"}
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        try:
            serve = GroupStructuralSummary.from_dict(values["serve"])
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid serve group: {exc}."
            ) from exc
        try:
            aborted = GroupStructuralSummary.from_dict(values["aborted"])
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid aborted group: {exc}."
            ) from exc
        return cls(serve=serve, aborted=aborted)


@dataclass(frozen=True, slots=True)
class CheckpointEvaluationReport:
    """Versioned deterministic report for one PhaseDocument evaluation."""

    schema_version: int = CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION
    dataset_split: str = SPLIT_DEV
    sessions: tuple[str, ...] = field(default_factory=tuple)
    n_attempts: int = 0
    per_stage: tuple[StageAggregate, ...] = field(default_factory=tuple)
    overall: StageAggregate = field(
        default_factory=lambda: StageAggregate(stage="overall")
    )
    n_calibrated: int = 0
    brier_score: float | None = None
    calibration_bins: tuple[CalibrationBin, ...] = field(default_factory=tuple)
    structural_complete: int = 0
    structural_partial: int = 0
    structural_incomplete: int = 0
    structural_unavailable: int = 0
    separation: StructuralSeparation | None = None
    order_violations: tuple[OrderViolation, ...] = field(default_factory=tuple)
    attempts: tuple[AttemptEvaluation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = "checkpoint_evaluation_report"
        version = self.schema_version
        if not _is_int(version):
            raise CheckpointEvaluationError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION:
            if version > CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION:
                raise CheckpointEvaluationError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version "
                    f"{CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION}."
                )
            raise CheckpointEvaluationError(
                f"{name}: unsupported schema_version {version!r}; expected "
                f"version {CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION}."
            )
        if self.dataset_split not in ALLOWED_SPLITS:
            raise CheckpointEvaluationError(
                f"{name}: 'dataset_split' must be one of "
                f"{sorted(ALLOWED_SPLITS)}, got {self.dataset_split!r}."
            )
        raw_sessions = self.sessions
        if not isinstance(raw_sessions, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'sessions' must be a list or tuple."
            )
        sessions = tuple(raw_sessions)
        for entry in sessions:
            _check_non_blank(name, "'sessions' entry", entry)
        if tuple(sorted(sessions)) != sessions:
            raise CheckpointEvaluationError(
                f"{name}: 'sessions' must be sorted, got {list(sessions)!r}."
            )
        if len(set(sessions)) != len(sessions):
            raise CheckpointEvaluationError(
                f"{name}: 'sessions' must not contain duplicates."
            )
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(
            self, "n_attempts", _check_count(name, "n_attempts", self.n_attempts)
        )
        raw_stages = self.per_stage
        if not isinstance(raw_stages, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'per_stage' must be a list or tuple."
            )
        stages = tuple(raw_stages)
        for entry in stages:
            if not isinstance(entry, StageAggregate):
                raise CheckpointEvaluationError(
                    f"{name}: every per-stage entry must be a StageAggregate."
                )
        if [entry.stage for entry in stages] != list(STAGE_ORDER):
            raise CheckpointEvaluationError(
                f"{name}: 'per_stage' must cover {list(STAGE_ORDER)} in "
                f"order, got {[entry.stage for entry in stages]!r}."
            )
        object.__setattr__(self, "per_stage", stages)
        if not isinstance(self.overall, StageAggregate):
            raise CheckpointEvaluationError(
                f"{name}: 'overall' must be a StageAggregate."
            )
        if self.overall.stage != "overall":
            raise CheckpointEvaluationError(
                f"{name}: 'overall' aggregate must use stage 'overall'."
            )
        object.__setattr__(
            self, "n_calibrated", _check_count(name, "n_calibrated", self.n_calibrated)
        )
        if self.brier_score is None:
            if self.n_calibrated != 0:
                raise CheckpointEvaluationError(
                    f"{name}: 'brier_score' must be set when calibrated."
                )
        else:
            if not _is_number(self.brier_score) or not math.isfinite(
                self.brier_score
            ):
                raise CheckpointEvaluationError(
                    f"{name}: 'brier_score' must be finite or None."
                )
            number = float(self.brier_score)
            if number < 0.0 or number > 1.0:
                raise CheckpointEvaluationError(
                    f"{name}: 'brier_score' must lie in [0, 1]."
                )
            object.__setattr__(self, "brier_score", number)
            if self.n_calibrated == 0:
                raise CheckpointEvaluationError(
                    f"{name}: 'brier_score' must be None when nothing was "
                    f"calibrated."
                )
        raw_bins = self.calibration_bins
        if not isinstance(raw_bins, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'calibration_bins' must be a list or tuple."
            )
        bins = tuple(raw_bins)
        if len(bins) != len(CALIBRATION_BIN_EDGES) - 1:
            raise CheckpointEvaluationError(
                f"{name}: 'calibration_bins' must have "
                f"{len(CALIBRATION_BIN_EDGES) - 1} bins, got {len(bins)}."
            )
        for index, entry in enumerate(bins):
            if not isinstance(entry, CalibrationBin):
                raise CheckpointEvaluationError(
                    f"{name}: every bin must be a CalibrationBin."
                )
            if entry.bin_index != index:
                raise CheckpointEvaluationError(
                    f"{name}: bin order must be 0..N-1, got "
                    f"{entry.bin_index!r} at position {index}."
                )
        object.__setattr__(self, "calibration_bins", bins)
        for key in (
            "structural_complete",
            "structural_partial",
            "structural_incomplete",
            "structural_unavailable",
        ):
            object.__setattr__(
                self, key, _check_count(name, key, getattr(self, key))
            )
        structural_total = (
            self.structural_complete
            + self.structural_partial
            + self.structural_incomplete
            + self.structural_unavailable
        )
        if structural_total != self.n_attempts:
            raise CheckpointEvaluationError(
                f"{name}: structural counts ({structural_total}) must sum to "
                f"'n_attempts' ({self.n_attempts})."
            )
        if self.separation is not None and not isinstance(
            self.separation, StructuralSeparation
        ):
            raise CheckpointEvaluationError(
                f"{name}: 'separation' must be a StructuralSeparation or None."
            )
        raw_violations = self.order_violations
        if not isinstance(raw_violations, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'order_violations' must be a list or tuple."
            )
        violations = tuple(raw_violations)
        for entry in violations:
            if not isinstance(entry, OrderViolation):
                raise CheckpointEvaluationError(
                    f"{name}: every violation must be an OrderViolation."
                )
        object.__setattr__(self, "order_violations", violations)
        raw_attempts = self.attempts
        if not isinstance(raw_attempts, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'attempts' must be a list or tuple."
            )
        attempts = tuple(raw_attempts)
        for entry in attempts:
            if not isinstance(entry, AttemptEvaluation):
                raise CheckpointEvaluationError(
                    f"{name}: every attempt must be an AttemptEvaluation."
                )
        if [entry.attempt_id for entry in attempts] != sorted(
            entry.attempt_id for entry in attempts
        ):
            raise CheckpointEvaluationError(
                f"{name}: 'attempts' must be sorted by attempt_id."
            )
        if len(attempts) != self.n_attempts:
            raise CheckpointEvaluationError(
                f"{name}: 'attempts' length must equal 'n_attempts'."
            )
        object.__setattr__(self, "attempts", attempts)

    def stage(self, key: str) -> StageAggregate:
        """Return the per-stage aggregate for canonical stage ``key``."""
        for entry in self.per_stage:
            if entry.stage == key:
                return entry
        raise KeyError(f"checkpoint_evaluation_report: unknown stage {key!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": [entry.to_dict() for entry in self.attempts],
            "brier_score": self.brier_score,
            "calibration_bins": [entry.to_dict() for entry in self.calibration_bins],
            "dataset_split": self.dataset_split,
            "n_attempts": self.n_attempts,
            "n_calibrated": self.n_calibrated,
            "order_violations": [entry.to_dict() for entry in self.order_violations],
            "overall": self.overall.to_dict(),
            "per_stage": [entry.to_dict() for entry in self.per_stage],
            "schema_version": self.schema_version,
            "separation": (
                self.separation.to_dict() if self.separation is not None else None
            ),
            "sessions": list(self.sessions),
            "structural_complete": self.structural_complete,
            "structural_incomplete": self.structural_incomplete,
            "structural_partial": self.structural_partial,
            "structural_unavailable": self.structural_unavailable,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CheckpointEvaluationReport:
        name = "checkpoint_evaluation_report"
        if not isinstance(values, dict):
            raise CheckpointEvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempts",
            "brier_score",
            "calibration_bins",
            "dataset_split",
            "n_attempts",
            "n_calibrated",
            "order_violations",
            "overall",
            "per_stage",
            "schema_version",
            "separation",
            "sessions",
            "structural_complete",
            "structural_incomplete",
            "structural_partial",
            "structural_unavailable",
        }
        missing = sorted(known - set(values))
        if missing:
            raise CheckpointEvaluationError(
                f"{name}: missing required keys {missing!r}."
            )
        _check_no_unknown_keys(name, values, known)
        _check_schema_version(
            name, values, CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION
        )
        if values["dataset_split"] not in ALLOWED_SPLITS:
            raise CheckpointEvaluationError(
                f"{name}: 'dataset_split' must be one of "
                f"{sorted(ALLOWED_SPLITS)}, got {values['dataset_split']!r}."
            )
        try:
            per_stage = tuple(
                StageAggregate.from_dict(entry) for entry in values["per_stage"]
            )
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid per_stage: {exc}."
            ) from exc
        try:
            overall = StageAggregate.from_dict(values["overall"])
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid overall: {exc}."
            ) from exc
        raw_bins = values["calibration_bins"]
        if not isinstance(raw_bins, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'calibration_bins' must be a list."
            )
        try:
            bins = tuple(CalibrationBin.from_dict(entry) for entry in raw_bins)
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid calibration bin: {exc}."
            ) from exc
        raw_violations = values["order_violations"]
        if not isinstance(raw_violations, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'order_violations' must be a list."
            )
        try:
            violations = tuple(
                OrderViolation.from_dict(entry) for entry in raw_violations
            )
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid order violation: {exc}."
            ) from exc
        raw_attempts = values["attempts"]
        if not isinstance(raw_attempts, (list, tuple)):
            raise CheckpointEvaluationError(
                f"{name}: 'attempts' must be a list."
            )
        try:
            attempts = tuple(
                AttemptEvaluation.from_dict(entry) for entry in raw_attempts
            )
        except CheckpointEvaluationError as exc:
            raise CheckpointEvaluationError(
                f"{name}: invalid attempt: {exc}."
            ) from exc
        raw_separation = values["separation"]
        separation: StructuralSeparation | None
        if raw_separation is None:
            separation = None
        elif isinstance(raw_separation, dict):
            try:
                separation = StructuralSeparation.from_dict(raw_separation)
            except CheckpointEvaluationError as exc:
                raise CheckpointEvaluationError(
                    f"{name}: invalid separation: {exc}."
                ) from exc
        else:
            raise CheckpointEvaluationError(
                f"{name}: 'separation' must be an object or None, got "
                f"{type(raw_separation).__name__}."
            )
        return cls(
            schema_version=values["schema_version"],
            dataset_split=values["dataset_split"],
            sessions=tuple(values["sessions"]),
            n_attempts=values["n_attempts"],
            per_stage=per_stage,
            overall=overall,
            n_calibrated=values["n_calibrated"],
            brier_score=values["brier_score"],
            calibration_bins=bins,
            structural_complete=values["structural_complete"],
            structural_partial=values["structural_partial"],
            structural_incomplete=values["structural_incomplete"],
            structural_unavailable=values["structural_unavailable"],
            separation=separation,
            order_violations=violations,
            attempts=attempts,
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> CheckpointEvaluationReport:
        return cls.from_dict(
            _loads_object("checkpoint_evaluation_report", data)
        )


def _build_stage_aggregate(
    stage: str,
    *,
    n_available: int,
    n_unavailable: int,
    n_ambiguous: int,
    n_predicted_available: int,
    n_predicted_unavailable: int,
    n_both_available: int,
    n_missing: int,
    n_extra: int,
    accepted: int,
    evaluated_keyframe: int,
    overlaps: Sequence[tuple[float, float]],
    timing_signed: Sequence[float],
    timing_abs_sorted: Sequence[float],
) -> StageAggregate:
    if evaluated_keyframe:
        rate: float | None = accepted / evaluated_keyframe
    else:
        rate = None
    if overlaps:
        mean_iou = math.fsum(item[0] for item in overlaps) / len(overlaps)
        mean_inter = math.fsum(item[1] for item in overlaps) / len(overlaps)
    else:
        mean_iou = None
        mean_inter = None
    if timing_signed:
        mean_signed: float | None = math.fsum(timing_signed) / len(timing_signed)
        mean_abs: float | None = math.fsum(timing_abs_sorted) / len(
            timing_abs_sorted
        )
        ordered = sorted(timing_abs_sorted)
        median_abs: float | None = median_of_sorted(ordered)
        p90_abs: float | None = nearest_rank_percentile(
            ordered, HIGH_PERCENTILE_Q
        )
    else:
        mean_signed = None
        mean_abs = None
        median_abs = None
        p90_abs = None
    return StageAggregate(
        stage=stage,
        n_annotated_available=n_available,
        n_annotated_unavailable=n_unavailable,
        n_annotated_ambiguous=n_ambiguous,
        n_predicted_available=n_predicted_available,
        n_predicted_unavailable=n_predicted_unavailable,
        n_both_available=n_both_available,
        n_machine_missing_manual_available=n_missing,
        n_machine_available_manual_not_available=n_extra,
        n_keyframe_evaluated=evaluated_keyframe,
        n_keyframe_accepted=accepted,
        accepted_keyframe_rate=rate,
        n_overlap_evaluated=len(overlaps),
        mean_iou=mean_iou,
        mean_intersection_seconds=mean_inter,
        n_timing_evaluated=len(timing_signed),
        mean_signed_error_seconds=mean_signed,
        mean_abs_error_seconds=mean_abs,
        median_abs_error_seconds=median_abs,
        p90_abs_error_seconds=p90_abs,
    )


def evaluate_phase_document(
    document: PhaseDocument,
    manifest: PhaseAnnotationManifest,
) -> CheckpointEvaluationReport:
    """Evaluate one :class:`PhaseDocument` against one validated manifest.

    The manifest attempt-id set must exactly equal the document
    attempt-id set, and every manifest attempt range must exactly equal
    the document's unpadded ``attempt_range``. The input document is
    never modified. Reports are deterministic and JSON-friendly.
    """
    name = "evaluate_phase_document"
    if not isinstance(document, PhaseDocument):
        raise CheckpointEvaluationError(
            f"{name}: 'document' must be a PhaseDocument, "
            f"got {type(document).__name__}."
        )
    if not isinstance(manifest, PhaseAnnotationManifest):
        raise CheckpointEvaluationError(
            f"{name}: 'manifest' must be a PhaseAnnotationManifest, "
            f"got {type(manifest).__name__}."
        )
    if len(manifest) == 0:
        raise CheckpointEvaluationError(
            f"{name}: manifest contains no annotations to evaluate."
        )
    doc_by_id: dict[str, AttemptPhase] = {}
    for attempt in document.attempts:
        if not isinstance(attempt, AttemptPhase):
            raise CheckpointEvaluationError(
                f"{name}: document attempts must be AttemptPhase values."
            )
        doc_by_id[attempt.attempt_id] = attempt
    manifest_ids = set(manifest.attempt_ids)
    doc_ids = set(doc_by_id)
    missing = sorted(manifest_ids - doc_ids)
    extra = sorted(doc_ids - manifest_ids)
    if missing or extra:
        raise CheckpointEvaluationError(
            f"{name}: attempt mismatch: manifest attempts missing from the "
            f"phase document: {missing!r}; document attempts absent from "
            f"the manifest: {extra!r}. Match attempts deterministically by "
            f"identity before scoring."
        )
    for attempt_id in sorted(manifest_ids):
        rows = manifest.rows_for_attempt(attempt_id)
        first = rows[0]
        machine = doc_by_id[attempt_id]
        if (
            first.attempt_start_seconds != machine.attempt_range.start_seconds
            or first.attempt_end_seconds != machine.attempt_range.end_seconds
        ):
            raise CheckpointEvaluationError(
                f"{name}: range linkage mismatch for attempt "
                f"{attempt_id!r}: manifest "
                f"[{first.attempt_start_seconds!r}, "
                f"{first.attempt_end_seconds!r}) != document "
                f"[{machine.attempt_range.start_seconds!r}, "
                f"{machine.attempt_range.end_seconds!r})."
            )

    # Per-attempt detail plus pooled accumulators.
    attempt_evaluations: list[AttemptEvaluation] = []
    per_stage_pairs: dict[str, list[StagePairDetail]] = {
        key: [] for key in STAGE_ORDER
    }
    calibration_pairs: list[tuple[float, float]] = []  # (confidence, correctness)
    violations: list[OrderViolation] = []
    structural_counts = {
        "complete": 0,
        "partial": 0,
        "incomplete": 0,
        "unavailable": 0,
    }
    label_status: dict[str, dict[str, int]] = {}

    for attempt_id in sorted(manifest_ids):
        stage_keys = sorted(
            entry.stage for entry in manifest.rows_for_attempt(attempt_id)
        )
        if stage_keys != sorted(STAGE_ORDER):
            raise CheckpointEvaluationError(
                f"{name}: attempt {attempt_id!r} must annotate exactly "
                f"the eight stage keys {list(STAGE_ORDER)}, got "
                f"{stage_keys!r}."
            )
        rows = {entry.stage: entry for entry in manifest.rows_for_attempt(attempt_id)}
        machine = doc_by_id[attempt_id]
        first_row = manifest.rows_for_attempt(attempt_id)[0]
        structural_counts[machine.structural_status] += 1
        label = first_row.attempt_label
        if label is not None:
            bucket = label_status.setdefault(
                label, {"complete": 0, "partial": 0, "incomplete": 0,
                        "unavailable": 0, "count": 0}
            )
            bucket[machine.structural_status] += 1
            bucket["count"] += 1
        details: list[StagePairDetail] = []
        machine_intervals: list[tuple[str, float, float]] = []
        machine_keyframes: list[tuple[str, float]] = []
        manual_intervals: list[tuple[str, float, float]] = []
        manual_keyframes: list[tuple[str, float]] = []
        for key in STAGE_ORDER:
            row = rows[key]
            stage = machine.stage(key)
            machine_available = stage.availability in ("available", "partial")
            if machine_available:
                assert stage.interval is not None
                machine_interval = (
                    float(stage.interval.start_seconds),
                    float(stage.interval.end_seconds),
                )
                machine_intervals.append((key, *machine_interval))
                machine_keyframe: float | None = (
                    float(stage.keyframe_seconds)
                    if stage.keyframe_seconds is not None
                    else None
                )
                if machine_keyframe is not None:
                    machine_keyframes.append((key, machine_keyframe))
            else:
                machine_interval = (None, None)  # type: ignore[assignment]
                machine_keyframe = None
                if (
                    stage.interval is not None
                    or stage.keyframe_seconds is not None
                ):
                    raise CheckpointEvaluationError(
                        f"{name}: internal machine state for "
                        f"{attempt_id!r}/{key!r} is inconsistent."
                    )
            accepted: bool | None = None
            iou_value: float | None = None
            inter_value: float | None = None
            signed: float | None = None
            abs_err: float | None = None
            if row.status == STATUS_AVAILABLE and machine_available:
                assert row.interval_start_seconds is not None
                assert row.interval_end_seconds is not None
                assert machine_interval[0] is not None
                inter_value = interval_intersection_seconds(
                    float(row.interval_start_seconds),
                    float(row.interval_end_seconds),
                    float(machine_interval[0]),
                    float(machine_interval[1]),
                )
                iou_value = interval_iou(
                    float(row.interval_start_seconds),
                    float(row.interval_end_seconds),
                    float(machine_interval[0]),
                    float(machine_interval[1]),
                )
                if machine_keyframe is not None:
                    accepted = bool(
                        float(row.interval_start_seconds)
                        <= machine_keyframe
                        < float(row.interval_end_seconds)
                    )
                    calibration_pairs.append(
                        (float(stage.confidence), 1.0 if accepted else 0.0)
                    )
                    if row.manual_keyframe_seconds is not None:
                        signed = machine_keyframe - float(
                            row.manual_keyframe_seconds
                        )
                        abs_err = abs(signed)
                manual_intervals.append(
                    (key, float(row.interval_start_seconds),
                     float(row.interval_end_seconds))
                )
                if row.manual_keyframe_seconds is not None:
                    manual_keyframes.append(
                        (key, float(row.manual_keyframe_seconds))
                    )
            elif row.status == STATUS_AVAILABLE and not machine_available:
                manual_intervals.append(
                    (key, float(row.interval_start_seconds),
                     float(row.interval_end_seconds))
                )
                if row.manual_keyframe_seconds is not None:
                    manual_keyframes.append(
                        (key, float(row.manual_keyframe_seconds))
                    )
            else:
                # Unavailable/ambiguous manual rows: honestly null pair
                # metrics; never scored as visual claims.
                pass
            detail = StagePairDetail(
                stage=key,
                manual_status=row.status,
                manual_interval_start_seconds=row.interval_start_seconds,
                manual_interval_end_seconds=row.interval_end_seconds,
                manual_keyframe_seconds=row.manual_keyframe_seconds,
                machine_availability=stage.availability,
                machine_interval_start_seconds=machine_interval[0],
                machine_interval_end_seconds=machine_interval[1],
                machine_keyframe_seconds=machine_keyframe,
                machine_confidence=float(stage.confidence),
                accepted=accepted,
                iou=iou_value,
                intersection_seconds=inter_value,
                signed_error_seconds=signed,
                abs_error_seconds=abs_err,
            )
            details.append(detail)
            per_stage_pairs[key].append(detail)
        # Order checks over available sequences only; skips never violate.
        for side, intervals, keyframes in (
            ("machine", machine_intervals, machine_keyframes),
            ("manual", manual_intervals, manual_keyframes),
        ):
            previous_end: float | None = None
            previous_key: str | None = None
            for key, start, end in intervals:
                if previous_end is not None and start < previous_end:
                    assert previous_key is not None
                    violations.append(
                        OrderViolation(
                            attempt_id=attempt_id,
                            side=side,  # type: ignore[arg-type]
                            detail=(
                                f"{side} stage {key!r} interval starts at "
                                f"{start!r} before previous stage "
                                f"{previous_key!r} ends at {previous_end!r}."
                            ),
                        )
                    )
                    break
                previous_end = end
                previous_key = key
            previous_kf: float | None = None
            for key, value in keyframes:
                if previous_kf is not None and value < previous_kf:
                    violations.append(
                        OrderViolation(
                            attempt_id=attempt_id,
                            side=side,  # type: ignore[arg-type]
                            detail=(
                                f"{side} stage {key!r} keyframe ({value!r}) "
                                f"precedes earlier keyframe "
                                f"({previous_kf!r})."
                            ),
                        )
                    )
                    break
                previous_kf = value
        attempt_evaluations.append(
            AttemptEvaluation(
                attempt_id=attempt_id,
                session_id=first_row.session_id,
                media=first_row.media,
                attempt_start_seconds=first_row.attempt_start_seconds,
                attempt_end_seconds=first_row.attempt_end_seconds,
                attempt_label=first_row.attempt_label,
                structural_status=machine.structural_status,
                stages=tuple(details),
            )
        )

    violations = sorted(
        violations, key=lambda entry: (entry.attempt_id, entry.side, entry.detail)
    )

    def _aggregate(stage: str, pairs: Sequence[StagePairDetail]) -> StageAggregate:
        n_available = sum(1 for item in pairs if item.manual_status == STATUS_AVAILABLE)
        n_unavailable = sum(
            1 for item in pairs if item.manual_status == STATUS_UNAVAILABLE
        )
        n_ambiguous = sum(
            1 for item in pairs if item.manual_status == STATUS_AMBIGUOUS
        )
        n_pred_available = sum(
            1 for item in pairs if item.machine_availability != "unavailable"
        )
        n_pred_unavailable = len(pairs) - n_pred_available
        n_both = sum(
            1
            for item in pairs
            if item.manual_status == STATUS_AVAILABLE
            and item.machine_availability != "unavailable"
        )
        n_missing = sum(
            1
            for item in pairs
            if item.manual_status == STATUS_AVAILABLE
            and item.machine_availability == "unavailable"
        )
        n_extra = sum(
            1
            for item in pairs
            if item.manual_status != STATUS_AVAILABLE
            and item.machine_availability != "unavailable"
        )
        accepted_count = sum(1 for item in pairs if item.accepted is True)
        evaluated_count = sum(1 for item in pairs if item.accepted is not None)
        overlaps = [
            (float(item.iou), float(item.intersection_seconds))
            for item in pairs
            if item.iou is not None and item.intersection_seconds is not None
        ]
        signed_errors = [
            float(item.signed_error_seconds)
            for item in pairs
            if item.signed_error_seconds is not None
        ]
        abs_errors = [
            float(item.abs_error_seconds)
            for item in pairs
            if item.abs_error_seconds is not None
        ]
        return _build_stage_aggregate(
            stage,
            n_available=n_available,
            n_unavailable=n_unavailable,
            n_ambiguous=n_ambiguous,
            n_predicted_available=n_pred_available,
            n_predicted_unavailable=n_pred_unavailable,
            n_both_available=n_both,
            n_missing=n_missing,
            n_extra=n_extra,
            accepted=accepted_count,
            evaluated_keyframe=evaluated_count,
            overlaps=overlaps,
            timing_signed=signed_errors,
            timing_abs_sorted=abs_errors,
        )

    per_stage = tuple(
        _aggregate(key, per_stage_pairs[key]) for key in STAGE_ORDER
    )
    pooled = [item for key in STAGE_ORDER for item in per_stage_pairs[key]]
    overall = _aggregate("overall", pooled)

    # Calibration over accepted-keyframe evaluable pairs (deterministic bins).
    n_calibrated = len(calibration_pairs)
    if n_calibrated:
        ordered_pairs = sorted(calibration_pairs, key=lambda item: (item[0], item[1]))
        brier: float | None = math.fsum(
            (conf - correct) ** 2 for conf, correct in ordered_pairs
        ) / n_calibrated
    else:
        ordered_pairs = []
        brier = None
    bins: list[CalibrationBin] = []
    width = 1.0 / (len(CALIBRATION_BIN_EDGES) - 1)
    for index in range(len(CALIBRATION_BIN_EDGES) - 1):
        lo = CALIBRATION_BIN_EDGES[index]
        hi = CALIBRATION_BIN_EDGES[index + 1]
        in_bin = [
            pair
            for pair in ordered_pairs
            if (min(int(pair[0] * 5), 4) == index)
        ]
        _ = width
        if in_bin:
            mean_conf = math.fsum(pair[0] for pair in in_bin) / len(in_bin)
            mean_acc = math.fsum(pair[1] for pair in in_bin) / len(in_bin)
            gap = abs(mean_conf - mean_acc)
            bins.append(
                CalibrationBin(
                    bin_index=index,
                    lo=lo,
                    hi=hi,
                    count=len(in_bin),
                    mean_confidence=mean_conf,
                    mean_accuracy=mean_acc,
                    abs_gap=gap,
                )
            )
        else:
            bins.append(
                CalibrationBin(
                    bin_index=index,
                    lo=lo,
                    hi=hi,
                    count=0,
                    mean_confidence=None,
                    mean_accuracy=None,
                    abs_gap=None,
                )
            )

    if label_status:
        def _group(label: str) -> GroupStructuralSummary:
            bucket = label_status.get(
                label,
                {"complete": 0, "partial": 0, "incomplete": 0,
                 "unavailable": 0, "count": 0},
            )
            count = bucket["count"]
            return GroupStructuralSummary(
                attempt_label=label,  # type: ignore[arg-type]
                count=count,
                complete=bucket["complete"],
                partial=bucket["partial"],
                incomplete=bucket["incomplete"],
                unavailable=bucket["unavailable"],
                complete_rate=(bucket["complete"] / count if count else None),
            )

        separation: StructuralSeparation | None = StructuralSeparation(
            serve=_group(LABEL_SERVE), aborted=_group(LABEL_ABORTED)
        )
    else:
        separation = None

    return CheckpointEvaluationReport(
        schema_version=CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION,
        dataset_split=manifest.dataset_split,
        sessions=manifest.sessions,
        n_attempts=len(attempt_evaluations),
        per_stage=per_stage,
        overall=overall,
        n_calibrated=n_calibrated,
        brier_score=brier,
        calibration_bins=tuple(bins),
        structural_complete=structural_counts["complete"],
        structural_partial=structural_counts["partial"],
        structural_incomplete=structural_counts["incomplete"],
        structural_unavailable=structural_counts["unavailable"],
        separation=separation,
        order_violations=tuple(violations),
        attempts=tuple(attempt_evaluations),
    )
