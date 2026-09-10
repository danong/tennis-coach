"""Deterministic serve-attempt evaluation: manifests, matching, reports.

M3.4. This module is intentionally pure: no file I/O, no subprocesses, no
framework objects, no detector thresholds. It validates session-disjoint
annotation manifests and computes deterministic range metrics over
source-time intervals.

Manifest model
--------------

An :class:`AnnotationManifest` carries one dataset split (``"dev"`` or
``"heldout"``) and a list of :class:`Annotation` values. Each annotation
references a media file by a *relative* local path, names a recording
session, and marks a half-open ``[start_seconds, end_seconds)`` source-time
range with a label:

- ``"serve"``: a positive ground-truth attempt.
- ``"ambiguous"``: an excluded region (e.g. a borderline motion). It never
  counts as ground truth, and an unmatched prediction overlapping one with
  IoU ``>=`` threshold is excused from the false-positive count.

Session discipline: development and held-out manifests must not share a
``session_id``. Use :func:`validate_session_disjoint` to enforce this;
tuning against held-out labels is out of scope for this module.

Matching and metrics
--------------------

Predicted ranges (for example planned attempt ``detected_range`` values)
are matched one-to-one against positive ground-truth ranges with greedy
IoU matching at :data:`IOU_THRESHOLD` (0.5, per ``docs/design.md`` §8).
All candidate pairs with IoU ``>=`` threshold are considered in order of
decreasing IoU; ties break by lower truth index, then lower prediction
index. Each truth and each prediction participates in at most one match.
Matching is performed independently within each stratum (recording
session), never across sessions.

Reported scalars (see :class:`RangeMetrics` / :class:`OverallMetrics`):

- ``precision``: ``TP / (TP + FP)``; ``None`` when no scored predictions.
- ``recall``: ``TP / num_truth``; ``None`` when no ground truth.
- ``mean_abs_onset_error_seconds`` / ``mean_abs_end_error_seconds``: mean
  absolute ``pred.start - truth.start`` / ``pred.end - truth.end`` over
  matches; ``None`` when there are no matches. Signed means are reported
  alongside as bias diagnostics.
- ``retained_duration_ratio``: matched intersection seconds over total
  ground-truth seconds; ``None`` when there is no ground truth.
- ``extra_seconds_per_attempt``: non-overlapping predicted seconds
  averaged over scored (non-excused) predictions; ``0.0`` when there are
  none. Excused ambiguous predictions contribute neither duration nor
  denominator.

All aggregation uses :func:`math.fsum` over deterministically ordered
inputs, so reports are bit-for-bit reproducible for identical inputs.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from serve_review.domain import MediaRange

__all__ = [
    "ANNOTATION_MANIFEST_SCHEMA_VERSION",
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "IOU_THRESHOLD",
    "LABEL_SERVE",
    "LABEL_AMBIGUOUS",
    "ALLOWED_LABELS",
    "SPLIT_DEV",
    "SPLIT_HELDOUT",
    "ALLOWED_SPLITS",
    "EvaluationError",
    "Annotation",
    "AnnotationManifest",
    "MatchEntry",
    "RangeMetrics",
    "OverallMetrics",
    "StratumInput",
    "StratumReport",
    "EvaluationReport",
    "intersection_seconds",
    "iou",
    "match_ranges",
    "evaluate_ranges",
    "evaluate_report",
    "evaluate_manifest",
    "group_truth_by_session",
    "group_ambiguous_by_session",
    "validate_session_disjoint",
]

#: Version of the annotation manifest schema.
ANNOTATION_MANIFEST_SCHEMA_VERSION = 1

#: Version of the machine-readable evaluation report schema.
EVALUATION_REPORT_SCHEMA_VERSION = 1

#: Fixed one-to-one matching threshold (design §8). Not tunable here.
IOU_THRESHOLD = 0.5

#: Positive ground-truth label.
LABEL_SERVE = "serve"

#: Excluded-region label: never ground truth; may excuse predictions.
LABEL_AMBIGUOUS = "ambiguous"

#: Labels accepted on annotations.
ALLOWED_LABELS = frozenset({LABEL_SERVE, LABEL_AMBIGUOUS})

#: Development split name.
SPLIT_DEV = "dev"

#: Held-out split name.
SPLIT_HELDOUT = "heldout"

#: Splits accepted on manifests.
ALLOWED_SPLITS = frozenset({SPLIT_DEV, SPLIT_HELDOUT})

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


class EvaluationError(ValueError):
    """Raised when evaluation input or a persisted document is invalid."""


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
            raise EvaluationError(f"{name}: invalid UTF-8 JSON payload.") from exc
    if not isinstance(data, str):
        raise EvaluationError(
            f"{name}: JSON payload must be str or bytes, "
            f"got {type(data).__name__}."
        )
    try:
        decoded = json.loads(data)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{name}: invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise EvaluationError(
            f"{name}: JSON object is required, got {type(decoded).__name__}."
        )
    return decoded


def _check_schema_version(name: str, values: Mapping[str, Any], current: int) -> None:
    if "schema_version" not in values:
        raise EvaluationError(f"{name}: missing required key 'schema_version'.")
    version = values["schema_version"]
    if not _is_int(version):
        raise EvaluationError(
            f"{name}: 'schema_version' must be an integer, got {version!r}."
        )
    if version != current:
        if version > current:
            raise EvaluationError(
                f"{name}: unsupported newer schema_version {version!r}; "
                f"this build supports version {current}."
            )
        raise EvaluationError(
            f"{name}: unsupported schema_version {version!r}; "
            f"expected version {current}."
        )


def _check_no_unknown_keys(
    name: str, values: Mapping[str, Any], known: set[str]
) -> None:
    unknown = sorted(set(values) - known)
    if unknown:
        raise EvaluationError(f"{name}: unknown keys {unknown!r}.")


def _check_non_blank(name: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(
            f"{name}: {key!r} must be a non-blank string, got {value!r}."
        )
    return value


def _check_time_bound(name: str, key: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value) or value < 0:
        raise EvaluationError(
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
        raise EvaluationError(
            f"{name}: 'media' must be a relative local path, got {value!r}."
        )
    if ".." in PurePosixPath(text).parts:
        raise EvaluationError(
            f"{name}: 'media' must not escape its directory with '..', "
            f"got {value!r}."
        )
    if "\\" in text:
        raise EvaluationError(
            f"{name}: 'media' must use POSIX-style relative paths, "
            f"got {value!r}."
        )
    return text


def _check_label(name: str, value: Any) -> str:
    if value not in ALLOWED_LABELS:
        raise EvaluationError(
            f"{name}: 'label' must be one of {sorted(ALLOWED_LABELS)}, "
            f"got {value!r}."
        )
    return value


def _check_threshold(name: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise EvaluationError(
            f"{name}: IoU threshold must be a finite number, got {value!r}."
        )
    threshold = float(value)
    if not 0.0 < threshold <= 1.0:
        raise EvaluationError(
            f"{name}: IoU threshold must lie in (0, 1], got {value!r}."
        )
    return threshold


def _check_range_sequence(
    name: str, key: str, values: Any
) -> tuple[MediaRange, ...]:
    if isinstance(values, MediaRange) or not isinstance(values, (list, tuple)):
        raise EvaluationError(
            f"{name}: {key!r} must be a list or tuple of MediaRange, "
            f"got {type(values).__name__}."
        )
    items = tuple(values)
    for entry in items:
        if not isinstance(entry, MediaRange):
            raise EvaluationError(
                f"{name}: every entry of {key!r} must be a MediaRange, "
                f"got {type(entry).__name__}."
            )
    return items


def _check_count(name: str, key: str, value: Any) -> int:
    if not _is_int(value) or value < 0:
        raise EvaluationError(
            f"{name}: {key!r} must be an integer >= 0, got {value!r}."
        )
    return value


def _check_ratio_or_none(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(value):
        raise EvaluationError(
            f"{name}: {key!r} must be a finite number in [0, 1] or None, "
            f"got {value!r}."
        )
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise EvaluationError(
            f"{name}: {key!r} must lie in [0, 1] or be None, got {value!r}."
        )
    return number


def _check_error_or_none(name: str, key: str, value: Any) -> float | None:
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(value):
        raise EvaluationError(
            f"{name}: {key!r} must be a finite number or None, "
            f"got {value!r}."
        )
    return float(value)


def _check_extra_seconds(name: str, value: Any) -> float:
    if not _is_number(value) or not math.isfinite(value) or value < 0:
        raise EvaluationError(
            f"{name}: 'extra_seconds_per_attempt' must be a finite number "
            f">= 0, got {value!r}."
        )
    return float(value)


@dataclass(frozen=True, slots=True)
class Annotation:
    """One labeled source-time interval within a recording session."""

    session_id: str = ""
    media: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    label: str = LABEL_SERVE

    def __post_init__(self) -> None:
        name = "annotation"
        object.__setattr__(
            self, "session_id", _check_non_blank(name, "'session_id'", self.session_id)
        )
        object.__setattr__(
            self, "media", _check_relative_media(name, self.media)
        )
        start = _check_time_bound(name, "'start_seconds'", self.start_seconds)
        end = _check_time_bound(name, "'end_seconds'", self.end_seconds)
        if not start < end:
            raise EvaluationError(
                f"{name}: 'start_seconds' ({start!r}) must be less than "
                f"'end_seconds' ({end!r}) for a half-open range."
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "label", _check_label(name, self.label))

    def to_range(self) -> MediaRange:
        """Return this annotation's interval as a :class:`MediaRange`."""
        return MediaRange(
            start_seconds=self.start_seconds, end_seconds=self.end_seconds
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_seconds": self.end_seconds,
            "label": self.label,
            "media": self.media,
            "session_id": self.session_id,
            "start_seconds": self.start_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Annotation:
        name = "annotation"
        if not isinstance(values, dict):
            raise EvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"end_seconds", "label", "media", "session_id", "start_seconds"}
        missing = sorted(known - set(values))
        if missing:
            raise EvaluationError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        return cls(
            session_id=values["session_id"],
            media=values["media"],
            start_seconds=values["start_seconds"],
            end_seconds=values["end_seconds"],
            label=values["label"],
        )


@dataclass(frozen=True, slots=True)
class AnnotationManifest:
    """Versioned set of annotations for one dataset split."""

    schema_version: int = ANNOTATION_MANIFEST_SCHEMA_VERSION
    dataset_split: str = SPLIT_DEV
    annotations: tuple[Annotation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = "annotation_manifest"
        version = self.schema_version
        if not _is_int(version):
            raise EvaluationError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != ANNOTATION_MANIFEST_SCHEMA_VERSION:
            if version > ANNOTATION_MANIFEST_SCHEMA_VERSION:
                raise EvaluationError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version "
                    f"{ANNOTATION_MANIFEST_SCHEMA_VERSION}."
                )
            raise EvaluationError(
                f"{name}: unsupported schema_version {version!r}; expected "
                f"version {ANNOTATION_MANIFEST_SCHEMA_VERSION}."
            )
        if self.dataset_split not in ALLOWED_SPLITS:
            raise EvaluationError(
                f"{name}: 'dataset_split' must be one of "
                f"{sorted(ALLOWED_SPLITS)}, got {self.dataset_split!r}."
            )
        raw = self.annotations
        if isinstance(raw, Annotation) or not isinstance(raw, (list, tuple)):
            raise EvaluationError(
                f"{name}: 'annotations' must be a list or tuple of "
                f"Annotation, got {type(raw).__name__}."
            )
        items = tuple(raw)
        for entry in items:
            if not isinstance(entry, Annotation):
                raise EvaluationError(
                    f"{name}: every annotation must be an Annotation, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "annotations", items)

    def __len__(self) -> int:
        return len(self.annotations)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.annotations)

    @property
    def sessions(self) -> tuple[str, ...]:
        """Sorted session identifiers referenced by this manifest."""
        return tuple(sorted({entry.session_id for entry in self.annotations}))

    def serve_annotations(self) -> tuple[Annotation, ...]:
        """Annotations labeled ``"serve"``, in manifest order."""
        return tuple(
            entry for entry in self.annotations if entry.label == LABEL_SERVE
        )

    def ambiguous_annotations(self) -> tuple[Annotation, ...]:
        """Annotations labeled ``"ambiguous"``, in manifest order."""
        return tuple(
            entry for entry in self.annotations if entry.label == LABEL_AMBIGUOUS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotations": [entry.to_dict() for entry in self.annotations],
            "dataset_split": self.dataset_split,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> AnnotationManifest:
        name = "annotation_manifest"
        if not isinstance(values, dict):
            raise EvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"annotations", "dataset_split", "schema_version"}
        missing = sorted(known - set(values))
        if missing:
            raise EvaluationError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        _check_schema_version(name, values, ANNOTATION_MANIFEST_SCHEMA_VERSION)
        raw = values["annotations"]
        if not isinstance(raw, (list, tuple)):
            raise EvaluationError(
                f"{name}: 'annotations' must be a list of annotation "
                f"objects, got {type(raw).__name__}."
            )
        try:
            parsed = tuple(Annotation.from_dict(entry) for entry in raw)
        except EvaluationError as exc:
            raise EvaluationError(f"{name}: invalid annotation: {exc}.") from exc
        split = values["dataset_split"]
        if split not in ALLOWED_SPLITS:
            raise EvaluationError(
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
    def from_json(cls, data: str | bytes | bytearray) -> AnnotationManifest:
        return cls.from_dict(_loads_object("annotation_manifest", data))


def validate_session_disjoint(
    first: AnnotationManifest, second: AnnotationManifest
) -> None:
    """Reject any shared ``session_id`` between two manifests.

    Raises :class:`EvaluationError` listing the leaked sessions. Pass the
    development manifest and the held-out manifest to guard the M3 gate.
    """
    name = "session_disjoint"
    if not isinstance(first, AnnotationManifest):
        raise EvaluationError(
            f"{name}: first manifest must be an AnnotationManifest, "
            f"got {type(first).__name__}."
        )
    if not isinstance(second, AnnotationManifest):
        raise EvaluationError(
            f"{name}: second manifest must be an AnnotationManifest, "
            f"got {type(second).__name__}."
        )
    leaked = sorted(set(first.sessions) & set(second.sessions))
    if leaked:
        raise EvaluationError(
            f"{name}: session leakage between manifests: {leaked!r}. "
            f"Split labels by recording session, not by nearby clips."
        )


def intersection_seconds(first: MediaRange, second: MediaRange) -> float:
    """Half-open overlap duration in seconds (``0.0`` when disjoint)."""
    name = "intersection"
    if not isinstance(first, MediaRange):
        raise EvaluationError(
            f"{name}: first range must be a MediaRange, "
            f"got {type(first).__name__}."
        )
    if not isinstance(second, MediaRange):
        raise EvaluationError(
            f"{name}: second range must be a MediaRange, "
            f"got {type(second).__name__}."
        )
    return max(
        0.0,
        min(first.end_seconds, second.end_seconds)
        - max(first.start_seconds, second.start_seconds),
    )


def iou(first: MediaRange, second: MediaRange) -> float:
    """Intersection over union for two half-open ranges in ``[0, 1]``.."""
    inter = intersection_seconds(first, second)
    union = first.duration_seconds + second.duration_seconds - inter
    if union <= 0.0:
        return 0.0
    return inter / union


@dataclass(frozen=True, slots=True)
class MatchEntry:
    """One one-to-one truth/prediction pairing at or above threshold."""

    truth_index: int = 0
    pred_index: int = 0
    iou: float = 0.0
    onset_error_seconds: float = 0.0
    end_error_seconds: float = 0.0
    intersection_seconds: float = 0.0

    def __post_init__(self) -> None:
        name = "match_entry"
        for key in ("truth_index", "pred_index"):
            value = getattr(self, key)
            if not _is_int(value) or value < 0:
                raise EvaluationError(
                    f"{name}: {key!r} must be an integer >= 0, "
                    f"got {value!r}."
                )
        iou_value = self.iou
        if (
            not _is_number(iou_value)
            or not math.isfinite(iou_value)
            or iou_value < 0.0
            or iou_value > 1.0
        ):
            raise EvaluationError(
                f"{name}: 'iou' must lie in [0, 1], got {iou_value!r}."
            )
        object.__setattr__(self, "iou", float(iou_value))
        for key in (
            "onset_error_seconds",
            "end_error_seconds",
        ):
            value = getattr(self, key)
            if not _is_number(value) or not math.isfinite(value):
                raise EvaluationError(
                    f"{name}: {key!r} must be a finite number, "
                    f"got {value!r}."
                )
            object.__setattr__(self, key, float(value))
        inter = self.intersection_seconds
        if not _is_number(inter) or not math.isfinite(inter) or inter < 0:
            raise EvaluationError(
                f"{name}: 'intersection_seconds' must be a finite number "
                f">= 0, got {inter!r}."
            )
        object.__setattr__(self, "intersection_seconds", float(inter))

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_error_seconds": self.end_error_seconds,
            "intersection_seconds": self.intersection_seconds,
            "iou": self.iou,
            "onset_error_seconds": self.onset_error_seconds,
            "pred_index": self.pred_index,
            "truth_index": self.truth_index,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MatchEntry:
        name = "match_entry"
        if not isinstance(values, dict):
            raise EvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "end_error_seconds",
            "intersection_seconds",
            "iou",
            "onset_error_seconds",
            "pred_index",
            "truth_index",
        }
        missing = sorted(known - set(values))
        if missing:
            raise EvaluationError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        return cls(
            truth_index=values["truth_index"],
            pred_index=values["pred_index"],
            iou=values["iou"],
            onset_error_seconds=values["onset_error_seconds"],
            end_error_seconds=values["end_error_seconds"],
            intersection_seconds=values["intersection_seconds"],
        )


def match_ranges(
    truth: Sequence[MediaRange],
    predictions: Sequence[MediaRange],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[MatchEntry, ...]:
    """Greedily match truth to predictions one-to-one at ``iou_threshold``.

    Candidate pairs with IoU ``>=`` threshold are claimed in order of
    decreasing IoU; ties break by lower truth index, then lower
    prediction index. Each side participates at most once. Indices refer
    to input order. Output is sorted by ``(truth_index, pred_index)``.
    """
    name = "match_ranges"
    truth_items = _check_range_sequence(name, "'truth'", truth)
    pred_items = _check_range_sequence(name, "'predictions'", predictions)
    threshold = _check_threshold(name, iou_threshold)
    scored: list[tuple[float, int, int]] = []
    for truth_index, truth_range in enumerate(truth_items):
        for pred_index, pred_range in enumerate(pred_items):
            value = iou(truth_range, pred_range)
            if value >= threshold:
                scored.append((value, truth_index, pred_index))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    matches: list[MatchEntry] = []
    for value, truth_index, pred_index in scored:
        if truth_index in used_truth or pred_index in used_pred:
            continue
        used_truth.add(truth_index)
        used_pred.add(pred_index)
        truth_range = truth_items[truth_index]
        pred_range = pred_items[pred_index]
        matches.append(
            MatchEntry(
                truth_index=truth_index,
                pred_index=pred_index,
                iou=value,
                onset_error_seconds=(
                    pred_range.start_seconds - truth_range.start_seconds
                ),
                end_error_seconds=(
                    pred_range.end_seconds - truth_range.end_seconds
                ),
                intersection_seconds=intersection_seconds(
                    truth_range, pred_range
                ),
            )
        )
    matches.sort(key=lambda entry: (entry.truth_index, entry.pred_index))
    return tuple(matches)


def _excused_pred_indices(
    predictions: tuple[MediaRange, ...],
    matched_pred: set[int],
    ambiguous: tuple[MediaRange, ...],
    threshold: float,
) -> tuple[int, ...]:
    """Indices of unmatched predictions overlapping ambiguity at threshold."""
    excused: list[int] = []
    for index, pred_range in enumerate(predictions):
        if index in matched_pred:
            continue
        if any(iou(pred_range, other) >= threshold for other in ambiguous):
            excused.append(index)
    return tuple(excused)


def _scalar_metrics(
    truth: tuple[MediaRange, ...],
    predictions: tuple[MediaRange, ...],
    matches: tuple[MatchEntry, ...],
    excused: tuple[int, ...],
) -> dict[str, Any]:
    """Shared scalar aggregates over one pooling level."""
    num_truth = len(truth)
    num_predictions = len(predictions)
    num_excused = len(excused)
    true_positives = len(matches)
    false_positives = num_predictions - true_positives - num_excused
    false_negatives = num_truth - true_positives
    scored_predictions = num_predictions - num_excused
    precision = (
        true_positives / scored_predictions if scored_predictions > 0 else None
    )
    recall = true_positives / num_truth if num_truth > 0 else None
    if matches:
        onset_errors = [entry.onset_error_seconds for entry in matches]
        end_errors = [entry.end_error_seconds for entry in matches]
        mean_abs_onset = math.fsum(abs(value) for value in onset_errors) / len(
            matches
        )
        mean_abs_end = math.fsum(abs(value) for value in end_errors) / len(matches)
        mean_signed_onset = math.fsum(onset_errors) / len(matches)
        mean_signed_end = math.fsum(end_errors) / len(matches)
    else:
        mean_abs_onset = None
        mean_abs_end = None
        mean_signed_onset = None
        mean_signed_end = None
    truth_duration = math.fsum(entry.duration_seconds for entry in truth)
    matched_intersection = math.fsum(entry.intersection_seconds for entry in matches)
    retained = matched_intersection / truth_duration if truth_duration > 0 else None
    excused_set = frozenset(excused)
    scored_duration = math.fsum(
        entry.duration_seconds
        for index, entry in enumerate(predictions)
        if index not in excused_set
    )
    extra = (
        (scored_duration - matched_intersection) / scored_predictions
        if scored_predictions > 0
        else 0.0
    )
    return {
        "num_truth": num_truth,
        "num_predictions": num_predictions,
        "num_excused_ambiguous": num_excused,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "mean_abs_onset_error_seconds": mean_abs_onset,
        "mean_abs_end_error_seconds": mean_abs_end,
        "mean_signed_onset_error_seconds": mean_signed_onset,
        "mean_signed_end_error_seconds": mean_signed_end,
        "retained_duration_ratio": retained,
        "extra_seconds_per_attempt": extra,
    }


_SCALAR_INT_KEYS = (
    "num_truth",
    "num_predictions",
    "num_excused_ambiguous",
    "true_positives",
    "false_positives",
    "false_negatives",
)


_SCALAR_FLOAT_KEYS = (
    "precision",
    "recall",
    "mean_abs_onset_error_seconds",
    "mean_abs_end_error_seconds",
    "mean_signed_onset_error_seconds",
    "mean_signed_end_error_seconds",
    "retained_duration_ratio",
    "extra_seconds_per_attempt",
)


def _scalars_payload(source: Any) -> dict[str, Any]:
    """Collect shared scalar fields via getattr (slots-safe)."""
    return {
        key: getattr(source, key)
        for key in (*_SCALAR_INT_KEYS, *_SCALAR_FLOAT_KEYS)
    }


def _validate_scalars(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate shared scalar fields; return a cleaned copy."""
    cleaned: dict[str, Any] = {}
    for key in _SCALAR_INT_KEYS:
        cleaned[key] = _check_count(name, key, payload[key])
    if (
        cleaned["true_positives"] > cleaned["num_truth"]
        or cleaned["true_positives"]
        > cleaned["num_predictions"] - cleaned["num_excused_ambiguous"]
        or cleaned["false_positives"]
        != cleaned["num_predictions"]
        - cleaned["true_positives"]
        - cleaned["num_excused_ambiguous"]
        or cleaned["false_negatives"]
        != cleaned["num_truth"] - cleaned["true_positives"]
    ):
        raise EvaluationError(f"{name}: inconsistent count fields.")
    cleaned["precision"] = _check_ratio_or_none(name, "precision", payload["precision"])
    cleaned["recall"] = _check_ratio_or_none(name, "recall", payload["recall"])
    scored = cleaned["num_predictions"] - cleaned["num_excused_ambiguous"]
    if (cleaned["precision"] is None) != (scored == 0):
        raise EvaluationError(
            f"{name}: 'precision' must be None exactly when there are no "
            f"scored predictions."
        )
    if (cleaned["recall"] is None) != (cleaned["num_truth"] == 0):
        raise EvaluationError(
            f"{name}: 'recall' must be None exactly when there is no "
            f"ground truth."
        )
    has_matches = cleaned["true_positives"] > 0
    for key in (
        "mean_abs_onset_error_seconds",
        "mean_abs_end_error_seconds",
        "mean_signed_onset_error_seconds",
        "mean_signed_end_error_seconds",
    ):
        cleaned[key] = _check_error_or_none(name, key, payload[key])
        if (cleaned[key] is None) != (not has_matches):
            raise EvaluationError(
                f"{name}: {key!r} must be None exactly when there are no "
                f"matches."
            )
    for key in ("mean_abs_onset_error_seconds", "mean_abs_end_error_seconds"):
        if cleaned[key] is not None and cleaned[key] < 0.0:
            raise EvaluationError(
                f"{name}: {key!r} must be >= 0, got {cleaned[key]!r}."
            )
    cleaned["retained_duration_ratio"] = _check_ratio_or_none(
        name, "retained_duration_ratio", payload["retained_duration_ratio"]
    )
    if cleaned["retained_duration_ratio"] is not None and cleaned["num_truth"] == 0:
        raise EvaluationError(
            f"{name}: 'retained_duration_ratio' must be None when there is "
            f"no ground truth."
        )
    cleaned["extra_seconds_per_attempt"] = _check_extra_seconds(
        name, payload["extra_seconds_per_attempt"]
    )
    if scored == 0 and cleaned["extra_seconds_per_attempt"] != 0.0:
        raise EvaluationError(
            f"{name}: 'extra_seconds_per_attempt' must be 0.0 when there "
            f"are no scored predictions."
        )
    return cleaned


@dataclass(frozen=True, slots=True)
class RangeMetrics:
    """Single-stratum metrics plus the one-to-one match list."""

    num_truth: int = 0
    num_predictions: int = 0
    num_excused_ambiguous: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float | None = None
    recall: float | None = None
    mean_abs_onset_error_seconds: float | None = None
    mean_abs_end_error_seconds: float | None = None
    mean_signed_onset_error_seconds: float | None = None
    mean_signed_end_error_seconds: float | None = None
    retained_duration_ratio: float | None = None
    extra_seconds_per_attempt: float = 0.0
    matches: tuple[MatchEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = "range_metrics"
        cleaned = _validate_scalars(name, _scalars_payload(self))
        for key, value in cleaned.items():
            object.__setattr__(self, key, value)
        raw = self.matches
        if isinstance(raw, MatchEntry) or not isinstance(raw, (list, tuple)):
            raise EvaluationError(
                f"{name}: 'matches' must be a list or tuple of MatchEntry, "
                f"got {type(raw).__name__}."
            )
        items = tuple(raw)
        for entry in items:
            if not isinstance(entry, MatchEntry):
                raise EvaluationError(
                    f"{name}: every match must be a MatchEntry, "
                    f"got {type(entry).__name__}."
                )
        if len(items) != self.true_positives:
            raise EvaluationError(
                f"{name}: 'matches' length ({len(items)}) must equal "
                f"'true_positives' ({self.true_positives})."
            )
        object.__setattr__(self, "matches", items)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_seconds_per_attempt": self.extra_seconds_per_attempt,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "matches": [entry.to_dict() for entry in self.matches],
            "mean_abs_end_error_seconds": self.mean_abs_end_error_seconds,
            "mean_abs_onset_error_seconds": self.mean_abs_onset_error_seconds,
            "mean_signed_end_error_seconds": self.mean_signed_end_error_seconds,
            "mean_signed_onset_error_seconds": self.mean_signed_onset_error_seconds,
            "num_excused_ambiguous": self.num_excused_ambiguous,
            "num_predictions": self.num_predictions,
            "num_truth": self.num_truth,
            "precision": self.precision,
            "recall": self.recall,
            "retained_duration_ratio": self.retained_duration_ratio,
            "true_positives": self.true_positives,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RangeMetrics:
        name = "range_metrics"
        if not isinstance(values, dict):
            raise EvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "extra_seconds_per_attempt",
            "false_negatives",
            "false_positives",
            "matches",
            "mean_abs_end_error_seconds",
            "mean_abs_onset_error_seconds",
            "mean_signed_end_error_seconds",
            "mean_signed_onset_error_seconds",
            "num_excused_ambiguous",
            "num_predictions",
            "num_truth",
            "precision",
            "recall",
            "retained_duration_ratio",
            "true_positives",
        }
        missing = sorted(known - set(values))
        if missing:
            raise EvaluationError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        raw_matches = values["matches"]
        if not isinstance(raw_matches, (list, tuple)):
            raise EvaluationError(
                f"{name}: 'matches' must be a list of match objects, "
                f"got {type(raw_matches).__name__}."
            )
        try:
            parsed = tuple(MatchEntry.from_dict(entry) for entry in raw_matches)
        except EvaluationError as exc:
            raise EvaluationError(f"{name}: invalid match: {exc}.") from exc
        return cls(
            num_truth=values["num_truth"],
            num_predictions=values["num_predictions"],
            num_excused_ambiguous=values["num_excused_ambiguous"],
            true_positives=values["true_positives"],
            false_positives=values["false_positives"],
            false_negatives=values["false_negatives"],
            precision=values["precision"],
            recall=values["recall"],
            mean_abs_onset_error_seconds=values["mean_abs_onset_error_seconds"],
            mean_abs_end_error_seconds=values["mean_abs_end_error_seconds"],
            mean_signed_onset_error_seconds=values[
                "mean_signed_onset_error_seconds"
            ],
            mean_signed_end_error_seconds=values["mean_signed_end_error_seconds"],
            retained_duration_ratio=values["retained_duration_ratio"],
            extra_seconds_per_attempt=values["extra_seconds_per_attempt"],
            matches=parsed,
        )


@dataclass(frozen=True, slots=True)
class OverallMetrics:
    """Pooled scalars across strata (no pooled match list)."""

    num_truth: int = 0
    num_predictions: int = 0
    num_excused_ambiguous: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float | None = None
    recall: float | None = None
    mean_abs_onset_error_seconds: float | None = None
    mean_abs_end_error_seconds: float | None = None
    mean_signed_onset_error_seconds: float | None = None
    mean_signed_end_error_seconds: float | None = None
    retained_duration_ratio: float | None = None
    extra_seconds_per_attempt: float = 0.0

    def __post_init__(self) -> None:
        name = "overall_metrics"
        cleaned = _validate_scalars(name, _scalars_payload(self))
        for key, value in cleaned.items():
            object.__setattr__(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_seconds_per_attempt": self.extra_seconds_per_attempt,
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "mean_abs_end_error_seconds": self.mean_abs_end_error_seconds,
            "mean_abs_onset_error_seconds": self.mean_abs_onset_error_seconds,
            "mean_signed_end_error_seconds": self.mean_signed_end_error_seconds,
            "mean_signed_onset_error_seconds": self.mean_signed_onset_error_seconds,
            "num_excused_ambiguous": self.num_excused_ambiguous,
            "num_predictions": self.num_predictions,
            "num_truth": self.num_truth,
            "precision": self.precision,
            "recall": self.recall,
            "retained_duration_ratio": self.retained_duration_ratio,
            "true_positives": self.true_positives,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> OverallMetrics:
        name = "overall_metrics"
        if not isinstance(values, dict):
            raise EvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "extra_seconds_per_attempt",
            "false_negatives",
            "false_positives",
            "mean_abs_end_error_seconds",
            "mean_abs_onset_error_seconds",
            "mean_signed_end_error_seconds",
            "mean_signed_onset_error_seconds",
            "num_excused_ambiguous",
            "num_predictions",
            "num_truth",
            "precision",
            "recall",
            "retained_duration_ratio",
            "true_positives",
        }
        missing = sorted(known - set(values))
        if missing:
            raise EvaluationError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        return cls(**{key: values[key] for key in known})


@dataclass(frozen=True, slots=True)
class StratumInput:
    """Caller-supplied truth/predictions for one stratum (session)."""

    truth: tuple[MediaRange, ...] = field(default_factory=tuple)
    predictions: tuple[MediaRange, ...] = field(default_factory=tuple)
    ambiguous: tuple[MediaRange, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = "stratum_input"
        object.__setattr__(
            self, "truth", _check_range_sequence(name, "'truth'", self.truth)
        )
        object.__setattr__(
            self,
            "predictions",
            _check_range_sequence(name, "'predictions'", self.predictions),
        )
        object.__setattr__(
            self,
            "ambiguous",
            _check_range_sequence(name, "'ambiguous'", self.ambiguous),
        )


@dataclass(frozen=True, slots=True)
class StratumReport:
    """Per-stratum metrics with its match list."""

    stratum: str = ""
    metrics: RangeMetrics = field(default_factory=RangeMetrics)

    def __post_init__(self) -> None:
        name = "stratum_report"
        object.__setattr__(
            self, "stratum", _check_non_blank(name, "'stratum'", self.stratum)
        )
        if not isinstance(self.metrics, RangeMetrics):
            raise EvaluationError(
                f"{name}: 'metrics' must be RangeMetrics, "
                f"got {type(self.metrics).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": self.metrics.to_dict(), "stratum": self.stratum}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StratumReport:
        name = "stratum_report"
        if not isinstance(values, dict):
            raise EvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"metrics", "stratum"}
        missing = sorted(known - set(values))
        if missing:
            raise EvaluationError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        try:
            metrics = RangeMetrics.from_dict(values["metrics"])
        except EvaluationError as exc:
            raise EvaluationError(f"{name}: invalid metrics: {exc}.") from exc
        return cls(stratum=values["stratum"], metrics=metrics)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Versioned machine-readable evaluation report."""

    schema_version: int = EVALUATION_REPORT_SCHEMA_VERSION
    iou_threshold: float = IOU_THRESHOLD
    overall: OverallMetrics = field(default_factory=OverallMetrics)
    strata: tuple[StratumReport, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        name = "evaluation_report"
        version = self.schema_version
        if not _is_int(version):
            raise EvaluationError(
                f"{name}: 'schema_version' must be an integer, got {version!r}."
            )
        if version != EVALUATION_REPORT_SCHEMA_VERSION:
            if version > EVALUATION_REPORT_SCHEMA_VERSION:
                raise EvaluationError(
                    f"{name}: unsupported newer schema_version {version!r}; "
                    f"this build supports version "
                    f"{EVALUATION_REPORT_SCHEMA_VERSION}."
                )
            raise EvaluationError(
                f"{name}: unsupported schema_version {version!r}; expected "
                f"version {EVALUATION_REPORT_SCHEMA_VERSION}."
            )
        object.__setattr__(
            self, "iou_threshold", _check_threshold(name, self.iou_threshold)
        )
        if not isinstance(self.overall, OverallMetrics):
            raise EvaluationError(
                f"{name}: 'overall' must be OverallMetrics, "
                f"got {type(self.overall).__name__}."
            )
        raw = self.strata
        if isinstance(raw, StratumReport) or not isinstance(raw, (list, tuple)):
            raise EvaluationError(
                f"{name}: 'strata' must be a list or tuple of StratumReport, "
                f"got {type(raw).__name__}."
            )
        items = tuple(raw)
        if not items:
            raise EvaluationError(f"{name}: at least one stratum is required.")
        for entry in items:
            if not isinstance(entry, StratumReport):
                raise EvaluationError(
                    f"{name}: every stratum must be a StratumReport, "
                    f"got {type(entry).__name__}."
                )
        names = [entry.stratum for entry in items]
        if len(set(names)) != len(names):
            raise EvaluationError(
                f"{name}: duplicate stratum names {names!r}."
            )
        object.__setattr__(
            self, "strata", tuple(sorted(items, key=lambda entry: entry.stratum))
        )

    def stratum(self, name: str) -> StratumReport:
        """Return the report for one stratum, or raise :class:`KeyError`."""
        for entry in self.strata:
            if entry.stratum == name:
                return entry
        raise KeyError(f"evaluation_report: unknown stratum {name!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "overall": self.overall.to_dict(),
            "schema_version": self.schema_version,
            "strata": [entry.to_dict() for entry in self.strata],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> EvaluationReport:
        name = "evaluation_report"
        if not isinstance(values, dict):
            raise EvaluationError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {"iou_threshold", "overall", "schema_version", "strata"}
        missing = sorted(known - set(values))
        if missing:
            raise EvaluationError(f"{name}: missing required keys {missing!r}.")
        _check_no_unknown_keys(name, values, known)
        _check_schema_version(name, values, EVALUATION_REPORT_SCHEMA_VERSION)
        try:
            overall = OverallMetrics.from_dict(values["overall"])
        except EvaluationError as exc:
            raise EvaluationError(f"{name}: invalid overall: {exc}.") from exc
        raw_strata = values["strata"]
        if not isinstance(raw_strata, (list, tuple)):
            raise EvaluationError(
                f"{name}: 'strata' must be a list of stratum objects, "
                f"got {type(raw_strata).__name__}."
            )
        try:
            parsed = tuple(
                StratumReport.from_dict(entry) for entry in raw_strata
            )
        except EvaluationError as exc:
            raise EvaluationError(f"{name}: invalid stratum: {exc}.") from exc
        return cls(
            schema_version=values["schema_version"],
            iou_threshold=values["iou_threshold"],
            overall=overall,
            strata=parsed,
        )

    def to_json(self) -> str:
        return _dumps_deterministic(self.to_dict())

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> EvaluationReport:
        return cls.from_dict(_loads_object("evaluation_report", data))


def evaluate_ranges(
    truth: Sequence[MediaRange],
    predictions: Sequence[MediaRange],
    *,
    ambiguous_ranges: Sequence[MediaRange] = (),
    iou_threshold: float = IOU_THRESHOLD,
) -> RangeMetrics:
    """Score one stratum: match, excuse ambiguity, and aggregate scalars."""
    name = "evaluate_ranges"
    truth_items = _check_range_sequence(name, "'truth'", truth)
    pred_items = _check_range_sequence(name, "'predictions'", predictions)
    ambiguous_items = _check_range_sequence(
        name, "'ambiguous_ranges'", ambiguous_ranges
    )
    threshold = _check_threshold(name, iou_threshold)
    matches = match_ranges(truth_items, pred_items, iou_threshold=threshold)
    matched_pred = {entry.pred_index for entry in matches}
    excused = _excused_pred_indices(
        pred_items, matched_pred, ambiguous_items, threshold
    )
    return RangeMetrics(
        **_scalar_metrics(truth_items, pred_items, matches, excused),
        matches=matches,
    )


def evaluate_report(
    strata: Mapping[str, StratumInput],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> EvaluationReport:
    """Score each stratum independently and pool overall scalars.

    ``strata`` maps a non-blank stratum name (typically a recording
    session) to its truth/predictions/ambiguous ranges. Matching never
    crosses strata. Overall scalars pool per-stratum matches and counts
    with :func:`math.fsum` over strata sorted by name.
    """
    name = "evaluate_report"
    threshold = _check_threshold(name, iou_threshold)
    if not isinstance(strata, Mapping):
        raise EvaluationError(
            f"{name}: 'strata' must be a mapping of stratum name to "
            f"StratumInput, got {type(strata).__name__}."
        )
    if not strata:
        raise EvaluationError(f"{name}: at least one stratum is required.")
    ordered_names = sorted(strata)
    for key in ordered_names:
        if not isinstance(key, str) or not key.strip():
            raise EvaluationError(
                f"{name}: stratum names must be non-blank strings, "
                f"got {key!r}."
            )
        if not isinstance(strata[key], StratumInput):
            raise EvaluationError(
                f"{name}: stratum {key!r} must be a StratumInput, "
                f"got {type(strata[key]).__name__}."
            )
    per_stratum: list[StratumReport] = []
    pooled_truth: list[MediaRange] = []
    pooled_predictions: list[MediaRange] = []
    pooled_matches: list[MatchEntry] = []
    pooled_excused_count = 0
    for key in ordered_names:
        entry = strata[key]
        metrics = evaluate_ranges(
            entry.truth,
            entry.predictions,
            ambiguous_ranges=entry.ambiguous,
            iou_threshold=threshold,
        )
        per_stratum.append(StratumReport(stratum=key, metrics=metrics))
        pooled_truth.extend(entry.truth)
        pooled_predictions.extend(entry.predictions)
        pooled_matches.extend(metrics.matches)
        pooled_excused_count += metrics.num_excused_ambiguous
    # Re-derive excused flags on pooled predictions is unnecessary: counts
    # and durations pool per stratum. Rebuild pooled scalars directly so
    # overall never re-matches across sessions.
    total_truth = len(pooled_truth)
    total_pred = len(pooled_predictions)
    total_tp = math.fsum(1 for _ in pooled_matches)
    total_tp_int = int(total_tp)
    total_fp = total_pred - total_tp_int - pooled_excused_count
    total_fn = total_truth - total_tp_int
    scored = total_pred - pooled_excused_count
    overall_precision = total_tp_int / scored if scored > 0 else None
    overall_recall = total_tp_int / total_truth if total_truth > 0 else None
    if pooled_matches:
        onset = [entry.onset_error_seconds for entry in pooled_matches]
        end = [entry.end_error_seconds for entry in pooled_matches]
        abs_onset = math.fsum(abs(value) for value in onset) / len(pooled_matches)
        abs_end = math.fsum(abs(value) for value in end) / len(pooled_matches)
        signed_onset = math.fsum(onset) / len(pooled_matches)
        signed_end = math.fsum(end) / len(pooled_matches)
    else:
        abs_onset = abs_end = signed_onset = signed_end = None
    truth_duration = math.fsum(entry.duration_seconds for entry in pooled_truth)
    matched_inter = math.fsum(entry.intersection_seconds for entry in pooled_matches)
    retained = matched_inter / truth_duration if truth_duration > 0 else None
    # Scored predicted duration pools per-stratum scored durations, which
    # requires each stratum's excused indices. Recompute deterministically
    # from the inputs in stratum order.
    scored_duration = 0.0
    for key in ordered_names:
        entry = strata[key]
        stratum_matches = match_ranges(
            entry.truth, entry.predictions, iou_threshold=threshold
        )
        matched = {item.pred_index for item in stratum_matches}
        excused = _excused_pred_indices(
            tuple(entry.predictions), matched, tuple(entry.ambiguous), threshold
        )
        excused_set = frozenset(excused)
        scored_duration += math.fsum(
            item.duration_seconds
            for index, item in enumerate(entry.predictions)
            if index not in excused_set
        )
    extra = (scored_duration - matched_inter) / scored if scored > 0 else 0.0
    overall = OverallMetrics(
        num_truth=total_truth,
        num_predictions=total_pred,
        num_excused_ambiguous=pooled_excused_count,
        true_positives=total_tp_int,
        false_positives=total_fp,
        false_negatives=total_fn,
        precision=overall_precision,
        recall=overall_recall,
        mean_abs_onset_error_seconds=abs_onset,
        mean_abs_end_error_seconds=abs_end,
        mean_signed_onset_error_seconds=signed_onset,
        mean_signed_end_error_seconds=signed_end,
        retained_duration_ratio=retained,
        extra_seconds_per_attempt=extra,
    )
    return EvaluationReport(
        schema_version=EVALUATION_REPORT_SCHEMA_VERSION,
        iou_threshold=threshold,
        overall=overall,
        strata=tuple(per_stratum),
    )


def group_truth_by_session(
    manifest: AnnotationManifest,
) -> dict[str, tuple[MediaRange, ...]]:
    """Map each manifest session to its positive ranges.

    Ranges within a session are sorted by ``(media, start, end)`` so
    match indices are stable regardless of authoring order.
    """
    return _group_by_session(manifest, LABEL_SERVE)


def group_ambiguous_by_session(
    manifest: AnnotationManifest,
) -> dict[str, tuple[MediaRange, ...]]:
    """Map each manifest session to its ambiguous ranges (sorted)."""
    return _group_by_session(manifest, LABEL_AMBIGUOUS)


def _group_by_session(
    manifest: AnnotationManifest, label: str
) -> dict[str, tuple[MediaRange, ...]]:
    name = "group_by_session"
    if not isinstance(manifest, AnnotationManifest):
        raise EvaluationError(
            f"{name}: manifest must be an AnnotationManifest, "
            f"got {type(manifest).__name__}."
        )
    grouped: dict[str, list[tuple[str, float, float, MediaRange]]] = {}
    for entry in manifest.annotations:
        if entry.label != label:
            continue
        grouped.setdefault(entry.session_id, []).append(
            (
                entry.media,
                entry.start_seconds,
                entry.end_seconds,
                entry.to_range(),
            )
        )
    ordered: dict[str, tuple[MediaRange, ...]] = {}
    for session in sorted(grouped):
        items = sorted(grouped[session], key=lambda item: (item[0], item[1], item[2]))
        ordered[session] = tuple(item[3] for item in items)
    return ordered


def evaluate_manifest(
    manifest: AnnotationManifest,
    predictions_by_session: Mapping[str, Sequence[MediaRange]] | None = None,
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> EvaluationReport:
    """Score a manifest's sessions against per-session predictions.

    Sessions without predictions score zero true positives (every truth
    range is a false negative). Prediction sessions absent from the
    manifest raise :class:`EvaluationError` instead of silently scoring
    as false positives, so session typos surface as errors.
    """
    name = "evaluate_manifest"
    if not isinstance(manifest, AnnotationManifest):
        raise EvaluationError(
            f"{name}: manifest must be an AnnotationManifest, "
            f"got {type(manifest).__name__}."
        )
    threshold = _check_threshold(name, iou_threshold)
    predictions: dict[str, tuple[MediaRange, ...]] = {}
    if predictions_by_session is not None:
        if not isinstance(predictions_by_session, Mapping):
            raise EvaluationError(
                f"{name}: 'predictions_by_session' must be a mapping of "
                f"session id to ranges, got "
                f"{type(predictions_by_session).__name__}."
            )
        for session, ranges in predictions_by_session.items():
            if not isinstance(session, str) or not session.strip():
                raise EvaluationError(
                    f"{name}: prediction session ids must be non-blank "
                    f"strings, got {session!r}."
                )
            predictions[session] = _check_range_sequence(
                name, f"predictions[{session!r}]", ranges
            )
    known_sessions = set(manifest.sessions)
    unknown = sorted(set(predictions) - known_sessions)
    if unknown:
        raise EvaluationError(
            f"{name}: predictions reference sessions absent from the "
            f"manifest: {unknown!r}."
        )
    truth_by_session = group_truth_by_session(manifest)
    ambiguous_by_session = group_ambiguous_by_session(manifest)
    strata: dict[str, StratumInput] = {}
    for session in manifest.sessions:
        strata[session] = StratumInput(
            truth=truth_by_session.get(session, ()),
            predictions=predictions.get(session, ()),
            ambiguous=ambiguous_by_session.get(session, ()),
        )
    if not strata:
        raise EvaluationError(
            f"{name}: manifest contains no sessions to evaluate."
        )
    return evaluate_report(strata, iou_threshold=threshold)
