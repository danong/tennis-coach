"""Pure scalar comparison contracts and operations for ServeComparisonV1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from serve_review.fingerprint import (
    COMPARISON_DOMAIN,
    LAYOUT_ID,
    METRIC_NAMES,
    METRIC_UNITS,
    RESAMPLING_METHOD,
    SAMPLES_PER_SEGMENT,
    SCHEMA,
    SCHEMA_VERSION,
    SEGMENT_LAYOUT,
    SHAPE,
    CHANNEL_NAMES,
    CHANNEL_UNITS,
    ServeFingerprintV1,
)


PAIRWISE_SCHEMA = "serve-pairwise-comparison-v1"
BASELINE_SCHEMA = "serve-baseline-comparison-v1"

CAPTURE_CONTEXTS = (
    "asserted_compatible",
    "not_asserted",
    "asserted_incompatible",
)
SEMANTIC_PROVENANCE_POLICY = "require-equal-measurement-provenance-v1"
SEMANTIC_PROVENANCE_FIELDS = (
    "waveform_method_version",
    "waveform_config_id",
    "checkpoint_method_version",
    "checkpoint_config_id",
    "body_model_name",
    "body_model_version",
)
PROVENANCE_FIELDS = (
    "coordinate_convention",
    *SEMANTIC_PROVENANCE_FIELDS,
)

PAIRWISE_MISSINGNESS_REASONS = (
    "comparison_incompatible",
    "candidate_metric_unavailable",
    "reference_metric_unavailable",
    "both_metrics_unavailable",
)
BASELINE_DESCRIPTIVE_REASONS = (
    "comparison_incompatible",
    "candidate_metric_unavailable",
    "no_usable_cohort_values",
)
BASELINE_ROBUST_REASONS = (
    *BASELINE_DESCRIPTIVE_REASONS,
    "insufficient_usable_n",
    "zero_mad",
)

# Identity exclusions are intentionally small and stable: curation-specific
# reasons (for example, handedness) belong to the caller and never enter this
# numerical operation.
BASELINE_EXCLUSION_REASONS = (
    "candidate_identity_leave_one_out",
    "schema_mismatch",
    "schema_version_mismatch",
    "comparison_domain_mismatch",
    "phase_layout_mismatch",
    "coordinate_convention_mismatch",
    "metric_inventory_mismatch",
    "semantic_provenance_mismatch",
    "capture_context_not_asserted",
    "capture_context_asserted_incompatible",
)

STRUCTURAL_REASONS = (
    "schema_mismatch",
    "schema_version_mismatch",
    "comparison_domain_mismatch",
    "phase_layout_mismatch",
    "coordinate_convention_mismatch",
    "metric_inventory_mismatch",
    "semantic_provenance_mismatch",
)


class ComparisonError(ValueError):
    """Invalid comparison value, policy, or compatibility input."""


def load_fingerprint(path: str | Path) -> ServeFingerprintV1:
    """Load one persisted V1 fingerprint from an explicitly selected path.

    The path is included in every loading error so a caller cannot mistake an
    invalid artifact for an absent cohort member.  This helper intentionally
    does no discovery or directory traversal.
    """

    try:
        fingerprint_path = Path(path).expanduser()
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{path!r}: invalid fingerprint path: {exc}.") from exc
    try:
        payload = fingerprint_path.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"{fingerprint_path}: could not read fingerprint: {exc}.") from exc
    try:
        return ServeFingerprintV1.from_json(payload)
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f"{fingerprint_path}: invalid fingerprint: {exc}") from exc


def load_fingerprints(paths: Iterable[str | Path]) -> list[ServeFingerprintV1]:
    """Load every fingerprint at the caller-supplied paths, in order.

    Invalid paths are errors; they are never skipped.  The function accepts a
    concrete path collection only and does not recursively discover files.
    """

    if isinstance(paths, (str, bytes, Path)):
        raise ComparisonError("fingerprint paths must be an explicit collection of paths.")
    try:
        selected_paths = list(paths)
    except TypeError as exc:
        raise ComparisonError("fingerprint paths must be an explicit collection of paths.") from exc
    return [load_fingerprint(path) for path in selected_paths]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ComparisonError(f"{name} must be a finite number.")
    return float(value)


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComparisonError(f"{name} must be a non-blank string.")
    return value


def _strict_mapping(value: Any, required: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != required:
        raise ComparisonError(f"{name} must contain exactly {sorted(required)}.")
    return value


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _tuple_strings(value: Sequence[str], name: str, allowed: Sequence[str] | None = None) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ComparisonError(f"{name} must be a sequence of strings.")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise ComparisonError(f"{name} must contain only strings.")
    if allowed is not None and any(item not in allowed for item in result):
        raise ComparisonError(f"{name} contains an unsupported value.")
    return result


@dataclass(frozen=True)
class FingerprintIdentity:
    """Stable identity of one persisted fingerprint (not a display label)."""

    source_fingerprint: str
    attempt_start_seconds: float
    attempt_end_seconds: float

    def __post_init__(self) -> None:
        _nonblank(self.source_fingerprint, "source_fingerprint")
        start = _finite(self.attempt_start_seconds, "attempt_start_seconds")
        end = _finite(self.attempt_end_seconds, "attempt_end_seconds")
        if end <= start:
            raise ComparisonError("attempt range must have positive duration.")
        object.__setattr__(self, "attempt_start_seconds", start)
        object.__setattr__(self, "attempt_end_seconds", end)

    @classmethod
    def from_fingerprint(cls, fingerprint: ServeFingerprintV1) -> "FingerprintIdentity":
        if not isinstance(fingerprint, ServeFingerprintV1):
            raise ComparisonError("fingerprint must be a ServeFingerprintV1.")
        return cls(fingerprint.source_fingerprint, fingerprint.attempt_start_seconds,
                   fingerprint.attempt_end_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_fingerprint": self.source_fingerprint,
            "attempt_start_seconds": self.attempt_start_seconds,
            "attempt_end_seconds": self.attempt_end_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FingerprintIdentity":
        value = _strict_mapping(payload, {"source_fingerprint", "attempt_start_seconds", "attempt_end_seconds"},
                                "fingerprint identity")
        return cls(value["source_fingerprint"], value["attempt_start_seconds"], value["attempt_end_seconds"])


@dataclass(frozen=True)
class ComparisonPolicy:
    """Caller-owned capture assertion and semantic provenance policy."""

    policy_id: str
    capture_context: str
    semantic_provenance: str = SEMANTIC_PROVENANCE_POLICY

    def __post_init__(self) -> None:
        _nonblank(self.policy_id, "policy_id")
        if self.capture_context not in CAPTURE_CONTEXTS:
            raise ComparisonError(f"capture_context must be one of {CAPTURE_CONTEXTS!r}.")
        if self.semantic_provenance != SEMANTIC_PROVENANCE_POLICY:
            raise ComparisonError(
                f"semantic_provenance is fixed to {SEMANTIC_PROVENANCE_POLICY!r} for V1."
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "policy_id": self.policy_id,
            "capture_context": self.capture_context,
            "semantic_provenance": self.semantic_provenance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComparisonPolicy":
        value = _strict_mapping(payload, {"policy_id", "capture_context", "semantic_provenance"},
                                "comparison policy")
        return cls(value["policy_id"], value["capture_context"], value["semantic_provenance"])


@dataclass(frozen=True)
class CompatibilityResult:
    """Structural facts, caller capture assertion, and provenance diagnostics."""

    structurally_compatible: bool
    capture_context_status: str
    capture_policy_id: str
    structural_reasons: tuple[str, ...] = ()
    provenance_differences: Mapping[str, tuple[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.structurally_compatible) is not bool:
            raise ComparisonError("structurally_compatible must be boolean.")
        if self.capture_context_status not in CAPTURE_CONTEXTS:
            raise ComparisonError(f"capture_context_status must be one of {CAPTURE_CONTEXTS!r}.")
        _nonblank(self.capture_policy_id, "capture_policy_id")
        reasons = _tuple_strings(self.structural_reasons, "structural_reasons", STRUCTURAL_REASONS)
        if len(set(reasons)) != len(reasons):
            raise ComparisonError("structural_reasons must not contain duplicates.")
        object.__setattr__(self, "structural_reasons", reasons)
        if not isinstance(self.provenance_differences, Mapping):
            raise ComparisonError("provenance_differences must be a mapping.")
        differences: dict[str, tuple[str, str]] = {}
        for field_name, values in self.provenance_differences.items():
            if field_name not in PROVENANCE_FIELDS:
                raise ComparisonError(f"unsupported provenance field {field_name!r}.")
            if not isinstance(values, (tuple, list)) or len(values) != 2:
                raise ComparisonError(f"provenance difference {field_name!r} must have two values.")
            left, right = values
            if not isinstance(left, str) or not isinstance(right, str):
                raise ComparisonError(f"provenance difference {field_name!r} values must be strings.")
            differences[field_name] = (left, right)
        object.__setattr__(self, "provenance_differences", differences)

    @property
    def compatible(self) -> bool:
        return self.structurally_compatible and self.capture_context_status == "asserted_compatible"

    def to_dict(self) -> dict[str, Any]:
        return {
            "structurally_compatible": self.structurally_compatible,
            "capture_context_status": self.capture_context_status,
            "capture_policy_id": self.capture_policy_id,
            "compatible": self.compatible,
            "structural_reasons": list(self.structural_reasons),
            "provenance_differences": {
                name: {"left": values[0], "right": values[1]}
                for name, values in self.provenance_differences.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CompatibilityResult":
        value = _strict_mapping(
            payload,
            {"structurally_compatible", "capture_context_status", "capture_policy_id",
             "compatible", "structural_reasons", "provenance_differences"},
            "compatibility",
        )
        expected = bool(value["structurally_compatible"]) and value["capture_context_status"] == "asserted_compatible"
        if type(value["compatible"]) is not bool or value["compatible"] != expected:
            raise ComparisonError("compatibility compatible field is inconsistent.")
        raw_differences = value["provenance_differences"]
        if not isinstance(raw_differences, Mapping):
            raise ComparisonError("provenance_differences must be an object.")
        differences = {}
        for name, raw in raw_differences.items():
            item = _strict_mapping(raw, {"left", "right"}, f"provenance difference {name}")
            differences[name] = (item["left"], item["right"])
        return cls(value["structurally_compatible"], value["capture_context_status"],
                   value["capture_policy_id"], tuple(value["structural_reasons"]), differences)


def check_compatibility(
    candidate: ServeFingerprintV1,
    reference: ServeFingerprintV1,
    *,
    policy: ComparisonPolicy,
) -> CompatibilityResult:
    """Check fixed structural contracts and caller-supplied capture context.

    All differing instance provenance fields are returned as diagnostics.  The
    six measurement-provenance fields named by V1's policy are structural hard
    gates. Extractor method/config are fixed by ``ServeFingerprintV1.from_json``
    and therefore cannot differ between parsed V1 values.
    """

    if not isinstance(candidate, ServeFingerprintV1) or not isinstance(reference, ServeFingerprintV1):
        raise ComparisonError("candidate and reference must be ServeFingerprintV1 values.")
    if not isinstance(policy, ComparisonPolicy):
        raise ComparisonError("policy must be a ComparisonPolicy.")

    candidate_payload = candidate.to_dict()
    reference_payload = reference.to_dict()
    reasons: list[str] = []

    if candidate_payload.get("schema") != SCHEMA or reference_payload.get("schema") != SCHEMA:
        reasons.append("schema_mismatch")
    if candidate_payload.get("schema_version") != SCHEMA_VERSION or reference_payload.get("schema_version") != SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if (candidate_payload.get("comparison_domain") != COMPARISON_DOMAIN or
            reference_payload.get("comparison_domain") != COMPARISON_DOMAIN or
            candidate_payload.get("comparison_domain") != reference_payload.get("comparison_domain")):
        reasons.append("comparison_domain_mismatch")
    expected_layout = {
        "layout_id": LAYOUT_ID,
        "samples_per_segment": SAMPLES_PER_SEGMENT,
        "resampling_method": RESAMPLING_METHOD,
        "segments": [{"id": s, "start_anchor": a, "end_anchor": b} for s, a, b in SEGMENT_LAYOUT],
    }
    candidate_layout = candidate_payload.get("phase_layout")
    reference_layout = reference_payload.get("phase_layout")
    if candidate_layout != expected_layout or reference_layout != expected_layout or candidate_layout != reference_layout:
        reasons.append("phase_layout_mismatch")
    expected_sequence = {
        "channel_order": list(CHANNEL_NAMES),
        "channel_units": [CHANNEL_UNITS[name] for name in CHANNEL_NAMES],
        "shape": list(SHAPE),
    }
    candidate_sequence = candidate_payload.get("normalized_sequence", {})
    reference_sequence = reference_payload.get("normalized_sequence", {})
    if (not isinstance(candidate_sequence, Mapping) or not isinstance(reference_sequence, Mapping) or
            any(candidate_sequence.get(name) != expected for name, expected in expected_sequence.items()) or
            any(reference_sequence.get(name) != expected for name, expected in expected_sequence.items())):
        reasons.append("phase_layout_mismatch")
    candidate_provenance = candidate.provenance
    reference_provenance = reference.provenance
    differences = {
        name: (candidate_provenance[name], reference_provenance[name])
        for name in PROVENANCE_FIELDS
        if candidate_provenance[name] != reference_provenance[name]
    }
    if candidate_provenance["coordinate_convention"] != reference_provenance["coordinate_convention"]:
        reasons.append("coordinate_convention_mismatch")
    if set(candidate.metrics) != set(METRIC_NAMES) or set(reference.metrics) != set(METRIC_NAMES):
        reasons.append("metric_inventory_mismatch")
    elif any(candidate.metrics[name].unit != reference.metrics[name].unit or
             candidate.metrics[name].unit != METRIC_UNITS[name] for name in METRIC_NAMES):
        reasons.append("metric_inventory_mismatch")
    if any(name in differences for name in SEMANTIC_PROVENANCE_FIELDS):
        reasons.append("semantic_provenance_mismatch")
    # Keep reason ordering deterministic even if future structural checks grow.
    reasons = list(dict.fromkeys(reasons))
    return CompatibilityResult(
        structurally_compatible=not reasons,
        capture_context_status=policy.capture_context,
        capture_policy_id=policy.policy_id,
        structural_reasons=tuple(reasons),
        provenance_differences=differences,
    )


# A descriptive alias keeps call sites readable without creating a second API.
assess_compatibility = check_compatibility


def _validate_metric_name(name: str) -> None:
    if name not in METRIC_NAMES:
        raise ComparisonError(f"unsupported metric name {name!r}.")


@dataclass(frozen=True)
class PairwiseMetric:
    unit: str
    candidate_value: float | None
    reference_value: float | None
    delta: float | None
    available: bool
    reason: str | None

    def __post_init__(self) -> None:
        _nonblank(self.unit, "metric unit")
        if type(self.available) is not bool:
            raise ComparisonError("pairwise metric available must be boolean.")
        if self.available:
            if self.reason is not None:
                raise ComparisonError("available pairwise metric reason must be null.")
            _finite(self.candidate_value, "candidate_value")
            _finite(self.reference_value, "reference_value")
            delta = _finite(self.delta, "delta")
            if not math.isclose(delta, float(self.candidate_value) - float(self.reference_value),
                                rel_tol=0.0, abs_tol=1e-12):
                raise ComparisonError("pairwise delta must equal candidate minus reference.")
            object.__setattr__(self, "candidate_value", float(self.candidate_value))
            object.__setattr__(self, "reference_value", float(self.reference_value))
            object.__setattr__(self, "delta", delta)
        else:
            if self.candidate_value is not None or self.reference_value is not None or self.delta is not None:
                raise ComparisonError("unavailable pairwise metric numeric fields must be null.")
            if self.reason not in PAIRWISE_MISSINGNESS_REASONS:
                raise ComparisonError("invalid pairwise metric reason.")

    def to_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "candidate_value": self.candidate_value,
                "reference_value": self.reference_value, "delta": self.delta,
                "available": self.available, "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseMetric":
        value = _strict_mapping(payload, {"unit", "candidate_value", "reference_value", "delta", "available", "reason"},
                                "pairwise metric")
        return cls(value["unit"], value["candidate_value"], value["reference_value"],
                   value["delta"], value["available"], value["reason"])


@dataclass(frozen=True)
class BaselineMetric:
    unit: str
    candidate_value: float | None
    usable_n: int
    median: float | None
    mad: float | None
    iqr: float | None
    percentile: float | None
    robust_deviation: float | None
    available: bool
    reason: str | None
    robust_deviation_available: bool
    robust_deviation_reason: str | None

    @property
    def robust_scale(self) -> float | None:
        """The deterministic 1.4826 * MAD scale, when descriptive values exist."""
        return None if self.mad is None else 1.4826 * self.mad

    def __post_init__(self) -> None:
        _nonblank(self.unit, "metric unit")
        if type(self.usable_n) is not int or isinstance(self.usable_n, bool) or self.usable_n < 0:
            raise ComparisonError("usable_n must be a non-negative integer.")
        if type(self.available) is not bool or type(self.robust_deviation_available) is not bool:
            raise ComparisonError("baseline metric availability must be boolean.")
        if self.available:
            if self.reason is not None or self.candidate_value is None or self.usable_n < 1:
                raise ComparisonError("available baseline metric has invalid availability fields.")
            for name in ("candidate_value", "median", "mad", "iqr", "percentile"):
                _finite(getattr(self, name), name)
            if float(self.mad) < 0.0 or float(self.iqr) < 0.0:
                raise ComparisonError("MAD and IQR must be non-negative.")
            if not 0.0 <= float(self.percentile) <= 100.0:
                raise ComparisonError("percentile must be between 0 and 100.")
        else:
            if self.reason not in BASELINE_DESCRIPTIVE_REASONS:
                raise ComparisonError("invalid baseline metric reason.")
            if any(getattr(self, name) is not None for name in ("candidate_value", "median", "mad", "iqr", "percentile")):
                raise ComparisonError("unavailable baseline descriptive values must be null.")
        if self.robust_deviation_available:
            if not self.available or self.usable_n < 5 or float(self.mad) <= 0.0:
                raise ComparisonError("available robust deviation requires an available qualified metric.")
            if self.robust_deviation_reason is not None:
                raise ComparisonError("available robust deviation reason must be null.")
            deviation = _finite(self.robust_deviation, "robust_deviation")
            expected = (float(self.candidate_value) - float(self.median)) / (1.4826 * float(self.mad))
            if not math.isclose(deviation, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ComparisonError("robust deviation does not match candidate, median, and MAD.")
        else:
            if self.robust_deviation is not None or self.robust_deviation_reason not in BASELINE_ROBUST_REASONS:
                raise ComparisonError("unavailable robust deviation fields are invalid.")
            expected_reason = self.reason if not self.available else (
                "insufficient_usable_n" if self.usable_n < 5 else
                "zero_mad" if float(self.mad) == 0.0 else None
            )
            if expected_reason is None or self.robust_deviation_reason != expected_reason:
                raise ComparisonError("robust deviation reason violates V1 precedence.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit, "candidate_value": self.candidate_value, "usable_n": self.usable_n,
            "median": self.median, "mad": self.mad, "iqr": self.iqr, "percentile": self.percentile,
            "robust_deviation": self.robust_deviation, "available": self.available, "reason": self.reason,
            "robust_deviation_available": self.robust_deviation_available,
            "robust_deviation_reason": self.robust_deviation_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BaselineMetric":
        keys = {"unit", "candidate_value", "usable_n", "median", "mad", "iqr", "percentile",
                "robust_deviation", "available", "reason", "robust_deviation_available", "robust_deviation_reason"}
        value = _strict_mapping(payload, keys, "baseline metric")
        return cls(*(value[name] for name in (
            "unit", "candidate_value", "usable_n", "median", "mad", "iqr", "percentile",
            "robust_deviation", "available", "reason", "robust_deviation_available", "robust_deviation_reason")))


@dataclass(frozen=True)
class ExcludedIdentity:
    identity: FingerprintIdentity
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, FingerprintIdentity):
            raise ComparisonError("excluded identity must be a FingerprintIdentity.")
        if self.reason not in BASELINE_EXCLUSION_REASONS:
            raise ComparisonError("invalid baseline exclusion reason.")

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity.to_dict(), "reason": self.reason}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExcludedIdentity":
        value = _strict_mapping(payload, {"identity", "reason"}, "excluded identity")
        return cls(FingerprintIdentity.from_dict(value["identity"]), value["reason"])


def _metrics_dict(metrics: Mapping[str, Any], value_type: type, name: str) -> dict[str, Any]:
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
        raise ComparisonError(f"{name} must contain exactly all V1 metrics.")
    result = {}
    for metric_name in METRIC_NAMES:
        value = metrics[metric_name]
        if not isinstance(value, value_type):
            raise ComparisonError(f"{name}[{metric_name!r}] has invalid type.")
        if value.unit != METRIC_UNITS[metric_name]:
            raise ComparisonError(f"{metric_name!r} has invalid unit.")
        result[metric_name] = value
    return result


@dataclass(frozen=True)
class ServePairwiseComparison:
    candidate_identity: FingerprintIdentity
    reference_identity: FingerprintIdentity
    compatibility: CompatibilityResult
    metrics: Mapping[str, PairwiseMetric]
    usable_metric_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_identity, FingerprintIdentity) or not isinstance(self.reference_identity, FingerprintIdentity):
            raise ComparisonError("pairwise identities must be FingerprintIdentity values.")
        if self.candidate_identity == self.reference_identity:
            raise ComparisonError("candidate and reference identities must differ.")
        if not isinstance(self.compatibility, CompatibilityResult):
            raise ComparisonError("compatibility must be a CompatibilityResult.")
        object.__setattr__(self, "metrics", _metrics_dict(self.metrics, PairwiseMetric, "pairwise metrics"))
        expected_count = sum(metric.available for metric in self.metrics.values())
        if type(self.usable_metric_count) is not int or self.usable_metric_count != expected_count:
            raise ComparisonError("usable_metric_count must equal the number of available metrics.")
        if not self.compatibility.compatible and expected_count:
            raise ComparisonError("incompatible pairwise comparisons cannot contain available metrics.")
        if self.compatibility.compatible and any(
            metric.reason == "comparison_incompatible" for metric in self.metrics.values()
        ):
            raise ComparisonError("compatible pairwise comparisons cannot use comparison_incompatible.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PAIRWISE_SCHEMA,
            "candidate_identity": self.candidate_identity.to_dict(),
            "reference_identity": self.reference_identity.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "usable_metric_count": self.usable_metric_count,
            "metrics": {name: self.metrics[name].to_dict() for name in METRIC_NAMES},
        }

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServePairwiseComparison":
        value = _strict_mapping(payload, {"schema", "candidate_identity", "reference_identity", "compatibility",
                                          "usable_metric_count", "metrics"},
                                "pairwise comparison")
        if value["schema"] != PAIRWISE_SCHEMA:
            raise ComparisonError("unsupported pairwise comparison schema.")
        raw = _strict_mapping(value["metrics"], set(METRIC_NAMES), "pairwise metrics")
        return cls(FingerprintIdentity.from_dict(value["candidate_identity"]),
                   FingerprintIdentity.from_dict(value["reference_identity"]),
                   CompatibilityResult.from_dict(value["compatibility"]),
                   {name: PairwiseMetric.from_dict(raw[name]) for name in METRIC_NAMES},
                   value["usable_metric_count"])

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> "ServePairwiseComparison":
        if isinstance(data, (bytes, bytearray)):
            data = bytes(data).decode("utf-8")
        if not isinstance(data, str):
            raise ComparisonError("JSON payload must be str or bytes.")
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ComparisonError(f"invalid comparison JSON: {exc}.") from exc
        if not isinstance(payload, dict):
            raise ComparisonError("comparison JSON must be an object.")
        return cls.from_dict(payload)


def compare_pairwise(
    candidate: ServeFingerprintV1,
    reference: ServeFingerprintV1,
    *,
    policy: ComparisonPolicy,
) -> ServePairwiseComparison:
    """Compare the fixed scalar metrics of one candidate and reference.

    Compatibility is a gate for every metric.  If the pair cannot be compared,
    the result still contains the complete fixed metric inventory, with each
    metric marked ``comparison_incompatible``.  For a compatible pair, metric
    availability is evaluated independently and available deltas always use
    the candidate-minus-reference convention.
    """

    if not isinstance(candidate, ServeFingerprintV1) or not isinstance(reference, ServeFingerprintV1):
        raise ComparisonError("candidate and reference must be ServeFingerprintV1 values.")
    if not isinstance(policy, ComparisonPolicy):
        raise ComparisonError("policy must be a ComparisonPolicy.")

    candidate_identity = FingerprintIdentity.from_fingerprint(candidate)
    reference_identity = FingerprintIdentity.from_fingerprint(reference)
    if candidate_identity == reference_identity:
        raise ComparisonError("candidate and reference identities must differ.")

    compatibility = check_compatibility(candidate, reference, policy=policy)
    metrics: dict[str, PairwiseMetric] = {}
    for name in METRIC_NAMES:
        unit = METRIC_UNITS[name]
        if not compatibility.compatible:
            metrics[name] = PairwiseMetric(unit, None, None, None, False, "comparison_incompatible")
            continue

        candidate_metric = candidate.metrics[name]
        reference_metric = reference.metrics[name]
        if candidate_metric.available and reference_metric.available:
            # MetricValue validates finite values when available.  Keep the
            # conversion here explicit so PairwiseMetric receives its public
            # float contract even when a caller supplied an integer value.
            candidate_value = float(candidate_metric.value)
            reference_value = float(reference_metric.value)
            metrics[name] = PairwiseMetric(
                unit,
                candidate_value,
                reference_value,
                candidate_value - reference_value,
                True,
                None,
            )
        elif not candidate_metric.available and not reference_metric.available:
            metrics[name] = PairwiseMetric(unit, None, None, None, False, "both_metrics_unavailable")
        elif not candidate_metric.available:
            metrics[name] = PairwiseMetric(unit, None, None, None, False, "candidate_metric_unavailable")
        else:
            metrics[name] = PairwiseMetric(unit, None, None, None, False, "reference_metric_unavailable")

    return ServePairwiseComparison(
        candidate_identity,
        reference_identity,
        compatibility,
        metrics,
        sum(metric.available for metric in metrics.values()),
    )


@dataclass(frozen=True)
class ServeBaselineComparison:
    candidate_identity: FingerprintIdentity
    cohort_id: str
    total_cohort_size: int
    eligible_cohort_size: int
    compatibility: CompatibilityResult
    metrics: Mapping[str, BaselineMetric]
    excluded_identities: tuple[ExcludedIdentity, ...] = ()
    ranking: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_identity, FingerprintIdentity):
            raise ComparisonError("candidate_identity must be a FingerprintIdentity.")
        _nonblank(self.cohort_id, "cohort_id")
        for name, value in (("total_cohort_size", self.total_cohort_size), ("eligible_cohort_size", self.eligible_cohort_size)):
            if type(value) is not int or value < 0:
                raise ComparisonError(f"{name} must be a non-negative integer.")
        if self.eligible_cohort_size > self.total_cohort_size:
            raise ComparisonError("eligible_cohort_size cannot exceed total_cohort_size.")
        if not isinstance(self.compatibility, CompatibilityResult):
            raise ComparisonError("compatibility must be a CompatibilityResult.")
        object.__setattr__(self, "metrics", _metrics_dict(self.metrics, BaselineMetric, "baseline metrics"))
        exclusions = tuple(self.excluded_identities)
        if any(not isinstance(item, ExcludedIdentity) for item in exclusions):
            raise ComparisonError("excluded_identities must contain ExcludedIdentity values.")
        object.__setattr__(self, "excluded_identities", exclusions)
        if len({item.identity for item in exclusions}) != len(exclusions):
            raise ComparisonError("excluded identities must not contain duplicates.")
        if self.total_cohort_size != self.eligible_cohort_size + len(exclusions):
            raise ComparisonError("cohort counts must equal eligible plus excluded identities.")
        ranking = _tuple_strings(self.ranking, "ranking", METRIC_NAMES)
        if len(set(ranking)) != len(ranking):
            raise ComparisonError("ranking must not contain duplicates.")
        expected_ranking = tuple(sorted(
            (name for name in METRIC_NAMES if self.metrics[name].robust_deviation_available),
            key=lambda name: (-abs(float(self.metrics[name].robust_deviation)), METRIC_NAMES.index(name)),
        ))
        if ranking != expected_ranking:
            raise ComparisonError("ranking must contain every available robust deviation in deterministic order.")
        object.__setattr__(self, "ranking", ranking)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_SCHEMA,
            "candidate_identity": self.candidate_identity.to_dict(),
            "cohort_id": self.cohort_id,
            "total_cohort_size": self.total_cohort_size,
            "eligible_cohort_size": self.eligible_cohort_size,
            "compatibility": self.compatibility.to_dict(),
            "excluded_identities": [item.to_dict() for item in self.excluded_identities],
            "ranking": list(self.ranking),
            "metrics": {name: self.metrics[name].to_dict() for name in METRIC_NAMES},
        }

    def to_json(self) -> str:
        return _json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServeBaselineComparison":
        value = _strict_mapping(payload, {"schema", "candidate_identity", "cohort_id", "total_cohort_size",
                                           "eligible_cohort_size", "compatibility", "excluded_identities", "ranking", "metrics"},
                                "baseline comparison")
        if value["schema"] != BASELINE_SCHEMA:
            raise ComparisonError("unsupported baseline comparison schema.")
        raw_metrics = _strict_mapping(value["metrics"], set(METRIC_NAMES), "baseline metrics")
        if not isinstance(value["excluded_identities"], list):
            raise ComparisonError("excluded_identities must be a list.")
        return cls(FingerprintIdentity.from_dict(value["candidate_identity"]), value["cohort_id"],
                   value["total_cohort_size"], value["eligible_cohort_size"],
                   CompatibilityResult.from_dict(value["compatibility"]),
                   {name: BaselineMetric.from_dict(raw_metrics[name]) for name in METRIC_NAMES},
                   tuple(ExcludedIdentity.from_dict(item) for item in value["excluded_identities"]),
                   tuple(value["ranking"]))

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> "ServeBaselineComparison":
        if isinstance(data, (bytes, bytearray)):
            data = bytes(data).decode("utf-8")
        if not isinstance(data, str):
            raise ComparisonError("JSON payload must be str or bytes.")
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ComparisonError(f"invalid comparison JSON: {exc}.") from exc
        if not isinstance(payload, dict):
            raise ComparisonError("comparison JSON must be an object.")
        return cls.from_dict(payload)


# The helpers below deliberately operate on finite values and make no use of
# statistics-library defaults.  In particular, quantile interpolation and
# equality for the percentile are part of the persisted comparison contract.
def _sorted_values(values: Sequence[float], name: str = "values") -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ComparisonError(f"{name} must be a sequence of finite numbers.")
    try:
        result = tuple(_finite(value, f"{name} value") for value in values)
    except TypeError as exc:
        raise ComparisonError(f"{name} must be a sequence of finite numbers.") from exc
    return tuple(sorted(result))


def median(values: Sequence[float]) -> float:
    """Return the exact V1 median of a non-empty finite sequence."""
    ordered = _sorted_values(values)
    if not ordered:
        raise ComparisonError("median requires at least one value.")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def mad(values: Sequence[float], center: float | None = None) -> float:
    """Return median absolute deviation around the supplied (or V1) median."""
    ordered = _sorted_values(values)
    if not ordered:
        raise ComparisonError("mad requires at least one value.")
    midpoint = median(ordered) if center is None else _finite(center, "center")
    return median(tuple(abs(value - midpoint) for value in ordered))


def robust_scale(mad_value: float) -> float:
    """Return the fixed V1 robust scale multiplier times MAD."""
    return 1.4826 * _finite(mad_value, "mad")


def linear_quantile(values: Sequence[float], probability: float) -> float:
    """Return V1's type-7/linear empirical quantile for ``probability``."""
    ordered = _sorted_values(values)
    if not ordered:
        raise ComparisonError("linear_quantile requires at least one value.")
    probability = _finite(probability, "probability")
    if not 0.0 <= probability <= 1.0:
        raise ComparisonError("probability must be between 0 and 1.")
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * probability
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    if lower == upper:
        return ordered[lower]
    fraction = h - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def interquartile_range(values: Sequence[float]) -> float:
    """Return the V1 linear empirical IQR."""
    return linear_quantile(values, 0.75) - linear_quantile(values, 0.25)


def midrank_percentile(values: Sequence[float], candidate: float) -> float:
    """Return V1's exact-equality empirical midrank percentile in percent."""
    ordered = _sorted_values(values)
    if not ordered:
        raise ComparisonError("midrank_percentile requires at least one value.")
    candidate = _finite(candidate, "candidate")
    less = sum(value < candidate for value in ordered)
    equal = sum(value == candidate for value in ordered)
    return 100.0 * (less + 0.5 * equal) / len(ordered)


# Explicit aliases make the deterministic nature apparent at call sites while
# retaining concise names for callers that use these as ordinary helpers.
deterministic_median = median
deterministic_mad = mad
deterministic_linear_quantile = linear_quantile
deterministic_iqr = interquartile_range
deterministic_midrank_percentile = midrank_percentile


def _baseline_compatibility(
    candidate: ServeFingerprintV1,
    members: Sequence[ServeFingerprintV1],
    *,
    policy: ComparisonPolicy,
    candidate_identity: FingerprintIdentity,
) -> tuple[CompatibilityResult, list[tuple[ServeFingerprintV1, CompatibilityResult]]]:
    """Assess every member and select a stable representative result.

    A baseline can contain a mixture of compatible and excluded records.  Its
    top-level compatibility therefore describes the first eligible member (or
    the first supplied member when all are excluded), while each excluded
    identity retains the stable ``comparison_incompatible`` reason.
    """
    assessments = [
        (member, check_compatibility(candidate, member, policy=policy))
        for member in members
    ]
    non_candidate = [
        item for member, item in assessments
        if FingerprintIdentity.from_fingerprint(member) != candidate_identity
    ]
    eligible = [item for item in non_candidate if item.compatible]
    representative = eligible[0] if eligible else (non_candidate[0] if non_candidate else assessments[0][1])
    return representative, assessments


def compare_to_baseline(
    candidate: ServeFingerprintV1,
    cohort: Sequence[ServeFingerprintV1],
    *,
    cohort_id: str,
    policy: ComparisonPolicy,
) -> ServeBaselineComparison:
    """Compare one candidate against a caller-supplied, leave-one-out cohort.

    Identity validation happens before any filtering, so malformed duplicate
    input cannot be hidden by candidate exclusion or compatibility filtering.
    """
    if not isinstance(candidate, ServeFingerprintV1):
        raise ComparisonError("candidate must be a ServeFingerprintV1.")
    if not isinstance(policy, ComparisonPolicy):
        raise ComparisonError("policy must be a ComparisonPolicy.")
    _nonblank(cohort_id, "cohort_id")
    if isinstance(cohort, (str, bytes)) or not isinstance(cohort, Sequence):
        raise ComparisonError("cohort must be a sequence of ServeFingerprintV1 values.")
    if not cohort:
        raise ComparisonError("cohort must contain at least one fingerprint.")
    if any(not isinstance(member, ServeFingerprintV1) for member in cohort):
        raise ComparisonError("cohort must contain only ServeFingerprintV1 values.")

    identities = [FingerprintIdentity.from_fingerprint(member) for member in cohort]
    seen: set[FingerprintIdentity] = set()
    for identity in identities:
        if identity in seen:
            raise ComparisonError(f"duplicate cohort identity: {identity!r}.")
        seen.add(identity)

    candidate_identity = FingerprintIdentity.from_fingerprint(candidate)
    if all(identity == candidate_identity for identity in identities):
        raise ComparisonError(
            "baseline cohort must contain at least one non-candidate fingerprint after leave-one-out."
        )
    top_compatibility, assessments = _baseline_compatibility(
        candidate, cohort, policy=policy, candidate_identity=candidate_identity,
    )
    excluded: list[ExcludedIdentity] = []
    eligible_members: list[ServeFingerprintV1] = []
    for member, identity, compatibility in zip(cohort, identities, (item[1] for item in assessments)):
        if identity == candidate_identity:
            excluded.append(ExcludedIdentity(identity, "candidate_identity_leave_one_out"))
        elif compatibility.compatible:
            eligible_members.append(member)
        else:
            if compatibility.structural_reasons:
                reason = compatibility.structural_reasons[0]
            elif compatibility.capture_context_status == "not_asserted":
                reason = "capture_context_not_asserted"
            else:
                reason = "capture_context_asserted_incompatible"
            excluded.append(ExcludedIdentity(identity, reason))

    metrics: dict[str, BaselineMetric] = {}
    baseline_has_eligible = bool(eligible_members)
    for name in METRIC_NAMES:
        unit = METRIC_UNITS[name]
        candidate_metric = candidate.metrics[name]
        values = [float(member.metrics[name].value) for member in eligible_members
                  if member.metrics[name].available]

        # Availability and robust-deviation reason precedence is intentionally
        # explicit rather than relying on a chain of incidental null checks.
        if not baseline_has_eligible:
            descriptive_reason = "comparison_incompatible"
        elif not candidate_metric.available:
            descriptive_reason = "candidate_metric_unavailable"
        elif not values:
            descriptive_reason = "no_usable_cohort_values"
        else:
            descriptive_reason = None

        if descriptive_reason is None:
            candidate_value = float(candidate_metric.value)
            ordered = tuple(sorted(values))
            center = median(ordered)
            mad_value = mad(ordered, center)
            iqr_value = interquartile_range(ordered)
            percentile = midrank_percentile(ordered, candidate_value)
            scale = robust_scale(mad_value)
            if len(ordered) < 5:
                robust_available, robust_reason, deviation = False, "insufficient_usable_n", None
            elif mad_value == 0.0:
                robust_available, robust_reason, deviation = False, "zero_mad", None
            else:
                robust_available, robust_reason = True, None
                deviation = (candidate_value - center) / scale
            metrics[name] = BaselineMetric(
                unit, candidate_value, len(ordered), center, mad_value, iqr_value,
                percentile, deviation, True, None, robust_available, robust_reason,
            )
        else:
            metrics[name] = BaselineMetric(
                unit, None, len(values), None, None, None, None, None,
                False, descriptive_reason, False, descriptive_reason,
            )

    ranking = tuple(
        sorted(
            (name for name in METRIC_NAMES if metrics[name].robust_deviation_available),
            key=lambda name: (-abs(float(metrics[name].robust_deviation)), METRIC_NAMES.index(name)),
        )
    )
    return ServeBaselineComparison(
        candidate_identity,
        cohort_id,
        len(cohort),
        len(eligible_members),
        top_compatibility,
        metrics,
        tuple(excluded),
        ranking,
    )
