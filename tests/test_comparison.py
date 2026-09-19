from __future__ import annotations

import copy
import json

import pytest
from click.testing import CliRunner

from serve_review.comparison import (
    CAPTURE_CONTEXTS,
    compare_to_baseline,
    compare_pairwise,
    SEMANTIC_PROVENANCE_POLICY,
    ComparisonError,
    ComparisonPolicy,
    FingerprintIdentity,
    PairwiseMetric,
    BaselineMetric,
    ExcludedIdentity,
    ServeBaselineComparison,
    ServePairwiseComparison,
    check_compatibility,
    interquartile_range,
    linear_quantile,
    mad,
    median,
    midrank_percentile,
    robust_scale,
    load_fingerprint,
)
from serve_review.cli.main import cli
from serve_review.fingerprint import (
    ANCHOR_NAMES,
    CHANNEL_NAMES,
    METRIC_UNITS,
    METRIC_NAMES,
    SEGMENT_LAYOUT,
    AnchorValue,
    MetricValue,
    SequenceSegment,
    ServeFingerprintV1,
)


def _fingerprint(*, source: str = "sha256:one", waveform_config: str = "waveform-v1") -> ServeFingerprintV1:
    anchors = {name: AnchorValue(True, float(i)) for i, name in enumerate(ANCHOR_NAMES)}
    metrics = {name: MetricValue(False, None, unit) for name, unit in METRIC_UNITS.items()}
    segments = tuple(
        SequenceSegment(
            id=name,
            available=False,
            duration_metric=f"{name}_duration",
            values=tuple(tuple(None for _ in CHANNEL_NAMES) for _ in range(16)),
            availability=tuple(tuple(False for _ in CHANNEL_NAMES) for _ in range(16)),
        )
        for name, _, _ in SEGMENT_LAYOUT
    )
    return ServeFingerprintV1(
        source_fingerprint=source,
        attempt_start_seconds=0.0,
        attempt_end_seconds=8.0,
        provenance={
            "coordinate_convention": "world-hip-v1",
            "waveform_method_version": "waveform-method-v1",
            "waveform_config_id": waveform_config,
            "checkpoint_method_version": "checkpoint-method-v1",
            "checkpoint_config_id": "checkpoint-config-v1",
            "body_model_name": "pose",
            "body_model_version": "1",
        },
        anchors=anchors,
        metrics=metrics,
        segments=segments,
    )


def test_policy_and_identity_are_frozen_and_validate_contract_values() -> None:
    policy = ComparisonPolicy("personal-session-2026-09-15", "asserted_compatible")
    assert policy.semantic_provenance == SEMANTIC_PROVENANCE_POLICY
    assert set(CAPTURE_CONTEXTS) == {"asserted_compatible", "not_asserted", "asserted_incompatible"}
    assert FingerprintIdentity.from_fingerprint(_fingerprint()).to_dict()["source_fingerprint"] == "sha256:one"
    with pytest.raises(ComparisonError):
        ComparisonPolicy("audit", "asserted_compatible", "unknown-policy")
    with pytest.raises(ComparisonError):
        FingerprintIdentity("sha256:x", 2.0, 1.0)


def test_semantic_provenance_mismatch_is_hard_gate_and_other_provenance_is_diagnostic() -> None:
    candidate = _fingerprint()
    reference = _fingerprint(waveform_config="waveform-v2")
    policy = ComparisonPolicy("session", "asserted_compatible")
    result = check_compatibility(candidate, reference, policy=policy)
    assert not result.structurally_compatible
    assert not result.compatible
    assert "semantic_provenance_mismatch" in result.structural_reasons
    assert result.provenance_differences["waveform_config_id"] == ("waveform-v1", "waveform-v2")

    payload = copy.deepcopy(candidate.to_dict())
    payload["provenance"]["waveform_method_version"] = "waveform-method-v2"
    changed = ServeFingerprintV1.from_dict(payload)
    diagnostic = check_compatibility(candidate, changed, policy=policy)
    assert not diagnostic.structurally_compatible
    assert diagnostic.provenance_differences["waveform_method_version"] == ("waveform-method-v1", "waveform-method-v2")


def test_capture_assertion_is_separate_from_structural_compatibility() -> None:
    candidate = _fingerprint()
    reference = _fingerprint(source="sha256:two")
    not_asserted = check_compatibility(
        candidate, reference, policy=ComparisonPolicy("audit", "not_asserted")
    )
    assert not not_asserted.compatible
    assert not_asserted.structurally_compatible
    incompatible = check_compatibility(
        candidate, reference, policy=ComparisonPolicy("audit", "asserted_incompatible")
    )
    assert not incompatible.compatible
    assert incompatible.capture_policy_id == "audit"


def test_pairwise_metric_enforces_candidate_minus_reference_and_round_trip() -> None:
    metric = PairwiseMetric("seconds", 0.31, 0.346, -0.036, True, None)
    assert PairwiseMetric.from_dict(metric.to_dict()) == metric
    with pytest.raises(ComparisonError):
        PairwiseMetric("seconds", 0.31, 0.346, 0.036, True, None)


def test_pairwise_result_contains_fixed_metric_inventory_and_deterministic_json() -> None:
    candidate = FingerprintIdentity.from_fingerprint(_fingerprint())
    reference = FingerprintIdentity.from_fingerprint(_fingerprint(source="sha256:two"))
    compatibility = check_compatibility(
        _fingerprint(), _fingerprint(source="sha256:two"),
        policy=ComparisonPolicy("audit", "asserted_compatible"),
    )
    metrics = {
        name: PairwiseMetric(unit, None, None, None, False, "both_metrics_unavailable")
        for name, unit in METRIC_UNITS.items()
    }
    result = ServePairwiseComparison(candidate, reference, compatibility, metrics, 0)
    assert tuple(result.metrics) == tuple(METRIC_UNITS)
    assert ServePairwiseComparison.from_json(result.to_json()) == result


def _with_metrics(fingerprint: ServeFingerprintV1, metrics: dict[str, MetricValue]) -> ServeFingerprintV1:
    return ServeFingerprintV1(
        source_fingerprint=fingerprint.source_fingerprint,
        attempt_start_seconds=fingerprint.attempt_start_seconds,
        attempt_end_seconds=fingerprint.attempt_end_seconds,
        provenance=fingerprint.provenance,
        anchors=fingerprint.anchors,
        metrics=metrics,
        segments=fingerprint.segments,
        config_id=fingerprint.config_id,
    )


def test_compare_pairwise_returns_all_fixed_metrics_and_candidate_minus_reference_deltas() -> None:
    candidate = _fingerprint()
    reference = _fingerprint(source="sha256:two")
    candidate = _with_metrics(
        candidate,
        {name: MetricValue(True, float(index + 1), unit) for index, (name, unit) in enumerate(METRIC_UNITS.items())},
    )
    reference = _with_metrics(
        reference,
        {name: MetricValue(True, float(index), unit) for index, (name, unit) in enumerate(METRIC_UNITS.items())},
    )

    result = compare_pairwise(candidate, reference, policy=ComparisonPolicy("audit", "asserted_compatible"))

    assert tuple(result.metrics) == METRIC_NAMES
    assert tuple(result.metrics) == tuple(METRIC_UNITS)
    for index, name in enumerate(METRIC_NAMES):
        metric = result.metrics[name]
        assert metric.unit == METRIC_UNITS[name]
        assert metric.available
        assert metric.candidate_value == float(index + 1)
        assert metric.reference_value == float(index)
        assert metric.delta == 1.0
        assert metric.reason is None


def test_compare_pairwise_accepts_deterministically_serialized_fingerprints() -> None:
    candidate = ServeFingerprintV1.from_json(_fingerprint(source="candidate").to_json())
    reference = ServeFingerprintV1.from_json(_fingerprint(source="reference").to_json())

    result = compare_pairwise(
        candidate,
        reference,
        policy=ComparisonPolicy("serialized", "asserted_compatible"),
    )

    assert result.compatibility.compatible
    assert "metric_inventory_mismatch" not in result.compatibility.structural_reasons


def test_compare_pairwise_applies_exact_metric_missingness_reasons() -> None:
    candidate = _fingerprint()
    reference = _fingerprint(source="sha256:two")
    names = list(METRIC_NAMES)
    candidate_metrics = dict(candidate.metrics)
    reference_metrics = dict(reference.metrics)
    candidate_metrics[names[0]] = MetricValue(True, 1.0, METRIC_UNITS[names[0]])
    reference_metrics[names[0]] = MetricValue(True, 0.5, METRIC_UNITS[names[0]])
    candidate_metrics[names[1]] = MetricValue(True, 1.0, METRIC_UNITS[names[1]])
    reference_metrics[names[1]] = MetricValue(False, None, METRIC_UNITS[names[1]])
    candidate_metrics[names[2]] = MetricValue(False, None, METRIC_UNITS[names[2]])
    reference_metrics[names[2]] = MetricValue(True, 0.5, METRIC_UNITS[names[2]])
    candidate = _with_metrics(candidate, candidate_metrics)
    reference = _with_metrics(reference, reference_metrics)

    result = compare_pairwise(candidate, reference, policy=ComparisonPolicy("audit", "asserted_compatible"))

    assert result.metrics[names[0]].available
    assert result.metrics[names[1]].reason == "reference_metric_unavailable"
    assert result.metrics[names[2]].reason == "candidate_metric_unavailable"
    for name in names[3:]:
        assert result.metrics[name].reason == "both_metrics_unavailable"
    for name in names[1:]:
        metric = result.metrics[name]
        assert metric.candidate_value is None
        assert metric.reference_value is None
        assert metric.delta is None


def test_compare_pairwise_incompatibility_marks_every_metric_and_rejects_self_comparison() -> None:
    candidate = _fingerprint()
    reference = _fingerprint(source="sha256:two")
    result = compare_pairwise(candidate, reference, policy=ComparisonPolicy("audit", "not_asserted"))
    assert not result.compatibility.compatible
    assert all(not metric.available for metric in result.metrics.values())
    assert all(metric.reason == "comparison_incompatible" for metric in result.metrics.values())

    with pytest.raises(ComparisonError):
        compare_pairwise(candidate, _fingerprint(), policy=ComparisonPolicy("audit", "asserted_compatible"))


def _with_one_metric(fingerprint: ServeFingerprintV1, name: str, value: float | None) -> ServeFingerprintV1:
    metrics = dict(fingerprint.metrics)
    metrics[name] = MetricValue(value is not None, value, METRIC_UNITS[name])
    return _with_metrics(fingerprint, metrics)


def test_baseline_rejects_duplicate_identities_and_reports_leave_one_out_and_incompatibility() -> None:
    candidate = _fingerprint(source="candidate")
    compatible = _fingerprint(source="compatible")
    incompatible = _fingerprint(source="incompatible", waveform_config="waveform-v2")
    policy = ComparisonPolicy("audit", "asserted_compatible")

    with pytest.raises(ComparisonError, match="duplicate cohort identity"):
        compare_to_baseline(candidate, [compatible, compatible], cohort_id="cohort", policy=policy)
    with pytest.raises(ComparisonError, match="non-candidate fingerprint"):
        compare_to_baseline(candidate, [candidate], cohort_id="cohort", policy=policy)

    result = compare_to_baseline(
        candidate, [candidate, compatible, incompatible], cohort_id="cohort", policy=policy
    )
    assert result.total_cohort_size == 3
    assert result.eligible_cohort_size == 1
    assert [(item.identity.source_fingerprint, item.reason) for item in result.excluded_identities] == [
        ("candidate", "candidate_identity_leave_one_out"),
        ("incompatible", "semantic_provenance_mismatch"),
    ]


def test_baseline_exact_statistics_and_midrank_percentile() -> None:
    name = METRIC_NAMES[0]
    candidate = _with_one_metric(_fingerprint(source="candidate"), name, 7.0)
    cohort = [_with_one_metric(_fingerprint(source=f"member-{value}"), name, float(value))
              for value in range(1, 7)]
    result = compare_to_baseline(
        candidate, cohort, cohort_id="cohort", policy=ComparisonPolicy("audit", "asserted_compatible")
    )
    metric = result.metrics[name]
    assert metric.usable_n == 6
    assert metric.median == 3.5
    assert metric.mad == 1.5
    assert metric.robust_scale == 1.4826 * 1.5
    assert metric.iqr == 2.5
    assert metric.percentile == 100.0
    assert metric.robust_deviation == pytest.approx((7.0 - 3.5) / (1.4826 * 1.5))
    assert median([1, 2, 3, 4]) == 2.5
    assert mad([1, 2, 3, 4]) == 1.0
    assert robust_scale(2.0) == 2.9652
    assert linear_quantile([1, 2, 3, 4], 0.25) == 1.75
    assert interquartile_range([1, 2, 3, 4]) == 1.5
    assert midrank_percentile([1, 2, 2, 4], 2) == 50.0


def test_baseline_reason_precedence_and_robust_edges() -> None:
    name = METRIC_NAMES[0]
    policy = ComparisonPolicy("audit", "asserted_compatible")
    candidate = _with_one_metric(_fingerprint(source="candidate"), name, 3.0)

    short = compare_to_baseline(
        candidate,
        [_with_one_metric(_fingerprint(source=f"member-{i}"), name, float(i)) for i in range(4)],
        cohort_id="short",
        policy=policy,
    ).metrics[name]
    assert short.available and short.robust_deviation_reason == "insufficient_usable_n"

    zero_mad = compare_to_baseline(
        candidate,
        [_with_one_metric(_fingerprint(source=f"member-{i}"), name, 2.0) for i in range(5)],
        cohort_id="zero",
        policy=policy,
    ).metrics[name]
    assert zero_mad.available and zero_mad.robust_deviation_reason == "zero_mad"

    unavailable_candidate = _with_one_metric(_fingerprint(source="missing"), name, None)
    no_cohort_values = compare_to_baseline(
        unavailable_candidate,
        [_with_one_metric(_fingerprint(source="member"), name, None)],
        cohort_id="missing",
        policy=policy,
    ).metrics[name]
    assert no_cohort_values.reason == "candidate_metric_unavailable"
    assert no_cohort_values.robust_deviation_reason == "candidate_metric_unavailable"

    incompatible = compare_to_baseline(
        candidate,
        [_fingerprint(source="bad", waveform_config="waveform-v2")],
        cohort_id="bad",
        policy=policy,
    ).metrics[name]
    assert incompatible.reason == "comparison_incompatible"
    assert incompatible.robust_deviation_reason == "comparison_incompatible"


def test_baseline_ranking_and_json_are_deterministic() -> None:
    candidate = _fingerprint(source="candidate")
    cohort = [_fingerprint(source=f"member-{i}") for i in range(5)]
    names = METRIC_NAMES[:2]
    candidate_metrics = dict(candidate.metrics)
    member_metrics = [dict(member.metrics) for member in cohort]
    for name in names:
        candidate_metrics[name] = MetricValue(True, 5.0, METRIC_UNITS[name])
        for index, metrics in enumerate(member_metrics):
            metrics[name] = MetricValue(True, float(index + 1), METRIC_UNITS[name])
    candidate = _with_metrics(candidate, candidate_metrics)
    cohort = [_with_metrics(member, metrics) for member, metrics in zip(cohort, member_metrics)]
    result = compare_to_baseline(
        candidate, cohort, cohort_id="cohort", policy=ComparisonPolicy("audit", "asserted_compatible")
    )
    assert result.ranking[:2] == names
    assert ServeBaselineComparison.from_json(result.to_json()) == result


def test_baseline_schema_rejects_impossible_robust_state_exclusion_and_ranking() -> None:
    with pytest.raises(ComparisonError):
        BaselineMetric(
            "seconds", None, 0, None, None, None, None, 1.0,
            False, "comparison_incompatible", True, None,
        )
    with pytest.raises(ComparisonError):
        ExcludedIdentity(FingerprintIdentity("source", 0.0, 1.0), "made_up")

    candidate = _fingerprint(source="candidate")
    cohort = [_with_one_metric(_fingerprint(source=f"member-{i}"), METRIC_NAMES[0], float(i))
              for i in range(5)]
    candidate = _with_one_metric(candidate, METRIC_NAMES[0], 9.0)
    result = compare_to_baseline(
        candidate, cohort, cohort_id="cohort",
        policy=ComparisonPolicy("audit", "asserted_compatible"),
    )
    payload = result.to_dict()
    payload["ranking"] = []
    with pytest.raises(ComparisonError):
        ServeBaselineComparison.from_dict(payload)


def test_loader_and_development_cli_emit_json_and_report_paths(tmp_path) -> None:
    candidate_path = tmp_path / "candidate.json"
    reference_path = tmp_path / "reference.json"
    candidate_path.write_text(_fingerprint(source="candidate").to_json(), encoding="utf-8")
    reference_path.write_text(_fingerprint(source="reference").to_json(), encoding="utf-8")

    assert load_fingerprint(candidate_path).source_fingerprint == "candidate"
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ComparisonError, match="bad.json"):
        load_fingerprint(bad_path)

    runner = CliRunner()
    pairwise = runner.invoke(cli, [
        "dev", "compare-pairwise", str(candidate_path), str(reference_path),
        "--policy-id", "smoke", "--capture-context", "asserted_compatible",
    ])
    assert pairwise.exit_code == 0, pairwise.output
    assert json.loads(pairwise.output)["schema"] == "serve-pairwise-comparison-v1"

    baseline = runner.invoke(cli, [
        "dev", "compare-baseline", str(candidate_path), str(reference_path),
        "--cohort-id", "tiny", "--policy-id", "smoke",
        "--capture-context", "asserted_compatible",
    ])
    assert baseline.exit_code == 0, baseline.output
    assert json.loads(baseline.output)["schema"] == "serve-baseline-comparison-v1"

    invalid = runner.invoke(cli, [
        "dev", "compare-pairwise", str(bad_path), str(reference_path),
        "--policy-id", "smoke", "--capture-context", "asserted_compatible",
    ])
    assert invalid.exit_code == 2
    assert "bad.json" in invalid.output
