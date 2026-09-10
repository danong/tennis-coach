"""Focused M3.4 tests for evaluation matching and reports.

Deterministic, offline, synthetic only: every range is hand-constructed
with hand-calculated expectations. No private footage, manifests, or
threshold tuning. Fixture manifests under ``tests/fixtures/annotations/``
are tiny synthetic examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve_review.domain import MediaRange
from serve_review.evaluation import (
    ANNOTATION_MANIFEST_SCHEMA_VERSION,
    EVALUATION_REPORT_SCHEMA_VERSION,
    IOU_THRESHOLD,
    Annotation,
    AnnotationManifest,
    EvaluationError,
    EvaluationReport,
    MatchEntry,
    OverallMetrics,
    RangeMetrics,
    StratumInput,
    StratumReport,
    evaluate_manifest,
    evaluate_ranges,
    evaluate_report,
    group_ambiguous_by_session,
    group_truth_by_session,
    intersection_seconds,
    iou,
    match_ranges,
    validate_session_disjoint,
)

FIXTURES = Path(__file__).parent / "fixtures" / "annotations"


def rng(start: float, end: float) -> MediaRange:
    return MediaRange(start_seconds=start, end_seconds=end)


def ann(
    session: str,
    media: str,
    start: float,
    end: float,
    label: str = "serve",
) -> Annotation:
    return Annotation(
        session_id=session,
        media=media,
        start_seconds=start,
        end_seconds=end,
        label=label,
    )


def manifest(split: str, entries: list[Annotation]) -> AnnotationManifest:
    return AnnotationManifest(
        schema_version=ANNOTATION_MANIFEST_SCHEMA_VERSION,
        dataset_split=split,
        annotations=tuple(entries),
    )


# --- IoU hand calculations --------------------------------------------------


def test_iou_threshold_is_pinned() -> None:
    assert IOU_THRESHOLD == 0.5


def test_iou_hand_calculated() -> None:
    assert iou(rng(10.0, 12.0), rng(10.0, 12.0)) == pytest.approx(1.0)
    assert iou(rng(0.0, 2.0), rng(5.0, 7.0)) == pytest.approx(0.0)
    # Adjacent half-open ranges do not overlap.
    assert iou(rng(0.0, 2.0), rng(2.0, 4.0)) == pytest.approx(0.0)
    assert intersection_seconds(rng(0.0, 2.0), rng(2.0, 4.0)) == pytest.approx(0.0)
    # Partial: inter 1.5, union 2.5 -> 0.6.
    assert iou(rng(10.0, 12.0), rng(10.5, 12.5)) == pytest.approx(0.6)
    # Containment: inter 2.0, union 4.0 -> 0.5.
    assert iou(rng(0.0, 4.0), rng(1.0, 3.0)) == pytest.approx(0.5)
    # Exact boundary: inter 1.0, union 2.0 -> 0.5.
    assert iou(rng(0.0, 2.0), rng(0.0, 1.0)) == pytest.approx(0.5)


def test_iou_rejects_non_ranges() -> None:
    with pytest.raises(EvaluationError):
        iou("nope", rng(0.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        intersection_seconds(rng(0.0, 1.0), None)  # type: ignore[arg-type]


# --- One-to-one matching ----------------------------------------------------


def test_matching_hand_calculated() -> None:
    # T0=[10,12] (dur 2), T1=[20,23] (dur 3).
    # P0=[10.5,12.5] IoU 0.6 with T0; P2=[20.2,22.8] IoU 2.6/3 with T1;
    # P1=[30,31] matches nothing.
    truth = [rng(10.0, 12.0), rng(20.0, 23.0)]
    preds = [rng(10.5, 12.5), rng(30.0, 31.0), rng(20.2, 22.8)]
    matches = match_ranges(truth, preds)
    assert [(m.truth_index, m.pred_index) for m in matches] == [(0, 0), (1, 2)]
    assert matches[0].iou == pytest.approx(0.6)
    assert matches[1].iou == pytest.approx(2.6 / 3.0)
    assert matches[0].onset_error_seconds == pytest.approx(0.5)
    assert matches[0].end_error_seconds == pytest.approx(0.5)
    assert matches[1].onset_error_seconds == pytest.approx(0.2)
    assert matches[1].end_error_seconds == pytest.approx(-0.2)
    assert matches[0].intersection_seconds == pytest.approx(1.5)
    assert matches[1].intersection_seconds == pytest.approx(2.6)


def test_matching_is_one_to_one() -> None:
    # One truth, two identical predictions: only the lower pred index wins.
    matches = match_ranges([rng(0.0, 2.0)], [rng(0.0, 2.0), rng(0.0, 2.0)])
    assert len(matches) == 1
    assert (matches[0].truth_index, matches[0].pred_index) == (0, 0)
    # One prediction, two identical truths: only the lower truth index wins.
    matches = match_ranges([rng(0.0, 2.0), rng(0.0, 2.0)], [rng(0.0, 2.0)])
    assert len(matches) == 1
    assert (matches[0].truth_index, matches[0].pred_index) == (0, 0)


def test_tie_breaking_is_deterministic() -> None:
    # Two truths and two predictions with symmetric equal-IoU pairings:
    # greedy order must resolve by (truth, pred) index, not input luck.
    truth = [rng(0.0, 2.0), rng(10.0, 12.0)]
    preds = [rng(10.0, 12.0), rng(0.0, 2.0)]
    first = match_ranges(truth, preds)
    second = match_ranges(truth, preds)
    assert first == second
    assert [(m.truth_index, m.pred_index) for m in first] == [(0, 1), (1, 0)]
    for entry in first:
        assert entry.iou == pytest.approx(1.0)


def test_threshold_boundary_is_inclusive() -> None:
    # IoU exactly 0.5 matches at the default threshold.
    assert match_ranges([rng(0.0, 2.0)], [rng(0.0, 1.0)])
    # The same pair misses under a stricter threshold.
    assert match_ranges([rng(0.0, 2.0)], [rng(0.0, 1.0)], iou_threshold=0.51) == ()
    # A weak overlap never matches.
    assert match_ranges([rng(0.0, 4.0)], [rng(0.0, 1.0)]) == ()


def test_matching_validates_threshold() -> None:
    with pytest.raises(EvaluationError):
        match_ranges([rng(0.0, 1.0)], [rng(0.0, 1.0)], iou_threshold=0.0)
    with pytest.raises(EvaluationError):
        match_ranges([rng(0.0, 1.0)], [rng(0.0, 1.0)], iou_threshold=1.5)
    with pytest.raises(EvaluationError):
        match_ranges([rng(0.0, 1.0)], [rng(0.0, 1.0)], iou_threshold=float("nan"))
    with pytest.raises(EvaluationError):
        match_ranges(["nope"], [rng(0.0, 1.0)])  # type: ignore[list-item]
    with pytest.raises(EvaluationError):
        match_ranges([rng(0.0, 1.0)], rng(0.0, 1.0))  # type: ignore[arg-type]


# --- Single-stratum metrics -------------------------------------------------


def test_evaluate_ranges_hand_calculated() -> None:
    truth = [rng(10.0, 12.0), rng(20.0, 23.0)]
    preds = [rng(10.5, 12.5), rng(30.0, 31.0), rng(20.2, 22.8)]
    metrics = evaluate_ranges(truth, preds)
    assert metrics.num_truth == 2
    assert metrics.num_predictions == 3
    assert metrics.num_excused_ambiguous == 0
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.precision == pytest.approx(2.0 / 3.0)
    assert metrics.recall == pytest.approx(1.0)
    # Onset errors 0.5, 0.2 -> mean abs 0.35, signed 0.35.
    assert metrics.mean_abs_onset_error_seconds == pytest.approx(0.35)
    assert metrics.mean_signed_onset_error_seconds == pytest.approx(0.35)
    # End errors 0.5, -0.2 -> mean abs 0.35, signed 0.15.
    assert metrics.mean_abs_end_error_seconds == pytest.approx(0.35)
    assert metrics.mean_signed_end_error_seconds == pytest.approx(0.15)
    # Retained: (1.5 + 2.6) / (2 + 3) = 0.82.
    assert metrics.retained_duration_ratio == pytest.approx(0.82)
    # Extra: ((2 + 1 + 2.6) - 4.1) / 3 = 0.5.
    assert metrics.extra_seconds_per_attempt == pytest.approx(0.5)
    assert len(metrics.matches) == 2


def test_empty_predictions() -> None:
    metrics = evaluate_ranges([rng(10.0, 12.0)], [])
    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.precision is None
    assert metrics.recall == pytest.approx(0.0)
    assert metrics.mean_abs_onset_error_seconds is None
    assert metrics.mean_abs_end_error_seconds is None
    assert metrics.mean_signed_onset_error_seconds is None
    assert metrics.mean_signed_end_error_seconds is None
    assert metrics.retained_duration_ratio == pytest.approx(0.0)
    assert metrics.extra_seconds_per_attempt == pytest.approx(0.0)
    assert metrics.matches == ()


def test_empty_truth() -> None:
    metrics = evaluate_ranges([], [rng(10.0, 12.0)])
    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.precision == pytest.approx(0.0)
    assert metrics.recall is None
    assert metrics.mean_abs_onset_error_seconds is None
    assert metrics.retained_duration_ratio is None
    assert metrics.extra_seconds_per_attempt == pytest.approx(2.0)


def test_empty_both() -> None:
    metrics = evaluate_ranges([], [])
    assert metrics.precision is None
    assert metrics.recall is None
    assert metrics.retained_duration_ratio is None
    assert metrics.extra_seconds_per_attempt == pytest.approx(0.0)
    assert metrics.mean_abs_onset_error_seconds is None
    assert metrics.matches == ()


def test_ambiguity_excuses_unmatched_predictions() -> None:
    truth = [rng(0.0, 2.0)]
    ambiguous = [rng(10.0, 12.0)]
    preds = [rng(0.2, 1.8), rng(10.1, 11.9)]
    metrics = evaluate_ranges(truth, preds, ambiguous_ranges=ambiguous)
    assert metrics.true_positives == 1
    assert metrics.num_excused_ambiguous == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    # Only the scored prediction counts: dur 1.6, intersection 1.6.
    assert metrics.extra_seconds_per_attempt == pytest.approx(0.0)
    assert metrics.retained_duration_ratio == pytest.approx(1.6 / 2.0)


def test_ambiguity_does_not_excuse_poor_overlap() -> None:
    # IoU([10.5,11.0],[10,12]) = 0.5/2 = 0.25 < 0.5: stays a false positive.
    metrics = evaluate_ranges(
        [rng(0.0, 2.0)],
        [rng(10.5, 11.0)],
        ambiguous_ranges=[rng(10.0, 12.0)],
    )
    assert metrics.num_excused_ambiguous == 0
    assert metrics.false_positives == 1
    assert metrics.precision == pytest.approx(0.0)


def test_ambiguity_never_counts_as_truth() -> None:
    metrics = evaluate_ranges([], [], ambiguous_ranges=[rng(0.0, 2.0)])
    assert metrics.num_truth == 0
    assert metrics.recall is None
    assert metrics.retained_duration_ratio is None


def test_evaluate_ranges_is_deterministic() -> None:
    truth = [rng(10.0, 12.0), rng(20.0, 23.0)]
    preds = [rng(10.5, 12.5), rng(30.0, 31.0), rng(20.2, 22.8)]
    first = evaluate_ranges(truth, preds)
    second = evaluate_ranges(truth, preds)
    assert first == second
    assert first.to_dict() == second.to_dict()


# --- Strata and overall pooling ---------------------------------------------


def test_stratified_report_hand_calculated() -> None:
    report = evaluate_report(
        {
            "sess-a": StratumInput(
                truth=(rng(10.0, 12.0),),
                predictions=(rng(10.5, 12.5), rng(30.0, 31.0)),
            ),
            "sess-b": StratumInput(
                truth=(rng(0.0, 4.0),),
                predictions=(rng(0.0, 1.0),),
            ),
        }
    )
    assert report.iou_threshold == pytest.approx(0.5)
    assert [entry.stratum for entry in report.strata] == ["sess-a", "sess-b"]

    sess_a = report.stratum("sess-a").metrics
    assert sess_a.true_positives == 1
    assert sess_a.false_positives == 1
    assert sess_a.precision == pytest.approx(0.5)
    assert sess_a.recall == pytest.approx(1.0)
    assert sess_a.retained_duration_ratio == pytest.approx(1.5 / 2.0)
    assert sess_a.extra_seconds_per_attempt == pytest.approx(0.75)

    sess_b = report.stratum("sess-b").metrics
    assert sess_b.true_positives == 0
    assert sess_b.precision == pytest.approx(0.0)
    assert sess_b.recall == pytest.approx(0.0)
    assert sess_b.mean_abs_onset_error_seconds is None
    assert sess_b.retained_duration_ratio == pytest.approx(0.0)
    assert sess_b.extra_seconds_per_attempt == pytest.approx(1.0)

    overall = report.overall
    assert overall.num_truth == 2
    assert overall.num_predictions == 3
    assert overall.true_positives == 1
    assert overall.false_positives == 2
    assert overall.false_negatives == 1
    assert overall.precision == pytest.approx(1.0 / 3.0)
    assert overall.recall == pytest.approx(0.5)
    assert overall.mean_abs_onset_error_seconds == pytest.approx(0.5)
    assert overall.mean_abs_end_error_seconds == pytest.approx(0.5)
    assert overall.retained_duration_ratio == pytest.approx(1.5 / 6.0)
    assert overall.extra_seconds_per_attempt == pytest.approx(2.5 / 3.0)


def test_strata_never_match_across_sessions() -> None:
    # Identical clock times in different sessions must not pair up.
    report = evaluate_report(
        {
            "sess-a": StratumInput(truth=(rng(0.0, 2.0),), predictions=()),
            "sess-b": StratumInput(truth=(), predictions=(rng(0.0, 2.0),)),
        }
    )
    assert report.overall.true_positives == 0
    assert report.overall.false_negatives == 1
    assert report.overall.false_positives == 1
    assert report.overall.precision == pytest.approx(0.0)
    assert report.overall.recall == pytest.approx(0.0)


def test_report_rejects_bad_strata() -> None:
    with pytest.raises(EvaluationError):
        evaluate_report({})
    with pytest.raises(EvaluationError):
        evaluate_report({"  ": StratumInput()})
    with pytest.raises(EvaluationError):
        evaluate_report({"sess-a": "nope"})  # type: ignore[dict-value]
    with pytest.raises(EvaluationError):
        evaluate_report(["nope"])  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        evaluate_report({"sess-a": StratumInput()}, iou_threshold=0.0)


def test_stratum_lookup_rejects_unknown() -> None:
    report = evaluate_report({"sess-a": StratumInput()})
    with pytest.raises(KeyError):
        report.stratum("sess-missing")


# --- Manifest validation ----------------------------------------------------


def test_manifest_round_trip() -> None:
    original = manifest(
        "dev",
        [
            ann("sess-a", "sess-a/cap01.mov", 10.0, 12.0),
            ann("sess-a", "sess-a/cap01.mov", 30.0, 31.0, "ambiguous"),
        ],
    )
    assert AnnotationManifest.from_dict(original.to_dict()) == original
    assert AnnotationManifest.from_json(original.to_json()) == original
    assert original.sessions == ("sess-a",)
    assert len(original.serve_annotations()) == 1
    assert len(original.ambiguous_annotations()) == 1
    # Deterministic JSON: sorted keys, stable payload.
    assert original.to_json() == manifest(
        "dev",
        [
            ann("sess-a", "sess-a/cap01.mov", 10.0, 12.0),
            ann("sess-a", "sess-a/cap01.mov", 30.0, 31.0, "ambiguous"),
        ],
    ).to_json()
    assert list(json.loads(original.to_json())) == sorted(
        json.loads(original.to_json())
    )


def test_manifest_schema_version_is_pinned() -> None:
    assert ANNOTATION_MANIFEST_SCHEMA_VERSION == 1
    assert EVALUATION_REPORT_SCHEMA_VERSION == 1


def test_manifest_rejects_absolute_paths() -> None:
    for bad in ("/abs/cap.mov", "/tmp/x.mov"):
        with pytest.raises(EvaluationError):
            ann("sess-a", bad, 1.0, 2.0)
    with pytest.raises(EvaluationError):
        Annotation.from_dict(
            {
                "session_id": "sess-a",
                "media": "/abs/cap.mov",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "label": "serve",
            }
        )


def test_manifest_rejects_parent_traversal() -> None:
    with pytest.raises(EvaluationError):
        ann("sess-a", "../outside/cap.mov", 1.0, 2.0)


def test_manifest_rejects_invalid_labels() -> None:
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", 1.0, 2.0, "Serve")
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", 1.0, 2.0, "toss")
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", 1.0, 2.0, "")


def test_manifest_rejects_invalid_times() -> None:
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", 2.0, 2.0)
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", 3.0, 2.0)
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", -1.0, 2.0)
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", 1.0, float("inf"))
    with pytest.raises(EvaluationError):
        ann("sess-a", "sess-a/cap.mov", float("nan"), 2.0)


def test_manifest_rejects_bad_shape() -> None:
    good = manifest("dev", [ann("sess-a", "sess-a/cap.mov", 1.0, 2.0)])
    payload = good.to_dict()
    with pytest.raises(EvaluationError):
        AnnotationManifest.from_dict({**payload, "schema_version": 999})
    with pytest.raises(EvaluationError):
        AnnotationManifest.from_dict({**payload, "schema_version": 0})
    with pytest.raises(EvaluationError):
        AnnotationManifest.from_dict({**payload, "dataset_split": "test"})
    with pytest.raises(EvaluationError):
        AnnotationManifest.from_dict({**payload, "extra": 1})
    with pytest.raises(EvaluationError):
        AnnotationManifest.from_dict({k: v for k, v in payload.items() if k != "annotations"})
    with pytest.raises(EvaluationError):
        AnnotationManifest.from_dict({**payload, "annotations": "nope"})
    with pytest.raises(EvaluationError):
        AnnotationManifest.from_json("{oops")
    with pytest.raises(EvaluationError):
        Annotation.from_dict({"session_id": "sess-a"})
    with pytest.raises(EvaluationError):
        Annotation.from_dict(
            {**good.annotations[0].to_dict(), "bogus": True}
        )
    with pytest.raises(EvaluationError):
        AnnotationManifest(
            schema_version=1, dataset_split="dev", annotations=["nope"]  # type: ignore[list-item]
        )
    with pytest.raises(EvaluationError):
        ann("  ", "sess-a/cap.mov", 1.0, 2.0)
    with pytest.raises(EvaluationError):
        ann("sess-a", "   ", 1.0, 2.0)


def test_session_disjoint_passes() -> None:
    dev = manifest("dev", [ann("sess-a", "sess-a/cap.mov", 1.0, 2.0)])
    heldout = manifest("heldout", [ann("sess-b", "sess-b/cap.mov", 1.0, 2.0)])
    validate_session_disjoint(dev, heldout)  # must not raise


def test_session_leakage_rejected() -> None:
    dev = manifest(
        "dev",
        [
            ann("sess-a", "sess-a/cap.mov", 1.0, 2.0),
            ann("sess-b", "sess-b/cap.mov", 1.0, 2.0),
        ],
    )
    heldout = manifest(
        "heldout", [ann("sess-b", "sess-b/other.mov", 5.0, 6.0)]
    )
    with pytest.raises(EvaluationError, match="sess-b"):
        validate_session_disjoint(dev, heldout)
    with pytest.raises(EvaluationError):
        validate_session_disjoint(dev, "nope")  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        validate_session_disjoint("nope", heldout)  # type: ignore[arg-type]


# --- Manifest-driven evaluation ---------------------------------------------


def test_evaluate_manifest_groups_by_session() -> None:
    doc = manifest(
        "dev",
        [
            ann("sess-a", "sess-a/cap.mov", 10.0, 12.0),
            ann("sess-a", "sess-a/cap.mov", 30.0, 31.0, "ambiguous"),
            ann("sess-b", "sess-b/cap.mov", 5.0, 7.0),
        ],
    )
    assert set(group_truth_by_session(doc)) == {"sess-a", "sess-b"}
    assert set(group_ambiguous_by_session(doc)) == {"sess-a"}
    report = evaluate_manifest(
        doc,
        {
            "sess-a": [rng(10.5, 12.5)],
            "sess-b": [rng(5.0, 7.0)],
        },
    )
    assert [entry.stratum for entry in report.strata] == ["sess-a", "sess-b"]
    assert report.overall.true_positives == 2
    assert report.overall.false_negatives == 0
    assert report.overall.false_positives == 0
    assert report.overall.precision == pytest.approx(1.0)
    assert report.overall.recall == pytest.approx(1.0)


def test_evaluate_manifest_missing_session_scores_misses() -> None:
    doc = manifest("dev", [ann("sess-a", "sess-a/cap.mov", 10.0, 12.0)])
    report = evaluate_manifest(doc, {})
    assert report.overall.true_positives == 0
    assert report.overall.false_negatives == 1
    assert report.overall.recall == pytest.approx(0.0)
    assert report.overall.precision is None


def test_evaluate_manifest_rejects_unknown_sessions() -> None:
    doc = manifest("dev", [ann("sess-a", "sess-a/cap.mov", 10.0, 12.0)])
    with pytest.raises(EvaluationError, match="sess-ghost"):
        evaluate_manifest(doc, {"sess-ghost": [rng(0.0, 1.0)]})
    with pytest.raises(EvaluationError):
        evaluate_manifest(doc, {"sess-a": ["nope"]})  # type: ignore[list-item]
    with pytest.raises(EvaluationError):
        evaluate_manifest("nope", {})  # type: ignore[arg-type]
    with pytest.raises(EvaluationError):
        evaluate_manifest(manifest("dev", []), {})


# --- Fixture manifests -------------------------------------------------------


def test_fixture_manifests_are_valid_and_disjoint() -> None:
    dev = AnnotationManifest.from_json((FIXTURES / "dev.json").read_text())
    heldout = AnnotationManifest.from_json((FIXTURES / "heldout.json").read_text())
    assert dev.dataset_split == "dev"
    assert heldout.dataset_split == "heldout"
    assert dev.sessions == ("sess-a", "sess-b")
    assert heldout.sessions == ("sess-c",)
    validate_session_disjoint(dev, heldout)
    # Synthetic predictions: one hit, one miss, one false positive.
    report = evaluate_manifest(
        dev,
        {
            "sess-a": [rng(10.5, 12.5), rng(40.0, 41.0)],
            "sess-b": [rng(5.0, 7.0)],
        },
    )
    assert report.overall.num_truth == 3
    assert report.overall.true_positives == 2
    assert report.overall.false_positives == 1
    assert report.overall.false_negatives == 1
    assert report.overall.precision == pytest.approx(2.0 / 3.0)
    assert report.overall.recall == pytest.approx(2.0 / 3.0)


# --- Report codec ------------------------------------------------------------


def test_report_json_round_trip() -> None:
    report = evaluate_report(
        {
            "sess-a": StratumInput(
                truth=(rng(10.0, 12.0),),
                predictions=(rng(10.5, 12.5), rng(30.0, 31.0)),
                ambiguous=(rng(50.0, 52.0),),
            ),
            "sess-b": StratumInput(truth=(rng(0.0, 4.0),)),
        }
    )
    assert EvaluationReport.from_dict(report.to_dict()) == report
    assert EvaluationReport.from_json(report.to_json()) == report
    # Strata serialize in sorted order with sorted keys.
    payload = json.loads(report.to_json())
    assert [entry["stratum"] for entry in payload["strata"]] == ["sess-a", "sess-b"]
    assert list(payload) == sorted(payload)


def test_report_rejects_newer_and_unknown_versions() -> None:
    report = evaluate_report({"sess-a": StratumInput()})
    with pytest.raises(EvaluationError):
        EvaluationReport.from_dict(
            {**report.to_dict(), "schema_version": 999}
        )
    with pytest.raises(EvaluationError):
        EvaluationReport.from_dict({**report.to_dict(), "extra": 1})
    with pytest.raises(EvaluationError):
        EvaluationReport.from_dict(
            {k: v for k, v in report.to_dict().items() if k != "overall"}
        )
    tampered = report.to_dict()
    tampered["strata"] = [
        {**tampered["strata"][0], "metrics": {**tampered["strata"][0]["metrics"], "precision": 0.12345}}
    ]
    # Hand-edited scalars that break count/None-pattern consistency fail.
    with pytest.raises(EvaluationError):
        EvaluationReport.from_dict(tampered)
    with pytest.raises(EvaluationError):
        EvaluationReport.from_json("{oops")
    with pytest.raises(EvaluationError):
        EvaluationReport(
            schema_version=1,
            iou_threshold=0.5,
            overall=report.overall,
            strata=(
                StratumReport(stratum="dup", metrics=report.strata[0].metrics),
                StratumReport(stratum="dup", metrics=report.strata[0].metrics),
            ),
        )
    with pytest.raises(EvaluationError):
        EvaluationReport(
            schema_version=1,
            iou_threshold=0.5,
            overall=report.overall,
            strata=(),
        )


def test_metrics_and_match_codecs_reject_bad_shapes() -> None:
    metrics = evaluate_ranges([rng(0.0, 2.0)], [rng(0.0, 2.0)])
    with pytest.raises(EvaluationError):
        RangeMetrics.from_dict({**metrics.to_dict(), "bogus": 1})
    with pytest.raises(EvaluationError):
        RangeMetrics.from_dict(
            {k: v for k, v in metrics.to_dict().items() if k != "matches"}
        )
    with pytest.raises(EvaluationError):
        OverallMetrics.from_dict({**metrics.to_dict(), "matches": []})
    with pytest.raises(EvaluationError):
        MatchEntry.from_dict(metrics.matches[0].to_dict() | {"extra": 0})
    with pytest.raises(EvaluationError):
        MatchEntry(
            truth_index=-1,
            pred_index=0,
            iou=0.5,
            onset_error_seconds=0.0,
            end_error_seconds=0.0,
            intersection_seconds=1.0,
        )
    with pytest.raises(EvaluationError):
        StratumReport.from_dict({"stratum": "sess-a"})
    with pytest.raises(EvaluationError):
        StratumReport(stratum="  ", metrics=metrics)
    with pytest.raises(EvaluationError):
        StratumInput(truth=["nope"])  # type: ignore[list-item]
    with pytest.raises(EvaluationError):
        group_truth_by_session("nope")  # type: ignore[arg-type]
