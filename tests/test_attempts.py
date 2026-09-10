"""Focused M3.3 tests for the attempt document and padding planner.

Deterministic, offline, synthetic only: candidates are built directly
as :class:`CandidateRange` values, never from private footage.
"""

from __future__ import annotations

import json

import pytest

from serve_review.detection.plan import (
    DEFAULT_METHOD_VERSION,
    PLAN_SCHEMA_VERSION,
    PlanConfig,
    PlanError,
    plan_attempts,
    to_export_plan,
    union_effective_ranges,
)
from serve_review.detection.ranges import CandidateRange
from serve_review.domain import (
    Attempt,
    AttemptDocument,
    AttemptError,
    DomainError,
    ExportPlan,
    MediaRange,
    SchemaVersionError,
    SourceMetadata,
)


def make_source(**overrides) -> SourceMetadata:
    fields = {
        "fingerprint": "sha256:attempts-fixture",
        "duration_seconds": 120.0,
        "width": 1080,
        "height": 1920,
        "frame_rate_num": 120,
        "frame_rate_den": 1,
        "video_codec": "hevc",
        "rotation_degrees": 0,
    }
    fields.update(overrides)
    return SourceMetadata(**fields)


def cand(start: float, end: float) -> CandidateRange:
    return CandidateRange(start_seconds=start, end_seconds=end)


# --- Config codecs ----------------------------------------------------------


def test_plan_schema_version_is_pinned() -> None:
    assert PLAN_SCHEMA_VERSION == 1
    assert PlanConfig().schema_version == 1


def test_plan_config_json_round_trip() -> None:
    config = PlanConfig(padding_seconds=1.5)
    assert PlanConfig.from_dict(config.to_dict()) == config
    assert PlanConfig.from_json(config.to_json()) == config
    assert list(json.loads(config.to_json())) == sorted(json.loads(config.to_json()))


def test_plan_config_validation() -> None:
    with pytest.raises(PlanError):
        PlanConfig(padding_seconds=-0.1)
    with pytest.raises(PlanError):
        PlanConfig(padding_seconds=float("nan"))
    with pytest.raises(PlanError):
        PlanConfig(padding_seconds=float("inf"))
    with pytest.raises(PlanError):
        PlanConfig(padding_seconds="1")  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        PlanConfig(padding_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        PlanConfig(schema_version=999)
    with pytest.raises(PlanError):
        PlanConfig.from_dict({"padding_seconds": 1.0})
    with pytest.raises(PlanError):
        PlanConfig.from_dict(
            {"padding_seconds": 1.0, "schema_version": 1, "extra": 0}
        )
    with pytest.raises(PlanError):
        PlanConfig.from_dict(
            {"padding_seconds": 1.0, "schema_version": 999}
        )
    with pytest.raises(PlanError):
        PlanConfig.from_dict("nope")  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        PlanConfig.from_json("{oops")


# --- Zero padding -----------------------------------------------------------


def test_zero_padding_preserves_detected_ranges() -> None:
    source = make_source()
    config = PlanConfig(padding_seconds=0.0)
    doc = plan_attempts([cand(10.0, 12.0), cand(30.0, 31.5)], source, config)
    assert len(doc) == 2
    for attempt, (start, end) in zip(doc, [(10.0, 12.0), (30.0, 31.5)]):
        assert attempt.detected_range == MediaRange(start, end)
        assert attempt.effective_range == MediaRange(start, end)
    assert doc.export_ranges == (
        MediaRange(10.0, 12.0),
        MediaRange(30.0, 31.5),
    )
    assert doc.padding_seconds == pytest.approx(0.0)


def test_symmetric_padding_expands_inside_bounds() -> None:
    source = make_source()
    doc = plan_attempts([cand(10.0, 12.0)], source, PlanConfig(padding_seconds=1.0))
    attempt = doc[0]
    assert attempt.detected_range == MediaRange(10.0, 12.0)
    assert attempt.effective_range == MediaRange(9.0, 13.0)


# --- Source boundaries ------------------------------------------------------


def test_padding_clamps_to_source_bounds() -> None:
    source = make_source(duration_seconds=120.0)
    doc = plan_attempts(
        [cand(0.2, 1.0), cand(118.5, 119.5)],
        source,
        PlanConfig(padding_seconds=1.0),
    )
    assert doc[0].effective_range == MediaRange(0.0, 2.0)
    assert doc[1].effective_range == MediaRange(117.5, 120.0)
    # Detected ranges are preserved, never rewritten by clamping.
    assert doc[0].detected_range == MediaRange(0.2, 1.0)
    assert doc[1].detected_range == MediaRange(118.5, 119.5)


def test_padding_clamps_fully_at_edges() -> None:
    source = make_source(duration_seconds=10.0)
    doc = plan_attempts([cand(0.0, 0.5)], source, PlanConfig(padding_seconds=5.0))
    assert doc[0].effective_range == MediaRange(0.0, 5.5)
    doc2 = plan_attempts([cand(9.5, 10.0)], source, PlanConfig(padding_seconds=5.0))
    assert doc2[0].effective_range == MediaRange(4.5, 10.0)


def test_candidate_beyond_source_duration_rejected() -> None:
    source = make_source(duration_seconds=10.0)
    with pytest.raises(PlanError):
        plan_attempts([cand(9.0, 11.0)], source, PlanConfig(padding_seconds=0.0))
    with pytest.raises(PlanError):
        plan_attempts([cand(10.0, 11.0)], source, PlanConfig(padding_seconds=0.0))


# --- Overlap union ----------------------------------------------------------


def test_overlapping_effective_ranges_union_for_export() -> None:
    source = make_source()
    # Detected ranges are disjoint, but padding=1 makes them overlap:
    # [10,11]->[9,12], [11.5,12.5]->[10.5,13.5] overlap.
    doc = plan_attempts(
        [cand(10.0, 11.0), cand(11.5, 12.5)],
        source,
        PlanConfig(padding_seconds=1.0),
    )
    assert doc[0].detected_range == MediaRange(10.0, 11.0)
    assert doc[1].detected_range == MediaRange(11.5, 12.5)
    assert doc[0].effective_range == MediaRange(9.0, 12.0)
    assert doc[1].effective_range == MediaRange(10.5, 13.5)
    assert doc.export_ranges == (MediaRange(9.0, 13.5),)


def test_adjacent_effective_ranges_stay_separate() -> None:
    # Effective [10,12] and [12,14] touch but do not overlap.
    merged = union_effective_ranges([MediaRange(10.0, 12.0), MediaRange(12.0, 14.0)])
    assert merged == (MediaRange(10.0, 12.0), MediaRange(12.0, 14.0))
    source = make_source()
    doc = plan_attempts(
        [cand(10.0, 12.0), cand(12.0, 14.0)],
        source,
        PlanConfig(padding_seconds=0.0),
    )
    assert doc.export_ranges == (MediaRange(10.0, 12.0), MediaRange(12.0, 14.0))
    # Adjacent export ranges convert to a valid ExportPlan.
    plan = to_export_plan(doc)
    assert isinstance(plan, ExportPlan)
    assert plan.ranges == doc.export_ranges


def test_chain_overlap_unions_transitively() -> None:
    merged = union_effective_ranges(
        [MediaRange(0.0, 2.0), MediaRange(1.0, 3.0), MediaRange(2.5, 4.0)]
    )
    assert merged == (MediaRange(0.0, 4.0),)


def test_export_plan_conversion_matches_union() -> None:
    source = make_source()
    doc = plan_attempts(
        [cand(10.0, 11.0), cand(11.5, 12.5), cand(50.0, 51.0)],
        source,
        PlanConfig(padding_seconds=1.0),
    )
    plan = to_export_plan(doc)
    assert plan.source_fingerprint == source.fingerprint
    assert plan.source_duration_seconds == pytest.approx(source.duration_seconds)
    assert plan.ranges == doc.export_ranges
    assert len(plan) == 2


# --- Ordering and stable IDs -------------------------------------------------


def test_unsorted_input_is_sorted_with_stable_ids() -> None:
    source = make_source()
    doc = plan_attempts(
        [cand(30.0, 31.0), cand(10.0, 11.0), cand(20.0, 21.0)],
        source,
        PlanConfig(padding_seconds=0.0),
    )
    assert [a.attempt_id for a in doc] == ["serve-001", "serve-002", "serve-003"]
    assert [a.detected_range.start_seconds for a in doc] == [10.0, 20.0, 30.0]


def test_stable_rerun_is_deterministic() -> None:
    source = make_source()
    candidates = [cand(10.0, 12.0), cand(30.0, 31.5)]
    config = PlanConfig(padding_seconds=1.0)
    first = plan_attempts(candidates, source, config)
    second = plan_attempts(candidates, source, config)
    assert first == second
    assert first.to_json() == second.to_json()
    assert [a.to_dict() for a in first] == [a.to_dict() for a in second]
    # Reversed input order still yields the same document.
    third = plan_attempts(list(reversed(candidates)), source, config)
    assert third == first


# --- Confidence and evidence --------------------------------------------------


def test_confidence_and_evidence_serialization() -> None:
    source = make_source()
    doc = plan_attempts(
        [cand(10.0, 12.0), cand(30.0, 31.0)],
        source,
        PlanConfig(padding_seconds=0.5),
        confidences=[0.85, None],
        evidences=[{"motion_peak": 0.9}, {}],
    )
    assert doc[0].confidence == pytest.approx(0.85)
    assert doc[0].evidence == {"motion_peak": 0.9}
    assert doc[1].confidence is None
    assert doc[1].evidence == {}
    payload = doc.to_dict()
    assert payload["attempts"][0]["confidence"] == pytest.approx(0.85)
    assert payload["attempts"][0]["evidence"] == {"motion_peak": 0.9}
    assert AttemptDocument.from_dict(payload) == doc
    assert AttemptDocument.from_json(doc.to_json()) == doc
    # Deterministic JSON: evidence keys sorted.
    doc2 = plan_attempts(
        [cand(10.0, 12.0)],
        source,
        PlanConfig(padding_seconds=0.0),
        evidences=[{"z": 1.0, "a": 2.0}],
    )
    assert list(json.loads(doc2.to_json())["attempts"][0]["evidence"]) == ["a", "z"]


def test_default_confidence_is_unknown() -> None:
    source = make_source()
    doc = plan_attempts([cand(5.0, 6.0)], source, PlanConfig(padding_seconds=0.0))
    assert doc[0].confidence is None
    assert doc[0].evidence == {}


def test_attempt_json_round_trip() -> None:
    attempt = Attempt(
        attempt_id="serve-001",
        detected_range=MediaRange(10.0, 12.0),
        effective_range=MediaRange(9.0, 13.0),
        confidence=0.5,
        evidence={"motion_peak": 0.7},
    )
    assert Attempt.from_dict(attempt.to_dict()) == attempt
    assert Attempt.from_json(attempt.to_json()) == attempt


# --- Empty detection ----------------------------------------------------------


def test_empty_detection_yields_empty_plan() -> None:
    source = make_source()
    doc = plan_attempts([], source, PlanConfig(padding_seconds=1.0))
    assert doc.attempts == ()
    assert doc.export_ranges == ()
    assert len(doc) == 0
    assert doc.total_export_duration_seconds == pytest.approx(0.0)
    assert AttemptDocument.from_dict(doc.to_dict()) == doc
    assert AttemptDocument.from_json(doc.to_json()) == doc
    with pytest.raises((AttemptError, PlanError)):
        to_export_plan(doc)
    with pytest.raises((AttemptError, PlanError)):
        doc.to_export_plan()


# --- Document codecs and validation -------------------------------------------


def test_document_json_round_trip_preserves_ranges() -> None:
    source = make_source()
    doc = plan_attempts(
        [cand(10.0, 11.0), cand(11.5, 12.5)],
        source,
        PlanConfig(padding_seconds=1.0),
        confidences=[0.9, 0.4],
        evidences=[{"motion_peak": 0.95}, {"motion_peak": 0.5}],
    )
    decoded = AttemptDocument.from_json(doc.to_json())
    assert decoded == doc
    # Union did not rewrite detected ranges.
    assert decoded[0].detected_range == MediaRange(10.0, 11.0)
    assert decoded[1].detected_range == MediaRange(11.5, 12.5)
    assert decoded.method_version == DEFAULT_METHOD_VERSION
    assert decoded.source_fingerprint == source.fingerprint


def test_document_rejects_newer_and_unknown_versions() -> None:
    source = make_source()
    doc = plan_attempts([cand(10.0, 11.0)], source, PlanConfig(padding_seconds=0.0))
    with pytest.raises(SchemaVersionError):
        AttemptDocument.from_dict({**doc.to_dict(), "schema_version": 999})
    nested = doc.to_dict()
    nested["attempts"] = [{**nested["attempts"][0], "schema_version": 999}]
    with pytest.raises((SchemaVersionError, AttemptError)):
        AttemptDocument.from_dict(nested)
    nested2 = doc.to_dict()
    nested2["export_ranges"] = [{**nested2["export_ranges"][0], "schema_version": 99}]
    with pytest.raises(AttemptError):
        AttemptDocument.from_dict(nested2)
    with pytest.raises(AttemptError):
        AttemptDocument.from_dict({**doc.to_dict(), "extra": 1})
    with pytest.raises(AttemptError):
        Attempt.from_dict({**doc[0].to_dict(), "bogus": True})
    with pytest.raises((DomainError, AttemptError)):
        AttemptDocument.from_json("{oops")


def test_document_rejects_inconsistent_export_union() -> None:
    source = make_source()
    doc = plan_attempts(
        [cand(10.0, 11.0), cand(11.5, 12.5)],
        source,
        PlanConfig(padding_seconds=1.0),
    )
    tampered = {
        **doc.to_dict(),
        "export_ranges": [MediaRange(9.0, 12.0).to_dict()],
    }
    with pytest.raises(AttemptError):
        AttemptDocument.from_dict(tampered)


def test_attempt_validation_errors() -> None:
    with pytest.raises(AttemptError):
        Attempt(
            attempt_id="bad-1",
            detected_range=MediaRange(1.0, 2.0),
            effective_range=MediaRange(1.0, 2.0),
        )
    with pytest.raises(AttemptError):
        Attempt(
            attempt_id="serve-001",
            detected_range=MediaRange(1.0, 2.0),
            effective_range=MediaRange(1.5, 2.5),  # does not contain detected
        )
    with pytest.raises(AttemptError):
        Attempt(
            attempt_id="serve-001",
            detected_range=MediaRange(1.0, 2.0),
            effective_range=MediaRange(0.5, 2.5),
            confidence=1.5,
        )
    with pytest.raises(AttemptError):
        Attempt(
            attempt_id="serve-001",
            detected_range=MediaRange(1.0, 2.0),
            effective_range=MediaRange(0.5, 2.5),
            evidence={"": 1.0},
        )
    with pytest.raises(AttemptError):
        Attempt(
            attempt_id="serve-001",
            detected_range=MediaRange(1.0, 2.0),
            effective_range=MediaRange(0.5, 2.5),
            evidence={"peak": float("nan")},  # type: ignore[dict-item]
        )
    with pytest.raises(AttemptError):
        Attempt.from_dict({"attempt_id": "serve-001"})


def test_planner_input_validation() -> None:
    source = make_source()
    with pytest.raises(PlanError):
        plan_attempts(["nope"], source)  # type: ignore[list-item]
    with pytest.raises(PlanError):
        plan_attempts([cand(1.0, 2.0)], "nope")  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        plan_attempts([cand(1.0, 2.0)], source, config="nope")  # type: ignore[arg-type]
    with pytest.raises(PlanError):
        plan_attempts(
            [cand(1.0, 2.0), cand(3.0, 4.0)],
            source,
            PlanConfig(padding_seconds=0.0),
            confidences=[0.5],
        )
    with pytest.raises(PlanError):
        plan_attempts(
            [cand(1.0, 2.0)],
            source,
            PlanConfig(padding_seconds=0.0),
            confidences=[2.0],
        )
    with pytest.raises(PlanError):
        plan_attempts(
            [cand(1.0, 2.0)],
            source,
            PlanConfig(padding_seconds=0.0),
            evidences=[{"peak": float("inf")}],
        )
    with pytest.raises(PlanError):
        plan_attempts(
            [cand(1.0, 2.0)],
            source,
            PlanConfig(padding_seconds=0.0),
            evidences=[{"peak": 0.5}, {"extra": 0.1}],
        )
    with pytest.raises(PlanError):
        plan_attempts([cand(1.0, 2.0)], source, method_version="  ")
    with pytest.raises(PlanError):
        union_effective_ranges(["nope"])  # type: ignore[list-item]
    with pytest.raises(PlanError):
        to_export_plan("nope")  # type: ignore[arg-type]
