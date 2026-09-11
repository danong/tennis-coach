"""Focused M4.1 tests for the eight-stage phase schema.

Deterministic, offline, synthetic only; no private footage.
"""

from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType

import pytest

from serve_review.checkpoints import (
    PHASE_SCHEMA_VERSION,
    STAGE_AVAILABILITIES,
    STAGE_KEYS,
    STAGE_ORDER,
    STAGE_PROVENANCES,
    STRUCTURAL_STATUSES,
    AttemptPhase,
    PhaseDocument,
    PhaseError,
    StagePhase,
)
from serve_review.domain import (
    DomainError,
    MediaRange,
    SchemaVersionError,
)


def span(start: float, end: float) -> MediaRange:
    return MediaRange(start_seconds=start, end_seconds=end)


def avail_stage(
    start: float,
    end: float,
    *,
    keyframe: float | None = None,
    provenance: str = "body_pose",
    availability: str = "available",
    confidence: float = 0.8,
    uncertainty: float | None = None,
    evidence: list[str] | tuple[str, ...] = ("cue_a",),
    limitations: list[str] | tuple[str, ...] = ("body_pose_estimate",),
) -> StagePhase:
    return StagePhase(
        availability=availability,
        provenance=provenance,
        confidence=confidence,
        interval=span(start, end),
        keyframe_seconds=keyframe,
        temporal_uncertainty_seconds=uncertainty,
        evidence=evidence,
        limitations=limitations,
    )


def unavail_stage(*, provenance: str = "body_pose") -> StagePhase:
    return StagePhase(
        availability="unavailable",
        provenance=provenance,
        confidence=0.0,
        interval=None,
        keyframe_seconds=None,
        temporal_uncertainty_seconds=None,
        evidence=(),
        limitations=("no_observation",),
    )


def full_stages() -> dict[str, StagePhase]:
    return {
        "start": avail_stage(10.0, 11.0, keyframe=10.5),
        "release": avail_stage(11.0, 12.0, keyframe=11.5),
        "loading": avail_stage(12.0, 13.0),
        "cocking": avail_stage(13.0, 14.0),
        "acceleration": avail_stage(14.0, 15.0, keyframe=14.5),
        "contact": avail_stage(
            15.0,
            16.0,
            keyframe=15.5,
            provenance="audio_transient",
            uncertainty=0.05,
            evidence=("audio_peak",),
            limitations=("audio_anchored_not_visual",),
        ),
        "deceleration": avail_stage(16.0, 17.0),
        "finish": avail_stage(17.0, 18.0, keyframe=17.5),
    }


def make_attempt(
    stages: dict[str, StagePhase] | None = None,
    *,
    status: str = "complete",
    attempt_id: str = "serve-001",
    attempt_range: MediaRange | None = None,
    anomalies: list[str] | tuple[str, ...] = (),
) -> AttemptPhase:
    return AttemptPhase(
        attempt_id=attempt_id,
        attempt_range=attempt_range or span(10.0, 18.0),
        method_version="phase-test-v1",
        config_id="cfg-001",
        stages=stages if stages is not None else full_stages(),
        structural_status=status,
        anomalies=anomalies,
    )


def make_doc(*attempts: AttemptPhase) -> PhaseDocument:
    return PhaseDocument(
        source_fingerprint="sha256:phase-fixture",
        source_duration_seconds=120.0,
        attempts=list(attempts),
    )


# --- Keys/order -------------------------------------------------------------


def test_stage_keys_exact_order() -> None:
    assert STAGE_ORDER == (
        "start",
        "release",
        "loading",
        "cocking",
        "acceleration",
        "contact",
        "deceleration",
        "finish",
    )
    assert STAGE_KEYS == STAGE_ORDER
    assert len(STAGE_ORDER) == 8
    assert STAGE_AVAILABILITIES == ("available", "partial", "unavailable")
    assert STAGE_PROVENANCES == (
        "body_pose",
        "audio_transient",
        "body_pose_audio",
        "manual",
    )
    assert STRUCTURAL_STATUSES == ("complete", "partial", "incomplete", "unavailable")
    assert PHASE_SCHEMA_VERSION == 1


def test_attempt_stages_exposed_in_canonical_order() -> None:
    attempt = make_attempt()
    assert list(attempt.stages.keys()) == list(STAGE_ORDER)
    assert attempt.stage("contact").provenance == "audio_transient"
    with pytest.raises(PhaseError):
        attempt.stage("toss")


def test_concise_public_keys_reject_long_names() -> None:
    stages = full_stages()
    stages["toss_release"] = stages.pop("release")  # type: ignore[assignment]
    with pytest.raises(PhaseError):
        make_attempt(stages=stages)


# --- Golden round trips ------------------------------------------------------


def test_golden_deterministic_json_round_trips() -> None:
    stage = avail_stage(1.0, 2.0, keyframe=1.5)
    assert StagePhase.from_dict(stage.to_dict()) == stage
    assert StagePhase.from_json(stage.to_json()) == stage
    assert StagePhase.from_json(stage.to_json().encode("utf-8")) == stage
    assert stage.to_json() == stage.to_json()
    assert list(json.loads(stage.to_json())) == sorted(json.loads(stage.to_json()))

    attempt = make_attempt()
    encoded = attempt.to_json()
    assert attempt.to_json() == encoded
    assert AttemptPhase.from_json(encoded) == attempt
    # JSON object keys sort alphabetically, but decoded mapping is canonical.
    assert list(json.loads(encoded)["stages"]) == sorted(json.loads(encoded)["stages"])
    assert list(AttemptPhase.from_json(encoded).stages) == list(STAGE_ORDER)

    doc = make_doc(attempt)
    doc_json = doc.to_json()
    assert doc.to_json() == doc_json
    assert list(json.loads(doc_json)) == sorted(json.loads(doc_json))
    assert PhaseDocument.from_json(doc_json) == doc
    assert PhaseDocument.from_json(doc_json.encode("utf-8")) == doc


def test_partial_and_unavailable_documents() -> None:
    stages = full_stages()
    stages["finish"] = unavail_stage()
    partial = make_attempt(stages=stages, status="partial")
    assert partial.structural_status == "partial"
    assert PhaseDocument.from_json(make_doc(partial).to_json()) == make_doc(partial)

    all_unavail = {key: unavail_stage() for key in STAGE_ORDER}
    unavailable = make_attempt(
        stages=all_unavail, status="unavailable", attempt_range=span(0.0, 8.0)
    )
    assert unavailable.structural_status == "unavailable"
    assert AttemptPhase.from_dict(unavailable.to_dict()) == unavailable

    mixed = {key: unavail_stage() for key in STAGE_ORDER}
    mixed["loading"] = avail_stage(
        0.0, 1.0, availability="partial", evidence=("weak_cue",)
    )
    incomplete = make_attempt(
        stages=mixed, status="incomplete", attempt_range=span(0.0, 8.0)
    )
    assert incomplete.structural_status == "incomplete"

    empty = PhaseDocument(
        source_fingerprint="sha256:empty",
        source_duration_seconds=60.0,
        attempts=(),
    )
    assert len(empty) == 0
    assert PhaseDocument.from_json(empty.to_json()) == empty


# --- Unavailable honesty ------------------------------------------------------


def test_unavailable_forbids_interval_keyframe_uncertainty() -> None:
    with pytest.raises(PhaseError):
        StagePhase(
            availability="unavailable",
            provenance="body_pose",
            confidence=0.0,
            interval=span(1.0, 2.0),
        )
    with pytest.raises(PhaseError):
        StagePhase(
            availability="unavailable",
            provenance="body_pose",
            confidence=0.0,
            interval=None,
            keyframe_seconds=1.5,
        )
    with pytest.raises(PhaseError):
        StagePhase(
            availability="unavailable",
            provenance="body_pose",
            confidence=0.0,
            interval=None,
            temporal_uncertainty_seconds=0.1,
        )
    # Honest unavailable needs no fabricated times.
    assert unavail_stage().interval is None
    assert unavail_stage().keyframe_seconds is None


def test_available_requires_interval() -> None:
    with pytest.raises(PhaseError):
        StagePhase(
            availability="available",
            provenance="body_pose",
            confidence=0.5,
            interval=None,
        )
    with pytest.raises(PhaseError):
        StagePhase(
            availability="partial",
            provenance="body_pose",
            confidence=0.5,
            interval=None,
        )


# --- Range/keyframe/order ------------------------------------------------------


def test_stage_interval_must_lie_inside_attempt() -> None:
    stages = full_stages()
    stages["start"] = avail_stage(9.0, 11.0)  # starts before attempt
    with pytest.raises(PhaseError):
        make_attempt(stages=stages)
    stages2 = full_stages()
    stages2["finish"] = avail_stage(17.0, 19.0)  # ends after attempt
    with pytest.raises(PhaseError):
        make_attempt(stages=stages2)


def test_keyframe_must_lie_inside_interval() -> None:
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, keyframe=2.0)  # half-open end excluded
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, keyframe=0.9)
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, keyframe=2.5)
    assert avail_stage(1.0, 2.0, keyframe=1.0).keyframe_seconds == pytest.approx(1.0)


@pytest.mark.parametrize("keyframe", [float("nan"), float("inf"), True, "1.5", -1.0])
def test_invalid_keyframe_types(keyframe) -> None:
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, keyframe=keyframe)  # type: ignore[arg-type]


def test_chronological_stage_ordering_enforced() -> None:
    stages = full_stages()
    stages["loading"] = avail_stage(11.5, 12.5)  # overlaps release [11,12)
    with pytest.raises(PhaseError):
        make_attempt(stages=stages)
    # Keyframes inside ordered intervals stay chronological.
    ordered = full_stages()
    ordered["start"] = avail_stage(10.0, 11.0, keyframe=10.9)
    ordered["release"] = avail_stage(11.0, 12.0, keyframe=11.0)
    assert make_attempt(stages=ordered).structural_status == "complete"
    # Skipping unavailable stages still orders remaining ones.
    skipped = full_stages()
    skipped["loading"] = unavail_stage()
    ok = make_attempt(stages=skipped, status="partial")
    assert ok.stage("release").interval.end_seconds <= ok.stage(  # type: ignore[union-attr]
        "cocking"
    ).interval.start_seconds


# --- Confidence/provenance/availability/uncertainty ------------------------------


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf"), float("-inf"), True, False, "0.5", None],
)
def test_invalid_confidence(confidence) -> None:
    with pytest.raises((PhaseError, SchemaVersionError)):
        StagePhase(
            availability="available",
            provenance="body_pose",
            confidence=confidence,  # type: ignore[arg-type]
            interval=span(1.0, 2.0),
        )


@pytest.mark.parametrize(
    "provenance", ["body-kinematic", "visual", "racket-visual", "", None, True, 7]
)
def test_invalid_provenance_values(provenance) -> None:
    with pytest.raises(PhaseError):
        StagePhase(
            availability="available",
            provenance=provenance,  # type: ignore[arg-type]
            confidence=0.5,
            interval=span(1.0, 2.0),
        )


@pytest.mark.parametrize("availability", ["avail", "Available", "", None, True, 0])
def test_invalid_availability_values(availability) -> None:
    with pytest.raises(PhaseError):
        StagePhase(
            availability=availability,  # type: ignore[arg-type]
            provenance="body_pose",
            confidence=0.5,
            interval=span(1.0, 2.0),
        )


@pytest.mark.parametrize(
    "uncertainty", [-0.1, float("nan"), float("inf"), True, "0.1"]
)
def test_invalid_uncertainty_values(uncertainty) -> None:
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, uncertainty=uncertainty)  # type: ignore[arg-type]


# --- Honest provenance constraints ----------------------------------------------


def test_body_pose_contact_never_claims_visual_contact() -> None:
    stages = full_stages()
    stages["contact"] = avail_stage(
        15.0, 16.0, keyframe=15.5, provenance="body_pose", uncertainty=0.05
    )
    with pytest.raises(PhaseError):
        make_attempt(stages=stages)


@pytest.mark.parametrize("provenance", ["audio_transient", "body_pose_audio"])
def test_audio_contact_requires_anchor_and_uncertainty(provenance: str) -> None:
    no_key = full_stages()
    no_key["contact"] = avail_stage(
        15.0, 16.0, provenance=provenance, uncertainty=0.05
    )
    with pytest.raises(PhaseError):
        make_attempt(stages=no_key)
    no_unc = full_stages()
    no_unc["contact"] = avail_stage(
        15.0, 16.0, keyframe=15.5, provenance=provenance
    )
    with pytest.raises(PhaseError):
        make_attempt(stages=no_unc)
    ok = full_stages()
    ok["contact"] = avail_stage(
        15.0, 16.0, keyframe=15.5, provenance=provenance, uncertainty=0.0
    )
    assert make_attempt(stages=ok).structural_status == "complete"


def test_manual_contact_follows_optional_keyframe_rule() -> None:
    stages = full_stages()
    stages["contact"] = avail_stage(15.0, 16.0, provenance="manual")
    assert make_attempt(stages=stages).stage("contact").keyframe_seconds is None
    stages2 = full_stages()
    stages2["contact"] = avail_stage(
        15.0, 16.0, keyframe=15.2, provenance="manual", uncertainty=0.1
    )
    assert make_attempt(stages=stages2).stage("contact").keyframe_seconds == pytest.approx(
        15.2
    )


def test_release_cocking_carry_limitations_not_renamed_keys() -> None:
    attempt = make_attempt()
    assert set(attempt.stages) == set(STAGE_ORDER)
    assert attempt.stage("release").limitations == ("body_pose_estimate",)
    assert "racket" not in attempt.stage("cocking").provenance


# --- Status/anomaly ---------------------------------------------------------------


@pytest.mark.parametrize("status", ["done", "COMPLETE", "", None, True, 3])
def test_invalid_structural_status_values(status) -> None:
    with pytest.raises(PhaseError):
        make_attempt(status=status)  # type: ignore[arg-type]


def test_inconsistent_status_availability_rejected() -> None:
    with pytest.raises(PhaseError):
        make_attempt(status="partial")  # all available must be complete
    stages = full_stages()
    stages["finish"] = unavail_stage()
    with pytest.raises(PhaseError):
        make_attempt(stages=stages, status="complete")
    with pytest.raises(PhaseError):
        make_attempt(stages=stages, status="unavailable")
    with pytest.raises(PhaseError):
        make_attempt(stages=stages, status="incomplete")
    all_unavail = {key: unavail_stage() for key in STAGE_ORDER}
    with pytest.raises(PhaseError):
        make_attempt(
            stages=all_unavail, status="partial", attempt_range=span(0.0, 8.0)
        )


@pytest.mark.parametrize(
    "anomalies", [["dup", "dup"], ["", ], ["ok", ""], ["ok", 7], "truncated", [True]]
)
def test_invalid_anomalies(anomalies) -> None:
    with pytest.raises(PhaseError):
        make_attempt(anomalies=anomalies)  # type: ignore[arg-type]


def test_duplicate_evidence_limitations_rejected() -> None:
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, evidence=("a", "a"))
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, limitations=("l", "l"))
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, evidence=("  ",))
    with pytest.raises(PhaseError):
        avail_stage(1.0, 2.0, evidence="cue")  # type: ignore[arg-type]


# --- Versions/keys/JSON --------------------------------------------------------------


def test_version_rejection() -> None:
    stage = avail_stage(1.0, 2.0)
    with pytest.raises(SchemaVersionError):
        StagePhase.from_dict({**stage.to_dict(), "schema_version": 2})
    with pytest.raises(SchemaVersionError):
        StagePhase.from_dict({**stage.to_dict(), "schema_version": 0})
    attempt = make_attempt()
    with pytest.raises(SchemaVersionError):
        AttemptPhase.from_dict({**attempt.to_dict(), "schema_version": 99})
    doc = make_doc(attempt)
    with pytest.raises(SchemaVersionError):
        PhaseDocument.from_dict({**doc.to_dict(), "schema_version": 2})
    with pytest.raises(SchemaVersionError):
        PhaseDocument.from_dict({**doc.to_dict(), "schema_version": 0})
    nested = attempt.to_dict()
    nested["stages"]["start"] = {**nested["stages"]["start"], "schema_version": 99}
    with pytest.raises((SchemaVersionError, PhaseError)):
        AttemptPhase.from_dict(nested)


def test_missing_and_unknown_keys_rejected() -> None:
    stage = avail_stage(1.0, 2.0).to_dict()
    missing = dict(stage)
    del missing["confidence"]
    with pytest.raises(PhaseError):
        StagePhase.from_dict(missing)
    with pytest.raises(PhaseError):
        StagePhase.from_dict({**stage, "extra": 1})
    attempt = make_attempt().to_dict()
    with pytest.raises(PhaseError):
        AttemptPhase.from_dict({**attempt, "bogus": True})
    incomplete = dict(attempt)
    del incomplete["stages"]
    with pytest.raises(PhaseError):
        AttemptPhase.from_dict(incomplete)
    doc = make_doc(make_attempt()).to_dict()
    with pytest.raises(PhaseError):
        PhaseDocument.from_dict({**doc, "extra": 0})
    with pytest.raises(PhaseError):
        AttemptPhase.from_dict({**attempt, "stages": {"start": {}}})
    with pytest.raises(PhaseError):
        AttemptPhase.from_dict(
            {**attempt, "stages": {**attempt["stages"], "toss": {}}}
        )


def test_malformed_json_rejected() -> None:
    with pytest.raises(DomainError):
        StagePhase.from_json("{oops")
    with pytest.raises(DomainError):
        AttemptPhase.from_json("not json")
    with pytest.raises(DomainError):
        PhaseDocument.from_json(b"{bad")
    with pytest.raises(DomainError):
        PhaseDocument.from_json("[1,2]")
    with pytest.raises(DomainError):
        StagePhase.from_json("[1]")


def test_source_bounds_unique_sorted_non_overlapping() -> None:
    good = make_attempt()
    with pytest.raises(PhaseError):
        make_doc(good, good)  # duplicate id + overlap
    second_stages: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = 20.0 + i, 21.0 + i
        if key == "contact":
            second_stages[key] = avail_stage(
                s, e, keyframe=s + 0.5, provenance="audio_transient",
                uncertainty=0.05, evidence=("audio_peak",),
                limitations=("audio_anchored_not_visual",),
            )
        elif i % 2 == 0:
            second_stages[key] = avail_stage(s, e, keyframe=s + 0.5)
        else:
            second_stages[key] = avail_stage(s, e)
    second = make_attempt(
        attempt_id="serve-002", attempt_range=span(20.0, 28.0),
        stages=second_stages,
    )
    doc = make_doc(good, second)
    assert [a.attempt_id for a in doc] == ["serve-001", "serve-002"]
    # Overlap: second range starts inside the first attempt.
    overlap_stages: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = 16.0 + i, 17.0 + i
        if key == "contact":
            overlap_stages[key] = avail_stage(
                s, e, keyframe=s + 0.5, provenance="audio_transient",
                uncertainty=0.05, evidence=("audio_peak",),
                limitations=("audio_anchored_not_visual",),
            )
        else:
            overlap_stages[key] = avail_stage(s, e)
    overlap = make_attempt(
        attempt_id="serve-002", attempt_range=span(16.0, 24.0),
        stages=overlap_stages,
    )
    with pytest.raises(PhaseError):
        make_doc(good, overlap)
    # Unsorted.
    with pytest.raises(PhaseError):
        make_doc(second, good)
    # Beyond source duration.
    far_stages: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = 119.0 + i * 0.25, 119.25 + i * 0.25
        if key == "contact":
            far_stages[key] = avail_stage(
                s, e, keyframe=s + 0.125, provenance="audio_transient",
                uncertainty=0.02, evidence=("audio_peak",),
                limitations=("audio_anchored_not_visual",),
            )
        else:
            far_stages[key] = avail_stage(s, e)
    far = make_attempt(
        attempt_id="serve-001", attempt_range=span(119.0, 121.0),
        stages=far_stages,
    )
    with pytest.raises(PhaseError):
        PhaseDocument(
            source_fingerprint="sha256:x",
            source_duration_seconds=120.0,
            attempts=(far,),
        )
    with pytest.raises(PhaseError):
        PhaseDocument(
            source_fingerprint="  ",
            source_duration_seconds=120.0,
            attempts=(),
        )
    with pytest.raises(PhaseError):
        PhaseDocument(
            source_fingerprint="sha256:x",
            source_duration_seconds=0.0,
            attempts=(),
        )


# --- Immutability/aliasing -------------------------------------------------------------


def test_no_mutation_or_aliasing_surprises() -> None:
    attempt = make_attempt()
    with pytest.raises(dataclasses.FrozenInstanceError):
        attempt.structural_status = "partial"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        attempt.stage("start").confidence = 0.1  # type: ignore[misc]
    # Exposed stage mapping is read-only: normal dict mutation is forbidden.
    assert isinstance(attempt.stages, MappingProxyType)
    with pytest.raises(TypeError):
        attempt.stages["start"] = unavail_stage()  # type: ignore[index]
    with pytest.raises((TypeError, AttributeError)):
        attempt.stages.pop("start")  # type: ignore[attr-defined]

    raw_stages = full_stages()
    attempt2 = make_attempt(stages=raw_stages)
    raw_stages["start"] = unavail_stage()
    assert attempt2.stage("start").availability == "available"

    evidence = ["cue_a", "cue_b"]
    stage = avail_stage(1.0, 2.0, evidence=evidence)  # type: ignore[arg-type]
    evidence.append("cue_c")
    assert stage.evidence == ("cue_a", "cue_b")

    payload = attempt.to_dict()
    payload["stages"]["start"]["confidence"] = 0.01
    assert attempt.stage("start").confidence == pytest.approx(0.8)

    raw_list = [attempt]
    doc = make_doc(*raw_list)
    raw_list.clear()
    assert len(doc) == 1
    assert doc[0] == attempt
    assert list(doc) == [attempt]
