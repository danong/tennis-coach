"""Focused M4.6 tests for manual-gate checkpoint evaluation.

Deterministic, offline, synthetic only: every interval, keyframe, and
expected scalar is hand-constructed with hand-calculated expectations.
No private footage, manifests, or solver tuning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from serve_review.checkpoint_evaluation import (
    CHECKPOINT_ANNOTATION_SCHEMA_VERSION,
    CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION,
    CheckpointEvaluationError,
    CheckpointEvaluationReport,
    PhaseAnnotation,
    PhaseAnnotationManifest,
    evaluate_phase_document,
    interval_intersection_seconds,
    interval_iou,
    median_of_sorted,
    nearest_rank_percentile,
    validate_session_disjoint,
)
from serve_review.domain import STAGE_ORDER, AttemptPhase, MediaRange, PhaseDocument, StagePhase

FIXTURES = Path(__file__).parent / "fixtures" / "checkpoint_annotations"


# --- builders ---------------------------------------------------------------

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
) -> StagePhase:
    return StagePhase(
        availability=availability,
        provenance=provenance,
        confidence=confidence,
        interval=span(start, end),
        keyframe_seconds=keyframe,
        temporal_uncertainty_seconds=uncertainty,
        evidence=("cue_a",),
        limitations=("body_pose_estimate",),
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


def full_stages(a0: float, *, contact_conf: float = 0.8) -> dict[str, StagePhase]:
    stages: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = a0 + i, a0 + i + 1
        if key == "contact":
            stages[key] = avail_stage(
                s, e, keyframe=s + 0.5, provenance="audio_transient",
                confidence=contact_conf, uncertainty=0.05,
            )
        else:
            stages[key] = avail_stage(s, e, keyframe=s + 0.5)
    return stages


def make_attempt(
    attempt_id: str,
    a0: float,
    stages: dict[str, StagePhase] | None = None,
    *,
    status: str = "complete",
) -> AttemptPhase:
    return AttemptPhase(
        attempt_id=attempt_id,
        attempt_range=span(a0, a0 + 8),
        method_version="phase-test-v1",
        config_id="cfg-001",
        stages=stages if stages is not None else full_stages(a0),
        structural_status=status,
        anomalies=(),
    )


def make_doc(*attempts: AttemptPhase) -> PhaseDocument:
    return PhaseDocument(
        source_fingerprint="sha256:checkpoint-test",
        source_duration_seconds=500.0,
        attempts=list(attempts),
    )


def row(
    session: str,
    media: str,
    aid: str,
    a0: float,
    stage: str,
    status: str,
    istart: float | None,
    iend: float | None,
    kframe: float | None = None,
    conf: float | None = None,
    label: str | None = "serve",
) -> PhaseAnnotation:
    return PhaseAnnotation(
        session_id=session,
        media=media,
        attempt_id=aid,
        attempt_start_seconds=a0,
        attempt_end_seconds=a0 + 8,
        stage=stage,
        status=status,
        interval_start_seconds=istart,
        interval_end_seconds=iend,
        manual_keyframe_seconds=kframe,
        confidence=conf,
        attempt_label=label,
    )


def full_rows(session: str, media: str, aid: str, a0: float, label: str | None = "serve") -> list[PhaseAnnotation]:
    out: list[PhaseAnnotation] = []
    for i, key in enumerate(STAGE_ORDER):
        s, e = a0 + i, a0 + i + 1
        out.append(row(session, media, aid, a0, key, "available", s, e, s + 0.5, 0.7, label))
    return out


def manifest(split: str, entries: list[PhaseAnnotation]) -> PhaseAnnotationManifest:
    return PhaseAnnotationManifest(
        schema_version=CHECKPOINT_ANNOTATION_SCHEMA_VERSION,
        dataset_split=split,
        annotations=tuple(entries),
    )


# --- helpers: interval math -------------------------------------------------

def test_interval_math_hand_calculated() -> None:
    assert interval_intersection_seconds(10.0, 11.0, 10.2, 10.9) == pytest.approx(0.7)
    assert interval_intersection_seconds(0.0, 1.0, 1.0, 2.0) == pytest.approx(0.0)
    # inter 0.7, union 1.0 -> 0.7
    assert interval_iou(10.0, 11.0, 10.2, 10.9) == pytest.approx(0.7)
    assert interval_iou(0.0, 1.0, 1.0, 2.0) == pytest.approx(0.0)
    assert interval_iou(0.0, 2.0, 0.0, 2.0) == pytest.approx(1.0)
    # inter 1.0, union 3.0 -> 1/3
    assert interval_iou(0.0, 2.0, 1.0, 3.0) == pytest.approx(1.0 / 3.0)


def test_median_and_percentile_hand_calculated() -> None:
    assert median_of_sorted([]) is None
    assert median_of_sorted([0.2]) == pytest.approx(0.2)
    assert median_of_sorted([0.1, 0.2, 0.3]) == pytest.approx(0.2)
    # even n: mean of two middles (0.2+0.3)/2 = 0.25
    assert median_of_sorted([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.25)
    assert nearest_rank_percentile([], 0.9) is None
    # n=3: ceil(0.9*3)=3 -> index 2 -> 0.3
    assert nearest_rank_percentile([0.1, 0.2, 0.3], 0.9) == pytest.approx(0.3)
    # n=10: ceil(9.0)=9 -> index 8 -> 9.0
    assert nearest_rank_percentile([float(i) for i in range(1, 11)], 0.9) == pytest.approx(9.0)
    # median as q=0.5 nearest-rank would give index 0 for n=2; median differs by design
    assert nearest_rank_percentile([0.1, 0.9], 0.5) == pytest.approx(0.1)
    with pytest.raises(CheckpointEvaluationError):
        nearest_rank_percentile([0.1], 0.0)
    with pytest.raises(CheckpointEvaluationError):
        nearest_rank_percentile([0.1], 1.5)


# --- accepted interval / keyframe cases -------------------------------------

def test_accepted_keyframe_inside_interval() -> None:
    # Manual start [10,11) kf 10.5; machine [10.2,10.9) kf 10.6 conf 0.8.
    a0 = 10.0
    stages = full_stages(a0)
    stages["start"] = avail_stage(10.2, 10.9, keyframe=10.6, confidence=0.8)
    doc = make_doc(make_attempt("serve-001", a0, stages))
    rows = full_rows("sess-a", "sess-a/cap01.mov", "serve-001", a0)
    man = manifest("dev", rows)
    report = evaluate_phase_document(doc, man)
    start = report.stage("start")
    assert start.n_keyframe_evaluated == 1
    assert start.n_keyframe_accepted == 1
    assert start.accepted_keyframe_rate == pytest.approx(1.0)
    assert start.mean_iou == pytest.approx(0.7)
    assert start.mean_intersection_seconds == pytest.approx(0.7)
    assert start.mean_signed_error_seconds == pytest.approx(0.1)
    assert start.mean_abs_error_seconds == pytest.approx(0.1)
    detail = report.attempts[0].stages[0]
    assert detail.accepted is True
    assert detail.iou == pytest.approx(0.7)
    assert detail.signed_error_seconds == pytest.approx(0.1)
    assert detail.abs_error_seconds == pytest.approx(0.1)


def test_rejected_keyframe_outside_interval() -> None:
    # Manual start [10,11) kf 10.5; machine [11,12) kf 11.5 -> outside.
    # Other machine stages shift accordingly to stay chronological:
    # release..finish occupy [12,19)? That breaks attempt range [10,18).
    # Instead use a dedicated attempt range [10,20) is illegal (stages must
    # tile 8s). So build a custom attempt range [10,20) with 10s span and
    # 8 stages placed chronologically inside it.
    a0, a1 = 10.0, 20.0
    stages: dict[str, StagePhase] = {}
    machine_layout = [(11.0, 12.0, 11.5), (12.0, 13.0, 12.5), (13.0, 14.0, None),
                      (14.0, 15.0, None), (15.0, 16.0, None), (16.0, 17.0, 16.5),
                      (17.0, 18.0, None), (18.0, 19.0, None)]
    for key, (s, e, k) in zip(STAGE_ORDER, machine_layout):
        if key == "contact":
            stages[key] = avail_stage(s, e, keyframe=k, provenance="audio_transient",
                                      confidence=0.8, uncertainty=0.05)
        elif k is None:
            stages[key] = avail_stage(s, e)
        else:
            stages[key] = avail_stage(s, e, keyframe=k)
    attempt = AttemptPhase(
        attempt_id="serve-001", attempt_range=span(a0, a1),
        method_version="phase-test-v1", config_id="cfg-001",
        stages=stages, structural_status="complete", anomalies=(),
    )
    doc = make_doc(attempt)
    rows: list[PhaseAnnotation] = []
    manual_layout = [(10.0, 11.0, 10.5), (12.0, 13.0, 12.5), (13.0, 14.0, None),
                     (14.0, 15.0, None), (15.0, 16.0, None), (16.0, 17.0, 16.5),
                     (17.0, 18.0, None), (18.0, 19.0, None)]
    for key, (s, e, k) in zip(STAGE_ORDER, manual_layout):
        rows.append(PhaseAnnotation(
            session_id="sess-a", media="sess-a/cap01.mov", attempt_id="serve-001",
            attempt_start_seconds=a0, attempt_end_seconds=a1, stage=key,
            status="available", interval_start_seconds=s, interval_end_seconds=e,
            manual_keyframe_seconds=k, confidence=0.7, attempt_label="serve"))
    report = evaluate_phase_document(doc, manifest("dev", rows))
    start = report.stage("start")
    assert start.n_keyframe_evaluated == 1
    assert start.n_keyframe_accepted == 0
    assert start.accepted_keyframe_rate == pytest.approx(0.0)
    assert start.mean_iou == pytest.approx(0.0)
    assert start.mean_intersection_seconds == pytest.approx(0.0)
    assert start.mean_signed_error_seconds == pytest.approx(1.0)
    assert start.mean_abs_error_seconds == pytest.approx(1.0)
    assert report.attempts[0].stages[0].accepted is False


# --- timing signs / errors / percentiles ------------------------------------

def test_timing_signs_and_percentiles_hand_calculated() -> None:
    # Errors +0.1, -0.2, +0.3 over start/release/loading; rest unavailable.
    a0 = 0.0
    stages: dict[str, StagePhase] = {}
    keyframes = {"start": 0.6, "release": 1.3, "loading": 2.8}
    for i, key in enumerate(STAGE_ORDER):
        s, e = a0 + i, a0 + i + 1
        if key in keyframes:
            if key == "contact":
                raise AssertionError("contact not in this trio")
            stages[key] = avail_stage(s, e, keyframe=keyframes[key], confidence=0.8)
        else:
            prov = "audio_transient" if key == "contact" else "body_pose"
            stages[key] = unavail_stage(provenance=prov)
    doc = make_doc(make_attempt("serve-001", a0, stages, status="partial"))
    rows: list[PhaseAnnotation] = []
    for i, key in enumerate(STAGE_ORDER):
        s, e = a0 + i, a0 + i + 1
        if key in ("start", "release", "loading"):
            rows.append(row("sess-a", "sess-a/cap01.mov", "serve-001", a0, key,
                            "available", s, e, s + 0.5, 0.7, "serve"))
        else:
            rows.append(row("sess-a", "sess-a/cap01.mov", "serve-001", a0, key,
                            "unavailable", None, None, None, None, "serve"))
    report = evaluate_phase_document(doc, manifest("dev", rows))
    overall = report.overall
    assert overall.n_timing_evaluated == 3
    assert overall.mean_signed_error_seconds == pytest.approx(0.2 / 3.0)
    assert overall.mean_abs_error_seconds == pytest.approx(0.2)
    assert overall.median_abs_error_seconds == pytest.approx(0.2)
    assert overall.p90_abs_error_seconds == pytest.approx(0.3)
    assert report.stage("start").mean_signed_error_seconds == pytest.approx(0.1)
    assert report.stage("release").mean_signed_error_seconds == pytest.approx(-0.2)
    assert report.stage("loading").mean_abs_error_seconds == pytest.approx(0.3)
    # Even-n median: add a fourth error +0.4 via second attempt on stage start.
    a1 = 100.0
    stages2: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = a1 + i, a1 + i + 1
        if key == "start":
            stages2[key] = avail_stage(s, e, keyframe=s + 0.9, confidence=0.8)
        else:
            prov = "audio_transient" if key == "contact" else "body_pose"
            stages2[key] = unavail_stage(provenance=prov)
    doc2 = make_doc(
        make_attempt("serve-001", a0, stages, status="partial"),
        make_attempt("serve-002", a1, stages2, status="partial"),
    )
    rows2 = list(rows)
    for i, key in enumerate(STAGE_ORDER):
        s, e = a1 + i, a1 + i + 1
        if key == "start":
            rows2.append(row("sess-a", "sess-a/cap01.mov", "serve-002", a1, key,
                             "available", s, e, s + 0.5, 0.7, "serve"))
        else:
            rows2.append(row("sess-a", "sess-a/cap01.mov", "serve-002", a1, key,
                             "unavailable", None, None, None, None, "serve"))
    report2 = evaluate_phase_document(doc2, manifest("dev", rows2))
    start2 = report2.stage("start")
    # abs errors [0.1, 0.4] -> median 0.25
    assert start2.n_timing_evaluated == 2
    assert start2.median_abs_error_seconds == pytest.approx(0.25)
    assert start2.p90_abs_error_seconds == pytest.approx(0.4)


# --- availability / missing / ambiguous -------------------------------------

def test_availability_missing_and_ambiguous_honest_nulls() -> None:
    a0 = 0.0
    stages: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = a0 + i, a0 + i + 1
        if key == "start":
            stages[key] = unavail_stage()  # missing vs manual available
        elif key == "release":
            stages[key] = avail_stage(s, e, keyframe=s + 0.5, confidence=0.9)
        elif key == "loading":
            stages[key] = avail_stage(s, e, keyframe=s + 0.5, confidence=0.4)
        else:
            prov = "audio_transient" if key == "contact" else "body_pose"
            stages[key] = unavail_stage(provenance=prov)
    # 2 available + 6 unavailable -> partial
    doc = make_doc(make_attempt("serve-001", a0, stages, status="partial"))
    rows: list[PhaseAnnotation] = [
        row("sess-a", "sess-a/cap01.mov", "serve-001", a0, "start",
            "available", 0.0, 1.0, 0.5, 0.7, "serve"),
        row("sess-a", "sess-a/cap01.mov", "serve-001", a0, "release",
            "unavailable", None, None, None, None, "serve"),
        row("sess-a", "sess-a/cap01.mov", "serve-001", a0, "loading",
            "ambiguous", None, None, None, None, "serve"),
    ]
    for i, key in enumerate(STAGE_ORDER[3:]):
        rows.append(row("sess-a", "sess-a/cap01.mov", "serve-001", a0, key,
                        "unavailable", None, None, None, None, "serve"))
    report = evaluate_phase_document(doc, manifest("dev", rows))
    start = report.stage("start")
    assert start.n_annotated_available == 1
    assert start.n_predicted_unavailable == 1
    assert start.n_machine_missing_manual_available == 1
    assert start.n_keyframe_evaluated == 0
    assert start.accepted_keyframe_rate is None
    assert start.mean_iou is None
    assert report.attempts[0].stages[0].accepted is None
    assert report.attempts[0].stages[0].iou is None
    release = report.stage("release")
    # manual unavailable + machine available: honest extra, no scoring
    assert release.n_annotated_unavailable == 1
    assert release.n_predicted_available == 1
    assert release.n_machine_available_manual_not_available == 1
    assert release.n_keyframe_evaluated == 0
    assert report.attempts[0].stages[1].accepted is None
    loading = report.stage("loading")
    assert loading.n_annotated_ambiguous == 1
    assert loading.n_machine_available_manual_not_available == 1
    assert loading.n_keyframe_evaluated == 0


def test_partial_and_fully_unavailable_documents() -> None:
    a0 = 0.0
    partial_stages: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = a0 + i, a0 + i + 1
        if key in ("start", "release"):
            partial_stages[key] = avail_stage(s, e, keyframe=s + 0.5)
        else:
            prov = "audio_transient" if key == "contact" else "body_pose"
            partial_stages[key] = unavail_stage(provenance=prov)
    doc = make_doc(make_attempt("serve-001", a0, partial_stages, status="partial"))
    report = evaluate_phase_document(doc, manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", a0)))
    assert report.structural_partial == 1
    assert report.overall.n_machine_missing_manual_available == 6
    assert report.overall.n_both_available == 2

    empty_stages = {k: unavail_stage(provenance=("audio_transient" if k == "contact" else "body_pose")) for k in STAGE_ORDER}
    doc2 = make_doc(make_attempt("serve-001", a0, empty_stages, status="unavailable"))
    report2 = evaluate_phase_document(doc2, manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", a0)))
    assert report2.structural_unavailable == 1
    assert report2.overall.accepted_keyframe_rate is None
    assert report2.overall.mean_iou is None
    assert report2.overall.mean_abs_error_seconds is None
    assert report2.n_calibrated == 0
    assert report2.brier_score is None
    assert all(b.count == 0 for b in report2.calibration_bins)

    # incomplete: one partial-availability machine stage, rest unavailable
    mixed: dict[str, StagePhase] = {}
    for i, key in enumerate(STAGE_ORDER):
        s, e = a0 + i, a0 + i + 1
        if key == "start":
            mixed[key] = avail_stage(s, e, keyframe=s + 0.5, availability="partial", confidence=0.3)
        else:
            prov = "audio_transient" if key == "contact" else "body_pose"
            mixed[key] = unavail_stage(provenance=prov)
    doc3 = make_doc(make_attempt("serve-001", a0, mixed, status="incomplete"))
    report3 = evaluate_phase_document(doc3, manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", a0)))
    assert report3.structural_incomplete == 1
    assert report3.stage("start").n_predicted_available == 1


# --- ordering violations ----------------------------------------------------

def test_manual_order_violation_flagged_machine_clean() -> None:
    a0 = 0.0
    doc = make_doc(make_attempt("serve-001", a0, full_stages(a0)))
    rows = full_rows("sess-a", "sess-a/cap01.mov", "serve-001", a0)
    # Make release overlap start: release [0.5,1.5) starts before start ends.
    bad = [r for r in rows if r.stage != "release"]
    bad.append(row("sess-a", "sess-a/cap01.mov", "serve-001", a0, "release",
                   "available", 0.5, 1.5, 1.0, 0.7, "serve"))
    report = evaluate_phase_document(doc, manifest("dev", bad))
    manual = [v for v in report.order_violations if v.side == "manual"]
    machine = [v for v in report.order_violations if v.side == "machine"]
    assert len(manual) >= 1
    assert manual[0].attempt_id == "serve-001"
    assert machine == []


# --- confidence calibration -------------------------------------------------

def test_confidence_calibration_brier_and_bins_hand_calculated() -> None:
    # 4 attempts, one evaluable 'start' pair each.
    # (conf, accepted): (0.1,T),(0.9,T),(0.8,F),(0.3,F) -> Brier 0.3875.
    confs = [0.1, 0.9, 0.8, 0.3]
    accepted_flags = [True, True, False, False]
    attempts = []
    rows: list[PhaseAnnotation] = []
    for n, (conf, ok) in enumerate(zip(confs, accepted_flags)):
        aid = f"serve-{n + 1:03d}"
        a0 = float(n * 10)
        stages: dict[str, StagePhase] = {}
        kf = a0 + 0.5 if ok else a0 + 0.9
        for i, key in enumerate(STAGE_ORDER):
            s, e = a0 + i, a0 + i + 1
            if key == "start":
                stages[key] = avail_stage(s, e, keyframe=kf, confidence=conf)
            else:
                prov = "audio_transient" if key == "contact" else "body_pose"
                stages[key] = unavail_stage(provenance=prov)
        attempts.append(make_attempt(aid, a0, stages, status="partial"))
        for i, key in enumerate(STAGE_ORDER):
            s, e = a0 + i, a0 + i + 1
            if key == "start":
                # narrow manual interval so kf=0.9 rejects
                rows.append(row("sess-a", "sess-a/cap01.mov", aid, a0, key,
                                "available", a0 + 0.4, a0 + 0.6, a0 + 0.5, 0.7, "serve"))
            else:
                rows.append(row("sess-a", "sess-a/cap01.mov", aid, a0, key,
                                "unavailable", None, None, None, None, "serve"))
    report = evaluate_phase_document(make_doc(*attempts), manifest("dev", rows))
    assert report.n_calibrated == 4
    assert report.brier_score == pytest.approx(0.3875)
    bins = {b.bin_index: b for b in report.calibration_bins}
    assert bins[0].count == 1
    assert bins[0].mean_confidence == pytest.approx(0.1)
    assert bins[0].mean_accuracy == pytest.approx(1.0)
    assert bins[0].abs_gap == pytest.approx(0.9)
    assert bins[1].count == 1
    assert bins[1].mean_confidence == pytest.approx(0.3)
    assert bins[1].mean_accuracy == pytest.approx(0.0)
    assert bins[1].abs_gap == pytest.approx(0.3)
    assert bins[2].count == 0 and bins[2].mean_confidence is None
    assert bins[3].count == 0 and bins[3].mean_confidence is None
    assert bins[4].count == 2
    assert bins[4].mean_confidence == pytest.approx(0.85)
    assert bins[4].mean_accuracy == pytest.approx(0.5)
    assert bins[4].abs_gap == pytest.approx(0.35)


# --- serve versus aborted separation ----------------------------------------

def test_serve_aborted_structural_separation() -> None:
    doc = make_doc(
        make_attempt("serve-001", 0.0, full_stages(0.0)),
        make_attempt("serve-002", 100.0,
                     {k: unavail_stage(provenance=("audio_transient" if k == "contact" else "body_pose")) for k in STAGE_ORDER},
                     status="unavailable"),
    )
    rows = full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0, "serve") + \
        full_rows("sess-b", "sess-b/cap02.mov", "serve-002", 100.0, "aborted")
    report = evaluate_phase_document(doc, manifest("dev", rows))
    assert report.separation is not None
    assert report.separation.serve.count == 1
    assert report.separation.serve.complete == 1
    assert report.separation.serve.complete_rate == pytest.approx(1.0)
    assert report.separation.aborted.count == 1
    assert report.separation.aborted.unavailable == 1
    assert report.separation.aborted.complete_rate == pytest.approx(0.0)
    assert report.structural_complete == 1
    assert report.structural_unavailable == 1


def test_separation_none_when_ungraded() -> None:
    doc = make_doc(make_attempt("serve-001", 0.0, full_stages(0.0)))
    report = evaluate_phase_document(
        doc, manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0, None)))
    assert report.separation is None


# --- manifest validation ----------------------------------------------------

def test_manifest_round_trip_deterministic() -> None:
    original = manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0))
    assert PhaseAnnotationManifest.from_dict(original.to_dict()) == original
    assert PhaseAnnotationManifest.from_json(original.to_json()) == original
    assert PhaseAnnotationManifest.from_json(original.to_json().encode("utf-8")) == original
    assert original.to_json() == manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0)).to_json()
    assert list(json.loads(original.to_json())) == sorted(json.loads(original.to_json()))
    assert original.sessions == ("sess-a",)
    assert original.attempt_ids == ("serve-001",)


def test_schema_versions_pinned() -> None:
    assert CHECKPOINT_ANNOTATION_SCHEMA_VERSION == 1
    assert CHECKPOINT_EVALUATION_REPORT_SCHEMA_VERSION == 1


def test_manifest_rejects_absolute_and_traversal_paths() -> None:
    for bad in ("/abs/cap.mov", "../outside/cap.mov", "a\\b.mov", "C:\\x.mov", "C:/x.mov"):
        with pytest.raises(CheckpointEvaluationError):
            row("sess-a", bad, "serve-001", 0.0, "start", "available", 0.0, 1.0, 0.5)
    with pytest.raises(CheckpointEvaluationError):
        row("  ", "sess-a/cap.mov", "serve-001", 0.0, "start", "available", 0.0, 1.0, 0.5)


def test_manifest_rejects_bad_versions_and_splits() -> None:
    good = manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0))
    payload = good.to_dict()
    with pytest.raises(CheckpointEvaluationError):
        PhaseAnnotationManifest.from_dict({**payload, "schema_version": 999})
    with pytest.raises(CheckpointEvaluationError):
        PhaseAnnotationManifest.from_dict({**payload, "schema_version": 0})
    with pytest.raises(CheckpointEvaluationError):
        PhaseAnnotationManifest.from_dict({**payload, "dataset_split": "test"})
    with pytest.raises(CheckpointEvaluationError):
        PhaseAnnotationManifest.from_dict({**payload, "extra": 1})
    with pytest.raises(CheckpointEvaluationError):
        PhaseAnnotationManifest.from_dict(
            {k: v for k, v in payload.items() if k != "annotations"})
    with pytest.raises(CheckpointEvaluationError):
        PhaseAnnotationManifest.from_json("{oops")


def test_manifest_rejects_bad_stage_status_times() -> None:
    with pytest.raises(CheckpointEvaluationError):
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "toss", "available", 0.0, 1.0, 0.5)
    with pytest.raises(CheckpointEvaluationError):
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "weird", 0.0, 1.0, 0.5)
    with pytest.raises(CheckpointEvaluationError):
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "available", 1.0, 1.0, None)
    with pytest.raises(CheckpointEvaluationError):
        # interval outside attempt [0,8)
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "available", 7.5, 8.5, 7.6)
    with pytest.raises(CheckpointEvaluationError):
        # keyframe outside manual interval
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "available", 0.0, 1.0, 1.0)
    with pytest.raises(CheckpointEvaluationError):
        # interval supplied for unavailable
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "unavailable", 0.0, 1.0)
    with pytest.raises(CheckpointEvaluationError):
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "ambiguous", None, None, 0.5)
    with pytest.raises(CheckpointEvaluationError):
        row("sess-a", "sess-a/cap.mov", "bad-id", 0.0, "start", "available", 0.0, 1.0, 0.5)
    with pytest.raises(CheckpointEvaluationError):
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "available", 0.0, 1.0, 0.5, 1.5)
    with pytest.raises(CheckpointEvaluationError):
        row("sess-a", "sess-a/cap.mov", "serve-001", 0.0, "start", "available", 0.0, 1.0, 0.5, None, "weird")


def test_duplicate_and_conflicting_annotations_rejected() -> None:
    base = full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0)
    with pytest.raises(CheckpointEvaluationError, match="duplicate"):
        manifest("dev", base + [base[0]])
    altered = list(base)
    altered.append(row("sess-a", "sess-a/cap01.mov", "serve-001", 0.0, "start",
                       "available", 0.0, 0.9, 0.5, 0.7, "serve"))
    with pytest.raises(CheckpointEvaluationError, match="conflicting"):
        manifest("dev", altered)
    linkage = [r for r in base if not (r.attempt_id == "serve-001" and r.stage == "start")]
    linkage.append(row("sess-other", "other/cap.mov", "serve-001", 0.0, "start",
                       "available", 0.0, 1.0, 0.5, 0.7, "serve"))
    with pytest.raises(CheckpointEvaluationError, match="linkage"):
        manifest("dev", linkage)


def test_session_disjoint_and_leakage() -> None:
    dev = manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0))
    heldout = manifest("heldout", full_rows("sess-c", "sess-c/cap03.mov", "serve-001", 6.0))
    validate_session_disjoint(dev, heldout)
    leaked = manifest("heldout", full_rows("sess-a", "sess-a/other.mov", "serve-009", 50.0))
    with pytest.raises(CheckpointEvaluationError, match="sess-a"):
        validate_session_disjoint(dev, leaked)
    with pytest.raises(CheckpointEvaluationError):
        validate_session_disjoint(dev, "nope")  # type: ignore[arg-type]


# --- linkage / mismatch -----------------------------------------------------

def test_range_linkage_mismatch_rejected() -> None:
    doc = make_doc(make_attempt("serve-001", 0.0, full_stages(0.0)))
    rows = full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.5)
    with pytest.raises(CheckpointEvaluationError, match="linkage"):
        evaluate_phase_document(doc, manifest("dev", rows))


def test_attempt_mismatch_rejected() -> None:
    doc = make_doc(make_attempt("serve-001", 0.0, full_stages(0.0)))
    # Unknown attempt id.
    with pytest.raises(CheckpointEvaluationError, match="mismatch"):
        evaluate_phase_document(doc, manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-009", 0.0)))
    # Document has an extra attempt absent from the manifest.
    doc2 = make_doc(make_attempt("serve-001", 0.0, full_stages(0.0)),
                    make_attempt("serve-002", 100.0, full_stages(100.0)))
    with pytest.raises(CheckpointEvaluationError, match="mismatch"):
        evaluate_phase_document(doc2, manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0)))
    # Manifest missing one of eight stages.
    short = [r for r in full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0) if r.stage != "finish"]
    with pytest.raises(CheckpointEvaluationError, match="eight"):
        evaluate_phase_document(doc, manifest("dev", short))
    with pytest.raises(CheckpointEvaluationError):
        evaluate_phase_document(doc, manifest("dev", []))
    with pytest.raises(CheckpointEvaluationError):
        evaluate_phase_document("nope", manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0)))  # type: ignore[arg-type]


# --- tie determinism + report JSON ------------------------------------------

def test_tie_determinism_and_report_json() -> None:
    doc = make_doc(make_attempt("serve-001", 0.0, full_stages(0.0)))
    man = manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0))
    first = evaluate_phase_document(doc, man)
    second = evaluate_phase_document(doc, man)
    assert first == second
    assert first.to_json() == second.to_json()
    assert CheckpointEvaluationReport.from_dict(first.to_dict()) == first
    assert CheckpointEvaluationReport.from_json(first.to_json()) == first
    assert CheckpointEvaluationReport.from_json(first.to_json().encode("utf-8")) == first
    payload = json.loads(first.to_json())
    assert list(payload) == sorted(payload)
    assert [e["stage"] for e in payload["per_stage"]] == list(STAGE_ORDER)
    assert [a["attempt_id"] for a in payload["attempts"]] == ["serve-001"]
    # Input ordering must not affect the report: shuffled manifest rows.
    shuffled = list(manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0)).annotations)[::-1]
    third = evaluate_phase_document(doc, manifest("dev", shuffled))
    assert third.to_json() == first.to_json()


def test_report_rejects_tampered_versions_and_keys() -> None:
    doc = make_doc(make_attempt("serve-001", 0.0, full_stages(0.0)))
    report = evaluate_phase_document(doc, manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0)))
    with pytest.raises(CheckpointEvaluationError):
        CheckpointEvaluationReport.from_dict({**report.to_dict(), "schema_version": 999})
    with pytest.raises(CheckpointEvaluationError):
        CheckpointEvaluationReport.from_dict({**report.to_dict(), "extra": 1})
    with pytest.raises(CheckpointEvaluationError):
        CheckpointEvaluationReport.from_dict(
            {k: v for k, v in report.to_dict().items() if k != "overall"})
    with pytest.raises(CheckpointEvaluationError):
        CheckpointEvaluationReport.from_json("{oops")
    with pytest.raises(KeyError):
        report.stage("toss")


def test_never_alters_phase_document() -> None:
    attempt = make_attempt("serve-001", 0.0, full_stages(0.0))
    doc = make_doc(attempt)
    before = doc.to_json()
    man = manifest("dev", full_rows("sess-a", "sess-a/cap01.mov", "serve-001", 0.0))
    evaluate_phase_document(doc, man)
    assert doc.to_json() == before
    assert doc[0] == attempt


# --- fixtures ----------------------------------------------------------------

def test_fixture_manifests_valid_and_disjoint() -> None:
    dev = PhaseAnnotationManifest.from_json((FIXTURES / "dev.json").read_text())
    heldout = PhaseAnnotationManifest.from_json((FIXTURES / "heldout.json").read_text())
    assert dev.dataset_split == "dev"
    assert heldout.dataset_split == "heldout"
    assert dev.sessions == ("sess-a", "sess-b")
    assert heldout.sessions == ("sess-c",)
    validate_session_disjoint(dev, heldout)
    # Build the matching two-attempt document for the dev fixture.
    doc = make_doc(
        make_attempt("serve-001", 10.0, full_stages(10.0)),
        make_attempt("serve-002", 20.0, full_stages(20.0)),
    )
    report = evaluate_phase_document(doc, dev)
    assert report.n_attempts == 2
    assert report.overall.accepted_keyframe_rate == pytest.approx(1.0)
    assert report.structural_complete == 2
    assert report.separation is not None
    assert report.separation.serve.count == 1
    assert report.separation.aborted.count == 1
