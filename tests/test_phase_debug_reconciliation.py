"""Focused tests for M4 Unit 2 solver reconciliation (representation only).

Deterministic, offline, synthetic only; no private footage, no CLI,
no pipeline/rendering, no solver/evidence behavior changes. Covers the
required Unit 2 outcome: the accepted Unit 1 artifact durably explains
the existing DP path (unary/transition/skip shares, predecessor spans,
contact-relative offsets, deterministic explanations) with the artifact
total reconciling to ``PhaseSolverResult.total_score`` within the
documented tolerance, while solver selection is provably unchanged.
"""

from __future__ import annotations

import json

import pytest

from serve_review.checkpoints.evidence import (
    EVIDENCE_DEFAULT_CONFIG_ID,
    EVIDENCE_METHOD_VERSION,
    PhaseEvidence,
    StageCandidate,
)
from serve_review.checkpoints.phase_debug import (
    PHASE_DEBUG_SCHEMA_VERSION,
    RECONCILIATION_METHOD_VERSION,
    RECONCILIATION_OBJECTIVE_TOLERANCE,
    PhaseDebugArtifact,
    PhaseDebugError,
)
from serve_review.checkpoints.phase_debug_builder import (
    build_phase_debug_artifact,
)
from serve_review.checkpoints.phase_debug_reconciliation import (
    explain_solver_path,
)
from serve_review.checkpoints.phase_features import (
    PHASE_FEATURES_METHOD_VERSION,
    PhaseFeatureGrid,
    PhaseFeatureSample,
)
from serve_review.checkpoints.phase_solver import (
    PHASE_SOLVER_METHOD_VERSION,
    PhaseSolverConfig,
    PhaseSolverResult,
    solve_with_diagnostics,
)
from serve_review.domain import (
    STAGE_ORDER,
    Attempt,
    AttemptPhase,
    MediaRange,
    StagePhase,
)

ATTEMPT_START = 10.0
ATTEMPT_END = 12.0
GRID_RATE_HZ = 30.0
CONTACT_STAGE = "contact"


def _attempt_range() -> MediaRange:
    return MediaRange(start_seconds=ATTEMPT_START, end_seconds=ATTEMPT_END)


def _attempt() -> Attempt:
    detected = _attempt_range()
    return Attempt(
        attempt_id="serve-001",
        detected_range=detected,
        effective_range=MediaRange(
            start_seconds=ATTEMPT_START, end_seconds=ATTEMPT_END
        ),
        confidence=0.9,
        evidence={"motion": 1.0},
    )


def _keyframe(index: int) -> float:
    return 10.20 + 0.15 * index


def _grid_sample(time: float) -> PhaseFeatureSample:
    return PhaseFeatureSample(
        time_seconds=time,
        observed=True,
        observation_quality=0.9,
        interpolation_span_seconds=0.0,
        source_time_offset_seconds=0.0,
        temporal_uncertainty_seconds=0.002,
        derivative_confidence=0.8,
        derivative_quality=0.7,
        wrist_left_x=0.10,
        wrist_left_y=-0.20,
        wrist_left_vx=0.50,
        wrist_left_vy=-1.00,
        wrist_right_x=0.20,
        wrist_right_y=-0.30,
        wrist_right_vx=0.40,
        wrist_right_vy=-0.80,
        elbow_angle_right=90.0,
        knee_angle_left=150.0,
        knee_angle_right=140.0,
        shoulder_axis_camera_dx=1.0,
        shoulder_axis_camera_dy=0.1,
        hip_axis_camera_dx=1.0,
        hip_axis_camera_dy=-0.05,
        torso_rotation_camera_deg=5.0,
        audio_energy=0.30,
        audio_candidate=False,
    )


def _grid() -> PhaseFeatureGrid:
    step = 1.0 / GRID_RATE_HZ
    count = int(round((ATTEMPT_END - ATTEMPT_START) / step))
    samples = tuple(
        _grid_sample(ATTEMPT_START + index * step) for index in range(count)
    )
    return PhaseFeatureGrid(
        attempt_range=_attempt_range(),
        method_version=PHASE_FEATURES_METHOD_VERSION,
        config_id="phase-features-default-v2",
        source_frame_rate_hz=240.0,
        pose_observation_rate_hz=GRID_RATE_HZ,
        grid_rate_hz=GRID_RATE_HZ,
        position_window_samples=3,
        derivative_window_samples=3,
        direct_observation_tolerance_seconds=0.015,
        samples=samples,
    )


def _candidate(
    stage: str, keyframe: float, *, score: float = 0.70
) -> StageCandidate:
    return StageCandidate(
        stage=stage,
        keyframe_seconds=keyframe,
        interval=MediaRange(
            start_seconds=keyframe - 0.02, end_seconds=keyframe + 0.02
        ),
        score=score,
        observation_quality=0.8,
        derivative_quality=0.7,
        evidence=("cue_a",),
        limitations=("camera_relative_only",),
        provenance="audio_transient" if stage == "contact" else "body_pose",
        temporal_uncertainty_seconds=0.02,
    )


def _evidence(
    *,
    keyframes: dict[str, float] | None = None,
    scores: dict[str, float] | None = None,
    drop: tuple[str, ...] = (),
) -> PhaseEvidence:
    members: list[StageCandidate] = []
    for index, stage in enumerate(STAGE_ORDER):
        if stage in drop:
            continue
        keyframe = _keyframe(index) if keyframes is None else keyframes[stage]
        score = 0.70 - 0.02 * index
        if scores is not None and stage in scores:
            score = scores[stage]
        members.append(_candidate(stage, keyframe, score=score))
    return PhaseEvidence(
        attempt_range=_attempt_range(),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=tuple(members),
    )


def _manual_exact() -> dict[str, float]:
    return {stage: _keyframe(i) for i, stage in enumerate(STAGE_ORDER)}


def _solve(
    evidence: PhaseEvidence, config: PhaseSolverConfig | None = None
) -> tuple[PhaseSolverResult, PhaseSolverConfig]:
    cfg = config if config is not None else PhaseSolverConfig()
    result = solve_with_diagnostics(_attempt(), _grid(), evidence, cfg)
    return result, cfg


def _contact_time(result: PhaseSolverResult) -> float | None:
    return result.attempt_phase.stages["contact"].keyframe_seconds


# --- Complete path ---


def test_complete_path_reconciles_exactly() -> None:
    evidence = _evidence()
    result, config = _solve(evidence)
    # Sanity: the synthetic clean path selects everything.
    assert all(value is not None for value in result.selected_keyframes.values())
    artifact = build_phase_debug_artifact(
        _grid(),
        evidence,
        result,
        _manual_exact(),
        solver_result=result,
        solver_config=config,
    )
    assert artifact.schema_version == PHASE_DEBUG_SCHEMA_VERSION == 3
    assert artifact.total_objective == pytest.approx(
        result.total_score, abs=RECONCILIATION_OBJECTIVE_TOLERANCE
    )
    assert artifact.solver_config_id == config.config_id
    assert artifact.solver_method_version == PHASE_SOLVER_METHOD_VERSION
    assert (
        artifact.reconciliation_method_version == RECONCILIATION_METHOD_VERSION
    )
    total = 0.0
    previous_selected: str | None = None
    for index, stage in enumerate(STAGE_ORDER):
        record = artifact.stages[stage]
        total += float(record.objective_contribution)
        # Selected: unary is the exact-match score, skip is null.
        expected_score = 0.70 - 0.02 * index
        assert record.unary_contribution == pytest.approx(expected_score)
        assert record.skip_contribution is None
        assert record.selected_score == pytest.approx(expected_score)
        assert record.selected_rank == 1
        expected_transition = 0.0 if previous_selected is None else config.transition_bonus
        assert record.transition_contribution == pytest.approx(expected_transition)
        assert record.objective_contribution == pytest.approx(
            expected_score + expected_transition,
            abs=RECONCILIATION_OBJECTIVE_TOLERANCE,
        )
        assert record.predecessor_stage == previous_selected
        assert record.selection_explanation == "selected_candidate_rank_1"
        assert record.unavailable_reason is None
        previous_selected = stage
    assert total == pytest.approx(
        result.total_score, abs=RECONCILIATION_OBJECTIVE_TOLERANCE
    )
    # First selected stage carries an honest 0.0 transition (no bonus).
    assert artifact.stages["start"].transition_contribution == pytest.approx(0.0)


def test_skipped_path_without_candidates() -> None:
    evidence = _evidence(drop=("finish", "deceleration"))
    result, config = _solve(evidence)
    assert result.selected_keyframes["finish"] is None
    assert result.selected_keyframes["deceleration"] is None
    artifact = build_phase_debug_artifact(
        _grid(), evidence, result, _manual_exact(), solver_result=result,
        solver_config=config,
    )
    for stage in ("finish", "deceleration"):
        record = artifact.stages[stage]
        assert record.selected_time_seconds is None
        assert record.unary_contribution is None
        assert record.transition_contribution is None
        assert record.skip_contribution == pytest.approx(-config.skip_penalty)
        assert record.objective_contribution == pytest.approx(
            -config.skip_penalty
        )
        assert record.predecessor_stage is None
        assert record.selection_explanation == "no_candidate_available"
        assert record.unavailable_reason == "no_candidate_available"
        assert record.anomalies == (f"no_candidate_{stage}",)
        assert record.selected_contact_offset_seconds is None
    # Selected survivors still reconcile.
    assert artifact.total_objective == pytest.approx(
        result.total_score, abs=RECONCILIATION_OBJECTIVE_TOLERANCE
    )
    summed = sum(
        float(record.objective_contribution)
        for record in artifact.stages.values()
    )
    assert summed == pytest.approx(
        result.total_score, abs=RECONCILIATION_OBJECTIVE_TOLERANCE
    )


def test_nonadjacent_transition_spans_skipped_stages() -> None:
    # Drop two consecutive middles: the surviving pair spans 3 stages.
    evidence = _evidence(drop=("loading", "cocking"))
    result, config = _solve(evidence)
    assert result.selected_keyframes["loading"] is None
    assert result.selected_keyframes["cocking"] is None
    assert result.selected_keyframes["release"] is not None
    assert result.selected_keyframes["acceleration"] is not None
    artifact = build_phase_debug_artifact(
        _grid(), evidence, result, None, solver_result=result,
        solver_config=config,
    )
    acceleration = artifact.stages["acceleration"]
    # The predecessor skips the dropped span yet still earns one bonus.
    assert acceleration.predecessor_stage == "release"
    assert acceleration.transition_contribution == pytest.approx(
        config.transition_bonus
    )
    assert acceleration.unary_contribution is not None
    assert acceleration.skip_contribution is None
    for stage in ("loading", "cocking"):
        record = artifact.stages[stage]
        assert record.skip_contribution == pytest.approx(-config.skip_penalty)
        assert record.selection_explanation == "no_candidate_available"
    assert artifact.total_objective == pytest.approx(
        result.total_score, abs=RECONCILIATION_OBJECTIVE_TOLERANCE
    )


def test_skipped_for_consistency_keeps_candidates() -> None:
    # Out-of-order middle keyframes force the DP to skip one survivor
    # despite candidates existing on both sides.
    keyframes = {stage: _keyframe(i) for i, stage in enumerate(STAGE_ORDER)}
    keyframes["loading"] = 10.80
    keyframes["cocking"] = 10.60  # earlier than loading: infeasible together
    evidence = _evidence(keyframes=keyframes)
    result, config = _solve(evidence)
    skipped = [
        stage
        for stage in STAGE_ORDER
        if result.selected_keyframes[stage] is None
    ]
    assert skipped, "expected the DP to skip an inconsistent stage"
    artifact = build_phase_debug_artifact(
        _grid(), evidence, result, None, solver_result=result,
        solver_config=config,
    )
    for stage in skipped:
        record = artifact.stages[stage]
        assert len(record.candidates) >= 1
        assert record.unary_contribution is None
        assert record.transition_contribution is None
        assert record.skip_contribution == pytest.approx(
            -config.skip_cost(stage)
        )
        assert record.selection_explanation == "skipped_for_global_consistency"
        assert record.anomalies == (f"skipped_{stage}_for_consistency",)
        assert record.unavailable_reason is None
    assert artifact.total_objective == pytest.approx(
        result.total_score, abs=RECONCILIATION_OBJECTIVE_TOLERANCE
    )


def test_contact_skip_penalty_and_null_offsets() -> None:
    evidence = _evidence(drop=("contact",))
    result, config = _solve(evidence)
    assert result.selected_keyframes["contact"] is None
    artifact = build_phase_debug_artifact(
        _grid(), evidence, result, _manual_exact(), solver_result=result,
        solver_config=config,
    )
    contact = artifact.stages["contact"]
    assert contact.skip_contribution == pytest.approx(
        -config.contact_skip_penalty
    )
    assert contact.skip_contribution != pytest.approx(-config.skip_penalty)
    assert contact.selection_explanation == "no_candidate_available"
    # Without a contact anchor every contact-relative offset is honestly null,
    # even for present selected/manual times.
    for record in artifact.stages.values():
        assert record.selected_contact_offset_seconds is None
        assert record.manual_contact_offset_seconds is None
    assert artifact.total_objective == pytest.approx(
        result.total_score, abs=RECONCILIATION_OBJECTIVE_TOLERANCE
    )


def test_contact_relative_offsets_selected_and_manual() -> None:
    evidence = _evidence()
    result, config = _solve(evidence)
    contact = _contact_time(result)
    assert contact is not None
    manual = _manual_exact()
    manual["cocking"] = _keyframe(3) + 0.005  # nonmatching manual still offsets
    artifact = build_phase_debug_artifact(
        _grid(), evidence, result, manual, solver_result=result,
        solver_config=config,
    )
    for index, stage in enumerate(STAGE_ORDER):
        record = artifact.stages[stage]
        assert record.selected_contact_offset_seconds == pytest.approx(
            _keyframe(index) - float(contact)
        )
        expected_manual = manual[stage] - float(contact)
        assert record.manual_contact_offset_seconds == pytest.approx(
            expected_manual
        )
    # Contact itself offsets to exactly zero on both sides.
    assert artifact.stages["contact"].selected_contact_offset_seconds == (
        pytest.approx(0.0)
    )
    assert artifact.stages["contact"].manual_contact_offset_seconds == (
        pytest.approx(0.0)
    )
    # Stages before contact are negative, after are positive.
    assert artifact.stages["start"].selected_contact_offset_seconds < 0.0
    assert artifact.stages["finish"].selected_contact_offset_seconds > 0.0


# --- Invalid / mismatched inputs are rejected ---


def test_rejects_config_mismatch() -> None:
    evidence = _evidence()
    result, _ = _solve(evidence)
    other = PhaseSolverConfig(
        config_id="phase-solver-other-v1",
        skip_penalty=0.30,
        contact_skip_penalty=0.80,
        min_transition_gap_seconds=0.0,
        max_transition_gap_seconds=1.50,
        transition_bonus=0.05,
        available_observation_floor=0.50,
        body_support_quality_floor=0.30,
        body_elevation_margin=0.0,
        body_min_wrist_speed=0.05,
        contact_body_confidence_bonus=0.05,
        contact_body_confidence_penalty=0.15,
    )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), evidence, result, None, solver_result=result,
            solver_config=other,
        )
    with pytest.raises(PhaseDebugError):
        explain_solver_path(evidence, result, other)


def test_rejects_selected_mismatch() -> None:
    evidence = _evidence()
    result, config = _solve(evidence)
    other_result, _ = _solve(_evidence(scores={"start": 0.10}))
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), evidence, result, None, solver_result=other_result,
            solver_config=config,
        )


def test_rejects_single_solver_input() -> None:
    evidence = _evidence()
    result, config = _solve(evidence)
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), evidence, result, None, solver_result=result,
            solver_config=None,
        )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), evidence, result, None, solver_result=None,
            solver_config=config,
        )


def test_rejects_range_mismatch() -> None:
    evidence = _evidence()
    result, config = _solve(evidence)
    other_evidence = PhaseEvidence(
        attempt_range=MediaRange(start_seconds=0.0, end_seconds=2.0),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=(),
    )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), other_evidence, result, None, solver_result=result,
            solver_config=config,
        )


def _fake_result_with_keyframes(
    keyframes: dict[str, float | None], total: float
) -> PhaseSolverResult:
    stages: dict[str, StagePhase] = {}
    for index, stage in enumerate(STAGE_ORDER):
        keyframe = keyframes[stage]
        if keyframe is None:
            stages[stage] = StagePhase(
                availability="unavailable",
                provenance="audio_transient" if stage == "contact" else "body_pose",
                confidence=0.0,
                interval=None,
                keyframe_seconds=None,
                temporal_uncertainty_seconds=None,
                evidence=(),
                limitations=("no_candidate_available",),
            )
        else:
            stages[stage] = StagePhase(
                availability="available",
                provenance="audio_transient" if stage == "contact" else "body_pose",
                confidence=0.7,
                interval=MediaRange(
                    start_seconds=float(keyframe) - 0.02,
                    end_seconds=float(keyframe) + 0.02,
                ),
                keyframe_seconds=float(keyframe),
                temporal_uncertainty_seconds=0.02,
                evidence=("cue_a",),
                limitations=(),
            )
    available = sum(1 for stage in stages.values() if stage.availability == "available")
    if available == len(STAGE_ORDER):
        status = "complete"
    elif available == 0:
        status = "unavailable"
    else:
        status = "partial"
    phase = AttemptPhase(
        attempt_id="serve-001",
        attempt_range=_attempt_range(),
        method_version=PHASE_SOLVER_METHOD_VERSION,
        config_id=PhaseSolverConfig().config_id,
        stages=stages,  # type: ignore[arg-type]
        structural_status=status,  # type: ignore[arg-type]
        anomalies=(),
    )
    return PhaseSolverResult(
        attempt_phase=phase,
        total_score=float(total),
        selected_keyframes=dict(keyframes),
        method_version=PHASE_SOLVER_METHOD_VERSION,
        config_id=PhaseSolverConfig().config_id,
    )


def test_rejects_selected_without_exact_candidate() -> None:
    evidence = _evidence()
    config = PhaseSolverConfig()
    keyframes: dict[str, float | None] = {
        stage: _keyframe(i) for i, stage in enumerate(STAGE_ORDER)
    }
    keyframes["loading"] = _keyframe(2) + 0.005  # no exact candidate
    fake = _fake_result_with_keyframes(keyframes, total=1.0)
    with pytest.raises(PhaseDebugError):
        explain_solver_path(evidence, fake, config)
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), evidence, fake, None, solver_result=fake,
            solver_config=config,
        )


def test_rejects_infeasible_chosen_transition() -> None:
    # Gap infeasibility (not overlap): start -> release gap 1.70 exceeds
    # max_gap*span = 1.50, yet intervals stay chronological so the domain
    # object itself is valid and only the replay rejects it.
    far_evidence = PhaseEvidence(
        attempt_range=_attempt_range(),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=(
            _candidate("start", 10.10, score=0.70),
            _candidate("release", 11.80, score=0.70),
        ),
    )
    config = PhaseSolverConfig()
    keyframes: dict[str, float | None] = {stage: None for stage in STAGE_ORDER}
    keyframes["start"] = 10.10
    keyframes["release"] = 11.80
    fake = _fake_result_with_keyframes(keyframes, total=1.0)
    with pytest.raises(PhaseDebugError):
        explain_solver_path(far_evidence, fake, config)


def test_rejects_total_mismatch() -> None:
    evidence = _evidence()
    result, config = _solve(evidence)
    keyframes = dict(result.selected_keyframes)
    fake = _fake_result_with_keyframes(
        {key: value for key, value in keyframes.items()},
        total=float(result.total_score) + 5.0,
    )
    with pytest.raises(PhaseDebugError):
        explain_solver_path(evidence, fake, config)


def test_rejects_wrong_input_types() -> None:
    evidence = _evidence()
    result, config = _solve(evidence)
    with pytest.raises(PhaseDebugError):
        explain_solver_path(object(), result, config)  # type: ignore[arg-type]
    with pytest.raises(PhaseDebugError):
        explain_solver_path(evidence, object(), config)  # type: ignore[arg-type]
    with pytest.raises(PhaseDebugError):
        explain_solver_path(evidence, result, object())  # type: ignore[arg-type]


# --- Roundtrip / determinism / Unit 1 preservation / selection unchanged ---


def test_json_roundtrip_and_determinism() -> None:
    evidence = _evidence(drop=("cocking",))
    result, config = _solve(evidence)
    first = build_phase_debug_artifact(
        _grid(), evidence, result, _manual_exact(), solver_result=result,
        solver_config=config,
    )
    second = build_phase_debug_artifact(
        _grid(), evidence, result, _manual_exact(), solver_result=result,
        solver_config=config,
    )
    assert first == second
    payload = first.to_json()
    assert payload.endswith("\n")
    assert payload == json.dumps(json.loads(payload), sort_keys=True, indent=2) + "\n"
    clone = PhaseDebugArtifact.from_json(payload)
    assert clone == first
    assert clone.total_objective == pytest.approx(result.total_score)
    assert clone.stages["cocking"].skip_contribution == pytest.approx(
        -config.skip_penalty
    )
    # v2 payloads without v3 keys are rejected as incomplete.
    legacy = json.loads(payload)
    del legacy["solver_config_id"]
    with pytest.raises(PhaseDebugError):
        PhaseDebugArtifact.from_dict(legacy)
    stage_payload = first.stages["start"].to_dict()
    del stage_payload["unary_contribution"]
    with pytest.raises(PhaseDebugError):
        first.stages["start"].__class__.from_dict(stage_payload)


def test_unreconciled_builder_preserves_unit1_semantics() -> None:
    evidence = _evidence()
    result, _ = _solve(evidence)
    artifact = build_phase_debug_artifact(
        _grid(), evidence, result, _manual_exact()
    )
    assert artifact.total_objective is None
    assert artifact.solver_config_id is None
    assert artifact.solver_method_version is None
    assert artifact.reconciliation_method_version is None
    for record in artifact.stages.values():
        assert record.unary_contribution is None
        assert record.transition_contribution is None
        assert record.skip_contribution is None
        assert record.objective_contribution is None
        # Explanation/predecessor/offsets remain honestly populated.
        assert record.selection_explanation
        assert record.selected_contact_offset_seconds is not None or (
            record.selected_time_seconds is None
            or artifact.stages["contact"].selected_time_seconds is None
        )
    # Manual exact-match semantics unchanged from Unit 1.
    assert artifact.stages["cocking"].manual_score == pytest.approx(0.64)
    assert artifact.stages["cocking"].manual_rank == 1


def test_solver_selection_is_unchanged() -> None:
    evidence = _evidence()
    before, config = _solve(evidence)
    before_keyframes = dict(before.selected_keyframes)
    before_total = float(before.total_score)
    artifact = build_phase_debug_artifact(
        _grid(), evidence, before, _manual_exact(), solver_result=before,
        solver_config=config,
    )
    # The artifact restates the same times; the solver re-run is identical.
    for stage in STAGE_ORDER:
        assert artifact.stages[stage].selected_time_seconds == (
            pytest.approx(before_keyframes[stage])
            if before_keyframes[stage] is not None
            else None
        )
        if before_keyframes[stage] is not None:
            assert artifact.stages[stage].selected_score is not None
    after = solve_with_diagnostics(_attempt(), _grid(), evidence, config)
    assert dict(after.selected_keyframes) == before_keyframes
    assert float(after.total_score) == pytest.approx(before_total)
    assert after.attempt_phase == before.attempt_phase
    # Inputs were not mutated by the builder.
    assert dict(before.selected_keyframes) == before_keyframes
