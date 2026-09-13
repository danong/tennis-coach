"""Focused tests for the deterministic phase-debug builder (M4 Unit 1).

Deterministic, offline, synthetic only; no private footage, no CLI,
no pipeline/rendering, no solver/evidence behavior changes. Covers the
required outcome: all eight StageDebug records populated from one
PhaseFeatureGrid, PhaseEvidence, selected AttemptPhase (or
PhaseSolverResult), and frozen manual query times, with durable
validated manual_score/manual_rank (exact match only, never
interpolated), support snapshots, candidate summaries/ranks, bounded
trace windows, derived unavailable reasons/anomalies, strict
range/method linkage, deterministic ordering, and honest
allowlist-only channel mapping.
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
    MAX_TRACE_SAMPLES,
    PHASE_DEBUG_DEFAULT_CONFIG_ID,
    PHASE_DEBUG_METHOD_VERSION,
    PHASE_DEBUG_SCHEMA_VERSION,
    TRACE_CHANNEL_ALLOWLIST_V1,
    TRACE_SAMPLING_METHOD_V1,
    PhaseDebugArtifact,
    PhaseDebugError,
)
from serve_review.checkpoints.phase_debug_builder import (
    TRACE_WINDOW_HALF_SECONDS,
    build_phase_debug_artifact,
)
from serve_review.checkpoints.phase_features import (
    PHASE_FEATURES_METHOD_VERSION,
    PhaseFeatureGrid,
    PhaseFeatureSample,
)
from serve_review.checkpoints.phase_solver import PhaseSolverResult
from serve_review.domain import STAGE_ORDER, AttemptPhase, MediaRange, StagePhase

ATTEMPT_START = 10.0
ATTEMPT_END = 12.0
GRID_RATE_HZ = 30.0


def _attempt_range() -> MediaRange:
    return MediaRange(start_seconds=ATTEMPT_START, end_seconds=ATTEMPT_END)


def _keyframe(index: int) -> float:
    return 10.20 + 0.15 * index


def _grid_sample(time: float, *, observed: bool = True) -> PhaseFeatureSample:
    if observed:
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
    return PhaseFeatureSample(
        time_seconds=time,
        observed=False,
        observation_quality=0.4,
        interpolation_span_seconds=0.066,
        source_time_offset_seconds=0.0,
        temporal_uncertainty_seconds=0.034,
        derivative_confidence=0.3,
        derivative_quality=0.3,
        wrist_left_x=0.10,
        wrist_left_y=-0.20,
        wrist_left_vx=None,
        wrist_left_vy=None,
        wrist_right_x=None,
        wrist_right_y=None,
        wrist_right_vx=None,
        wrist_right_vy=None,
        elbow_angle_right=None,
        knee_angle_left=None,
        knee_angle_right=None,
        shoulder_axis_camera_dx=None,
        shoulder_axis_camera_dy=None,
        hip_axis_camera_dx=None,
        hip_axis_camera_dy=None,
        torso_rotation_camera_deg=None,
        audio_energy=0.30,
        audio_candidate=False,
    )


def _grid(rate_hz: float = GRID_RATE_HZ) -> PhaseFeatureGrid:
    step = 1.0 / rate_hz
    count = int(round((ATTEMPT_END - ATTEMPT_START) / step))
    samples = tuple(
        _grid_sample(ATTEMPT_START + index * step) for index in range(count)
    )
    return PhaseFeatureGrid(
        attempt_range=_attempt_range(),
        method_version=PHASE_FEATURES_METHOD_VERSION,
        config_id="phase-features-default-v2",
        source_frame_rate_hz=240.0,
        pose_observation_rate_hz=rate_hz,
        grid_rate_hz=rate_hz,
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
    *, scores: dict[str, float] | None = None, drop: tuple[str, ...] = ()
) -> PhaseEvidence:
    members: list[StageCandidate] = []
    for index, stage in enumerate(STAGE_ORDER):
        if stage in drop:
            continue
        score = 0.70 - 0.02 * index
        if scores is not None and stage in scores:
            score = scores[stage]
        members.append(_candidate(stage, _keyframe(index), score=score))
    return PhaseEvidence(
        attempt_range=_attempt_range(),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=tuple(members),
    )


def _selected_stage(
    stage: str, index: int, *, keyframe: float | None = None
) -> StagePhase:
    moment = _keyframe(index) if keyframe is None else keyframe
    if moment is None:
        raise AssertionError("unreachable")
    return StagePhase(
        availability="available",
        provenance="audio_transient" if stage == "contact" else "body_pose",
        confidence=0.70 - 0.02 * index,
        interval=MediaRange(
            start_seconds=moment - 0.02, end_seconds=moment + 0.02
        ),
        keyframe_seconds=moment,
        temporal_uncertainty_seconds=0.02,
        evidence=("cue_a",),
        limitations=(),
    )


def _unavailable_stage(*, limitations: tuple[str, ...]) -> StagePhase:
    return StagePhase(
        availability="unavailable",
        provenance="body_pose",
        confidence=0.0,
        interval=None,
        keyframe_seconds=None,
        temporal_uncertainty_seconds=None,
        evidence=(),
        limitations=limitations,
    )


def _structural_status(stages: dict[str, StagePhase]) -> str:
    available = sum(1 for s in stages.values() if s.availability == "available")
    unavailable = sum(
        1 for s in stages.values() if s.availability == "unavailable"
    )
    if available == len(STAGE_ORDER):
        return "complete"
    if unavailable == len(STAGE_ORDER):
        return "unavailable"
    if available >= 1:
        return "partial"
    return "incomplete"


def _selected(
    *,
    keyframes: dict[str, float | None] | None = None,
    unavailable: dict[str, tuple[str, ...]] | None = None,
    anomalies: tuple[str, ...] = (),
) -> AttemptPhase:
    keyframes = keyframes or {}
    unavailable = unavailable or {}
    stages: dict[str, StagePhase] = {}
    for index, stage in enumerate(STAGE_ORDER):
        if stage in unavailable:
            stages[stage] = _unavailable_stage(
                limitations=unavailable[stage]
            )
            continue
        override = keyframes.get(stage, _keyframe(index))
        assert override is not None
        stages[stage] = _selected_stage(stage, index, keyframe=override)
    return AttemptPhase(
        attempt_id="serve-001",
        attempt_range=_attempt_range(),
        method_version="phase-solver-v1",
        config_id="phase-solver-default-v1",
        stages=stages,  # type: ignore[arg-type]
        structural_status=_structural_status(stages),
        anomalies=anomalies,
    )


def _manual_exact() -> dict[str, float]:
    return {stage: _keyframe(i) for i, stage in enumerate(STAGE_ORDER)}


# --- Complete build ---


def test_complete_build_populates_all_stages() -> None:
    artifact = build_phase_debug_artifact(
        _grid(), _evidence(), _selected(), _manual_exact()
    )
    assert isinstance(artifact, PhaseDebugArtifact)
    assert set(artifact.stages.keys()) == set(STAGE_ORDER)
    assert set(artifact.traces.keys()) == set(STAGE_ORDER)
    assert artifact.attempt_id == "serve-001"
    assert artifact.method_version == PHASE_DEBUG_METHOD_VERSION
    assert artifact.config_id == PHASE_DEBUG_DEFAULT_CONFIG_ID
    assert artifact.schema_version == PHASE_DEBUG_SCHEMA_VERSION
    for index, stage in enumerate(STAGE_ORDER):
        record = artifact.stages[stage]
        assert record.stage == stage
        assert record.manual_time_seconds == pytest.approx(_keyframe(index))
        assert record.selected_time_seconds == pytest.approx(_keyframe(index))
        # Exact candidate-keyframe match on both sides.
        assert record.manual_score == pytest.approx(0.70 - 0.02 * index)
        assert record.manual_rank == 1
        assert record.selected_score == pytest.approx(0.70 - 0.02 * index)
        assert record.selected_rank == 1
        assert len(record.candidates) == 1
        assert record.candidates[0].rank == 1
        assert record.unavailable_reason is None
        assert record.anomalies == ()
        assert record.objective_contribution is None
        assert record.manual_support is not None
        assert record.selected_support is not None
    assert artifact.total_objective is None
    clone = PhaseDebugArtifact.from_json(artifact.to_json())
    assert clone == artifact


def test_deferred_objective_and_config_passthrough() -> None:
    artifact = build_phase_debug_artifact(
        _grid(),
        _evidence(),
        _selected(),
        _manual_exact(),
        config_id="phase-debug-custom-v1",
    )
    assert artifact.config_id == "phase-debug-custom-v1"
    assert artifact.total_objective is None
    assert all(
        record.objective_contribution is None
        for record in artifact.stages.values()
    )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), _evidence(), _selected(), _manual_exact(), config_id="  "
        )


# --- Manual semantics: exact match only, never interpolated ---


def test_manual_exact_match_yields_score_and_rank() -> None:
    manual = {stage: None for stage in STAGE_ORDER}
    manual["cocking"] = _keyframe(3)
    artifact = build_phase_debug_artifact(_grid(), _evidence(), _selected(), manual)
    record = artifact.stages["cocking"]
    assert record.manual_score == pytest.approx(0.64)
    assert record.manual_rank == 1
    assert artifact.stages["start"].manual_score is None
    assert artifact.stages["start"].manual_rank is None


def test_manual_nonmatching_time_yields_null_null() -> None:
    manual = _manual_exact()
    manual["cocking"] = _keyframe(3) + 0.005  # near but not exact
    artifact = build_phase_debug_artifact(_grid(), _evidence(), _selected(), manual)
    record = artifact.stages["cocking"]
    assert record.manual_score is None
    assert record.manual_rank is None
    # The manual query time and its support snapshot are still recorded.
    assert record.manual_time_seconds == pytest.approx(_keyframe(3) + 0.005)
    assert record.manual_support is not None


def test_manual_between_candidates_never_interpolates() -> None:
    members = [
        _candidate("cocking", 10.50, score=0.80),
        _candidate("cocking", 10.60, score=0.60),
    ]
    evidence = PhaseEvidence(
        attempt_range=_attempt_range(),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=tuple(
            _candidate(stage, _keyframe(i))
            for i, stage in enumerate(STAGE_ORDER)
            if stage != "cocking"
        )
        + tuple(members),
    )
    manual = {stage: None for stage in STAGE_ORDER}
    manual["cocking"] = 10.55  # strictly between the two keyframes
    artifact = build_phase_debug_artifact(_grid(), evidence, _selected(), manual)
    record = artifact.stages["cocking"]
    assert record.manual_score is None
    assert record.manual_rank is None
    # Deterministic rank order is preserved in the summaries.
    assert [c.rank for c in record.candidates] == [1, 2]
    assert [c.unary_score for c in record.candidates] == [0.80, 0.60]


def test_manual_match_middle_rank() -> None:
    members = [
        _candidate("cocking", 10.50, score=0.80),
        _candidate("cocking", 10.60, score=0.60),
        _candidate("cocking", 10.70, score=0.40),
    ]
    evidence = PhaseEvidence(
        attempt_range=_attempt_range(),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=tuple(
            _candidate(stage, _keyframe(i))
            for i, stage in enumerate(STAGE_ORDER)
            if stage != "cocking"
        )
        + tuple(members),
    )
    manual = {stage: None for stage in STAGE_ORDER}
    manual["cocking"] = 10.60
    artifact = build_phase_debug_artifact(_grid(), evidence, _selected(), manual)
    record = artifact.stages["cocking"]
    assert record.manual_score == pytest.approx(0.60)
    assert record.manual_rank == 2


def test_manual_null_times_yield_null_score_rank() -> None:
    artifact = build_phase_debug_artifact(_grid(), _evidence(), _selected(), None)
    for record in artifact.stages.values():
        assert record.manual_time_seconds is None
        assert record.manual_score is None
        assert record.manual_rank is None
        assert record.manual_support is None


# --- Selected score/rank follow the same exact-match rule ---


def test_selected_nonmatching_time_yields_null_null() -> None:
    keyframes = {stage: _keyframe(i) for i, stage in enumerate(STAGE_ORDER)}
    keyframes["loading"] = _keyframe(2) + 0.005
    selected = _selected(keyframes=keyframes)
    artifact = build_phase_debug_artifact(
        _grid(), _evidence(), selected, _manual_exact()
    )
    record = artifact.stages["loading"]
    assert record.selected_time_seconds == pytest.approx(_keyframe(2) + 0.005)
    assert record.selected_score is None
    assert record.selected_rank is None
    assert record.unavailable_reason is None  # selected info still present


def test_solver_result_wrapper_accepted() -> None:
    selected = _selected(anomalies=("contact_body_support_missing",))
    wrapped = PhaseSolverResult(
        attempt_phase=selected,
        total_score=1.25,
        selected_keyframes={
            stage: selected.stages[stage].keyframe_seconds
            for stage in STAGE_ORDER
        },
        method_version="phase-solver-v1",
        config_id="phase-solver-default-v1",
    )
    from_attempt = build_phase_debug_artifact(
        _grid(), _evidence(), selected, _manual_exact()
    )
    from_wrapped = build_phase_debug_artifact(
        _grid(), _evidence(), wrapped, _manual_exact()
    )
    assert from_wrapped == from_attempt
    assert from_wrapped.anomalies == ("contact_body_support_missing",)


def test_rejects_invalid_selected_wrapper() -> None:
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(_grid(), _evidence(), object(), _manual_exact())


# --- Unavailable reason / anomalies derived from inputs ---


def test_unavailable_without_candidates_derives_reason() -> None:
    evidence = _evidence(drop=("finish",))
    selected = _selected(
        unavailable={"finish": ("no_candidate_available",)}
    )
    artifact = build_phase_debug_artifact(_grid(), evidence, selected, _manual_exact())
    record = artifact.stages["finish"]
    assert record.selected_time_seconds is None
    assert record.selected_score is None
    assert record.selected_rank is None
    assert record.candidates == ()
    assert record.unavailable_reason == "no_candidate_available"
    assert record.anomalies == ("no_candidate_finish",)
    clone = PhaseDebugArtifact.from_json(artifact.to_json())
    assert clone == artifact


def test_unavailable_with_candidates_means_skipped() -> None:
    selected = _selected(
        unavailable={"deceleration": ("skipped_for_global_consistency",)}
    )
    artifact = build_phase_debug_artifact(_grid(), _evidence(), selected, None)
    record = artifact.stages["deceleration"]
    assert len(record.candidates) == 1
    assert record.unavailable_reason is None
    assert record.anomalies == ("skipped_deceleration_for_consistency",)


def test_unavailable_reason_echoes_selected_limitations() -> None:
    selected = _selected(
        unavailable={"finish": ("body_support_occluded", "camera_relative_only")}
    )
    artifact = build_phase_debug_artifact(
        _grid(), _evidence(drop=("finish",)), selected, None
    )
    record = artifact.stages["finish"]
    assert (
        record.unavailable_reason
        == "body_support_occluded; camera_relative_only"
    )


# --- Support snapshots ---


def test_support_snapshots_match_queries() -> None:
    artifact = build_phase_debug_artifact(
        _grid(), _evidence(), _selected(), _manual_exact()
    )
    for index, stage in enumerate(STAGE_ORDER):
        record = artifact.stages[stage]
        for snapshot, label in (
            (record.manual_support, "manual"),
            (record.selected_support, "selected"),
        ):
            assert snapshot is not None, (stage, label)
            assert snapshot.queried_time_seconds == pytest.approx(_keyframe(index))
            assert snapshot.support_time_seconds is not None
            assert snapshot.support_offset_seconds == pytest.approx(
                snapshot.support_time_seconds - _keyframe(index)
            )
            assert set(snapshot.feature_values.keys()) <= set(
                TRACE_CHANNEL_ALLOWLIST_V1
            )


def test_support_values_allowlist_only_and_honest() -> None:
    artifact = build_phase_debug_artifact(
        _grid(), _evidence(), _selected(), _manual_exact()
    )
    values = dict(artifact.stages["start"].selected_support.feature_values)  # type: ignore[union-attr]
    assert values["left_wrist_x_camera"] == pytest.approx(0.10)
    assert values["left_wrist_y_camera"] == pytest.approx(-0.20)
    assert values["left_wrist_upward_velocity_camera"] == pytest.approx(1.00)
    assert values["left_wrist_speed_camera"] == pytest.approx(
        (0.50**2 + 1.00**2) ** 0.5
    )
    assert values["right_elbow_flexion"] == pytest.approx(0.50)
    assert values["audio_transient_energy"] == pytest.approx(0.30)
    assert values["audio_candidate"] is False
    # Channels without an honest grid source stay missing, never zero-filled.
    assert "face_height_threshold_camera" not in values
    assert "right_wrist_arc_length" not in values


def test_support_missingness_empty_grid() -> None:
    empty = PhaseFeatureGrid(
        attempt_range=_attempt_range(),
        method_version=PHASE_FEATURES_METHOD_VERSION,
        config_id="phase-features-default-v2",
        source_frame_rate_hz=240.0,
        pose_observation_rate_hz=0.0,
        grid_rate_hz=GRID_RATE_HZ,
        position_window_samples=3,
        derivative_window_samples=3,
        direct_observation_tolerance_seconds=0.015,
        samples=(),
    )
    artifact = build_phase_debug_artifact(
        empty, _evidence(), _selected(), _manual_exact()
    )
    for record in artifact.stages.values():
        assert record.manual_support is None
        assert record.selected_support is None
        assert len(artifact.traces[record.stage].samples) == 0


# --- Bounded traces ---


def test_trace_windows_bounded_and_increasing() -> None:
    artifact = build_phase_debug_artifact(
        _grid(), _evidence(), _selected(), _manual_exact()
    )
    for index, stage in enumerate(STAGE_ORDER):
        trace = artifact.traces[stage]
        assert len(trace.windows) == 2
        assert {w.kind for w in trace.windows} == {"manual", "selected"}
        assert trace.sampling_method == TRACE_SAMPLING_METHOD_V1
        assert trace.truncated is False
        assert trace.source_sample_count == len(trace.samples)
        times = [s.time_seconds for s in trace.samples]
        assert times == sorted(times)
        assert len(set(times)) == len(times)
        for sample in trace.samples:
            assert set(sample.values.keys()) <= set(TRACE_CHANNEL_ALLOWLIST_V1)
        for window in trace.windows:
            width = (
                window.interval.end_seconds - window.interval.start_seconds
            )
            assert width <= 2 * TRACE_WINDOW_HALF_SECONDS + 1e-9


def test_trace_truncation_is_bounded_and_deterministic() -> None:
    dense = _grid(rate_hz=1200.0)
    manual = {stage: None for stage in STAGE_ORDER}
    manual["finish"] = 11.60  # disjoint from the selected finish window
    first = build_phase_debug_artifact(dense, _evidence(), _selected(), manual)
    second = build_phase_debug_artifact(dense, _evidence(), _selected(), manual)
    trace = first.traces["finish"]
    assert trace == second.traces["finish"]
    assert trace.truncated is True
    assert trace.source_sample_count > MAX_TRACE_SAMPLES
    assert len(trace.samples) == MAX_TRACE_SAMPLES
    times = [s.time_seconds for s in trace.samples]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_trace_windows_absent_when_times_null() -> None:
    artifact = build_phase_debug_artifact(_grid(), _evidence(), _selected(), None)
    selected_only = [s for s in STAGE_ORDER]
    assert selected_only  # stages still all present
    for stage in STAGE_ORDER:
        trace = artifact.traces[stage]
        assert len(trace.windows) == 1
        assert trace.windows[0].kind == "selected"


# --- Strict linkage and input validation ---


def test_rejects_range_mismatch() -> None:
    other = MediaRange(start_seconds=0.0, end_seconds=2.0)
    evidence = PhaseEvidence(
        attempt_range=other,
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=(),
    )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(_grid(), evidence, _selected(), None)
    bad_range_selected = AttemptPhase(
        attempt_id="serve-001",
        attempt_range=MediaRange(start_seconds=9.0, end_seconds=11.0),
        method_version="phase-solver-v1",
        config_id="phase-solver-default-v1",
        stages={s: _unavailable_stage(limitations=("no_candidate_available",)) for s in STAGE_ORDER},  # type: ignore[arg-type]
        structural_status="unavailable",
        anomalies=(),
    )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(_grid(), _evidence(), bad_range_selected, None)


def test_rejects_method_mismatch() -> None:
    grid = _grid()
    object.__setattr__(grid, "method_version", "phase-features-v9")
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(grid, _evidence(), _selected(), None)
    evidence = _evidence()
    object.__setattr__(evidence, "method_version", "phase-evidence-v9")
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(_grid(), evidence, _selected(), None)


def test_rejects_bad_manual_times() -> None:
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), _evidence(), _selected(), {"start": 10.2, "bogus": 1.0}
        )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), _evidence(), _selected(), {"start": float("nan")}
        )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), _evidence(), _selected(), {"start": 99.0}  # outside attempt
        )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(
            _grid(), _evidence(), _selected(), ["not-a-mapping"]  # type: ignore[arg-type]
        )
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact("not-a-grid", _evidence(), _selected(), None)  # type: ignore[arg-type]


def test_rejects_wrong_input_types() -> None:
    with pytest.raises(PhaseDebugError):
        build_phase_debug_artifact(_grid(), object(), _selected(), None)  # type: ignore[arg-type]


# --- Determinism and codec ---


def test_builder_determinism_and_codec() -> None:
    first = build_phase_debug_artifact(
        _grid(), _evidence(), _selected(), _manual_exact()
    )
    second = build_phase_debug_artifact(
        _grid(), _evidence(), _selected(), _manual_exact()
    )
    assert first == second
    payload = first.to_json()
    assert payload.endswith("\n")
    assert payload == json.dumps(json.loads(payload), sort_keys=True, indent=2) + "\n"
    assert PhaseDebugArtifact.from_json(payload) == first
    # Manual fields survive the durable round trip.
    record = PhaseDebugArtifact.from_json(payload).stages["cocking"]
    assert record.manual_score == pytest.approx(0.64)
    assert record.manual_rank == 1


def test_candidate_summaries_carry_ranks_and_provenance() -> None:
    artifact = build_phase_debug_artifact(
        _grid(), _evidence(), _selected(), _manual_exact()
    )
    contact = artifact.stages["contact"].candidates[0]
    assert contact.provenance == "audio_transient"
    assert contact.unary_score == pytest.approx(0.60)
    assert contact.rank == 1
    start = artifact.stages["start"].candidates[0]
    assert start.provenance == "body_pose"
    assert start.evidence_ids == ("cue_a",)
    assert start.limitation_ids == ("camera_relative_only",)
