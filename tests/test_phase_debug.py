"""Focused tests for phase-debug diagnostic schemas (representation only).

Deterministic, offline, synthetic only; no private footage, no CLI,
no pipeline/rendering, no production phase behavior.
"""

from __future__ import annotations

import json
from types import MappingProxyType

import pytest

from serve_review.checkpoints import phase_debug as debug_module
from serve_review.checkpoints.phase_debug import (
    MAX_TRACE_SAMPLES,
    MAX_WINDOWS_PER_STAGE,
    PHASE_DEBUG_METHOD_VERSION,
    PHASE_DEBUG_SCHEMA_VERSION,
    TRACE_CHANNEL_ALLOWLIST_V1,
    TRACE_SAMPLING_METHOD_V1,
    CandidateSummary,
    PhaseDebugArtifact,
    PhaseDebugError,
    StageDebug,
    StageTrace,
    SupportSnapshot,
    TraceSample,
    TraceWindow,
)
from serve_review.domain import STAGE_ORDER, MediaRange

REQUIRED_CHANNELS = (
    "left_wrist_x_camera",
    "left_wrist_y_camera",
    "left_wrist_upward_velocity_camera",
    "left_wrist_speed_camera",
    "face_height_threshold_camera",
    "combined_knee_flexion",
    "shoulder_line_tilt_camera_deg",
    "hip_line_tilt_camera_deg",
    "shoulder_hip_angle_difference_camera_deg",
    "right_elbow_flexion",
    "right_wrist_x_camera",
    "right_wrist_y_camera",
    "right_wrist_elevation_camera",
    "right_wrist_upward_velocity_camera",
    "right_wrist_speed_camera",
    "right_wrist_arc_length",
    "torso_rotation_proxy_camera",
    "audio_transient_energy",
    "audio_candidate",
)

ATTEMPT_START = 10.0
ATTEMPT_END = 12.0


def _range(start: float, end: float) -> MediaRange:
    return MediaRange(start_seconds=start, end_seconds=end)


def _attempt_range() -> MediaRange:
    return _range(ATTEMPT_START, ATTEMPT_END)


def _manual_time(index: int) -> float:
    return 10.10 + 0.10 * index


def _selected_time(index: int) -> float:
    return 10.12 + 0.10 * index


def _support(
    queried: float,
    *,
    offset: float = 0.002,
    observed: bool = True,
    values: dict | None = None,
) -> SupportSnapshot:
    support = queried + offset
    span = 0.0 if observed else 0.010
    uncertainty = max(abs(offset), span / 2.0) + 0.001
    if values is None:
        values = {"left_wrist_x_camera": 0.11, "audio_candidate": False}
    return SupportSnapshot(
        queried_time_seconds=queried,
        support_time_seconds=support,
        support_offset_seconds=offset,
        observed=observed,
        observation_quality=0.9 if observed else 0.6,
        interpolation_span_seconds=span,
        temporal_uncertainty_seconds=uncertainty,
        derivative_confidence=0.8,
        feature_values=dict(values),
    )


def _candidate(stage: str, keyframe: float, *, rank: int | None = 1) -> CandidateSummary:
    provenance = "audio_transient" if stage == "contact" else "body_pose"
    return CandidateSummary(
        stage=stage,
        interval=_range(keyframe - 0.02, keyframe + 0.02),
        keyframe_seconds=keyframe,
        unary_score=0.7,
        rank=rank,
        observation_quality=0.8,
        derivative_quality=0.7,
        provenance=provenance,
        temporal_uncertainty_seconds=0.02,
        evidence_ids=("cue_a",),
        limitation_ids=("camera_relative_only",),
    )


def _stage_complete(stage: str, index: int) -> StageDebug:
    manual = _manual_time(index)
    selected = _selected_time(index)
    return StageDebug(
        stage=stage,
        manual_time_seconds=manual,
        selected_time_seconds=selected,
        manual_support=_support(manual, offset=0.002, observed=True),
        selected_support=_support(selected, offset=-0.001, observed=True),
        candidates=(_candidate(stage, selected),),
        selected_score=0.7,
        selected_rank=1,
        objective_contribution=0.71,
        unavailable_reason=None,
        anomalies=(),
    )


def _trace(stage: str, index: int) -> StageTrace:
    manual = _manual_time(index)
    selected = _selected_time(index)
    windows = (
        TraceWindow(kind="manual", interval=_range(manual - 0.03, manual + 0.03)),
        TraceWindow(kind="selected", interval=_range(selected - 0.03, selected + 0.03)),
    )
    samples = (
        TraceSample(
            time_seconds=selected - 0.01,
            values={"left_wrist_x_camera": 0.1},
        ),
        TraceSample(
            time_seconds=selected,
            values={"left_wrist_x_camera": 0.2, "audio_candidate": False},
        ),
        TraceSample(
            time_seconds=selected + 0.01,
            values={"right_wrist_speed_camera": 1.5},
        ),
    )
    return StageTrace(
        stage=stage,
        windows=windows,
        samples=samples,
        source_sample_count=3,
        sampling_method=TRACE_SAMPLING_METHOD_V1,
        truncated=False,
    )


def _complete_artifact() -> PhaseDebugArtifact:
    stages = {stage: _stage_complete(stage, i) for i, stage in enumerate(STAGE_ORDER)}
    traces = {stage: _trace(stage, i) for i, stage in enumerate(STAGE_ORDER)}
    return PhaseDebugArtifact(
        attempt_id="serve-001",
        attempt_range=_attempt_range(),
        method_version=PHASE_DEBUG_METHOD_VERSION,
        config_id="phase-debug-default-v1",
        stages=stages,  # type: ignore[arg-type]
        traces=traces,  # type: ignore[arg-type]
        anomalies=(),
        total_objective=5.5,
    )


def _sparse_artifact() -> PhaseDebugArtifact:
    stages: dict[str, StageDebug] = {}
    traces: dict[str, StageTrace] = {}
    for i, stage in enumerate(STAGE_ORDER):
        if stage in ("start", "contact"):
            stages[stage] = _stage_complete(stage, i)
            traces[stage] = _trace(stage, i)
            continue
        manual = _manual_time(i)
        stages[stage] = StageDebug(
            stage=stage,
            manual_time_seconds=manual,
            selected_time_seconds=None,
            manual_support=_support(manual, offset=0.002, observed=False),
            selected_support=None,
            candidates=(),
            selected_score=None,
            selected_rank=None,
            objective_contribution=None,
            unavailable_reason="no_candidate_available",
            anomalies=(f"no_candidate_{stage}",),
        )
        traces[stage] = StageTrace(
            stage=stage,
            windows=(),
            samples=(),
            source_sample_count=0,
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=False,
        )
    return PhaseDebugArtifact(
        attempt_id="serve-002",
        attempt_range=_attempt_range(),
        method_version=PHASE_DEBUG_METHOD_VERSION,
        config_id="phase-debug-default-v1",
        stages=stages,  # type: ignore[arg-type]
        traces=traces,  # type: ignore[arg-type]
        anomalies=("sparse_debug",),
        total_objective=None,
    )


# --- Allowlist ---


def test_allowlist_contains_required_channels() -> None:
    for channel in REQUIRED_CHANNELS:
        assert channel in TRACE_CHANNEL_ALLOWLIST_V1
    assert len(TRACE_CHANNEL_ALLOWLIST_V1) >= len(REQUIRED_CHANNELS)
    assert len(set(TRACE_CHANNEL_ALLOWLIST_V1)) == len(TRACE_CHANNEL_ALLOWLIST_V1)
    # Single fixed v1 allowlist shared by support and trace values.
    assert debug_module.SUPPORT_FEATURE_ALLOWLIST_V1 is TRACE_CHANNEL_ALLOWLIST_V1


def test_allowlist_rejects_arbitrary_names() -> None:
    with pytest.raises(PhaseDebugError):
        SupportSnapshot(
            queried_time_seconds=10.1,
            support_time_seconds=10.102,
            support_offset_seconds=0.002,
            observed=True,
            observation_quality=0.9,
            interpolation_span_seconds=0.0,
            temporal_uncertainty_seconds=0.003,
            derivative_confidence=0.8,
            feature_values={"mystery_channel": 1.0},
        )
    with pytest.raises(PhaseDebugError):
        TraceSample(time_seconds=10.1, values={"not_a_channel": 0.5})


def test_trace_channels_use_same_allowlist() -> None:
    sample = TraceSample(
        time_seconds=10.1,
        values={name: 0.25 for name in REQUIRED_CHANNELS if name != "audio_candidate"},
    )
    assert set(sample.values.keys()) == {
        name for name in REQUIRED_CHANNELS if name != "audio_candidate"
    }
    with pytest.raises(PhaseDebugError):
        TraceSample(time_seconds=10.1, values={"audio_transient_power": 2.0})


# --- Support snapshots ---


def test_support_snapshot_all_fields() -> None:
    snapshot = _support(10.2, offset=-0.003, observed=False)
    assert snapshot.queried_time_seconds == pytest.approx(10.2)
    assert snapshot.support_time_seconds == pytest.approx(10.197)
    assert snapshot.support_offset_seconds == pytest.approx(-0.003)
    assert snapshot.observed is False
    assert snapshot.observation_quality == pytest.approx(0.6)
    assert snapshot.interpolation_span_seconds == pytest.approx(0.010)
    assert snapshot.temporal_uncertainty_seconds is not None
    assert snapshot.temporal_uncertainty_seconds >= abs(-0.003)
    assert snapshot.temporal_uncertainty_seconds >= 0.010 / 2.0
    assert snapshot.derivative_confidence == pytest.approx(0.8)
    assert dict(snapshot.feature_values) == {
        "left_wrist_x_camera": 0.11,
        "audio_candidate": False,
    }
    clone = SupportSnapshot.from_dict(snapshot.to_dict())
    assert clone == snapshot


def test_support_missingness_all_null_empty() -> None:
    snapshot = SupportSnapshot(
        queried_time_seconds=10.5,
        support_time_seconds=None,
        support_offset_seconds=None,
        observed=False,
        observation_quality=None,
        interpolation_span_seconds=None,
        temporal_uncertainty_seconds=None,
        derivative_confidence=None,
        feature_values={},
    )
    assert snapshot.support_time_seconds is None
    assert snapshot.support_offset_seconds is None
    assert dict(snapshot.feature_values) == {}
    clone = SupportSnapshot.from_json(snapshot.to_json())
    assert clone == snapshot


def test_support_offset_exact_signed() -> None:
    queried = 11.0
    snapshot = _support(queried, offset=0.004)
    assert snapshot.support_offset_seconds == pytest.approx(
        snapshot.support_time_seconds - queried  # type: ignore[operator]
    )
    with pytest.raises(PhaseDebugError):
        SupportSnapshot(
            queried_time_seconds=queried,
            support_time_seconds=queried + 0.004,
            support_offset_seconds=0.009,  # wrong signed offset
            observed=True,
            observation_quality=0.9,
            interpolation_span_seconds=0.0,
            temporal_uncertainty_seconds=0.009,
            derivative_confidence=0.8,
            feature_values={"left_wrist_x_camera": 0.0},
        )


def test_support_uncertainty_floors() -> None:
    with pytest.raises(PhaseDebugError):
        SupportSnapshot(
            queried_time_seconds=10.0,
            support_time_seconds=10.005,
            support_offset_seconds=0.005,
            observed=True,
            observation_quality=0.9,
            interpolation_span_seconds=0.0,
            temporal_uncertainty_seconds=0.001,  # narrower than |offset|
            derivative_confidence=0.8,
            feature_values={"left_wrist_x_camera": 0.0},
        )
    with pytest.raises(PhaseDebugError):
        SupportSnapshot(
            queried_time_seconds=10.0,
            support_time_seconds=10.001,
            support_offset_seconds=0.001,
            observed=False,
            observation_quality=0.5,
            interpolation_span_seconds=0.02,
            temporal_uncertainty_seconds=0.001,  # narrower than span/2
            derivative_confidence=0.5,
            feature_values={"left_wrist_x_camera": 0.0},
        )


def test_support_interpolation_semantics() -> None:
    with pytest.raises(PhaseDebugError):
        SupportSnapshot(
            queried_time_seconds=10.0,
            support_time_seconds=10.001,
            support_offset_seconds=0.001,
            observed=True,
            observation_quality=0.9,
            interpolation_span_seconds=0.01,  # observed must be 0.0
            temporal_uncertainty_seconds=0.01,
            derivative_confidence=0.8,
            feature_values={"left_wrist_x_camera": 0.0},
        )
    with pytest.raises(PhaseDebugError):
        SupportSnapshot(
            queried_time_seconds=10.0,
            support_time_seconds=10.001,
            support_offset_seconds=0.001,
            observed=False,
            observation_quality=1.0,  # non-observed cannot claim 1.0
            interpolation_span_seconds=0.01,
            temporal_uncertainty_seconds=0.01,
            derivative_confidence=0.5,
            feature_values={"left_wrist_x_camera": 0.0},
        )


def test_support_mappings_immutable() -> None:
    snapshot = _support(10.3)
    assert isinstance(snapshot.feature_values, MappingProxyType)
    with pytest.raises(TypeError):
        snapshot.feature_values["left_wrist_x_camera"] = 9.9  # type: ignore[index]


# --- Candidate summaries ---


def test_candidate_summary_all_fields() -> None:
    candidate = _candidate("cocking", 10.42)
    assert candidate.interval is not None
    assert candidate.keyframe_seconds == pytest.approx(10.42)
    assert candidate.unary_score == pytest.approx(0.7)
    assert candidate.rank == 1
    assert candidate.observation_quality == pytest.approx(0.8)
    assert candidate.derivative_quality == pytest.approx(0.7)
    assert candidate.provenance == "body_pose"
    assert candidate.temporal_uncertainty_seconds == pytest.approx(0.02)
    assert candidate.evidence_ids == ("cue_a",)
    assert candidate.limitation_ids == ("camera_relative_only",)
    clone = CandidateSummary.from_dict(candidate.to_dict())
    assert clone == candidate


def test_candidate_contact_provenance() -> None:
    candidate = _candidate("contact", 10.62)
    assert candidate.provenance == "audio_transient"
    with pytest.raises(PhaseDebugError):
        CandidateSummary(
            stage="contact",
            interval=_range(10.6, 10.64),
            keyframe_seconds=10.62,
            unary_score=0.5,
            rank=1,
            observation_quality=0.8,
            derivative_quality=0.7,
            provenance="nope",
            temporal_uncertainty_seconds=0.02,
            evidence_ids=("cue_a",),
            limitation_ids=("camera_relative_only",),
        )


def test_candidate_missingness() -> None:
    candidate = CandidateSummary(
        stage="loading",
        interval=None,
        keyframe_seconds=None,
        unary_score=None,
        rank=None,
        observation_quality=None,
        derivative_quality=None,
        provenance=None,
        temporal_uncertainty_seconds=None,
        evidence_ids=(),
        limitation_ids=(),
    )
    assert candidate.interval is None
    clone = CandidateSummary.from_json(candidate.to_json())
    assert clone == candidate
    with pytest.raises(PhaseDebugError):
        CandidateSummary(
            stage="loading",
            interval=None,
            keyframe_seconds=10.2,  # keyframe without interval
            unary_score=None,
            rank=None,
            observation_quality=None,
            derivative_quality=None,
            provenance=None,
            temporal_uncertainty_seconds=None,
            evidence_ids=(),
            limitation_ids=(),
        )


def test_candidate_range_keyframe_validation() -> None:
    with pytest.raises(PhaseDebugError):
        CandidateSummary(
            stage="start",
            interval=_range(10.1, 10.2),
            keyframe_seconds=10.5,  # outside half-open interval
            unary_score=0.5,
            rank=1,
            observation_quality=0.8,
            derivative_quality=0.7,
            provenance="body_pose",
            temporal_uncertainty_seconds=0.02,
            evidence_ids=("cue_a",),
            limitation_ids=("camera_relative_only",),
        )
    with pytest.raises(PhaseDebugError):
        CandidateSummary(
            stage="start",
            interval=_range(10.1, 10.2),
            keyframe_seconds=None,  # required when interval present
            unary_score=0.5,
            rank=1,
            observation_quality=0.8,
            derivative_quality=0.7,
            provenance="body_pose",
            temporal_uncertainty_seconds=0.02,
            evidence_ids=("cue_a",),
            limitation_ids=("camera_relative_only",),
        )


def test_candidate_evidence_limitation_validation() -> None:
    with pytest.raises(PhaseDebugError):
        _candidate("start", 10.12).to_dict() and CandidateSummary(
            stage="start",
            interval=_range(10.1, 10.14),
            keyframe_seconds=10.12,
            unary_score=0.5,
            rank=1,
            observation_quality=0.8,
            derivative_quality=0.7,
            provenance="body_pose",
            temporal_uncertainty_seconds=0.02,
            evidence_ids=("cue_a", "cue_a"),  # duplicate
            limitation_ids=(),
        )
    with pytest.raises(PhaseDebugError):
        CandidateSummary(
            stage="start",
            interval=_range(10.1, 10.14),
            keyframe_seconds=10.12,
            unary_score=0.5,
            rank=1,
            observation_quality=0.8,
            derivative_quality=0.7,
            provenance="body_pose",
            temporal_uncertainty_seconds=0.02,
            evidence_ids=("  ",),
            limitation_ids=(),
        )


def test_candidate_rank_determinism() -> None:
    first = _candidate("start", 10.12, rank=2)
    second = _candidate("start", 10.13, rank=1)
    with pytest.raises(PhaseDebugError):
        StageDebug(
            stage="start",
            manual_time_seconds=10.10,
            selected_time_seconds=10.12,
            manual_support=_support(10.10),
            selected_support=_support(10.12),
            candidates=(first, second),  # not in rank order
            selected_score=0.5,
            selected_rank=1,
            objective_contribution=0.1,
            unavailable_reason=None,
            anomalies=(),
        )
    with pytest.raises(PhaseDebugError):
        StageDebug(
            stage="start",
            manual_time_seconds=10.10,
            selected_time_seconds=10.12,
            manual_support=_support(10.10),
            selected_support=_support(10.12),
            candidates=(_candidate("start", 10.12, rank=1), _candidate("start", 10.13, rank=1)),
            selected_score=0.5,
            selected_rank=1,
            objective_contribution=0.1,
            unavailable_reason=None,
            anomalies=(),
        )


# --- Traces ---


def test_trace_windows_bounded() -> None:
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(
                TraceWindow(kind="manual", interval=_range(10.1, 10.2)),
                TraceWindow(kind="selected", interval=_range(10.2, 10.3)),
                TraceWindow(kind="manual", interval=_range(10.3, 10.4)),
            ),
            samples=(),
            source_sample_count=0,
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=False,
        )
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(
                TraceWindow(kind="manual", interval=_range(10.1, 10.2)),
                TraceWindow(kind="manual", interval=_range(10.2, 10.3)),
            ),
            samples=(),
            source_sample_count=0,
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=False,
        )


def test_trace_samples_bounded_strictly_increasing() -> None:
    many = tuple(
        TraceSample(time_seconds=10.0 + 0.001 * i, values={}) for i in range(242)
    )
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(),
            samples=many,
            source_sample_count=242,
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=False,
        )
    assert MAX_TRACE_SAMPLES == 241
    assert MAX_WINDOWS_PER_STAGE == 2
    ok = tuple(
        TraceSample(time_seconds=10.0 + 0.001 * i, values={}) for i in range(241)
    )
    trace = StageTrace(
        stage="start",
        windows=(),
        samples=ok,
        source_sample_count=241,
        sampling_method=TRACE_SAMPLING_METHOD_V1,
        truncated=False,
    )
    assert len(trace.samples) == 241
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(),
            samples=(
                TraceSample(time_seconds=10.2, values={}),
                TraceSample(time_seconds=10.2, values={}),  # not increasing
            ),
            source_sample_count=2,
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=False,
        )


def test_trace_source_count_truncation_identity() -> None:
    samples = (TraceSample(time_seconds=10.2, values={}),)
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(),
            samples=samples,
            source_sample_count=0,  # less than len(samples)
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=False,
        )
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(),
            samples=samples,
            source_sample_count=5,  # must equal len when not truncated
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=False,
        )
    truncated = StageTrace(
        stage="start",
        windows=(),
        samples=samples,
        source_sample_count=5,
        sampling_method=TRACE_SAMPLING_METHOD_V1,
        truncated=True,
    )
    assert truncated.truncated is True
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(),
            samples=samples,
            source_sample_count=1,  # truncated requires strictly greater
            sampling_method=TRACE_SAMPLING_METHOD_V1,
            truncated=True,
        )
    with pytest.raises(PhaseDebugError):
        StageTrace(
            stage="start",
            windows=(),
            samples=samples,
            source_sample_count=1,
            sampling_method="  ",  # method identity required
            truncated=False,
        )


# --- Artifacts ---


def test_complete_artifact_roundtrip() -> None:
    artifact = _complete_artifact()
    assert set(artifact.stages.keys()) == set(STAGE_ORDER)
    assert set(artifact.traces.keys()) == set(STAGE_ORDER)
    assert [stage.stage for stage in artifact.stages.values()] == list(STAGE_ORDER)
    payload = artifact.to_dict()
    assert list(payload["stages"].keys()) == list(STAGE_ORDER)
    clone = PhaseDebugArtifact.from_dict(json.loads(json.dumps(payload)))
    assert clone == artifact


def test_sparse_artifact_missingness() -> None:
    artifact = _sparse_artifact()
    assert artifact.stages["loading"].selected_time_seconds is None
    assert artifact.stages["loading"].unavailable_reason == "no_candidate_available"
    assert artifact.stages["loading"].candidates == ()
    assert artifact.traces["loading"].samples == ()
    clone = PhaseDebugArtifact.from_json(artifact.to_json())
    assert clone == artifact
    assert clone.total_objective is None


def test_codec_determinism() -> None:
    artifact = _complete_artifact()
    first = artifact.to_json()
    second = PhaseDebugArtifact.from_json(first).to_json()
    assert first == second
    assert first.endswith("\n")
    assert json.loads(first) == json.loads(second)
    # Deterministic key order: sorted keys at top level.
    assert first == json.dumps(json.loads(first), sort_keys=True, indent=2) + "\n"


def test_unknown_missing_keys_rejected() -> None:
    artifact = _complete_artifact()
    payload = artifact.to_dict()
    bad = dict(payload)
    bad["surprise"] = 1
    with pytest.raises(PhaseDebugError):
        PhaseDebugArtifact.from_dict(bad)
    for key in ("stages", "traces", "attempt_range", "attempt_id"):
        bad = dict(payload)
        del bad[key]
        with pytest.raises(PhaseDebugError):
            PhaseDebugArtifact.from_dict(bad)
    support_payload = _support(10.1).to_dict()
    bad_support = dict(support_payload)
    bad_support["extra"] = 0
    with pytest.raises(PhaseDebugError):
        SupportSnapshot.from_dict(bad_support)
    candidate_payload = _candidate("start", 10.12).to_dict()
    bad_candidate = dict(candidate_payload)
    bad_candidate["extra"] = 0
    with pytest.raises(PhaseDebugError):
        CandidateSummary.from_dict(bad_candidate)


def test_canonical_stages_and_order() -> None:
    assert tuple(STAGE_ORDER) == (
        "start",
        "release",
        "loading",
        "cocking",
        "acceleration",
        "contact",
        "deceleration",
        "finish",
    )
    artifact = _complete_artifact()
    payload = artifact.to_dict()
    del payload["stages"]["contact"]
    with pytest.raises(PhaseDebugError):
        PhaseDebugArtifact.from_dict(payload)
    payload = artifact.to_dict()
    payload["stages"]["contact"] = _stage_complete("start", 5).to_dict()
    with pytest.raises(PhaseDebugError):
        PhaseDebugArtifact.from_dict(payload)
    # Unordered selected times rejected.
    stages = {stage: _stage_complete(stage, i) for i, stage in enumerate(STAGE_ORDER)}
    bad_first = StageDebug(
        stage="start",
        manual_time_seconds=10.10,
        selected_time_seconds=11.90,  # later than every later stage
        manual_support=_support(10.10),
        selected_support=_support(11.90),
        candidates=(_candidate("start", 11.90),),
        selected_score=0.5,
        selected_rank=1,
        objective_contribution=0.1,
        unavailable_reason=None,
        anomalies=(),
    )
    stages["start"] = bad_first
    traces = {stage: _trace(stage, i) for i, stage in enumerate(STAGE_ORDER)}
    with pytest.raises(PhaseDebugError):
        PhaseDebugArtifact(
            attempt_id="serve-001",
            attempt_range=_attempt_range(),
            method_version=PHASE_DEBUG_METHOD_VERSION,
            config_id="phase-debug-default-v1",
            stages=stages,  # type: ignore[arg-type]
            traces=traces,  # type: ignore[arg-type]
            anomalies=(),
            total_objective=None,
        )


def test_stage_unavailable_reason_required() -> None:
    with pytest.raises(PhaseDebugError):
        StageDebug(
            stage="finish",
            manual_time_seconds=None,
            selected_time_seconds=None,
            manual_support=None,
            selected_support=None,
            candidates=(),
            selected_score=None,
            selected_rank=None,
            objective_contribution=None,
            unavailable_reason=None,  # required when nothing selected
            anomalies=(),
        )
    with pytest.raises(PhaseDebugError):
        StageDebug(
            stage="finish",
            manual_time_seconds=10.8,
            selected_time_seconds=10.82,
            manual_support=_support(10.8),
            selected_support=_support(10.82),
            candidates=(_candidate("finish", 10.82),),
            selected_score=0.5,
            selected_rank=1,
            objective_contribution=0.1,
            unavailable_reason="stale",  # must be null when selected present
            anomalies=(),
        )


def test_artifact_mappings_immutable() -> None:
    artifact = _complete_artifact()
    assert isinstance(artifact.stages, MappingProxyType)
    assert isinstance(artifact.traces, MappingProxyType)
    with pytest.raises(TypeError):
        artifact.stages["start"] = artifact.stages["start"]  # type: ignore[index]
    with pytest.raises(Exception):
        artifact.attempt_id = "serve-999"  # type: ignore[misc]


def test_strict_finite_range_validation() -> None:
    with pytest.raises(PhaseDebugError):
        SupportSnapshot(
            queried_time_seconds=float("inf"),
            support_time_seconds=None,
            support_offset_seconds=None,
            observed=False,
            observation_quality=None,
            interpolation_span_seconds=None,
            temporal_uncertainty_seconds=None,
            derivative_confidence=None,
            feature_values={},
        )
    with pytest.raises(PhaseDebugError):
        CandidateSummary(
            stage="start",
            interval=_range(10.1, 10.14),
            keyframe_seconds=10.12,
            unary_score=2.0,  # out of [0, 1]
            rank=1,
            observation_quality=0.8,
            derivative_quality=0.7,
            provenance="body_pose",
            temporal_uncertainty_seconds=0.02,
            evidence_ids=("cue_a",),
            limitation_ids=(),
        )
    with pytest.raises(PhaseDebugError):
        PhaseDebugArtifact(
            attempt_id="bad-id",
            attempt_range=_attempt_range(),
            method_version=PHASE_DEBUG_METHOD_VERSION,
            config_id="phase-debug-default-v1",
            stages={s: _stage_complete(s, i) for i, s in enumerate(STAGE_ORDER)},  # type: ignore[arg-type]
            traces={s: _trace(s, i) for i, s in enumerate(STAGE_ORDER)},  # type: ignore[arg-type]
            anomalies=(),
            total_objective=None,
        )


def test_schema_version_rejected() -> None:
    payload = _support(10.1).to_dict()
    payload["schema_version"] = 999
    with pytest.raises(PhaseDebugError):
        SupportSnapshot.from_dict(payload)
    payload = _complete_artifact().to_dict()
    payload["schema_version"] = 999
    with pytest.raises(PhaseDebugError):
        PhaseDebugArtifact.from_dict(payload)


# --- Manual score/rank durable fields ---


def test_manual_score_rank_roundtrip() -> None:
    stage = _stage_complete("cocking", 3)
    assert stage.manual_score is None
    assert stage.manual_rank is None
    scored = StageDebug(
        stage="cocking",
        manual_time_seconds=stage.manual_time_seconds,
        selected_time_seconds=stage.selected_time_seconds,
        manual_support=stage.manual_support,
        selected_support=stage.selected_support,
        candidates=stage.candidates,
        manual_score=0.62,
        manual_rank=2,
        selected_score=stage.selected_score,
        selected_rank=stage.selected_rank,
        objective_contribution=stage.objective_contribution,
        unavailable_reason=None,
        anomalies=(),
    )
    assert scored.manual_score == pytest.approx(0.62)
    assert scored.manual_rank == 2
    clone = StageDebug.from_json(scored.to_json())
    assert clone == scored
    assert clone.manual_score == pytest.approx(0.62)
    assert clone.manual_rank == 2


def test_manual_score_rank_validation() -> None:
    base = _stage_complete("start", 0)
    with pytest.raises(PhaseDebugError):
        StageDebug(
            stage="start",
            manual_time_seconds=base.manual_time_seconds,
            selected_time_seconds=base.selected_time_seconds,
            manual_support=base.manual_support,
            selected_support=base.selected_support,
            candidates=base.candidates,
            manual_score=2.0,  # out of [0, 1]
            manual_rank=None,
            selected_score=None,
            selected_rank=None,
            objective_contribution=None,
            unavailable_reason=None,
            anomalies=(),
        )
    with pytest.raises(PhaseDebugError):
        StageDebug(
            stage="start",
            manual_time_seconds=base.manual_time_seconds,
            selected_time_seconds=base.selected_time_seconds,
            manual_support=base.manual_support,
            selected_support=base.selected_support,
            candidates=base.candidates,
            manual_score=0.5,
            manual_rank=0,  # must be >= 1
            selected_score=None,
            selected_rank=None,
            objective_contribution=None,
            unavailable_reason=None,
            anomalies=(),
        )
    with pytest.raises(PhaseDebugError):
        StageDebug(
            stage="start",
            manual_time_seconds=base.manual_time_seconds,
            selected_time_seconds=base.selected_time_seconds,
            manual_support=base.manual_support,
            selected_support=base.selected_support,
            candidates=base.candidates,
            manual_score=0.5,
            manual_rank=True,  # bools are not valid ranks
            selected_score=None,
            selected_rank=None,
            objective_contribution=None,
            unavailable_reason=None,
            anomalies=(),
        )


def test_manual_fields_required_in_codec() -> None:
    payload = _stage_complete("start", 0).to_dict()
    assert "manual_score" in payload
    assert "manual_rank" in payload
    for key in ("manual_score", "manual_rank"):
        bad = dict(payload)
        del bad[key]
        with pytest.raises(PhaseDebugError):
            StageDebug.from_dict(bad)
    bad = dict(payload)
    bad["manual_extra"] = 1
    with pytest.raises(PhaseDebugError):
        StageDebug.from_dict(bad)
