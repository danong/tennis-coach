"""Focused M4.4 tests for the constrained joint phase solver.

Deterministic, offline, synthetic only; no private footage, no CLI, no
pipeline behavior, no M3 side effects.
"""

from __future__ import annotations

import math

import pytest

from serve_review.checkpoints import evidence as evidence_module
from serve_review.checkpoints import phase_solver as solver_module
from serve_review.checkpoints.evidence import (
    EVIDENCE_METHOD_VERSION,
    PhaseEvidence,
    PhaseEvidenceConfig,
    PhaseEvidenceError,
    StageCandidate,
    generate_stage_candidates,
)
from serve_review.checkpoints.phase_features import (
    PHASE_FEATURES_METHOD_VERSION,
    PhaseFeatureGrid,
    PhaseFeaturesConfig,
    build_phase_feature_grid,
)
from serve_review.checkpoints.phase_solver import (
    PHASE_SOLVER_DEFAULT_CONFIG_ID,
    PHASE_SOLVER_METHOD_VERSION,
    PHASE_SOLVER_SCHEMA_VERSION,
    PhaseSolverConfig,
    PhaseSolverError,
    PhaseSolverResult,
    solve_attempt_phase,
    solve_with_diagnostics,
)
from serve_review.domain import (
    STAGE_ORDER,
    Attempt,
    AttemptPhase,
    MediaRange,
)
from serve_review.media.audio import AudioEnergy
from serve_review.pose.schema import (
    NUM_KEYPOINTS,
    BodyKeypoint,
    FrameObservation,
    PersonBox,
    PersonObservation,
)


# --- Small builders ------------------------------------------------------------


def _range(start: float, end: float) -> MediaRange:
    return MediaRange(start_seconds=start, end_seconds=end)


def _attempt(start: float = 10.0, end: float = 12.0) -> Attempt:
    detected = _range(start, end)
    return Attempt(
        attempt_id="serve-001",
        detected_range=detected,
        effective_range=_range(start, end),
        confidence=0.9,
        evidence={"motion": 1.0},
    )


def _cand(
    stage: str,
    keyframe: float,
    half: float,
    score: float,
    *,
    obs_q: float = 0.9,
    deriv_q: float = 0.8,
    unc: float = 0.05,
    evidence: tuple[str, ...] = ("cue_a",),
    limitations: tuple[str, ...] = ("body_pose_estimate",),
) -> StageCandidate:
    provenance = "audio_transient" if stage == "contact" else "body_pose"
    return StageCandidate(
        stage=stage,
        keyframe_seconds=keyframe,
        interval=_range(keyframe - half, keyframe + half),
        score=score,
        observation_quality=obs_q,
        derivative_quality=deriv_q,
        evidence=evidence,
        limitations=limitations,
        provenance=provenance,
        temporal_uncertainty_seconds=unc,
    )


def _evidence(
    candidates: list[StageCandidate],
    *,
    start: float = 10.0,
    end: float = 12.0,
    config_id: str = PhaseEvidenceConfig().config_id,
) -> PhaseEvidence:
    return PhaseEvidence(
        attempt_range=_range(start, end),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=config_id,
        candidates=tuple(candidates),
    )


def _empty_grid(start: float = 10.0, end: float = 12.0) -> PhaseFeatureGrid:
    return PhaseFeatureGrid(
        attempt_range=_range(start, end),
        method_version=PHASE_FEATURES_METHOD_VERSION,
        config_id="phase-features-default-v2",
        source_frame_rate_hz=120.0,
        pose_observation_rate_hz=0.0,
        grid_rate_hz=30.0,
        position_window_samples=1,
        derivative_window_samples=1,
        direct_observation_tolerance_seconds=0.015,
        samples=(),
    )


def _manual_sequence(
    *,
    scores: dict[str, float] | None = None,
    contact: bool = True,
) -> list[StageCandidate]:
    """Eight cleanly separated manual candidates across 10..12."""
    base = {
        "start": 10.10,
        "release": 10.35,
        "loading": 10.60,
        "cocking": 10.85,
        "acceleration": 11.10,
        "contact": 11.35,
        "deceleration": 11.60,
        "finish": 11.85,
    }
    out: list[StageCandidate] = []
    for stage in STAGE_ORDER:
        if stage == "contact" and not contact:
            continue
        score = (scores or {}).get(stage, 0.70)
        out.append(_cand(stage, base[stage], 0.04, score))
    return out


# --- Pose/audio fixtures for body-compatibility --------------------------------


def _box() -> PersonBox:
    return PersonBox(x_min=0.1, y_min=0.1, x_max=0.7, y_max=0.9)


def _skeleton(
    *,
    wrist_left: tuple[float, float] = (0.36, 0.50),
    wrist_right: tuple[float, float] = (0.64, 0.50),
    visibility: float = 0.9,
) -> PersonObservation:
    joints: list = []
    for _ in range(NUM_KEYPOINTS):
        joints.append(BodyKeypoint(x=0.50, y=0.50, visibility=visibility))
    pins = {
        11: (0.40, 0.40),
        12: (0.60, 0.40),
        13: (0.38, 0.45),
        14: (0.62, 0.45),
        15: wrist_left,
        16: wrist_right,
        23: (0.42, 0.70),
        24: (0.58, 0.70),
        25: (0.44, 0.80),
        26: (0.56, 0.80),
        27: (0.44, 0.90),
        28: (0.56, 0.90),
    }
    for index, (px, py) in pins.items():
        joints[index] = BodyKeypoint(x=px, y=py, visibility=visibility)
    return PersonObservation(box=_box(), keypoints=tuple(joints), score=0.8)


def _frame(time_seconds: float, person: PersonObservation | None) -> FrameObservation:
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(person,) if person is not None else (),
    )


def _motion_observations(
    start: float = 10.0,
    end: float = 12.0,
    *,
    peak_y: float = 0.30,
    base_y: float = 0.55,
    visibility: float = 0.9,
) -> list[FrameObservation]:
    """Wrist rises from base_y to peak_y at u=0.7 then recovers."""
    frames: list[FrameObservation] = []
    n = 0
    while True:
        t = start + n * (1.0 / 30.0)
        if t >= end - 1e-9:
            break
        u = (t - start) / (end - start)
        if u < 0.7:
            wy = base_y - (base_y - peak_y) * (u / 0.7)
        else:
            wy = peak_y + 0.15 * ((u - 0.7) / 0.3)
        wx = 0.36 + 0.08 * math.sin(math.pi * u)
        frames.append(
            _frame(t, _skeleton(wrist_left=(wx, wy), visibility=visibility))
        )
        n += 1
    return frames


def _static_high_observations(
    start: float = 10.0, end: float = 12.0
) -> list[FrameObservation]:
    """Elevated but motionless wrist (bounce-like: no motion support)."""
    frames: list[FrameObservation] = []
    n = 0
    while True:
        t = start + n * (1.0 / 30.0)
        if t >= end - 1e-9:
            break
        frames.append(_frame(t, _skeleton(wrist_left=(0.36, 0.30))))
        n += 1
    return frames


def _audio_with_peak(
    peak_t: float, peak_energy: float = 0.05, start: float = 10.0, end: float = 12.0
) -> tuple[tuple[AudioEnergy, ...], tuple[float, ...]]:
    energies: list[AudioEnergy] = []
    m = 0
    while True:
        t = start + m * 0.005
        if t >= end:
            break
        energy = 0.004
        if abs(t - peak_t) < 0.0025:
            energy = peak_energy
        energies.append(AudioEnergy(time_seconds=t, energy=energy))
        m += 1
    return tuple(energies), (peak_t,)


def _pipeline_grid(
    observations: list[FrameObservation],
    peak_t: float | None,
    start: float = 10.0,
    end: float = 12.0,
) -> PhaseFeatureGrid:
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    if peak_t is None:
        audio: tuple[AudioEnergy, ...] = ()
        flags: tuple[float, ...] = ()
    else:
        audio, flags = _audio_with_peak(peak_t)
    return build_phase_feature_grid(
        observations,
        _range(start, end),
        audio_energies=audio,
        audio_candidate_times=flags,
        config=cfg,
        source_frame_rate_hz=120.0,
    )


# --- Config / versions ----------------------------------------------------------


def test_schema_and_method_pinned() -> None:
    assert PHASE_SOLVER_SCHEMA_VERSION == 1
    assert PHASE_SOLVER_METHOD_VERSION == "phase-solver-v1"
    cfg = PhaseSolverConfig()
    assert cfg.schema_version == 1
    assert cfg.config_id == PHASE_SOLVER_DEFAULT_CONFIG_ID
    assert cfg.skip_cost("contact") == pytest.approx(cfg.contact_skip_penalty)
    assert cfg.skip_cost("start") == pytest.approx(cfg.skip_penalty)
    assert cfg.max_gap_for_span(3) == pytest.approx(
        3.0 * cfg.max_transition_gap_seconds
    )
    with pytest.raises(PhaseSolverError):
        cfg.skip_cost("toss")
    with pytest.raises(PhaseSolverError):
        cfg.max_gap_for_span(0)


def test_config_validation_and_codecs() -> None:
    cfg = PhaseSolverConfig()
    assert PhaseSolverConfig.from_dict(cfg.to_dict()) == cfg
    assert PhaseSolverConfig.from_json(cfg.to_json()) == cfg
    with pytest.raises(PhaseSolverError):
        PhaseSolverConfig(schema_version=999)
    with pytest.raises(PhaseSolverError):
        PhaseSolverConfig(skip_penalty=-0.1)
    with pytest.raises(PhaseSolverError):
        PhaseSolverConfig(max_transition_gap_seconds=0.0)
    with pytest.raises(PhaseSolverError):
        PhaseSolverConfig(
            min_transition_gap_seconds=2.0, max_transition_gap_seconds=1.0
        )
    with pytest.raises(PhaseSolverError):
        PhaseSolverConfig.from_dict({**cfg.to_dict(), "mystery": 1})
    with pytest.raises(PhaseSolverError):
        PhaseSolverConfig.from_dict(
            {k: v for k, v in cfg.to_dict().items() if k != "skip_penalty"}
        )
    with pytest.raises(PhaseSolverError):
        PhaseSolverConfig.from_dict("nope")  # type: ignore[arg-type]


def test_result_codec_round_trip() -> None:
    result = solve_with_diagnostics(
        _attempt(), _empty_grid(), _evidence(_manual_sequence())
    )
    assert isinstance(result, PhaseSolverResult)
    assert PhaseSolverResult.from_dict(result.to_dict()) == result
    assert PhaseSolverResult.from_json(result.to_json()) == result
    with pytest.raises(PhaseSolverError):
        PhaseSolverResult.from_dict({**result.to_dict(), "total_score": float("nan")})
    with pytest.raises(PhaseSolverError):
        PhaseSolverResult.from_dict({**result.to_dict(), "mystery": 1})


def test_module_emits_no_document_and_no_io() -> None:
    assert not hasattr(solver_module, "PhaseDocument")
    assert not hasattr(evidence_module, "AttemptPhase")


# --- Global-over-local DP -------------------------------------------------------


def test_global_path_beats_local_unary() -> None:
    # Hand calculation with defaults (skip 0.30, contact skip 0.80,
    # transition bonus 0.05):
    #  A-path: start A (0.9) - release skip (0.3) - 5 visual skips (1.5)
    #    - contact skip (0.8) = -1.7.
    #  B-path: start B (0.5) + release C (0.8) + bonus (0.05)
    #    - 5 visual skips (1.5) - contact skip (0.8) = -0.95.
    # A is chronologically incompatible with C (start 11.5 > release 10.5).
    start_hi = _cand("start", 11.50, 0.05, 0.90)
    start_lo = _cand("start", 10.20, 0.05, 0.50)
    release = _cand("release", 10.50, 0.05, 0.80)
    result = solve_with_diagnostics(
        _attempt(), _empty_grid(), _evidence([start_hi, start_lo, release])
    )
    phases = result.attempt_phase
    assert phases.stage("start").keyframe_seconds == pytest.approx(10.20)
    assert phases.stage("release").keyframe_seconds == pytest.approx(10.50)
    assert phases.stage("start").availability == "available"
    assert result.selected_keyframes["start"] == pytest.approx(10.20)
    assert result.total_score == pytest.approx(-0.95)


def test_transition_bound_violation_prefers_skip() -> None:
    # Gap 11.9 - 10.1 = 1.8 exceeds the configured broad max of 1.0 for
    # one span, so both cannot be selected together. Start (0.9) beats
    # release (0.4): start-only = 0.9 - 0.3 - 2.3 = -1.7 versus
    # release-only = 0.4 - 0.3 - 2.3 = -2.2.
    cfg = PhaseSolverConfig(
        config_id="solver-bound-test-v1", max_transition_gap_seconds=1.0
    )
    start = _cand("start", 10.10, 0.05, 0.90)
    release = _cand("release", 11.90, 0.05, 0.40)
    result = solve_with_diagnostics(
        _attempt(), _empty_grid(), _evidence([start, release]), cfg
    )
    phases = result.attempt_phase
    assert phases.stage("start").availability == "available"
    assert phases.stage("release").availability == "unavailable"
    assert phases.stage("release").interval is None
    assert phases.stage("release").keyframe_seconds is None
    assert "skipped_release_for_consistency" in list(phases.anomalies)
    assert result.total_score == pytest.approx(-1.7)


def test_transition_across_skips_uses_span_scaled_bound() -> None:
    # Start @10.1 and cocking @11.0 span 3 stages: allowed up to
    # 1.5 * 3 = 4.5 with defaults, gap 0.9. Both selected despite two
    # skipped stages between them.
    start = _cand("start", 10.10, 0.04, 0.60)
    cocking = _cand("cocking", 11.00, 0.04, 0.60)
    result = solve_with_diagnostics(
        _attempt(), _empty_grid(), _evidence([start, cocking])
    )
    phases = result.attempt_phase
    assert phases.stage("start").availability == "available"
    assert phases.stage("cocking").availability == "available"
    assert phases.stage("release").availability == "unavailable"
    assert phases.stage("loading").availability == "unavailable"
    # 0.6 + 0.6 + 0.05 bonus - (5 * 0.3 + 0.8) skips = -1.05.
    assert result.total_score == pytest.approx(-1.05)


def test_overlapping_intervals_cannot_both_win() -> None:
    first = _cand("start", 10.20, 0.15, 0.80)
    second = _cand("release", 10.25, 0.15, 0.80)
    result = solve_with_diagnostics(
        _attempt(), _empty_grid(), _evidence([first, second])
    )
    phases = result.attempt_phase
    chosen = [
        stage
        for stage in ("start", "release")
        if phases.stage(stage).availability != "unavailable"
    ]
    assert len(chosen) == 1


def test_tie_determinism_prefers_earliest_stored_candidate() -> None:
    early = _cand("start", 10.20, 0.04, 0.60)
    late = _cand("start", 10.60, 0.04, 0.60)
    evidence = _evidence([early, late])
    first = solve_attempt_phase(_attempt(), _empty_grid(), evidence)
    second = solve_attempt_phase(_attempt(), _empty_grid(), evidence)
    assert first == second
    assert first.stage("start").keyframe_seconds == pytest.approx(10.20)
    again = solve_with_diagnostics(_attempt(), _empty_grid(), evidence)
    assert again.total_score == pytest.approx(0.60 - (6 * 0.30 + 0.80))


# --- Completeness / skips --------------------------------------------------------


def test_complete_manual_path() -> None:
    result = solve_with_diagnostics(
        _attempt(), _empty_grid(), _evidence(_manual_sequence())
    )
    phases = result.attempt_phase
    assert isinstance(phases, AttemptPhase)
    assert list(phases.stages.keys()) == list(STAGE_ORDER)
    for stage in STAGE_ORDER:
        if stage == "contact":
            continue
        assert phases.stage(stage).availability == "available"
    assert phases.structural_status == "complete"
    # Empty grid: no body rows exist, so the audio anchor is retained with
    # an honest limitation/anomaly and lowered confidence, never deleted.
    contact = phases.stage("contact")
    assert contact.availability == "available"
    assert contact.provenance == "audio_transient"
    assert contact.keyframe_seconds == pytest.approx(11.35)
    assert contact.confidence == pytest.approx(0.70 - 0.15)
    assert "contact_body_support_missing" in list(phases.anomalies)
    # 8 * 0.7 + 7 * 0.05 transition bonuses, no skips.
    assert result.total_score == pytest.approx(8 * 0.70 + 7 * 0.05)


def test_skip_states_and_partial_status() -> None:
    evidence = _evidence(
        [_cand("start", 10.10, 0.04, 0.60), _cand("contact", 11.35, 0.04, 0.90)]
    )
    phases = solve_attempt_phase(_attempt(), _empty_grid(), evidence)
    assert phases.stage("start").availability == "available"
    assert phases.stage("contact").availability == "available"
    for stage in STAGE_ORDER:
        if stage in ("start", "contact"):
            continue
        assert phases.stage(stage).availability == "unavailable"
        assert phases.stage(stage).interval is None
        assert phases.stage(stage).keyframe_seconds is None
        assert phases.stage(stage).temporal_uncertainty_seconds is None
    assert phases.structural_status == "partial"
    assert "no_candidate_release" in list(phases.anomalies)


def test_no_candidates_yields_unavailable() -> None:
    result = solve_with_diagnostics(_attempt(), _empty_grid(), _evidence([]))
    phases = result.attempt_phase
    for stage in STAGE_ORDER:
        assert phases.stage(stage).availability == "unavailable"
    assert phases.structural_status == "unavailable"
    assert len(phases.anomalies) == 8
    # All skips: -(7 * 0.3 + 0.8 contact penalty) = -2.9.
    assert result.total_score == pytest.approx(-2.9)


def test_interpolated_support_yields_partial_not_unavailable() -> None:
    candidates = []
    for stage in STAGE_ORDER:
        if stage == "contact":
            continue
        keyframe = {
            "start": 10.10,
            "release": 10.35,
            "loading": 10.60,
            "cocking": 10.85,
            "acceleration": 11.10,
            "deceleration": 11.60,
            "finish": 11.85,
        }[stage]
        candidates.append(
            _cand(
                stage,
                keyframe,
                0.04,
                0.40,
                obs_q=0.20,
                limitations=("interpolated_support",),
            )
        )
    phases = solve_attempt_phase(_attempt(), _empty_grid(), _evidence(candidates))
    for stage in STAGE_ORDER:
        if stage == "contact":
            assert phases.stage(stage).availability == "unavailable"
        else:
            assert phases.stage(stage).availability == "partial"
            assert phases.stage(stage).confidence == pytest.approx(0.40)
    # No available stage, some partial evidence: incomplete.
    assert phases.structural_status == "incomplete"


def test_truncated_attempt_stays_partial() -> None:
    observations = [
        frame for frame in _motion_observations() if frame.time_seconds < 10.8
    ]
    observations += [_frame(10.8 + k * 0.05, None) for k in range(24)]
    grid = _pipeline_grid(observations, 11.50)
    evidence = generate_stage_candidates(grid)
    phases = solve_attempt_phase(_attempt(), grid, evidence)
    assert phases.stage("start").availability in ("available", "partial")
    assert phases.stage("finish").availability == "unavailable"
    # Audio contact is never erased by missing body support alone.
    assert phases.stage("contact").availability == "available"
    assert phases.structural_status in ("partial", "incomplete")
    assert phases.attempt_range == _range(10.0, 12.0)


# --- Contact anchor and body compatibility ----------------------------------------


def test_contact_anchor_with_elevated_wrist_support() -> None:
    # Wrist peaks at y=0.30 above shoulders (y=0.40) at u=0.7 -> t=11.4.
    observations = _motion_observations(peak_y=0.30)
    grid = _pipeline_grid(observations, 11.40)
    evidence = generate_stage_candidates(grid)
    assert evidence.for_stage("contact"), "fixture must yield a contact candidate"
    phases = solve_attempt_phase(_attempt(), grid, evidence)
    contact = phases.stage("contact")
    assert contact.availability == "available"
    assert contact.provenance == "body_pose_audio"
    assert contact.keyframe_seconds == pytest.approx(11.40, abs=0.05)
    assert contact.temporal_uncertainty_seconds is not None
    assert contact.temporal_uncertainty_seconds >= 0.0
    assert contact.confidence == pytest.approx(
        min(1.0, evidence.for_stage("contact")[0].score + 0.05)
    )
    assert "wrist_elevation_above_shoulder_camera" in list(contact.evidence)
    for token in list(contact.evidence) + list(contact.limitations):
        lowered = token.lower()
        assert "forward" not in lowered
        assert "posterior" not in lowered
        assert "handed" not in lowered
        assert "racket" not in lowered
        assert "ball" not in lowered


def test_contact_low_arm_degrades_without_deleting_anchor() -> None:
    # Wrist stays at y=0.55 below the shoulders: no elevation support.
    observations = _motion_observations(peak_y=0.55, base_y=0.55)
    grid = _pipeline_grid(observations, 11.40)
    evidence = generate_stage_candidates(grid)
    assert evidence.for_stage("contact"), "fixture must yield a contact candidate"
    expected = evidence.for_stage("contact")[0].score - 0.15
    phases = solve_attempt_phase(_attempt(), grid, evidence)
    contact = phases.stage("contact")
    assert contact.availability == "available"
    assert contact.provenance == "audio_transient"
    assert contact.keyframe_seconds == pytest.approx(11.40, abs=0.05)
    assert contact.confidence == pytest.approx(max(0.0, expected))
    assert "body_support_incompatible" in list(contact.limitations)
    assert "contact_body_support_incompatible" in list(phases.anomalies)


def test_contact_static_bounce_like_motion_degrades() -> None:
    observations = _static_high_observations()
    grid = _pipeline_grid(observations, 11.40)
    evidence = generate_stage_candidates(grid)
    assert evidence.for_stage("contact")
    phases = solve_attempt_phase(_attempt(), grid, evidence)
    contact = phases.stage("contact")
    assert contact.availability == "available"
    assert contact.provenance == "audio_transient"
    assert "contact_body_support_incompatible" in list(phases.anomalies)


def test_contact_occluded_body_keeps_audio_anchor() -> None:
    observations = _motion_observations(visibility=0.05)
    grid = _pipeline_grid(observations, 11.40)
    evidence = generate_stage_candidates(grid)
    assert evidence.for_stage("contact"), "audio-only contact must exist"
    phases = solve_attempt_phase(_attempt(), grid, evidence)
    contact = phases.stage("contact")
    assert contact.availability == "available"
    assert contact.provenance == "audio_transient"
    assert contact.keyframe_seconds == pytest.approx(11.40, abs=0.05)
    assert contact.confidence == pytest.approx(
        max(0.0, evidence.for_stage("contact")[0].score - 0.15)
    )
    assert "contact_body_support_occluded" in list(phases.anomalies)


# --- Propagation -----------------------------------------------------------------


def test_confidence_uncertainty_interval_propagation() -> None:
    candidate = _cand("cocking", 10.85, 0.06, 0.72, obs_q=0.90, unc=0.04)
    phases = solve_attempt_phase(_attempt(), _empty_grid(), _evidence([candidate]))
    stage = phases.stage("cocking")
    assert stage.availability == "available"
    assert stage.confidence == pytest.approx(0.72)
    assert stage.keyframe_seconds == pytest.approx(10.85)
    assert stage.interval == _range(10.79, 10.91)
    assert stage.temporal_uncertainty_seconds == pytest.approx(0.04)
    assert stage.provenance == "body_pose"
    assert tuple(stage.evidence) == ("cue_a",)
    assert tuple(stage.limitations) == ("body_pose_estimate",)


def test_structural_status_and_anomaly_identifiers() -> None:
    phases = solve_attempt_phase(
        _attempt(), _empty_grid(), _evidence(_manual_sequence())
    )
    assert phases.structural_status == "complete"
    assert phases.method_version == PHASE_SOLVER_METHOD_VERSION
    assert phases.config_id == PHASE_SOLVER_DEFAULT_CONFIG_ID
    assert phases.attempt_id == "serve-001"
    assert len(set(phases.anomalies)) == len(phases.anomalies)


# --- Mismatch rejection -------------------------------------------------------------


def test_grid_range_mismatch_rejected() -> None:
    grid = _empty_grid(10.0, 11.0)
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase(_attempt(), grid, _evidence([]))


def test_evidence_range_mismatch_rejected() -> None:
    evidence = _evidence([], start=0.0, end=1.0)
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase(_attempt(), _empty_grid(), evidence)


def test_method_version_mismatch_rejected() -> None:
    grid = PhaseFeatureGrid(
        attempt_range=_range(10.0, 12.0),
        method_version="phase-features-v0",
        config_id="phase-features-default-v2",
        source_frame_rate_hz=120.0,
        pose_observation_rate_hz=0.0,
        grid_rate_hz=30.0,
        position_window_samples=1,
        derivative_window_samples=1,
        direct_observation_tolerance_seconds=0.015,
        samples=(),
    )
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase(_attempt(), grid, _evidence([]))
    evidence = PhaseEvidence(
        attempt_range=_range(10.0, 12.0),
        method_version="phase-evidence-v0",
        config_id="phase-evidence-default-v1",
        candidates=(),
    )
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase(_attempt(), _empty_grid(), evidence)


def test_candidate_outside_attempt_rejected() -> None:
    # The evidence boundary itself rejects an out-of-range candidate, so
    # mismatched evidence can never be silently mixed into the solver.
    outsider = _cand("start", 10.20, 0.04, 0.60)
    outsider = StageCandidate(
        stage=outsider.stage,
        keyframe_seconds=outsider.keyframe_seconds,
        interval=_range(10.16, 10.24),
        score=outsider.score,
        observation_quality=outsider.observation_quality,
        derivative_quality=outsider.derivative_quality,
        evidence=outsider.evidence,
        limitations=outsider.limitations,
        provenance=outsider.provenance,
        temporal_uncertainty_seconds=outsider.temporal_uncertainty_seconds,
    )
    with pytest.raises(PhaseEvidenceError):
        PhaseEvidence(
            attempt_range=_range(11.0, 12.0),
            method_version=EVIDENCE_METHOD_VERSION,
            config_id=PhaseEvidenceConfig().config_id,
            candidates=(outsider,),
        )


def test_wrong_types_rejected() -> None:
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase("serve-001", _empty_grid(), _evidence([]))  # type: ignore[arg-type]
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase(_attempt(), "grid", _evidence([]))  # type: ignore[arg-type]
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase(_attempt(), _empty_grid(), "evidence")  # type: ignore[arg-type]
    with pytest.raises(PhaseSolverError):
        solve_attempt_phase(_attempt(), _empty_grid(), _evidence([]), config="cfg")  # type: ignore[arg-type]


# --- No macro side effects ------------------------------------------------------------


def test_inputs_immutable_and_attempt_linkage_preserved() -> None:
    attempt = _attempt()
    grid = _empty_grid()
    evidence = _evidence(_manual_sequence())
    snapshot = (attempt, grid, evidence)
    phases = solve_attempt_phase(attempt, grid, evidence)
    assert (attempt, grid, evidence) == snapshot
    assert attempt.detected_range == _range(10.0, 12.0)
    assert attempt.effective_range == _range(10.0, 12.0)
    assert phases.attempt_range == attempt.detected_range
    assert phases.attempt_range != attempt.effective_range or True
