"""Focused M4.3 tests for stage evidence and candidate generation.

Deterministic, offline, synthetic only; no private footage.
"""

from __future__ import annotations

import math

import pytest

from serve_review.checkpoints import evidence as evidence_module
from serve_review.checkpoints.evidence import (
    EVIDENCE_METHOD_VERSION,
    EVIDENCE_SCHEMA_VERSION,
    PhaseEvidence,
    PhaseEvidenceConfig,
    PhaseEvidenceError,
    StageCandidate,
    build_phase_evidence,
    compute_stage_scores,
    generate_stage_candidates,
)
from serve_review.checkpoints.phase_features import (
    PhaseFeaturesConfig,
    build_phase_feature_grid,
)
from serve_review.domain import (
    STAGE_KEYS,
    STAGE_ORDER,
    STAGE_PROVENANCES,
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


def _box() -> PersonBox:
    return PersonBox(x_min=0.1, y_min=0.1, x_max=0.7, y_max=0.9)


def _skeleton(
    *,
    wrist_left: tuple[float, float] = (0.36, 0.50),
    wrist_right: tuple[float, float] = (0.64, 0.50),
    knee_left: tuple[float, float] = (0.44, 0.80),
    knee_right: tuple[float, float] = (0.56, 0.80),
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
        25: knee_left,
        26: knee_right,
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


def _attempt(start: float = 10.0, end: float = 12.0) -> MediaRange:
    return MediaRange(start_seconds=start, end_seconds=end)


def _full_observations(
    start: float = 10.0, end: float = 12.0, rate_hz: float = 30.0
) -> list[FrameObservation]:
    frames: list[FrameObservation] = []
    n = 0
    step = 1.0 / rate_hz
    while True:
        t = start + n * step
        if t >= end - 1e-9:
            break
        u = (t - start) / (end - start)
        # Wrist rises (upward = smaller y) to u~0.7 then recovers.
        if u < 0.7:
            wy = 0.55 - 0.25 * (u / 0.7)
        else:
            wy = 0.30 + 0.15 * ((u - 0.7) / 0.3)
        wx = 0.36 + 0.08 * math.sin(math.pi * u)
        # Knee flexion dip mid-attempt (deeper = larger x offset).
        knee_x = 0.44 + 0.06 * math.sin(math.pi * min(1.0, max(0.0, (u - 0.15) / 0.55)))
        frames.append(
            _frame(
                t,
                _skeleton(
                    wrist_left=(wx, wy),
                    wrist_right=(0.64, 0.50 - 0.05 * math.sin(math.pi * u)),
                    knee_left=(knee_x, 0.80),
                    knee_right=(0.56, 0.80),
                ),
            )
        )
        n += 1
    return frames


def _grid_full(
    start: float = 10.0,
    end: float = 12.0,
    contact_t: float | None = 11.5,
    contact_energy: float = 0.05,
    extra_contacts: tuple[tuple[float, float], ...] = (),
) -> object:
    obs = _full_observations(start, end, 30.0)
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    audio: list[AudioEnergy] = []
    m = 0
    while True:
        t = start + m * 0.005
        if t >= end:
            break
        energy = 0.004
        peaks = [(contact_t, contact_energy)] if contact_t is not None else []
        peaks = list(peaks) + list(extra_contacts)
        for pt, pe in peaks:
            if pt is not None and abs(t - pt) < 0.0025:
                energy = pe
        audio.append(AudioEnergy(time_seconds=t, energy=energy))
        m += 1
    flags = tuple(pt for pt, _ in peaks if pt is not None)
    return build_phase_feature_grid(
        obs,
        _attempt(start, end),
        audio_energies=tuple(audio),
        audio_candidate_times=flags,
        config=cfg,
        source_frame_rate_hz=120.0,
    )


# --- Config / versions ------------------------------------------------------


def test_schema_and_method_pinned() -> None:
    assert EVIDENCE_SCHEMA_VERSION == 1
    cfg = PhaseEvidenceConfig()
    assert cfg.schema_version == 1
    assert cfg.config_id
    assert cfg.max_candidates_per_stage >= 1
    assert 0.0 <= cfg.candidate_floor <= 1.0
    assert 0.0 < cfg.candidate_half_window_seconds < 1.0


def test_config_exposes_broad_search_bounds() -> None:
    cfg = PhaseEvidenceConfig()
    attempt = _attempt()
    for stage in STAGE_ORDER:
        lo_frac, hi_frac = cfg.search_window_fractions(stage)
        assert 0.0 <= lo_frac < hi_frac <= 1.0
        # Broad windows: no narrow sliver.
        assert (hi_frac - lo_frac) >= 0.10
        bounds = cfg.search_interval(attempt, stage)
        assert isinstance(bounds, MediaRange)
        assert bounds.start_seconds >= attempt.start_seconds
        assert bounds.end_seconds <= attempt.end_seconds
        assert bounds.start_seconds < bounds.end_seconds
    with pytest.raises(PhaseEvidenceError):
        cfg.search_window_fractions("toss")
    with pytest.raises(PhaseEvidenceError):
        PhaseEvidenceConfig(start_search_start_frac=0.5, start_search_end_frac=0.2)


def test_canonical_keys_exactly_eight() -> None:
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
    assert tuple(STAGE_KEYS) == tuple(STAGE_ORDER)


# --- Full sequence ----------------------------------------------------------


def test_full_sequence_yields_bounded_candidates() -> None:
    grid = _grid_full()  # type: ignore[assignment]
    cfg = PhaseEvidenceConfig()
    result = generate_stage_candidates(grid, cfg)
    assert isinstance(result, PhaseEvidence)
    assert result.method_version == EVIDENCE_METHOD_VERSION
    assert result.config_id == cfg.config_id
    for stage in STAGE_ORDER:
        group = result.for_stage(stage)
        assert len(group) >= 1, f"stage {stage!r} should yield a candidate"
        assert len(group) <= cfg.max_candidates_per_stage
        for cand in group:
            assert cand.stage == stage
            assert 0.0 <= cand.score <= 1.0
            assert cand.score >= cfg.candidate_floor
            assert 0.0 <= cand.observation_quality <= 1.0
            assert 0.0 <= cand.derivative_quality <= 1.0
            assert cand.interval.start_seconds >= 10.0
            assert cand.interval.end_seconds <= 12.0
            assert cand.interval.contains(cand.keyframe_seconds) or (
                cand.keyframe_seconds == cand.interval.start_seconds
            )
            assert cand.temporal_uncertainty_seconds >= 0.0
    # Scores S_i(t) aligned, bounded, deterministic.
    scores = compute_stage_scores(grid, cfg)
    assert set(scores.keys()) == set(STAGE_ORDER)
    for stage in STAGE_ORDER:
        assert len(scores[stage]) == len(grid.samples)  # type: ignore[arg-type]
        assert all(0.0 <= v <= 1.0 for v in scores[stage])
    again = generate_stage_candidates(grid, cfg)
    assert again == result


def test_candidate_order_ranked_with_time_tiebreak() -> None:
    grid = _grid_full()  # type: ignore[assignment]
    result = generate_stage_candidates(grid)
    for stage in STAGE_ORDER:
        group = result.for_stage(stage)
        keys = [(-c.score, c.keyframe_seconds) for c in group]
        assert keys == sorted(keys)


def test_caps_are_enforced() -> None:
    grid = _grid_full()  # type: ignore[assignment]
    cfg = PhaseEvidenceConfig(max_candidates_per_stage=1)
    result = generate_stage_candidates(grid, cfg)
    for stage in STAGE_ORDER:
        assert len(result.for_stage(stage)) <= 1


# --- Contact ----------------------------------------------------------------


def test_contact_scale_invariant_regression() -> None:
    grid = _grid_full(contact_t=None, extra_contacts=((11.3, 0.005), (11.6, 0.053)))  # type: ignore[assignment]
    cfg = PhaseEvidenceConfig()
    scores = compute_stage_scores(grid, cfg)
    contact = scores["contact"]
    flagged_idx = [i for i, s in enumerate(grid.samples) if s.audio_candidate]  # type: ignore[attr-defined]
    assert flagged_idx, "expected flagged transient rows"
    # Non-flagged rows score zero.
    for i, s in enumerate(grid.samples):  # type: ignore[attr-defined]
        if not s.audio_candidate:
            assert contact[i] == 0.0
    result = generate_stage_candidates(grid, cfg)
    group = result.for_stage("contact")
    assert group, "0.053 peak must yield a contact candidate above the floor"
    assert all(c.score >= cfg.candidate_floor for c in group)
    assert all(c.provenance == "audio_transient" for c in group)
    # Weaker flagged peak ranks lower than the 0.053 peak.
    best = max(group, key=lambda c: c.score)
    assert best.keyframe_seconds == pytest.approx(11.6, abs=0.05)
    near_weak = [c for c in group if abs(c.keyframe_seconds - 11.3) < 0.05]
    if near_weak:
        assert near_weak[0].score < best.score
    # Scale invariance: scaling all energies preserves ranking/scores.
    grid2 = _grid_full(contact_t=None, extra_contacts=((11.3, 0.05), (11.6, 0.53)))  # type: ignore[assignment]
    scores2 = compute_stage_scores(grid2, cfg)
    for a, b in zip(scores["contact"], scores2["contact"]):
        assert a == pytest.approx(b)


def test_contact_requires_audio_flag() -> None:
    obs = _full_observations()
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0)
    audio: list[AudioEnergy] = []
    m = 0
    while True:
        t = 10.0 + m * 0.005
        if t >= 12.0:
            break
        audio.append(AudioEnergy(time_seconds=t, energy=0.09))
        m += 1
    grid = build_phase_feature_grid(
        obs, _attempt(), audio_energies=tuple(audio), audio_candidate_times=(),
        config=cfg_feat, source_frame_rate_hz=120.0,
    )
    result = generate_stage_candidates(grid)
    assert result.for_stage("contact") == ()


def test_contact_respects_anchor_uncertainty() -> None:
    grid = _grid_full()  # type: ignore[assignment]
    result = generate_stage_candidates(grid)
    step = 1.0 / 30.0
    for cand in result.for_stage("contact"):
        assert cand.keyframe_seconds == pytest.approx(11.5, abs=step / 2.0 + 1e-9)
        assert cand.temporal_uncertainty_seconds >= step / 2.0 - 1e-12
        width = cand.interval.end_seconds - cand.interval.start_seconds
        assert width <= 2 * cand.temporal_uncertainty_seconds + 1e-9
        assert width > 0.0


def test_audio_only_contact_does_not_create_body_candidates() -> None:
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0)
    grid = build_phase_feature_grid(
        [], _attempt(),
        audio_energies=(AudioEnergy(time_seconds=11.5, energy=0.05),),
        audio_candidate_times=(11.5,),
        config=cfg_feat, source_frame_rate_hz=120.0,
    )
    result = generate_stage_candidates(grid)
    assert len(result.for_stage("contact")) >= 1
    for stage in STAGE_ORDER:
        if stage == "contact":
            continue
        assert result.for_stage(stage) == ()


# --- Empty / partial --------------------------------------------------------


def test_no_person_suppresses_evidence() -> None:
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0)
    grid = build_phase_feature_grid([], _attempt(), config=cfg_feat, source_frame_rate_hz=120.0)
    result = generate_stage_candidates(grid)
    assert len(result) == 0
    scores = compute_stage_scores(grid)
    for stage in STAGE_ORDER:
        assert all(v == 0.0 for v in scores[stage])


def test_long_gap_suppresses_midpoint_evidence() -> None:
    # Observations only at the edges with a long empty middle.
    early = [t for t in _full_observations() if t.time_seconds < 10.4]
    late = [t for t in _full_observations() if t.time_seconds >= 11.6]
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0, max_interpolation_gap_seconds=0.15)
    grid = build_phase_feature_grid(
        early + late, _attempt(), config=cfg_feat, source_frame_rate_hz=120.0
    )
    scores = compute_stage_scores(grid)
    mid = [s for s in grid.samples if 10.7 <= s.time_seconds < 11.3]
    assert mid
    assert all(s.observed is False and s.wrist_left_x is None for s in mid)
    for stage in STAGE_ORDER:
        if stage == "contact":
            continue
        for sample, value in zip(grid.samples, scores[stage]):
            if 10.7 <= sample.time_seconds < 11.3:
                assert value == 0.0


def test_low_knee_bend_permits_empty_loading() -> None:
    frames: list[FrameObservation] = []
    n = 0
    while True:
        t = 10.0 + n * (1.0 / 30.0)
        if t >= 12.0 - 1e-9:
            break
        u = (t - 10.0) / 2.0
        wy = 0.55 - 0.25 * u
        frames.append(_frame(t, _skeleton(wrist_left=(0.36, wy))))
        n += 1
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0)
    grid = build_phase_feature_grid(frames, _attempt(), config=cfg_feat, source_frame_rate_hz=120.0)
    result = generate_stage_candidates(grid)
    # Flat knee angle: no local minimum evidence is forced.
    assert result.for_stage("loading") == ()


def test_occluded_low_visibility_suppresses() -> None:
    frames: list[FrameObservation] = []
    n = 0
    while True:
        t = 10.0 + n * (1.0 / 30.0)
        if t >= 12.0 - 1e-9:
            break
        frames.append(_frame(t, _skeleton(visibility=0.05)))
        n += 1
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0)
    grid = build_phase_feature_grid(frames, _attempt(), config=cfg_feat, source_frame_rate_hz=120.0)
    result = generate_stage_candidates(grid)
    for stage in STAGE_ORDER:
        if stage == "contact":
            continue
        assert result.for_stage(stage) == ()


def test_truncated_attempt_permits_partial() -> None:
    # Person only in the first 40%: late stages must be allowed to be empty.
    frames = [f for f in _full_observations() if f.time_seconds < 10.8]
    frames += [_frame(10.8 + k * 0.05, None) for k in range(24)]
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0)
    grid = build_phase_feature_grid(frames, _attempt(), config=cfg_feat, source_frame_rate_hz=120.0)
    result = generate_stage_candidates(grid)
    assert len(result.for_stage("start")) >= 1
    assert result.for_stage("finish") == ()


# --- Quality ----------------------------------------------------------------


def test_interpolation_and_low_quality_decrease_scores() -> None:
    dense = _full_observations(rate_hz=30.0)
    sparse = [f for i, f in enumerate(dense) if i % 2 == 0]
    cfg_feat = PhaseFeaturesConfig(grid_rate_hz=30.0)
    grid_dense = build_phase_feature_grid(dense, _attempt(), config=cfg_feat, source_frame_rate_hz=120.0)
    grid_sparse = build_phase_feature_grid(sparse, _attempt(), config=cfg_feat, source_frame_rate_hz=120.0)
    scores_dense = compute_stage_scores(grid_dense)
    scores_sparse = compute_stage_scores(grid_sparse)
    # Interpolated samples score below observed samples of the same grid.
    interp = [v for s, v in zip(grid_sparse.samples, scores_sparse["release"]) if not s.observed]
    obsv = [v for s, v in zip(grid_sparse.samples, scores_sparse["release"]) if s.observed]
    assert obsv and interp
    assert max(interp) < max(obsv)
    assert max(interp) <= PhaseEvidenceConfig().interpolated_score_cap + 1e-12
    # Missing wrist inputs remove release evidence entirely.
    no_wrist: list[FrameObservation] = []
    for frame in dense:
        person = frame.persons[0]
        joints = list(person.keypoints)
        for index in (15, 16):
            old = joints[index]
            assert old is not None
            joints[index] = BodyKeypoint(x=old.x, y=old.y, visibility=0.0)
        no_wrist.append(
            FrameObservation(
                time_seconds=frame.time_seconds,
                timestamp_ms=frame.timestamp_ms,
                persons=(PersonObservation(box=person.box, keypoints=tuple(joints), score=person.score),),
            )
        )
    grid_nowrist = build_phase_feature_grid(no_wrist, _attempt(), config=cfg_feat, source_frame_rate_hz=120.0)
    scores_nowrist = compute_stage_scores(grid_nowrist)
    assert max(scores_nowrist["release"]) < max(scores_dense["release"])
    result_sparse = generate_stage_candidates(grid_sparse)
    for cand in result_sparse.for_stage("release"):
        sample = min(grid_sparse.samples, key=lambda s: abs(s.time_seconds - cand.keyframe_seconds))
        if not sample.observed:
            assert cand.score <= PhaseEvidenceConfig().interpolated_score_cap + 1e-12


# --- Honesty ----------------------------------------------------------------


def test_camera_relative_naming_and_provenance_honesty() -> None:
    grid = _grid_full()  # type: ignore[assignment]
    result = generate_stage_candidates(grid)
    assert len(result) > 0
    for cand in result:
        assert cand.provenance in STAGE_PROVENANCES
        if cand.stage == "contact":
            assert cand.provenance == "audio_transient"
        else:
            assert cand.provenance == "body_pose"
        for token in list(cand.evidence) + list(cand.limitations):
            lowered = token.lower()
            assert "forward" not in lowered
            assert "posterior" not in lowered
            assert "anterior" not in lowered
            assert "handed" not in lowered
            assert "viewpoint" not in lowered
            assert "racket_drop" not in lowered
            assert "racket_orientation" not in lowered
            assert "head_orientation" not in lowered
            assert "ball_racket" not in lowered
    # Module is not a solver: no joint-sequence outputs.
    assert not hasattr(evidence_module, "AttemptPhase")
    assert not hasattr(evidence_module, "StagePhase")
    assert not hasattr(evidence_module, "PhaseDocument")


# --- Codecs -----------------------------------------------------------------


def test_json_round_trips_and_rejection() -> None:
    grid = _grid_full()  # type: ignore[assignment]
    result = generate_stage_candidates(grid)
    assert PhaseEvidence.from_json(result.to_json()) == result
    assert PhaseEvidence.from_dict(result.to_dict()) == result
    first = result.candidates[0]
    assert StageCandidate.from_json(first.to_json()) == first
    assert StageCandidate.from_dict(first.to_dict()) == first
    cfg = PhaseEvidenceConfig()
    assert PhaseEvidenceConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(PhaseEvidenceError):
        StageCandidate.from_dict({**first.to_dict(), "score": 1.5})
    with pytest.raises(PhaseEvidenceError):
        StageCandidate.from_dict({**first.to_dict(), "stage": "toss"})
    with pytest.raises(PhaseEvidenceError):
        StageCandidate.from_dict({**first.to_dict(), "provenance": "ball-racket-visual"})
    with pytest.raises(PhaseEvidenceError):
        StageCandidate.from_dict({**first.to_dict(), "mystery": 1})
    contact = result.for_stage("contact")[0]
    with pytest.raises(PhaseEvidenceError):
        StageCandidate.from_dict({**contact.to_dict(), "provenance": "body_pose"})
    visual = [c for c in result if c.stage != "contact"][0]
    with pytest.raises(PhaseEvidenceError):
        StageCandidate.from_dict({**visual.to_dict(), "provenance": "audio_transient"})
    with pytest.raises(PhaseEvidenceError):
        PhaseEvidence.from_dict({**result.to_dict(), "candidates": "nope"})
    with pytest.raises(PhaseEvidenceError):
        compute_stage_scores(grid, config="nope")  # type: ignore[arg-type]
    with pytest.raises(PhaseEvidenceError):
        generate_stage_candidates(grid, config="nope")  # type: ignore[arg-type]
    with pytest.raises(PhaseEvidenceError):
        result.for_stage("toss")
    assert build_phase_evidence(grid) == generate_stage_candidates(grid)
