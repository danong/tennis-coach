"""Focused M4.2 tests for attempt-local phase features.

Deterministic, offline, synthetic only; no private footage.
"""

from __future__ import annotations

import math

import pytest

from serve_review.checkpoints.phase_features import (
    PHASE_FEATURES_SCHEMA_VERSION,
    PhaseFeatureGrid,
    PhaseFeatureSample,
    PhaseFeaturesConfig,
    PhaseFeaturesError,
    build_phase_feature_grid,
    resolve_direct_observation_tolerance_seconds,
    window_samples_for_seconds,
)
from serve_review.domain import MediaRange
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
    visibility: float = 0.9,
) -> PersonObservation:
    joints: list = []
    for index in range(NUM_KEYPOINTS):
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


def _sine_observations(
    *, start: float, end: float, rate_hz: float, amplitude: float = 0.06, freq_hz: float = 0.5
) -> list[FrameObservation]:
    step = 1.0 / rate_hz
    frames: list[FrameObservation] = []
    n = 0
    while True:
        t = start + n * step
        if t >= end - 1e-9:
            if t < end and (end - t) > 1e-9:
                pass
            else:
                break
        rel = t - start
        x = 0.44 + amplitude * math.sin(2.0 * math.pi * freq_hz * rel)
        frames.append(_frame(t, _skeleton(wrist_left=(x, 0.50))))
        n += 1
    return frames


def _attempt(start: float = 10.0, end: float = 12.0) -> MediaRange:
    return MediaRange(start_seconds=start, end_seconds=end)


# --- Config/version -------------------------------------------------------


def test_schema_version_pinned() -> None:
    assert PHASE_FEATURES_SCHEMA_VERSION == 2
    assert PhaseFeaturesConfig().schema_version == 2
    assert PhaseFeatureGrid(
        attempt_range=_attempt(0.0, 0.5),
        source_frame_rate_hz=120.0,
        pose_observation_rate_hz=0.0,
        grid_rate_hz=30.0,
        position_window_samples=1,
        derivative_window_samples=1,
        direct_observation_tolerance_seconds=0.015,
        samples=(),
    ).schema_version == 2


def test_config_validation() -> None:
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(grid_rate_hz=0.0)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(max_interpolation_gap_seconds=-0.1)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(visibility_floor=1.5)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(position_smoothing_seconds=0.0)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(direct_observation_tolerance_seconds=0.0)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(direct_observation_tolerance_seconds=-0.01)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(config_id="  ")
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig(schema_version=99)
    assert PhaseFeaturesConfig.from_dict(PhaseFeaturesConfig().to_dict()) == PhaseFeaturesConfig()
    with pytest.raises(PhaseFeaturesError):
        PhaseFeaturesConfig.from_dict({**PhaseFeaturesConfig().to_dict(), "extra": 1})


def test_window_conversion_odd_and_cadence_dependent() -> None:
    # 0.12 s at 30 Hz -> round(3.6)=4 -> 5 (odd).
    assert window_samples_for_seconds(0.12, 30.0) == 5
    # Same seconds at 60 Hz -> round(7.2)=7 (odd, unchanged).
    assert window_samples_for_seconds(0.12, 60.0) == 7
    # Even rounding bumps by one: 0.10 s at 30 Hz -> 3 (odd already).
    assert window_samples_for_seconds(0.10, 30.0) == 3
    # 0.05 s at 30 Hz -> round(1.5)=2 -> 3.
    assert window_samples_for_seconds(0.05, 30.0) == 3
    # Time-based windows vary correctly: longer seconds -> larger window.
    assert window_samples_for_seconds(0.24, 30.0) > window_samples_for_seconds(0.12, 30.0)
    with pytest.raises(PhaseFeaturesError):
        window_samples_for_seconds(0.0, 30.0)
    with pytest.raises(PhaseFeaturesError):
        window_samples_for_seconds(0.12, 0.0)


def test_grid_records_canonical_rates_and_windows() -> None:
    obs = _sine_observations(start=10.0, end=12.0, rate_hz=30.0)
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    grid = build_phase_feature_grid(
        obs, _attempt(), config=cfg, source_frame_rate_hz=120.0
    )
    assert grid.source_frame_rate_hz == pytest.approx(120.0)
    assert grid.grid_rate_hz == pytest.approx(30.0)
    # 30 Hz observations over 2 s: (60-1)/span ~= 30 Hz.
    assert grid.pose_observation_rate_hz == pytest.approx(30.0, rel=0.05)
    assert grid.position_window_samples % 2 == 1
    assert grid.derivative_window_samples % 2 == 1
    assert grid.position_window_samples == window_samples_for_seconds(
        cfg.position_smoothing_seconds, cfg.grid_rate_hz
    )


# --- Cadence consistency --------------------------------------------------


def _peak_info(grid: PhaseFeatureGrid) -> tuple[PhaseFeatureSample, float]:
    scored = [(s.wrist_left_x, s) for s in grid if s.wrist_left_x is not None]
    assert scored
    best = max(scored, key=lambda pair: pair[0])
    return best[1], best[0]


@pytest.mark.parametrize("rate_hz", [30.0, 60.0, 120.0])
def test_cadence_consistent_extremum_and_derivatives(rate_hz: float) -> None:
    obs = _sine_observations(start=10.0, end=12.0, rate_hz=rate_hz)
    cfg = PhaseFeaturesConfig(
        grid_rate_hz=30.0,
        position_smoothing_seconds=0.05,
        derivative_window_seconds=0.10,
    )
    grid = build_phase_feature_grid(obs, _attempt(), config=cfg, source_frame_rate_hz=120.0)
    assert len(grid) == 60  # 2 s at 30 Hz.
    peak, _ = _peak_info(grid)
    # Sine peak at rel t=0.5 -> absolute 10.5 (normalized x max).
    assert peak.time_seconds == pytest.approx(10.5, abs=0.08)
    assert peak.wrist_left_x is not None and peak.wrist_left_x > 0.0
    # Velocity near zero at the extremum; acceleration negative (peak).
    assert peak.wrist_left_vx is not None
    assert abs(peak.wrist_left_vx) < 0.45
    assert peak.wrist_left_ax is not None
    assert peak.wrist_left_ax < -0.3


def test_irregular_source_times_preserve_extremum() -> None:
    base = _sine_observations(start=10.0, end=12.0, rate_hz=30.0)
    jittered: list[FrameObservation] = []
    for i, frame in enumerate(base):
        person = frame.persons[0] if frame.has_person else None
        # Deterministic jitter within +-5 ms, endpoints pinned.
        delta = 0.0 if i in (0, len(base) - 1) else (0.005 if i % 2 == 0 else -0.005)
        t = frame.time_seconds + delta
        jittered.append(
            FrameObservation(
                time_seconds=t,
                timestamp_ms=int(round(t * 1000)),
                persons=(person,) if person is not None else (),
            )
        )
    jittered.sort(key=lambda f: f.time_seconds)
    cfg = PhaseFeaturesConfig(
        grid_rate_hz=30.0,
        position_smoothing_seconds=0.05,
        derivative_window_seconds=0.10,
    )
    grid = build_phase_feature_grid(jittered, _attempt(), config=cfg, source_frame_rate_hz=120.0)
    peak, _ = _peak_info(grid)
    assert peak.time_seconds == pytest.approx(10.5, abs=0.10)
    assert peak.wrist_left_vx is not None and abs(peak.wrist_left_vx) < 0.6


# --- Interpolation vs forbidden gaps --------------------------------------


def test_short_span_interpolates_long_gap_does_not() -> None:
    cfg = PhaseFeaturesConfig(
        grid_rate_hz=30.0,
        max_interpolation_gap_seconds=0.15,
        position_smoothing_seconds=0.033,
        derivative_window_seconds=0.066,
    )
    # Dense 30 Hz observations with one short dropout (~0.067 s, 2 frames).
    obs = [_frame(10.0 + i / 30.0, _skeleton()) for i in range(30)]
    short = [f for f in obs if not (10.40 - 1e-9 <= f.time_seconds < 10.47 - 1e-9)]
    grid = build_phase_feature_grid(short, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    assert len(grid) == 30
    gap_samples = [s for s in grid if 10.40 <= s.time_seconds < 10.47]
    assert gap_samples
    for sample in gap_samples:
        assert sample.observed is False
        assert sample.wrist_left_x is not None  # bridged short span
        assert sample.interpolation_span_seconds == pytest.approx(0.10, abs=0.04)
        assert 0.0 < sample.observation_quality < 1.0
    # Long dropout (~0.5 s) exceeds the configured gap: never bridged.
    long_missing = [f for f in obs if not (10.30 <= f.time_seconds < 10.80)]
    grid2 = build_phase_feature_grid(long_missing, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    mid = [s for s in grid2 if 10.40 <= s.time_seconds < 10.70]
    assert mid
    for sample in mid:
        assert sample.observed is False
        assert sample.wrist_left_x is None
        assert sample.elbow_angle_left is None
        assert sample.observation_quality == 0.0
        assert sample.interpolation_span_seconds == 0.0


def test_no_person_and_low_visibility_stay_missing() -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    obs = [
        _frame(10.0, _skeleton()),
        _frame(10.10, None),  # explicit no-person
        _frame(10.20, _skeleton(visibility=0.05)),  # below default floor
        _frame(10.30, _skeleton()),
    ]
    grid = build_phase_feature_grid(obs, _attempt(10.0, 10.40), config=cfg, source_frame_rate_hz=120.0)
    by_time = {round(s.time_seconds, 6): s for s in grid}
    gap = by_time[round(10.10, 6)]
    assert gap.observed is False
    assert gap.wrist_left_x is None
    assert gap.observation_quality == 0.0
    dim = by_time[round(10.20, 6)]
    # Gated out: trajectories honestly missing even though "observed" row exists.
    assert dim.wrist_left_x is None
    assert dim.elbow_angle_left is None


# --- Half-open boundaries -------------------------------------------------


def test_exact_half_open_boundaries() -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=10.0)
    obs = [
        _frame(10.0, _skeleton()),
        _frame(10.5, _skeleton()),
        _frame(11.0, _skeleton()),  # exactly at end: excluded
    ]
    grid = build_phase_feature_grid(obs, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    times = [s.time_seconds for s in grid]
    assert len(times) == 10
    assert times[0] == pytest.approx(10.0)
    assert max(times) < 11.0
    assert all(10.0 <= t < 11.0 for t in times)
    # Observation at the half-open end never leaks into the grid.
    assert grid.pose_observation_rate_hz == pytest.approx(2.0, rel=0.01)


# --- Audio alignment ------------------------------------------------------


def test_timestamp_audio_alignment_and_candidates() -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    obs = _sine_observations(start=10.0, end=11.0, rate_hz=30.0)
    # Dense audio with a sharp peak at 10.50 s.
    audio: list[AudioEnergy] = []
    n = 0
    while True:
        t = 10.0 + n * 0.005
        if t >= 11.0:
            break
        energy = 5.0 if abs(t - 10.50) < 0.0025 else 0.05
        audio.append(AudioEnergy(time_seconds=t, energy=energy))
        n += 1
    grid = build_phase_feature_grid(
        obs,
        _attempt(10.0, 11.0),
        audio_energies=tuple(audio),
        audio_candidate_times=(10.50,),
        config=cfg,
        source_frame_rate_hz=120.0,
    )
    best = max(grid, key=lambda s: s.audio_energy if s.audio_energy is not None else -1.0)
    assert best.audio_energy is not None and best.audio_energy > 1.0
    assert best.time_seconds == pytest.approx(10.50, abs=1.0 / 30.0)
    flagged = [s for s in grid if s.audio_candidate]
    assert flagged
    assert any(abs(s.time_seconds - 10.50) <= 1.0 / 30.0 / 2.0 + 1e-9 for s in flagged)
    far = [s for s in grid if abs(s.time_seconds - 10.50) > 0.10]
    assert all(s.audio_candidate is False for s in far)


# --- Derivative boundary degradation --------------------------------------


def test_derivative_boundary_degradation() -> None:
    obs = _sine_observations(start=10.0, end=12.0, rate_hz=60.0)
    cfg = PhaseFeaturesConfig(
        grid_rate_hz=30.0,
        position_smoothing_seconds=0.12,
        derivative_window_seconds=0.12,
    )
    grid = build_phase_feature_grid(obs, _attempt(), config=cfg, source_frame_rate_hz=120.0)
    interior = [s.derivative_confidence for s in grid[5:-5]]
    edges = [grid[0].derivative_confidence, grid[-1].derivative_confidence]
    assert max(interior) == pytest.approx(1.0, abs=0.05)
    assert all(e < max(interior) for e in edges)
    assert all(0.0 <= s.derivative_confidence <= 1.0 for s in grid)
    assert all(0.0 <= s.derivative_quality <= 1.0 for s in grid)


# --- Normalization invariance ---------------------------------------------


def _affine_observation(person: PersonObservation, *, scale: float, dx: float, dy: float) -> PersonObservation:
    moved = []
    for joint in person.keypoints:
        assert joint is not None
        moved.append(
            BodyKeypoint(
                x=joint.x * scale + dx, y=joint.y * scale + dy, visibility=joint.visibility
            )
        )
    box = person.box
    moved_box = PersonBox(
        x_min=box.x_min * scale + dx,
        y_min=box.y_min * scale + dy,
        x_max=box.x_max * scale + dx,
        y_max=box.y_max * scale + dy,
    )
    return PersonObservation(box=moved_box, keypoints=tuple(moved), score=person.score)


def test_normalized_translation_scale_invariance() -> None:
    obs = _sine_observations(start=10.0, end=11.0, rate_hz=30.0)
    mapped = [
        FrameObservation(
            time_seconds=f.time_seconds,
            timestamp_ms=f.timestamp_ms,
            persons=(_affine_observation(f.persons[0], scale=0.8, dx=0.10, dy=0.05),),
        )
        for f in obs
    ]
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    plain = build_phase_feature_grid(obs, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    shifted = build_phase_feature_grid(mapped, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    for left, right in zip(plain, shifted):
        assert left.wrist_left_x == pytest.approx(right.wrist_left_x, rel=1e-6, abs=1e-9)
        assert left.wrist_left_y == pytest.approx(right.wrist_left_y, rel=1e-6, abs=1e-9)
        assert left.elbow_angle_left == pytest.approx(right.elbow_angle_left, abs=1e-6)


# --- Interpolated precision ------------------------------------------------


def test_interpolated_grid_cannot_claim_higher_precision() -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=60.0, max_interpolation_gap_seconds=0.15)
    obs = _sine_observations(start=10.0, end=11.0, rate_hz=30.0)
    grid = build_phase_feature_grid(obs, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    observed_q = [s.observation_quality for s in grid if s.observed]
    interp_q = [s.observation_quality for s in grid if not s.observed and s.wrist_left_x is not None]
    assert observed_q and interp_q
    assert max(interp_q) < min(observed_q)
    assert all(q < 1.0 for q in interp_q)
    for sample in grid:
        if not sample.observed:
            assert sample.observation_quality < 1.0
        if sample.observed:
            assert sample.interpolation_span_seconds == 0.0


# --- Missingness / determinism / immutability -------------------------------


def test_missingness_honest_and_deterministic() -> None:
    obs = _sine_observations(start=10.0, end=11.0, rate_hz=30.0)
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    first = build_phase_feature_grid(obs, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    second = build_phase_feature_grid(obs, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    assert first == second
    assert [s.to_dict() for s in first] == [s.to_dict() for s in second]
    assert PhaseFeatureGrid.from_dict(first.to_dict()) == first
    # No fabricated values across a fully empty attempt.
    empty = build_phase_feature_grid([], _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0)
    assert len(empty) == 30
    assert all(s.observed is False for s in empty)
    assert all(s.wrist_left_x is None for s in empty)
    assert all(s.observation_quality == 0.0 for s in empty)
    # Camera-relative naming: no anatomical forward/posterior claims.
    for key in PhaseFeatureSample(_attempt(10.0, 11.0).start_seconds).to_dict():
        lowered = key.lower()
        assert "forward" not in lowered
        assert "posterior" not in lowered
        assert "anterior" not in lowered


def test_rejects_unordered_and_bad_inputs() -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    obs = _sine_observations(start=10.0, end=11.0, rate_hz=30.0)
    with pytest.raises(PhaseFeaturesError):
        build_phase_feature_grid(list(reversed(obs)), _attempt(10.0, 11.0), config=cfg)
    with pytest.raises(PhaseFeaturesError):
        build_phase_feature_grid(obs, _attempt(10.0, 11.0), config="nope")  # type: ignore[arg-type]
    with pytest.raises(PhaseFeaturesError):
        build_phase_feature_grid(
            obs,
            _attempt(10.0, 11.0),
            audio_energies=(AudioEnergy(time_seconds=0.2, energy=0.1), AudioEnergy(time_seconds=0.1, energy=0.2)),
            config=cfg,
        )


# --- M4.2 repair: bounded direct-observation alignment ----------------------


def _drift_observations(
    *, start: float, count: int, rate_hz: float
) -> list[FrameObservation]:
    step = 1.0 / rate_hz
    return [_frame(start + n * step, _skeleton()) for n in range(count)]


def test_real_pose_cadence_29_979hz_classifies_direct() -> None:
    # Smoke pattern: 145 genuine 29.979 Hz samples on a nominal 30 Hz grid
    # must classify nearly all as direct observations, not interpolation.
    start, grid_hz, src_hz = 10.0, 30.0, 29.979
    count = 145
    obs = _drift_observations(start=start, count=count, rate_hz=src_hz)
    end = start + count / grid_hz  # 145 grid points over the same span
    cfg = PhaseFeaturesConfig(grid_rate_hz=grid_hz)
    grid = build_phase_feature_grid(
        obs, MediaRange(start_seconds=start, end_seconds=end), config=cfg,
        source_frame_rate_hz=120.0,
    )
    assert len(grid) == count
    observed = [s for s in grid if s.observed]
    # Nearly all compatible samples are direct-observed.
    assert len(observed) >= 130
    for sample in observed:
        assert sample.interpolation_span_seconds == 0.0
        assert sample.wrist_left_x is not None
        assert sample.observation_quality > 0.0
        assert abs(sample.source_time_offset_seconds) <= (
            grid.direct_observation_tolerance_seconds + 1e-12
        )
        assert sample.temporal_uncertainty_seconds + 1e-12 >= abs(
            sample.source_time_offset_seconds
        )
    # Canonical grid coordinates are preserved (uniform, half-open).
    step = 1.0 / grid_hz
    for index, sample in enumerate(grid):
        assert sample.time_seconds == pytest.approx(start + index * step, abs=1e-9)
    assert all(start <= s.time_seconds < end for s in grid)
    # Effective tolerance never bridges distinct frames/gaps.
    assert grid.direct_observation_tolerance_seconds < step / 2.0 + 1e-12
    assert grid.direct_observation_tolerance_seconds < 1.0 / src_hz / 2.0 + 1e-12


def test_direct_alignment_consumes_each_source_at_most_once() -> None:
    start, grid_hz, src_hz = 10.0, 30.0, 29.979
    obs = _drift_observations(start=start, count=145, rate_hz=src_hz)
    end = start + 145 / grid_hz
    cfg = PhaseFeaturesConfig(grid_rate_hz=grid_hz)
    grid = build_phase_feature_grid(
        obs, MediaRange(start_seconds=start, end_seconds=end), config=cfg,
        source_frame_rate_hz=120.0,
    )
    source_times = sorted(
        (s.time_seconds + s.source_time_offset_seconds)
        for s in grid if s.observed
    )
    # Deterministic ties, no double consumption: each source time unique
    # and coincident with a genuine qualified input observation.
    assert len(source_times) == len(set(round(t, 9) for t in source_times))
    obs_times = [f.time_seconds for f in obs]
    for claimed in source_times:
        assert min(abs(claimed - t) for t in obs_times) <= 1e-9
    assert len(source_times) <= len(obs_times)
    # Deterministic rerun consumes identically.
    again = build_phase_feature_grid(
        obs, MediaRange(start_seconds=start, end_seconds=end), config=cfg,
        source_frame_rate_hz=120.0,
    )
    assert [s.source_time_offset_seconds for s in grid] == [
        s.source_time_offset_seconds for s in again
    ]


def test_direct_alignment_tolerance_boundary_rejects() -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    step = 1.0 / 30.0
    # Two qualified sources bracketing a grid point; the nearer sits just
    # outside the effective tolerance so the grid point must not observe.
    effective = resolve_direct_observation_tolerance_seconds(
        cfg, grid_rate_hz=30.0, qualified_source_times=(10.0, 10.0 + step),
    )
    assert 0.0 < effective < step / 2.0
    just_outside = effective + 0.001
    assert just_outside < step / 2.0
    grid_time = 10.0 + step
    # Third source stays >33 ms from the probe so the global minimum gap
    # (and hence the effective tolerance) stays at the grid-capped value.
    obs = [
        _frame(10.0, _skeleton()),
        _frame(grid_time + just_outside, _skeleton()),
        _frame(10.099, _skeleton()),
    ]
    grid = build_phase_feature_grid(
        obs, MediaRange(start_seconds=10.0, end_seconds=10.10),
        config=cfg, source_frame_rate_hz=120.0,
    )
    by_time = {round(s.time_seconds, 9): s for s in grid}
    target = by_time[round(grid_time, 9)]
    assert target.observed is False
    assert target.source_time_offset_seconds == 0.0
    # Just inside the boundary the same geometry must observe.
    just_inside_offset = effective - 0.001
    assert just_inside_offset > 0.0
    obs2 = [
        _frame(10.0, _skeleton()),
        _frame(grid_time + just_inside_offset, _skeleton()),
        _frame(10.099, _skeleton()),
    ]
    grid2 = build_phase_feature_grid(
        obs2, MediaRange(start_seconds=10.0, end_seconds=10.10),
        config=cfg, source_frame_rate_hz=120.0,
    )
    by_time2 = {round(s.time_seconds, 9): s for s in grid2}
    target2 = by_time2[round(grid_time, 9)]
    assert target2.observed is True
    assert target2.source_time_offset_seconds == pytest.approx(
        just_inside_offset, abs=1e-9
    )


def test_direct_alignment_bounded_by_cadence() -> None:
    cfg = PhaseFeaturesConfig(
        grid_rate_hz=30.0, direct_observation_tolerance_seconds=1.0
    )
    # Even a wildly broad configured tolerance is capped below half steps.
    effective = resolve_direct_observation_tolerance_seconds(
        cfg, grid_rate_hz=30.0,
        qualified_source_times=tuple(10.0 + n / 30.0 for n in range(5)),
    )
    assert effective < (1.0 / 30.0) / 2.0 + 1e-12
    assert effective < 0.15 / 2.0 + 1e-12
    # Dense 120 Hz sources cap by the source spacing as well.
    dense = tuple(10.0 + n / 120.0 for n in range(9))
    effective_dense = resolve_direct_observation_tolerance_seconds(
        cfg, grid_rate_hz=30.0, qualified_source_times=dense,
    )
    assert effective_dense < (1.0 / 120.0) / 2.0 + 1e-12


@pytest.mark.parametrize("rate_hz", [30.0, 60.0, 120.0])
def test_direct_alignment_dense_and_irregular_cadence(rate_hz: float) -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    obs = _sine_observations(start=10.0, end=11.0, rate_hz=rate_hz)
    grid = build_phase_feature_grid(
        obs, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0
    )
    assert len(grid) == 30
    observed = [s for s in grid if s.observed]
    # Dense sources always offer a nearby qualified sample per grid point.
    assert len(observed) >= 25
    source_times = [
        s.time_seconds + s.source_time_offset_seconds for s in observed
    ]
    assert len(set(round(t, 9) for t in source_times)) == len(source_times)
    for sample in observed:
        assert sample.temporal_uncertainty_seconds + 1e-12 >= abs(
            sample.source_time_offset_seconds
        )
    # Irregular cadence: 30 Hz base with deterministic ±5 ms jitter stays
    # within the bounded tolerance and remains mostly direct-observed.
    if rate_hz == 30.0:
        jittered: list[FrameObservation] = []
        for i, frame in enumerate(obs):
            person = frame.persons[0] if frame.has_person else None
            delta = 0.0 if i in (0, len(obs) - 1) else (0.005 if i % 2 == 0 else -0.005)
            t = frame.time_seconds + delta
            jittered.append(
                FrameObservation(
                    time_seconds=t,
                    timestamp_ms=int(round(t * 1000)),
                    persons=(person,) if person is not None else (),
                )
            )
        jittered.sort(key=lambda f: f.time_seconds)
        grid_j = build_phase_feature_grid(
            [f for f in jittered if 10.0 <= f.time_seconds < 11.0],
            _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0,
        )
        assert sum(1 for s in grid_j if s.observed) >= 20


def test_source_offset_and_uncertainty_honesty() -> None:
    cfg = PhaseFeaturesConfig(grid_rate_hz=30.0)
    step = 1.0 / 30.0
    # Exact grid times claim zero offset but still carry spacing honesty.
    obs = _sine_observations(start=10.0, end=11.0, rate_hz=30.0)
    grid = build_phase_feature_grid(
        obs, _attempt(10.0, 11.0), config=cfg, source_frame_rate_hz=120.0
    )
    for sample in grid:
        assert math.isfinite(sample.source_time_offset_seconds)
        assert math.isfinite(sample.temporal_uncertainty_seconds)
        assert sample.temporal_uncertainty_seconds >= 0.0
        assert sample.temporal_uncertainty_seconds + 1e-12 >= abs(
            sample.source_time_offset_seconds
        )
        assert sample.temporal_uncertainty_seconds + 1e-12 >= (
            sample.interpolation_span_seconds / 2.0
        )
        if sample.observed:
            assert sample.interpolation_span_seconds == 0.0
            assert sample.temporal_uncertainty_seconds + 1e-12 >= step / 2.0 - 1e-9 or abs(
                sample.source_time_offset_seconds
            ) <= grid.direct_observation_tolerance_seconds + 1e-12
        else:
            assert sample.source_time_offset_seconds == 0.0
            if sample.wrist_left_x is None:
                assert sample.observation_quality == 0.0
    # Grid coordinates stay canonical; source time is recoverable.
    for sample in [s for s in grid if s.observed]:
        source_time = sample.time_seconds + sample.source_time_offset_seconds
        assert min(abs(source_time - f.time_seconds) for f in obs) <= 1e-9
    assert PhaseFeatureGrid.from_dict(grid.to_dict()) == grid
    assert grid.method_version == "phase-features-v2"


def test_sample_and_grid_codecs_validate_offset_honesty() -> None:
    base = PhaseFeatureSample(time_seconds=10.0)
    assert base.source_time_offset_seconds == 0.0
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(time_seconds=10.0, source_time_offset_seconds=0.01)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(
            time_seconds=10.0, observed=True, source_time_offset_seconds=0.01,
            temporal_uncertainty_seconds=0.001,
        )
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(
            time_seconds=10.0, observed=False, observation_quality=0.5,
            interpolation_span_seconds=0.10, temporal_uncertainty_seconds=0.01,
        )
    good = PhaseFeatureSample(
        time_seconds=10.0, observed=True, observation_quality=0.9,
        source_time_offset_seconds=0.005, temporal_uncertainty_seconds=0.016,
    )
    assert PhaseFeatureSample.from_dict(good.to_dict()) == good
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample.from_dict(
            {k: v for k, v in good.to_dict().items() if k != "source_time_offset_seconds"}
        )


# --- M4.2 numerical-bounds repair: post-smoothing physical clamp ---------


def test_smoothing_extrema_overshoot_clamped_to_physical_bounds() -> None:
    """Near-180 plateau overshoots the quadratic fit; emission stays bounded.

    Valid raw elbow angles ``[~179, 180, 180, 180, ~179]`` (wrist ``x``
    0.361 gives ~179 deg, 0.360 gives exactly 180 deg) fit to ~180.17
    at the centre without a post-smoothing bound, which strict schema
    validation correctly rejects. The repair clamps only the emitted
    smoothed angle positions to [0, 180] at the post-smoothing feature
    boundary; externally constructed out-of-range values must still
    reject.
    """
    wrist_xs = [0.361, 0.360, 0.360, 0.360, 0.361]
    obs = [
        _frame(10.0 + i / 30.0, _skeleton(wrist_left=(x, 0.50)))
        for i, x in enumerate(wrist_xs)
    ]
    cfg = PhaseFeaturesConfig(
        grid_rate_hz=30.0,
        position_smoothing_seconds=0.12,  # 5-sample odd window at 30 Hz
        derivative_window_seconds=0.12,
    )
    # Would raise PhaseFeaturesError("... must lie in [0, 180] ... got
    # 180.17...") before the post-smoothing clamp.
    grid = build_phase_feature_grid(
        obs,
        MediaRange(start_seconds=10.0, end_seconds=10.0 + 5 / 30.0),
        config=cfg,
        source_frame_rate_hz=30.0,
    )
    assert len(grid) == 5
    for sample in grid:
        for key in (
            "elbow_angle_left",
            "elbow_angle_right",
            "knee_angle_left",
            "knee_angle_right",
        ):
            value = getattr(sample, key)
            assert value is None or 0.0 <= value <= 180.0
    centre = grid[2]
    assert centre.elbow_angle_left is not None
    assert centre.elbow_angle_left <= 180.0
    assert centre.elbow_angle_left == pytest.approx(180.0, abs=1e-9)
    # Strict schema validation is not weakened: externally constructed
    # overshoots still reject on every bounded channel.
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(time_seconds=10.0, elbow_angle_left=180.054901949318)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(time_seconds=10.0, elbow_angle_left=200.0)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(time_seconds=10.0, elbow_angle_right=180.05)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(time_seconds=10.0, knee_angle_left=180.05)
    with pytest.raises(PhaseFeaturesError):
        PhaseFeatureSample(time_seconds=10.0, knee_angle_right=-0.5)
    assert PhaseFeatureGrid.from_dict(grid.to_dict()) == grid


def test_clamped_extrema_derivatives_stay_timestamp_aware() -> None:
    """Derivatives near a clamped extremum stay finite and slope-consistent.

    Deliberate treatment: derivatives come from the raw timestamp-aware
    local-quadratic fits (pre-clamp grid times), never from the clamped
    positions, so they remain deterministic and timestamp-aware. Near a
    clamped 180-degree peak the velocity stays finite and near zero
    while the acceleration stays finite and negative (peak curvature)
    instead of being artificially flattened by the clamp.
    """
    wrist_xs = [0.361, 0.360, 0.360, 0.360, 0.361]
    obs = [
        _frame(10.0 + i / 30.0, _skeleton(wrist_left=(x, 0.50)))
        for i, x in enumerate(wrist_xs)
    ]
    cfg = PhaseFeaturesConfig(
        grid_rate_hz=30.0,
        position_smoothing_seconds=0.12,
        derivative_window_seconds=0.12,
    )
    grid = build_phase_feature_grid(
        obs,
        MediaRange(start_seconds=10.0, end_seconds=10.0 + 5 / 30.0),
        config=cfg,
        source_frame_rate_hz=30.0,
    )
    centre = grid[2]
    assert centre.elbow_angle_left == pytest.approx(180.0, abs=1e-9)
    assert centre.elbow_angle_left_vel is not None
    assert centre.elbow_angle_left_acc is not None
    assert math.isfinite(centre.elbow_angle_left_vel)
    assert math.isfinite(centre.elbow_angle_left_acc)
    # Peak consistency: near-zero slope with negative curvature.
    assert abs(centre.elbow_angle_left_vel) < 5.0
    assert centre.elbow_angle_left_acc < 0.0
    for sample in grid:
        for key in (
            "elbow_angle_left_vel",
            "elbow_angle_right_vel",
            "knee_angle_left_vel",
            "knee_angle_right_vel",
            "elbow_angle_left_acc",
            "elbow_angle_right_acc",
            "knee_angle_left_acc",
            "knee_angle_right_acc",
        ):
            value = getattr(sample, key)
            assert value is None or math.isfinite(value)
    # Deterministic rerun yields identical derivatives.
    again = build_phase_feature_grid(
        obs,
        MediaRange(start_seconds=10.0, end_seconds=10.0 + 5 / 30.0),
        config=cfg,
        source_frame_rate_hz=30.0,
    )
    assert [s.elbow_angle_left_vel for s in grid] == [
        s.elbow_angle_left_vel for s in again
    ]
    assert [s.elbow_angle_left_acc for s in grid] == [
        s.elbow_angle_left_acc for s in again
    ]
    assert [s.elbow_angle_left for s in grid] == [
        s.elbow_angle_left for s in again
    ]
