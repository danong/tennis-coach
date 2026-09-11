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
    assert PHASE_FEATURES_SCHEMA_VERSION == 1
    assert PhaseFeaturesConfig().schema_version == 1
    assert PhaseFeatureGrid(
        attempt_range=_attempt(0.0, 0.5),
        source_frame_rate_hz=120.0,
        pose_observation_rate_hz=0.0,
        grid_rate_hz=30.0,
        position_window_samples=1,
        derivative_window_samples=1,
        samples=(),
    ).schema_version == 1


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
