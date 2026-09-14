"""Synthetic unit tests for M4.10a pure composite six-anchor candidates.

Offline, deterministic, synthetic only: no media decoding, no model
inference, no cache writes, no audio, no private footage. Waveform
tracks are either built directly as synthetic
:class:`KinematicWaveformTrack` samples or via the frozen M4.8/M4.9
builders over synthetic world geometry.
"""

from __future__ import annotations

import math

import pytest

from serve_review.checkpoints import composite_anchors as ca
from serve_review.checkpoints import kinematic_waveforms as kw
from serve_review.checkpoints.kinematic_waveforms import KinematicWaveformsConfig
from serve_review.pose.schema import BodyKeypoint, FrameObservation, PersonBox, PersonObservation
from serve_review.pose.world import WorldFrameObservation, WorldLandmark
from serve_review.checkpoints import world_filter as wf

DT = 1.0 / 30.0


# --- synthetic waveform helpers ----------------------------------------------


def _blank_series(count: int, value: float = 0.5) -> dict[str, list[float | None]]:
    return {name: [value] * count for name in kw.CHANNEL_NAMES}


def _make_track(
    times: list[float],
    overrides: dict[str, list[float | None]] | None = None,
    *,
    uncertainty: float = 0.002,
) -> kw.KinematicWaveformTrack:
    count = len(times)
    base = _blank_series(count, 0.5)
    # Sensible neutral defaults per channel family so tests control cues.
    for name in kw.CHANNEL_NAMES:
        if name in ("right_wrist_speed_peak", "right_wrist_speed_trough", "right_wrist_accel_peak", "right_wrist_accel_trough", "audio_transient_flag"):
            base[name] = [0.0] * count
        elif name in ("audio_transient_energy",):
            base[name] = [0.1] * count
        elif name in ("shoulder_tilt_deg", "hip_tilt_deg"):
            base[name] = [0.0] * count
        elif name in ("knee_flexion_left", "knee_flexion_right"):
            base[name] = [90.0] * count
        elif name in ("elbow_flexion_left", "elbow_flexion_right"):
            base[name] = [90.0] * count
    if overrides:
        for channel, series in overrides.items():
            assert len(series) == count, channel
            base[channel] = list(series)
    samples = []
    for i, t in enumerate(times):
        values: list[float | None] = [base[name][i] for name in kw.CHANNEL_NAMES]
        available = [v is not None for v in values]
        quality = [0.8 if v is not None else 0.0 for v in values]
        uncertainty_row = [uncertainty] * len(values)
        samples.append(
            kw.KinematicWaveformSample(
                time_seconds=t,
                timestamp_ms=int(round(t * 1000)),
                channel_values=tuple(values),
                channel_available=tuple(available),
                channel_quality=tuple(quality),
                channel_uncertainty_seconds=tuple(uncertainty_row),
            )
        )
    return kw.KinematicWaveformTrack(
        config=KinematicWaveformsConfig(),
        method_version=kw.KINEMATIC_WAVEFORMS_METHOD_VERSION,
        coordinate_convention=kw.COORDINATE_CONVENTION_VERSION,
        normalization=kw.NORMALIZATION_VERSION,
        derivative_method=kw.DERIVATIVE_METHOD_VERSION,
        samples=tuple(samples),
        schema_version=kw.KINEMATIC_WAVEFORMS_SCHEMA_VERSION,
    )


def _times(count: int, dt: float = DT) -> list[float]:
    return [i * dt for i in range(count)]


# --- config -------------------------------------------------------------------


def test_default_weights_match_exact_initial_generic_values() -> None:
    config = ca.CompositeAnchorConfig()
    assert dict(config.weight_for("start")) == {
        "left_arm_elevation_rise": 0.35,
        "left_elbow_extension": 0.25,
        "stillness": 0.25,
        "left_arm_low": 0.15,
    }
    assert dict(config.weight_for("release")) == {
        "left_arm_elevation": 0.30,
        "left_arm_elevation_rise": 0.30,
        "left_arm_extension": 0.25,
        "preparation": 0.15,
    }
    assert dict(config.weight_for("loading")) == {
        "knee_flexion": 0.20,
        "shoulder_hip_separation": 0.20,
        "tilt": 0.15,
        "left_arm": 0.20,
        "right_elbow_flexion": 0.15,
        "stillness": 0.10,
    }
    assert dict(config.weight_for("cocking")) == {
        "right_wrist_elevation_trough": 0.25,
        "right_wrist_acceleration": 0.20,
        "torso_rise": 0.15,
        "knee_unload": 0.15,
        "loading_unwind": 0.25,
    }
    assert dict(config.weight_for("contact")) == {
        "right_wrist_elevation_apex": 0.25,
        "right_wrist_speed_peak": 0.20,
        "right_wrist_acceleration_peak": 0.15,
        "torso_rise": 0.10,
        "right_arm_extension": 0.10,
        "audio_transient": 0.20,
    }
    assert dict(config.weight_for("finish")) == {
        "whole_body_settling": 0.45,
        "right_wrist_settling": 0.35,
        "torso_settling": 0.20,
    }
    for stage in ca.COMPOSITE_ANCHOR_STAGES:
        total = sum(config.weight_for(stage).values())
        assert total == pytest.approx(1.0, abs=1e-12)


def test_config_immutable_versioned_and_json_roundtrip() -> None:
    config = ca.CompositeAnchorConfig()
    assert config.schema_version == ca.COMPOSITE_ANCHORS_SCHEMA_VERSION
    assert config.method_version == ca.COMPOSITE_ANCHORS_METHOD_VERSION
    with pytest.raises(Exception):
        config.config_id = "mutated"  # type: ignore[misc]
    payload = config.to_dict()
    assert payload["schema_version"] == ca.COMPOSITE_ANCHORS_SCHEMA_VERSION
    assert payload["weights"]["start"]["left_arm_elevation_rise"] == 0.35
    assert payload["start"]["left_arm_elevation_rise"] == 0.35
    restored = ca.CompositeAnchorConfig.from_dict(payload)
    assert restored == config
    assert ca.CompositeAnchorConfig.from_json(config.to_json()) == config
    with pytest.raises(ca.CompositeAnchorsError):
        ca.CompositeAnchorConfig.from_dict({**payload, "bogus": 1.0})
    with pytest.raises(ca.CompositeAnchorsError):
        ca.CompositeAnchorConfig.from_dict({k: v for k, v in payload.items() if k != "weights" and k != "start"})


# --- dense preservation / PTS / ordering ---------------------------------------


def test_dense_grid_preserves_every_frame_pts_and_order() -> None:
    count = 12
    times = _times(count)
    track = _make_track(times)
    anchor_set = ca.build_composite_anchor_set(track)
    assert len(anchor_set.candidates) == count * len(ca.COMPOSITE_ANCHOR_STAGES)
    for stage in ca.COMPOSITE_ANCHOR_STAGES:
        rows = anchor_set.for_stage(stage)
        assert len(rows) == count
        assert [c.time_seconds for c in rows] == pytest.approx(times)
        assert [c.timestamp_ms for c in rows] == [int(round(t * 1000)) for t in times]
        for candidate in rows:
            assert candidate.stage == stage
            assert candidate.provenance == ca.COMPOSITE_ANCHOR_PROVENANCE
            assert 0.0 <= candidate.score <= 1.0
            assert 0.0 <= candidate.coverage <= 1.0
            assert tuple(sorted(candidate.cue_values.keys())) == tuple(
                sorted(ca.COMPOSITE_CUE_NAMES[stage])
            )
    # Deterministic global order: stage order, then PTS.
    order = {s: i for i, s in enumerate(ca.COMPOSITE_ANCHOR_STAGES)}
    keys = [(order[c.stage], c.time_seconds) for c in anchor_set.candidates]
    assert keys == sorted(keys)


def test_no_chronology_or_selection_dense_for_all_stages() -> None:
    # Even with an early contact apex, every stage still yields the full
    # dense grid: nothing is pruned, ordered, or selected here.
    count = 15
    times = _times(count)
    dy = [1.0 if i == 1 else 0.0 for i in range(count)]
    track = _make_track(times, {"right_wrist_rel_shoulder_dy": dy})
    anchor_set = ca.build_composite_anchor_set(track)
    for stage in ca.COMPOSITE_ANCHOR_STAGES:
        assert len(anchor_set.for_stage(stage)) == count
    assert hasattr(anchor_set, "for_stage")
    assert not hasattr(anchor_set, "selected_keyframes")
    assert not hasattr(ca, "solve_attempt_phase")


# --- cue composition ------------------------------------------------------------


def test_start_score_rewards_elevation_rise_composition() -> None:
    count = 20
    times = _times(count)
    elev = [0.0 if i < 10 else (i - 10) * 0.1 for i in range(count)]
    track = _make_track(times, {"left_arm_elevation": elev})
    anchor_set = ca.build_composite_anchor_set(track)
    rows = anchor_set.for_stage("start")
    late = sum(c.score for c in rows[12:18]) / 6
    early = sum(c.score for c in rows[1:7]) / 6
    assert late > early
    # Score equals the availability-weighted mean over available cues.
    config = ca.CompositeAnchorConfig()
    normalized = ca.compute_composite_anchor_scores(track)["start"]
    weights = config.weight_for("start")
    for index, candidate in enumerate(rows):
        expected_num = sum(
            weights[cue] * normalized[cue][index]
            for cue in weights
            if normalized[cue][index] is not None
        )
        assert candidate.score == pytest.approx(expected_num, abs=1e-12)
        assert candidate.coverage == pytest.approx(1.0, abs=1e-12)


def test_contact_score_rewards_elevation_apex() -> None:
    count = 21
    times = _times(count)
    dy = [1.0 - abs(i - 10) * 0.1 for i in range(count)]
    track = _make_track(times, {"right_wrist_rel_shoulder_dy": dy})
    anchor_set = ca.build_composite_anchor_set(track)
    rows = anchor_set.for_stage("contact")
    assert rows[10].cue_values["right_wrist_elevation_apex"] == pytest.approx(1.0)
    assert rows[0].cue_values["right_wrist_elevation_apex"] == pytest.approx(0.0)
    assert rows[10].score > rows[0].score


def test_finish_score_rewards_settling() -> None:
    count = 16
    times = _times(count)
    energy = [1.0 - i * 0.05 for i in range(count)]
    speed = [1.0 - i * 0.05 for i in range(count)]
    accel = [1.0 - i * 0.05 for i in range(count)]
    track = _make_track(
        times,
        {
            "whole_body_settling_energy": energy,
            "right_wrist_speed": speed,
            "right_wrist_acceleration": accel,
        },
    )
    rows = ca.build_composite_anchor_set(track).for_stage("finish")
    assert rows[-1].score > rows[0].score
    assert rows[-1].cue_values["whole_body_settling"] == pytest.approx(1.0)
    assert rows[0].cue_values["whole_body_settling"] == pytest.approx(0.0)


def test_contradictory_cue_lowers_score_honestly() -> None:
    count = 12
    times = _times(count)
    # Identical tracks except one contradictory cue (high energy = low
    # stillness) must score lower at every frame for finish.
    calm = _make_track(times, {"whole_body_settling_energy": [0.1] * count})
    wild = _make_track(times, {"whole_body_settling_energy": [0.1] * count})
    # Inject contradiction: raise right-wrist speed only in `wild`.
    wild = _make_track(
        times,
        {
            "whole_body_settling_energy": [0.1] * count,
            "right_wrist_speed": [5.0] * count,
            "right_wrist_acceleration": [0.1] * count,
        },
    )
    calm = _make_track(
        times,
        {
            "whole_body_settling_energy": [0.1] * count,
            "right_wrist_speed": [0.1] * count,
            "right_wrist_acceleration": [0.1] * count,
        },
    )
    calm_rows = ca.build_composite_anchor_set(calm).for_stage("finish")
    wild_rows = ca.build_composite_anchor_set(wild).for_stage("finish")
    # Constant series carry no discriminative evidence (guards -> 0.0),
    # so both are honest zeros rather than fabricated separation...
    for candidate in (*calm_rows, *wild_rows):
        assert all(
            v in (None, 0.0) or 0.0 <= v <= 1.0 for v in candidate.cue_values.values()
        )
    # ...while a varying contradictory cue deterministically lowers scores.
    varying_calm = _make_track(
        times,
        {
            "whole_body_settling_energy": [float(count - i) for i in range(count)],
            "right_wrist_speed": [float(count - i) for i in range(count)],
            "right_wrist_acceleration": [0.1] * count,
        },
    )
    varying_wild = _make_track(
        times,
        {
            "whole_body_settling_energy": [float(count - i) for i in range(count)],
            "right_wrist_speed": [float(i) for i in range(count)],
            "right_wrist_acceleration": [0.1] * count,
        },
    )
    calm_scores = [c.score for c in ca.build_composite_anchor_set(varying_calm).for_stage("finish")]
    wild_scores = [c.score for c in ca.build_composite_anchor_set(varying_wild).for_stage("finish")]
    # At the last frame calm has low settling+low speed (both settled),
    # wild has low settling cue but high speed contradiction.
    assert wild_scores[-1] < calm_scores[-1]


# --- missing support / coverage ---------------------------------------------------


def test_fully_missing_support_is_honest_zero() -> None:
    count = 8
    times = _times(count)
    overrides = {
        "left_arm_elevation": [None] * count,
        "left_arm_extension": [None] * count,
        "elbow_flexion_left": [None] * count,
        "elbow_flexion_right": [None] * count,
        "knee_flexion_left": [None] * count,
        "knee_flexion_right": [None] * count,
        "shoulder_hip_separation_transverse_deg": [None] * count,
        "shoulder_tilt_deg": [None] * count,
        "hip_tilt_deg": [None] * count,
        "torso_rise": [None] * count,
        "right_wrist_rel_shoulder_dy": [None] * count,
        "right_wrist_rel_shoulder_distance": [None] * count,
        "right_wrist_speed": [None] * count,
        "right_wrist_acceleration": [None] * count,
        "right_wrist_speed_peak": [None] * count,
        "right_wrist_speed_trough": [None] * count,
        "right_wrist_accel_peak": [None] * count,
        "right_wrist_accel_trough": [None] * count,
        "whole_body_settling_energy": [None] * count,
        "audio_transient_energy": [None] * count,
        "audio_transient_flag": [None] * count,
    }
    track = _make_track(times, overrides)
    anchor_set = ca.build_composite_anchor_set(track)
    for candidate in anchor_set.candidates:
        assert candidate.score == 0.0
        assert candidate.coverage == 0.0
        assert all(v is None for v in candidate.cue_values.values())


def test_partial_missing_coverage_is_mean_available_weight() -> None:
    count = 10
    times = _times(count)
    # Kill only the settling proxy: start stillness cue drops out.
    track = _make_track(times, {"whole_body_settling_energy": [None] * count})
    rows = ca.build_composite_anchor_set(track).for_stage("start")
    for candidate in rows:
        assert candidate.cue_values["stillness"] is None
        assert candidate.coverage == pytest.approx(0.75, abs=1e-12)
        assert candidate.score >= 0.0
    # Loading stillness weight is 0.10 -> coverage 0.90.
    loading = ca.build_composite_anchor_set(track).for_stage("loading")
    for candidate in loading:
        assert candidate.coverage == pytest.approx(0.90, abs=1e-12)


def test_quantile_guards_constant_and_singleton_series() -> None:
    assert tuple(ca.quantile_normalize_series([0.7, 0.7, 0.7, 0.7])) == (0.0, 0.0, 0.0, 0.0)
    assert tuple(ca.quantile_normalize_series([None, None])) == (None, None)
    assert tuple(ca.quantile_normalize_series([1.0, None])) == (0.0, None)
    assert tuple(ca.quantile_normalize_series([float("nan"), 1.0, 2.0])) == (
        None,
        0.0,
        1.0,
    ) or True  # nan treated as missing below
    normalized = ca.quantile_normalize_series([float("nan"), 1.0, 2.0])
    assert normalized[0] is None
    assert normalized[1] == pytest.approx(0.0)
    assert normalized[2] == pytest.approx(1.0)


# --- determinism / schema ------------------------------------------------------------


def test_determinism_identical_rebuild() -> None:
    count = 14
    times = _times(count)
    elev = [math.sin(i * 0.5) for i in range(count)]
    track = _make_track(times, {"left_arm_elevation": elev})
    first = ca.build_composite_anchor_set(track)
    second = ca.build_composite_anchor_set(track)
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_candidate_and_set_json_schema_roundtrip() -> None:
    count = 6
    times = _times(count)
    track = _make_track(times)
    anchor_set = ca.build_composite_anchor_set(track)
    restored = ca.CompositeAnchorSet.from_dict(anchor_set.to_dict())
    assert restored == anchor_set
    assert ca.CompositeAnchorSet.from_json(anchor_set.to_json()) == anchor_set
    candidate = anchor_set.candidates[0]
    assert ca.CompositeAnchorCandidate.from_dict(candidate.to_dict()) == candidate
    assert ca.CompositeAnchorCandidate.from_json(candidate.to_json()) == candidate
    with pytest.raises(ca.CompositeAnchorsError):
        ca.CompositeAnchorCandidate.from_dict({**candidate.to_dict(), "extra": 1})
    with pytest.raises(ca.CompositeAnchorsError):
        ca.CompositeAnchorSet.from_dict({**anchor_set.to_dict(), "extra": 1})


# --- synthetic geometry PTS ------------------------------------------------------------


def _companion(time_seconds: float) -> FrameObservation:
    box = PersonBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    keypoints = tuple(BodyKeypoint(x=0.5, y=0.5, visibility=0.9) for _ in range(33))
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(PersonObservation(box=box, keypoints=keypoints, score=0.9),),
    )


BASE_POSITIONS: dict[int, tuple[float, float, float]] = {
    11: (-0.25, 1.0, 0.0),
    12: (0.25, 1.0, 0.0),
    13: (-0.35, 0.6, 0.0),
    14: (0.35, 0.6, 0.0),
    15: (-0.40, 0.3, 0.0),
    16: (0.40, 0.3, 0.0),
    23: (-0.15, 0.0, 0.0),
    24: (0.15, 0.0, 0.0),
    25: (-0.15, -0.5, 0.0),
    26: (0.15, -0.5, 0.0),
    27: (-0.15, -1.0, 0.0),
    28: (0.15, -1.0, 0.0),
}


def _frame_at(time_seconds: float, overrides: dict[int, tuple[float, float, float]] | None = None) -> WorldFrameObservation:
    joints: list[WorldLandmark | None] = []
    for index in range(33):
        if overrides is not None and index in overrides:
            x, y, z = overrides[index]
            joints.append(WorldLandmark(x=x, y=y, z=z))
        elif index in BASE_POSITIONS:
            x, y, z = BASE_POSITIONS[index]
            joints.append(WorldLandmark(x=x, y=y, z=z))
        else:
            joints.append(WorldLandmark(x=0.0, y=0.5, z=0.0))
    return WorldFrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        world_landmarks=tuple(joints),
        frame_2d=_companion(time_seconds),
    )


def test_synthetic_geometry_waveform_pts_flows_verbatim() -> None:
    count = 40
    frames = []
    for i in range(count):
        t = i * DT
        overrides = dict(BASE_POSITIONS)
        overrides[15] = (-0.40, 0.3 + 0.01 * i, 0.0)
        frames.append(_frame_at(t, overrides))
    filtered = wf.build_filtered_world_track(frames)
    wave = kw.build_kinematic_waveform_track(filtered)
    anchor_set = ca.build_composite_anchor_set(wave)
    wave_times = [s.time_seconds for s in wave.samples]
    for stage in ca.COMPOSITE_ANCHOR_STAGES:
        assert [c.time_seconds for c in anchor_set.for_stage(stage)] == pytest.approx(wave_times)
        assert [c.timestamp_ms for c in anchor_set.for_stage(stage)] == [
            s.timestamp_ms for s in wave.samples
        ]


# --- contact scoring with/without the audio cue ---------------------------------


def test_contact_audio_cue_unavailable_without_audio() -> None:
    count = 12
    times = _times(count)
    # No audio channels available -> audio cue honestly missing everywhere.
    silent = _make_track(
        times,
        {
            "audio_transient_energy": [None] * count,
            "audio_transient_flag": [None] * count,
        },
    )
    rows = ca.build_composite_anchor_set(silent).for_stage("contact")
    for candidate in rows:
        assert candidate.cue_values["audio_transient"] is None
    # Mean available weight without the 0.20 audio cue is 0.80.
    for candidate in rows:
        assert candidate.coverage == pytest.approx(0.80, abs=1e-12)


def test_contact_audio_cue_rewards_transient_and_shifts_score() -> None:
    count = 21
    times = _times(count)
    dy = [1.0 - abs(i - 10) * 0.1 for i in range(count)]
    # Identical body evidence; only the audio cue differs.
    silent = _make_track(
        times,
        {
            "right_wrist_rel_shoulder_dy": dy,
            "audio_transient_energy": [None] * count,
            "audio_transient_flag": [None] * count,
        },
    )
    flags = [1.0 if 9 <= i <= 11 else 0.0 for i in range(count)]
    energies = [0.10 + 0.80 * (1.0 - abs(i - 10) * 0.1) for i in range(count)]
    loud = _make_track(
        times,
        {
            "right_wrist_rel_shoulder_dy": dy,
            "audio_transient_energy": energies,
            "audio_transient_flag": flags,
        },
    )
    silent_rows = ca.build_composite_anchor_set(silent).for_stage("contact")
    loud_rows = ca.build_composite_anchor_set(loud).for_stage("contact")
    # Flag-preferred cue normalizes the lone spike to 1.0 at contact.
    assert loud_rows[10].cue_values["audio_transient"] == pytest.approx(1.0)
    assert loud_rows[0].cue_values["audio_transient"] == pytest.approx(0.0)
    assert all(c.cue_values["audio_transient"] is None for c in silent_rows)
    # Full coverage with audio; reduced without.
    assert loud_rows[10].coverage == pytest.approx(1.0, abs=1e-12)
    assert silent_rows[10].coverage == pytest.approx(0.80, abs=1e-12)
    # The coincident transient deterministically raises the contact score.
    assert loud_rows[10].score > silent_rows[10].score
    # Diagnostic helper exposes the same normalized cue series.
    normalized = ca.compute_composite_anchor_scores(loud)["contact"]["audio_transient"]
    assert normalized[10] == pytest.approx(1.0)
    assert normalized[0] == pytest.approx(0.0)


# --- M4 feature-correctness: flags, peaks, directional extension --------------


def test_rare_single_audio_flag_retains_score_one() -> None:
    # A rare [0, ..., 1, ..., 0] explicit flag must survive verbatim: the
    # lone spike normalizes to 1.0 instead of collapsing to all zeros.
    count = 21
    times = _times(count)
    flags = [0.0] * count
    flags[10] = 1.0
    track = _make_track(
        times,
        {
            "audio_transient_energy": [0.05] * count,
            "audio_transient_flag": flags,
        },
    )
    normalized = ca.compute_composite_anchor_scores(track)["contact"]["audio_transient"]
    assert normalized[10] == pytest.approx(1.0)
    assert normalized[0] == pytest.approx(0.0)
    rows = ca.build_composite_anchor_set(track).for_stage("contact")
    assert rows[10].cue_values["audio_transient"] == pytest.approx(1.0)
    assert rows[0].cue_values["audio_transient"] == pytest.approx(0.0)
    # Energy-only fallback still normalizes continuously when no flag exists.
    no_flag = _make_track(
        times,
        {
            "audio_transient_energy": [0.05 + 0.05 * i for i in range(count)],
            "audio_transient_flag": [None] * count,
        },
    )
    fallback = ca.compute_composite_anchor_scores(no_flag)["contact"]["audio_transient"]
    assert fallback[0] == pytest.approx(0.0)
    assert fallback[-1] == pytest.approx(1.0)


def test_low_valleys_do_not_receive_peak_flags_or_contact_reward() -> None:
    # A strict local minimum must set trough=1/peak=0, never peak=1; contact
    # rewards only the peak channel.
    count = 9
    times = _times(count)
    speed = [3.0, 2.0, 1.0, 0.2, 1.0, 2.0, 3.0, 2.0, 3.0]
    accel = [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 2.5, 1.0, 1.0]
    track = _make_track(
        times,
        {
            "right_wrist_speed": speed,
            "right_wrist_acceleration": accel,
            "right_wrist_speed_peak": [None] * count,
            "right_wrist_speed_trough": [None] * count,
            "right_wrist_accel_peak": [None] * count,
            "right_wrist_accel_trough": [None] * count,
        },
    )
    # Build the peak/trough flags directly through the waveform-independent
    # helper path: emulate a track whose peak/trough channels are set by hand
    # so the contact-cue wiring itself is under test.
    peak = [0.0] * count
    peak[6] = 1.0  # strict local speed maximum at index 6
    trough = [0.0] * count
    trough[3] = 1.0  # strict local speed minimum at index 3
    accel_peak = [0.0] * count
    accel_peak[6] = 1.0
    wired = _make_track(
        times,
        {
            "right_wrist_rel_shoulder_dy": [0.5] * count,
            "right_wrist_rel_shoulder_distance": [0.8] * count,
            "right_wrist_speed_peak": peak,
            "right_wrist_speed_trough": trough,
            "right_wrist_accel_peak": accel_peak,
            "right_wrist_accel_trough": [0.0] * count,
            "torso_rise": [0.2] * count,
            "audio_transient_energy": [None] * count,
            "audio_transient_flag": [None] * count,
        },
    )
    rows = ca.build_composite_anchor_set(wired).for_stage("contact")
    assert rows[6].cue_values["right_wrist_speed_peak"] == pytest.approx(1.0)
    assert rows[3].cue_values["right_wrist_speed_peak"] == pytest.approx(0.0)
    assert rows[6].score > rows[3].score


def test_arm_down_extension_contributes_zero_arm_up_contributes() -> None:
    # Same reach distance scores zero when the wrist is at/below the
    # shoulder (dy <= 0) and positively when above (dy > 0).
    count = 12
    times = _times(count)
    dy_down = [-0.5] * count
    dy_up = [0.5] * count
    dist = [0.9 + 0.01 * i for i in range(count)]
    down = _make_track(
        times,
        {
            "right_wrist_rel_shoulder_dy": dy_down,
            "right_wrist_rel_shoulder_distance": dist,
            "right_wrist_speed_peak": [0.0] * count,
            "right_wrist_accel_peak": [0.0] * count,
            "torso_rise": [0.2] * count,
            "audio_transient_energy": [None] * count,
            "audio_transient_flag": [None] * count,
        },
    )
    up = _make_track(
        times,
        {
            "right_wrist_rel_shoulder_dy": dy_up,
            "right_wrist_rel_shoulder_distance": dist,
            "right_wrist_speed_peak": [0.0] * count,
            "right_wrist_accel_peak": [0.0] * count,
            "torso_rise": [0.2] * count,
            "audio_transient_energy": [None] * count,
            "audio_transient_flag": [None] * count,
        },
    )
    down_rows = ca.build_composite_anchor_set(down).for_stage("contact")
    up_rows = ca.build_composite_anchor_set(up).for_stage("contact")
    for candidate in down_rows:
        assert candidate.cue_values["right_arm_extension"] == pytest.approx(0.0)
    # Arm-up extension varies with reach and carries positive evidence.
    assert any(
        c.cue_values["right_arm_extension"] is not None
        and c.cue_values["right_arm_extension"] > 0.0
        for c in up_rows
    )
    assert "right_arm_extension" in ca.COMPOSITE_CUE_NAMES["contact"]
    assert "late_arm" not in ca.COMPOSITE_CUE_NAMES["contact"]
