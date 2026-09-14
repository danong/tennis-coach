"""Synthetic unit tests for the M4.9 deterministic waveform matrix.

Offline, deterministic, and synthetic only: no media decoding, no model
inference, no cache writes, no audio, no private footage. Filtered
tracks are built from synthetic :class:`WorldFrameObservation` rows via
the frozen M4.8 builder, then consumed by
:mod:`serve_review.checkpoints.kinematic_waveforms`.
"""

from __future__ import annotations

import math

import pytest

from serve_review.pose.schema import BodyKeypoint, FrameObservation, PersonBox, PersonObservation
from serve_review.pose.world import WorldFrameObservation, WorldLandmark
from serve_review.checkpoints import world_filter as wf
from serve_review.checkpoints import kinematic_waveforms as kw

FS = 240.0
DT = 1.0 / FS


def _companion(time_seconds: float) -> FrameObservation:
    box = PersonBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    keypoints = tuple(BodyKeypoint(x=0.5, y=0.5, visibility=0.9) for _ in range(33))
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(PersonObservation(box=box, keypoints=keypoints, score=0.9),),
    )


# Canonical skeleton: shoulder_mid=(0,1,0), hip_mid=(0,0,0) -> body 1.0 m.
BASE_POSITIONS: dict[int, tuple[float, float, float]] = {
    11: (-0.25, 1.0, 0.0),  # left_shoulder
    12: (0.25, 1.0, 0.0),  # right_shoulder
    13: (-0.35, 0.6, 0.0),  # left_elbow
    14: (0.35, 0.6, 0.0),  # right_elbow
    15: (-0.40, 0.3, 0.0),  # left_wrist
    16: (0.40, 0.3, 0.0),  # right_wrist
    23: (-0.15, 0.0, 0.0),  # left_hip
    24: (0.15, 0.0, 0.0),  # right_hip
    25: (-0.15, -0.5, 0.0),  # left_knee
    26: (0.15, -0.5, 0.0),  # right_knee
    27: (-0.15, -1.0, 0.0),  # left_ankle
    28: (0.15, -1.0, 0.0),  # right_ankle
}


def _frame_at(
    time_seconds: float,
    overrides: dict[int, tuple[float, float, float]] | None = None,
    *,
    missing: frozenset[int] = frozenset(),
) -> WorldFrameObservation:
    joints: list[WorldLandmark | None] = []
    for index in range(33):
        if index in missing:
            joints.append(None)
        elif overrides is not None and index in overrides:
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


def _constant_track(
    count: int = 40, overrides: dict[int, tuple[float, float, float]] | None = None
) -> wf.FilteredWorldTrack:
    frames = [_frame_at(i * DT, overrides) for i in range(count)]
    return wf.build_filtered_world_track(frames)


def _value(track: kw.KinematicWaveformTrack, row: int, channel: str) -> float | None:
    return track.samples[row].channel_values[kw.CHANNEL_INDEX[channel]]


def _padlen(track: wf.FilteredWorldTrack) -> int:
    import numpy as np

    return wf.default_sos_padlen(np.asarray(track.sos_coefficients))


# --- known angle / relative geometry ------------------------------------------


def test_right_angle_knee_flexion_is_90_degrees() -> None:
    # Right chain: hip=(0,1,0), knee=(0,0,0), ankle=(1,0,0) -> interior 90 -> flexion 90.
    overrides = {24: (0.0, 1.0, 0.0), 26: (0.0, 0.0, 0.0), 28: (1.0, 0.0, 0.0)}
    filtered = _constant_track(40, overrides)
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    for row in range(pad, 40 - pad):
        assert _value(track, row, "knee_flexion_right") == pytest.approx(90.0, abs=0.5)


def test_straight_limb_flexion_is_zero() -> None:
    # Collinear chain -> interior 180 -> flexion 0 (documented convention).
    overrides = {24: (0.0, 1.0, 0.0), 26: (0.0, 0.0, 0.0), 28: (0.0, -1.0, 0.0)}
    filtered = _constant_track(40, overrides)
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    for row in range(pad, 40 - pad):
        assert _value(track, row, "knee_flexion_right") == pytest.approx(0.0, abs=0.5)


def test_relative_geometry_components_and_distance() -> None:
    # Right wrist (1.0, 0.0, 0.0), right shoulder (0.0, 1.0, 0.0),
    # body length forced to 1.0 via canonical hips/shoulders midline...
    # Here override wrist/shoulder exactly; hips stay canonical except
    # shoulders adjusted to keep torso length 1.0.
    overrides = {
        11: (-0.25, 1.0, 0.0),
        12: (0.0, 1.0, 0.0),
        16: (1.0, 0.0, 0.0),
    }
    filtered = _constant_track(40, overrides)
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    # shoulder_mid=(-0.125,1,0), hip_mid=(0,0,0): body = sqrt(0.125^2+1) ~ 1.0078.
    import math as _math

    body = _math.hypot(0.125, 1.0)
    for row in range(pad, 40 - pad):
        dx = _value(track, row, "right_wrist_rel_shoulder_dx")
        dy = _value(track, row, "right_wrist_rel_shoulder_dy")
        dist = _value(track, row, "right_wrist_rel_shoulder_distance")
        assert dx == pytest.approx(1.0 / body, abs=0.02)
        # +y down: upward-positive dy = reference_y - wrist_y = +1.0/body.
        assert dy == pytest.approx(1.0 / body, abs=0.02)
        assert dist == pytest.approx(_math.hypot(1.0, 1.0) / body, abs=0.02)


def test_tilt_and_separation_on_level_geometry() -> None:
    filtered = _constant_track()
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    for row in range(pad, 40 - pad):
        assert _value(track, row, "shoulder_tilt_deg") == pytest.approx(0.0, abs=0.5)
        assert _value(track, row, "hip_tilt_deg") == pytest.approx(0.0, abs=0.5)
        assert _value(track, row, "shoulder_hip_separation_transverse_deg") == pytest.approx(
            0.0, abs=1.0
        )
        # +y down, upward-positive: torso = (hip - shoulder)/body = -1.0,
        # hip = (ankle - hip)/body = -1.0 for the canonical numbers.
        # MediaPipe world landmarks are hip-centered: neither channel measures
        # absolute court/body translation.
        assert _value(track, row, "torso_verticality") == pytest.approx(-1.0, abs=0.02)
        # Hip mid y=0, ankle mid y=-1 -> hip_ankle_vertical_extent = -1.0 body lengths.
        assert _value(track, row, "hip_ankle_vertical_extent") == pytest.approx(-1.0, abs=0.02)


def test_left_arm_elevation_and_extension() -> None:
    filtered = _constant_track()
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    for row in range(pad, 40 - pad):
        # +y down, upward-positive: shoulder_y - wrist_y = 1.0-0.3 = +0.7.
        assert _value(track, row, "left_arm_elevation") == pytest.approx(0.7, abs=0.03)
        assert _value(track, row, "left_arm_extension") == pytest.approx(
            math.hypot(0.15, 0.7), abs=0.03
        )


# --- irregular PTS derivatives -------------------------------------------------


def test_irregular_pts_centered_derivative_matches_analytic_rate() -> None:
    # Wrist x ramps linearly: x = 2*t meters; body length held at 1.0 m, so
    # normalized velocity must be 2.0 body-lengths/s despite irregular PTS.
    times = [0.0, 0.003, 0.009, 0.014, 0.022, 0.027]
    times += [0.027 + i * DT for i in range(1, 60)]
    frames = []
    for t in times:
        overrides = dict(BASE_POSITIONS)
        overrides[16] = (2.0 * t, 0.3, 0.0)
        frames.append(_frame_at(t, overrides))
    filtered = wf.build_filtered_world_track(frames)
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    mid = len(times) // 2
    for row in range(pad + 1, len(times) - pad - 1, 7):
        speed = _value(track, row, "right_wrist_speed")
        assert speed is not None
        assert speed == pytest.approx(2.0, rel=0.08)
    # Grid PTS preserved verbatim through both stages.
    assert [s.time_seconds for s in track.samples] == pytest.approx(times)
    assert _value(track, mid, "right_wrist_speed") == pytest.approx(2.0, rel=0.08)


def test_angle_velocity_on_linear_flexion_ramp() -> None:
    # Drive knee flexion linearly by swinging the ankle in x with time;
    # check the waveform velocity equals the centered PTS slope of the
    # waveform flexion series itself (operator self-consistency on the
    # irregular grid), and one-sided handling at run boundaries.
    count = 60
    times = [i * DT + (0.001 if i % 3 == 0 else 0.0) for i in range(count)]
    frames = []
    for i, t in enumerate(times):
        overrides = dict(BASE_POSITIONS)
        overrides[28] = (0.15 + 0.01 * i, -1.0, 0.0)
        frames.append(_frame_at(t, overrides))
    filtered = wf.build_filtered_world_track(frames)
    track = kw.build_kinematic_waveform_track(filtered)
    flex = [_value(track, r, "knee_flexion_right") for r in range(count)]
    vel = [_value(track, r, "knee_flexion_velocity_right") for r in range(count)]
    assert all(v is not None for v in flex)
    assert all(v is not None for v in vel)
    mid = count // 2
    assert flex[mid] is not None and vel[mid] is not None
    expected = (float(flex[mid + 1]) - float(flex[mid - 1])) / (times[mid + 1] - times[mid - 1])
    assert float(vel[mid]) == pytest.approx(expected, rel=1e-9)
    # Boundary uses the one-sided neighbor only.
    assert flex[0] is not None and flex[1] is not None and vel[0] is not None
    assert float(vel[0]) == pytest.approx(
        (float(flex[1]) - float(flex[0])) / (times[1] - times[0]), rel=1e-9
    )


def test_turning_point_indicators_fire_on_oscillation() -> None:
    count = 120
    frames = []
    for i in range(count):
        t = i * DT
        overrides = dict(BASE_POSITIONS)
        overrides[16] = (0.4 + 0.2 * math.sin(2.0 * math.pi * 2.0 * t), 0.3, 0.0)
        frames.append(_frame_at(t, overrides))
    filtered = wf.build_filtered_world_track(frames)
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    peaks = [_value(track, r, "right_wrist_speed_peak") for r in range(count)]
    troughs = [_value(track, r, "right_wrist_speed_trough") for r in range(count)]
    interior_peaks = peaks[pad + 1 : count - pad - 1]
    interior_troughs = troughs[pad + 1 : count - pad - 1]
    assert all(f in (0.0, 1.0) for f in interior_peaks)
    assert all(f in (0.0, 1.0) for f in interior_troughs)
    assert any(f == 1.0 for f in interior_peaks)  # speed maxima detected
    assert any(f == 1.0 for f in interior_troughs)  # speed minima detected
    assert any(f == 0.0 for f in interior_peaks)
    # Peaks and troughs never coincide on the same qualified triple.
    for peak, trough in zip(interior_peaks, interior_troughs):
        assert not (peak == 1.0 and trough == 1.0)
    # Endpoints can never be extrema (no full triple).
    assert _value(track, 0, "right_wrist_speed_peak") is None
    assert _value(track, count - 1, "right_wrist_speed_peak") is None
    assert _value(track, 0, "right_wrist_speed_trough") is None
    assert _value(track, count - 1, "right_wrist_speed_trough") is None


# --- missingness / quality propagation -----------------------------------------


def test_missing_joint_propagates_without_fabrication() -> None:
    count = 40
    frames = [
        _frame_at(i * DT, dict(BASE_POSITIONS), missing=frozenset({26}))
        for i in range(count)
    ]
    filtered = wf.build_filtered_world_track(frames)
    track = kw.build_kinematic_waveform_track(filtered)
    for row in range(count):
        sample = track.samples[row]
        pos = kw.CHANNEL_INDEX["knee_flexion_right"]
        assert sample.channel_values[pos] is None
        assert sample.channel_available[pos] is False
        assert sample.channel_quality[pos] == 0.0
        vpos = kw.CHANNEL_INDEX["knee_flexion_velocity_right"]
        assert sample.channel_values[vpos] is None
        assert sample.channel_available[vpos] is False
        # Independent left side stays available.
        lpos = kw.CHANNEL_INDEX["knee_flexion_left"]
        assert sample.channel_values[lpos] is not None
        assert sample.channel_available[lpos] is True
    # Settling proxy requires all 12 joints: a missing knee kills it everywhere.
    for row in range(count):
        assert _value(track, row, "whole_body_settling_energy") is None


def test_gap_breaks_derivative_stencil_honestly() -> None:
    # Long gap (> M4.8 0.10 s interpolation bound) splits segments: the
    # gap rows stay honestly unavailable and derivatives cannot bridge it.
    count = 80
    frames = []
    for i in range(count):
        if 30 <= i < 56:
            frames.append(
                _frame_at(i * DT, dict(BASE_POSITIONS), missing=frozenset({16}))
            )
        else:
            frames.append(_frame_at(i * DT, dict(BASE_POSITIONS)))
    filtered = wf.build_filtered_world_track(frames)
    track = kw.build_kinematic_waveform_track(filtered)
    for row in range(30, 56):
        assert _value(track, row, "right_wrist_speed") is None
        assert _value(track, row, "right_wrist_rel_shoulder_distance") is None
    # Run-boundary rows use the qualified one-sided neighbor only, so they
    # stay available; rows inside the gap (no support at all) do not.
    assert _value(track, 29, "right_wrist_speed") is not None
    assert _value(track, 56, "right_wrist_speed") is not None
    # Peak/trough flags need a full qualified triple: unavailable at boundaries.
    assert _value(track, 29, "right_wrist_speed_peak") is None
    assert _value(track, 56, "right_wrist_speed_peak") is None
    assert _value(track, 29, "right_wrist_speed_trough") is None
    assert _value(track, 56, "right_wrist_speed_trough") is None
    # Far from the gap, speed is available again (constant pose -> ~0).
    far = _value(track, 70, "right_wrist_speed")
    assert far is not None and far == pytest.approx(0.0, abs=0.05)


def test_degenerate_body_length_kills_normalized_only() -> None:
    # Collapse shoulders onto hips: torso reference ~0 -> normalized
    # channels unavailable, but pure angle channels survive.
    overrides = {
        11: (0.0, 0.0, 0.0),
        12: (0.0, 0.0, 0.0),
        23: (0.0, 0.0, 0.0),
        24: (0.0, 0.0, 0.0),
        25: (-0.15, -0.5, 0.0),
        26: (0.15, -0.5, 0.0),
        27: (-0.15, -1.0, 0.0),
        28: (0.15, -1.0, 0.0),
    }
    filtered = _constant_track(40, overrides)
    track = kw.build_kinematic_waveform_track(filtered)
    pad = _padlen(filtered)
    row = (pad + (40 - pad)) // 2
    assert _value(track, row, "torso_verticality") is None
    assert _value(track, row, "right_wrist_rel_shoulder_distance") is None
    assert _value(track, row, "right_wrist_speed") is None
    # Knee flexion needs no body reference: still available.
    assert _value(track, row, "knee_flexion_left") is not None


def test_no_fabricated_output_when_everything_missing() -> None:
    from serve_review.pose.schema import FrameObservation as _FO

    frames = []
    for i in range(40):
        t = i * DT
        joints = tuple(None for _ in range(33))
        frames.append(
            WorldFrameObservation(
                time_seconds=t,
                timestamp_ms=int(round(t * 1000)),
                world_landmarks=joints,
                frame_2d=FrameObservation(
                    time_seconds=t, timestamp_ms=int(round(t * 1000)), persons=()
                ),
            )
        )
    filtered = wf.build_filtered_world_track(frames)
    track = kw.build_kinematic_waveform_track(filtered)
    assert len(track.samples) == 40
    for sample in track.samples:
        assert all(v is None for v in sample.channel_values)
        assert all(a is False for a in sample.channel_available)
        assert all(q == 0.0 for q in sample.channel_quality)


def test_quality_reflects_edge_and_interpolated_support() -> None:
    # Short interior gap -> interpolated support with reduced quality;
    # derived channels must not exceed the supporting quality.
    count = 60
    base = [_frame_at(i * DT, dict(BASE_POSITIONS)) for i in range(count)]
    gapped = []
    for i, frame in enumerate(base):
        if i in (30, 31):
            joints = list(frame.world_landmarks)
            joints[16] = None
            gapped.append(
                WorldFrameObservation(
                    time_seconds=frame.time_seconds,
                    timestamp_ms=frame.timestamp_ms,
                    world_landmarks=tuple(joints),
                    frame_2d=frame.frame_2d,
                )
            )
        else:
            gapped.append(frame)
    filtered = wf.build_filtered_world_track(gapped)
    track = kw.build_kinematic_waveform_track(filtered)
    pos = kw.CHANNEL_INDEX["right_wrist_rel_shoulder_distance"]
    assert filtered.samples[30].interpolated[16] is True
    assert track.samples[30].channel_quality[pos] <= 0.5 + 1e-12
    assert track.samples[30].channel_quality[pos] > 0.0
    # Directly observed interior rows carry higher quality.
    assert track.samples[10].channel_quality[pos] >= track.samples[30].channel_quality[pos]
    # Uncertainty never narrower than the M4.8 support bound.
    assert (
        track.samples[30].channel_uncertainty_seconds[pos]
        >= filtered.samples[30].temporal_uncertainty_seconds - 1e-12
    )


# --- deterministic codec / config ----------------------------------------------


def test_channel_inventory_and_units_are_versioned() -> None:
    assert kw.KINEMATIC_WAVEFORMS_SCHEMA_VERSION == 3
    assert kw.KINEMATIC_WAVEFORMS_METHOD_VERSION == "kinematic-waveforms-v3"
    assert kw.COORDINATE_CONVENTION_VERSION == "mediapipe-world-hip-centered-v2"
    assert kw.NORMALIZATION_VERSION == "torso-length-normalization-v1"
    assert kw.DERIVATIVE_METHOD_VERSION == "centered-nonuniform-pts-v1"
    assert kw.NUM_CHANNELS == 40
    assert len(kw.CHANNEL_NAMES) == 40
    assert kw.CHANNEL_NAMES[-2:] == ("audio_transient_energy", "audio_transient_flag")
    assert set(kw.CHANNEL_UNITS) == set(kw.CHANNEL_NAMES)
    for required in (
        "knee_flexion_left",
        "knee_flexion_right",
        "elbow_flexion_left",
        "elbow_flexion_right",
        "shoulder_tilt_deg",
        "hip_tilt_deg",
        "shoulder_hip_separation_transverse_deg",
        "torso_verticality",
        "hip_ankle_vertical_extent",
        "left_arm_elevation",
        "left_arm_extension",
        "right_wrist_speed",
        "right_wrist_acceleration",
        "right_wrist_speed_peak",
        "right_wrist_speed_trough",
        "right_wrist_accel_peak",
        "right_wrist_accel_trough",
        "whole_body_settling_energy",
        "audio_transient_energy",
        "audio_transient_flag",
    ):
        assert required in kw.CHANNEL_INDEX


def test_config_codec_round_trip_and_rejection() -> None:
    config = kw.KinematicWaveformsConfig()
    assert kw.KinematicWaveformsConfig.from_dict(config.to_dict()) == config
    assert kw.KinematicWaveformsConfig.from_json(config.to_json()) == config
    with pytest.raises(kw.KinematicWaveformsError):
        kw.KinematicWaveformsConfig(coordinate_convention="other-v1")
    with pytest.raises(kw.KinematicWaveformsError):
        kw.KinematicWaveformsConfig(normalization="other-v1")
    with pytest.raises(kw.KinematicWaveformsError):
        kw.KinematicWaveformsConfig(derivative_method="other-v1")
    with pytest.raises(kw.KinematicWaveformsError):
        kw.KinematicWaveformsConfig(min_body_length_m=0.0)
    with pytest.raises(kw.KinematicWaveformsError):
        kw.KinematicWaveformsConfig.from_dict(
            {**config.to_dict(), "unexpected": True}
        )
    with pytest.raises(kw.KinematicWaveformsError):
        kw.KinematicWaveformsConfig.from_dict(
            {**config.to_dict(), "schema_version": 999}
        )


def test_track_codec_round_trip_and_determinism() -> None:
    filtered = _constant_track()
    first = kw.build_kinematic_waveform_track(filtered)
    second = kw.build_kinematic_waveform_track(filtered)
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()
    clone = kw.KinematicWaveformTrack.from_dict(first.to_dict())
    assert clone == first
    assert kw.KinematicWaveformTrack.from_json(first.to_json()) == first
    sample_clone = kw.KinematicWaveformSample.from_dict(first.samples[3].to_dict())
    assert sample_clone == first.samples[3]
    # PTS grid reproduced verbatim from M4.8.
    assert [s.time_seconds for s in first.samples] == [
        s.time_seconds for s in filtered.samples
    ]
    assert [s.timestamp_ms for s in first.samples] == [
        s.timestamp_ms for s in filtered.samples
    ]


def test_track_codec_rejects_unknown_keys_and_newer_schema() -> None:
    filtered = _constant_track()
    track = kw.build_kinematic_waveform_track(filtered)
    tampered = dict(track.to_dict())
    tampered["bogus"] = True
    with pytest.raises(kw.KinematicWaveformsError, match="unknown keys"):
        kw.KinematicWaveformTrack.from_dict(tampered)
    newer = dict(track.to_dict())
    newer["schema_version"] = 999
    with pytest.raises(kw.KinematicWaveformsError, match="newer"):
        kw.KinematicWaveformTrack.from_dict(newer)
    bad_names = dict(track.to_dict())
    bad_names["channel_names"] = list(reversed(bad_names["channel_names"]))
    with pytest.raises(kw.KinematicWaveformsError, match="channel_names"):
        kw.KinematicWaveformTrack.from_dict(bad_names)


def test_sample_rejects_inconsistent_availability() -> None:
    filtered = _constant_track()
    track = kw.build_kinematic_waveform_track(filtered)
    payload = track.samples[5].to_dict()
    bad = dict(payload)
    values = list(bad["channel_values"])
    available = list(bad["channel_available"])
    assert values[0] is not None and available[0] is True
    values[0] = None
    bad["channel_values"] = values
    with pytest.raises(kw.KinematicWaveformsError, match="available"):
        kw.KinematicWaveformSample.from_dict(bad)
    bad2 = dict(payload)
    values2 = list(bad2["channel_values"])
    available2 = list(bad2["channel_available"])
    available2[1] = False
    bad2["channel_available"] = available2
    quality2 = list(bad2["channel_quality"])
    quality2[1] = 0.0
    bad2["channel_quality"] = quality2
    assert values2[1] is not None
    with pytest.raises(kw.KinematicWaveformsError, match="unavailable"):
        kw.KinematicWaveformSample.from_dict(bad2)


def test_input_validation() -> None:
    filtered = _constant_track()
    with pytest.raises(kw.KinematicWaveformsError, match="FilteredWorldTrack"):
        kw.build_kinematic_waveform_track(["nope"])  # type: ignore[arg-type]
    with pytest.raises(kw.KinematicWaveformsError, match="KinematicWaveformsConfig"):
        kw.build_kinematic_waveform_track(filtered, config="nope")  # type: ignore[arg-type]
    with pytest.raises(kw.KinematicWaveformsError, match="unknown channel"):
        filtered_track = kw.build_kinematic_waveform_track(filtered)
        filtered_track.channel_series("no_such_channel")
    with pytest.raises(kw.KinematicWaveformsError, match="unknown channel"):
        filtered_track.samples[0].value("no_such_channel")


# --- synthetic audio alignment / codec / missing audio --------------------------


def _audio_positions() -> tuple[int, int]:
    return (
        kw.CHANNEL_INDEX["audio_transient_energy"],
        kw.CHANNEL_INDEX["audio_transient_flag"],
    )


def test_synthetic_audio_alignment_places_energy_and_peak_flag() -> None:
    count = 30
    filtered = _constant_track(count)
    times = [s.time_seconds for s in filtered.samples]
    energies: list[float | None] = [0.02] * count
    energies[15] = 0.90  # synthetic transient spike at one PTS
    track = kw.build_kinematic_waveform_track(filtered, audio_energies=energies)
    epos, fpos = _audio_positions()
    assert track.channel_series("audio_transient_energy") == tuple(energies)
    assert [s.time_seconds for s in track.samples] == times
    # Peak-picked flag: strict local maximum fires exactly at the spike.
    assert track.samples[15].channel_values[fpos] == 1.0
    assert track.samples[15].channel_available[fpos] is True
    assert track.samples[14].channel_values[fpos] == 0.0
    assert track.samples[16].channel_values[fpos] == 0.0
    # Edges can never carry a derived flag (no full triple).
    assert track.samples[0].channel_values[fpos] is None
    assert track.samples[count - 1].channel_values[fpos] is None
    assert track.samples[0].channel_available[fpos] is False
    # Energy availability/quality/uncertainty mirror honest support.
    assert track.samples[15].channel_available[epos] is True
    assert track.samples[15].channel_quality[epos] == pytest.approx(1.0)
    assert track.samples[15].channel_uncertainty_seconds[epos] >= 0.0


def test_audio_explicit_flag_pairs_and_audioenergy_objects_align() -> None:
    count = 10
    filtered = _constant_track(count)
    times = [s.time_seconds for s in filtered.samples]
    pairs = [(0.05, 0.0)] * count
    pairs[4] = (0.70, 1.0)
    track = kw.build_kinematic_waveform_track(filtered, audio_energies=pairs)
    epos, fpos = _audio_positions()
    assert track.samples[4].channel_values[epos] == pytest.approx(0.70)
    assert track.samples[4].channel_values[fpos] == 1.0
    assert track.samples[3].channel_values[fpos] == 0.0
    # AudioEnergy-like objects must match PTS one-for-one in order.
    from serve_review.media.audio import AudioEnergy as _AE

    aligned = [_AE(time_seconds=t, energy=(0.80 if i == 6 else 0.03)) for i, t in enumerate(times)]
    track2 = kw.build_kinematic_waveform_track(filtered, audio_energies=aligned)
    assert track2.samples[6].channel_values[epos] == pytest.approx(0.80)
    assert track2.samples[6].channel_values[fpos] == 1.0
    # Alias spelling carries the same payload.
    track3 = kw.build_kinematic_waveform_track(filtered, audio=list(aligned))
    assert track3.to_dict() == track2.to_dict()
    # Misaligned time raises rather than silently shifting support.
    shifted = [_AE(time_seconds=t + 0.5, energy=0.10) for t in times]
    with pytest.raises(kw.KinematicWaveformsError, match="[Aa]lign|match"):
        kw.build_kinematic_waveform_track(filtered, audio_energies=shifted)


def test_audio_codec_roundtrip_and_missing_audio_is_unavailable() -> None:
    filtered = _constant_track(12)
    epos, fpos = _audio_positions()
    # Absent audio: both channels unavailable on every sample, never zero.
    silent = kw.build_kinematic_waveform_track(filtered)
    for sample in silent.samples:
        assert sample.channel_values[epos] is None
        assert sample.channel_values[fpos] is None
        assert sample.channel_available[epos] is False
        assert sample.channel_available[fpos] is False
        assert sample.channel_quality[epos] == 0.0
        assert sample.channel_quality[fpos] == 0.0
    assert silent.channel_series("audio_transient_energy") == (None,) * 12
    # Present audio round-trips deterministically through both codecs.
    energies = [0.04 + 0.01 * (i % 3) for i in range(12)]
    energies[7] = 0.95
    loud = kw.build_kinematic_waveform_track(filtered, audio_energies=energies)
    assert kw.KinematicWaveformTrack.from_dict(loud.to_dict()) == loud
    assert kw.KinematicWaveformTrack.from_json(loud.to_json()) == loud
    assert kw.build_kinematic_waveform_track(filtered, audio_energies=energies).to_json() == loud.to_json()
    assert loud.samples[7].channel_values[fpos] == 1.0
    # Partial Nones stay missing without fabrication.
    gapped = list(energies)
    gapped[5] = None
    partial = kw.build_kinematic_waveform_track(filtered, audio_energies=gapped)
    assert partial.samples[5].channel_values[epos] is None
    assert partial.samples[5].channel_available[epos] is False
    # Misaligned shapes and invalid values raise.
    with pytest.raises(kw.KinematicWaveformsError, match="exactly one|per filtered"):
        kw.build_kinematic_waveform_track(filtered, audio_energies=[0.1] * 3)
    with pytest.raises(kw.KinematicWaveformsError, match="energy"):
        kw.build_kinematic_waveform_track(filtered, audio_energies=[-0.5] * 12)
    with pytest.raises(kw.KinematicWaveformsError, match="flag"):
        kw.build_kinematic_waveform_track(filtered, audio_energies=[(0.1, 2.0)] * 12)


# --- M4 feature-correctness: +y-down vertical convention -----------------------


def test_wrist_above_gives_positive_elevation_wrist_below_negative() -> None:
    # +y down: elevation = (shoulder_y - wrist_y)/body. Wrist above the
    # shoulder (smaller y) must read positive; below must read negative.
    up = dict(BASE_POSITIONS)
    up[15] = (-0.40, 0.2, 0.0)  # left wrist well above shoulder y=1.0
    track_up = kw.build_kinematic_waveform_track(_constant_track(40, {15: up[15]}))
    pad = _padlen(_constant_track(40))
    row = 20
    assert _value(track_up, row, "left_arm_elevation") is not None
    assert float(_value(track_up, row, "left_arm_elevation")) > 0.0
    down = dict(BASE_POSITIONS)
    down[15] = (-0.40, 1.8, 0.0)  # wrist below shoulder
    track_down = kw.build_kinematic_waveform_track(_constant_track(40, {15: down[15]}))
    assert float(_value(track_down, row, "left_arm_elevation")) < 0.0
    # Right-wrist dy follows the same upward-positive rule.
    up_r = kw.build_kinematic_waveform_track(_constant_track(40, {16: (-0.40, 0.2, 0.0)}))
    # right shoulder y=1.0, wrist y=0.2 -> dy = +0.8/body > 0.
    assert float(_value(up_r, row, "right_wrist_rel_shoulder_dy")) > 0.0
    down_r = kw.build_kinematic_waveform_track(_constant_track(40, {16: (0.40, 1.8, 0.0)}))
    assert float(_value(down_r, row, "right_wrist_rel_shoulder_dy")) < 0.0


def test_tilt_signs_match_y_down_convention() -> None:
    # Left shoulder above right shoulder in the world means left_y < right_y
    # with +y down; tilt must read positive (left higher).
    overrides = {11: (-0.25, 0.8, 0.0), 12: (0.25, 1.2, 0.0)}
    track = kw.build_kinematic_waveform_track(_constant_track(40, overrides))
    row = 20
    assert float(_value(track, row, "shoulder_tilt_deg")) > 0.0
    flipped = {11: (-0.25, 1.2, 0.0), 12: (0.25, 0.8, 0.0)}
    track2 = kw.build_kinematic_waveform_track(_constant_track(40, flipped))
    assert float(_value(track2, row, "shoulder_tilt_deg")) < 0.0
    # Same rule for the hip line.
    hips_up = {23: (-0.15, -0.2, 0.0), 24: (0.15, 0.2, 0.0)}
    track3 = kw.build_kinematic_waveform_track(_constant_track(40, hips_up))
    assert float(_value(track3, row, "hip_tilt_deg")) > 0.0
