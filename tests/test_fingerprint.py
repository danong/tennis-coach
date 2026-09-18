from __future__ import annotations

import copy

import pytest

from serve_review.fingerprint import (
    ANCHOR_NAMES,
    CHANNEL_NAMES,
    METRIC_NAMES,
    METRIC_UNITS,
    SEGMENT_LAYOUT,
    SEGMENT_NAMES,
    SHAPE,
    AnchorValue,
    MetricValue,
    SequenceSegment,
    ServeFingerprintError,
    ServeFingerprintV1,
    build_serve_fingerprint_v1,
    extract_scalar_metrics,
)
from serve_review.domain import AttemptPhase, MediaRange, StagePhase
from serve_review.checkpoints.kinematic_waveforms import (
    CHANNEL_INDEX as WAVEFORM_CHANNEL_INDEX,
    CHANNEL_NAMES as WAVEFORM_CHANNEL_NAMES,
    KinematicWaveformSample,
    KinematicWaveformTrack,
    KinematicWaveformsConfig,
)


def _sample_fingerprint() -> ServeFingerprintV1:
    anchors = {name: AnchorValue(True, float(i)) for i, name in enumerate(ANCHOR_NAMES)}
    metrics = {name: MetricValue(False, None, unit) for name, unit in METRIC_UNITS.items()}
    segments = tuple(
        SequenceSegment(
            id=name,
            available=False,
            duration_metric=f"{name}_duration",
            values=tuple(tuple(None for _ in CHANNEL_NAMES) for _ in range(16)),
            availability=tuple(tuple(False for _ in CHANNEL_NAMES) for _ in range(16)),
        )
        for name, _, _ in SEGMENT_LAYOUT
    )
    return ServeFingerprintV1(
        source_fingerprint="sha256:abc",
        attempt_start_seconds=0.0,
        attempt_end_seconds=8.0,
        provenance={
            "coordinate_convention": "mediapipe-world-hip-centered-v2",
            "waveform_method_version": "kinematic-waveforms-v3",
            "waveform_config_id": "kinematic-waveforms-default-v3",
            "checkpoint_method_version": "serve-waveform-v1",
            "checkpoint_config_id": "phase-solver-serve-default-v2",
            "body_model_name": "pose",
            "body_model_version": "1",
        },
        anchors=anchors,
        metrics=metrics,
        segments=segments,
    )


def test_v1_fixed_inventories_and_json_round_trip_are_deterministic() -> None:
    fingerprint = _sample_fingerprint()
    encoded = fingerprint.to_json()

    assert len(METRIC_NAMES) == 30
    assert tuple(METRIC_UNITS) == METRIC_NAMES
    assert len(CHANNEL_NAMES) == 12
    assert SEGMENT_NAMES == tuple(row[0] for row in SEGMENT_LAYOUT)
    assert SHAPE == (5, 16, 12)
    decoded = ServeFingerprintV1.from_json(encoded)
    assert decoded == fingerprint
    assert decoded.to_json() == encoded


def test_schema_rejects_inventory_shape_and_missingness_drift() -> None:
    payload = _sample_fingerprint().to_dict()
    bad_order = copy.deepcopy(payload)
    bad_order["normalized_sequence"]["channel_order"][0] = "audio_transient_energy"
    with pytest.raises(ServeFingerprintError):
        ServeFingerprintV1.from_dict(bad_order)

    bad_metric = copy.deepcopy(payload)
    del bad_metric["metrics"][METRIC_NAMES[0]]
    with pytest.raises(ServeFingerprintError):
        ServeFingerprintV1.from_dict(bad_metric)

    bad_missingness = copy.deepcopy(payload)
    bad_missingness["normalized_sequence"]["segments"][0]["values"][0][0] = 0.0
    with pytest.raises(ServeFingerprintError):
        ServeFingerprintV1.from_dict(bad_missingness)


def test_schema_rejects_structurally_unavailable_non_null_cells() -> None:
    values = [[None] * len(CHANNEL_NAMES) for _ in range(16)]
    masks = [[False] * len(CHANNEL_NAMES) for _ in range(16)]
    values[0][0] = 0.0
    with pytest.raises(ServeFingerprintError):
        SequenceSegment("start_to_release", False, "start_to_release_duration",
                        tuple(tuple(row) for row in values),
                        tuple(tuple(row) for row in masks))


def test_schema_rejects_partial_channel_support_in_available_segment() -> None:
    payload = _sample_fingerprint().to_dict()
    segment = payload["normalized_sequence"]["segments"][0]
    segment["available"] = True
    segment["availability"][0][0] = True
    segment["values"][0][0] = 0.0

    with pytest.raises(ServeFingerprintError):
        ServeFingerprintV1.from_dict(payload)


def _waveform(rows: list[dict[str, float | None]]) -> KinematicWaveformTrack:
    samples = []
    for i, supplied in enumerate(rows):
        values = [None] * len(WAVEFORM_CHANNEL_NAMES)
        available = [False] * len(values)
        quality = [0.0] * len(values)
        for channel, value in supplied.items():
            index = WAVEFORM_CHANNEL_INDEX[channel]
            values[index] = value
            available[index] = value is not None
            quality[index] = 1.0 if value is not None else 0.0
        samples.append(KinematicWaveformSample(
            time_seconds=float(i), timestamp_ms=i * 1000,
            channel_values=tuple(values), channel_available=tuple(available),
            channel_quality=tuple(quality),
            channel_uncertainty_seconds=tuple(0.0 for _ in values),
        ))
    return KinematicWaveformTrack(config=KinematicWaveformsConfig(), samples=tuple(samples))


def test_scalar_extraction_uses_exact_anchor_rows_closed_intervals_and_signs() -> None:
    names = (
        "knee_flexion_left", "knee_flexion_right", "knee_flexion_velocity_left",
        "knee_flexion_velocity_right", "shoulder_hip_separation_transverse_deg",
        "shoulder_tilt_deg", "hip_tilt_deg", "torso_verticality",
        "hip_ankle_vertical_extent", "left_arm_elevation", "left_arm_extension",
        "elbow_flexion_right", "elbow_flexion_velocity_right",
        "right_wrist_rel_shoulder_distance", "right_wrist_speed",
    )
    values = [{channel: float(i) for channel in names} for i in range(7)]
    values[1]["knee_flexion_left"] = 11.0
    values[2]["knee_flexion_left"] = 20.0
    values[3]["knee_flexion_left"] = 20.0  # earliest max wins
    values[2]["knee_flexion_velocity_left"] = -4.0
    values[3]["knee_flexion_velocity_left"] = -7.0
    values[4]["knee_flexion_velocity_left"] = -7.0  # earliest extension-rate peak
    values[3]["elbow_flexion_velocity_right"] = -3.0
    values[4]["elbow_flexion_velocity_right"] = -8.0
    values[2]["shoulder_hip_separation_transverse_deg"] = 45.0
    values[3]["shoulder_hip_separation_transverse_deg"] = 45.0
    values[4]["right_wrist_speed"] = 10.0
    values[5]["right_wrist_speed"] = 10.0  # earliest tie wins
    anchors = dict(zip(ANCHOR_NAMES, (0., 1., 2., 3., 5., 6.)))
    metrics = extract_scalar_metrics(_waveform(values), anchors)

    def v(name: str) -> float | None:
        return metrics[name].value

    assert tuple(metrics) == METRIC_NAMES
    assert v("start_to_release_duration") == 1.0
    assert v("knee_flexion_left_at_loading") == 20.0
    assert v("knee_flexion_left_max_release_to_cocking") == 20.0
    assert v("knee_extension_rate_left_peak_loading_to_contact") == 7.0
    assert v("shoulder_hip_separation_max_time_relative_to_contact") == -3.0
    assert v("hitting_elbow_extension_rate_peak_cocking_to_contact") == 8.0
    assert v("hitting_elbow_extension_peak_time_relative_to_contact") == -1.0
    assert v("right_wrist_speed_peak_time_relative_to_contact") == -1.0


def test_extension_rate_is_zero_without_extension_and_elbow_peak_time_is_unavailable() -> None:
    names = ("knee_flexion_velocity_left", "knee_flexion_velocity_right",
             "elbow_flexion_velocity_right")
    values = [{channel: float(i + 1) for channel in names} for i in range(7)]
    anchors = dict(zip(ANCHOR_NAMES, (0., 1., 2., 3., 5., 6.)))

    metrics = extract_scalar_metrics(_waveform(values), anchors)

    assert metrics["knee_extension_rate_left_peak_loading_to_contact"].value == 0.0
    assert metrics["knee_extension_rate_right_peak_loading_to_contact"].value == 0.0
    elbow_rate = metrics["hitting_elbow_extension_rate_peak_cocking_to_contact"]
    elbow_time = metrics["hitting_elbow_extension_peak_time_relative_to_contact"]
    assert elbow_rate.available and elbow_rate.value == 0.0
    assert not elbow_time.available and elbow_time.value is None


def test_scalar_metrics_are_unavailable_for_missing_anchor_or_interior_support() -> None:
    rows = [{"right_wrist_speed": float(i)} for i in range(5)]
    rows[3]["right_wrist_speed"] = None
    anchors = dict(zip(ANCHOR_NAMES, (0., 1., 1., 1., 4., None)))
    metrics = extract_scalar_metrics(_waveform(rows), anchors)
    assert not metrics["contact_to_finish_duration"].available
    assert metrics["contact_to_finish_duration"].value is None
    assert not metrics["right_wrist_speed_peak_cocking_to_contact"].available


def _phase(anchor_times: tuple[float, ...]) -> AttemptPhase:
    direct = dict(zip(("start", "release", "loading", "cocking", "contact", "finish"), anchor_times))
    times = {**direct, "acceleration": anchor_times[3] + 1.0,
             "deceleration": anchor_times[4] + 1.0}
    stages = {}
    for name in ("start", "release", "loading", "cocking", "acceleration",
                 "contact", "deceleration", "finish"):
        time = times[name]
        stages[name] = StagePhase(
            availability="available", provenance="body_pose", confidence=1.0,
            interval=MediaRange(time, time + 0.1), keyframe_seconds=time,
            temporal_uncertainty_seconds=0.0 if name == "contact" else None,
        )
    return AttemptPhase(attempt_id="serve-001", attempt_range=MediaRange(anchor_times[0], anchor_times[-1] + 0.1),
                        method_version="checkpoint-v1", config_id="checkpoint-default-v1",
                        stages=stages, structural_status="complete")


def test_sequence_is_fixed_shape_and_linearly_resampled_in_source_time() -> None:
    channel = "knee_flexion_left"
    waveform = _waveform([{channel: float(i * 3)} for i in range(11)])
    fingerprint = build_serve_fingerprint_v1(
        waveform, _phase((0., 2., 4., 6., 8., 10.)),
        source_fingerprint="sha256:sequence", body_model_name="pose", body_model_version="1",
    )
    assert tuple(len(segment.values) for segment in fingerprint.segments) == (16,) * 5
    assert all(len(row) == 12 for segment in fingerprint.segments for row in segment.values)
    first = fingerprint.segments[0]
    assert first.available
    column = CHANNEL_NAMES.index(channel)
    assert first.availability[0][column]
    assert first.values[0][column] == 0.0
    assert first.values[1][column] == pytest.approx(0.4)
    assert first.values[-1][column] == 6.0


def test_sequence_missing_anchor_and_interior_support_are_explicit() -> None:
    channel = "knee_flexion_left"
    rows = [{channel: float(i)} for i in range(11)]
    rows[1][channel] = None
    waveform = _waveform(rows)
    fingerprint = build_serve_fingerprint_v1(
        waveform, _phase((0., 2., 4., 6., 8., 10.)),
        source_fingerprint="sha256:sequence", body_model_name="pose", body_model_version="1",
    )
    first = fingerprint.segments[0]
    assert first.available
    column = CHANNEL_NAMES.index(channel)
    assert all(row[column] is None for row in first.values)
    assert all(not row[column] for row in first.availability)

    missing = dict(_phase((0., 2., 4., 6., 8., 10.)).stages)
    missing["release"] = StagePhase()
    phase = AttemptPhase(attempt_id="serve-001", attempt_range=MediaRange(0., 10.1),
                         method_version="checkpoint-v1", config_id="checkpoint-default-v1",
                         stages=missing, structural_status="partial")
    absent = build_serve_fingerprint_v1(
        waveform, phase, source_fingerprint="sha256:sequence", body_model_name="pose", body_model_version="1",
    )
    assert not absent.segments[0].available
    assert all(value is None for row in absent.segments[0].values for value in row)
    assert all(not value for row in absent.segments[0].availability for value in row)


def test_piecewise_linear_fixture_checks_scalars_resampling_and_gap_support() -> None:
    sampled_channel = "knee_flexion_left"
    gap_channel = "knee_flexion_right"
    scalar_channel = "right_wrist_speed"
    rows = [{} for _ in range(11)]
    rows[0].update({sampled_channel: 0.0, gap_channel: 2.0})
    rows[1].update({sampled_channel: 15.0, gap_channel: None})
    rows[2].update({sampled_channel: 0.0, gap_channel: 4.0})
    rows[6][scalar_channel] = 0.0
    rows[7][scalar_channel] = 12.0
    rows[8][scalar_channel] = 3.0
    fingerprint = build_serve_fingerprint_v1(
        _waveform(rows), _phase((0., 2., 4., 6., 8., 10.)),
        source_fingerprint="sha256:piecewise", body_model_name="pose", body_model_version="1",
    )

    # The first segment covers t=0..2. These native values are linear on
    # either side of t=1, so every normalized sample has an exact expectation.
    segment = fingerprint.segments[0]
    sampled_column = CHANNEL_NAMES.index(sampled_channel)
    expected = tuple(
        pytest.approx(30.0 * i / 15.0 if i <= 7 else 30.0 * (1.0 - i / 15.0))
        for i in range(16)
    )
    assert tuple(row[sampled_column] for row in segment.values) == expected
    assert all(row[sampled_column] for row in segment.availability)

    # The scalar maximum is the middle native point, with time relative to contact.
    assert fingerprint.metrics["right_wrist_speed_peak_cocking_to_contact"].value == 12.0
    assert fingerprint.metrics["right_wrist_speed_peak_time_relative_to_contact"].value == -1.0

    # A missing native row invalidates the entire channel/segment series.
    gap_column = CHANNEL_NAMES.index(gap_channel)
    assert all(row[gap_column] is None for row in segment.values)
    assert all(not row[gap_column] for row in segment.availability)
