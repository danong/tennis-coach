"""Focused M4.8 tests for segment-safe Butterworth world-track filtering.

Deterministic, offline, and synthetic only: no media decoding, no model
inference, no cache writes, no private footage. Observations are built
directly from :class:`WorldFrameObservation` rows with exact PTS grids.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from serve_review.pose.schema import BodyKeypoint, FrameObservation, PersonBox, PersonObservation
from serve_review.pose.world import WorldFrameObservation, WorldLandmark
from serve_review.checkpoints import world_filter as wf

FS = 240.0  # synthetic native rate (slow-motion-like)
DT = 1.0 / FS


def _companion(time_seconds: float, *, person: bool = True) -> FrameObservation:
    if person:
        box = PersonBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
        keypoints = tuple(
            BodyKeypoint(x=0.5, y=0.5, visibility=0.9) for _ in range(33)
        )
        persons = (PersonObservation(box=box, keypoints=keypoints, score=0.9),)
    else:
        persons = ()
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=persons,
    )


def _frame(
    time_seconds: float,
    values: dict[int, tuple[float, float, float]] | None = None,
    *,
    missing: frozenset[int] = frozenset(),
    person: bool = True,
) -> WorldFrameObservation:
    joints: list[WorldLandmark | None] = []
    for index in range(33):
        if index in missing or not person:
            joints.append(None)
        elif values is not None and index in values:
            x, y, z = values[index]
            joints.append(WorldLandmark(x=x, y=y, z=z))
        else:
            joints.append(WorldLandmark(x=0.0, y=1.0, z=0.0))
    return WorldFrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        world_landmarks=tuple(joints),
        frame_2d=_companion(time_seconds, person=person),
    )


def _sine_track(
    count: int,
    freq_hz: float,
    *,
    amplitude: float = 1.0,
    joint: int = 0,
    start: float = 0.0,
    step: float = DT,
    noise: dict[int, tuple[float, float]] | None = None,
) -> list[WorldFrameObservation]:
    """Track with a sine on one joint/x axis, constants elsewhere."""
    frames: list[WorldFrameObservation] = []
    for i in range(count):
        t = start + i * step
        value = amplitude * math.sin(2.0 * math.pi * freq_hz * t)
        extra: dict[int, tuple[float, float, float]] = {joint: (value, 1.0, 0.0)}
        if noise:
            for j, (noise_freq, noise_amp) in noise.items():
                base = extra.get(j, (0.0, 1.0, 0.0))
                wobble = noise_amp * math.sin(2.0 * math.pi * noise_freq * t)
                extra[j] = (base[0] + wobble, base[1], base[2])
        frames.append(_frame(t, extra))
    return frames


def _series(track: wf.FilteredWorldTrack, joint: int, axis: int) -> np.ndarray:
    out = []
    for sample in track.samples:
        landmark = sample.filtered[joint]
        assert landmark is not None
        out.append((landmark.x, landmark.y, landmark.z)[axis])
    return np.asarray(out, dtype=np.float64)


# --- config -----------------------------------------------------------------


def test_config_defaults_match_approved_design() -> None:
    config = wf.WorldFilterConfig()
    assert config.filter_order == 4
    assert config.cutoff_hz == 12.0
    assert config.min_segment_samples == 21
    assert wf.WORLD_FILTER_SCHEMA_VERSION == 1
    assert wf.WORLD_FILTER_METHOD_VERSION == "butterworth-sos-zerophase-v1"
    assert wf.WORLD_FILTER_DEFAULT_ORDER == 4
    assert wf.WORLD_FILTER_DEFAULT_CUTOFF_HZ == 12.0
    assert wf.WORLD_FILTER_DEFAULT_MIN_SEGMENT_SAMPLES == 21


def test_config_validation_rejects_bad_values() -> None:
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig(filter_order=0)
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig(filter_order=-2)
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig(cutoff_hz=0.0)
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig(cutoff_hz=float("nan"))
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig(min_segment_samples=0)
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig(max_interpolation_gap_seconds=-0.1)
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig(config_id="   ")
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig.from_dict(
            {
                "config_id": "x",
                "cutoff_hz": 12.0,
                "filter_order": 4,
                "max_interpolation_gap_seconds": 0.1,
                "min_segment_samples": 21,
                "schema_version": 1,
                "unexpected": True,
            }
        )
    with pytest.raises(wf.WorldFilterError):
        wf.WorldFilterConfig.from_dict(
            {
                "config_id": "x",
                "cutoff_hz": 12.0,
                "filter_order": 4,
                "max_interpolation_gap_seconds": 0.1,
                "min_segment_samples": 21,
                "schema_version": 999,
            }
        )


def test_config_codec_round_trip() -> None:
    config = wf.WorldFilterConfig()
    assert wf.WorldFilterConfig.from_dict(config.to_dict()) == config
    assert wf.WorldFilterConfig.from_json(config.to_json()) == config


# --- frequency behavior ------------------------------------------------------


def test_passband_sine_preserved_with_zero_phase() -> None:
    count = 240
    frames = _sine_track(count, 2.0)
    track = wf.build_filtered_world_track(frames)
    assert track.sample_rate_hz == pytest.approx(FS)
    assert len(track.samples) == count
    assert all(sample.available[0] for sample in track.samples)
    filtered = _series(track, 0, 0)
    raw = np.array([math.sin(2.0 * math.pi * 2.0 * (i * DT)) for i in range(count)])
    padlen = wf.default_sos_padlen(np.asarray(track.sos_coefficients))
    interior = slice(padlen, count - padlen)
    rmse = float(np.sqrt(np.mean((filtered[interior] - raw[interior]) ** 2)))
    assert rmse < 0.02
    # Zero phase: interior peaks align (no group-delay shift).
    assert int(np.argmax(filtered[interior])) == int(np.argmax(raw[interior]))
    # Cross-correlation peaks at lag zero.
    centered_f = filtered[interior] - filtered[interior].mean()
    centered_r = raw[interior] - raw[interior].mean()
    lags = np.correlate(centered_f, centered_r, mode="full")
    best_lag = int(np.argmax(lags)) - (len(centered_r) - 1)
    assert best_lag == 0


def test_stopband_high_frequency_attenuated() -> None:
    count = 240
    frames = _sine_track(count, 2.0, noise={0: (40.0, 0.5)})
    track = wf.build_filtered_world_track(frames)
    filtered = _series(track, 0, 0)
    pure = np.array([math.sin(2.0 * math.pi * 2.0 * (i * DT)) for i in range(count)])
    raw = np.array([sample.world_landmarks[0].x for sample in frames], dtype=float)
    padlen = wf.default_sos_padlen(np.asarray(track.sos_coefficients))
    interior = slice(padlen, count - padlen)
    raw_rmse = float(np.sqrt(np.mean((raw[interior] - pure[interior]) ** 2)))
    filt_rmse = float(np.sqrt(np.mean((filtered[interior] - pure[interior]) ** 2)))
    assert raw_rmse > 0.3  # sanity: the 40 Hz component is really there
    assert filt_rmse < 0.1 * raw_rmse
    # Residual 40 Hz amplitude in the output is small (least-squares fit).
    t = np.arange(count)[interior] * DT
    basis = np.stack(
        [np.sin(2.0 * math.pi * 40.0 * t), np.cos(2.0 * math.pi * 40.0 * t)],
        axis=1,
    )
    coef, _, _, _ = np.linalg.lstsq(basis, filtered[interior], rcond=None)
    assert float(np.hypot(*coef)) < 0.05


def test_deterministic_repeated_build() -> None:
    frames = _sine_track(120, 3.0, noise={0: (40.0, 0.2)})
    first = wf.build_filtered_world_track(frames)
    second = wf.build_filtered_world_track(frames)
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_sos_provenance_recorded() -> None:
    frames = _sine_track(60, 2.0)
    track = wf.build_filtered_world_track(frames)
    sos = np.asarray(track.sos_coefficients)
    assert sos.shape == (2, 6)  # order 4 -> two second-order sections
    assert track.method_version == "butterworth-sos-zerophase-v1"


# --- segments and gaps --------------------------------------------------------


def test_long_missing_span_splits_segments_without_leakage() -> None:
    frames: list[WorldFrameObservation] = []
    for i in range(140):
        t = i * DT
        level = 0.0 if i < 70 else 5.0
        if 55 <= i < 85:
            frames.append(_frame(t, {5: (level, 1.0, 0.0)}, missing=frozenset({5})))
        else:
            frames.append(_frame(t, {5: (level, 1.0, 0.0)}))
    track = wf.build_filtered_world_track(frames)
    segments = [seg for seg in track.segments if seg.joint_index == 5]
    assert len(segments) == 2
    assert all(seg.filtered for seg in segments)
    assert segments[0].end_index == 54
    assert segments[1].start_index == 85
    # No filtered value inside the long missing span.
    for row in range(55, 85):
        sample = track.samples[row]
        assert sample.observed[5] is False
        assert sample.available[5] is False
        assert sample.filtered[5] is None
        assert sample.filter_confidence[5] == 0.0
        assert sample.observation_quality[5] == 0.0
    # No step-response leakage across the split.
    tail = track.samples[50].filtered[5]
    head = track.samples[90].filtered[5]
    assert tail is not None and head is not None
    assert abs(tail.x - 0.0) < 0.3
    assert abs(head.x - 5.0) < 0.3


def test_long_pts_jump_splits_segments() -> None:
    frames = _sine_track(60, 2.0)
    second = _sine_track(60, 2.0, start=60 * DT + 0.5)
    track = wf.build_filtered_world_track(frames + second)
    segments = [seg for seg in track.segments if seg.joint_index == 0]
    assert len(segments) == 2
    assert segments[0].end_index == 59
    assert segments[1].start_index == 60
    assert all(seg.filtered for seg in segments)
    # Canonical grid reproduces every supplied PTS verbatim, in order.
    assert [s.time_seconds for s in track.samples] == [
        f.time_seconds for f in frames + second
    ]


def test_short_segment_stays_unfiltered_but_honest() -> None:
    before = None
    frames = _sine_track(10, 2.0)
    before = [frame.to_dict() for frame in frames]
    track = wf.build_filtered_world_track(frames)
    for sample in track.samples:
        assert sample.observed[0] is True
        assert sample.available[0] is False
        assert sample.filtered[0] is None
        assert sample.filter_confidence[0] == 0.0
        assert sample.observation_quality[0] == 1.0
    segments = [seg for seg in track.segments if seg.joint_index == 0]
    assert len(segments) == 1
    assert segments[0].filtered is False
    assert "too_short" in segments[0].reason
    # Source rows are never altered.
    assert [frame.to_dict() for frame in frames] == before


def test_short_gap_interpolated_with_reduced_quality() -> None:
    frames = _sine_track(80, 2.0)
    gapped = [
        _frame(f.time_seconds, {0: (f.world_landmarks[0].x, 1.0, 0.0)})
        if not 30 <= i < 33
        else _frame(f.time_seconds, {}, missing=frozenset({0}))
        for i, f in enumerate(frames)
    ]
    track = wf.build_filtered_world_track(gapped)
    for row in (30, 31, 32):
        sample = track.samples[row]
        assert sample.observed[0] is False
        assert sample.interpolated[0] is True
        assert sample.available[0] is True
        assert sample.filtered[0] is not None
        assert sample.filter_confidence[0] == pytest.approx(0.5)
        assert sample.observation_quality[0] == pytest.approx(0.5)
        assert sample.interpolation_span_seconds > 0.0
        assert sample.temporal_uncertainty_seconds == pytest.approx(
            sample.interpolation_span_seconds / 2.0
        )
    # Neighbors remain directly observed with full quality.
    assert track.samples[29].observed[0] is True
    assert track.samples[29].interpolated[0] is False
    assert track.samples[29].interpolation_span_seconds == 0.0
    assert track.samples[33].observation_quality[0] == pytest.approx(1.0)


def test_no_person_frames_are_unqualified_everywhere() -> None:
    frames = _sine_track(40, 2.0)
    frames[10] = _frame(10 * DT, person=False)
    frames[11] = _frame(11 * DT, person=False)
    track = wf.build_filtered_world_track(frames)
    for row in (10, 11):
        sample = track.samples[row]
        assert all(o is False for o in sample.observed)
        # A two-frame no-person run is a short qualified gap: interpolated
        # support with reduced quality, never claimed as observed.
        assert all(g is True for g in sample.interpolated)
        assert all(a is True for a in sample.available)
        assert all(v is not None for v in sample.filtered)
        assert all(q == pytest.approx(0.5) for q in sample.observation_quality)
        # 2D companion linkage is retained even for missing rows.
        assert sample.frame_2d.has_person is False


def test_long_no_person_span_stays_unavailable() -> None:
    frames = _sine_track(140, 2.0)
    for i in range(55, 85):
        frames[i] = _frame(i * DT, person=False)
    track = wf.build_filtered_world_track(frames)
    segments = [seg for seg in track.segments if seg.joint_index == 0]
    assert len(segments) == 2
    for row in range(55, 85):
        sample = track.samples[row]
        assert sample.interpolated[0] is False
        assert sample.available[0] is False
        assert sample.filtered[0] is None


def test_missing_joints_never_fabricated() -> None:
    frames = _sine_track(60, 2.0)
    for i, frame in enumerate(frames):
        joints = list(frame.world_landmarks)
        joints[7] = None
        frames[i] = WorldFrameObservation(
            time_seconds=frame.time_seconds,
            timestamp_ms=frame.timestamp_ms,
            world_landmarks=tuple(joints),
            frame_2d=frame.frame_2d,
        )
    track = wf.build_filtered_world_track(frames)
    assert not [seg for seg in track.segments if seg.joint_index == 7]
    for sample in track.samples:
        assert sample.observed[7] is False
        assert sample.available[7] is False
        assert sample.filtered[7] is None
    # Other joints still filter normally.
    assert all(sample.available[0] for sample in track.samples)


# --- grid, edges, independence -------------------------------------------------


def test_irregular_pts_grid_preserved_exactly() -> None:
    rng = np.random.RandomState(7)
    jitter = rng.uniform(-0.001, 0.001, size=100)
    times = np.cumsum(np.full(100, DT) + np.concatenate([[0.0], jitter[1:]]))
    times = np.maximum.accumulate(times)
    frames = [
        _frame(float(t), {0: (math.sin(2.0 * math.pi * 2.0 * float(t)), 1.0, 0.0)})
        for t in times
    ]
    track = wf.build_filtered_world_track(frames)
    assert [s.time_seconds for s in track.samples] == [float(t) for t in times]
    assert [s.timestamp_ms for s in track.samples] == [
        int(round(float(t) * 1000)) for t in times
    ]
    assert all(sample.available[0] for sample in track.samples)
    assert track.sample_rate_hz == pytest.approx(FS, rel=0.05)


def test_edge_samples_carry_reduced_confidence() -> None:
    frames = _sine_track(200, 2.0)
    track = wf.build_filtered_world_track(frames)
    padlen = wf.default_sos_padlen(np.asarray(track.sos_coefficients))
    first = track.samples[0]
    assert first.edge[0] is True
    assert first.filter_confidence[0] == pytest.approx(0.5)
    mid = track.samples[100]
    assert mid.edge[0] is False
    assert mid.filter_confidence[0] == pytest.approx(1.0)
    last = track.samples[-1]
    assert last.edge[0] is True
    assert last.filter_confidence[0] == pytest.approx(0.5)
    # Padding margin matches the SciPy default extension length.
    assert padlen == 15
    assert track.samples[padlen - 1].edge[0] is True
    assert track.samples[padlen].edge[0] is False


def test_axes_filtered_independently() -> None:
    frames: list[WorldFrameObservation] = []
    for i in range(120):
        t = i * DT
        frames.append(
            _frame(
                t,
                {
                    3: (
                        math.sin(2.0 * math.pi * 2.0 * t),
                        2.0,
                        0.25 * math.sin(2.0 * math.pi * 40.0 * t),
                    )
                },
            )
        )
    track = wf.build_filtered_world_track(frames)
    xs = _series(track, 3, 0)
    ys = _series(track, 3, 1)
    zs = _series(track, 3, 2)
    padlen = wf.default_sos_padlen(np.asarray(track.sos_coefficients))
    interior = slice(padlen, 120 - padlen)
    # Constant y passes untouched; noisy z is smoothed toward zero.
    assert np.max(np.abs(ys[interior] - 2.0)) < 1e-9
    assert float(np.max(np.abs(zs[interior]))) < 0.05
    assert float(np.max(np.abs(xs[interior]))) > 0.5


def test_companion_linkage_retained() -> None:
    frames = _sine_track(30, 2.0)
    track = wf.build_filtered_world_track(frames)
    for sample, frame in zip(track.samples, frames):
        assert sample.frame_2d == frame.frame_2d
        assert sample.time_seconds == frame.time_seconds
        assert sample.timestamp_ms == frame.timestamp_ms


# --- input validation -----------------------------------------------------------


def test_input_validation() -> None:
    with pytest.raises(wf.WorldFilterError, match="at least one frame"):
        wf.build_filtered_world_track([])
    with pytest.raises(wf.WorldFilterError, match="WorldFrameObservation"):
        wf.build_filtered_world_track(["nope"])  # type: ignore[list-item]
    frames = _sine_track(30, 2.0)
    with pytest.raises(wf.WorldFilterError, match="strictly increasing"):
        wf.build_filtered_world_track([frames[5], frames[2]])
    with pytest.raises(wf.WorldFilterError, match="WorldFilterConfig"):
        wf.build_filtered_world_track(frames, config="nope")  # type: ignore[arg-type]


def test_cutoff_above_nyquist_rejected() -> None:
    # 5 Hz sampling -> Nyquist 2.5 Hz < 12 Hz cutoff: honest failure.
    frames = _sine_track(30, 1.0, step=0.2)
    with pytest.raises(wf.WorldFilterError, match="Nyquist"):
        wf.build_filtered_world_track(frames)


def test_single_frame_track_is_honestly_unavailable() -> None:
    track = wf.build_filtered_world_track([_frame(0.0, {0: (1.0, 2.0, 3.0)})])
    sample = track.samples[0]
    assert sample.observed[0] is True
    assert sample.available[0] is False
    assert sample.filtered[0] is None
    assert track.sample_rate_hz == 0.0


# --- codecs ---------------------------------------------------------------------


def test_track_codec_round_trip() -> None:
    frames = _sine_track(40, 2.0)
    track = wf.build_filtered_world_track(frames)
    clone = wf.FilteredWorldTrack.from_dict(track.to_dict())
    assert clone == track
    assert wf.FilteredWorldTrack.from_json(track.to_json()) == track
    sample_clone = wf.FilteredWorldSample.from_dict(track.samples[5].to_dict())
    assert sample_clone == track.samples[5]
    segment_clone = wf.FilterSegment.from_dict(track.segments[0].to_dict())
    assert segment_clone == track.segments[0]


def test_track_codec_rejects_unknown_keys_and_newer_schema() -> None:
    frames = _sine_track(40, 2.0)
    track = wf.build_filtered_world_track(frames)
    payload = track.to_dict()
    tampered = dict(payload)
    tampered["bogus"] = True
    with pytest.raises(wf.WorldFilterError, match="unknown keys"):
        wf.FilteredWorldTrack.from_dict(tampered)
    newer = dict(payload)
    newer["schema_version"] = 999
    with pytest.raises(wf.WorldFilterError, match="newer"):
        wf.FilteredWorldTrack.from_dict(newer)
    sample_payload = dict(track.samples[0].to_dict())
    sample_payload["schema_version"] = 999
    with pytest.raises(wf.WorldFilterError, match="newer"):
        wf.FilteredWorldSample.from_dict(sample_payload)


def test_sample_codec_rejects_inconsistent_support() -> None:
    frames = _sine_track(40, 2.0)
    track = wf.build_filtered_world_track(frames)
    payload = track.samples[20].to_dict()
    # Available-but-null and valued-but-unavailable are both rejected.
    bad = dict(payload)
    bad_filtered = list(bad["filtered"])
    bad_filtered[0] = None
    bad["filtered"] = bad_filtered
    with pytest.raises(wf.WorldFilterError, match="available"):
        wf.FilteredWorldSample.from_dict(bad)
    bad2 = dict(payload)
    bad2_available = list(bad2["available"])
    bad2_available[1] = False
    bad2["available"] = bad2_available
    bad2_conf = list(bad2["filter_confidence"])
    bad2_conf[1] = 0.0
    bad2["filter_confidence"] = bad2_conf
    with pytest.raises(wf.WorldFilterError, match="unavailable"):
        wf.FilteredWorldSample.from_dict(bad2)


def test_design_helpers() -> None:
    assert wf.estimate_sample_rate_hz([0.0, 0.1, 0.2, 0.3]) == pytest.approx(10.0)
    with pytest.raises(wf.WorldFilterError):
        wf.estimate_sample_rate_hz([1.0])
    with pytest.raises(wf.WorldFilterError):
        wf.estimate_sample_rate_hz([0.0, 0.0])
    sos = wf.design_filter_sos(wf.WorldFilterConfig(), FS)
    assert sos.shape == (2, 6)
    assert wf.default_sos_padlen(sos) == 15
