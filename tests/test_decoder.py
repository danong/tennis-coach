"""Focused tests for the macro state decoder (synthetic only)."""

from __future__ import annotations

import pytest

from serve_review.detection.decoder import (
    DECODER_SCHEMA_VERSION,
    REASON_ABORTED,
    REASON_SHADOW,
    DecodeResult,
    DecoderConfig,
    DecoderError,
    ShadowRecord,
    decode_sequence,
    pair_frames,
)
from serve_review.detection.features import FeatureFrame
from serve_review.detection.ranges import CandidateRange
from serve_review.media.audio import AudioEnergy

STEP = 0.1
QUIET = 0.05
LOUD = 0.9


def _feat(
    moment: float,
    *,
    person: bool = True,
    visible: float = 1.0,
    torso: float | None = 0.0,
    overhead: float | None = 0.0,
    elbow_speed: float | None = 0.1,
    elbow_flex: float | None = 150.0,
    rest: float | None = 0.9,
) -> FeatureFrame:
    if not person:
        return FeatureFrame(
            time_seconds=moment, has_person=False, visible_fraction=0.0
        )
    if rest is None:
        return FeatureFrame(
            time_seconds=moment,
            has_person=True,
            visible_fraction=visible,
            torso_scale=1.0,
            player_scale=1.0,
            wrist_speed=0.1,
            elbow_speed=elbow_speed,
            body_motion=0.2,
            overhead_evidence=overhead,
            rest_evidence=None,
            motion_evidence=None,
            elbow_flexion_left=elbow_flex,
            elbow_flexion_right=elbow_flex,
            shoulder_tilt=0.0,
            torso_displacement=torso,
        )
    motion = 1.0 - float(rest)
    return FeatureFrame(
        time_seconds=moment,
        has_person=True,
        visible_fraction=visible,
        torso_scale=1.0,
        player_scale=1.0,
        wrist_speed=0.1,
        elbow_speed=elbow_speed,
        body_motion=0.2,
        overhead_evidence=overhead,
        rest_evidence=float(rest),
        motion_evidence=motion,
        elbow_flexion_left=elbow_flex,
        elbow_flexion_right=elbow_flex,
        shoulder_tilt=0.0,
        torso_displacement=torso,
    )


def _rest_run(start: float, count: int) -> list[FeatureFrame]:
    return [_feat(start + i * STEP) for i in range(count)]


def _audio(times: list[float], energy: float) -> list[AudioEnergy]:
    return [AudioEnergy(time_seconds=t, energy=energy) for t in times]


def _valid_serve(start: float = 1.0) -> tuple[list[FeatureFrame], list[AudioEnergy]]:
    """Build one canonical valid serve: prep at start, accel +0.3, exit +0.8."""
    frames: list[FeatureFrame] = []
    frames += _rest_run(0.0, 5)  # 0.0..0.4 idle
    # Preparation: torso activity, low rest, not overhead, slow elbow.
    for i in range(3):  # 1.0,1.1,1.2 with start=1.0
        frames.append(
            _feat(
                start + i * STEP,
                torso=0.5,
                overhead=0.0,
                elbow_speed=0.4,
                elbow_flex=120.0,
                rest=0.15,
            )
        )
    # Acceleration into overhead quadrant.
    accel = start + 0.3
    frames.append(
        _feat(
            accel,
            torso=0.5,
            overhead=1.0,
            elbow_speed=5.0,
            elbow_flex=170.0,
            rest=0.1,
        )
    )
    # Follow-through overhead then drop.
    frames.append(
        _feat(
            accel + 0.1,
            torso=0.3,
            overhead=1.0,
            elbow_speed=2.0,
            elbow_flex=160.0,
            rest=0.2,
        )
    )
    frames.append(
        _feat(
            accel + 0.2,
            torso=0.1,
            overhead=1.0,
            elbow_speed=1.0,
            elbow_flex=150.0,
            rest=0.4,
        )
    )
    exit_t = start + 0.8
    frames.append(
        _feat(
            exit_t,
            torso=0.0,
            overhead=0.0,
            elbow_speed=0.1,
            elbow_flex=150.0,
            rest=0.9,
        )
    )
    frames += _rest_run(exit_t + STEP, 5)
    frames.sort(key=lambda f: f.time_seconds)
    # Audio: quiet everywhere except a transient near acceleration.
    audio = [AudioEnergy(time_seconds=f.time_seconds, energy=QUIET) for f in frames]
    audio.append(AudioEnergy(time_seconds=accel + 0.05, energy=LOUD))
    audio.sort(key=lambda a: a.time_seconds)
    # Deduplicate the transient time if it collides with a frame time.
    seen: dict[float, float] = {}
    for sample in audio:
        prev = seen.get(sample.time_seconds)
        if prev is None or sample.energy > prev:
            seen[sample.time_seconds] = sample.energy
    audio = [AudioEnergy(time_seconds=t, energy=e) for t, e in sorted(seen.items())]
    return frames, audio


def test_schema_version_pinned() -> None:
    assert DECODER_SCHEMA_VERSION == 2
    assert DecoderConfig().schema_version == 2
    assert DecoderConfig().audio_transient_ratio == pytest.approx(5.0)
    assert DecoderConfig().audio_transient_floor == pytest.approx(0.01)


def test_empty_input_yields_empty_output() -> None:
    result = decode_sequence([], [])
    assert result.ranges == ()
    assert result.shadows == ()
    assert decode_sequence([], _audio([0.1], QUIET)).ranges == ()


def test_strictly_increasing_times_required() -> None:
    frames, audio = _valid_serve()
    with pytest.raises(DecoderError):
        decode_sequence(list(reversed(frames)), audio)
    with pytest.raises(DecoderError):
        decode_sequence(frames, list(reversed(audio)))
    with pytest.raises(DecoderError):
        decode_sequence([_feat(0.5), _feat(0.5)], [])
    with pytest.raises(DecoderError):
        pair_frames(["nope"], [])  # type: ignore[list-item]
    with pytest.raises(DecoderError):
        decode_sequence(frames, ["nope"])  # type: ignore[list-item]
    with pytest.raises(DecoderError):
        decode_sequence(frames, audio, config="nope")  # type: ignore[arg-type]


def test_full_valid_serve_traversal() -> None:
    frames, audio = _valid_serve(start=1.0)
    result = decode_sequence(frames, audio)
    assert len(result.ranges) == 1
    assert len(result.shadows) == 0
    item = result.ranges[0]
    assert isinstance(item, CandidateRange)
    assert item.start_seconds == pytest.approx(1.0)
    assert item.end_seconds == pytest.approx(1.8)
    # Planner compatibility: CandidateRange round trip.
    assert CandidateRange.from_dict(item.to_dict()) == item


def test_boundary_timestamps_equal_transition_times() -> None:
    frames, audio = _valid_serve(start=2.0)
    result = decode_sequence(frames, audio)
    assert len(result.ranges) == 1
    frame_times = {f.time_seconds for f in frames}
    assert result.ranges[0].start_seconds in frame_times
    assert result.ranges[0].end_seconds in frame_times
    assert result.ranges[0].start_seconds == 2.0
    assert result.ranges[0].end_seconds == 2.8


def test_silent_overhead_routes_to_shadow_no_export() -> None:
    frames, _ = _valid_serve(start=1.0)
    quiet = [AudioEnergy(time_seconds=f.time_seconds, energy=QUIET) for f in frames]
    result = decode_sequence(frames, quiet)
    assert result.ranges == ()
    assert len(result.shadows) == 1
    assert result.shadows[0].reason == REASON_SHADOW


def test_missing_audio_never_fabricates() -> None:
    frames, _ = _valid_serve(start=1.0)
    result = decode_sequence(frames, [])
    assert result.ranges == ()
    assert len(result.shadows) == 1
    assert result.shadows[0].reason == REASON_SHADOW


def test_toss_without_acceleration_aborts() -> None:
    frames: list[FeatureFrame] = []
    frames += _rest_run(0.0, 5)
    for i in range(4):  # preparation-like lift, never accelerates overhead
        frames.append(
            _feat(
                0.5 + i * STEP,
                torso=0.5,
                overhead=0.0,
                elbow_speed=0.4,
                elbow_flex=120.0,
                rest=0.2,
            )
        )
    frames += _rest_run(0.9, 6)
    audio = [AudioEnergy(time_seconds=f.time_seconds, energy=QUIET) for f in frames]
    result = decode_sequence(frames, audio)
    assert result.ranges == ()
    assert len(result.shadows) == 1
    assert result.shadows[0].reason == REASON_ABORTED


def test_audio_outside_window_rejects() -> None:
    frames, _ = _valid_serve(start=1.0)
    # Transient 0.6s after acceleration: outside the +-0.4s window.
    audio = [AudioEnergy(time_seconds=f.time_seconds, energy=QUIET) for f in frames]
    audio.append(AudioEnergy(time_seconds=1.3 + 0.6, energy=LOUD))
    audio.sort(key=lambda a: a.time_seconds)
    result = decode_sequence(frames, audio)
    assert result.ranges == ()
    assert len(result.shadows) == 1
    assert result.shadows[0].reason == REASON_SHADOW


def test_audio_window_is_versioned() -> None:
    frames, _ = _valid_serve(start=1.0)
    audio = [AudioEnergy(time_seconds=f.time_seconds, energy=QUIET) for f in frames]
    audio.append(AudioEnergy(time_seconds=1.3 + 0.6, energy=LOUD))
    audio.sort(key=lambda a: a.time_seconds)
    wide = DecoderConfig(audio_window_seconds=0.7)
    result = decode_sequence(frames, audio, wide)
    assert len(result.ranges) == 1


def test_dropout_bridging_short_dip() -> None:
    frames, audio = _valid_serve(start=1.0)
    # Two-frame visibility dip inside preparation/acceleration.
    bridged = [
        _feat(1.1, person=False) if f.time_seconds == pytest.approx(1.1) else f
        for f in frames
    ]
    bridged = [
        _feat(f.time_seconds, visible=0.1, torso=0.5, overhead=0.0,
              elbow_speed=0.4, elbow_flex=120.0, rest=0.15)
        if f.time_seconds == pytest.approx(1.2) else f
        for f in bridged
    ]
    result = decode_sequence(bridged, audio)
    assert len(result.ranges) == 1
    assert result.ranges[0].start_seconds == pytest.approx(1.0)


def test_long_dropout_collapses_to_shadow_or_abort() -> None:
    frames, audio = _valid_serve(start=1.0)
    gap = [_feat(1.05 + i * 0.05, person=False) for i in range(6)]
    # Replace the preparation window with a 6-frame no-person gap.
    kept = [f for f in frames if not (1.0 <= f.time_seconds <= 1.3)]
    merged = sorted(kept + gap, key=lambda f: f.time_seconds)
    result = decode_sequence(merged, audio)
    # A 6-frame gap exceeds the 4-frame hysteresis: no valid serve.
    assert result.ranges == ()


def test_back_to_back_serves() -> None:
    first_frames, _ = _valid_serve(start=1.0)
    second_frames, _ = _valid_serve(start=4.0)
    # Keep idle gap between serves; drop the leading idle of the second.
    frames = [f for f in first_frames if f.time_seconds < 3.0]
    frames += [f for f in second_frames if f.time_seconds >= 4.0 - 0.5]
    frames.sort(key=lambda f: f.time_seconds)
    audio = [AudioEnergy(time_seconds=f.time_seconds, energy=QUIET) for f in frames]
    audio.append(AudioEnergy(time_seconds=1.35, energy=LOUD))
    audio.append(AudioEnergy(time_seconds=4.35, energy=LOUD))
    audio.sort(key=lambda a: a.time_seconds)
    result = decode_sequence(frames, audio)
    assert len(result.ranges) == 2
    assert result.ranges[0].end_seconds < result.ranges[1].start_seconds


def test_missing_data_honesty() -> None:
    frames = [
        _feat(0.0, person=False),
        _feat(0.1, rest=None, overhead=None, torso=None,
              elbow_speed=None, elbow_flex=None),
        _feat(0.2, person=False),
    ]
    audio: list[AudioEnergy] = []
    result = decode_sequence(frames, audio)
    assert result.ranges == ()
    assert result.shadows == ()
    # Unknown-geometry frames never open preparation on their own.
    pairs = pair_frames(frames, audio)
    assert all(p.audio_energy is None for p in pairs)


def test_wrist_speed_never_triggers_preparation() -> None:
    # High wrist speed with no torso displacement and no overhead must
    # stay idle: preparation uses proximal geometry, not wrist speed.
    frames = [
        _feat(0.0 + i * STEP, torso=0.0, overhead=0.0, elbow_speed=0.1,
              elbow_flex=150.0, rest=0.9) for i in range(10)
    ]
    for f in frames:
        assert f.wrist_speed == pytest.approx(0.1)
    sporty = [
        FeatureFrame(
            time_seconds=f.time_seconds,
            has_person=True,
            visible_fraction=1.0,
            torso_scale=1.0,
            player_scale=1.0,
            wrist_speed=25.0,
            elbow_speed=0.1,
            body_motion=0.2,
            overhead_evidence=0.0,
            rest_evidence=0.9,
            motion_evidence=0.1,
            elbow_flexion_left=150.0,
            elbow_flexion_right=150.0,
            shoulder_tilt=0.0,
            torso_displacement=0.0,
        )
        for f in frames
    ]
    result = decode_sequence(sporty, [])
    assert result.ranges == ()
    assert result.shadows == ()


def test_pairing_missing_audio_is_unknown() -> None:
    frames = _rest_run(0.0, 3)
    audio = [AudioEnergy(time_seconds=0.0, energy=QUIET)]
    pairs = pair_frames(frames, audio)
    assert pairs[0].audio_energy == pytest.approx(QUIET)
    assert pairs[1].audio_energy is None
    assert pairs[2].audio_energy is None


def test_determinism() -> None:
    frames, audio = _valid_serve(start=1.0)
    first = decode_sequence(frames, audio)
    second = decode_sequence(frames, audio)
    assert first == second
    assert [r.to_dict() for r in first.ranges] == [r.to_dict() for r in second.ranges]


def test_config_codec_round_trip() -> None:
    config = DecoderConfig()
    assert DecoderConfig.from_dict(config.to_dict()) == config
    assert DecoderConfig.from_json(config.to_json()) == config
    with pytest.raises(DecoderError):
        DecoderConfig(dropout_hysteresis_frames=2)
    with pytest.raises(DecoderError):
        DecoderConfig(dropout_hysteresis_frames=6)
    with pytest.raises(DecoderError):
        DecoderConfig(audio_window_seconds=0.0)
    with pytest.raises(DecoderError):
        DecoderConfig(audio_transient_ratio=0.0)
    with pytest.raises(DecoderError):
        DecoderConfig(audio_transient_ratio=-1.0)
    with pytest.raises(DecoderError):
        DecoderConfig(audio_transient_floor=-0.001)
    with pytest.raises(DecoderError):
        DecoderConfig.from_dict({**config.to_dict(), "extra": 1})
    with pytest.raises(DecoderError):
        DecoderConfig.from_dict({"torso_displacement_threshold": 0.2})
    with pytest.raises(DecoderError):
        DecoderConfig.from_dict(
            {**config.to_dict(), "audio_transient_threshold": 0.3}
        )
    with pytest.raises(DecoderError):
        DecoderConfig(schema_version=999)
    with pytest.raises(DecoderError):
        DecoderConfig(schema_version=1)


def test_shadow_and_result_codec_round_trip() -> None:
    shadow = ShadowRecord(start_seconds=1.0, end_seconds=1.8, reason=REASON_SHADOW)
    assert ShadowRecord.from_dict(shadow.to_dict()) == shadow
    assert ShadowRecord.from_json(shadow.to_json()) == shadow
    with pytest.raises(DecoderError):
        ShadowRecord(start_seconds=1.0, end_seconds=1.0, reason=REASON_SHADOW)
    with pytest.raises(DecoderError):
        ShadowRecord(start_seconds=1.0, end_seconds=1.8, reason="nope")  # type: ignore[arg-type]
    frames, audio = _valid_serve(start=1.0)
    result = decode_sequence(frames, audio)
    assert DecodeResult.from_dict(result.to_dict()) == result
    assert DecodeResult.from_json(result.to_json()) == result
    empty = DecodeResult(ranges=(), shadows=())
    assert DecodeResult.from_dict(empty.to_dict()) == empty


# --- Real-scale phone-audio transients (measured single-serve anchor) ---

# Physical phone-audio scale: a 20 ms impact spike peaks near 0.053
# over a ~0.009 bed. The version-1 absolute 0.3 threshold sat ~6x above
# the real peak and forced every such serve into the shadow log.
REAL_PEAK = 0.053
REAL_BED = 0.009


def _real_scale_serve(
    start: float = 3.0, *, peak: float = REAL_PEAK, bed: float = REAL_BED
) -> tuple[list[FeatureFrame], list[AudioEnergy]]:
    """Serve-shaped pose track with phone-scale audio bed plus one spike."""
    frames: list[FeatureFrame] = []
    frames += [_feat(0.0 + i * STEP) for i in range(5)]
    for i in range(3):
        frames.append(
            _feat(
                start + i * STEP,
                torso=0.5,
                overhead=0.0,
                elbow_speed=0.4,
                elbow_flex=120.0,
                rest=0.15,
            )
        )
    accel = start + 0.3
    frames.append(
        _feat(
            accel,
            torso=0.5,
            overhead=1.0,
            elbow_speed=5.0,
            elbow_flex=170.0,
            rest=0.1,
        )
    )
    frames.append(
        _feat(
            accel + 0.1,
            torso=0.3,
            overhead=1.0,
            elbow_speed=2.0,
            elbow_flex=160.0,
            rest=0.2,
        )
    )
    frames.append(
        _feat(
            accel + 0.2,
            torso=0.1,
            overhead=1.0,
            elbow_speed=1.0,
            elbow_flex=150.0,
            rest=0.4,
        )
    )
    exit_t = start + 0.8
    frames.append(
        _feat(
            exit_t,
            torso=0.0,
            overhead=0.0,
            elbow_speed=0.1,
            elbow_flex=150.0,
            rest=0.9,
        )
    )
    frames += [_feat(exit_t + STEP + i * STEP) for i in range(5)]
    frames.sort(key=lambda f: f.time_seconds)
    audio = [AudioEnergy(time_seconds=f.time_seconds, energy=bed) for f in frames]
    audio.append(AudioEnergy(time_seconds=accel + 0.05, energy=peak))
    audio.sort(key=lambda a: a.time_seconds)
    return frames, audio


def test_real_scale_peak_validates_single_serve() -> None:
    frames, audio = _real_scale_serve(start=3.0)
    result = decode_sequence(frames, audio)
    assert len(result.ranges) == 1
    assert result.shadows == ()
    assert result.ranges[0].start_seconds == pytest.approx(3.0)
    assert result.ranges[0].end_seconds == pytest.approx(3.8)


def test_straddled_skirt_level_peak_rejects_to_shadow() -> None:
    # Coarse pose-grid sampling that straddles the spike reads only the
    # ~0.009 skirts: no sample rises above the bed, so the hypothesis
    # must stay a shadow instead of a valid serve.
    frames, _ = _real_scale_serve(start=3.0)
    skirts = [AudioEnergy(time_seconds=f.time_seconds, energy=REAL_BED) for f in frames]
    result = decode_sequence(frames, skirts)
    assert result.ranges == ()
    assert len(result.shadows) == 1
    assert result.shadows[0].reason == REASON_SHADOW


def test_absolute_floor_rejects_near_silence() -> None:
    # A large ratio against a near-zero baseline must not validate:
    # the 0.004 blip clears ratio 5 over the 0.0005 bed but not the
    # absolute floor.
    frames, _ = _real_scale_serve(start=1.0)
    near_silent = [
        AudioEnergy(time_seconds=f.time_seconds, energy=0.0005) for f in frames
    ]
    near_silent.append(AudioEnergy(time_seconds=1.35, energy=0.004))
    near_silent.sort(key=lambda a: a.time_seconds)
    result = decode_sequence(frames, near_silent)
    assert result.ranges == ()
    assert len(result.shadows) == 1


def _long_drive_serve(
    *, drive_end: float = 2.5, peak_time: float | None = None
) -> tuple[list[FeatureFrame], list[AudioEnergy]]:
    """Serve with a long overhead drive (span >> +-0.4 s window)."""
    start = 1.0
    frames: list[FeatureFrame] = []
    frames += _rest_run(0.0, 5)  # 0.0..0.4 idle
    for i in range(3):  # preparation 1.0,1.1,1.2
        frames.append(
            _feat(
                start + i * STEP,
                torso=0.5,
                overhead=0.0,
                elbow_speed=0.4,
                elbow_flex=120.0,
                rest=0.15,
            )
        )
    moment = start + 0.3  # first acceleration trigger at 1.3
    while moment <= drive_end + 1e-9:
        frames.append(
            _feat(
                moment,
                torso=0.5,
                overhead=1.0,
                elbow_speed=5.0,
                elbow_flex=170.0,
                rest=0.1,
            )
        )
        moment += STEP
    exit_t = drive_end + 0.5
    frames.append(
        _feat(
            exit_t,
            torso=0.0,
            overhead=0.0,
            elbow_speed=0.1,
            elbow_flex=150.0,
            rest=0.9,
        )
    )
    frames += _rest_run(exit_t + STEP, 5)
    frames.sort(key=lambda f: f.time_seconds)
    audio = [AudioEnergy(time_seconds=f.time_seconds, energy=QUIET) for f in frames]
    if peak_time is not None:
        audio.append(AudioEnergy(time_seconds=peak_time, energy=LOUD))
        audio.sort(key=lambda a: a.time_seconds)
    return frames, audio


def test_long_drive_late_contact_validates() -> None:
    # Acceleration span 1.3..2.5 (1.2 s >> 0.4 s window); a transient at
    # the drive end is ~1.2 s after onset, so a first-trigger anchor
    # would shadow it. The latest-trigger anchor follows the drive.
    frames, audio = _long_drive_serve(drive_end=2.5, peak_time=2.55)
    result = decode_sequence(frames, audio)
    assert len(result.ranges) == 1
    assert result.shadows == ()
    assert result.ranges[0].start_seconds == pytest.approx(1.0)
    assert result.ranges[0].end_seconds == pytest.approx(3.0)


def test_long_drive_transient_far_after_end_shadows() -> None:
    # Same long drive, but the transient sits 0.7 s past the last
    # acceleration trigger: outside the +-0.4 s window even from the
    # latest anchor, so the hypothesis must shadow.
    frames, audio = _long_drive_serve(drive_end=2.5, peak_time=3.25)
    result = decode_sequence(frames, audio)
    assert result.ranges == ()
    assert len(result.shadows) == 1
    assert result.shadows[0].reason == REASON_SHADOW


def test_transient_ratio_is_versioned() -> None:
    frames, audio = _real_scale_serve(start=3.0)
    # 0.053 over the 0.009 bed is ratio ~5.9: the default 5.0 passes
    # while a strict 20.0 rejects to shadow.
    assert len(decode_sequence(frames, audio).ranges) == 1
    strict = DecoderConfig(audio_transient_ratio=20.0)
    rejected = decode_sequence(frames, audio, strict)
    assert rejected.ranges == ()
    assert len(rejected.shadows) == 1
    assert strict.to_dict()["audio_transient_ratio"] == pytest.approx(20.0)
