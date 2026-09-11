"""Tests for the video-aligned audio transient feature sampler.

All media is generated synthetically in temporary directories with
``tests/media_factory.py`` (sine beds plus ``aevalsrc`` impulse clicks at
known times, at 44.1/48 kHz, mono/stereo); no private footage, inference,
CLI, caches, or network access are used. Real FFmpeg/ffprobe run the
integration paths; pure unit paths cover RMS math, command shape,
schedule/window validation, and error/cancellation contracts.
"""

from __future__ import annotations

import math
import shutil
import statistics
import subprocess
from pathlib import Path

import pytest

from serve_review.media.audio import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    DEFAULT_WINDOW_SECONDS,
    MAX_BUFFERED_SAMPLES,
    AudioCancelled,
    AudioEnergy,
    AudioError,
    audio_filter_graph,
    build_audio_decode_args,
    iter_audio_energy,
    probe_audio_stream,
    rms_energy,
    validate_audio_schedule,
    validate_window_seconds,
)
from media_factory import (
    FixtureError,
    FixtureSpec,
    audio_input_spec,
    build_av_fixture_args,
    ffmpeg_available,
    generate_av_fixture,
    generate_fixture,
    landscape_spec,
)

NEEDS_TOOLS = pytest.mark.skipif(
    not ffmpeg_available() or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for audio sampler integration tests",
)

DURATION = 1.0
RATE_30HZ = tuple(k / 30.0 for k in range(30))
CLICK_TIME_441 = 0.5
CLICK_TIME_480 = 0.7
CLICK_TOLERANCE = 0.05

# In-band gated burst modeling an impact transient (2 kHz energy the
# 1.0-3.5 kHz bandpass keeps) over a low out-of-band hum the bandpass
# rejects, so the click stands far above the bed floor.
BED_220 = "0.2*sin(2*PI*220*t)"
CLICK_441_EXPR = (
    f"{BED_220}+if(lt(abs(t-{CLICK_TIME_441}),0.004),sin(2*PI*2000*t),0)"
)
CLICK_480_MONO = (
    f"{BED_220}+if(lt(abs(t-{CLICK_TIME_480}),0.004),sin(2*PI*2000*t),0)"
)


def _tiny_spec() -> FixtureSpec:
    return FixtureSpec(
        width=64, height=48, duration_seconds=DURATION, fps=30
    )


def _sample(video: Path, schedule=RATE_30HZ, **kwargs):
    return list(iter_audio_energy(video, schedule, **kwargs))


def _median(values: list[float]) -> float:
    return statistics.median(values)


# --- Pure RMS behavior (no media required) ---


def test_rms_energy_known_values() -> None:
    assert rms_energy([]) == 0.0
    assert rms_energy([0.0, 0.0, 0.0]) == 0.0
    assert rms_energy([1.0, 1.0]) == pytest.approx(1.0)
    assert rms_energy([-1.0, 1.0]) == pytest.approx(1.0)
    assert rms_energy([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))
    assert rms_energy((0.5, -0.5, 0.5, -0.5)) == pytest.approx(0.5)
    first = rms_energy([0.1, 0.4, 0.9])
    assert rms_energy([0.1, 0.4, 0.9]) == first  # deterministic


def test_rms_energy_rejects_invalid() -> None:
    for bad in ("audio", b"\x00", None, 5, True):
        with pytest.raises(AudioError, match="[Ss]amples"):
            rms_energy(bad)  # type: ignore[arg-type]
    for bad_samples in (
        [0.5, float("nan")],
        [float("inf")],
        ["x"],
        [None],
        [True],
    ):
        with pytest.raises(AudioError, match="[Ss]ample"):
            rms_energy(bad_samples)  # type: ignore[list-item]


def test_audio_energy_validates_fields() -> None:
    assert AudioEnergy(time_seconds=0.5, energy=0.1).energy == pytest.approx(0.1)
    for bad_time in (-0.1, float("nan"), float("inf"), "0.5", None, True):
        with pytest.raises(AudioError, match="time_seconds"):
            AudioEnergy(time_seconds=bad_time, energy=0.1)  # type: ignore[arg-type]
    for bad_energy in (-0.1, float("nan"), float("inf"), "0.1", None, True):
        with pytest.raises(AudioError, match="energy"):
            AudioEnergy(time_seconds=0.5, energy=bad_energy)  # type: ignore[arg-type]


def test_filter_graph_pins_bandpass() -> None:
    assert BANDPASS_LOW_HZ == pytest.approx(1000.0)
    assert BANDPASS_HIGH_HZ == pytest.approx(3500.0)
    graph = audio_filter_graph()
    assert graph == "highpass=f=1000,lowpass=f=3500"
    assert audio_filter_graph() == graph  # deterministic


def test_decode_args_shape_demuxes_raw_audio() -> None:
    args = build_audio_decode_args("/tmp/src.mov")
    assert isinstance(args, list)
    assert all(isinstance(part, str) for part in args)
    assert args[0] == "ffmpeg"
    assert args[args.index("-map") + 1] == "0:a:0"
    assert "0:v" not in args
    assert "-ss" not in args  # no seeking: raw demux from the timeline origin
    assert "atempo" not in " ".join(args)  # no slow-mo processing
    assert "asetrate" not in " ".join(args)
    assert args[args.index("-af") + 1] == audio_filter_graph()
    assert args[args.index("-ac") + 1] == "1"
    assert args[args.index("-f") + 1] == "f32le"
    assert "pcm_f32le" in args
    assert args[-1] == "-"
    assert build_audio_decode_args("/tmp/src.mov") == args  # deterministic


def test_decode_args_rejects_bad_inputs() -> None:
    with pytest.raises(AudioError, match="ffmpeg"):
        build_audio_decode_args("/tmp/src.mov", ffmpeg="")
    with pytest.raises(AudioError, match="[Vv]ideo"):
        build_audio_decode_args("")


def test_schedule_validation() -> None:
    assert validate_audio_schedule([0.0, 0.5], 1.0) == (0.0, 0.5)
    assert validate_audio_schedule((0.1, 0.25, 0.9), 1.0) == (0.1, 0.25, 0.9)
    for bad in ([], (), 0.5, "0.5", [0.5, 0.5], [0.6, 0.5], [-0.1], [1.0],
                [1.5], [0.1, float("nan")], [True], [None], ["x"]):
        with pytest.raises(AudioError, match="[Ss]chedule"):
            validate_audio_schedule(bad, 1.0)  # type: ignore[arg-type]
    for bad_duration in (0, -1.0, float("nan"), float("inf"), "1", None, True):
        with pytest.raises(AudioError, match="duration"):
            validate_audio_schedule([0.1], bad_duration)  # type: ignore[arg-type]


def test_window_validation() -> None:
    assert validate_window_seconds(DEFAULT_WINDOW_SECONDS) == pytest.approx(0.010)
    assert validate_window_seconds(0.005) == pytest.approx(0.005)
    for bad in (0, -0.01, 0.0005, 1.0, float("nan"), float("inf"),
                "0.01", None, True):
        with pytest.raises(AudioError, match="window_seconds"):
            validate_window_seconds(bad)  # type: ignore[arg-type]


def test_memory_bound_is_streaming_scale() -> None:
    # One pipe chunk plus per-timestamp scalars: kilobytes, never the track.
    assert MAX_BUFFERED_SAMPLES * 4 <= 256 * 1024


def test_missing_tool_raises_actionable(tmp_path: Path) -> None:
    video = tmp_path / "src.mov"
    video.write_bytes(b"fake-bytes")
    with pytest.raises(AudioError, match="not found.*install FFmpeg"):
        iter_audio_energy(video, [0.1], ffmpeg="no-such-ffmpeg-xyz")
    with pytest.raises(AudioError, match="not found"):
        iter_audio_energy(video, [0.1], ffprobe="no-such-ffprobe-xyz")


def test_invalid_hook_and_missing_file(tmp_path: Path) -> None:
    video = tmp_path / "src.mov"
    video.write_bytes(b"fake-bytes")
    with pytest.raises(AudioError, match="is_cancelled"):
        iter_audio_energy(video, [0.1], is_cancelled="yes")  # type: ignore[arg-type]
    with pytest.raises(AudioError, match="window_seconds"):
        iter_audio_energy(video, [0.1], window_seconds=5.0)
    with pytest.raises(AudioError, match="does not exist"):
        iter_audio_energy(tmp_path / "missing.mov", [0.1])
    with pytest.raises(AudioError, match="does not exist"):
        probe_audio_stream(tmp_path / "missing.mov")


def test_av_fixture_args_are_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "av.mov"
    spec = _tiny_spec()
    first = build_av_fixture_args(output, spec, audio_expr="0")
    assert build_av_fixture_args(output, spec, audio_expr="0") == first
    assert all(isinstance(part, str) for part in first)
    assert first[0] == "ffmpeg"
    assert "aevalsrc" in " ".join(first)
    assert "testsrc2" in " ".join(first)
    stereo = build_av_fixture_args(
        output, spec, audio_expr="0|0", sample_rate_hz=48000, channels=2
    )
    assert stereo != first
    with pytest.raises(FixtureError, match="[Ss]ample rate"):
        build_av_fixture_args(output, spec, audio_expr="0", sample_rate_hz=22050)
    with pytest.raises(FixtureError, match="[Cc]hannels"):
        build_av_fixture_args(output, spec, audio_expr="0", channels=3)
    with pytest.raises(FixtureError, match="[Ee]xpression"):
        build_av_fixture_args(output, spec, audio_expr="  ")
    with pytest.raises(FixtureError, match="[Ee]xpression"):
        build_av_fixture_args(output, spec, audio_expr="it's")
    assert "aevalsrc='0'" in audio_input_spec("0")


def test_generate_uses_argument_array_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import media_factory as factory

    captured: dict = {}

    def _fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        Path(args[-1]).write_bytes(b"fake-av-fixture")
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(factory.subprocess, "run", _fake_run)
    output = tmp_path / "nested" / "av.mov"
    assert generate_av_fixture(output, _tiny_spec(), audio_expr="0") == output
    assert output.is_file()
    assert isinstance(captured["args"], list)
    assert "shell" not in captured["kwargs"]


# --- Integration: real FFmpeg/ffprobe in temporary directories ---


@NEEDS_TOOLS
def test_probe_reports_audio_track(tmp_path: Path) -> None:
    mono = generate_av_fixture(
        tmp_path / "mono.mov", _tiny_spec(), audio_expr=CLICK_441_EXPR
    )
    info = probe_audio_stream(mono)
    assert info.sample_rate_hz == 44100
    assert info.channels == 1
    assert info.duration_seconds == pytest.approx(DURATION, abs=0.08)
    stereo = generate_av_fixture(
        tmp_path / "stereo.mov",
        _tiny_spec(),
        audio_expr=f"{CLICK_480_MONO}|{CLICK_480_MONO}",
        sample_rate_hz=48000,
        channels=2,
    )
    stereo_info = probe_audio_stream(stereo)
    assert stereo_info.sample_rate_hz == 48000
    assert stereo_info.channels == 2


@NEEDS_TOOLS
def test_click_detection_mono_44100(tmp_path: Path) -> None:
    video = generate_av_fixture(
        tmp_path / "click441.mov", _tiny_spec(), audio_expr=CLICK_441_EXPR
    )
    out = _sample(video)
    assert len(out) == len(RATE_30HZ)
    assert all(isinstance(item, AudioEnergy) for item in out)
    peak = max(out, key=lambda item: item.energy)
    assert peak.time_seconds == pytest.approx(CLICK_TIME_441, abs=CLICK_TOLERANCE)
    floor = _median([item.energy for item in out])
    assert peak.energy > 10 * floor


@NEEDS_TOOLS
def test_click_detection_stereo_48000(tmp_path: Path) -> None:
    video = generate_av_fixture(
        tmp_path / "click48.mov",
        _tiny_spec(),
        audio_expr=f"{CLICK_480_MONO}|{CLICK_480_MONO}",
        sample_rate_hz=48000,
        channels=2,
    )
    out = _sample(video)
    assert len(out) == len(RATE_30HZ)
    peak = max(out, key=lambda item: item.energy)
    assert peak.time_seconds == pytest.approx(CLICK_TIME_480, abs=CLICK_TOLERANCE)
    floor = _median([item.energy for item in out])
    assert peak.energy > 10 * floor


@NEEDS_TOOLS
def test_silence_yields_near_zero_energy(tmp_path: Path) -> None:
    video = generate_av_fixture(tmp_path / "silent.mov", _tiny_spec(), audio_expr="0")
    out = _sample(video)
    assert len(out) == len(RATE_30HZ)
    assert max(item.energy for item in out) < 0.002


@NEEDS_TOOLS
def test_bandpass_rejects_out_of_band_tones(tmp_path: Path) -> None:
    spec = _tiny_spec()
    low = generate_av_fixture(
        tmp_path / "tone200.mov", spec, audio_expr="0.5*sin(2*PI*200*t)"
    )
    mid = generate_av_fixture(
        tmp_path / "tone2000.mov", spec, audio_expr="0.5*sin(2*PI*2000*t)"
    )
    high = generate_av_fixture(
        tmp_path / "tone8000.mov",
        spec,
        audio_expr="0.5*sin(2*PI*8000*t)|0.5*sin(2*PI*8000*t)",
        sample_rate_hz=48000,
        channels=2,
    )
    mean_low = statistics.mean(item.energy for item in _sample(low))
    mean_mid = statistics.mean(item.energy for item in _sample(mid))
    mean_high = statistics.mean(item.energy for item in _sample(high))
    assert mean_mid / mean_low > 10  # 200 Hz hum rejected by the highpass
    assert mean_mid / mean_high > 3  # 8 kHz hiss rejected by the lowpass


@NEEDS_TOOLS
def test_alignment_matches_caller_timestamps(tmp_path: Path) -> None:
    video = generate_av_fixture(
        tmp_path / "align.mov", _tiny_spec(), audio_expr=CLICK_441_EXPR
    )
    out = _sample(video)
    assert [item.time_seconds for item in out] == list(RATE_30HZ)
    assert all(math.isfinite(item.energy) and item.energy >= 0 for item in out)
    sparse = (0.1, 0.25, 0.9)
    sparse_out = _sample(video, sparse)
    assert [item.time_seconds for item in sparse_out] == list(sparse)
    dense = _sample(video, RATE_30HZ, window_seconds=0.005)
    assert [item.time_seconds for item in dense] == list(RATE_30HZ)
    assert max(dense, key=lambda item: item.energy).time_seconds == pytest.approx(
        CLICK_TIME_441, abs=CLICK_TOLERANCE
    )


@NEEDS_TOOLS
def test_missing_audio_raises_actionable(tmp_path: Path) -> None:
    silent_video = generate_fixture(tmp_path / "video-only.mov", landscape_spec())
    with pytest.raises(AudioError, match="no audio stream"):
        probe_audio_stream(silent_video)
    with pytest.raises(AudioError, match="no audio stream"):
        list(iter_audio_energy(silent_video, [0.1]))


@NEEDS_TOOLS
def test_cancellation_before_start(tmp_path: Path) -> None:
    video = generate_av_fixture(
        tmp_path / "cancel.mov", _tiny_spec(), audio_expr=CLICK_441_EXPR
    )
    with pytest.raises(AudioCancelled):
        list(iter_audio_energy(video, RATE_30HZ, is_cancelled=lambda: True))


@NEEDS_TOOLS
def test_cancellation_midstream(tmp_path: Path) -> None:
    video = generate_av_fixture(
        tmp_path / "cancel2.mov", _tiny_spec(), audio_expr=CLICK_441_EXPR
    )
    calls = {"count": 0}

    def _hook() -> bool:
        calls["count"] += 1
        return calls["count"] >= 2

    with pytest.raises(AudioCancelled):
        list(iter_audio_energy(video, RATE_30HZ, is_cancelled=_hook))
    assert calls["count"] >= 2


@NEEDS_TOOLS
def test_malformed_input_raises_actionable(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"this is not media")
    with pytest.raises(AudioError):
        probe_audio_stream(junk)
    with pytest.raises(AudioError):
        list(iter_audio_energy(junk, [0.1]))


@NEEDS_TOOLS
def test_results_are_deterministic(tmp_path: Path) -> None:
    video = generate_av_fixture(
        tmp_path / "repeat.mov", _tiny_spec(), audio_expr=CLICK_441_EXPR
    )
    first = [(item.time_seconds, item.energy) for item in _sample(video)]
    second = [(item.time_seconds, item.energy) for item in _sample(video)]
    assert first == second


@NEEDS_TOOLS
def test_source_file_is_not_modified(tmp_path: Path) -> None:
    video = generate_av_fixture(
        tmp_path / "immutable.mov", _tiny_spec(), audio_expr=CLICK_441_EXPR
    )
    before = (video.stat().st_size, video.stat().st_mtime_ns)
    _sample(video)
    assert (video.stat().st_size, video.stat().st_mtime_ns) == before
