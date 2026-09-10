"""Tests for the timestamped frame sampler (M2.1).

All media is generated synthetically in temporary directories with
``tests/media_factory.py``; no private footage, model inference, CLI,
caches, or network access are used. Real FFmpeg/ffprobe run the integration paths;
pure unit paths cover schedules, rotation math, and command shape.
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from serve_review.media import frames as frames_module
from serve_review.media.frames import (
    DEFAULT_SAMPLE_RATE_HZ,
    MAX_BUFFERED_FRAMES,
    FrameCancelled,
    FrameError,
    SampledFrame,
    build_rawvideo_decode_args,
    build_uniform_schedule,
    iter_frames_at_rate,
    iter_sampled_frames,
    rotate_rgb_frame,
    validate_schedule,
)
from serve_review.media.probe import probe_source
from media_factory import (
    ffmpeg_available,
    generate_fixture,
    landscape_spec,
    portrait_spec,
)

NEEDS_TOOLS = pytest.mark.skipif(
    not ffmpeg_available() or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for frame sampler integration tests",
)

FRAME_EPS = 0.003


def _make_fixture(
    tmp_path: Path,
    *,
    orientation: str = "landscape",
    duration_seconds: float = 1.0,
    fps: int = 30,
    name: str = "source.mov",
) -> Path:
    spec = (
        landscape_spec(duration_seconds=duration_seconds, fps=fps)
        if orientation == "landscape"
        else portrait_spec(duration_seconds=duration_seconds, fps=fps)
    )
    return generate_fixture(tmp_path / name, spec)


def _times(frames: list[SampledFrame]) -> list[float]:
    return [frame.time_seconds for frame in frames]


# --- Pure schedule behavior (no media required) ---


def test_uniform_schedule_counts_follow_rate_not_nominal_fps() -> None:
    thirty = build_uniform_schedule(1.0, 30.0)
    sixty = build_uniform_schedule(1.0, 60.0)
    assert len(thirty) == 30
    assert len(sixty) == 60
    assert thirty[0] == pytest.approx(0.0)
    assert thirty[1] - thirty[0] == pytest.approx(1.0 / 30.0)
    assert sixty[1] - sixty[0] == pytest.approx(1.0 / 60.0)
    # Every 30 Hz point coincides with every other 60 Hz point.
    for point in thirty:
        assert any(abs(point - candidate) < 1e-9 for candidate in sixty)
    assert DEFAULT_SAMPLE_RATE_HZ == pytest.approx(30.0)


def test_uniform_schedule_rejects_invalid_inputs() -> None:
    for bad_rate in (0, -30, float("nan"), float("inf"), "30", None, True):
        with pytest.raises(FrameError):
            build_uniform_schedule(1.0, bad_rate)  # type: ignore[arg-type]
    for bad_duration in (0, -1.0, float("nan"), float("inf"), "1", None):
        with pytest.raises(FrameError):
            build_uniform_schedule(bad_duration, 30.0)  # type: ignore[arg-type]
    with pytest.raises(FrameError):
        build_uniform_schedule(1.0, 1000.0)  # above MAX_SAMPLE_RATE_HZ
    with pytest.raises(FrameError):
        build_uniform_schedule(1.0, 30.0, start_seconds=1.0)  # start at duration
    with pytest.raises(FrameError):
        build_uniform_schedule(1.0, 30.0, start_seconds=-0.1)


def test_validate_schedule_requires_strictly_increasing_in_bounds_times() -> None:
    assert validate_schedule([0.0, 0.5], 1.0) == (0.0, 0.5)
    for bad in ([], (), 0.5, "0.5", [0.5, 0.5], [0.6, 0.5], [-0.1], [1.0], [1.5],
                [0.1, float("nan")], [True], [None]):
        with pytest.raises(FrameError):
            validate_schedule(bad, 1.0)  # type: ignore[arg-type]


def test_decode_command_shape_uses_passthrough_and_no_seek() -> None:
    args = build_rawvideo_decode_args("/tmp/src.mov")
    assert args[0] == "ffmpeg"
    assert all(isinstance(part, str) for part in args)
    assert "-f" in args and "rawvideo" in args
    assert args[args.index("-pix_fmt") + 1] == "rgb24"
    assert args[args.index("-fps_mode") + 1] == "passthrough"
    assert "-ss" not in args
    assert "-c" not in " ".join(args) or "copy" not in args
    assert build_rawvideo_decode_args("/tmp/src.mov") == args


def test_rotate_rgb_frame_maps_pixels_deterministically() -> None:
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
    assert rotate_rgb_frame(image, 0).shape == (2, 3, 3)
    assert np.array_equal(rotate_rgb_frame(image, 0), image)
    assert rotate_rgb_frame(image, 90).shape == (3, 2, 3)
    assert rotate_rgb_frame(image, 270).shape == (3, 2, 3)
    assert rotate_rgb_frame(image, 180).shape == (2, 3, 3)
    assert np.array_equal(rotate_rgb_frame(image, 90), np.rot90(image, k=3))
    assert np.array_equal(rotate_rgb_frame(image, 270), np.rot90(image, k=1))
    assert np.array_equal(rotate_rgb_frame(image, 180), np.rot90(image, k=2))
    assert rotate_rgb_frame(image, 90).flags["C_CONTIGUOUS"]
    with pytest.raises(FrameError):
        rotate_rgb_frame(image, 45)
    with pytest.raises(FrameError):
        rotate_rgb_frame(image, "90")  # type: ignore[arg-type]


# --- Integration: portrait/landscape coverage at 30 Hz ---


@NEEDS_TOOLS
def test_portrait_and_landscape_30hz_shapes_order_and_bounds(
    tmp_path: Path,
) -> None:
    portrait = _make_fixture(tmp_path, orientation="portrait", name="portrait.mov")
    landscape = _make_fixture(tmp_path, orientation="landscape", name="landscape.mov")
    portrait_meta = probe_source(portrait)
    landscape_meta = probe_source(landscape)

    portrait_frames = list(iter_frames_at_rate(portrait, 30.0))
    landscape_frames = list(iter_frames_at_rate(landscape, 30.0))

    assert len(portrait_frames) == 30
    assert len(landscape_frames) == 30
    for frame in portrait_frames:
        assert isinstance(frame.image, np.ndarray)
        assert frame.image.dtype == np.uint8
        assert frame.image.shape == (320, 240, 3)
        assert (frame.width, frame.height) == (240, 320)
        assert frame.nbytes == 320 * 240 * 3
    for frame in landscape_frames:
        assert frame.image.shape == (240, 320, 3)
        assert (frame.width, frame.height) == (320, 240)
    for frames in (portrait_frames, landscape_frames):
        times = _times(frames)
        assert times[0] == pytest.approx(0.0, abs=FRAME_EPS)
        assert all(later > earlier for earlier, later in zip(times, times[1:]))
        assert times[-1] == pytest.approx(29 / 30.0, abs=FRAME_EPS)
        assert all(0.0 <= moment < 1.0 for moment in times)
        for frame in frames:
            assert frame.timestamp_ms == int(round(frame.time_seconds * 1000))
    # Visible timestamp identity: the test pattern changes over time.
    assert not np.array_equal(portrait_frames[0].image, portrait_frames[-1].image)
    assert portrait_meta.width == 240 and landscape_meta.width == 320


@NEEDS_TOOLS
def test_canonical_timestamps_come_from_container_not_index(tmp_path: Path) -> None:
    import json
    import subprocess

    video = _make_fixture(
        tmp_path, orientation="landscape", duration_seconds=1.0, fps=24
    )
    frames = list(iter_frames_at_rate(video, 30.0))
    # Only 24 source frames exist; requesting 30 Hz must not invent frames.
    assert len(frames) == 24
    times = _times(frames)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    # Canonical grid is 1/24 s, not the requested 1/30 s grid.
    for index, moment in enumerate(times):
        assert moment == pytest.approx(index / 24.0, abs=FRAME_EPS)
    assert times[1] == pytest.approx(1 / 24.0, abs=FRAME_EPS)
    assert abs(times[1] - 1 / 30.0) > 0.005
    # Cross-check against ffprobe integer-derived timestamps directly.
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time",
            "-of", "json", str(video),
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    expected = [float(entry["best_effort_timestamp_time"])
                for entry in json.loads(completed.stdout)["frames"]]
    assert len(expected) == len(times)
    for got, want in zip(times, expected):
        assert got == pytest.approx(want, abs=1e-6)


@NEEDS_TOOLS
def test_requested_60hz_does_not_invent_frames(tmp_path: Path) -> None:
    video = _make_fixture(tmp_path, orientation="landscape")
    at_30 = list(iter_frames_at_rate(video, 30.0))
    at_60 = list(iter_frames_at_rate(video, 60.0))
    assert len(at_30) == 30
    assert len(at_60) == 30  # bounded by the 30 decoded source frames
    assert _times(at_60) == pytest.approx(_times(at_30), abs=1e-9)
    at_10 = list(iter_frames_at_rate(video, 10.0))
    assert len(at_10) == 10
    assert all(later > earlier for earlier, later in zip(_times(at_10), _times(at_10)[1:]))


@NEEDS_TOOLS
def test_variable_schedule_coverage_selects_nearest_forward_frame(
    tmp_path: Path,
) -> None:
    video = _make_fixture(tmp_path, orientation="landscape")
    schedule = (0.05, 0.5, 0.95)
    frames = list(iter_sampled_frames(video, schedule))
    assert len(frames) == 3
    times = _times(frames)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    interval = 1 / 30.0
    for target, moment in zip(schedule, times):
        assert moment >= target - 0.002  # tolerance for ffprobe rounding
        assert moment < target + interval + 0.002
    # Explicit metadata path agrees with implicit probing.
    metadata = probe_source(video)
    again = list(iter_sampled_frames(video, schedule, source=metadata))
    assert _times(again) == pytest.approx(times, abs=1e-9)


@NEEDS_TOOLS
def test_rotation_override_normalizes_orientation(tmp_path: Path) -> None:
    video = _make_fixture(tmp_path, orientation="landscape")
    upright = list(iter_sampled_frames(video, (0.1, 0.5)))
    rotated = list(iter_sampled_frames(video, (0.1, 0.5), rotation_degrees=90))
    assert upright[0].image.shape == (240, 320, 3)
    assert rotated[0].image.shape == (320, 240, 3)
    assert (rotated[0].width, rotated[0].height) == (240, 320)
    assert _times(rotated) == pytest.approx(_times(upright), abs=1e-9)
    flipped = list(iter_sampled_frames(video, (0.1,), rotation_degrees=180))
    assert flipped[0].image.shape == (240, 320, 3)
    with pytest.raises(FrameError):
        iter_sampled_frames(video, (0.1,), rotation_degrees=45)


@NEEDS_TOOLS
def test_timestamp_ms_kept_separate_from_canonical_seconds(
    tmp_path: Path,
) -> None:
    video = _make_fixture(tmp_path, orientation="landscape")
    frames = list(iter_frames_at_rate(video, 30.0))
    # 1/30 s is not an integral millisecond: float stays canonical.
    assert frames[1].time_seconds == pytest.approx(1 / 30.0, abs=FRAME_EPS)
    assert frames[1].timestamp_ms == 33
    assert abs(frames[1].time_seconds * 1000 - frames[1].timestamp_ms) > 0.1


# --- Cancellation, bounds, and errors ---


@NEEDS_TOOLS
def test_cancellation_before_and_midstream(tmp_path: Path) -> None:
    video = _make_fixture(tmp_path, orientation="landscape")
    schedule = tuple(index / 30.0 for index in range(30))
    with pytest.raises(FrameCancelled):
        list(iter_sampled_frames(video, schedule, is_cancelled=lambda: True))

    state = {"calls": 0}

    def _cancel_after_two() -> bool:
        state["calls"] += 1
        return state["calls"] > 3

    stream = iter_sampled_frames(video, schedule, is_cancelled=_cancel_after_two)
    assert isinstance(stream, Iterator)
    first = next(stream)
    second = next(stream)
    assert first.time_seconds < second.time_seconds
    with pytest.raises(FrameCancelled):
        next(stream)
    with pytest.raises(FrameError, match="is_cancelled"):
        iter_sampled_frames(video, schedule, is_cancelled=42)  # type: ignore[arg-type]


@NEEDS_TOOLS
def test_bounded_streaming_holds_one_frame_at_a_time(tmp_path: Path) -> None:
    assert MAX_BUFFERED_FRAMES == 1
    assert "MAX_BUFFERED_FRAMES" in frames_module.__doc__
    video = _make_fixture(
        tmp_path, orientation="landscape", duration_seconds=2.0, name="long.mov"
    )
    # A single-point schedule on a longer video yields exactly one bounded frame.
    stream = iter_sampled_frames(video, (0.0,))
    assert isinstance(stream, Iterator)
    assert not inspect.isgeneratorfunction(iter_sampled_frames)
    single = list(stream)
    assert len(single) == 1
    assert single[0].nbytes == 320 * 240 * 3
    # Incremental iteration works without retaining the whole video.
    stream = iter_frames_at_rate(video, 30.0)
    first = next(stream)
    assert first.time_seconds == pytest.approx(0.0, abs=FRAME_EPS)
    rest = list(stream)
    assert len(rest) == 2 * 30 - 1
    assert all(frame.nbytes == 320 * 240 * 3 for frame in rest)


@NEEDS_TOOLS
def test_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(FrameError, match="does not exist"):
        iter_sampled_frames(tmp_path / "missing.mov", (0.1,))
    video = _make_fixture(tmp_path, orientation="landscape")
    with pytest.raises(FrameError, match="at least one"):
        iter_sampled_frames(video, [])
    with pytest.raises(FrameError, match="outside the source timeline"):
        iter_sampled_frames(video, (5.0,))
    with pytest.raises(FrameError, match="strictly increasing"):
        iter_sampled_frames(video, (0.5, 0.2))
    corrupt = tmp_path / "corrupt.mov"
    corrupt.write_bytes(b"not-a-video" * 100)
    with pytest.raises(FrameError):
        list(iter_sampled_frames(corrupt, (0.1,)))
