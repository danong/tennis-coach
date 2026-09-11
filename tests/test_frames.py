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
    # Counter-clockwise display-matrix convention (matches FFmpeg
    # autorotate/export): 90 is one CCW quarter turn, 270 is three.
    assert np.array_equal(rotate_rgb_frame(image, 90), np.rot90(image, k=1))
    assert np.array_equal(rotate_rgb_frame(image, 270), np.rot90(image, k=3))
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


def test_decode_args_use_stored_rgb_without_autorotate() -> None:
    """Regression: decode must carry stored-orientation RGB bytes.

    FFmpeg auto-rotates display-matrix footage by default; without
    ``-noautorotate`` a portrait phone clip would arrive pre-rotated and
    the sampler's single manual upright rotation would double-rotate
    (garbled stride/pixels) so the VIDEO landmarker sees no pose. The
    sampler contract is therefore: ``-noautorotate`` before ``-i`` plus
    ``rgb24``/``passthrough`` with no seek. Fails on the prior arg
    array that omitted the flag.
    """
    args = build_rawvideo_decode_args("/tmp/src.mov")
    assert "-noautorotate" in args
    assert args.index("-noautorotate") < args.index("-i")
    assert args[args.index("-pix_fmt") + 1] == "rgb24"
    assert args[args.index("-fps_mode") + 1] == "passthrough"
    assert "-ss" not in args


def test_sampler_upright_rgb_contract_accepted_by_pose_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: synthetic stored frames yield upright RGB accepted by pose.

    Uses fake ffprobe/Popen bytes (no private footage, no binaries) to
    isolate the representation/transform contract: stored-orientation
    ``rgb24`` bytes plus a 90-degree counter-clockwise display rotation
    (FFmpeg display-matrix convention) must yield exactly one manual CCW
    rotation into upright display-oriented,
    contiguous ``uint8`` ``H x W x 3`` frames with canonical timestamps
    and independently owned pixel buffers. Every yielded frame is then
    fed through a fake contract-compatible pose boundary that enforces
    the same checks as ``MediaPipePoseBackend.infer`` (``uint8`` RGB,
    ``(H, W, 3)``, positive dims, contiguous SRGB-suitable buffer, and
    strictly increasing ``timestamp_ms``). A channel swap, double
    rotation, non-contiguous/reused buffer, or timestamp disorder would
    fail loudly here instead of surfacing as zero downstream poses.
    """
    from types import SimpleNamespace
    import io
    import subprocess as subprocess_module

    stored_width, stored_height = 4, 2
    rotation = 90
    frame_times = [0.0, 1 / 30.0, 2 / 30.0]

    def _stored_frame(index: int) -> np.ndarray:
        base = np.empty((stored_height, stored_width, 3), dtype=np.uint8)
        for y in range(stored_height):
            for x in range(stored_width):
                for c in range(3):
                    base[y, x, c] = (x * 17 + y * 31 + c * 47 + index * 53) % 256
        return base

    stored_frames = [_stored_frame(i) for i in range(len(frame_times))]
    payload = b"".join(frame.tobytes(order="C") for frame in stored_frames)

    captured_args: list[list[str]] = []

    class _FakeStdout(io.BytesIO):
        def close(self) -> None:  # keep BytesIO readable semantics
            pass

    class _FakeStderr:
        def read(self, *args: object, **kwargs: object) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class _FakeProc:
        def __init__(self, args: list[str]) -> None:
            captured_args.append(list(args))
            self.stdout: object = _FakeStdout(payload)
            self.stderr: object = _FakeStderr()

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

    video = tmp_path / "clip.mov"
    video.write_bytes(b"fake-video-bytes")
    metadata = SimpleNamespace(
        duration_seconds=1.0,
        width=stored_width,
        height=stored_height,
        rotation_degrees=0,
        fingerprint="sha256:fake",
    )
    monkeypatch.setattr(
        frames_module, "_resolve_metadata", lambda *a, **k: metadata
    )
    monkeypatch.setattr(
        frames_module, "_decode_frame_times", lambda *a, **k: list(frame_times)
    )
    monkeypatch.setattr(frames_module, "_ensure_tool", lambda *a, **k: None)
    monkeypatch.setattr(
        subprocess_module, "Popen", lambda args, **k: _FakeProc(list(args))
    )

    schedule = tuple(frame_times)
    frames = list(
        iter_sampled_frames(video, schedule, rotation_degrees=rotation)
    )
    assert len(frames) == len(frame_times)
    # Decode used the stored-RGB contract (fails prior: flag missing).
    assert captured_args, "expected one ffmpeg invocation"
    used = captured_args[0]
    assert "-noautorotate" in used
    assert used.index("-noautorotate") < used.index("-i")
    assert used[used.index("-pix_fmt") + 1] == "rgb24"
    # Upright display orientation: 90 CCW swaps stored (W=4,H=2) to (W=2,H=4).
    for index, frame in enumerate(frames):
        expected = np.ascontiguousarray(np.rot90(stored_frames[index], k=1))
        assert frame.width == stored_height
        assert frame.height == stored_width
        assert frame.image.dtype == np.uint8
        assert frame.image.ndim == 3 and frame.image.shape[2] == 3
        assert frame.image.shape == (stored_width, stored_height, 3)
        assert frame.image.flags["C_CONTIGUOUS"]
        assert np.array_equal(frame.image, expected)
        assert frame.time_seconds == frame_times[index]
        assert frame.timestamp_ms == int(round(frame_times[index] * 1000))
    times = [frame.time_seconds for frame in frames]
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    # Independently owned buffers: mutating one frame never corrupts another.
    assert not np.shares_memory(frames[0].image, frames[1].image)
    assert not np.shares_memory(frames[1].image, frames[2].image)
    before = frames[1].image.copy()
    frames[0].image[0, 0, 0] ^= 0xFF
    assert np.array_equal(frames[1].image, before)

    # Fake contract-compatible pose boundary (mirrors the adapter checks).
    class _ContractPoseBoundary:
        def __init__(self) -> None:
            self.calls: list[int] = []
            self._last_ms: int | None = None

        def infer(self, image: np.ndarray, time_seconds: float) -> SimpleNamespace:
            if not isinstance(image, np.ndarray):
                raise AssertionError("pose boundary requires np.ndarray")
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise AssertionError(f"bad SRGB buffer {image.dtype} {image.shape}")
            if image.shape[0] <= 0 or image.shape[1] <= 0:
                raise AssertionError("bad dims")
            if not image.flags["C_CONTIGUOUS"]:
                raise AssertionError("pose boundary requires C-contiguous SRGB")
            wrapped = np.ascontiguousarray(image, dtype=np.uint8)  # mp.Image(SRGB)
            assert wrapped.shape == image.shape
            stamp_ms = int(round(float(time_seconds) * 1000))
            if self._last_ms is not None and not stamp_ms > self._last_ms:
                raise AssertionError("VIDEO stamps must strictly increase")
            self._last_ms = stamp_ms
            self.calls.append(stamp_ms)
            return SimpleNamespace(time_seconds=float(time_seconds), persons=(object(),))

    boundary = _ContractPoseBoundary()
    accepted = [boundary.infer(frame.image, frame.time_seconds) for frame in frames]
    assert len(accepted) == len(frames)  # 3/3 detections, not 0/N
    assert boundary.calls == sorted(boundary.calls)
    assert len(set(boundary.calls)) == len(frames)


def _make_rotated_fixture(
    tmp_path: Path, *, width: int, height: int, rotation: int, name: str
) -> Path:
    """Generate a synthetic fixture carrying Display Matrix rotation.

    The stored pixels stay ``width`` x ``height`` (with the deterministic
    top-left corner patch from ``media_factory`` as the asymmetric
    marker); rotation metadata is attached with FFmpeg's
    ``-display_rotation`` input option plus stream copy, so ffprobe
    reports a ``side_data_list`` Display Matrix entry (90, -180, -90 for
    90/180/270). Rotation 0 is the plain fixture.
    """
    import subprocess as _subprocess

    from media_factory import FixtureSpec  # local import: test helper only

    base = generate_fixture(
        tmp_path / f"{name}-base.mov",
        FixtureSpec(
            width=width, height=height, duration_seconds=0.5, fps=10
        ),
    )
    if rotation == 0:
        return base
    out = tmp_path / f"{name}-{rotation}.mov"
    completed = _subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            f"-display_rotation:v:0", str(rotation),
            "-i", str(base), "-c", "copy", str(out),
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    return out


def _decode_first_frame_raw(
    video: Path, *, autorotate: bool, out_height: int, out_width: int
) -> np.ndarray:
    """Decode the first frame to ``uint8`` RGB with/without autorotate."""
    import subprocess as _subprocess

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if not autorotate:
        args.append("-noautorotate")
    args += [
        "-i", str(video), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    completed = _subprocess.run(args, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr[-500:]
    raw = bytes(completed.stdout)
    assert len(raw) == out_height * out_width * 3
    return np.frombuffer(raw, dtype=np.uint8).reshape(
        (out_height, out_width, 3)
    ).copy()


def test_sampler_orientation_contract_version_bumped() -> None:
    """The 90/270 convention fix bumps the sampler orientation identity.

    Version 1 treated probed rotation as clockwise, flipping upright
    frames from rotation-90/270 sources by 180 degrees relative to the
    FFmpeg autorotate/export orientation. Pose observations sampled under
    version 1 from such sources are stale (flipped coordinates) and must
    be quarantined/re-extracted, never silently validated.
    """
    assert frames_module.SAMPLER_ORIENTATION_VERSION == 2
    assert "SAMPLER_ORIENTATION_VERSION" in frames_module.__all__


@NEEDS_TOOLS
def test_sampler_matches_ffmpeg_autorotate_at_all_rotations(
    tmp_path: Path,
) -> None:
    """Regression: sampled upright frames match the autorotate ground truth.

    Synthetic landscape (320x240 stored) and portrait (240x320 stored)
    fixtures carry an asymmetric top-left corner marker plus Display
    Matrix rotation 0/90/180/270. Ground truth is the default FFmpeg
    decode (autorotate on -- the same orientation the export path
    produces, since export never passes ``-noautorotate``). The sampler
    decodes with ``-noautorotate`` and applies one manual rotation; its
    upright output must equal the ground truth at all four rotations.
    The prior clockwise convention matched at 0/180 but was 180 degrees
    off at 90/270, so this test also asserts the old orientation would
    mismatch there (sensitivity check).
    """
    cases = [
        (320, 240, "landscape"),
        (240, 320, "portrait"),
    ]
    for width, height, label in cases:
        for rotation in (0, 90, 180, 270):
            video = _make_rotated_fixture(
                tmp_path,
                width=width, height=height,
                rotation=rotation,
                name=f"{label}-{rotation}",
            )
            metadata = probe_source(video)
            assert metadata.rotation_degrees == rotation
            assert (metadata.width, metadata.height) == (width, height)
            # Ground truth: FFmpeg default (autorotate) decode orientation.
            if rotation in (90, 270):
                upright_w, upright_h = height, width
            else:
                upright_w, upright_h = width, height
            expected = _decode_first_frame_raw(
                video, autorotate=True,
                out_height=upright_h, out_width=upright_w,
            )
            sampled = list(iter_sampled_frames(video, (0.0,)))[0]
            assert (sampled.width, sampled.height) == (upright_w, upright_h)
            assert sampled.image.shape == (upright_h, upright_w, 3)
            mean_abs = float(
                np.abs(
                    sampled.image.astype(np.int16)
                    - expected.astype(np.int16)
                ).mean()
            )  # lossless transpose of identical stored bytes: ~0.
            assert mean_abs < 8.0, (
                f"{label} rotation {rotation}: sampler differs from "
                f"autorotate ground truth (MAD={mean_abs:.2f})"
            )
            if rotation in (90, 270):
                # Sensitivity: the prior clockwise convention (90<->270
                # swapped) sits 180 degrees off here, far from ground truth.
                stored = _decode_first_frame_raw(
                    video, autorotate=False,
                    out_height=height, out_width=width,
                )
                flipped = rotate_rgb_frame(
                    stored, 270 if rotation == 90 else 90
                )
                flipped_mad = float(
                    np.abs(
                        flipped.astype(np.int16)
                        - expected.astype(np.int16)
                    ).mean()
                )
                assert flipped_mad > 50.0, (
                    f"{label} rotation {rotation}: sensitivity check failed "
                    f"(flipped MAD={flipped_mad:.2f}); the marker may be "
                    "too weak to catch a 180-degree flip"
                )
    # The sampler decode contract is unchanged: stored-orientation bytes
    # via ``-noautorotate`` before ``-i``, bounded single-frame streaming.
    args = build_rawvideo_decode_args("/tmp/src.mov")
    assert "-noautorotate" in args
    assert args.index("-noautorotate") < args.index("-i")
    assert MAX_BUFFERED_FRAMES == 1


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


# --- Hardware-accelerated decode with ffmpeg-side early dropping ---


def test_decode_args_hwaccel_before_input_and_fps_matches_schedule() -> None:
    from serve_review.media.frames import FRAME_MATCH_TOLERANCE_SECONDS

    args = build_rawvideo_decode_args("/tmp/src.mov")
    assert args[0] == "ffmpeg"
    assert "-hwaccel" in args
    assert args[args.index("-hwaccel") + 1] == "videotoolbox"
    assert args.index("-hwaccel") < args.index("-i")
    assert "-noautorotate" in args
    assert args.index("-noautorotate") < args.index("-i")
    assert "-vf" not in args  # arbitrary schedules stay passthrough
    assert args[args.index("-pix_fmt") + 1] == "rgb24"
    assert args[args.index("-fps_mode") + 1] == "passthrough"
    assert FRAME_MATCH_TOLERANCE_SECONDS == pytest.approx(0.001)

    schedule = build_uniform_schedule(1.0, 30.0)
    rate = 30.0
    filtered = build_rawvideo_decode_args("/tmp/src.mov", rate_hz=rate)
    assert filtered.index("-hwaccel") < filtered.index("-i")
    assert filtered.index("-noautorotate") < filtered.index("-i")
    assert "-vf" in filtered
    vf_value = filtered[filtered.index("-vf") + 1]
    assert "round=up" in vf_value
    # fps value matches the scheduled rate.
    assert vf_value.startswith(f"fps={rate:g}")
    assert len(schedule) == 30
    with pytest.raises(FrameError):
        build_rawvideo_decode_args("/tmp/src.mov", rate_hz=0)  # type: ignore[arg-type]
    with pytest.raises(FrameError):
        build_rawvideo_decode_args("/tmp/src.mov", rate_hz=1000.0)
    assert build_rawvideo_decode_args("/tmp/src.mov", rate_hz=10.0) == \
        build_rawvideo_decode_args("/tmp/src.mov", rate_hz=10.0)


@NEEDS_TOOLS
def test_fps_dropping_keeps_canonical_times_within_tolerance(tmp_path: Path) -> None:
    """30 fps source sampled at 10 Hz: dropped frames keep exact times."""
    from serve_review.media.frames import FRAME_MATCH_TOLERANCE_SECONDS

    video = _make_fixture(tmp_path, orientation="landscape")
    rate = 10.0
    frames = list(iter_frames_at_rate(video, rate))
    schedule = build_uniform_schedule(1.0, rate)
    assert len(frames) == len(schedule) == 10
    times = _times(frames)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    for target, moment in zip(schedule, times):
        assert abs(moment - target) <= FRAME_MATCH_TOLERANCE_SECONDS
        assert moment >= target - FRAME_MATCH_TOLERANCE_SECONDS
    # fps filter used round=up (first frame at/after target); the default
    # round=near would sit one source interval (33 ms) off here.
    assert times[1] == pytest.approx(0.1, abs=FRAME_MATCH_TOLERANCE_SECONDS)


@NEEDS_TOOLS
def test_120fps_to_30hz_keeps_canonical_times(tmp_path: Path) -> None:
    """Real 120 fps fixture sampled at 30 Hz stays within tolerance."""
    import subprocess as _subprocess

    from serve_review.media.frames import FRAME_MATCH_TOLERANCE_SECONDS

    out = tmp_path / "high120.mov"
    completed = _subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "testsrc2=size=320x240:rate=120:duration=0.5",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23", str(out),
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-500:]
    rate = 30.0
    frames = list(iter_frames_at_rate(out, rate))
    schedule = build_uniform_schedule(0.5, rate)
    assert len(frames) == len(schedule) == 15
    times = _times(frames)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    for target, moment in zip(schedule, times):
        assert abs(moment - target) <= FRAME_MATCH_TOLERANCE_SECONDS


def test_240fps_style_schedule_maps_filtered_outputs_to_source_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic 240 Hz source times at a 30 Hz schedule stay exact."""
    from types import SimpleNamespace
    import io
    import subprocess as subprocess_module

    from serve_review.media.frames import FRAME_MATCH_TOLERANCE_SECONDS

    stored_width, stored_height = 8, 8
    frame_times = [index / 240.0 for index in range(120)]  # 0.5 s at 240 Hz
    schedule = tuple(index / 30.0 for index in range(15))  # 0.5 s at 30 Hz
    assert len(schedule) == 15

    def _stored_frame(index: int) -> np.ndarray:
        base = np.empty((stored_height, stored_width, 3), dtype=np.uint8)
        for y in range(stored_height):
            for x in range(stored_width):
                for c in range(3):
                    base[y, x, c] = (x * 13 + y * 29 + c * 37 + index * 11) % 256
        return base

    source_frames = [_stored_frame(i) for i in range(len(frame_times))]
    # Uniform 240 -> 30 Hz selects every 8th source frame.
    selected = [index * 8 for index in range(15)]
    payload = b"".join(
        source_frames[i].tobytes(order="C") for i in selected
    )
    captured: list[list[str]] = []

    class _FakeStdout(io.BytesIO):
        def close(self) -> None:
            pass

    class _FakeStderr:
        def read(self, *args: object, **kwargs: object) -> bytes:
            return b""

        def close(self) -> None:
            pass

    class _FakeProc:
        def __init__(self, args: list[str]) -> None:
            captured.append(list(args))
            self.stdout: object = _FakeStdout(payload)
            self.stderr: object = _FakeStderr()

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

    video = tmp_path / "clip.mov"
    video.write_bytes(b"fake-video-bytes")
    metadata = SimpleNamespace(
        duration_seconds=0.5,
        width=stored_width,
        height=stored_height,
        rotation_degrees=0,
        fingerprint="sha256:fake",
    )
    monkeypatch.setattr(
        frames_module, "_resolve_metadata", lambda *a, **k: metadata
    )
    monkeypatch.setattr(
        frames_module, "_decode_frame_times", lambda *a, **k: list(frame_times)
    )
    monkeypatch.setattr(frames_module, "_ensure_tool", lambda *a, **k: None)
    monkeypatch.setattr(
        subprocess_module, "Popen", lambda args, **k: _FakeProc(list(args))
    )
    frames = list(iter_sampled_frames(video, schedule))
    assert len(frames) == len(schedule) == 15
    assert captured and "-hwaccel" in captured[0]
    assert captured[0].index("-hwaccel") < captured[0].index("-i")
    vf_value = captured[0][captured[0].index("-vf") + 1]
    assert vf_value.startswith("fps=30") and "round=up" in vf_value
    for frame, target, source_index in zip(frames, schedule, selected):
        assert abs(frame.time_seconds - target) <= FRAME_MATCH_TOLERANCE_SECONDS
        assert frame.time_seconds == pytest.approx(
            frame_times[source_index], abs=1e-9
        )
        assert np.array_equal(frame.image, source_frames[source_index])


@NEEDS_TOOLS
def test_filtered_orientation_matches_autorotate_at_90_270(tmp_path: Path) -> None:
    """Filtered (fps-dropped) upright frames match the export orientation."""
    for rotation in (90, 270):
        video = _make_rotated_fixture(
            tmp_path, width=320, height=240, rotation=rotation,
            name=f"filt-{rotation}",
        )
        frames = list(iter_frames_at_rate(video, 10.0))
        assert len(frames) == 5  # 0.5 s fixture at 10 Hz
        expected = _decode_first_frame_raw(
            video, autorotate=True, out_height=320, out_width=240,
        )
        assert frames[0].image.shape == (320, 240, 3)
        mad = float(
            np.abs(
                frames[0].image.astype(np.int16) - expected.astype(np.int16)
            ).mean()
        )
        assert mad < 8.0, f"rotation {rotation}: MAD={mad:.2f}"
        args = build_rawvideo_decode_args(str(video), rate_hz=10.0)
        assert args.index("-hwaccel") < args.index("-i")
        assert args.index("-noautorotate") < args.index("-i")


@NEEDS_TOOLS
def test_filtered_cancellation_midstream(tmp_path: Path) -> None:
    video = _make_fixture(tmp_path, orientation="landscape")
    with pytest.raises(FrameCancelled):
        list(iter_frames_at_rate(video, 10.0, is_cancelled=lambda: True))
    state = {"calls": 0}

    def _cancel_after_two() -> bool:
        state["calls"] += 1
        return state["calls"] > 3

    stream = iter_frames_at_rate(video, 10.0, is_cancelled=_cancel_after_two)
    first = next(stream)
    second = next(stream)
    assert first.time_seconds < second.time_seconds
    with pytest.raises(FrameCancelled):
        next(stream)


def test_hwaccel_unavailable_fails_actionably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VideoToolbox init failure raises an actionable error, no fallback."""
    from types import SimpleNamespace
    import io
    import subprocess as subprocess_module

    stored_width, stored_height = 320, 240
    frame_times = [index / 30.0 for index in range(30)]
    schedule = tuple(index / 30.0 for index in range(30))

    class _EmptyStdout(io.BytesIO):
        def close(self) -> None:
            pass

    class _FailStderr:
        def read(self, *args: object, **kwargs: object) -> bytes:
            return (
                b"[vist#0:0/h264] No device available for decoder: "
                b"device type videotoolbox needed for codec h264. "
                b"Hardware device setup failed for decoder."
            )

        def close(self) -> None:
            pass

    seen: list[list[str]] = []

    class _FailProc:
        def __init__(self, args: list[str]) -> None:
            seen.append(list(args))
            self.stdout: object = _EmptyStdout(b"")
            self.stderr: object = _FailStderr()

        def poll(self) -> int | None:
            return 1

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 1

    video = tmp_path / "clip.mov"
    video.write_bytes(b"fake-video-bytes")
    metadata = SimpleNamespace(
        duration_seconds=1.0,
        width=stored_width,
        height=stored_height,
        rotation_degrees=0,
        fingerprint="sha256:fake",
    )
    monkeypatch.setattr(
        frames_module, "_resolve_metadata", lambda *a, **k: metadata
    )
    monkeypatch.setattr(
        frames_module, "_decode_frame_times", lambda *a, **k: list(frame_times)
    )
    monkeypatch.setattr(frames_module, "_ensure_tool", lambda *a, **k: None)
    monkeypatch.setattr(
        subprocess_module, "Popen", lambda args, **k: _FailProc(list(args))
    )
    with pytest.raises(FrameError, match="VideoToolbox"):
        list(iter_sampled_frames(video, schedule))
    assert seen and "-hwaccel" in seen[0]
    assert seen[0][seen[0].index("-hwaccel") + 1] == "videotoolbox"


@NEEDS_TOOLS
def test_upsampling_still_does_not_invent_frames(tmp_path: Path) -> None:
    """60 Hz on a 30 fps source yields 30 canonical frames, never 60."""
    video = _make_fixture(tmp_path, orientation="landscape")
    at_60 = list(iter_frames_at_rate(video, 60.0))
    at_30 = list(iter_frames_at_rate(video, 30.0))
    assert len(at_60) == len(at_30) == 30
    assert _times(at_60) == pytest.approx(_times(at_30), abs=1e-9)
