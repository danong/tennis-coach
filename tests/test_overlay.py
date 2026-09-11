"""Focused overlay tests: renderer, MP4 writer, and extract integration.

Deterministic, offline, and independent of private footage, model
weights, and network access. Pose inference is fully faked; caches and
overlay outputs live in temporary directories. Real FFmpeg encoding is
exercised only for tiny synthetic frames and is skipped when FFmpeg is
unavailable; failure/cleanup paths use injected fakes.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_review.domain import SourceMetadata
from serve_review.media.frames import SampledFrame, build_uniform_schedule
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose import overlay as overlay_module
from serve_review.pose.extract import (
    ExtractionCancelled,
    ExtractionError,
    extract_poses,
)
from serve_review.pose.overlay import (
    BUNDLED_POSE_CONNECTIONS,
    DOT_COLOR,
    LINE_COLOR,
    OverlayCancelled,
    OverlayError,
    build_overlay_ffmpeg_args,
    normalized_to_pixel,
    pose_connections,
    render_frame,
    write_overlay_video,
)
from serve_review.pose.schema import (
    NUM_KEYPOINTS,
    BodyKeypoint,
    FrameObservation,
    PersonBox,
    PersonObservation,
)


# --- Helpers --------------------------------------------------------------


def make_metadata(duration: float = 0.2) -> SourceMetadata:
    return SourceMetadata(
        fingerprint="sha256:test-overlay-source",
        duration_seconds=duration,
        width=320,
        height=240,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
        time_base_num=1,
        time_base_den=90000,
    )


def make_person(spread: bool = False) -> PersonObservation:
    joints: list[BodyKeypoint | None] = []
    for i in range(NUM_KEYPOINTS):
        if spread:
            # Unique pixels on a 64x64 grid: 8 columns x 5 rows, 7px apart.
            col = 2 + (i % 8) * 7
            row = 2 + (i // 8) * 7
            joints.append(
                BodyKeypoint(x=col / 63.0, y=row / 63.0, visibility=0.9)
            )
        else:
            joints.append(
                BodyKeypoint(
                    x=min(0.99, max(0.01, 0.2 + 0.01 * i)),
                    y=0.4,
                    visibility=0.8,
                )
            )
    return PersonObservation(
        box=PersonBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.9),
        keypoints=tuple(joints),
        score=0.8,
    )


def make_observation(time_seconds: float, *, persons: bool = True) -> FrameObservation:
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(make_person(),) if persons else (),
    )


def make_sampled_frame(
    time_seconds: float, width: int = 32, height: int = 24, fill: int = 7
) -> SampledFrame:
    return SampledFrame(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        width=width,
        height=height,
        image=np.full((height, width, 3), fill, dtype=np.uint8),
    )


class FakeBackend:
    def __init__(self, *, no_person: bool = False) -> None:
        self._no_person = no_person
        self.calls: list[int] = []
        self.closed = False

    @property
    def model_name(self) -> str:
        return "fake-heavy"

    @property
    def model_version(self) -> str:
        return "fake-v1"

    def infer(self, image, time_seconds: float) -> FrameObservation:
        self.calls.append(int(round(float(time_seconds) * 1000)))
        return make_observation(float(time_seconds), persons=not self._no_person)

    def close(self) -> None:
        self.closed = True


def echo_factory(width: int = 32, height: int = 24, fill: int = 7):
    def _factory(remaining: tuple[float, ...], metadata):
        return iter(
            make_sampled_frame(t, width=width, height=height, fill=fill)
            for t in remaining
        )

    return _factory


def _touch_video(tmp_path: Path) -> Path:
    video = tmp_path / "clip.mov"
    video.write_bytes(b"fake-video-bytes")
    return video


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


needs_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="FFmpeg and ffprobe are required"
)


# --- Connection table -----------------------------------------------------


def test_bundled_connections_cover_all_33_joints() -> None:
    assert len(BUNDLED_POSE_CONNECTIONS) == 35
    assert BUNDLED_POSE_CONNECTIONS == tuple(sorted(BUNDLED_POSE_CONNECTIONS))
    covered: set[int] = set()
    for first, second in BUNDLED_POSE_CONNECTIONS:
        assert 0 <= first < NUM_KEYPOINTS
        assert 0 <= second < NUM_KEYPOINTS
        assert first != second
        covered.add(first)
        covered.add(second)
    assert covered == set(range(NUM_KEYPOINTS))


def test_pose_connections_prefers_mediapipe_table_or_bundled() -> None:
    first = pose_connections()
    second = pose_connections()
    assert first == second  # deterministic across calls
    assert len(first) == 35
    covered: set[int] = set()
    for a, b in first:
        assert 0 <= a < NUM_KEYPOINTS and 0 <= b < NUM_KEYPOINTS
        covered.update((a, b))
    assert covered == set(range(NUM_KEYPOINTS))
    # The pinned mediapipe 0.10.35 ships no solutions tables, so the
    # bundled topology is the active source; either way every joint is
    # covered and the table is sorted/deduplicated.
    assert first == tuple(sorted({(min(a, b), max(a, b)) for a, b in first}))


# --- Normalized-to-pixel mapping ------------------------------------------


def test_normalized_to_pixel_mapping() -> None:
    assert normalized_to_pixel(0.0, 0.0, 64, 48) == (0, 0)
    assert normalized_to_pixel(1.0, 1.0, 64, 48) == (63, 47)
    assert normalized_to_pixel(0.5, 0.5, 65, 49) == (32, 24)
    # Origin is top-left, x right and y down, never mirrored.
    left = normalized_to_pixel(0.0, 0.5, 100, 100)[0]
    right = normalized_to_pixel(1.0, 0.5, 100, 100)[0]
    assert left < right
    top = normalized_to_pixel(0.5, 0.0, 100, 100)[1]
    bottom = normalized_to_pixel(0.5, 1.0, 100, 100)[1]
    assert top < bottom


def test_normalized_to_pixel_clamps_and_rejects() -> None:
    assert normalized_to_pixel(-0.2, 1.4, 10, 10) == (0, 9)
    with pytest.raises(OverlayError):
        normalized_to_pixel(float("nan"), 0.5, 10, 10)
    with pytest.raises(OverlayError):
        normalized_to_pixel(0.5, 0.5, 0, 10)


# --- Renderer -------------------------------------------------------------


def test_render_is_deterministic_and_preserves_input() -> None:
    image = np.full((24, 32, 3), 7, dtype=np.uint8)
    before = image.copy()
    obs = make_observation(0.0)
    first = render_frame(image, obs)
    second = render_frame(image, obs)
    assert np.array_equal(first, second)
    assert np.array_equal(image, before)  # input never mutated
    assert first is not image
    assert first.dtype == np.uint8 and first.shape == (24, 32, 3)


def test_render_draws_all_33_joints_as_dots() -> None:
    width, height = 64, 64
    image = np.zeros((height, width, 3), dtype=np.uint8)
    person = make_person(spread=True)
    obs = FrameObservation(
        time_seconds=0.0, timestamp_ms=0, persons=(person,)
    )
    rendered = render_frame(image, obs)
    dot = np.array(DOT_COLOR, dtype=np.uint8)
    for index, joint in enumerate(person.keypoints):
        assert joint is not None
        col, row = normalized_to_pixel(joint.x, joint.y, width, height)
        assert tuple(rendered[row, col]) == tuple(dot), (
            f"joint {index} dot missing at pixel {(col, row)}"
        )
    # Dots sit on top of skeleton lines at joint centers.


def test_render_draws_connection_midpoints_as_lines() -> None:
    width, height = 64, 64
    image = np.zeros((height, width, 3), dtype=np.uint8)
    joints: list[BodyKeypoint | None] = [None] * NUM_KEYPOINTS
    joints[11] = BodyKeypoint(x=0.2, y=0.2, visibility=0.9)  # left_shoulder
    joints[12] = BodyKeypoint(x=0.8, y=0.2, visibility=0.9)  # right_shoulder
    person = PersonObservation(
        box=PersonBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.3),
        keypoints=tuple(joints),
        score=0.9,
    )
    obs = FrameObservation(time_seconds=0.0, timestamp_ms=0, persons=(person,))
    rendered = render_frame(image, obs, connections=[(11, 12)])
    start = normalized_to_pixel(0.2, 0.2, width, height)
    end = normalized_to_pixel(0.8, 0.2, width, height)
    mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
    assert tuple(rendered[mid[1], mid[0]]) == LINE_COLOR
    # Joint centers remain dots drawn over the line.
    dot = np.array(DOT_COLOR, dtype=np.uint8)
    assert tuple(rendered[start[1], start[0]]) == tuple(dot)
    assert tuple(rendered[end[1], end[0]]) == tuple(dot)


def test_render_missing_joint_suppresses_connected_segments() -> None:
    width, height = 64, 64
    background = np.full((height, width, 3), 11, dtype=np.uint8)
    joints: list[BodyKeypoint | None] = [None] * NUM_KEYPOINTS
    joints[11] = BodyKeypoint(x=0.2, y=0.2, visibility=0.9)
    # Joint 12 stays missing: the (11, 12) segment must not be fabricated.
    person = PersonObservation(
        box=PersonBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.3),
        keypoints=tuple(joints),
        score=0.9,
    )
    obs = FrameObservation(time_seconds=0.0, timestamp_ms=0, persons=(person,))
    rendered = render_frame(background, obs, connections=[(11, 12)])
    line = np.array(LINE_COLOR, dtype=np.uint8)
    assert not np.any(np.all(rendered == line, axis=2))
    # The single present joint still draws its dot.
    col, row = normalized_to_pixel(0.2, 0.2, width, height)
    assert tuple(rendered[row, col]) == DOT_COLOR


def test_render_no_person_passthrough() -> None:
    image = np.full((12, 16, 3), 42, dtype=np.uint8)
    obs = make_observation(0.0, persons=False)
    rendered = render_frame(image, obs)
    assert np.array_equal(rendered, image)
    assert rendered is not image


def test_render_none_observation_passthrough() -> None:
    image = np.full((12, 16, 3), 42, dtype=np.uint8)
    rendered = render_frame(image, None)
    assert np.array_equal(rendered, image)


# --- FFmpeg argument construction -----------------------------------------


def test_build_overlay_ffmpeg_args() -> None:
    args = build_overlay_ffmpeg_args(
        "out.mp4", width=32, height=24, fps=30.0, ffmpeg="ffmpeg"
    )
    assert args == build_overlay_ffmpeg_args(
        "out.mp4", width=32, height=24, fps=30.0, ffmpeg="ffmpeg"
    )
    assert isinstance(args, list)
    assert args[0] == "ffmpeg"
    assert "-i" in args and args[args.index("-i") + 1] == "-"
    assert "libx264" in args
    assert "yuv420p" in args
    assert "32x24" in args
    assert "copy" not in [entry.lower() for entry in args]
    assert args[-1] == "out.mp4"


def test_build_overlay_ffmpeg_args_rejects_bad_inputs() -> None:
    with pytest.raises(OverlayError):
        build_overlay_ffmpeg_args("out.mp4", width=0, height=24, fps=30.0)
    with pytest.raises(OverlayError):
        build_overlay_ffmpeg_args("out.mp4", width=32, height=24, fps=0)
    with pytest.raises(OverlayError):
        build_overlay_ffmpeg_args(
            "out.mp4", width=32, height=24, fps=30.0, video_encoder="copy"
        )


# --- Writer behavior ------------------------------------------------------


def _tmp_leftovers(directory: Path) -> list[Path]:
    return [p for p in directory.iterdir() if ".tmp-" in p.name]


class _FakeStdin(io.BytesIO):
    def close(self) -> None:  # keep buffer readable after close
        pass

    def really_close(self) -> None:
        super().close()


class _FakeProc:
    def __init__(self, rc: int = 0, output: str | None = None) -> None:
        self.stdin = _FakeStdin()
        self.stderr = io.BytesIO(b"" if rc == 0 else b"fake ffmpeg boom")
        self._rc = rc
        self._output = output

    def poll(self):
        return self._rc

    def wait(self, timeout=None):
        if self._rc == 0 and self._output is not None:
            Path(self._output).write_bytes(b"fake-mp4-bytes")
        return self._rc

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


def _fake_popen_ok(args, **kwargs):
    return _FakeProc(0, output=args[-1])


def _annotated_frames(count: int, width: int = 16, height: int = 16):
    frames = []
    for _ in range(count):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        frames.append(render_frame(image, make_observation(0.0)))
    return frames


def test_write_overlay_rejects_empty_frames(tmp_path: Path) -> None:
    with pytest.raises(OverlayError, match="empty"):
        write_overlay_video(
            iter([]), tmp_path / "overlay.mp4", width=16, height=16, fps=30.0
        )
    assert not (tmp_path / "overlay.mp4").exists()


def test_write_overlay_rejects_bad_paths(tmp_path: Path) -> None:
    frames = _annotated_frames(1)
    with pytest.raises(OverlayError, match="non-blank"):
        write_overlay_video(
            iter(frames), "  ", width=16, height=16, fps=30.0
        )
    with pytest.raises(OverlayError, match="directory"):
        write_overlay_video(
            iter(frames), tmp_path, width=16, height=16, fps=30.0
        )
    with pytest.raises(OverlayError, match=".mp4"):
        write_overlay_video(
            iter(frames), tmp_path / "overlay.mov", width=16, height=16, fps=30.0
        )
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not-a-directory")
    with pytest.raises(OverlayError, match="could not write overlay"):
        write_overlay_video(
            iter(_annotated_frames(1)),
            blocker / "overlay.mp4",
            width=16,
            height=16,
            fps=30.0,
        )


def test_write_overlay_rejects_dimension_mismatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(overlay_module.subprocess, "Popen", _fake_popen_ok)
    dest = tmp_path / "overlay.mp4"
    frames = _annotated_frames(1) + [
        np.zeros((8, 8, 3), dtype=np.uint8),
    ]
    with pytest.raises(OverlayError, match="mixed dimensions"):
        write_overlay_video(
            iter(frames), dest, width=16, height=16, fps=30.0
        )
    assert not dest.exists()
    assert _tmp_leftovers(tmp_path) == []


def test_write_overlay_reports_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(overlay_module.subprocess, "Popen", _fake_popen_ok)
    seen: list[tuple[int, int | None]] = []
    dest = tmp_path / "sub" / "overlay.mp4"
    write_overlay_video(
        iter(_annotated_frames(3)),
        dest,
        width=16,
        height=16,
        fps=30.0,
        total_frames=3,
        progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert _tmp_leftovers(tmp_path / "sub") == []


def test_write_overlay_ffmpeg_failure_cleans_up(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        overlay_module.subprocess, "Popen", lambda *a, **k: _FakeProc(1)
    )
    dest = tmp_path / "overlay.mp4"
    with pytest.raises(OverlayError, match="ffmpeg failed"):
        write_overlay_video(
            iter(_annotated_frames(2)), dest, width=16, height=16, fps=30.0
        )
    assert not dest.exists()
    assert _tmp_leftovers(tmp_path) == []


def test_write_overlay_missing_ffmpeg_cleans_up(tmp_path: Path) -> None:
    dest = tmp_path / "overlay.mp4"
    with pytest.raises(OverlayError, match="was not found"):
        write_overlay_video(
            iter(_annotated_frames(2)),
            dest,
            width=16,
            height=16,
            fps=30.0,
            ffmpeg="ffmpeg-missing-xyz",
        )
    assert not dest.exists()
    assert _tmp_leftovers(tmp_path) == []


def test_write_overlay_cancelled_before_start(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(
        overlay_module.subprocess,
        "Popen",
        lambda *a, **k: calls.append(a) or _FakeProc(0, output=a[0][-1]),
    )
    dest = tmp_path / "overlay.mp4"
    with pytest.raises(OverlayCancelled):
        write_overlay_video(
            iter(_annotated_frames(2)),
            dest,
            width=16,
            height=16,
            fps=30.0,
            is_cancelled=lambda: True,
        )
    assert not dest.exists()
    assert _tmp_leftovers(tmp_path) == []


def test_write_overlay_cancelled_midstream_cleans_up(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(overlay_module.subprocess, "Popen", _fake_popen_ok)
    state = {"done": 0}

    def _cancel() -> bool:
        return state["done"] >= 1

    def _progress(done: int, total: int | None) -> None:
        state["done"] = done

    dest = tmp_path / "overlay.mp4"
    with pytest.raises(OverlayCancelled):
        write_overlay_video(
            iter(_annotated_frames(4)),
            dest,
            width=16,
            height=16,
            fps=30.0,
            progress_callback=_progress,
            is_cancelled=_cancel,
        )
    assert not dest.exists()
    assert _tmp_leftovers(tmp_path) == []


@needs_ffmpeg
def test_write_overlay_success_encodes_h264(tmp_path: Path) -> None:
    dest = tmp_path / "overlay.mp4"
    seen: list[tuple[int, int | None]] = []
    result = write_overlay_video(
        iter(_annotated_frames(3)),
        dest,
        width=16,
        height=16,
        fps=30.0,
        total_frames=3,
        progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert result == dest
    assert dest.is_file() and dest.stat().st_size > 0
    assert seen == [(1, 3), (2, 3), (3, 3)]
    assert _tmp_leftovers(tmp_path) == []
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    stream = json.loads(completed.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (16, 16)
    assert stream["codec_name"] == "h264"


# --- extract_poses overlay integration (fake backend) ---------------------


def test_extract_overlay_defaults_off(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    result = extract_poses(
        video,
        cache,
        sample_rate_hz=30.0,
        probe_fn=lambda path: make_metadata(),
        frame_factory=echo_factory(),
        backend_factory=FakeBackend,
    )
    assert result.overlay_path is None
    assert list(tmp_path.glob("*.mp4")) == []


@needs_ffmpeg
def test_extract_overlay_writes_mp4_over_sampled_frames(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    overlay = tmp_path / "diag" / "overlay.mp4"
    seen: list[tuple[int, int | None]] = []
    result = extract_poses(
        video,
        cache,
        sample_rate_hz=30.0,
        probe_fn=lambda path: make_metadata(),
        frame_factory=echo_factory(width=32, height=24),
        backend_factory=FakeBackend,
        overlay_path=overlay,
        overlay_progress_callback=lambda done, total: seen.append((done, total)),
    )
    assert result.overlay_path == overlay
    assert overlay.is_file() and overlay.stat().st_size > 0
    assert seen == [(i + 1, 6) for i in range(6)]
    assert _tmp_leftovers(tmp_path / "diag") == []
    snapshot = cache_module.load_cache(cache)
    assert snapshot.complete is True
    assert len(snapshot.frames) == 6
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name",
            "-of",
            "json",
            str(overlay),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    stream = json.loads(completed.stdout)["streams"][0]
    # Same dimensions as the sampled frames used for inference.
    assert (stream["width"], stream["height"]) == (32, 24)
    assert stream["codec_name"] == "h264"


@needs_ffmpeg
def test_extract_overlay_cache_hit_renders_full_schedule(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    schedules: list[tuple[float, ...]] = []

    def _factory(remaining: tuple[float, ...], metadata):
        schedules.append(tuple(remaining))
        return iter(make_sampled_frame(t) for t in remaining)

    first = FakeBackend()
    plain = extract_poses(
        video,
        cache,
        probe_fn=lambda path: make_metadata(),
        frame_factory=_factory,
        backend_factory=lambda: first,
    )
    assert plain.overlay_path is None
    full = build_uniform_schedule(0.2, 30.0)
    assert [s for s in schedules] != []

    overlay = tmp_path / "overlay.mp4"
    second = FakeBackend()
    result = extract_poses(
        video,
        cache,
        probe_fn=lambda path: make_metadata(),
        frame_factory=_factory,
        backend_factory=lambda: second,
        overlay_path=overlay,
    )
    assert result.cache_hit is True
    assert second.calls == []  # no re-inference on a cache hit
    assert result.overlay_path == overlay
    assert overlay.is_file() and overlay.stat().st_size > 0
    # The overlay pass re-streamed the full schedule (upright, same rate).
    assert tuple(schedules[-1]) == pytest.approx(tuple(full))
    assert _tmp_leftovers(tmp_path) == []


def test_extract_overlay_ffmpeg_failure_cleans_up(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    overlay = tmp_path / "overlay.mp4"
    with pytest.raises(ExtractionError, match="could not write pose overlay"):
        extract_poses(
            video,
            cache,
            probe_fn=lambda path: make_metadata(),
            frame_factory=echo_factory(),
            backend_factory=FakeBackend,
            overlay_path=overlay,
            ffmpeg="ffmpeg-missing-xyz",
        )
    assert not overlay.exists()
    assert _tmp_leftovers(tmp_path) == []
    # The complete pose cache was already published before the overlay.
    snapshot = cache_module.load_cache(cache)
    assert snapshot.complete is True
    assert len(snapshot.frames) == 6


def test_extract_overlay_unwritable_path_fails_before_inference(
    tmp_path: Path,
) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not-a-directory")
    backend = FakeBackend()
    with pytest.raises(ExtractionError, match="could not write pose overlay"):
        extract_poses(
            video,
            cache,
            probe_fn=lambda path: make_metadata(),
            frame_factory=echo_factory(),
            backend_factory=lambda: backend,
            overlay_path=blocker / "overlay.mp4",
        )
    assert backend.calls == []
    assert not cache.exists()


def test_extract_overlay_rejects_bad_destinations(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    backend = FakeBackend()
    with pytest.raises(ExtractionError, match=".mp4"):
        extract_poses(
            video,
            tmp_path / "pose.jsonl",
            probe_fn=lambda path: make_metadata(),
            frame_factory=echo_factory(),
            backend_factory=lambda: backend,
            overlay_path=tmp_path / "overlay.mov",
        )
    with pytest.raises(ExtractionError, match="directory"):
        extract_poses(
            video,
            tmp_path / "pose.jsonl",
            probe_fn=lambda path: make_metadata(),
            frame_factory=echo_factory(),
            backend_factory=lambda: backend,
            overlay_path=tmp_path,
        )
    assert backend.calls == []


def test_extract_overlay_cancelled_on_cache_hit(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    extract_poses(
        video,
        cache,
        probe_fn=lambda path: make_metadata(),
        frame_factory=echo_factory(),
        backend_factory=FakeBackend,
    )
    overlay = tmp_path / "overlay.mp4"
    with pytest.raises(ExtractionCancelled):
        extract_poses(
            video,
            cache,
            probe_fn=lambda path: make_metadata(),
            frame_factory=echo_factory(),
            backend_factory=FakeBackend,
            overlay_path=overlay,
            is_cancelled=lambda: True,
        )
    assert not overlay.exists()
    assert _tmp_leftovers(tmp_path) == []
    assert cache_module.load_cache(cache).complete is True


def test_extract_overlay_time_misalignment_is_rejected(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"

    def _shifting_factory(remaining: tuple[float, ...], metadata):
        # First call (inference) echoes the schedule; the overlay pass
        # shifts every frame time, which must never be misaligned.
        _shifting_factory.calls.append(tuple(remaining))
        shift = 0.0 if len(_shifting_factory.calls) == 1 else 0.25
        return iter(make_sampled_frame(t + shift) for t in remaining)

    _shifting_factory.calls = []

    with pytest.raises(ExtractionError, match="misalign"):
        extract_poses(
            video,
            cache,
            probe_fn=lambda path: make_metadata(),
            frame_factory=_shifting_factory,
            backend_factory=FakeBackend,
            overlay_path=tmp_path / "overlay.mp4",
            ffmpeg="ffmpeg-missing-xyz",  # unreachable: mismatch raises first
        )
    assert not (tmp_path / "overlay.mp4").exists()


def test_extract_overlay_no_person_frames_render_unannotated(
    tmp_path: Path, monkeypatch
) -> None:
    rendered: list[np.ndarray] = []
    real_render = overlay_module.render_frame

    def _spy(image, observation):
        out = real_render(image, observation)
        rendered.append(out.copy())
        return out

    monkeypatch.setattr(overlay_module, "render_frame", _spy)
    monkeypatch.setattr(overlay_module.subprocess, "Popen", _fake_popen_ok)
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    result = extract_poses(
        video,
        cache,
        probe_fn=lambda path: make_metadata(duration=0.1),
        frame_factory=echo_factory(width=16, height=16, fill=9),
        backend_factory=lambda: FakeBackend(no_person=True),
        overlay_path=tmp_path / "overlay.mp4",
    )
    assert result.overlay_path is not None
    assert len(rendered) == 3
    for frame in rendered:
        # No-person frames pass through unannotated: identical to input.
        assert np.all(frame == 9)
