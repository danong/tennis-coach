"""Focused M2.5 tests for the pose extraction coordinator.

Deterministic, offline, and independent of private footage, model
weights, and network access. Frame sampling and pose inference are
fully faked; caches live in temporary directories. Generated media is
not required because the sampler is injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_review.domain import SourceMetadata
from serve_review.media.frames import SampledFrame
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose.backend import InferenceError
from serve_review.pose.extract import (
    ExtractionCancelled,
    ExtractionError,
    default_cache_path_for,
    extract_poses,
)
from serve_review.pose.schema import (
    BodyKeypoint,
    CacheIdentity,
    FrameObservation,
    PersonBox,
    PersonObservation,
)


def make_metadata(duration: float = 1.0, fingerprint: str = "sha256:test-source") -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
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


def make_person(seed: float = 0.5) -> PersonObservation:
    joints = tuple(
        BodyKeypoint(x=min(0.99, max(0.01, 0.2 + 0.01 * i)), y=0.4, visibility=0.8)
        if i != 0 or True else None
        for i in range(33)
    )
    return PersonObservation(
        box=PersonBox(x_min=0.2, y_min=0.2, x_max=0.8, y_max=0.9),
        keypoints=joints,
        score=0.8,
    )


def make_frame_observation(time_seconds: float, *, persons: bool = True) -> FrameObservation:
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(make_person(),) if persons else (),
    )


def make_sampled_frame(time_seconds: float) -> SampledFrame:
    return SampledFrame(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        width=8,
        height=6,
        image=np.full((6, 8, 3), 7, dtype=np.uint8),
    )


class FakeBackend:
    """Serialized fake pose backend with ordered VIDEO stamps."""

    def __init__(
        self,
        *,
        model_name: str = "fake-heavy",
        model_version: str = "fake-v1",
        no_person_times: set[float] | None = None,
        fail_at: float | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_version = model_version
        self._no_person = set(no_person_times or ())
        self._fail_at = fail_at
        self.calls: list[int] = []
        self.last_ms: int | None = None
        self.closed = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def infer(self, image, time_seconds: float) -> FrameObservation:
        stamp = int(round(float(time_seconds) * 1000))
        if self.last_ms is not None and not stamp > self.last_ms:
            raise AssertionError(f"backend calls not serialized/ordered: {self.last_ms} -> {stamp}")
        self.last_ms = stamp
        self.calls.append(stamp)
        if self._fail_at is not None and float(time_seconds) == pytest.approx(self._fail_at):
            raise InferenceError(f"fake inference failure at t={time_seconds}")
        has_person = float(time_seconds) not in self._no_person
        # Match approx membership for float schedules.
        for moment in self._no_person:
            if abs(float(moment) - float(time_seconds)) < 1e-9:
                has_person = False
        return make_frame_observation(float(time_seconds), persons=has_person)

    def close(self) -> None:
        self.closed = True


def make_probe(metadata: SourceMetadata):
    def _probe(path: Path) -> SourceMetadata:
        return metadata
    return _probe


def make_frame_factory(times: list[float] | None = None):
    """Build a factory honoring the requested remaining schedule.

    When ``times`` is None, the factory yields one SampledFrame per
    requested schedule point (the common fake path).
    """

    def _factory(remaining: tuple[float, ...], metadata: SourceMetadata):
        points = list(remaining) if times is None else [t for t in times if t in set(remaining) or True]
        # Default: echo the remaining schedule exactly.
        if times is None:
            points = list(remaining)
        return iter(make_sampled_frame(t) for t in points)

    return _factory


def _touch_video(tmp_path: Path, name: str = "clip.mov") -> Path:
    video = tmp_path / name
    video.write_bytes(b"fake-video-bytes")
    return video


# --- Validation -----------------------------------------------------------


def test_default_cache_path_layout(tmp_path: Path) -> None:
    assert default_cache_path_for("refs/session.mov") == Path("output/session/cache/pose-v1.jsonl")
    assert default_cache_path_for(tmp_path / "clip.mov", tmp_path / "out") == (
        tmp_path / "out" / "clip" / "cache" / "pose-v1.jsonl"
    )


def test_rejects_missing_video(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="does not exist"):
        extract_poses(tmp_path / "missing.mov", tmp_path / "pose.jsonl")


def test_rejects_bad_sample_rate(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    for bad in (0, -5.0, 1000.0, float("nan"), True, "30"):
        with pytest.raises(ExtractionError, match="sample rate"):
            extract_poses(video, tmp_path / "pose.jsonl", sample_rate_hz=bad)  # type: ignore[arg-type]


def test_rejects_start_outside_timeline(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    meta = make_metadata(duration=1.0)
    backend = FakeBackend()
    with pytest.raises(ExtractionError, match="outside the source timeline"):
        extract_poses(
            video,
            tmp_path / "pose.jsonl",
            sampling_start_seconds=1.0,
            probe_fn=make_probe(meta),
            frame_factory=make_frame_factory(),
            backend_factory=lambda: backend,
        )


# --- Complete extraction --------------------------------------------------


def test_complete_extraction_writes_complete_cache(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)  # 6 frames at 30 Hz
    backend = FakeBackend()
    progress: list[tuple[int, int]] = []

    result = extract_poses(
        video,
        cache,
        sample_rate_hz=30.0,
        probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(),
        backend_factory=lambda: backend,
        progress_callback=lambda done, total, obs: progress.append((done, total)),
    )
    assert result.complete is True
    assert result.cache_hit is False
    assert result.frame_count == 6
    assert result.inferred_frames == 6
    assert result.cached_frames == 0
    assert backend.closed is True
    # Serialized strictly increasing VIDEO stamps.
    assert backend.calls == sorted(backend.calls)
    assert len(set(backend.calls)) == 6
    # Progress reported per inferred frame against the full schedule.
    assert [done for done, _ in progress] == [1, 2, 3, 4, 5, 6]
    assert all(total == 6 for _, total in progress)
    snapshot = cache_module.load_cache(cache)
    assert snapshot.complete is True
    assert len(snapshot.frames) == 6
    times = [f.time_seconds for f in snapshot.frames]
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    for frame in snapshot.frames:
        assert frame.timestamp_ms == int(round(frame.time_seconds * 1000))
    # Source untouched.
    assert video.read_bytes() == b"fake-video-bytes"


def test_no_person_frames_are_explicit(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.1)  # 3 frames at 30 Hz
    from serve_review.media.frames import build_uniform_schedule

    schedule = build_uniform_schedule(0.1, 30.0)
    assert len(schedule) == 3
    backend = FakeBackend(no_person_times={schedule[1]})
    result = extract_poses(
        video,
        cache,
        sample_rate_hz=30.0,
        probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(),
        backend_factory=lambda: backend,
    )
    assert result.frame_count == 3
    snapshot = cache_module.load_cache(cache)
    assert len(snapshot.frames) == 3
    assert snapshot.frames[1].persons == ()
    assert snapshot.frames[1].has_person is False
    assert snapshot.frames[0].has_person is True


def test_cache_hit_skips_inference(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)
    first = FakeBackend()
    first_result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(), backend_factory=lambda: first,
    )
    assert first_result.cache_hit is False
    before = cache.read_text(encoding="utf-8")

    second = FakeBackend()
    calls: list = []

    def _exploding_factory(remaining, metadata):
        calls.append(tuple(remaining))
        raise AssertionError("frame source must not run on a cache hit")

    result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=_exploding_factory, backend_factory=lambda: second,
    )
    assert result.cache_hit is True
    assert result.inferred_frames == 0
    assert result.frame_count == first_result.frame_count
    assert second.calls == []
    assert second.closed is True
    assert cache.read_text(encoding="utf-8") == before


def test_partial_resume_infers_only_remaining(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)
    from serve_review.media.frames import build_uniform_schedule

    full = build_uniform_schedule(0.2, 30.0)
    assert len(full) == 6
    backend_a = FakeBackend()
    identity = CacheIdentity(
        source_fingerprint=meta.fingerprint,
        model_name=backend_a.model_name,
        model_version=backend_a.model_version,
        sampling_rate_hz=30.0,
        sampling_start_seconds=0.0,
    )
    partial = [make_frame_observation(t) for t in full[:2]]
    cache_module.write_partial_cache(cache, identity, partial)

    seen: list[tuple[float, ...]] = []
    backend_b = FakeBackend()

    def _factory(remaining: tuple[float, ...], metadata):
        seen.append(tuple(remaining))
        return iter(make_sampled_frame(t) for t in remaining)

    result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=_factory, backend_factory=lambda: backend_b,
    )
    assert result.cache_hit is False
    assert result.cached_frames == 2
    assert result.inferred_frames == 4
    assert result.frame_count == 6
    assert len(seen) == 1
    assert list(seen[0]) == list(full[2:])
    # Only remaining stamps were submitted, strictly increasing.
    assert len(backend_b.calls) == 4
    snapshot = cache_module.load_cache(cache)
    assert snapshot.complete is True
    assert [f.time_seconds for f in snapshot.frames] == pytest.approx(list(full))


def test_overwrite_replaces_existing_cache(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)
    first = FakeBackend()
    extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(), backend_factory=lambda: first,
    )
    second = FakeBackend(model_version="fake-v2")
    result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(), backend_factory=lambda: second,
        overwrite=True,
    )
    assert result.cache_hit is False
    assert result.inferred_frames == 6
    snapshot = cache_module.load_cache(cache)
    assert snapshot.identity.model_version == "fake-v2"
    assert snapshot.complete is True


# --- Stale / corrupt recovery ---------------------------------------------


def test_stale_cache_is_quarantined_and_restarted(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)
    stale_identity = CacheIdentity(
        source_fingerprint="sha256:other-video",
        model_name="fake-heavy",
        model_version="fake-v1",
        sampling_rate_hz=30.0,
        sampling_start_seconds=0.0,
    )
    cache_module.write_complete_cache(
        cache, stale_identity, [make_frame_observation(0.0)]
    )
    backend = FakeBackend()
    result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(), backend_factory=lambda: backend,
    )
    assert result.cache_hit is False
    assert result.frame_count == 6
    assert (tmp_path / "pose-v1.jsonl.corrupt").is_file()
    snapshot = cache_module.load_cache(cache)
    assert snapshot.identity.source_fingerprint == meta.fingerprint
    assert snapshot.complete is True


def test_corrupt_cache_is_quarantined_and_restarted(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    cache.write_text("this is { not json\n", encoding="utf-8")
    meta = make_metadata(duration=0.2)
    backend = FakeBackend()
    result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(), backend_factory=lambda: backend,
    )
    assert result.frame_count == 6
    assert (tmp_path / "pose-v1.jsonl.corrupt").is_file()
    assert cache_module.load_cache(cache).complete is True


# --- Cancellation / errors -------------------------------------------------


def test_cancellation_leaves_resumable_partial(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=1.0)  # 30 frames at 30 Hz
    backend = FakeBackend()
    state = {"n": 0}

    def _cancel() -> bool:
        return state["n"] >= 3

    def _progress(done: int, total: int, obs) -> None:
        state["n"] = done

    with pytest.raises(ExtractionCancelled):
        extract_poses(
            video, cache, probe_fn=make_probe(meta),
            frame_factory=make_frame_factory(), backend_factory=lambda: backend,
            progress_callback=_progress, is_cancelled=_cancel,
        )
    snapshot = cache_module.load_cache(cache)
    assert snapshot.complete is False
    assert len(snapshot.frames) == 3
    # Resume completes without repeating cached stamps.
    backend2 = FakeBackend()
    result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=make_frame_factory(), backend_factory=lambda: backend2,
    )
    assert result.complete is True
    assert result.frame_count == 30
    assert result.cached_frames == 3
    assert result.inferred_frames == 27


def test_inference_error_preserves_partial_and_reports(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)
    from serve_review.media.frames import build_uniform_schedule

    full = build_uniform_schedule(0.2, 30.0)
    backend = FakeBackend(fail_at=full[2])
    with pytest.raises(ExtractionError, match="pose inference failed"):
        extract_poses(
            video, cache, probe_fn=make_probe(meta),
            frame_factory=make_frame_factory(), backend_factory=lambda: backend,
        )
    # Two frames succeeded before the failure; partial remains resumable.
    snapshot = cache_module.load_cache(cache)
    assert snapshot.complete is False
    assert len(snapshot.frames) == 2


def test_backend_altering_canonical_time_is_rejected(tmp_path: Path) -> None:
    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)

    class ShiftingBackend(FakeBackend):
        def infer(self, image, time_seconds: float):
            obs = super().infer(image, time_seconds)
            return FrameObservation(
                time_seconds=float(time_seconds) + 0.5,
                timestamp_ms=int(round((float(time_seconds) + 0.5) * 1000)),
                persons=obs.persons,
            )

    with pytest.raises(ExtractionError, match="canonical time"):
        extract_poses(
            video, cache, probe_fn=make_probe(meta),
            frame_factory=make_frame_factory(),
            backend_factory=lambda: ShiftingBackend(),
        )


def test_frame_source_is_streamed_not_retained(tmp_path: Path) -> None:
    import inspect

    video = _touch_video(tmp_path)
    cache = tmp_path / "pose-v1.jsonl"
    meta = make_metadata(duration=0.2)
    received: list[str] = []

    def _factory(remaining: tuple[float, ...], metadata):
        assert isinstance(remaining, tuple)
        received.append("called")

        def _gen():
            for t in remaining:
                yield make_sampled_frame(t)

        stream = _gen()
        assert inspect.isgenerator(stream)
        return stream

    backend = FakeBackend()
    result = extract_poses(
        video, cache, probe_fn=make_probe(meta),
        frame_factory=_factory, backend_factory=lambda: backend,
    )
    assert received == ["called"]
    assert result.frame_count == 6
