"""Focused M4.7 tests for dense native-frame world extraction.

Deterministic, offline, and independent of private footage, model
weights, and network access. Frame decoding is faked through the
injectable ``native_frame_factory`` (or by monkeypatching the ffprobe
timestamp listing); pose inference is faked through the injectable
``backend_factory`` whose ``infer_world`` maps through the real
:func:`result_to_world_frame_observation` so world/2D pairing,
missingness, and the never-substitute-2D-z contract are exercised for
real. Real FFmpeg/ffprobe run only inside ``tests/test_frames.py``
native-iterator integration tests, never here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_review.domain import Attempt, AttemptDocument, MediaRange, SourceMetadata
from serve_review.media import frames as frames_module
from serve_review.pose import mediapipe as mediapipe_module
from serve_review.pose import world as world_module
from serve_review.pose import world_extract as world_extract_module
from serve_review.pose.world import WorldFrameObservation

DURATION = 10.0
START = 2.0
END = 3.0
# Irregular (VFR-style) native grid inside [START, END): never uniform.
NATIVE_TIMES = (2.0, 2.033, 2.041, 2.09, 2.15, 2.151, 2.3, 2.7, 2.99)


def _metadata(fingerprint: str = "sha256:test-source") -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=DURATION,
        width=320,
        height=240,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
    )


def _attempt(
    attempt_id: str = "serve-001",
    start: float = START,
    end: float = END,
    pad: float = 0.0,
) -> Attempt:
    return Attempt(
        attempt_id=attempt_id,
        detected_range=MediaRange(start, end),
        effective_range=MediaRange(start - pad, end + pad),
    )


def _document(
    metadata: SourceMetadata,
    attempts: tuple[Attempt, ...] | None = None,
) -> AttemptDocument:
    entries = attempts if attempts is not None else (_attempt(),)
    return AttemptDocument(
        source_fingerprint=metadata.fingerprint,
        source_duration_seconds=metadata.duration_seconds,
        padding_seconds=0.0,
        attempts=entries,
        export_ranges=tuple(
            MediaRange(a.effective_range.start_seconds, a.effective_range.end_seconds)
            for a in entries
        ),
    )


def _write_inputs(tmp_path: Path, metadata: SourceMetadata, document: AttemptDocument):
    video = tmp_path / "source.mov"
    video.write_bytes(b"fake-video-bytes")
    attempts_path = tmp_path / "attempts.json"
    attempts_path.write_text(document.to_json(), encoding="utf-8")
    return video, attempts_path


def _pose_2d(seed: float = 0.0):
    pose = []
    for index in range(33):
        x = min(0.99, max(0.01, 0.05 + 0.9 * (index / 32) + seed))
        pose.append(SimpleNamespace(x=x, y=0.10 + 0.01 * (index % 5), visibility=0.9))
    return pose


def _pose_world(seed: float = 0.0):
    return [
        SimpleNamespace(
            x=0.01 * index - 0.16 + seed,
            y=1.0 + 0.004 * index,
            z=-0.2 + 0.003 * index,
        )
        for index in range(33)
    ]


def _world_result(empty: bool = False, seed: float = 0.0):
    if empty:
        return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
    return SimpleNamespace(
        pose_landmarks=[_pose_2d(seed)], pose_world_landmarks=[_pose_world(seed)]
    )


class FakeFrame:
    def __init__(self, time_seconds: float) -> None:
        self.time_seconds = float(time_seconds)
        self.image = np.full((4, 4, 3), 7, dtype=np.uint8)


class FakeBackend:
    """Fake serialized backend mapping through the real world mapper."""

    def __init__(
        self,
        results: list,
        *,
        model_name: str = mediapipe_module.MODEL_NAME,
        model_version: str = mediapipe_module.MODEL_VERSION,
    ) -> None:
        self._results = list(results)
        self.calls: list[float] = []
        self.model_name = model_name
        self.model_version = model_version
        self.closed = False

    def infer_world(self, image: np.ndarray, time_seconds: float):
        self.calls.append(float(time_seconds))
        index = len(self.calls) - 1
        outcome = self._results[index] if index < len(self._results) else self._results[-1]
        if isinstance(outcome, BaseException):
            raise outcome
        return mediapipe_module.result_to_world_frame_observation(
            outcome, time_seconds=float(time_seconds)
        )

    def close(self) -> None:
        self.closed = True


def _patch_native_times(monkeypatch: pytest.MonkeyPatch, times=NATIVE_TIMES):
    monkeypatch.setattr(
        frames_module, "native_frame_times", lambda *a, **k: tuple(times)
    )


def _factory(results, seen: dict | None = None):
    def _make(remaining: tuple[float, ...], metadata: SourceMetadata):
        if seen is not None:
            seen.setdefault("ranges", []).append(tuple(remaining))
        return iter([FakeFrame(t) for t in remaining])

    backend = FakeBackend(results)
    return _make, backend


def test_happy_path_writes_complete_cache_with_exact_native_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    seen: dict = {}
    frame_factory, backend = _factory([_world_result(seed=i * 0.01) for i in range(20)], seen)

    result = world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=cache,
        probe_fn=lambda path: metadata,
        native_frame_factory=frame_factory,
        backend_factory=lambda: backend,
    )
    assert result.complete is True
    assert result.cache_hit is False
    assert result.frame_count == len(NATIVE_TIMES)
    assert result.inferred_frames == len(NATIVE_TIMES)
    assert result.cached_frames == 0
    assert result.attempt_id == "serve-001"
    assert (result.attempt_start_seconds, result.attempt_end_seconds) == (START, END)
    assert backend.closed is True
    # Factory saw exactly the native grid (irregular: no uniform schedule).
    assert seen["ranges"] == [tuple(NATIVE_TIMES)]
    snapshot = world_module.load_world_cache(cache)
    assert snapshot.complete is True
    assert tuple(f.time_seconds for f in snapshot.frames) == tuple(NATIVE_TIMES)
    assert snapshot.identity.source_fingerprint == metadata.fingerprint
    assert snapshot.identity.model_name == mediapipe_module.MODEL_NAME
    assert snapshot.identity.representation == "dense-world-v1"
    for frame in snapshot.frames:
        assert isinstance(frame, WorldFrameObservation)
        assert frame.world_present_count == 33
        assert frame.has_person is True
        assert frame.frame_2d.time_seconds == frame.time_seconds
    # Backend stamps strictly increase (shared VIDEO ordering).
    assert backend.calls == sorted(backend.calls)
    assert len(set(round(t * 1000) for t in backend.calls)) == len(backend.calls)
    assert video.read_bytes() == b"fake-video-bytes"
    assert json.loads(attempts_path.read_text(encoding="utf-8"))["attempts"][0][
        "attempt_id"
    ] == "serve-001"


def test_uses_detected_range_not_effective_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    # Padded effective range is wider; extraction must still cover only [2, 3).
    document = _document(metadata, (_attempt(pad=0.0),))
    padded = Attempt(
        attempt_id="serve-001",
        detected_range=MediaRange(START, END),
        effective_range=MediaRange(START - 1.0, END + 1.0),
    )
    document = _document(metadata, (padded,))
    video, attempts_path = _write_inputs(tmp_path, metadata, document)
    _patch_native_times(monkeypatch)
    seen: dict = {}
    frame_factory, backend = _factory([_world_result() for _ in range(20)], seen)
    result = world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=tmp_path / "world.jsonl",
        probe_fn=lambda path: metadata,
        native_frame_factory=frame_factory,
        backend_factory=lambda: backend,
    )
    assert (result.attempt_start_seconds, result.attempt_end_seconds) == (START, END)
    assert seen["ranges"] == [tuple(NATIVE_TIMES)]
    assert all(START <= t < END for t in backend.calls)


def test_fingerprint_mismatch_rejected(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata()
    other = _metadata(fingerprint="sha256:other")
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    _patch_native_times(monkeypatch)
    with pytest.raises(world_extract_module.WorldExtractionInputError, match="fingerprint"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=tmp_path / "world.jsonl",
            probe_fn=lambda path: other,
            native_frame_factory=_factory([_world_result()])[0],
            backend_factory=lambda: FakeBackend([_world_result()]),
        )
    assert not (tmp_path / "world.jsonl").exists()


def test_unknown_attempt_id_rejected(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    _patch_native_times(monkeypatch)
    with pytest.raises(world_extract_module.WorldExtractionInputError, match="unknown attempt"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-009",
            cache_path=tmp_path / "world.jsonl",
            probe_fn=lambda path: metadata,
            native_frame_factory=_factory([_world_result()])[0],
            backend_factory=lambda: FakeBackend([_world_result()]),
        )


def test_invalid_attempt_id_format_rejected(tmp_path: Path, monkeypatch) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    _patch_native_times(monkeypatch)
    with pytest.raises(world_extract_module.WorldExtractionInputError, match="attempt id"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="001",
            cache_path=tmp_path / "world.jsonl",
            probe_fn=lambda path: metadata,
        )


def test_missing_and_no_person_rows_stay_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    times = NATIVE_TIMES[:3]
    _patch_native_times(monkeypatch, times)
    mid_world = _pose_world()
    mid_world[5] = SimpleNamespace(x=None, y=None, z=None)
    results = [
        _world_result(empty=True),
        SimpleNamespace(
            pose_landmarks=[_pose_2d()], pose_world_landmarks=[list(mid_world)]
        ),
        _world_result(seed=0.02),
    ]
    frame_factory, backend = _factory(results)
    result = world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=tmp_path / "world.jsonl",
        probe_fn=lambda path: metadata,
        native_frame_factory=frame_factory,
        backend_factory=lambda: backend,
    )
    assert result.frame_count == 3
    snapshot = world_module.load_world_cache(tmp_path / "world.jsonl")
    assert snapshot.frames[0].has_person is False
    assert snapshot.frames[0].world_present_count == 0
    assert snapshot.frames[1].world_landmarks[5] is None
    assert snapshot.frames[1].world_present_count == 32
    assert snapshot.frames[2].world_present_count == 33


def test_world_rows_come_from_world_stream_never_2d_z(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    _patch_native_times(monkeypatch, NATIVE_TIMES[:1])
    pose_2d = _pose_2d()
    for landmark in pose_2d:
        landmark.z = 0.999  # type: ignore[attr-defined]
    world_pose = _pose_world()
    before = (world_pose[0].x, world_pose[0].y, world_pose[0].z)
    frame_factory, backend = _factory(
        [SimpleNamespace(pose_landmarks=[pose_2d], pose_world_landmarks=[world_pose])]
    )
    world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=tmp_path / "world.jsonl",
        probe_fn=lambda path: metadata,
        native_frame_factory=frame_factory,
        backend_factory=lambda: backend,
    )
    stored = world_module.load_world_cache(tmp_path / "world.jsonl").frames[0]
    first = stored.world_landmarks[0]
    assert first is not None
    assert (first.x, first.y, first.z) == pytest.approx(before)
    assert first.z != pytest.approx(0.999)


def test_complete_matching_cache_is_hit_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    frame_factory, backend = _factory([_world_result(seed=i * 0.01) for i in range(20)])
    first = world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=cache,
        probe_fn=lambda path: metadata,
        native_frame_factory=frame_factory,
        backend_factory=lambda: backend,
    )
    assert first.cache_hit is False

    calls: list = []

    def _boom_factory(remaining, meta):
        calls.append(tuple(remaining))
        raise AssertionError("must not stream on a cache hit")

    second_backend = FakeBackend([_world_result()])

    second = world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=cache,
        probe_fn=lambda path: metadata,
        native_frame_factory=_boom_factory,
        backend_factory=lambda: second_backend,
    )
    assert second.cache_hit is True
    assert second.frame_count == len(NATIVE_TIMES)
    assert second.inferred_frames == 0
    assert second_backend.calls == []
    assert calls == []


def test_partial_cache_resumes_after_last_stored_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    prefix = NATIVE_TIMES[:4]
    identity = world_module.WorldCacheIdentity(
        source_fingerprint=metadata.fingerprint,
        model_name=mediapipe_module.MODEL_NAME,
        model_version=mediapipe_module.MODEL_VERSION,
    )
    prefix_obs = [
        mediapipe_module.result_to_world_frame_observation(_world_result(), time_seconds=t)
        for t in prefix
    ]
    world_module.write_partial_world_cache(cache, identity, prefix_obs)

    seen: dict = {}
    frame_factory, backend = _factory([_world_result(seed=0.5) for _ in range(20)], seen)
    result = world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=cache,
        probe_fn=lambda path: metadata,
        native_frame_factory=frame_factory,
        backend_factory=lambda: backend,
    )
    assert result.cache_hit is False
    assert result.cached_frames == 4
    assert result.inferred_frames == len(NATIVE_TIMES) - 4
    assert seen["ranges"] == [tuple(NATIVE_TIMES[4:])]
    snapshot = world_module.load_world_cache(cache)
    assert snapshot.complete is True
    assert tuple(f.time_seconds for f in snapshot.frames) == tuple(NATIVE_TIMES)


def test_no_resume_fails_on_partial_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    identity = world_module.WorldCacheIdentity(
        source_fingerprint=metadata.fingerprint,
        model_name=mediapipe_module.MODEL_NAME,
        model_version=mediapipe_module.MODEL_VERSION,
    )
    world_module.write_partial_world_cache(
        cache,
        identity,
        [
            mediapipe_module.result_to_world_frame_observation(
                _world_result(), time_seconds=NATIVE_TIMES[0]
            )
        ],
    )
    with pytest.raises(world_extract_module.WorldExtractionCollisionError, match="no-resume"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=cache,
            probe_fn=lambda path: metadata,
            native_frame_factory=_factory([_world_result()])[0],
            backend_factory=lambda: FakeBackend([_world_result()]),
            no_resume=True,
        )


def test_complete_cache_for_other_range_collides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    identity = world_module.WorldCacheIdentity(
        source_fingerprint=metadata.fingerprint,
        model_name=mediapipe_module.MODEL_NAME,
        model_version=mediapipe_module.MODEL_VERSION,
    )
    other_times = (2.0, 2.5, 2.9)
    world_module.write_complete_world_cache(
        cache,
        identity,
        [
            mediapipe_module.result_to_world_frame_observation(
                _world_result(), time_seconds=t
            )
            for t in other_times
        ],
    )
    with pytest.raises(world_extract_module.WorldExtractionCollisionError, match="overwrite"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=cache,
            probe_fn=lambda path: metadata,
            native_frame_factory=_factory([_world_result()])[0],
            backend_factory=lambda: FakeBackend([_world_result()]),
        )
    # Cache preserved, not clobbered.
    assert tuple(
        f.time_seconds for f in world_module.load_world_cache(cache).frames
    ) == other_times


def test_overwrite_replaces_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    identity = world_module.WorldCacheIdentity(
        source_fingerprint=metadata.fingerprint,
        model_name=mediapipe_module.MODEL_NAME,
        model_version=mediapipe_module.MODEL_VERSION,
    )
    world_module.write_complete_world_cache(
        cache,
        identity,
        [
            mediapipe_module.result_to_world_frame_observation(
                _world_result(), time_seconds=t
            )
            for t in (2.0, 2.5)
        ],
    )
    frame_factory, backend = _factory([_world_result() for _ in range(20)])
    result = world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=cache,
        probe_fn=lambda path: metadata,
        native_frame_factory=frame_factory,
        backend_factory=lambda: backend,
        overwrite=True,
    )
    assert result.cache_hit is False
    assert result.frame_count == len(NATIVE_TIMES)


def test_overwrite_and_no_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    video = tmp_path / "source.mov"
    video.write_bytes(b"x")
    with pytest.raises(world_extract_module.WorldExtractionInputError, match="mutually"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=tmp_path / "attempts.json",
            attempt_id="serve-001",
            cache_path=tmp_path / "world.jsonl",
            overwrite=True,
            no_resume=True,
        )


def test_cancellation_before_start_writes_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    with pytest.raises(world_extract_module.WorldExtractionCancelled):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=cache,
            probe_fn=lambda path: metadata,
            native_frame_factory=_factory([_world_result()])[0],
            backend_factory=lambda: FakeBackend([_world_result()]),
            is_cancelled=lambda: True,
        )
    assert not cache.exists()


def test_cancellation_midstream_leaves_resumable_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch)
    state = {"calls": 0}

    def _cancel_after_two() -> bool:
        state["calls"] += 1
        return state["calls"] > 3

    class _TwoThenCancel(FakeBackend):
        def infer_world(self, image, time_seconds):
            if len(self.calls) >= 2:
                raise world_extract_module.WorldExtractionCancelled("stop")
            return super().infer_world(image, time_seconds)

    with pytest.raises(world_extract_module.WorldExtractionCancelled):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=cache,
            probe_fn=lambda path: metadata,
            native_frame_factory=_factory([_world_result() for _ in range(20)])[0],
            backend_factory=lambda: _TwoThenCancel([_world_result() for _ in range(20)]),
            is_cancelled=_cancel_after_two,
        )
    snapshot = world_module.load_world_cache(cache)
    assert snapshot.complete is False
    assert len(snapshot.frames) == 2
    assert tuple(f.time_seconds for f in snapshot.frames) == tuple(NATIVE_TIMES[:2])


def test_inference_failure_preserves_partial_not_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    cache = tmp_path / "world.jsonl"
    _patch_native_times(monkeypatch, NATIVE_TIMES[:4])
    results = [_world_result(), _world_result(), RuntimeError("boom")]
    with pytest.raises(world_extract_module.WorldExtractionError, match="boom"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=cache,
            probe_fn=lambda path: metadata,
            native_frame_factory=_factory(results)[0],
            backend_factory=lambda: FakeBackend(results),
        )
    snapshot = world_module.load_world_cache(cache)
    assert snapshot.complete is False
    assert len(snapshot.frames) == 2


def test_backend_without_infer_world_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from serve_review.pose.mediapipe import MediaPipePoseBackend

    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    _patch_native_times(monkeypatch, NATIVE_TIMES[:1])

    class _SparseOnly:
        model_name = mediapipe_module.MODEL_NAME
        model_version = mediapipe_module.MODEL_VERSION

        def close(self) -> None:
            pass

    with pytest.raises(world_extract_module.WorldExtractionError, match="infer_world"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=tmp_path / "world.jsonl",
            probe_fn=lambda path: metadata,
            native_frame_factory=_factory([_world_result()])[0],
            backend_factory=lambda: _SparseOnly(),
        )
    assert MediaPipePoseBackend is not None  # sparse adapter untouched


def test_progress_callback_reports_done_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    _patch_native_times(monkeypatch, NATIVE_TIMES[:3])
    seen: list = []
    world_extract_module.extract_attempt_world(
        video,
        attempts_path=attempts_path,
        attempt_id="serve-001",
        cache_path=tmp_path / "world.jsonl",
        probe_fn=lambda path: metadata,
        native_frame_factory=_factory([_world_result() for _ in range(5)])[0],
        backend_factory=lambda: FakeBackend([_world_result() for _ in range(5)]),
        progress_callback=lambda done, total, obs: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_cache_path_colliding_with_inputs_is_rejected(tmp_path: Path) -> None:
    metadata = _metadata()
    video, attempts_path = _write_inputs(tmp_path, metadata, _document(metadata))
    with pytest.raises(world_extract_module.WorldExtractionInputError, match="collides"):
        world_extract_module.extract_attempt_world(
            video,
            attempts_path=attempts_path,
            attempt_id="serve-001",
            cache_path=attempts_path,
            probe_fn=lambda path: metadata,
        )
