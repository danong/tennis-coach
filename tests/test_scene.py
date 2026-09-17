from __future__ import annotations

import pytest

from serve_review.media.audio import AudioEnergy
from serve_review.pose.schema import (
    BodyKeypoint,
    FrameObservation,
    PersonBox,
    PersonObservation,
)
from serve_review.pose.world import (
    WorldCacheIdentity,
    WorldCacheSnapshot,
    WorldFrameObservation,
    WorldLandmark,
)
from serve_review.scene import (
    Point2D,
    Racket2D,
    SceneFrame,
    SceneTrack,
    build_scene_track,
    build_scene_track_with_racketvision_observations,
)
from serve_review.tracking.racketvision import RacketVisionFrameObservation


def _world_frame(time: float) -> WorldFrameObservation:
    box = PersonBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    person = PersonObservation(
        box=box,
        keypoints=tuple(BodyKeypoint(x=0.5, y=0.5, visibility=0.9) for _ in range(33)),
        score=0.9,
    )
    companion = FrameObservation(
        time_seconds=time,
        timestamp_ms=round(time * 1000),
        persons=(person,),
    )
    return WorldFrameObservation(
        time_seconds=time,
        timestamp_ms=round(time * 1000),
        world_landmarks=tuple(WorldLandmark(x=0.0, y=1.0, z=0.0) for _ in range(33)),
        frame_2d=companion,
    )


def _snapshot(*times: float) -> WorldCacheSnapshot:
    return WorldCacheSnapshot(
        identity=WorldCacheIdentity(
            source_fingerprint="sha256:test",
            model_name="test-model",
            model_version="1",
        ),
        frames=tuple(_world_frame(time) for time in times),
        complete=True,
    )


def test_build_scene_track_reuses_body_and_joins_aligned_audio() -> None:
    snapshot = _snapshot(0.0, 0.1)
    scene = build_scene_track(
        snapshot,
        audio=(AudioEnergy(0.0, 0.2), AudioEnergy(0.1, 0.8)),
        audio_transients=(False, True),
    )

    assert scene.source_fingerprint == "sha256:test"
    assert scene.available_modalities == frozenset({"body_2d", "body_3d", "audio"})
    assert scene.frames[0].body_2d is snapshot.frames[0].frame_2d
    assert scene.frames[1].body_3d == snapshot.frames[1].world_landmarks
    assert scene.frames[1].time_seconds == 0.1
    assert scene.frames[1].audio_energy == 0.8
    assert scene.frames[1].audio_transient is True


def test_missing_modalities_and_no_detection_are_distinct() -> None:
    frame = SceneFrame(time_seconds=0.0, body_2d=None, body_3d=None)
    unavailable = SceneTrack("sha256:test", (frame,), frozenset())
    available_no_detection = SceneTrack("sha256:test", (frame,), frozenset({"ball_2d"}))

    assert "ball_2d" not in unavailable.available_modalities
    assert "ball_2d" in available_no_detection.available_modalities
    assert available_no_detection.frames[0].ball_2d is None


def test_point_and_racket_validate_normalized_coordinates() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        Point2D(x=1.1, y=0.5, confidence=1.0)
    with pytest.raises(ValueError, match="ordered box"):
        Racket2D((0.8, 0.2, 0.2, 0.9), 0.9, {})


def test_scene_rejects_mismatched_body_timestamp_and_nonincreasing_pts() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        SceneFrame(time_seconds=0.2, body_2d=_world_frame(0.1).frame_2d, body_3d=None)
    with pytest.raises(ValueError, match="strictly increasing"):
        SceneTrack(
            "sha256:test",
            (SceneFrame(0.1, None, None), SceneFrame(0.1, None, None)),
            frozenset(),
        )


def test_builder_rejects_audio_timestamp_mismatch_and_length_mismatch() -> None:
    snapshot = _snapshot(0.0, 0.1)
    with pytest.raises(ValueError, match="exactly match"):
        build_scene_track(
            snapshot, audio=(AudioEnergy(0.0, 0.2), AudioEnergy(0.11, 0.8))
        )
    with pytest.raises(ValueError, match="exactly one"):
        build_scene_track(snapshot, audio=(AudioEnergy(0.0, 0.2),))


def test_typed_racketvision_observations_attach_without_losing_nulls() -> None:
    racket = Racket2D((0.1, 0.2, 0.5, 0.8), 0.7, {})
    scene = build_scene_track_with_racketvision_observations(
        _snapshot(1.0, 1.1),
        (
            RacketVisionFrameObservation(1.0, None, None),
            RacketVisionFrameObservation(1.1, Point2D(0.4, 0.5, 0.8), racket),
        ),
        audio=(AudioEnergy(1.0, 0.2), AudioEnergy(1.1, 0.3)),
    )
    assert scene.source_fingerprint == "sha256:test"
    assert scene.available_modalities == frozenset(
        {"body_2d", "body_3d", "ball_2d", "racket_2d", "audio"}
    )
    assert scene.frames[0].ball_2d is None
    assert scene.frames[0].racket_2d is None
    assert scene.frames[1].ball_2d == Point2D(0.4, 0.5, 0.8)
    assert scene.frames[1].racket_2d is racket
    assert scene.frames[1].audio_energy == 0.3


def test_typed_racketvision_observations_require_exact_rows_and_timestamps() -> None:
    observation = RacketVisionFrameObservation(0.0, None, None)
    with pytest.raises(ValueError, match="exactly one"):
        build_scene_track_with_racketvision_observations(_snapshot(0.0, 0.1), (observation,))
    with pytest.raises(ValueError, match="exactly match"):
        build_scene_track_with_racketvision_observations(
            _snapshot(0.0), (RacketVisionFrameObservation(0.00001, None, None),)
        )


def test_scene_rejects_invalid_audio_values() -> None:
    with pytest.raises(ValueError, match="audio_energy"):
        SceneFrame(0.0, None, None, audio_energy=-1.0)
    with pytest.raises(ValueError, match="audio_transient"):
        SceneFrame(0.0, None, None, audio_transient=1)  # type: ignore[arg-type]
