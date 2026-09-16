from __future__ import annotations

import csv

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
    build_scene_track_with_racketvision,
)


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


def _write_racketvision_csv(path, rows, *, smoothed=False) -> None:
    fields = ["Frame", "X", "Y", "Visibility", "Confidence"]
    if smoothed:
        fields += ["SmoothX", "SmoothY"]
    fields += ["BBox1", "BBox2", "BBox3", "BBox4", "BBoxConfidence"]
    fields += [
        f"{name}{axis}"
        for name in ("Top", "Bottom", "Handle", "Left", "Right")
        for axis in ("X", "Y")
    ]
    fields += [
        f"{name}Confidence" for name in ("Top", "Bottom", "Handle", "Left", "Right")
    ]
    if smoothed:
        fields += [f"SmoothBBox{i}" for i in range(1, 5)]
        fields += [
            f"Smooth{name}{axis}"
            for name in ("Top", "Bottom", "Handle", "Left", "Right")
            for axis in ("X", "Y")
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_racketvision_adapter_joins_raw_and_smoothed_rows_by_scene_order(
    tmp_path,
) -> None:
    rows = [
        {"Frame": 0, "X": 20, "Y": 30, "Visibility": 0, "Confidence": 0},
        {
            "Frame": 1,
            "X": 40,
            "Y": 50,
            "Visibility": 1,
            "Confidence": 0.8,
            "BBox1": 10,
            "BBox2": 20,
            "BBox3": 50,
            "BBox4": 80,
            "BBoxConfidence": 0.7,
            "TopX": 20,
            "TopY": 20,
            "TopConfidence": 0.6,
        },
    ]
    raw_path = tmp_path / "raw.csv"
    _write_racketvision_csv(raw_path, rows)
    scene = build_scene_track_with_racketvision(
        _snapshot(2.5, 2.6), raw_path, source_width=100, source_height=100
    )
    assert [frame.time_seconds for frame in scene.frames] == [2.5, 2.6]
    assert scene.frames[0].ball_2d is None
    assert scene.frames[1].ball_2d == Point2D(0.4, 0.5, 0.8)
    assert scene.frames[1].racket_2d is not None
    assert scene.frames[1].racket_2d.bbox == (0.1, 0.2, 0.5, 0.8)
    assert scene.frames[1].racket_2d.keypoints["Top"] == Point2D(0.2, 0.2, 0.6)

    smooth_rows = [{"Frame": 0, "SmoothX": 60, "SmoothY": 70}]
    smooth_path = tmp_path / "smooth.csv"
    _write_racketvision_csv(smooth_path, smooth_rows, smoothed=True)
    smoothed = build_scene_track_with_racketvision(
        _snapshot(9.0), smooth_path, source_width=100, source_height=100, smoothed=True
    )
    assert smoothed.frames[0].time_seconds == 9.0
    assert smoothed.frames[0].ball_2d == Point2D(0.6, 0.7, 0.0)


def test_racketvision_adapter_rejects_mismatch_malformed_data_and_crop(
    tmp_path,
) -> None:
    path = tmp_path / "bad.csv"
    _write_racketvision_csv(path, [{"Frame": 0, "X": 10, "Y": "bad", "Visibility": 1}])
    with pytest.raises(ValueError, match="not numeric"):
        build_scene_track_with_racketvision(
            _snapshot(0.0), path, source_width=100, source_height=100
        )
    with pytest.raises(ValueError, match="expected 2"):
        build_scene_track_with_racketvision(
            _snapshot(0.0, 0.1), path, source_width=100, source_height=100
        )
    with pytest.raises(ValueError, match="crop_top.*unsupported"):
        build_scene_track_with_racketvision(
            _snapshot(0.0), path, source_width=100, source_height=100, crop_top=4
        )


def test_scene_rejects_invalid_audio_values() -> None:
    with pytest.raises(ValueError, match="audio_energy"):
        SceneFrame(0.0, None, None, audio_energy=-1.0)
    with pytest.raises(ValueError, match="audio_transient"):
        SceneFrame(0.0, None, None, audio_transient=1)  # type: ignore[arg-type]
