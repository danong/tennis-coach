from __future__ import annotations

from dataclasses import replace

import pytest

from serve_review.pose.schema import (
    BodyKeypoint,
    FrameObservation,
    PersonBox,
    PersonObservation,
)
from serve_review.scene import Point2D, Racket2D, SceneFrame, SceneTrack
from serve_review.scene_diagnostics import (
    ball_hand_separation_evidence,
    ball_inside_racket_hoop_evidence,
    ball_to_left_wrist_evidence,
    ball_to_racket_hoop_evidence,
    build_scene_visual_diagnostics,
)


def _frame(
    time: float,
    *,
    ball: Point2D | None = None,
    racket: Racket2D | None = None,
    wrist: tuple[float, float, float] | None = (0.2, 0.2, 0.9),
) -> SceneFrame:
    base = SceneFrame(time, None, None, ball_2d=ball, racket_2d=racket)
    if wrist is None:
        return base
    # Use a small synthetic one-person observation with a controllable left wrist.
    box = PersonBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
    keypoints = [BodyKeypoint(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    x, y, confidence = wrist
    keypoints[15] = BodyKeypoint(x=x, y=y, visibility=confidence)
    person = PersonObservation(box=box, keypoints=tuple(keypoints), score=0.9)
    companion = FrameObservation(
        time_seconds=time,
        timestamp_ms=round(time * 1000),
        persons=(person,),
    )
    return replace(base, body_2d=companion)


def _scene(frames: tuple[SceneFrame, ...]) -> SceneTrack:
    return SceneTrack(
        "sha256:diagnostics", frames, frozenset({"body_2d", "ball_2d", "racket_2d"})
    )


def _racket() -> Racket2D:
    return Racket2D(
        bbox=(0.35, 0.35, 0.65, 0.65),
        bbox_confidence=0.9,
        keypoints={
            "Top": Point2D(0.5, 0.4, 0.9),
            "Right": Point2D(0.6, 0.5, 0.9),
            "Bottom": Point2D(0.5, 0.6, 0.9),
            "Left": Point2D(0.4, 0.5, 0.9),
        },
    )


def test_release_evidence_selects_distance_and_separation_frames() -> None:
    scene = _scene(
        (
            _frame(1.0, ball=Point2D(0.21, 0.2, 0.9)),
            _frame(1.1, ball=Point2D(0.3, 0.2, 0.9)),
            _frame(1.2, ball=Point2D(0.7, 0.2, 0.9)),
        )
    )
    distance = ball_to_left_wrist_evidence(scene, 1.15)
    separation = ball_hand_separation_evidence(scene, 1.15)

    assert distance.available is True
    assert distance.candidate_time_seconds == 1.0
    assert distance.delta_selected_minus_candidate_seconds == pytest.approx(0.15)
    assert separation.available is True
    assert separation.candidate_time_seconds == 1.2
    assert separation.value == pytest.approx(0.4)


def test_contact_evidence_selects_distance_and_inside_frames() -> None:
    scene = _scene(
        (
            _frame(2.0, ball=Point2D(0.5, 0.5, 0.9), racket=_racket()),
            _frame(2.1, ball=Point2D(0.8, 0.8, 0.9), racket=_racket()),
        )
    )
    distance = ball_to_racket_hoop_evidence(scene, 2.05)
    inside = ball_inside_racket_hoop_evidence(scene, 2.05)

    assert distance.available is True
    assert distance.candidate_time_seconds == 2.0
    assert distance.inside is True
    assert inside.available is True
    assert inside.candidate_time_seconds == 2.0
    assert inside.inside is True


def test_low_confidence_and_missing_visuals_do_not_fabricate_evidence() -> None:
    scene = _scene(
        (
            _frame(1.0, ball=Point2D(0.3, 0.3, 0.05), wrist=(0.2, 0.2, 0.9)),
            _frame(2.0, ball=None, racket=None, wrist=None),
        )
    )
    release = ball_to_left_wrist_evidence(scene, 1.0)
    contact = ball_to_racket_hoop_evidence(scene, 2.0)

    assert release.available is False
    assert release.candidate_time_seconds is None
    assert release.reason == "no_qualified_observations"
    assert contact.available is False
    assert contact.candidate_time_seconds is None


def test_visual_diagnostics_are_deterministic_and_auditable() -> None:
    scene = _scene((_frame(1.0, ball=Point2D(0.21, 0.2, 0.9)),))
    first = build_scene_visual_diagnostics(
        scene, selected_release_time_seconds=1.0, selected_contact_time_seconds=2.0
    )
    second = build_scene_visual_diagnostics(
        scene, selected_release_time_seconds=1.0, selected_contact_time_seconds=2.0
    )

    assert first == second
    assert first["config"]["release_window_seconds"] == 0.5
    assert first["release"]["ball_to_left_wrist"]["candidate_time_seconds"] == 1.0
    assert first["release"]["ball_to_left_wrist"][
        "delta_selected_minus_candidate_seconds"
    ] == pytest.approx(0.0)
    assert first["contact"]["ball_to_racket_hoop"]["available"] is False


def test_contact_track_persists_frame_offset_and_post_contact_separation() -> None:
    scene = _scene((
        _frame(2.0, ball=Point2D(0.5, 0.5, 0.9), racket=_racket()),
        _frame(2.1, ball=Point2D(0.8, 0.8, 0.9), racket=_racket()),
    ))

    result = build_scene_visual_diagnostics(
        scene, selected_release_time_seconds=None, selected_contact_time_seconds=2.0
    )
    track = result["contact"]["ball_racket_hoop_track"]
    nearest = track["nearest_qualified_observation"]

    assert track["available"] is True
    assert nearest["frame_index"] == 0
    assert nearest["frames_from_contact"] == 0
    assert nearest["seconds_from_contact"] == pytest.approx(0.0)
    assert nearest["confidence"] == pytest.approx(0.9)
    assert track["post_contact_observations"][0]["frames_from_contact"] == 1
    assert track["has_post_contact_distance_increase"] is True


def test_contact_track_keeps_weak_raw_observation_separate_from_qualified() -> None:
    weak_racket = Racket2D(
        bbox=(0.35, 0.35, 0.65, 0.65),
        bbox_confidence=0.9,
        keypoints={
            name: Point2D(point.x, point.y, 0.05)
            for name, point in _racket().keypoints.items()
        },
    )
    scene = _scene((
        _frame(2.0, ball=Point2D(0.5, 0.5, 0.9), racket=weak_racket),
    ))

    result = build_scene_visual_diagnostics(
        scene, selected_release_time_seconds=None, selected_contact_time_seconds=2.0
    )
    track = result["contact"]["ball_racket_hoop_track"]

    assert track["available"] is False
    assert track["nearest_qualified_observation"] is None
    assert track["nearest_raw_observation"]["qualified"] is False
    assert track["nearest_raw_observation"]["confidence"] == pytest.approx(0.05)
