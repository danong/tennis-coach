"""Deterministic visual evidence derived from an MVP :class:`SceneTrack`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from serve_review.pose.schema import JOINT_INDEX
from serve_review.scene import Point2D, Racket2D, SceneFrame, SceneTrack

__all__ = [
    "CONTACT_DIAGNOSTIC_WINDOW_SECONDS",
    "DEFAULT_MIN_OBSERVATION_CONFIDENCE",
    "RELEASE_DIAGNOSTIC_WINDOW_SECONDS",
    "VisualEvidence",
    "ball_hand_separation_evidence",
    "ball_inside_racket_hoop_evidence",
    "ball_to_left_wrist_evidence",
    "ball_to_racket_hoop_evidence",
    "build_scene_visual_diagnostics",
]

RELEASE_DIAGNOSTIC_WINDOW_SECONDS = 0.50
CONTACT_DIAGNOSTIC_WINDOW_SECONDS = 0.25
DEFAULT_MIN_OBSERVATION_CONFIDENCE = 0.10
_LEFT_WRIST_INDEX = JOINT_INDEX["left_wrist"]
_HOOP_NAMES = ("Top", "Right", "Bottom", "Left")


@dataclass(frozen=True, slots=True)
class VisualEvidence:
    """One auditable visual candidate relative to a selected checkpoint."""

    metric: str
    available: bool
    confidence: float | None
    candidate_time_seconds: float | None
    selected_time_seconds: float | None
    delta_selected_minus_candidate_seconds: float | None
    value: float | None = None
    inside: bool | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "candidate_time_seconds": self.candidate_time_seconds,
            "confidence": self.confidence,
            "delta_selected_minus_candidate_seconds": self.delta_selected_minus_candidate_seconds,
            "inside": self.inside,
            "metric": self.metric,
            "reason": self.reason,
            "selected_time_seconds": self.selected_time_seconds,
            "value": self.value,
        }


def _unavailable(metric: str, selected: float | None, reason: str) -> VisualEvidence:
    return VisualEvidence(
        metric=metric,
        available=False,
        confidence=None,
        candidate_time_seconds=None,
        selected_time_seconds=selected,
        delta_selected_minus_candidate_seconds=None,
        reason=reason,
    )


def _selected_delta(selected: float | None, candidate: float) -> float | None:
    return None if selected is None else float(selected - candidate)


def _in_window(time: float, selected: float | None, radius: float) -> bool:
    return selected is not None and abs(time - selected) <= radius


def _left_wrist(frame: SceneFrame) -> Point2D | None:
    body = frame.body_2d
    if body is None or not body.persons:
        return None
    keypoint = body.persons[0].keypoints[_LEFT_WRIST_INDEX]
    if keypoint is None:
        return None
    return Point2D(keypoint.x, keypoint.y, keypoint.visibility)


def _distance(first: Point2D, second: Point2D) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _quality(*values: float) -> float:
    return min(values)


def _valid_point(point: Point2D | None, minimum: float) -> bool:
    return point is not None and point.confidence >= minimum


def ball_to_left_wrist_evidence(
    scene: SceneTrack,
    selected_time_seconds: float | None,
    *,
    window_seconds: float = RELEASE_DIAGNOSTIC_WINDOW_SECONDS,
    min_confidence: float = DEFAULT_MIN_OBSERVATION_CONFIDENCE,
) -> VisualEvidence:
    """Select the closest observed ball/wrist frame around release."""
    candidates: list[tuple[float, float, float]] = []
    for frame in scene.frames:
        if not _in_window(frame.time_seconds, selected_time_seconds, window_seconds):
            continue
        wrist = _left_wrist(frame)
        if not _valid_point(frame.ball_2d, min_confidence) or not _valid_point(
            wrist, min_confidence
        ):
            continue
        assert frame.ball_2d is not None and wrist is not None
        candidates.append(
            (
                _distance(frame.ball_2d, wrist),
                frame.time_seconds,
                _quality(frame.ball_2d.confidence, wrist.confidence),
            )
        )
    if not candidates:
        return _unavailable(
            "ball_to_left_wrist_distance",
            selected_time_seconds,
            "no_qualified_observations",
        )
    value, candidate, confidence = min(candidates, key=lambda item: (item[0], item[1]))
    return VisualEvidence(
        metric="ball_to_left_wrist_distance",
        available=True,
        confidence=confidence,
        candidate_time_seconds=candidate,
        selected_time_seconds=selected_time_seconds,
        delta_selected_minus_candidate_seconds=_selected_delta(
            selected_time_seconds, candidate
        ),
        value=value,
    )


def ball_hand_separation_evidence(
    scene: SceneTrack,
    selected_time_seconds: float | None,
    *,
    window_seconds: float = RELEASE_DIAGNOSTIC_WINDOW_SECONDS,
    min_confidence: float = DEFAULT_MIN_OBSERVATION_CONFIDENCE,
) -> VisualEvidence:
    """Select the largest observed frame-to-frame ball/wrist distance increase."""
    previous: tuple[float, float] | None = None
    candidates: list[tuple[float, float, float]] = []
    for frame in scene.frames:
        wrist = _left_wrist(frame)
        if not _in_window(frame.time_seconds, selected_time_seconds, window_seconds):
            previous = None
            continue
        if not _valid_point(frame.ball_2d, min_confidence) or not _valid_point(
            wrist, min_confidence
        ):
            previous = None
            continue
        assert frame.ball_2d is not None and wrist is not None
        distance = _distance(frame.ball_2d, wrist)
        confidence = _quality(frame.ball_2d.confidence, wrist.confidence)
        if previous is not None:
            prior_distance, prior_confidence = previous
            candidates.append(
                (
                    distance - prior_distance,
                    frame.time_seconds,
                    min(confidence, prior_confidence),
                )
            )
        previous = (distance, confidence)
    if not candidates:
        return _unavailable(
            "ball_hand_separation",
            selected_time_seconds,
            "no_qualified_consecutive_observations",
        )
    value, candidate, confidence = max(candidates, key=lambda item: (item[0], -item[1]))
    return VisualEvidence(
        metric="ball_hand_separation",
        available=True,
        confidence=confidence,
        candidate_time_seconds=candidate,
        selected_time_seconds=selected_time_seconds,
        delta_selected_minus_candidate_seconds=_selected_delta(
            selected_time_seconds, candidate
        ),
        value=value,
    )


def _hoop_points(racket: Racket2D, minimum: float) -> tuple[Point2D, ...] | None:
    points = tuple(racket.keypoints.get(name) for name in _HOOP_NAMES)
    if any(not _valid_point(point, minimum) for point in points):
        return None
    return tuple(point for point in points if point is not None)


def _hoop_center(points: tuple[Point2D, ...]) -> Point2D:
    return Point2D(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
        min(point.confidence for point in points),
    )


def _inside_convex_quad(point: Point2D, polygon: tuple[Point2D, ...]) -> bool:
    signs = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (second.x - first.x) * (point.y - first.y) - (second.y - first.y) * (
            point.x - first.x
        )
        signs.append(cross)
    return all(value >= -1e-12 for value in signs) or all(
        value <= 1e-12 for value in signs
    )


def _hoop_observations(
    scene: SceneTrack,
    selected_time_seconds: float | None,
    window_seconds: float,
    min_confidence: float,
) -> list[tuple[float, float, bool, float]]:
    observations = []
    for frame in scene.frames:
        if not _in_window(frame.time_seconds, selected_time_seconds, window_seconds):
            continue
        if not _valid_point(frame.ball_2d, min_confidence) or frame.racket_2d is None:
            continue
        points = _hoop_points(frame.racket_2d, min_confidence)
        if points is None:
            continue
        assert frame.ball_2d is not None
        center = _hoop_center(points)
        confidence = _quality(
            frame.ball_2d.confidence, *(point.confidence for point in points)
        )
        observations.append(
            (
                frame.time_seconds,
                _distance(frame.ball_2d, center),
                _inside_convex_quad(frame.ball_2d, points),
                confidence,
            )
        )
    return observations


def ball_to_racket_hoop_evidence(
    scene: SceneTrack,
    selected_time_seconds: float | None,
    *,
    window_seconds: float = CONTACT_DIAGNOSTIC_WINDOW_SECONDS,
    min_confidence: float = DEFAULT_MIN_OBSERVATION_CONFIDENCE,
) -> VisualEvidence:
    """Select the closest observed ball-to-hoop-center frame around contact."""
    observations = _hoop_observations(
        scene, selected_time_seconds, window_seconds, min_confidence
    )
    if not observations:
        return _unavailable(
            "ball_to_racket_hoop_distance",
            selected_time_seconds,
            "no_qualified_observations",
        )
    candidate, value, inside, confidence = min(
        observations, key=lambda item: (item[1], item[0])
    )
    return VisualEvidence(
        metric="ball_to_racket_hoop_distance",
        available=True,
        confidence=confidence,
        candidate_time_seconds=candidate,
        selected_time_seconds=selected_time_seconds,
        delta_selected_minus_candidate_seconds=_selected_delta(
            selected_time_seconds, candidate
        ),
        value=value,
        inside=inside,
    )


def ball_inside_racket_hoop_evidence(
    scene: SceneTrack,
    selected_time_seconds: float | None,
    *,
    window_seconds: float = CONTACT_DIAGNOSTIC_WINDOW_SECONDS,
    min_confidence: float = DEFAULT_MIN_OBSERVATION_CONFIDENCE,
) -> VisualEvidence:
    """Select the earliest qualified frame whose ball lies inside the hoop."""
    observations = _hoop_observations(
        scene, selected_time_seconds, window_seconds, min_confidence
    )
    inside = [item for item in observations if item[2]]
    if not inside:
        return _unavailable(
            "ball_inside_racket_hoop",
            selected_time_seconds,
            "no_qualified_inside_observation",
        )
    candidate, value, is_inside, confidence = min(inside, key=lambda item: item[0])
    return VisualEvidence(
        metric="ball_inside_racket_hoop",
        available=True,
        confidence=confidence,
        candidate_time_seconds=candidate,
        selected_time_seconds=selected_time_seconds,
        delta_selected_minus_candidate_seconds=_selected_delta(
            selected_time_seconds, candidate
        ),
        value=value,
        inside=is_inside,
    )


def build_scene_visual_diagnostics(
    scene: SceneTrack,
    *,
    selected_release_time_seconds: float | None,
    selected_contact_time_seconds: float | None,
) -> dict[str, Any]:
    """Build deterministic release/contact visual diagnostics."""
    evidence = (
        ball_to_left_wrist_evidence(scene, selected_release_time_seconds),
        ball_hand_separation_evidence(scene, selected_release_time_seconds),
        ball_to_racket_hoop_evidence(scene, selected_contact_time_seconds),
        ball_inside_racket_hoop_evidence(scene, selected_contact_time_seconds),
    )
    return {
        "config": {
            "contact_window_seconds": CONTACT_DIAGNOSTIC_WINDOW_SECONDS,
            "min_observation_confidence": DEFAULT_MIN_OBSERVATION_CONFIDENCE,
            "release_window_seconds": RELEASE_DIAGNOSTIC_WINDOW_SECONDS,
        },
        "release": {
            "selected_time_seconds": selected_release_time_seconds,
            "ball_to_left_wrist": evidence[0].to_dict(),
            "ball_hand_separation": evidence[1].to_dict(),
        },
        "contact": {
            "selected_time_seconds": selected_contact_time_seconds,
            "ball_to_racket_hoop": evidence[2].to_dict(),
            "ball_inside_racket_hoop": evidence[3].to_dict(),
        },
    }
