"""PTS-preserving image-space features derived from SceneTrack observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from serve_review.pose.schema import JOINT_INDEX
from serve_review.scene import SceneFrame, SceneTrack

__all__ = ["SceneFeatureSeries", "build_scene_feature_series"]

_LEFT_WRIST_INDEX = JOINT_INDEX["left_wrist"]
_HOOP_NAMES = ("Top", "Right", "Bottom", "Left")


@dataclass(frozen=True, slots=True)
class SceneFeatureSeries:
    """Observation-level image-space features on SceneTrack's exact PTS grid."""

    time_seconds: tuple[float, ...]
    racket_handle_hoop_vertical_orientation: tuple[float | None, ...]
    left_wrist_ball_distance: tuple[float | None, ...]
    racket_hoop_ball_distance: tuple[float | None, ...]


def _left_wrist_xy(frame: SceneFrame) -> tuple[float, float] | None:
    body = frame.body_2d
    if body is None or not body.persons:
        return None
    point = body.persons[0].keypoints[_LEFT_WRIST_INDEX]
    if point is None:
        return None
    return (float(point.x), float(point.y))


def _hoop_center_xy(frame: SceneFrame) -> tuple[float, float] | None:
    racket = frame.racket_2d
    if racket is None:
        return None
    points = [racket.keypoints.get(name) for name in _HOOP_NAMES]
    available = [point for point in points if point is not None]
    if not available:
        return None
    return (
        sum(point.x for point in available) / len(available),
        sum(point.y for point in available) / len(available),
    )


def build_scene_feature_series(scene_track: SceneTrack) -> SceneFeatureSeries:
    """Derive unweighted ball/racket features on SceneTrack's exact PTS grid.

    ``racket_handle_hoop_vertical_orientation`` is the signed image-space
    vertical component of the unit vector from handle to hoop center. With
    upright image coordinates, ``1`` means the handle is directly above the
    hoop, ``0`` is horizontal, and ``-1`` means it is below. Distances are
    normalized-image Euclidean distances. Missing source observations stay
    ``None`` independently per feature.
    """
    if not isinstance(scene_track, SceneTrack):
        raise TypeError(
            "build_scene_feature_series: 'scene_track' must be a SceneTrack, "
            f"got {type(scene_track).__name__}."
        )

    orientation: list[float | None] = []
    hand_ball: list[float | None] = []
    racket_ball: list[float | None] = []
    for frame in scene_track.frames:
        wrist = _left_wrist_xy(frame)
        hoop = _hoop_center_xy(frame)
        ball = frame.ball_2d
        handle = (
            frame.racket_2d.keypoints.get("Handle")
            if frame.racket_2d is not None
            else None
        )

        if handle is None or hoop is None:
            orientation.append(None)
        else:
            dx = hoop[0] - handle.x
            dy = hoop[1] - handle.y
            length = math.hypot(dx, dy)
            orientation.append(dy / length if length > 0.0 else None)

        hand_ball.append(
            math.hypot(ball.x - wrist[0], ball.y - wrist[1])
            if ball is not None and wrist is not None
            else None
        )
        racket_ball.append(
            math.hypot(ball.x - hoop[0], ball.y - hoop[1])
            if ball is not None and hoop is not None
            else None
        )

    return SceneFeatureSeries(
        time_seconds=tuple(frame.time_seconds for frame in scene_track.frames),
        racket_handle_hoop_vertical_orientation=tuple(orientation),
        left_wrist_ball_distance=tuple(hand_ball),
        racket_hoop_ball_distance=tuple(racket_ball),
    )
