"""Minimal timestamped multimodal scene representation.

This module is an in-memory compatibility view over existing artifacts.  It
keeps the dense-world cache as the source of the native video timeline and
joins optional audio observations without introducing a new cache format.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from serve_review.media.audio import AudioEnergy
from serve_review.pose.schema import FrameObservation
from serve_review.pose.world import (
    NUM_WORLD_LANDMARKS,
    WorldCacheSnapshot,
    WorldLandmark,
)

__all__ = [
    "SCENE_MODALITIES",
    "Point2D",
    "Racket2D",
    "SceneFrame",
    "SceneTrack",
    "build_scene_track",
    "build_scene_track_with_racketvision",
    "build_scene_track_with_racketvision_observations",
    "load_racketvision_csv",
]

SCENE_MODALITIES = frozenset({"body_2d", "body_3d", "ball_2d", "racket_2d", "audio"})


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return result


def _confidence(name: str, value: object) -> float:
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    return result


@dataclass(frozen=True, slots=True)
class Point2D:
    """A point in normalized, upright, unmirrored image coordinates."""

    x: float
    y: float
    confidence: float

    def __post_init__(self) -> None:
        x = _finite("Point2D.x", self.x)
        y = _finite("Point2D.y", self.y)
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"Point2D coordinates must be in [0, 1], got {(x, y)!r}.")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(
            self, "confidence", _confidence("Point2D.confidence", self.confidence)
        )


@dataclass(frozen=True, slots=True)
class Racket2D:
    """One racket detection in normalized image coordinates."""

    bbox: tuple[float, float, float, float]
    bbox_confidence: float
    keypoints: Mapping[str, Point2D | None]

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, (tuple, list)) or len(self.bbox) != 4:
            raise ValueError("Racket2D.bbox must contain four coordinates.")
        bbox = tuple(
            _finite(f"Racket2D.bbox[{i}]", value) for i, value in enumerate(self.bbox)
        )
        x1, y1, x2, y2 = bbox
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(
                f"Racket2D.bbox must be an ordered box in [0, 1], got {bbox!r}."
            )
        if not isinstance(self.keypoints, Mapping):
            raise TypeError("Racket2D.keypoints must be a mapping.")
        if any(not isinstance(name, str) or not name for name in self.keypoints):
            raise ValueError("Racket2D.keypoints names must be non-empty strings.")
        if any(
            point is not None and not isinstance(point, Point2D)
            for point in self.keypoints.values()
        ):
            raise ValueError("Racket2D.keypoints values must be Point2D or None.")
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(
            self,
            "bbox_confidence",
            _confidence("Racket2D.bbox_confidence", self.bbox_confidence),
        )


@dataclass(frozen=True, slots=True)
class SceneFrame:
    """One native video frame, with optional observations from each modality."""

    time_seconds: float
    body_2d: FrameObservation | None
    body_3d: tuple[WorldLandmark | None, ...] | None
    ball_2d: Point2D | None = None
    racket_2d: Racket2D | None = None
    audio_energy: float | None = None
    audio_transient: bool | None = None

    def __post_init__(self) -> None:
        time = _finite("SceneFrame.time_seconds", self.time_seconds)
        if time < 0.0:
            raise ValueError("SceneFrame.time_seconds must be >= 0.")
        object.__setattr__(self, "time_seconds", time)
        if self.body_2d is not None and not isinstance(self.body_2d, FrameObservation):
            raise ValueError("SceneFrame.body_2d must be FrameObservation or None.")
        if self.body_2d is not None and self.body_2d.time_seconds != time:
            raise ValueError("SceneFrame.body_2d timestamp must match SceneFrame PTS.")
        if self.body_3d is not None:
            if (
                not isinstance(self.body_3d, (tuple, list))
                or len(self.body_3d) != NUM_WORLD_LANDMARKS
            ):
                raise ValueError(
                    f"SceneFrame.body_3d must contain {NUM_WORLD_LANDMARKS} joints."
                )
            if any(
                joint is not None and not isinstance(joint, WorldLandmark)
                for joint in self.body_3d
            ):
                raise ValueError(
                    "SceneFrame.body_3d joints must be WorldLandmark or None."
                )
            object.__setattr__(self, "body_3d", tuple(self.body_3d))
        if self.ball_2d is not None and not isinstance(self.ball_2d, Point2D):
            raise ValueError("SceneFrame.ball_2d must be Point2D or None.")
        if self.racket_2d is not None and not isinstance(self.racket_2d, Racket2D):
            raise ValueError("SceneFrame.racket_2d must be Racket2D or None.")
        if self.audio_energy is not None:
            energy = _finite("SceneFrame.audio_energy", self.audio_energy)
            if energy < 0.0:
                raise ValueError("SceneFrame.audio_energy must be >= 0.")
            object.__setattr__(self, "audio_energy", energy)
        if self.audio_transient is not None and not isinstance(
            self.audio_transient, bool
        ):
            raise ValueError("SceneFrame.audio_transient must be bool or None.")


@dataclass(frozen=True, slots=True)
class SceneTrack:
    """Strict native-frame scene sequence; ``time_seconds`` is its only clock."""

    source_fingerprint: str
    frames: tuple[SceneFrame, ...]
    available_modalities: frozenset[str]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_fingerprint, str)
            or not self.source_fingerprint.strip()
        ):
            raise ValueError("SceneTrack.source_fingerprint must be non-blank.")
        if not isinstance(self.frames, (tuple, list)):
            raise TypeError("SceneTrack.frames must be a sequence.")
        frames = tuple(self.frames)
        if any(not isinstance(frame, SceneFrame) for frame in frames):
            raise ValueError("SceneTrack.frames must contain SceneFrame values.")
        for earlier, later in pairwise(frames):
            if not later.time_seconds > earlier.time_seconds:
                raise ValueError("SceneTrack frame times must be strictly increasing.")
        modalities = frozenset(self.available_modalities)
        if not modalities <= SCENE_MODALITIES:
            raise ValueError(
                f"Unknown scene modalities: {sorted(modalities - SCENE_MODALITIES)!r}."
            )
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "available_modalities", modalities)


def _csv_number(row: Mapping[str, str], column: str, *, required: bool) -> float | None:
    raw = row.get(column, "")
    if raw is None or not str(raw).strip():
        if required:
            raise ValueError(f"RacketVision CSV column {column!r} is missing a value.")
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"RacketVision CSV column {column!r} is not numeric: {raw!r}."
        ) from exc
    if not math.isfinite(value):
        if required:
            raise ValueError(f"RacketVision CSV column {column!r} is not finite.")
        return None
    return value


def _pixel_point(
    row: Mapping[str, str],
    x_column: str,
    y_column: str,
    confidence_column: str,
    width: float,
    height: float,
    *,
    required: bool,
) -> Point2D | None:
    x = _csv_number(row, x_column, required=required)
    y = _csv_number(row, y_column, required=required)
    confidence = _csv_number(row, confidence_column, required=False)
    if x is None or y is None:
        if required:
            raise ValueError(f"RacketVision point {x_column}/{y_column} is incomplete.")
        return None
    if confidence is None:
        confidence = 0.0
    return Point2D(x=x / width, y=y / height, confidence=confidence)


def load_racketvision_csv(
    csv_path: str | Path,
    scene_frames: Sequence[SceneFrame],
    *,
    source_width: float,
    source_height: float,
    smoothed: bool = False,
    crop_top: int = 0,
) -> tuple[SceneFrame, ...]:
    """Attach one raw or smoothed RacketVision CSV to native scene frames.

    CSV row order is the only join key.  PTS is copied from ``scene_frames``;
    this function never uses the CSV's frame number or an FPS estimate.
    """
    if crop_top != 0:
        raise ValueError(
            "RacketVision CSV with crop_top is unsupported: the CSV lacks a "
            "source-coordinate transform."
        )
    width = _finite("source_width", source_width)
    height = _finite("source_height", source_height)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("source_width and source_height must be positive.")
    if not isinstance(scene_frames, (tuple, list)):
        raise TypeError("scene_frames must be a list or tuple of SceneFrame values.")
    base = tuple(scene_frames)
    if any(not isinstance(frame, SceneFrame) for frame in base):
        raise ValueError("scene_frames must contain SceneFrame values.")
    path = Path(csv_path)
    try:
        handle = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not open RacketVision CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Frame" not in reader.fieldnames:
            raise ValueError("RacketVision CSV must contain a 'Frame' column.")
        rows = list(reader)
    if len(rows) != len(base):
        raise ValueError(
            f"RacketVision CSV has {len(rows)} rows, expected {len(base)} scene frames."
        )

    prefix = "Smooth" if smoothed else ""
    keypoint_names = ("Top", "Bottom", "Handle", "Left", "Right")
    result: list[SceneFrame] = []
    for index, (row, frame) in enumerate(zip(rows, base)):
        frame_number = _csv_number(row, "Frame", required=True)
        if frame_number != index or not frame_number.is_integer():
            raise ValueError(
                f"RacketVision CSV Frame at row {index} must be {index}, got {frame_number!r}."
            )
        ball = _pixel_point(
            row,
            f"{prefix}X" if smoothed else "X",
            f"{prefix}Y" if smoothed else "Y",
            "Confidence",
            width,
            height,
            required=(
                (_csv_number(row, "Visibility", required=False) or 0.0) > 0.0
                if not smoothed
                else _csv_number(row, "SmoothX", required=False) is not None
                and _csv_number(row, "SmoothY", required=False) is not None
            ),
        )
        if (
            not smoothed
            and (_csv_number(row, "Visibility", required=False) or 0.0) <= 0.0
        ):
            ball = None
        racket_prefix = "Smooth" if smoothed else ""
        bbox_columns = tuple(f"{racket_prefix}BBox{i}" for i in range(1, 5))
        bbox_values = [
            _csv_number(row, column, required=False) for column in bbox_columns
        ]
        bbox_confidence = _csv_number(row, "BBoxConfidence", required=False) or 0.0
        racket_present = all(value is not None for value in bbox_values)
        if (
            bbox_confidence > 0.0 or any(value is not None for value in bbox_values)
        ) and not racket_present:
            raise ValueError(f"RacketVision racket box is incomplete at row {index}.")
        racket = None
        if racket_present:
            bbox = tuple(
                value / (width if column_index % 2 == 0 else height)
                for column_index, value in enumerate(bbox_values)
            )
            keypoints = {}
            for name in keypoint_names:
                confidence = _csv_number(row, f"{name}Confidence", required=False)
                keypoints[name] = _pixel_point(
                    row,
                    f"{racket_prefix}{name}X",
                    f"{racket_prefix}{name}Y",
                    f"{name}Confidence",
                    width,
                    height,
                    required=(confidence or 0.0) > 0.0,
                )
            racket = Racket2D(
                bbox=bbox, bbox_confidence=bbox_confidence, keypoints=keypoints
            )
        result.append(
            SceneFrame(
                time_seconds=frame.time_seconds,
                body_2d=frame.body_2d,
                body_3d=frame.body_3d,
                ball_2d=ball,
                racket_2d=racket,
                audio_energy=frame.audio_energy,
                audio_transient=frame.audio_transient,
            )
        )
    return tuple(result)


def build_scene_track_with_racketvision(
    world_snapshot: WorldCacheSnapshot,
    racketvision_csv: str | Path,
    *,
    source_width: float,
    source_height: float,
    smoothed: bool = False,
    crop_top: int = 0,
    audio: Sequence[AudioEnergy] | None = None,
    audio_transients: Sequence[bool | None] | None = None,
) -> SceneTrack:
    """Build a scene and attach a strictly frame-aligned RacketVision CSV."""
    scene = build_scene_track(
        world_snapshot, audio=audio, audio_transients=audio_transients
    )
    frames = load_racketvision_csv(
        racketvision_csv,
        scene.frames,
        source_width=source_width,
        source_height=source_height,
        smoothed=smoothed,
        crop_top=crop_top,
    )
    return SceneTrack(
        source_fingerprint=scene.source_fingerprint,
        frames=frames,
        available_modalities=scene.available_modalities
        | frozenset({"ball_2d", "racket_2d"}),
    )


def build_scene_track_with_racketvision_observations(
    world_snapshot: WorldCacheSnapshot,
    observations: Sequence[object],
    *,
    audio: Sequence[AudioEnergy] | None = None,
    audio_transients: Sequence[bool | None] | None = None,
) -> SceneTrack:
    """Build a scene with one typed RacketVision observation per world frame.

    Observations are joined strictly by sequence position.  Their timestamps
    must be exactly equal to the native world-frame timestamps; no FPS or
    tolerance-based alignment is performed.
    """
    # Keep this import local: racketvision uses Point2D and Racket2D from this
    # module, so importing it at module load time would create a cycle.
    from serve_review.tracking.racketvision import RacketVisionFrameObservation

    if not isinstance(observations, (tuple, list)):
        raise TypeError("observations must be a list or tuple of typed values.")
    scene = build_scene_track(
        world_snapshot, audio=audio, audio_transients=audio_transients
    )
    if len(observations) != len(scene.frames):
        raise ValueError(
            "RacketVision observations must contain exactly one value per world frame."
        )
    result: list[SceneFrame] = []
    for index, (frame, observation) in enumerate(zip(scene.frames, observations)):
        if not isinstance(observation, RacketVisionFrameObservation):
            raise TypeError(
                "observations must contain RacketVisionFrameObservation values."
            )
        if observation.time_seconds != frame.time_seconds:
            raise ValueError(
                f"RacketVision observation timestamp at row {index} must exactly "
                "match the world frame timestamp."
            )
        result.append(
            SceneFrame(
                time_seconds=frame.time_seconds,
                body_2d=frame.body_2d,
                body_3d=frame.body_3d,
                ball_2d=observation.ball_2d,
                racket_2d=observation.racket_2d,
                audio_energy=frame.audio_energy,
                audio_transient=frame.audio_transient,
            )
        )
    return SceneTrack(
        source_fingerprint=scene.source_fingerprint,
        frames=tuple(result),
        available_modalities=scene.available_modalities
        | frozenset({"ball_2d", "racket_2d"}),
    )


def build_scene_track(
    world_snapshot: WorldCacheSnapshot,
    *,
    audio: Sequence[AudioEnergy] | None = None,
    audio_transients: Sequence[bool | None] | None = None,
) -> SceneTrack:
    """Join existing dense-world frames and already-aligned audio by PTS."""
    if not isinstance(world_snapshot, WorldCacheSnapshot):
        raise TypeError("world_snapshot must be a WorldCacheSnapshot.")
    frames = world_snapshot.frames
    if audio is not None:
        if not isinstance(audio, (tuple, list)):
            raise TypeError("audio must be a list or tuple of AudioEnergy values.")
        if len(audio) != len(frames) or any(
            not isinstance(item, AudioEnergy) for item in audio
        ):
            raise ValueError(
                "audio must contain exactly one AudioEnergy per scene frame."
            )
        for frame, item in zip(frames, audio):
            if item.time_seconds != frame.time_seconds:
                raise ValueError("audio timestamps must exactly match scene frame PTS.")
    if audio_transients is not None:
        if len(audio_transients) != len(frames):
            raise ValueError(
                "audio_transients must contain exactly one value per scene frame."
            )
        if any(
            value is not None and not isinstance(value, bool)
            for value in audio_transients
        ):
            raise ValueError("audio_transients values must be bool or None.")
    scene_frames = tuple(
        SceneFrame(
            time_seconds=frame.time_seconds,
            body_2d=frame.frame_2d,
            body_3d=frame.world_landmarks,
            audio_energy=item.energy if audio is not None else None,
            audio_transient=(
                audio_transients[index] if audio_transients is not None else None
            ),
        )
        for index, (frame, item) in enumerate(
            zip(frames, audio or (None,) * len(frames))
        )
    )
    modalities = {"body_2d", "body_3d"}
    if audio is not None:
        modalities.add("audio")
    return SceneTrack(
        source_fingerprint=world_snapshot.identity.source_fingerprint,
        frames=scene_frames,
        available_modalities=frozenset(modalities),
    )
