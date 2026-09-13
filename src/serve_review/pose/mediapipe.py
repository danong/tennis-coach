"""MediaPipe Pose Landmarker Heavy adapter (M2.4).

Adapts MediaPipe Tasks ``PoseLandmarker`` ``VIDEO`` mode to the portable
M2.2 observation schemas. Design notes:

- Exactly one landmarker per backend instance, guarded by a
  :class:`threading.Lock` so concurrent callers serialize ``VIDEO``
  calls instead of racing the estimator.
- Canonical ``time_seconds`` (float) is preserved exactly on every
  :class:`FrameObservation`; ``timestamp_ms`` is only the
  ``round(time_seconds * 1000)`` convenience submitted to
  ``detect_for_video`` and stored alongside. Strictly increasing
  milliseconds are enforced -- repeats fail loudly.
- The landmarker emits no person boxes and no per-person detection
  score, so the adapter derives an axis-aligned bounding box from the
  present joints and reports the mean present-joint visibility as the
  person ``score`` proxy. Both derivations are documented on the
  returned objects' provenance (see :func:`landmarks_to_person`) and
  must never be mistaken for detector output.
- Missing data is explicit: an empty ``persons`` list records
  no-person frames; a ``None`` joint records a missing landmark. A
  detected pose with zero usable joints is skipped (it carries no
  observation), which likewise yields an empty ``persons`` entry when
  nothing else was detected.
- This module never reads private footage and performs no downloads.
  Importing it does not import MediaPipe; the runtime is imported
  lazily inside :func:`open_mediapipe_backend` (and the default image
  wrapper) so pure mapping stays testable offline with injected fakes.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from serve_review.pose.backend import (
    BackendClosedError,
    InferenceError,
    MappingError,
    ModelFileError,
    PoseBackendError,
    check_timestamp_order,
    timestamp_ms_for,
    verify_model_file,
)
from serve_review.pose.schema import (
    NUM_KEYPOINTS,
    BodyKeypoint,
    CacheIdentity,
    FrameObservation,
    PersonBox,
    PersonObservation,
)
from serve_review.pose.world import (
    NUM_WORLD_LANDMARKS,
    WorldCacheIdentity,
    WorldFrameObservation,
    WorldLandmark,
)

__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "MODEL_ARTIFACT_FILENAME",
    "EXPECTED_MODEL_SHA256",
    "DEFAULT_MODEL_PATH",
    "NUM_EXPECTED_LANDMARKS",
    "NUM_EXPECTED_WORLD_LANDMARKS",
    "BOX_PAD",
    "MediaPipePoseBackend",
    "build_cache_identity",
    "build_world_cache_identity",
    "landmark_to_keypoint",
    "landmarks_to_person",
    "open_mediapipe_backend",
    "result_to_frame_observation",
    "result_to_world_frame_observation",
    "visibility_of",
    "world_landmark_to_point",
]

#: Cache/model identity matching ``models/manifest.json``.
MODEL_NAME = "mediapipe-pose-landmarker-heavy"
#: Approved Heavy artifact filename.
MODEL_ARTIFACT_FILENAME = "pose_landmarker_heavy.task"
#: Pinned SHA-256 of the approved artifact (models/manifest.json).
EXPECTED_MODEL_SHA256 = (
    "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"
)
#: Estimator version pinning the exact approved bytes (deterministic,
#: offline; a different artifact fails verification before it can run).
MODEL_VERSION = f"heavy-{EXPECTED_MODEL_SHA256[:16]}"
#: Default location of the approved local artifact (M2.3 downloader
#: deferred, so no automatic download is attempted).
DEFAULT_MODEL_PATH = Path("models/pose_landmarker_heavy.task")
#: Landmarks required per pose (MediaPipe Pose order, 0..32).
NUM_EXPECTED_LANDMARKS = NUM_KEYPOINTS
#: World landmarks required per pose (MediaPipe pose_world_landmarks order).
NUM_EXPECTED_WORLD_LANDMARKS = NUM_WORLD_LANDMARKS
#: Half-size padding applied only to degenerate box axes (single joint
#: or perfectly collinear joints) so the box stays valid.
BOX_PAD = 0.01


def visibility_of(landmark: Any) -> float:
    """Resolve an honest ``[0, 1]`` visibility for one landmark.

    Prefers ``visibility`` and falls back to ``presence`` (both are
    optional on MediaPipe ``NormalizedLandmark``); missing, boolean,
    non-finite, or non-numeric values resolve to ``0.0`` (unknown means
    not visibly evidenced, never confident). Finite values are clamped
    to ``[0, 1]``. Accepts attribute-style or mapping-style landmarks.
    """
    for key in ("visibility", "presence"):
        value: Any = None
        if isinstance(landmark, dict):
            value = landmark.get(key)
        else:
            value = getattr(landmark, key, None)
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        return min(1.0, max(0.0, number))
    return 0.0


def _coordinates_of(landmark: Any) -> tuple[Any, Any]:
    if isinstance(landmark, dict):
        return landmark.get("x"), landmark.get("y")
    return getattr(landmark, "x", None), getattr(landmark, "y", None)


def landmark_to_keypoint(landmark: Any) -> BodyKeypoint | None:
    """Map one landmark to a :class:`BodyKeypoint` or ``None``.

    ``None`` (explicitly missing) is returned when either coordinate is
    absent, boolean, non-numeric, non-finite, or outside ``[0, 1]`` in
    the upright unmirrored normalized frame. Out-of-range coordinates
    are reported as missing rather than clamped so edge estimates are
    never silently fabricated into bounds.
    """
    raw_x, raw_y = _coordinates_of(landmark)
    for raw in (raw_x, raw_y):
        if raw is None or isinstance(raw, bool):
            return None
        if not isinstance(raw, (int, float)):
            return None
        if not math.isfinite(float(raw)):
            return None
    x = float(raw_x)
    y = float(raw_y)
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return None
    return BodyKeypoint(x=x, y=y, visibility=visibility_of(landmark))


def landmarks_to_person(pose_landmarks: Any) -> PersonObservation | None:
    """Map one 33-landmark pose to a person (or ``None`` when unusable).

    Requires exactly ``33`` landmarks in MediaPipe Pose order; any other
    count raises :class:`MappingError` instead of truncating or padding.
    Returns ``None`` when zero joints are usable -- the caller then
    records a no-person frame rather than a fabricated all-missing
    person (which the schema rejects).

    The ``box`` is the axis-aligned bounding box of the present joints
    (degenerate axes are padded by :data:`BOX_PAD` and clamped); the
    ``score`` is the mean visibility of the present joints, an honest
    proxy documented here because the landmarker emits no per-person
    detection confidence.
    """
    if not isinstance(pose_landmarks, (list, tuple)):
        raise MappingError(
            "pose landmarks must be a list or tuple of "
            f"{NUM_EXPECTED_LANDMARKS} entries in MediaPipe Pose order, "
            f"got {type(pose_landmarks).__name__}."
        )
    joints_in = list(pose_landmarks)
    if len(joints_in) != NUM_EXPECTED_LANDMARKS:
        raise MappingError(
            f"pose landmarker contract violation: expected "
            f"{NUM_EXPECTED_LANDMARKS} landmarks per pose, got "
            f"{len(joints_in)}; refusing to truncate or pad."
        )
    keypoints = tuple(landmark_to_keypoint(entry) for entry in joints_in)
    present = [(joint.x, joint.y) for joint in keypoints if joint is not None]
    if not present:
        return None
    xs = [point[0] for point in present]
    ys = [point[1] for point in present]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max - x_min < 1e-9:
        x_min = max(0.0, x_min - BOX_PAD)
        x_max = min(1.0, x_max + BOX_PAD)
    if y_max - y_min < 1e-9:
        y_min = max(0.0, y_min - BOX_PAD)
        y_max = min(1.0, y_max + BOX_PAD)
    box = PersonBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
    score = sum(visibility_of(entry) for entry, joint in zip(joints_in, keypoints) if joint is not None) / len(present)
    return PersonObservation(box=box, keypoints=keypoints, score=float(score))


def result_to_frame_observation(result: Any, *, time_seconds: float) -> FrameObservation:
    """Map one landmarker result to a :class:`FrameObservation`.

    ``result`` is duck-typed (``.pose_landmarks``: sequence of poses);
    ``None`` or an empty list means no person was detected and yields an
    empty ``persons`` list. Pose order is preserved. The canonical float
    is stored exactly; ``timestamp_ms`` is ``round(time_seconds*1000)``.
    """
    if (
        isinstance(time_seconds, bool)
        or not isinstance(time_seconds, (int, float))
        or not math.isfinite(float(time_seconds))
        or float(time_seconds) < 0
    ):
        raise MappingError(
            f"invalid canonical time_seconds {time_seconds!r}: expected a "
            "finite number of seconds >= 0."
        )
    moment = float(time_seconds)
    raw_poses: Any = None
    if isinstance(result, dict):
        raw_poses = result.get("pose_landmarks", [])
    else:
        raw_poses = getattr(result, "pose_landmarks", None)
    if raw_poses is None:
        raw_poses = []
    if not isinstance(raw_poses, (list, tuple)):
        raise MappingError(
            "landmarker result 'pose_landmarks' must be a list of poses, "
            f"got {type(raw_poses).__name__}."
        )
    persons: list[PersonObservation] = []
    for pose in raw_poses:
        person = landmarks_to_person(pose)
        if person is not None:
            persons.append(person)
    return FrameObservation(
        time_seconds=moment,
        timestamp_ms=int(round(moment * 1000)),
        persons=tuple(persons),
    )


def build_cache_identity(
    source_fingerprint: str,
    sampling_rate_hz: float,
    sampling_start_seconds: float,
) -> CacheIdentity:
    """Build the M2.2 cache identity pinned to this adapter's model."""
    return CacheIdentity(
        source_fingerprint=source_fingerprint,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=sampling_rate_hz,
        sampling_start_seconds=sampling_start_seconds,
    )


def build_world_cache_identity(source_fingerprint: str) -> WorldCacheIdentity:
    """Build the M4.7 dense-world cache identity pinned to this adapter."""
    return WorldCacheIdentity(
        source_fingerprint=source_fingerprint,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )


def _world_coordinates_of(landmark: Any) -> tuple[Any, Any, Any]:
    if isinstance(landmark, dict):
        return landmark.get("x"), landmark.get("y"), landmark.get("z")
    return (
        getattr(landmark, "x", None),
        getattr(landmark, "y", None),
        getattr(landmark, "z", None),
    )


def world_landmark_to_point(landmark: Any) -> WorldLandmark | None:
    """Map one ``pose_world_landmarks`` entry to a :class:`WorldLandmark`.

    Returns ``None`` (explicitly missing) when any of ``x``/``y``/``z``
    is absent, boolean, non-numeric, or non-finite. World coordinates
    are hip-centered meters with no ``[0, 1]`` range contract, so no
    clamping is applied. This function reads only the world landmark's
    own ``x``/``y``/``z`` and never substitutes a 2D normalized ``z``
    (MediaPipe normalized landmarks carry no depth in meters).
    """
    raw_x, raw_y, raw_z = _world_coordinates_of(landmark)
    for raw in (raw_x, raw_y, raw_z):
        if raw is None or isinstance(raw, bool):
            return None
        if not isinstance(raw, (int, float)):
            return None
        if not math.isfinite(float(raw)):
            return None
    return WorldLandmark(x=float(raw_x), y=float(raw_y), z=float(raw_z))


def _pose_streams_of(result: Any) -> tuple[Any, Any]:
    if isinstance(result, dict):
        return result.get("pose_landmarks"), result.get(
            "pose_world_landmarks"
        )
    return getattr(result, "pose_landmarks", None), getattr(
        result, "pose_world_landmarks", None
    )


def result_to_world_frame_observation(
    result: Any, *, time_seconds: float
) -> WorldFrameObservation:
    """Map one landmarker result to a :class:`WorldFrameObservation`.

    Reads the synchronized ``pose_landmarks`` (normalized 2D companion
    with visibility) plus ``pose_world_landmarks`` (primary meters)
    streams from one result. The dense-world observation is
    single-person: exactly zero or one pose per stream is accepted.
    More than one pose raises :class:`MappingError` rather than
    silently selecting a person. A pose-count mismatch between the two
    streams raises instead of pairing across streams, and any pose
    whose landmark count differs from 33 raises instead of truncating
    or padding. Per-joint missingness stays per-joint (``None``); a
    world stream with zero usable joints alongside usable 2D joints
    (or vice versa) raises as a stream mismatch instead of emitting a
    fabricated half observation. World coordinates are taken only from
    the world stream -- a 2D ``z`` is never substituted.
    """
    if (
        isinstance(time_seconds, bool)
        or not isinstance(time_seconds, (int, float))
        or not math.isfinite(float(time_seconds))
        or float(time_seconds) < 0
    ):
        raise MappingError(
            f"invalid canonical time_seconds {time_seconds!r}: expected a "
            "finite number of seconds >= 0."
        )
    moment = float(time_seconds)
    raw_2d, raw_world = _pose_streams_of(result)
    if raw_2d is None:
        raw_2d = []
    if raw_world is None:
        raw_world = []
    if not isinstance(raw_2d, (list, tuple)):
        raise MappingError(
            "landmarker result 'pose_landmarks' must be a list of poses, "
            f"got {type(raw_2d).__name__}."
        )
    if not isinstance(raw_world, (list, tuple)):
        raise MappingError(
            "landmarker result 'pose_world_landmarks' must be a list of "
            f"poses, got {type(raw_world).__name__}."
        )
    poses_2d = list(raw_2d)
    poses_world = list(raw_world)
    if len(poses_2d) != len(poses_world):
        raise MappingError(
            "pose/world stream count mismatch: "
            f"pose_landmarks holds {len(poses_2d)} pose(s) but "
            f"pose_world_landmarks holds {len(poses_world)}; refusing to "
            "pair across streams or substitute 2D depth."
        )
    if len(poses_2d) > 1:
        raise MappingError(
            "dense-world observation is single-person: got "
            f"{len(poses_2d)} poses; refusing to silently select one."
        )
    companion = result_to_frame_observation(result, time_seconds=moment)
    if not poses_2d:
        empty_world = tuple([None] * NUM_EXPECTED_WORLD_LANDMARKS)
        return WorldFrameObservation(
            time_seconds=moment,
            timestamp_ms=int(round(moment * 1000)),
            world_landmarks=empty_world,
            frame_2d=companion,
        )
    pose_2d = poses_2d[0]
    pose_world = poses_world[0]
    if not isinstance(pose_2d, (list, tuple)):
        raise MappingError(
            "pose landmarks must be a list or tuple of "
            f"{NUM_EXPECTED_LANDMARKS} entries, got "
            f"{type(pose_2d).__name__}."
        )
    if not isinstance(pose_world, (list, tuple)):
        raise MappingError(
            "pose world landmarks must be a list or tuple of "
            f"{NUM_EXPECTED_WORLD_LANDMARKS} entries, got "
            f"{type(pose_world).__name__}."
        )
    if len(list(pose_2d)) != NUM_EXPECTED_LANDMARKS:
        raise MappingError(
            "pose landmarker contract violation: expected "
            f"{NUM_EXPECTED_LANDMARKS} landmarks per 2D pose, got "
            f"{len(list(pose_2d))}; refusing to truncate or pad."
        )
    if len(list(pose_world)) != NUM_EXPECTED_WORLD_LANDMARKS:
        raise MappingError(
            "pose landmarker contract violation: expected "
            f"{NUM_EXPECTED_WORLD_LANDMARKS} landmarks per world pose, got "
            f"{len(list(pose_world))}; refusing to truncate or pad."
        )
    world_joints = tuple(
        world_landmark_to_point(entry) for entry in list(pose_world)
    )
    person_2d = landmarks_to_person(pose_2d)
    world_usable = sum(1 for joint in world_joints if joint is not None)
    if (person_2d is None) != (world_usable == 0):
        raise MappingError(
            "pose/world stream mismatch: 2D usable="
            f"{person_2d is not None}, world usable joints={world_usable}; "
            "refusing to emit a half-fabricated observation."
        )
    if person_2d is None:
        # Both streams empty of usable joints: honest no-person frame.
        # Re-derive the companion from the same 2D pose so visibility
        # handling stays identical to the 2D path.
        return WorldFrameObservation(
            time_seconds=moment,
            timestamp_ms=int(round(moment * 1000)),
            world_landmarks=world_joints,
            frame_2d=companion,
        )
    return WorldFrameObservation(
        time_seconds=moment,
        timestamp_ms=int(round(moment * 1000)),
        world_landmarks=world_joints,
        frame_2d=companion,
    )


def _default_image_wrapper(image: np.ndarray) -> Any:
    """Wrap an RGB ``uint8`` array as a MediaPipe ``Image`` (lazy import)."""
    try:
        import mediapipe as mp  # noqa: PLC0415
    except ImportError as exc:
        raise PoseBackendError(
            "mediapipe is required for Pose Landmarker inference; run "
            "`mise run setup` (uv owns dependencies) and retry."
        ) from exc
    if not isinstance(image, np.ndarray):
        raise InferenceError(
            "invalid frame image: expected a numpy uint8 RGB array, "
            f"got {type(image).__name__}."
        )
    frame = np.ascontiguousarray(image, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise InferenceError(
            "invalid frame image: expected shape (height, width, 3), "
            f"got {frame.shape!r}."
        )
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise InferenceError(
            f"invalid frame image shape {frame.shape!r}: dimensions must "
            "be positive."
        )
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)


def _check_rgb_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise InferenceError(
            "invalid frame image: expected a numpy uint8 RGB array, "
            f"got {type(image).__name__}."
        )
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise InferenceError(
            "invalid frame image: expected a uint8 array with shape "
            f"(height, width, 3), got dtype={image.dtype} "
            f"shape={image.shape}."
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InferenceError(
            f"invalid frame image shape {image.shape!r}: dimensions must "
            "be positive."
        )
    return image


class MediaPipePoseBackend:
    """Serialized adapter around one Pose Landmarker (VIDEO mode).

    At most one estimator is ever called, and every ``detect_for_video``
    call holds an internal lock so concurrent threads cannot interleave
    timestamps. Integer milliseconds are enforced strictly increasing;
    the canonical float travels untouched onto the observation.
    """

    def __init__(
        self,
        landmarker: Any,
        *,
        model_name: str = MODEL_NAME,
        model_version: str = MODEL_VERSION,
        image_wrapper: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        if landmarker is None:
            raise PoseBackendError(
                "invalid landmarker: expected a PoseLandmarker with "
                "'detect_for_video', got None."
            )
        if not callable(getattr(landmarker, "detect_for_video", None)):
            raise PoseBackendError(
                "invalid landmarker: expected an object with a callable "
                f"'detect_for_video', got {type(landmarker).__name__}."
            )
        if not isinstance(model_name, str) or not model_name.strip():
            raise PoseBackendError(
                f"invalid model_name {model_name!r}: expected a non-blank "
                "string."
            )
        if not isinstance(model_version, str) or not model_version.strip():
            raise PoseBackendError(
                f"invalid model_version {model_version!r}: expected a "
                "non-blank string."
            )
        if image_wrapper is not None and not callable(image_wrapper):
            raise PoseBackendError(
                f"invalid image_wrapper {image_wrapper!r}: expected a "
                "callable or None."
            )
        self._landmarker = landmarker
        self._model_name = model_name
        self._model_version = model_version
        self._image_wrapper = image_wrapper or _default_image_wrapper
        self._lock = threading.Lock()
        self._last_ms: int | None = None
        self._closed = False

    @property
    def model_name(self) -> str:
        """Return the estimator name used for cache identity."""
        return self._model_name

    @property
    def model_version(self) -> str:
        """Return the estimator version used for cache identity."""
        return self._model_version

    @property
    def last_timestamp_ms(self) -> int | None:
        """Return the last submitted VIDEO timestamp (or ``None``)."""
        with self._lock:
            return self._last_ms

    def infer(self, image: np.ndarray, time_seconds: float) -> FrameObservation:
        """Run one ordered VIDEO inference for ``time_seconds``.

        Validates the RGB frame, converts canonical time to integer
        milliseconds, rejects non-increasing stamps, serializes the
        estimator call, and maps the result (preserving the canonical
        float). Raises :class:`BackendClosedError` after :meth:`close`.
        """
        frame = _check_rgb_image(image)
        stamp_ms = timestamp_ms_for(time_seconds)
        moment = float(time_seconds)
        with self._lock:
            if self._closed:
                raise BackendClosedError(
                    "pose backend is closed; create a new backend instead "
                    "of reusing a released landmarker."
                )
            check_timestamp_order(self._last_ms, stamp_ms)
            try:
                wrapped = self._image_wrapper(frame)
            except PoseBackendError:
                raise
            except Exception as exc:
                raise InferenceError(
                    f"could not wrap frame at t={moment}s for MediaPipe: "
                    f"{exc}."
                ) from exc
            try:
                result = self._landmarker.detect_for_video(wrapped, stamp_ms)
            except Exception as exc:
                raise InferenceError(
                    f"Pose Landmarker VIDEO inference failed at "
                    f"t={moment}s ({stamp_ms} ms): {exc}."
                ) from exc
            try:
                observation = result_to_frame_observation(result, time_seconds=moment)
            except PoseBackendError:
                raise
            except Exception as exc:
                raise InferenceError(
                    f"could not map Pose Landmarker output at t={moment}s: "
                    f"{exc}."
                ) from exc
            self._last_ms = stamp_ms
            return observation

    def infer_world(
        self, image: np.ndarray, time_seconds: float
    ) -> WorldFrameObservation:
        """Run one ordered VIDEO inference returning a world observation.

        Shares the serialized estimator, image wrapper, and strictly
        increasing ``timestamp_ms`` contract with :meth:`infer` (one
        ``_last_ms`` across both entry points so interleaved 2D/world
        calls still order loudly). The result is mapped with
        :func:`result_to_world_frame_observation`, preserving the
        canonical float and pairing synchronized ``pose_landmarks``
        (2D companion) with ``pose_world_landmarks`` (primary meters).
        Missing/no-person rows stay honest (empty 2D companion plus
        all-missing world joints); world coordinates come only from the
        world stream, never from a 2D ``z``. Raises
        :class:`BackendClosedError` after :meth:`close`.
        """
        frame = _check_rgb_image(image)
        stamp_ms = timestamp_ms_for(time_seconds)
        moment = float(time_seconds)
        with self._lock:
            if self._closed:
                raise BackendClosedError(
                    "pose backend is closed; create a new backend instead "
                    "of reusing a released landmarker."
                )
            check_timestamp_order(self._last_ms, stamp_ms)
            try:
                wrapped = self._image_wrapper(frame)
            except PoseBackendError:
                raise
            except Exception as exc:
                raise InferenceError(
                    f"could not wrap frame at t={moment}s for MediaPipe: "
                    f"{exc}."
                ) from exc
            try:
                result = self._landmarker.detect_for_video(wrapped, stamp_ms)
            except Exception as exc:
                raise InferenceError(
                    f"Pose Landmarker VIDEO inference failed at "
                    f"t={moment}s ({stamp_ms} ms): {exc}."
                ) from exc
            try:
                observation = result_to_world_frame_observation(
                    result, time_seconds=moment
                )
            except PoseBackendError:
                raise
            except Exception as exc:
                raise InferenceError(
                    f"could not map Pose Landmarker output at t={moment}s: "
                    f"{exc}."
                ) from exc
            self._last_ms = stamp_ms
            return observation

    def close(self) -> None:
        """Release the landmarker (idempotent; never raises on repeat)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            closer = getattr(self._landmarker, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    def __enter__(self) -> MediaPipePoseBackend:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close()
        return False


def open_mediapipe_backend(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    *,
    num_poses: int = 1,
    expected_sha256: str = EXPECTED_MODEL_SHA256,
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
) -> MediaPipePoseBackend:
    """Validate the approved artifact and open exactly one landmarker.

    Verifies file existence and SHA-256 (mismatch refuses to run),
    imports MediaPipe lazily, and builds ``PoseLandmarkerOptions`` in
    ``VIDEO`` running mode. Raises actionable
    :class:`~serve_review.pose.backend.PoseBackendError` subclasses for
    missing/unverified models, bad options, and unavailable runtimes.
    The caller owns the returned backend and must :meth:`close` it.
    """
    if isinstance(num_poses, bool) or not isinstance(num_poses, int) or num_poses < 1:
        raise PoseBackendError(
            f"invalid num_poses {num_poses!r}: expected an integer >= 1."
        )
    try:
        verified = verify_model_file(model_path, expected_sha256=expected_sha256)
    except PoseBackendError as exc:
        raise ModelFileError(
            f"{exc} The approved Heavy model lives at "
            f"'{MODEL_ARTIFACT_FILENAME}' (see models/manifest.json); the "
            "M2.3 downloader is deferred, so place the file locally -- no "
            "automatic download is attempted."
        ) from exc
    try:
        import mediapipe as mp  # noqa: PLC0415
    except ImportError as exc:
        raise PoseBackendError(
            "mediapipe is required for Pose Landmarker inference; run "
            "`mise run setup` (uv owns dependencies) and retry."
        ) from exc
    try:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(verified)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=num_poses,
        )
    except Exception as exc:
        raise PoseBackendError(
            f"could not configure Pose Landmarker from {verified}: {exc}."
        ) from exc
    try:
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    except Exception as exc:
        raise PoseBackendError(
            f"could not load Pose Landmarker model from {verified}: {exc}."
        ) from exc
    return MediaPipePoseBackend(
        landmarker, model_name=model_name, model_version=model_version
    )
