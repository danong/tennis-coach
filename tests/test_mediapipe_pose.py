"""Focused M2.4 tests for the MediaPipe Pose Landmarker adapter.

Deterministic, offline, and independent of private footage, model
weights, and network access. Every landmarker result is a fake/injected
duck-typed object; the real ``.task`` artifact is never loaded (it is
not present in the test workspace). Model-file checks use tiny
temporary files only.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_review.pose import mediapipe as mp_pose
from serve_review.pose.backend import (
    BackendClosedError,
    InferenceError,
    MappingError,
    ModelFileError,
    ModelHashMismatchError,
    PoseBackendError,
    TimestampOrderError,
    check_timestamp_order,
    timestamp_ms_for,
    verify_model_file,
)
from serve_review.pose.mediapipe import (
    DEFAULT_MODEL_PATH,
    EXPECTED_MODEL_SHA256,
    MODEL_ARTIFACT_FILENAME,
    MODEL_NAME,
    MODEL_VERSION,
    MediaPipePoseBackend,
    build_cache_identity,
    landmark_to_keypoint,
    landmarks_to_person,
    result_to_frame_observation,
    visibility_of,
)
from serve_review.pose.schema import (
    NUM_KEYPOINTS,
    CacheIdentity,
    FrameObservation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fake landmark/result factories (duck-typed; never MediaPipe objects).
# ---------------------------------------------------------------------------


def make_landmark(
    x: float | None = 0.5,
    y: float | None = 0.5,
    visibility: float | None = 0.9,
    presence: float | None = None,
    style: str = "namespace",
) -> object:
    if style == "dict":
        payload: dict[str, object] = {"x": x, "y": y}
        if visibility is not None:
            payload["visibility"] = visibility
        if presence is not None:
            payload["presence"] = presence
        return payload
    return SimpleNamespace(x=x, y=y, visibility=visibility, presence=presence)


def make_pose(
    *,
    count: int = NUM_KEYPOINTS,
    style: str = "namespace",
    visibility: float | None = 0.9,
    seed_offset: float = 0.0,
) -> list:
    pose = []
    for index in range(count):
        x = 0.05 + 0.9 * (index / (NUM_KEYPOINTS - 1)) + seed_offset
        x = min(0.99, max(0.01, x))
        y = 0.10 + 0.01 * (index % 5)
        pose.append(make_landmark(x, y, visibility, style=style))
    return pose


def make_result(poses: list | None) -> SimpleNamespace:
    return SimpleNamespace(pose_landmarks=poses)


class FakeLandmarker:
    """Injectable stand-in for PoseLandmarker (VIDEO mode)."""

    def __init__(self, results: list) -> None:
        self._results = list(results)
        self.calls: list[tuple[object, int]] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()
        self.closed = False

    def detect_for_video(self, image: object, timestamp_ms: int) -> object:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.calls.append((image, timestamp_ms))
            index = len(self.calls) - 1
            if index < len(self._results):
                outcome = self._results[index]
            else:
                outcome = self._results[-1]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            with self._guard:
                self.active -= 1

    def close(self) -> None:
        self.closed = True


def make_backend(results: list, **kwargs) -> tuple[MediaPipePoseBackend, FakeLandmarker]:
    fake = FakeLandmarker(results)
    backend = MediaPipePoseBackend(fake, image_wrapper=lambda frame: frame, **kwargs)
    return backend, fake


def rgb_frame(value: int = 128) -> np.ndarray:
    return np.full((8, 10, 3), value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Timestamp helpers (canonical float vs integer milliseconds).
# ---------------------------------------------------------------------------


def test_timestamp_ms_rounding_matches_sampler_convention() -> None:
    assert timestamp_ms_for(0.0) == 0
    assert timestamp_ms_for(0.033) == 33
    assert timestamp_ms_for(1.0) == 1000
    assert timestamp_ms_for(0.0005) in (0, 1)  # documents banker's rounding edge
    assert timestamp_ms_for(2.345678) == 2346


def test_timestamp_ms_rejects_bad_canonical_time() -> None:
    for bad in (-1.0, float("nan"), float("inf"), True, "33", None):
        with pytest.raises(TimestampOrderError):
            timestamp_ms_for(bad)  # type: ignore[arg-type]


def test_check_timestamp_order_accepts_strictly_increasing() -> None:
    check_timestamp_order(None, 0)
    check_timestamp_order(0, 1)
    check_timestamp_order(33, 66)


def test_check_timestamp_order_rejects_repeat_and_backward() -> None:
    with pytest.raises(TimestampOrderError, match="strictly increasing"):
        check_timestamp_order(33, 33)
    with pytest.raises(TimestampOrderError, match="strictly increasing"):
        check_timestamp_order(100, 99)


# ---------------------------------------------------------------------------
# Visibility resolution.
# ---------------------------------------------------------------------------


def test_visibility_prefers_visibility_then_presence() -> None:
    assert visibility_of(make_landmark(visibility=0.7, presence=0.2)) == pytest.approx(0.7)
    assert visibility_of(make_landmark(visibility=None, presence=0.4)) == pytest.approx(0.4)
    assert visibility_of(make_landmark(visibility=None, presence=None)) == 0.0


def test_visibility_clamps_and_rejects_non_finite() -> None:
    assert visibility_of(make_landmark(visibility=1.8)) == 1.0
    assert visibility_of(make_landmark(visibility=-0.5)) == 0.0
    assert visibility_of(make_landmark(visibility=float("nan"))) == 0.0
    assert visibility_of(make_landmark(visibility=float("inf"))) == 0.0
    assert visibility_of(make_landmark(visibility=True)) == 0.0  # type: ignore[arg-type]
    assert visibility_of({"x": 0.5, "y": 0.5}) == 0.0


# ---------------------------------------------------------------------------
# Coordinate mapping.
# ---------------------------------------------------------------------------


def test_landmark_mapping_preserves_coordinates() -> None:
    keypoint = landmark_to_keypoint(make_landmark(0.25, 0.75, 0.6))
    assert keypoint is not None
    assert keypoint.x == pytest.approx(0.25)
    assert keypoint.y == pytest.approx(0.75)
    assert keypoint.visibility == pytest.approx(0.6)


def test_landmark_mapping_supports_dict_style() -> None:
    keypoint = landmark_to_keypoint(make_landmark(0.2, 0.4, 0.5, style="dict"))
    assert keypoint is not None
    assert (keypoint.x, keypoint.y) == pytest.approx((0.2, 0.4))


@pytest.mark.parametrize(
    "x,y",
    [
        (None, 0.5),
        (0.5, None),
        (float("nan"), 0.5),
        (0.5, float("inf")),
        (True, 0.5),
        (-0.01, 0.5),
        (1.01, 0.5),
        (0.5, -0.2),
        (0.5, 1.5),
        ("0.5", 0.5),
    ],
)
def test_landmark_mapping_marks_bad_coordinates_missing(x: object, y: object) -> None:
    assert landmark_to_keypoint(SimpleNamespace(x=x, y=y, visibility=0.9)) is None


def test_landmarks_to_person_maps_33_in_order() -> None:
    pose = make_pose()
    person = landmarks_to_person(pose)
    assert person is not None
    assert len(person.keypoints) == NUM_KEYPOINTS
    first = person.keypoints[0]
    last = person.keypoints[32]
    assert first is not None and last is not None
    assert first.x < last.x  # order preserved left-to-right in factory
    assert person.joint("nose") is first
    assert person.joint("right_foot_index") is last
    assert person.present_count == NUM_KEYPOINTS


def test_landmarks_to_person_rejects_wrong_count() -> None:
    with pytest.raises(MappingError, match="33"):
        landmarks_to_person(make_pose(count=32))
    with pytest.raises(MappingError, match="33"):
        landmarks_to_person(make_pose(count=34))
    with pytest.raises(MappingError, match="list or tuple"):
        landmarks_to_person({"pose": []})


def test_landmarks_to_person_marks_missing_joints() -> None:
    pose = make_pose()
    pose[3] = make_landmark(None, None, None)
    pose[10] = make_landmark(2.0, 0.5, 0.9)  # out of range -> missing
    person = landmarks_to_person(pose)
    assert person is not None
    assert person.keypoints[3] is None
    assert person.keypoints[10] is None
    assert person.present_count == NUM_KEYPOINTS - 2


def test_landmarks_to_person_skips_fully_missing_pose() -> None:
    pose = [make_landmark(None, None, None) for _ in range(NUM_KEYPOINTS)]
    assert landmarks_to_person(pose) is None


def test_person_score_is_mean_present_visibility_proxy() -> None:
    pose = make_pose()
    for index in range(0, NUM_KEYPOINTS, 3):
        pose[index] = make_landmark(0.5, 0.5, 0.3)
    person = landmarks_to_person(pose)
    assert person is not None
    expected = sum(0.3 if i % 3 == 0 else 0.9 for i in range(NUM_KEYPOINTS)) / NUM_KEYPOINTS
    assert person.score == pytest.approx(expected)


def test_person_box_bounds_present_joints() -> None:
    pose = [make_landmark(None, None, None) for _ in range(NUM_KEYPOINTS)]
    pose[0] = make_landmark(0.2, 0.3, 0.9)
    pose[1] = make_landmark(0.6, 0.8, 0.9)
    person = landmarks_to_person(pose)
    assert person is not None
    assert person.box.x_min == pytest.approx(0.2)
    assert person.box.x_max == pytest.approx(0.6)
    assert person.box.y_min == pytest.approx(0.3)
    assert person.box.y_max == pytest.approx(0.8)


def test_person_box_pads_degenerate_single_joint() -> None:
    pose = [make_landmark(None, None, None) for _ in range(NUM_KEYPOINTS)]
    pose[5] = make_landmark(0.5, 0.5, 0.9)
    person = landmarks_to_person(pose)
    assert person is not None
    assert person.box.x_min < person.box.x_max
    assert person.box.y_min < person.box.y_max
    assert 0.0 <= person.box.x_min and person.box.x_max <= 1.0


# ---------------------------------------------------------------------------
# Result -> FrameObservation mapping (order, missing, canonical time).
# ---------------------------------------------------------------------------


def test_empty_and_none_results_mean_no_person() -> None:
    for poses in ([], None):
        observation = result_to_frame_observation(make_result(poses), time_seconds=0.5)
        assert observation.persons == ()
        assert observation.has_person is False
        assert observation.time_seconds == pytest.approx(0.5)
        assert observation.timestamp_ms == 500


def test_result_mapping_preserves_canonical_float() -> None:
    moment = 0.123456
    observation = result_to_frame_observation(make_result([make_pose()]), time_seconds=moment)
    assert observation.time_seconds == moment  # exact float, not ms-derived
    assert observation.timestamp_ms == int(round(moment * 1000))
    assert observation.time_seconds != observation.timestamp_ms / 1000


def test_result_mapping_preserves_pose_order() -> None:
    first = make_pose(seed_offset=0.0, visibility=0.9)
    second = make_pose(seed_offset=0.05, visibility=0.4)
    observation = result_to_frame_observation(
        make_result([first, second]), time_seconds=1.0
    )
    assert len(observation.persons) == 2
    assert observation.persons[0].score == pytest.approx(0.9)
    assert observation.persons[1].score == pytest.approx(0.4)


def test_result_mapping_skips_unusable_poses_honestly() -> None:
    usable = make_pose()
    unusable = [make_landmark(None, None, None) for _ in range(NUM_KEYPOINTS)]
    observation = result_to_frame_observation(
        make_result([unusable, usable]), time_seconds=0.1
    )
    assert len(observation.persons) == 1
    assert observation.persons[0].present_count == NUM_KEYPOINTS
    empty = result_to_frame_observation(make_result([unusable]), time_seconds=0.1)
    assert empty.persons == ()


def test_result_mapping_rejects_bad_time_and_shape() -> None:
    with pytest.raises(MappingError):
        result_to_frame_observation(make_result([]), time_seconds=-0.5)
    with pytest.raises(MappingError, match="list of poses"):
        result_to_frame_observation(
            SimpleNamespace(pose_landmarks={"a": 1}), time_seconds=0.5
        )
    with pytest.raises(MappingError, match="33"):
        result_to_frame_observation(
            make_result([make_pose(count=10)]), time_seconds=0.5
        )


def test_mapping_is_deterministic() -> None:
    poses = [make_pose(seed_offset=0.02)]
    left = result_to_frame_observation(make_result(poses), time_seconds=0.25)
    right = result_to_frame_observation(make_result(poses), time_seconds=0.25)
    assert left == right
    assert left.to_json() == right.to_json()


# ---------------------------------------------------------------------------
# Backend behavior: ordering, serialization, validation, lifecycle.
# ---------------------------------------------------------------------------


def test_backend_infer_maps_and_records_ordered_calls() -> None:
    backend, fake = make_backend([make_result([make_pose()]), make_result([])])
    first = backend.infer(rgb_frame(), 0.033)
    second = backend.infer(rgb_frame(), 0.066)
    assert isinstance(first, FrameObservation)
    assert first.has_person is True
    assert first.timestamp_ms == 33
    assert second.has_person is False
    assert second.timestamp_ms == 66
    assert [stamp for _, stamp in fake.calls] == [33, 66]
    assert backend.last_timestamp_ms == 66
    backend.close()


def test_backend_rejects_non_increasing_ms() -> None:
    backend, _ = make_backend([make_result([])])
    backend.infer(rgb_frame(), 0.033)
    with pytest.raises(TimestampOrderError, match="strictly increasing"):
        backend.infer(rgb_frame(), 0.033)  # same ms: repeat
    with pytest.raises(TimestampOrderError, match="strictly increasing"):
        backend.infer(rgb_frame(), 0.010)  # step backwards
    # Backend still usable at a later stamp after rejected calls.
    observation = backend.infer(rgb_frame(), 0.100)
    assert observation.timestamp_ms == 100
    backend.close()


def test_backend_serializes_concurrent_calls() -> None:
    poses = [make_result([make_pose()]) for _ in range(8)]
    backend, fake = make_backend(poses)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            backend.infer(rgb_frame(index % 250), (index + 1) * 0.05)
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    # Stagger submissions so VIDEO stamps stay strictly increasing even
    # though estimator execution overlaps without the lock.
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
        thread.join()
    assert errors == []
    assert fake.max_active == 1
    backend.close()


def test_backend_validates_image_before_calling_landmarker() -> None:
    backend, fake = make_backend([make_result([])])
    with pytest.raises(InferenceError, match="uint8"):
        backend.infer(np.zeros((4, 4, 3), dtype=np.float32), 0.05)
    with pytest.raises(InferenceError, match="shape"):
        backend.infer(np.zeros((4, 4), dtype=np.uint8), 0.05)
    with pytest.raises(InferenceError, match="numpy"):
        backend.infer([[0, 0, 0]], 0.05)  # type: ignore[arg-type]
    assert fake.calls == []
    backend.close()


def test_backend_wraps_inference_failures_actionably() -> None:
    backend, _ = make_backend([RuntimeError("boom")])
    with pytest.raises(InferenceError, match="VIDEO inference failed"):
        backend.infer(rgb_frame(), 0.05)
    backend.close()


def test_backend_close_is_idempotent_and_blocks_infer() -> None:
    backend, fake = make_backend([make_result([])])
    backend.close()
    backend.close()
    assert fake.closed is True
    with pytest.raises(BackendClosedError, match="closed"):
        backend.infer(rgb_frame(), 0.05)


def test_backend_context_manager_closes() -> None:
    fake = FakeLandmarker([make_result([])])
    with MediaPipePoseBackend(fake, image_wrapper=lambda frame: frame) as backend:
        assert backend.model_name == MODEL_NAME
        assert backend.model_version == MODEL_VERSION
    assert fake.closed is True


def test_backend_rejects_invalid_landmarker() -> None:
    with pytest.raises(PoseBackendError, match="detect_for_video"):
        MediaPipePoseBackend(object())
    with pytest.raises(PoseBackendError, match="got None"):
        MediaPipePoseBackend(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Model identity and metadata.
# ---------------------------------------------------------------------------


def test_model_constants_match_manifest() -> None:
    manifest_path = REPO_ROOT / "models" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {model["id"]: model for model in payload["models"]}
    entry = entries[MODEL_NAME]
    assert entry["artifact"] == MODEL_ARTIFACT_FILENAME
    assert entry["sha256"] == EXPECTED_MODEL_SHA256
    assert MODEL_VERSION.startswith("heavy-")
    assert EXPECTED_MODEL_SHA256 in MODEL_VERSION or MODEL_VERSION == (
        f"heavy-{EXPECTED_MODEL_SHA256[:16]}"
    )
    assert DEFAULT_MODEL_PATH.name == MODEL_ARTIFACT_FILENAME


def test_verify_model_file_missing_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "pose_landmarker_heavy.task"
    with pytest.raises(ModelFileError, match="not found"):
        verify_model_file(missing, expected_sha256=EXPECTED_MODEL_SHA256)


def test_verify_model_file_hash_mismatch_refuses_to_run(tmp_path: Path) -> None:
    candidate = tmp_path / MODEL_ARTIFACT_FILENAME
    candidate.write_bytes(b"tiny-sanitized-stand-in-bytes")
    with pytest.raises(ModelHashMismatchError, match="SHA-256 verification"):
        verify_model_file(candidate, expected_sha256=EXPECTED_MODEL_SHA256)


def test_verify_model_file_accepts_matching_digest(tmp_path: Path) -> None:
    candidate = tmp_path / MODEL_ARTIFACT_FILENAME
    candidate.write_bytes(b"tiny-sanitized-stand-in-bytes")
    import hashlib

    digest = hashlib.sha256(b"tiny-sanitized-stand-in-bytes").hexdigest()
    assert verify_model_file(candidate, expected_sha256=digest) == candidate


def test_open_backend_rejects_bad_num_poses_without_touching_disk(tmp_path: Path) -> None:
    with pytest.raises(PoseBackendError, match="num_poses"):
        mp_pose.open_mediapipe_backend(tmp_path / "model.task", num_poses=0)


def test_open_backend_missing_model_is_actionable(tmp_path: Path) -> None:
    missing = tmp_path / MODEL_ARTIFACT_FILENAME
    with pytest.raises(ModelFileError, match="M2.3 downloader is deferred"):
        mp_pose.open_mediapipe_backend(missing, num_poses=1)


def test_build_cache_identity_pins_adapter_model() -> None:
    identity = build_cache_identity("sha256:abc123", 30.0, 0.0)
    assert isinstance(identity, CacheIdentity)
    assert identity.model_name == MODEL_NAME
    assert identity.model_version == MODEL_VERSION
    assert identity.sampling_rate_hz == pytest.approx(30.0)
