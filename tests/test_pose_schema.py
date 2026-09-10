"""Focused M2.2 tests for portable pose observation schemas.

Deterministic, offline, and independent of private footage, model
execution, and caches on disk.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from serve_review.pose.schema import (
    JOINT_INDEX,
    JOINT_NAMES,
    NUM_KEYPOINTS,
    POSE_SCHEMA_VERSION,
    BodyKeypoint,
    CacheIdentity,
    FrameObservation,
    KeypointError,
    PersonBox,
    PersonBoxError,
    PersonObservation,
    PersonObservationError,
    FrameObservationError,
    CacheIdentityError,
    PoseError,
    SchemaVersionError,
)


def make_box(**overrides) -> PersonBox:
    fields = {"x_min": 0.1, "y_min": 0.2, "x_max": 0.6, "y_max": 0.9}
    fields.update(overrides)
    return PersonBox(**fields)


def make_keypoint(**overrides) -> BodyKeypoint:
    fields = {"x": 0.5, "y": 0.5, "visibility": 0.9}
    fields.update(overrides)
    return BodyKeypoint(**fields)


def make_joints(*, missing: set[int] | None = None) -> tuple:
    absent = missing or set()
    joints: list = []
    for index in range(NUM_KEYPOINTS):
        if index in absent:
            joints.append(None)
        else:
            joints.append(
                BodyKeypoint(
                    x=0.05 + 0.9 * (index / (NUM_KEYPOINTS - 1)),
                    y=0.1,
                    visibility=0.8,
                )
            )
    return tuple(joints)


def make_person(**overrides) -> PersonObservation:
    fields = {
        "box": make_box(),
        "keypoints": make_joints(missing={0, 5, 32}),
        "score": 0.75,
    }
    fields.update(overrides)
    return PersonObservation(**fields)


def make_frame(**overrides) -> FrameObservation:
    time_seconds = overrides.pop("time_seconds", 1.0)
    fields = {
        "time_seconds": time_seconds,
        "timestamp_ms": int(round(float(time_seconds) * 1000)),
        "persons": (make_person(),),
    }
    fields.update(overrides)
    return FrameObservation(**fields)


def make_identity(**overrides) -> CacheIdentity:
    fields = {
        "source_fingerprint": "sha256:abc123",
        "model_name": "pose-landmarker-heavy",
        "model_version": "0.10.32-test",
        "sampling_rate_hz": 30.0,
        "sampling_start_seconds": 0.0,
    }
    fields.update(overrides)
    return CacheIdentity(**fields)


# --- Joint order ----------------------------------------------------------


def test_joint_order_is_33_mediapipe_entries() -> None:
    assert POSE_SCHEMA_VERSION == 1
    assert NUM_KEYPOINTS == 33
    assert len(JOINT_NAMES) == 33
    assert len(set(JOINT_NAMES)) == 33
    assert JOINT_NAMES[0] == "nose"
    assert JOINT_NAMES[11] == "left_shoulder"
    assert JOINT_NAMES[12] == "right_shoulder"
    assert JOINT_NAMES[32] == "right_foot_index"
    assert JOINT_INDEX["nose"] == 0
    assert JOINT_INDEX["right_foot_index"] == 32


# --- PersonBox ------------------------------------------------------------


def test_person_box_is_immutable_and_reports_size() -> None:
    box = make_box()
    assert box.width == pytest.approx(0.5)
    assert box.height == pytest.approx(0.7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        box.x_min = 0.0  # type: ignore[misc]


def test_person_box_accepts_boundary_coordinates() -> None:
    full = PersonBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    assert full.width == pytest.approx(1.0)
    assert PersonBox.from_dict(full.to_dict()) == full


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_min": -0.01, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
        {"x_min": 0.0, "y_min": 0.0, "x_max": 1.01, "y_max": 0.5},
        {"x_min": 0.5, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},  # zero width
        {"x_min": 0.7, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},  # inverted x
        {"x_min": 0.0, "y_min": 0.6, "x_max": 0.5, "y_max": 0.6},  # zero height
        {"x_min": 0.0, "y_min": 0.8, "x_max": 0.5, "y_max": 0.5},  # inverted y
        {"x_min": float("nan"), "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
        {"x_min": 0.0, "y_min": float("inf"), "x_max": 0.5, "y_max": 0.5},
        {"x_min": True, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
        {"x_min": "0.1", "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
        {"x_min": None, "y_min": 0.0, "x_max": 0.5, "y_max": 0.5},
    ],
)
def test_person_box_rejects_bad_coordinates(kwargs) -> None:
    with pytest.raises((PersonBoxError, SchemaVersionError)):
        make_box(**kwargs)


def test_person_box_json_round_trip_is_deterministic() -> None:
    box = make_box()
    first = box.to_json()
    assert box.to_json() == first
    assert list(json.loads(first)) == sorted(json.loads(first))
    assert PersonBox.from_json(first) == box
    assert PersonBox.from_json(first.encode("utf-8")) == box
    assert PersonBox.from_dict(box.to_dict()) == box
    with pytest.raises(PoseError):
        PersonBox.from_json("not json{")


def test_person_box_rejects_missing_unknown_and_newer_keys() -> None:
    payload = make_box().to_dict()
    del payload["x_max"]
    with pytest.raises(PersonBoxError):
        PersonBox.from_dict(payload)
    with pytest.raises(PersonBoxError):
        PersonBox.from_dict({**make_box().to_dict(), "extra": 1})
    with pytest.raises(SchemaVersionError):
        PersonBox.from_dict({**make_box().to_dict(), "schema_version": 999})
    with pytest.raises(SchemaVersionError):
        PersonBox.from_dict({**make_box().to_dict(), "schema_version": 0})


# --- BodyKeypoint ---------------------------------------------------------


def test_body_keypoint_accepts_boundaries_and_round_trips() -> None:
    corner = BodyKeypoint(x=0.0, y=1.0, visibility=0.0)
    assert BodyKeypoint.from_dict(corner.to_dict()) == corner
    assert BodyKeypoint.from_json(corner.to_json()) == corner
    full = BodyKeypoint(x=1.0, y=1.0, visibility=1.0)
    assert BodyKeypoint.from_dict(full.to_dict()) == full


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x": -0.01, "y": 0.5, "visibility": 0.5},
        {"x": 1.01, "y": 0.5, "visibility": 0.5},
        {"x": 0.5, "y": 0.5, "visibility": -0.1},
        {"x": 0.5, "y": 0.5, "visibility": 1.1},
        {"x": float("nan"), "y": 0.5, "visibility": 0.5},
        {"x": 0.5, "y": float("inf"), "visibility": 0.5},
        {"x": 0.5, "y": 0.5, "visibility": float("nan")},
        {"x": True, "y": 0.5, "visibility": 0.5},
        {"x": "0.5", "y": 0.5, "visibility": 0.5},
        {"x": 0.5, "y": 0.5, "visibility": None},
    ],
)
def test_body_keypoint_rejects_bad_values(kwargs) -> None:
    with pytest.raises(KeypointError):
        make_keypoint(**kwargs)


def test_body_keypoint_rejects_missing_and_unknown_keys() -> None:
    payload = make_keypoint().to_dict()
    del payload["visibility"]
    with pytest.raises(KeypointError):
        BodyKeypoint.from_dict(payload)
    with pytest.raises(KeypointError):
        BodyKeypoint.from_dict({**make_keypoint().to_dict(), "schema_version": 1})


# --- PersonObservation ----------------------------------------------------


def test_person_observation_tracks_explicit_missing_joints() -> None:
    person = make_person()
    assert person.present_count == NUM_KEYPOINTS - 3
    assert person.keypoints[0] is None
    assert isinstance(person.keypoints[1], BodyKeypoint)
    assert person.joint("nose") is None
    assert isinstance(person.joint("left_eye"), BodyKeypoint)
    # Missing joints serialize as explicit nulls in JOINT_NAMES order.
    payload = person.to_dict()
    assert len(payload["keypoints"]) == 33
    assert payload["keypoints"][0] is None
    assert payload["keypoints"][1]["x"] == pytest.approx(
        0.05 + 0.9 * (1 / 32)
    )
    with pytest.raises(PersonObservationError):
        person.joint("racket_tip")


def test_person_observation_rejects_all_missing_joints() -> None:
    with pytest.raises(PersonObservationError):
        make_person(keypoints=tuple([None] * NUM_KEYPOINTS))


@pytest.mark.parametrize("count", [0, 1, 32, 34, 64])
def test_person_observation_requires_exactly_33_joints(count: int) -> None:
    joints = tuple([make_keypoint()] * count)
    with pytest.raises(PersonObservationError):
        make_person(keypoints=joints)


def test_person_observation_rejects_bad_joint_types_and_scores() -> None:
    joints = list(make_joints())
    joints[3] = {"x": 0.5, "y": 0.5, "visibility": 0.5}  # type: ignore[list-item]
    with pytest.raises(PersonObservationError):
        make_person(keypoints=tuple(joints))
    for bad_score in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None):
        with pytest.raises(PersonObservationError):
            make_person(score=bad_score)
    with pytest.raises(PersonObservationError):
        make_person(box={"x_min": 0})  # type: ignore[arg-type]


def test_person_observation_json_round_trip_is_deterministic() -> None:
    person = make_person()
    first = person.to_json()
    assert person.to_json() == first
    assert list(json.loads(first)) == sorted(json.loads(first))
    assert PersonObservation.from_json(first) == person
    assert PersonObservation.from_dict(person.to_dict()) == person


def test_person_observation_rejects_missing_unknown_and_newer_keys() -> None:
    payload = make_person().to_dict()
    del payload["score"]
    with pytest.raises(PersonObservationError):
        PersonObservation.from_dict(payload)
    with pytest.raises(PersonObservationError):
        PersonObservation.from_dict({**make_person().to_dict(), "extra": 1})
    with pytest.raises(SchemaVersionError):
        PersonObservation.from_dict(
            {**make_person().to_dict(), "schema_version": 999}
        )
    nested = make_person().to_dict()
    nested["box"] = {**nested["box"], "schema_version": 99}
    with pytest.raises(PersonObservationError):
        PersonObservation.from_dict(nested)
    nested2 = make_person().to_dict()
    nested2["keypoints"][1] = {"x": 2.0, "y": 0.5, "visibility": 0.5}
    with pytest.raises(PersonObservationError):
        PersonObservation.from_dict(nested2)


# --- FrameObservation -----------------------------------------------------


def test_frame_observation_allows_explicit_no_person_frame() -> None:
    empty = make_frame(persons=())
    assert empty.has_person is False
    assert make_frame().has_person is True
    assert FrameObservation.from_dict(empty.to_dict()) == empty


def test_frame_observation_keeps_canonical_time_separate_from_ms() -> None:
    # 1/30 s is not an integral millisecond: the float stays canonical.
    frame = FrameObservation(
        time_seconds=1 / 30.0, timestamp_ms=33, persons=()
    )
    assert frame.time_seconds == pytest.approx(1 / 30.0)
    assert frame.timestamp_ms == 33
    assert abs(frame.time_seconds * 1000 - frame.timestamp_ms) > 0.1
    assert FrameObservation.from_dict(frame.to_dict()) == frame


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_seconds": -0.1, "timestamp_ms": -100, "persons": ()},
        {"time_seconds": float("nan"), "timestamp_ms": 0, "persons": ()},
        {"time_seconds": float("inf"), "timestamp_ms": 0, "persons": ()},
        {"time_seconds": True, "timestamp_ms": 0, "persons": ()},
        {"time_seconds": "1.0", "timestamp_ms": 1000, "persons": ()},
        # timestamp_ms must equal round(time_seconds * 1000) exactly.
        {"time_seconds": 1.0, "timestamp_ms": 1001, "persons": ()},
        {"time_seconds": 1.0, "timestamp_ms": 999, "persons": ()},
        {"time_seconds": 1.0, "timestamp_ms": 1000.0, "persons": ()},
        {"time_seconds": 1.0, "timestamp_ms": -1, "persons": ()},
    ],
)
def test_frame_observation_rejects_bad_times(kwargs) -> None:
    with pytest.raises(FrameObservationError):
        FrameObservation(**kwargs)


def test_frame_observation_json_round_trip_is_deterministic() -> None:
    frame = make_frame()
    first = frame.to_json()
    assert frame.to_json() == first
    assert list(json.loads(first)) == sorted(json.loads(first))
    assert FrameObservation.from_json(first) == frame
    assert FrameObservation.from_json(first.encode("utf-8")) == frame
    with pytest.raises(PoseError):
        FrameObservation.from_json("{oops")


def test_frame_observation_rejects_missing_unknown_and_newer_keys() -> None:
    payload = make_frame().to_dict()
    del payload["persons"]
    with pytest.raises(FrameObservationError):
        FrameObservation.from_dict(payload)
    with pytest.raises(FrameObservationError):
        FrameObservation.from_dict({**make_frame().to_dict(), "bogus": True})
    with pytest.raises(SchemaVersionError):
        FrameObservation.from_dict(
            {**make_frame().to_dict(), "schema_version": 999}
        )
    with pytest.raises(FrameObservationError):
        FrameObservation.from_dict({**make_frame().to_dict(), "persons": "nope"})


# --- CacheIdentity --------------------------------------------------------


def test_cache_identity_is_immutable_and_round_trips() -> None:
    identity = make_identity()
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.model_name = "other"  # type: ignore[misc]
    first = identity.to_json()
    assert identity.to_json() == first
    assert list(json.loads(first)) == sorted(json.loads(first))
    assert CacheIdentity.from_json(first) == identity
    assert CacheIdentity.from_dict(identity.to_dict()) == identity


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_fingerprint": ""},
        {"source_fingerprint": "   "},
        {"source_fingerprint": None},
        {"model_name": ""},
        {"model_version": ""},
        {"sampling_rate_hz": 0},
        {"sampling_rate_hz": -30.0},
        {"sampling_rate_hz": float("nan")},
        {"sampling_rate_hz": float("inf")},
        {"sampling_rate_hz": "30"},
        {"sampling_rate_hz": True},
        {"sampling_start_seconds": -0.1},
        {"sampling_start_seconds": float("nan")},
        {"sampling_start_seconds": True},
    ],
)
def test_cache_identity_rejects_invalid_values(kwargs) -> None:
    with pytest.raises((CacheIdentityError, SchemaVersionError)):
        make_identity(**kwargs)


def test_cache_identity_rejects_missing_unknown_and_newer_keys() -> None:
    payload = make_identity().to_dict()
    del payload["model_name"]
    with pytest.raises(CacheIdentityError):
        CacheIdentity.from_dict(payload)
    with pytest.raises(CacheIdentityError):
        CacheIdentity.from_dict({**make_identity().to_dict(), "extra": 1})
    with pytest.raises(SchemaVersionError):
        CacheIdentity.from_dict(
            {**make_identity().to_dict(), "schema_version": 999}
        )
