"""Focused M3.1 tests for temporal feature extraction.

Deterministic, offline, and independent of private footage. All pose
sequences are synthetic.
"""

from __future__ import annotations

import pytest

from serve_review.detection.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureConfig,
    FeatureError,
    FeatureFrame,
    extract_features,
)
from serve_review.pose.schema import (
    NUM_KEYPOINTS,
    BodyKeypoint,
    FrameObservation,
    PersonBox,
    PersonObservation,
)


def _box() -> PersonBox:
    return PersonBox(x_min=0.1, y_min=0.1, x_max=0.7, y_max=0.9)


def _joints(
    *,
    wrist_y: float = 0.3,
    shoulder_y: float = 0.4,
    elbow_y: float = 0.35,
    missing: set[int] | None = None,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> tuple:
    absent = missing or set()
    joints: list = []
    for index in range(NUM_KEYPOINTS):
        if index in absent:
            joints.append(None)
            continue
        # Deterministic spread; key joints get explicit geometry.
        x = 0.30 + 0.01 * (index % 7) + shift_x
        y = 0.45 + 0.005 * (index % 5) + shift_y
        joints.append(BodyKeypoint(x=x, y=y, visibility=0.9))
    # Pin the kinematic skeleton (names -> indices per M2 schema).
    # 11 left_shoulder, 12 right_shoulder, 13 left_elbow, 14 right_elbow,
    # 15 left_wrist, 16 right_wrist, 23 left_hip, 24 right_hip.
    pins = {
        11: (0.40 + shift_x, shoulder_y + shift_y),
        12: (0.60 + shift_x, shoulder_y + shift_y),
        13: (0.38 + shift_x, elbow_y + shift_y),
        14: (0.62 + shift_x, elbow_y + shift_y),
        15: (0.36 + shift_x, wrist_y + shift_y),
        16: (0.64 + shift_x, wrist_y + shift_y),
        23: (0.42 + shift_x, 0.70 + shift_y),
        24: (0.58 + shift_x, 0.70 + shift_y),
    }
    for index, (px, py) in pins.items():
        if index in absent:
            continue
        joints[index] = BodyKeypoint(x=px, y=py, visibility=0.9)
    return tuple(joints)


def _person(**kwargs) -> PersonObservation:
    return PersonObservation(box=_box(), keypoints=_joints(**kwargs), score=0.8)


def _frame(time_seconds: float, person: PersonObservation | None) -> FrameObservation:
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(person,) if person is not None else (),
    )


def _static_sequence(n: int = 4, step: float = 0.1) -> list:
    return [_frame(i * step, _person()) for i in range(n)]


# --- Versioning -----------------------------------------------------------


def test_feature_schema_version_is_pinned() -> None:
    assert FEATURE_SCHEMA_VERSION == 1
    assert FeatureConfig().schema_version == 1
    assert FeatureFrame(time_seconds=0.0, has_person=False, visible_fraction=0.0).schema_version == 1


def test_feature_config_validates_window_and_reference() -> None:
    with pytest.raises(FeatureError):
        FeatureConfig(smoothing_window=0)
    with pytest.raises(FeatureError):
        FeatureConfig(smoothing_window=-2)
    with pytest.raises(FeatureError):
        FeatureConfig(smoothing_window=True)  # type: ignore[arg-type]
    with pytest.raises(FeatureError):
        FeatureConfig(motion_reference=0.0)
    with pytest.raises(FeatureError):
        FeatureConfig(motion_reference=float("inf"))
    with pytest.raises(FeatureError):
        FeatureConfig(schema_version=999)
    assert FeatureConfig.from_dict(FeatureConfig().to_dict()) == FeatureConfig()
    with pytest.raises(FeatureError):
        FeatureConfig.from_dict({**FeatureConfig().to_dict(), "extra": 1})


def test_extract_rejects_unordered_times_and_bad_types() -> None:
    frames = _static_sequence(2)
    swapped = [frames[1], frames[0]]
    with pytest.raises(FeatureError):
        extract_features(swapped)
    duplicate = [_frame(0.5, _person()), _frame(0.5, _person())]
    with pytest.raises(FeatureError):
        extract_features(duplicate)
    with pytest.raises(FeatureError):
        extract_features(["nope"])  # type: ignore[list-item]
    with pytest.raises(FeatureError):
        extract_features(_static_sequence(1), config="nope")  # type: ignore[arg-type]


def test_empty_sequence_yields_empty_tuple() -> None:
    assert extract_features([]) == ()


# --- Critical no-person rule ----------------------------------------------


def test_visible_no_person_visible_with_default_smoothing() -> None:
    config = FeatureConfig()  # default smoothing_window
    assert config.smoothing_window > 1
    frames = [
        _frame(0.0, _person()),
        _frame(0.1, None),
        _frame(0.2, _person()),
    ]
    features = extract_features(frames, config)
    assert len(features) == 3
    assert features[0].has_person is True
    assert features[1].has_person is False
    assert features[2].has_person is True
    # Smoothing must never bleed neighbor visibility into the gap.
    assert features[1].visible_fraction == 0.0
    assert features[1].torso_scale is None
    assert features[1].player_scale is None
    assert features[1].wrist_speed is None
    assert features[1].elbow_speed is None
    assert features[1].body_motion is None
    assert features[1].overhead_evidence is None
    assert features[1].rest_evidence is None
    assert features[1].motion_evidence is None
    # Neighbors stay fully visible: the gap is excluded from their
    # averages rather than dragging them down.
    assert features[0].visible_fraction == pytest.approx(1.0)
    assert features[2].visible_fraction == pytest.approx(1.0)


def test_no_person_stays_zero_for_large_windows() -> None:
    frames = [
        _frame(0.0, _person()),
        _frame(0.1, None),
        _frame(0.2, _person()),
    ]
    for window in (1, 3, 5, 7):
        features = extract_features(frames, FeatureConfig(smoothing_window=window))
        assert features[1].visible_fraction == 0.0
        assert features[1].overhead_evidence is None
        assert features[1].rest_evidence is None
        assert features[1].motion_evidence is None


def test_feature_frame_rejects_fabricated_no_person_evidence() -> None:
    with pytest.raises(FeatureError):
        FeatureFrame(
            time_seconds=0.0,
            has_person=False,
            visible_fraction=0.1,
        )
    with pytest.raises(FeatureError):
        FeatureFrame(
            time_seconds=0.0,
            has_person=False,
            visible_fraction=0.0,
            overhead_evidence=0.0,
        )
    with pytest.raises(FeatureError):
        FeatureFrame(
            time_seconds=0.0,
            has_person=False,
            visible_fraction=0.0,
            wrist_speed=0.0,
        )


# --- Determinism ----------------------------------------------------------


def test_extraction_is_deterministic() -> None:
    frames = _static_sequence(6)
    first = extract_features(frames)
    second = extract_features(frames)
    assert first == second
    assert [item.to_dict() for item in first] == [
        item.to_dict() for item in second
    ]


def test_feature_frame_json_codec_round_trips() -> None:
    features = extract_features(_static_sequence(3))
    for item in features:
        assert FeatureFrame.from_dict(item.to_dict()) == item


# --- Normalization invariance ---------------------------------------------


def _affine_person(*, scale: float, dx: float, dy: float) -> PersonObservation:
    joints = _joints()
    moved = []
    for joint in joints:
        if joint is None:
            moved.append(None)
            continue
        moved.append(
            BodyKeypoint(
                x=joint.x * scale + dx,
                y=joint.y * scale + dy,
                visibility=joint.visibility,
            )
        )
    box = _box()
    moved_box = PersonBox(
        x_min=box.x_min * scale + dx,
        y_min=box.y_min * scale + dy,
        x_max=box.x_max * scale + dx,
        y_max=box.y_max * scale + dy,
    )
    return PersonObservation(
        box=moved_box, keypoints=tuple(moved), score=0.8
    )


def test_translation_and_scale_normalization_invariance() -> None:
    # Two frames with identical relative motion; the second sequence is a
    # uniform affine map (scale 0.6 + translation) of the first, kept in
    # [0, 1] by construction.
    base = [
        _frame(0.0, _person(shift_x=0.0)),
        _frame(0.1, _person(shift_x=0.02)),
    ]
    mapped = [
        FrameObservation(
            time_seconds=frame.time_seconds,
            timestamp_ms=frame.timestamp_ms,
            persons=(
                _affine_person(scale=0.6, dx=0.15, dy=0.10)
                if index == 0
                else PersonObservation(
                    box=PersonBox(
                        x_min=0.1 * 0.6 + 0.15 + 0.02 * 0.6,
                        y_min=0.1 * 0.6 + 0.10,
                        x_max=0.7 * 0.6 + 0.15 + 0.02 * 0.6,
                        y_max=0.9 * 0.6 + 0.10,
                    ),
                    keypoints=tuple(
                        None
                        if joint is None
                        else BodyKeypoint(
                            x=(joint.x + (0.02 if index == 1 else 0.0)) * 0.6 + 0.15,
                            y=joint.y * 0.6 + 0.10,
                            visibility=joint.visibility,
                        )
                        for joint in _joints(
                            shift_x=0.02 if index == 1 else 0.0
                        )
                    ),
                    score=0.8,
                ),
            ),
        )
        for index, frame in enumerate(base)
    ]
    config = FeatureConfig(smoothing_window=1)
    plain = extract_features(base, config)
    shifted = extract_features(mapped, config)
    for field in (
        "visible_fraction",
        "wrist_speed",
        "elbow_speed",
        "body_motion",
        "overhead_evidence",
        "rest_evidence",
        "motion_evidence",
    ):
        assert getattr(shifted[1], field) == pytest.approx(
            getattr(plain[1], field), rel=1e-6, abs=1e-9
        )
    # Raw scales track the affine factor; normalized rates do not.
    assert shifted[1].torso_scale == pytest.approx(
        plain[1].torso_scale * 0.6, rel=1e-6
    )


# --- Unequal time steps ---------------------------------------------------


def test_source_time_derivatives_handle_unequal_steps() -> None:
    config = FeatureConfig(smoothing_window=1)
    slow = [
        _frame(0.0, _person(shift_x=0.0)),
        _frame(0.1, _person(shift_x=0.02)),
    ]
    fast_dt = [
        _frame(0.0, _person(shift_x=0.0)),
        _frame(0.2, _person(shift_x=0.02)),
    ]
    slow_speed = extract_features(slow, config)[1].wrist_speed
    fast_speed = extract_features(fast_dt, config)[1].wrist_speed
    assert slow_speed is not None and fast_speed is not None
    # Same normalized displacement over twice the source time: half rate.
    assert slow_speed == pytest.approx(2.0 * fast_speed, rel=1e-6)
    # First frame has no prior sample: speed is honestly missing.
    assert extract_features(slow, config)[0].wrist_speed is None
    assert extract_features(slow, config)[0].body_motion is None


# --- Missing joints -------------------------------------------------------


def test_missing_joints_propagate_honestly() -> None:
    config = FeatureConfig(smoothing_window=1)
    # Wrists 15/16 and elbows 13/14 missing: arm speeds and overhead
    # have no evidence, but the frame is still a person frame.
    missing_arms = {13, 14, 15, 16}
    frames = [
        _frame(0.0, _person(missing=missing_arms)),
        _frame(0.1, _person(missing=missing_arms)),
    ]
    features = extract_features(frames, config)
    assert features[1].has_person is True
    assert features[1].wrist_speed is None
    assert features[1].elbow_speed is None
    assert features[1].overhead_evidence is None
    # Other joints still support whole-body motion and visibility < 1.
    assert features[1].body_motion is not None
    assert features[1].visible_fraction == pytest.approx(
        (NUM_KEYPOINTS - len(missing_arms)) / NUM_KEYPOINTS
    )
    # Missing torso joints remove scale and therefore speeds.
    missing_torso = {11, 12, 23, 24}
    torso_frames = [
        _frame(0.0, PersonObservation(box=_box(), keypoints=_joints(missing=missing_torso), score=0.8)),
        _frame(0.1, PersonObservation(box=_box(), keypoints=_joints(missing=missing_torso), score=0.8)),
    ]
    torso_features = extract_features(torso_frames, config)
    assert torso_features[1].torso_scale is None
    # Box height still backs player_scale.
    assert torso_features[1].player_scale == pytest.approx(0.8)


def test_overhead_and_rest_evidence_semantics() -> None:
    config = FeatureConfig(smoothing_window=1)
    overhead = extract_features(
        [_frame(0.0, _person(wrist_y=0.20, shoulder_y=0.40))], config
    )[0]
    assert overhead.overhead_evidence == 1.0
    low = extract_features(
        [_frame(0.0, _person(wrist_y=0.60, shoulder_y=0.40))], config
    )[0]
    assert low.overhead_evidence == 0.0
    # Rest/motion pair is complementary and in [0, 1].
    moving = extract_features(
        [
            _frame(0.0, _person(shift_x=0.0)),
            _frame(0.05, _person(shift_x=0.05)),
        ],
        config,
    )[1]
    assert moving.motion_evidence is not None
    assert moving.rest_evidence is not None
    assert moving.motion_evidence + moving.rest_evidence == pytest.approx(1.0)
    assert 0.0 <= moving.motion_evidence <= 1.0
    # A static frame pair yields (near-)zero motion and full rest.
    static = extract_features(
        [
            _frame(0.0, _person(shift_x=0.0)),
            _frame(0.1, _person(shift_x=0.0)),
        ],
        config,
    )[1]
    assert static.body_motion == pytest.approx(0.0)
    assert static.motion_evidence == pytest.approx(0.0)
    assert static.rest_evidence == pytest.approx(1.0)
