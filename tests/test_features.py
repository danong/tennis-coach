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
    assert FEATURE_SCHEMA_VERSION == 2
    assert FeatureConfig().schema_version == 2
    assert FeatureFrame(time_seconds=0.0, has_person=False, visible_fraction=0.0).schema_version == 2


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


# --- Version-2 proximal geometry -------------------------------------------


def _custom_person(pins: dict[int, tuple[float, float]], vis: float = 0.9) -> PersonObservation:
    joints: list = []
    for index in range(NUM_KEYPOINTS):
        if index in pins:
            px, py = pins[index]
            joints.append(BodyKeypoint(x=px, y=py, visibility=vis))
        else:
            joints.append(
                BodyKeypoint(x=0.50 + 0.001 * index, y=0.50, visibility=vis)
            )
    return PersonObservation(box=_box(), keypoints=tuple(joints), score=0.8)


def _angle_pose(*, elbow_xy=None, shoulder_xy=None, wrist_xy=None,
                hip_xy=None, knee_xy=None, ankle_xy=None,
                vis: float = 0.9) -> PersonObservation:
    pins: dict[int, tuple[float, float]] = {
        11: (0.40, 0.40), 12: (0.60, 0.40),
        23: (0.42, 0.70), 24: (0.58, 0.70),
        13: (0.38, 0.50), 14: (0.62, 0.50),
        15: (0.36, 0.60), 16: (0.64, 0.60),
        25: (0.44, 0.80), 26: (0.56, 0.80),
        27: (0.44, 0.90), 28: (0.56, 0.90),
    }
    if shoulder_xy is not None:
        pins[13 - 2] = shoulder_xy  # 11 left_shoulder
    if elbow_xy is not None:
        pins[13] = elbow_xy
    if wrist_xy is not None:
        pins[15] = wrist_xy
    if hip_xy is not None:
        pins[23] = hip_xy
    if knee_xy is not None:
        pins[25] = knee_xy
    if ankle_xy is not None:
        pins[27] = ankle_xy
    return _custom_person(pins, vis=vis)


def test_feature_config_visibility_floor_validation() -> None:
    assert FeatureConfig().visibility_floor == pytest.approx(0.4)
    for bad in (-0.1, 1.5, float("nan"), float("inf"), True, "high"):
        with pytest.raises(FeatureError):
            FeatureConfig(visibility_floor=bad)  # type: ignore[arg-type]
    assert FeatureConfig(visibility_floor=0.0).visibility_floor == 0.0
    assert FeatureConfig(visibility_floor=1.0).visibility_floor == 1.0
    assert FeatureConfig.from_dict(FeatureConfig().to_dict()) == FeatureConfig()
    with pytest.raises(FeatureError):
        FeatureConfig.from_dict({**FeatureConfig().to_dict(), "extra": 1})
    # A version-1 config dict (no visibility floor) is rejected, not
    # silently reinterpreted.
    v1 = {
        "motion_reference": 2.0,
        "schema_version": 1,
        "smoothing_window": 3,
    }
    with pytest.raises(FeatureError):
        FeatureConfig.from_dict(v1)
    with pytest.raises(FeatureError):
        FeatureConfig.from_dict(
            {**FeatureConfig().to_dict(), "schema_version": 999}
        )


def test_joint_angle_correctness_on_hand_computed_poses() -> None:
    config = FeatureConfig(smoothing_window=1)
    # Right angle at the left elbow: shoulder above, wrist to the side.
    bent = extract_features(
        [_frame(0.0, _angle_pose(
            shoulder_xy=(0.40, 0.40),
            elbow_xy=(0.40, 0.50),
            wrist_xy=(0.50, 0.50),
        ))],
        config,
    )[0]
    assert bent.elbow_flexion_left == pytest.approx(90.0, abs=1e-9)
    # Straight arm reports 180 (fully extended).
    straight = extract_features(
        [_frame(0.0, _angle_pose(
            shoulder_xy=(0.40, 0.40),
            elbow_xy=(0.40, 0.50),
            wrist_xy=(0.40, 0.60),
        ))],
        config,
    )[0]
    assert straight.elbow_flexion_left == pytest.approx(180.0, abs=1e-9)
    # Right angle at the left knee.
    knee = extract_features(
        [_frame(0.0, _angle_pose(
            hip_xy=(0.45, 0.60),
            knee_xy=(0.45, 0.70),
            ankle_xy=(0.55, 0.70),
        ))],
        config,
    )[0]
    assert knee.knee_flexion_left == pytest.approx(90.0, abs=1e-9)
    # Untouched right side still computes from its default pins.
    assert bent.elbow_flexion_right is not None
    assert bent.knee_flexion_left is not None
    # Missing required joints propagate None.
    missing = extract_features(
        [_frame(0.0, _person(missing={11, 13, 15, 23, 25, 27}))],
        config,
    )[0]
    assert missing.elbow_flexion_left is None
    assert missing.knee_flexion_left is None
    # Degenerate limb (elbow coincident with shoulder) yields None.
    degenerate = extract_features(
        [_frame(0.0, _angle_pose(
            shoulder_xy=(0.40, 0.40),
            elbow_xy=(0.40, 0.40),
            wrist_xy=(0.50, 0.50),
        ))],
        config,
    )[0]
    assert degenerate.elbow_flexion_left is None


def test_shoulder_tilt_level_and_hand_computed_lean() -> None:
    config = FeatureConfig(smoothing_window=1)
    level = extract_features(
        [_frame(0.0, _custom_person({
            11: (0.40, 0.40), 12: (0.60, 0.40),
            23: (0.42, 0.70), 24: (0.58, 0.70),
        }))],
        config,
    )[0]
    assert level.shoulder_tilt == pytest.approx(0.0, abs=1e-9)
    # Right shoulder dropped by 0.1: hand-computed tilt is
    # asin(-0.025 / (hypot(0.2, 0.1) * 0.25)) == -26.565051... degrees.
    import math as _math

    dropped = extract_features(
        [_frame(0.0, _custom_person({
            11: (0.40, 0.40), 12: (0.60, 0.50),
            23: (0.42, 0.70), 24: (0.58, 0.70),
        }))],
        config,
    )[0]
    expected = _math.degrees(
        _math.asin(-0.025 / (_math.hypot(0.2, 0.1) * 0.25))
    )
    assert dropped.shoulder_tilt == pytest.approx(expected, abs=1e-9)
    assert dropped.shoulder_tilt < 0.0  # right shoulder lower -> negative
    # Missing torso joints propagate None.
    missing = extract_features(
        [_frame(0.0, _person(missing={11, 12, 23, 24}))],
        config,
    )[0]
    assert missing.shoulder_tilt is None


def _rotated_person(angle_deg: float, dx: float, dy: float) -> PersonObservation:
    import math as _math

    joints = _joints()
    theta = _math.radians(angle_deg)
    cos_t, sin_t = _math.cos(theta), _math.sin(theta)
    cx, cy = 0.50, 0.50
    moved = []
    for joint in joints:
        assert joint is not None
        px, py = joint.x - cx, joint.y - cy
        moved.append(
            BodyKeypoint(
                x=(px * cos_t - py * sin_t) + cx + dx,
                y=(px * sin_t + py * cos_t) + cy + dy,
                visibility=joint.visibility,
            )
        )
    return PersonObservation(box=_box(), keypoints=tuple(moved), score=0.8)


def test_shoulder_tilt_invariant_under_rigid_rotation_and_translation() -> None:
    import math as _math

    config = FeatureConfig(smoothing_window=1)
    base = extract_features([_frame(0.0, _person())], config)[0]
    # A 20-degree rigid rotation plus translation keeps every joint in
    # [0, 1] by construction (checked below) while screen-space
    # shoulder orientation changes substantially.
    tilted = _rotated_person(20.0, 0.05, -0.03)
    for joint in tilted.keypoints:
        assert joint is not None
        assert 0.0 <= joint.x <= 1.0
        assert 0.0 <= joint.y <= 1.0
    rotated = extract_features([_frame(0.0, tilted)], config)[0]
    assert rotated.shoulder_tilt is not None
    assert base.shoulder_tilt is not None
    assert rotated.shoulder_tilt == pytest.approx(base.shoulder_tilt, abs=1e-9)
    # Screen-space geometry really did move: raw shoulder y differs.
    assert tilted.keypoints[11].y != pytest.approx(
        _person().keypoints[11].y
    )
    # Pure translation alone also leaves tilt unchanged.
    shifted = extract_features(
        [_frame(0.0, _person(shift_x=0.03, shift_y=0.02))], config
    )[0]
    assert shifted.shoulder_tilt == pytest.approx(base.shoulder_tilt, abs=1e-9)
    # And joint angles are rigid-motion invariant too.
    for field in (
        "elbow_flexion_left",
        "elbow_flexion_right",
        "knee_flexion_left",
        "knee_flexion_right",
    ):
        assert getattr(rotated, field) == pytest.approx(
            getattr(base, field), abs=1e-9
        )


def test_visibility_floor_exclusion_before_angle_computation() -> None:
    config = FeatureConfig(smoothing_window=1)
    pose = _angle_pose(
        shoulder_xy=(0.40, 0.40),
        elbow_xy=(0.40, 0.50),
        wrist_xy=(0.50, 0.50),
    )
    assert extract_features([_frame(0.0, pose)], config)[0].elbow_flexion_left == pytest.approx(90.0, abs=1e-9)
    # Same geometry with the wrist below the default 0.4 floor is
    # excluded: the angle is honestly None.
    joints = list(pose.keypoints)
    dim = joints[15]
    assert dim is not None
    joints[15] = BodyKeypoint(x=dim.x, y=dim.y, visibility=0.05)
    dimmed = PersonObservation(box=pose.box, keypoints=tuple(joints), score=0.8)
    gated = extract_features([_frame(0.0, dimmed)], config)[0]
    assert gated.elbow_flexion_left is None
    # A permissive floor re-admits the same observation.
    open_config = FeatureConfig(smoothing_window=1, visibility_floor=0.0)
    assert extract_features(
        [_frame(0.0, dimmed)], open_config
    )[0].elbow_flexion_left == pytest.approx(90.0, abs=1e-9)
    # The version-1 auxiliary signals ignore the floor: wrist speed
    # still tracks the dimmed wrist across frames.
    pair = [
        _frame(0.0, dimmed),
        _frame(
            0.1,
            PersonObservation(
                box=dimmed.box,
                keypoints=tuple(
                    None
                    if joint is None
                    else (
                        BodyKeypoint(
                            x=joint.x + 0.02, y=joint.y, visibility=joint.visibility
                        )
                        if index == 15
                        else joint
                    )
                    for index, joint in enumerate(dimmed.keypoints)
                ),
                score=0.8,
            ),
        ),
    ]
    speeds = extract_features(pair, config)
    assert speeds[1].wrist_speed is not None


def test_torso_displacement_stillness_vs_motion() -> None:
    config = FeatureConfig(smoothing_window=1)
    # Static tripod: no 1.0 s history at first, then ~zero displacement.
    static = extract_features(
        [
            _frame(0.0, _person()),
            _frame(0.5, _person()),
            _frame(1.0, _person()),
            _frame(1.5, _person()),
        ],
        config,
    )
    assert static[0].torso_displacement is None
    assert static[1].torso_displacement is None
    assert static[2].torso_displacement == pytest.approx(0.0, abs=1e-9)
    assert static[3].torso_displacement == pytest.approx(0.0, abs=1e-9)
    # A 0.1 torso shift appearing at t=1.0 over a 0.3 torso scale gives
    # displacement ~= 1/3; the t=0.5 frame still has no 1.0 s history.
    moving = extract_features(
        [
            _frame(0.0, _person(shift_x=0.0)),
            _frame(0.5, _person(shift_x=0.0)),
            _frame(1.0, _person(shift_x=0.10)),
        ],
        config,
    )
    assert moving[1].torso_displacement is None
    assert moving[2].torso_displacement == pytest.approx(0.10 / 0.30, rel=1e-6)
    # Limb-only motion does not move the torso baseline: shifting just
    # the wrists leaves displacement at zero.
    base = _person()
    waved_joints = tuple(
        None
        if joint is None
        else (
            BodyKeypoint(x=joint.x + 0.05, y=joint.y, visibility=joint.visibility)
            if index in (15, 16)
            else joint
        )
        for index, joint in enumerate(base.keypoints)
    )
    waved = PersonObservation(box=base.box, keypoints=waved_joints, score=0.8)
    limb_only = extract_features(
        [
            _frame(0.0, base),
            _frame(0.5, base),
            _frame(1.0, waved),
        ],
        config,
    )
    assert limb_only[2].torso_displacement == pytest.approx(0.0, abs=1e-9)


def test_new_channels_no_person_and_codec_round_trips() -> None:
    frames = [
        _frame(0.0, _person()),
        _frame(0.1, None),
        _frame(0.2, _person()),
    ]
    for window in (1, 3, 7):
        features = extract_features(frames, FeatureConfig(smoothing_window=window))
        gap = features[1]
        assert gap.elbow_flexion_left is None
        assert gap.elbow_flexion_right is None
        assert gap.knee_flexion_left is None
        assert gap.knee_flexion_right is None
        assert gap.shoulder_tilt is None
        assert gap.torso_displacement is None
    # No-person construction with new fields set is rejected.
    with pytest.raises(FeatureError):
        FeatureFrame(
            time_seconds=0.0,
            has_person=False,
            visible_fraction=0.0,
            shoulder_tilt=0.0,
        )
    with pytest.raises(FeatureError):
        FeatureFrame(
            time_seconds=0.0,
            has_person=False,
            visible_fraction=0.0,
            torso_displacement=0.0,
        )
    # Range validation on the new fields.
    with pytest.raises(FeatureError):
        FeatureFrame(
            time_seconds=0.0, has_person=True, visible_fraction=1.0,
            elbow_flexion_left=200.0,
        )
    with pytest.raises(FeatureError):
        FeatureFrame(
            time_seconds=0.0, has_person=True, visible_fraction=1.0,
            shoulder_tilt=91.0,
        )
    # Full round trips carry the new channels.
    rich = extract_features(_static_sequence(4))
    for item in rich:
        assert FeatureFrame.from_dict(item.to_dict()) == item
    assert FeatureConfig.from_dict(FeatureConfig().to_dict()) == FeatureConfig()
    # Strict validation: unknown keys and newer versions rejected, and
    # a version-1 frame dict (missing the new keys) is rejected.
    good = rich[0].to_dict()
    with pytest.raises(FeatureError):
        FeatureFrame.from_dict({**good, "extra": 1.0})
    with pytest.raises(FeatureError):
        FeatureFrame.from_dict({**good, "schema_version": 999})
    v1_keys = {
        "body_motion", "elbow_speed", "has_person", "motion_evidence",
        "overhead_evidence", "player_scale", "rest_evidence",
        "schema_version", "time_seconds", "torso_scale",
        "visible_fraction", "wrist_speed",
    }
    with pytest.raises(FeatureError):
        FeatureFrame.from_dict({key: good[key] for key in v1_keys})


def test_new_channels_are_deterministic() -> None:
    frames = _static_sequence(6)
    first = extract_features(frames)
    second = extract_features(frames)
    for left, right in zip(first, second):
        assert left.elbow_flexion_left == right.elbow_flexion_left
        assert left.elbow_flexion_right == right.elbow_flexion_right
        assert left.knee_flexion_left == right.knee_flexion_left
        assert left.knee_flexion_right == right.knee_flexion_right
        assert left.shoulder_tilt == right.shoulder_tilt
        assert left.torso_displacement == right.torso_displacement
    assert first == second


def test_new_channels_smoothing_has_explicit_boundaries() -> None:
    # A single bent-elbow frame amid straight arms: with window=1 the
    # raw value shows exactly; wider windows blend honestly.
    poses = [
        _angle_pose(
            shoulder_xy=(0.40, 0.40),
            elbow_xy=(0.40, 0.50),
            wrist_xy=(0.40, 0.60),
        ),
        _angle_pose(
            shoulder_xy=(0.40, 0.40),
            elbow_xy=(0.40, 0.50),
            wrist_xy=(0.50, 0.50),
        ),
        _angle_pose(
            shoulder_xy=(0.40, 0.40),
            elbow_xy=(0.40, 0.50),
            wrist_xy=(0.40, 0.60),
        ),
    ]
    exact = extract_features(
        [_frame(0.0, poses[0]), _frame(0.1, poses[1]), _frame(0.2, poses[2])],
        FeatureConfig(smoothing_window=1),
    )
    assert exact[1].elbow_flexion_left == pytest.approx(90.0, abs=1e-9)
    assert exact[0].elbow_flexion_left == pytest.approx(180.0, abs=1e-9)
    wide = extract_features(
        [_frame(0.0, poses[0]), _frame(0.1, poses[1]), _frame(0.2, poses[2])],
        FeatureConfig(smoothing_window=3),
    )
    assert wide[1].elbow_flexion_left == pytest.approx((180.0 + 90.0 + 180.0) / 3)
