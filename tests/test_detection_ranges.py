"""Focused M3.2 tests for the candidate range state machine.

Deterministic, offline, synthetic only: every feature sequence is built
directly as :class:`FeatureFrame` values, never from private footage.
"""

from __future__ import annotations

import pytest

from serve_review.detection.features import FeatureFrame
from serve_review.detection.ranges import (
    RANGE_SCHEMA_VERSION,
    CandidateDetector,
    CandidateRange,
    RangeConfig,
    RangeError,
    find_candidates,
)

STEP = 0.1
REST = 0.05
SWING = 0.85


def _mk(
    moment: float,
    *,
    motion: float | None = REST,
    overhead: float | None = 0.0,
    person: bool = True,
    visible: float = 1.0,
) -> FeatureFrame:
    if not person:
        return FeatureFrame(
            time_seconds=moment, has_person=False, visible_fraction=0.0
        )
    if motion is None:
        return FeatureFrame(
            time_seconds=moment,
            has_person=True,
            visible_fraction=visible,
            overhead_evidence=overhead,
            rest_evidence=None,
            motion_evidence=None,
        )
    return FeatureFrame(
        time_seconds=moment,
        has_person=True,
        visible_fraction=visible,
        overhead_evidence=overhead,
        rest_evidence=1.0 - float(motion),
        motion_evidence=float(motion),
    )


def _cfg(**overrides) -> RangeConfig:
    base = {
        "active_threshold": 0.4,
        "inactive_threshold": 0.2,
        "min_duration_seconds": 0.6,
        "max_duration_seconds": 8.0,
        "merge_gap_seconds": 0.5,
        "min_visible_fraction": 0.3,
        "require_overhead": True,
    }
    base.update(overrides)
    return RangeConfig(**base)


def _burst(start: float, count: int, *, overhead: float | None = 1.0) -> list[FeatureFrame]:
    return [
        _mk(start + i * STEP, motion=SWING, overhead=overhead)
        for i in range(count)
    ]


def _rest(start: float, count: int) -> list[FeatureFrame]:
    return [_mk(start + i * STEP, motion=REST, overhead=0.0) for i in range(count)]


# --- Empty / ordering -----------------------------------------------------


def test_range_schema_version_is_pinned() -> None:
    assert RANGE_SCHEMA_VERSION == 1
    assert RangeConfig().schema_version == 1
    assert CandidateRange(start_seconds=0.0, end_seconds=0.5).schema_version == 1


def test_empty_input_yields_empty_result() -> None:
    assert find_candidates([]) == ()
    assert find_candidates([], _cfg()) == ()
    assert CandidateDetector(_cfg()).finish() == ()


def test_strictly_increasing_times_required() -> None:
    frames = _burst(0.8, 8) + _burst(0.0, 8)
    with pytest.raises(RangeError):
        find_candidates(frames)
    duplicate = [_mk(0.5, motion=SWING), _mk(0.5, motion=SWING)]
    with pytest.raises(RangeError):
        find_candidates(duplicate)
    detector = CandidateDetector(_cfg())
    detector.feed(_mk(1.0, motion=SWING))
    with pytest.raises(RangeError):
        detector.feed(_mk(0.9, motion=SWING))
    with pytest.raises(RangeError):
        find_candidates(["nope"])  # type: ignore[list-item]
    with pytest.raises(RangeError):
        find_candidates([_mk(0.0)], config="nope")  # type: ignore[arg-type]
    with pytest.raises(RangeError):
        CandidateDetector(config="nope")  # type: ignore[arg-type]
    with pytest.raises(RangeError):
        CandidateDetector(_cfg()).feed("nope")  # type: ignore[arg-type]


# --- Zero / one / multiple -------------------------------------------------


def test_zero_attempts_all_rest() -> None:
    frames = _rest(0.0, 30)
    assert find_candidates(frames, _cfg()) == ()


def test_single_attempt_span() -> None:
    frames = _rest(0.0, 10) + _burst(1.0, 11) + _rest(2.1, 10)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 1
    assert ranges[0].start_seconds == pytest.approx(1.0)
    assert ranges[0].end_seconds == pytest.approx(2.0)
    assert ranges[0].duration_seconds == pytest.approx(1.0)


def test_multiple_attempts_with_pause() -> None:
    frames = (
        _rest(0.0, 5)
        + _burst(0.5, 8)
        + _rest(1.3, 20)
        + _burst(3.3, 8)
        + _rest(4.1, 5)
    )
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 2
    assert ranges[0].start_seconds == pytest.approx(0.5)
    assert ranges[1].start_seconds == pytest.approx(3.3)
    assert ranges[0].end_seconds < ranges[1].start_seconds


def test_hysteresis_band_keeps_segment_alive() -> None:
    # Mid dip to 0.3 sits below active (0.4) but above inactive (0.2):
    # the open segment must survive it as one candidate.
    frames = _burst(10.0, 5)
    frames += [_mk(10.5 + i * STEP, motion=0.3, overhead=1.0) for i in range(3)]
    frames += _burst(10.8, 5)
    frames = _rest(9.0, 5) + frames + _rest(11.3, 5)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 1


def test_back_to_back_attempts_split_on_long_gap() -> None:
    frames = _burst(0.0, 8) + _rest(0.8, 8) + _burst(1.6, 8)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 2


def test_nearby_bursts_within_merge_gap_join() -> None:
    frames = _burst(0.0, 8) + _rest(0.8, 2) + _burst(1.0, 8)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 1
    assert ranges[0].start_seconds == pytest.approx(0.0)
    assert ranges[0].end_seconds == pytest.approx(1.7)


def test_truncated_first_and_last_attempts() -> None:
    # Sequence starts mid-swing: no leading rest required.
    leading = _burst(0.0, 8)
    ranges = find_candidates(leading, _cfg())
    assert len(ranges) == 1
    assert ranges[0].start_seconds == pytest.approx(0.0)
    # Sequence ends mid-swing: finish() flushes the open segment.
    trailing = _rest(0.0, 5) + _burst(0.5, 8)
    detector = CandidateDetector(_cfg())
    detector.feed_all(trailing)
    flushed = detector.finish()
    assert len(flushed) == 1
    assert flushed[0].start_seconds == pytest.approx(0.5)
    assert flushed[0].end_seconds == pytest.approx(1.2)
    # Second finish without new feeds is empty.
    assert detector.finish() == ()


# --- Duration bounds -------------------------------------------------------


def test_minimum_duration_rejects_blips() -> None:
    frames = _rest(0.0, 5) + _burst(0.5, 3) + _rest(0.8, 5)
    assert find_candidates(frames, _cfg()) == ()


def test_abbreviated_motion_allowed() -> None:
    # 8 frames span 0.7s >= min 0.6: a short but complete swing passes.
    frames = _rest(0.0, 5) + _burst(0.5, 8) + _rest(1.3, 5)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 1
    assert ranges[0].duration_seconds == pytest.approx(0.7)


def test_maximum_duration_splits_long_activity() -> None:
    config = _cfg(max_duration_seconds=1.0)
    frames = [_mk(i * STEP, motion=SWING, overhead=1.0) for i in range(31)]
    ranges = find_candidates(frames, config)
    assert len(ranges) >= 2
    for item in ranges:
        assert item.duration_seconds <= 1.0 + 1e-9
        assert item.duration_seconds >= config.min_duration_seconds - 1e-9
    assert ranges[0].start_seconds == pytest.approx(0.0)
    assert ranges[-1].end_seconds == pytest.approx(3.0)
    starts = [item.start_seconds for item in ranges]
    assert starts == sorted(starts)


# --- Toss / unrelated / missing --------------------------------------------


def test_aborted_toss_low_motion_rejected() -> None:
    # Overhead toss lift without swing-strength motion never opens.
    frames = _rest(0.0, 5)
    frames += [
        _mk(0.5 + i * STEP, motion=0.25, overhead=1.0) for i in range(6)
    ]
    frames += _rest(1.1, 5)
    assert find_candidates(frames, _cfg()) == ()


def test_isolated_overhead_blip_below_min_rejected() -> None:
    frames = _rest(0.0, 5) + _burst(0.5, 3) + _rest(0.8, 5)
    assert find_candidates(frames, _cfg()) == ()


def test_unrelated_motion_without_overhead_rejected() -> None:
    frames = _rest(0.0, 5) + _burst(0.5, 11, overhead=0.0) + _rest(1.6, 5)
    assert find_candidates(frames, _cfg()) == ()
    # The gate is the cause: disabling it keeps the same motion.
    kept = find_candidates(frames, _cfg(require_overhead=False))
    assert len(kept) == 1


def test_all_missing_overhead_kept_where_evidence_permits() -> None:
    frames = _rest(0.0, 5) + _burst(0.5, 8, overhead=None) + _rest(1.3, 5)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 1


def test_missing_samples_do_not_fabricate() -> None:
    frames = [
        _mk(0.0, person=False),
        _mk(0.1, motion=None, overhead=None),
        _mk(0.2, person=False),
    ]
    assert find_candidates(frames, _cfg()) == ()


def test_single_missing_frame_inside_serve_still_merged() -> None:
    frames = _rest(0.0, 5) + _burst(0.5, 5)
    frames += [_mk(1.0, person=False)]
    frames += _burst(1.1, 5) + _rest(1.6, 5)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 1
    assert ranges[0].start_seconds == pytest.approx(0.5)
    assert ranges[0].end_seconds == pytest.approx(1.5)


def test_long_missing_gap_splits_or_drops() -> None:
    first = _burst(0.5, 8)
    gap = [_mk(1.3 + i * STEP, person=False) for i in range(12)]
    second = _burst(2.5, 8)
    frames = _rest(0.0, 5) + first + gap + second + _rest(3.3, 5)
    ranges = find_candidates(frames, _cfg())
    assert len(ranges) == 2


def test_visibility_floor_blocks_low_coverage_motion() -> None:
    frames = [
        _mk(0.1 * i, motion=SWING, overhead=1.0, visible=0.05)
        for i in range(11)
    ]
    assert find_candidates(frames, _cfg()) == ()


# --- State machine semantics ------------------------------------------------


def test_reset_discards_buffered_state() -> None:
    detector = CandidateDetector(_cfg())
    detector.feed_all(_burst(0.0, 8))
    detector.reset()
    detector.feed_all(_rest(5.0, 10))
    assert detector.finish() == ()


def test_feed_finish_incremental_matches_pure_function() -> None:
    frames = _rest(0.0, 5) + _burst(0.5, 8) + _rest(1.3, 20) + _burst(3.3, 8)
    detector = CandidateDetector(_cfg())
    for frame in frames:
        detector.feed(frame)
    assert detector.finish() == find_candidates(frames, _cfg())


def test_determinism() -> None:
    frames = _rest(0.0, 5) + _burst(0.5, 8) + _rest(1.3, 6) + _burst(1.9, 9)
    first = find_candidates(frames, _cfg())
    second = find_candidates(frames, _cfg())
    assert first == second
    assert [item.to_dict() for item in first] == [
        item.to_dict() for item in second
    ]


# --- Codecs and validation --------------------------------------------------


def test_candidate_json_round_trip() -> None:
    item = CandidateRange(start_seconds=1.25, end_seconds=2.5)
    assert CandidateRange.from_dict(item.to_dict()) == item
    assert CandidateRange.from_json(item.to_json()) == item
    with pytest.raises(RangeError):
        CandidateRange(start_seconds=1.0, end_seconds=1.0)
    with pytest.raises(RangeError):
        CandidateRange(start_seconds=2.0, end_seconds=1.0)
    with pytest.raises(RangeError):
        CandidateRange(start_seconds=-0.1, end_seconds=0.5)
    with pytest.raises(RangeError):
        CandidateRange.from_dict(
            {"start_seconds": 0.0, "end_seconds": 0.5, "schema_version": 999}
        )
    with pytest.raises(RangeError):
        CandidateRange.from_dict(
            {"start_seconds": 0.0, "end_seconds": 0.5, "schema_version": 1,
             "extra": 0}
        )
    with pytest.raises(RangeError):
        CandidateRange.from_dict({"start_seconds": 0.0})


def test_config_json_round_trip() -> None:
    config = _cfg()
    assert RangeConfig.from_dict(config.to_dict()) == config
    assert RangeConfig.from_json(config.to_json()) == config


def test_config_validation() -> None:
    with pytest.raises(RangeError):
        _cfg(active_threshold=0.0)
    with pytest.raises(RangeError):
        _cfg(active_threshold=1.5)
    with pytest.raises(RangeError):
        _cfg(inactive_threshold=-0.1)
    with pytest.raises(RangeError):
        _cfg(active_threshold=0.3, inactive_threshold=0.5)
    with pytest.raises(RangeError):
        _cfg(min_duration_seconds=0.0)
    with pytest.raises(RangeError):
        _cfg(min_duration_seconds=2.0, max_duration_seconds=1.0)
    with pytest.raises(RangeError):
        _cfg(merge_gap_seconds=-0.1)
    with pytest.raises(RangeError):
        _cfg(min_visible_fraction=1.5)
    with pytest.raises(RangeError):
        _cfg(require_overhead="yes")  # type: ignore[arg-type]
    with pytest.raises(RangeError):
        RangeConfig(schema_version=999)
    with pytest.raises(RangeError):
        RangeConfig.from_dict({**_cfg().to_dict(), "extra": 1})
    with pytest.raises(RangeError):
        RangeConfig.from_dict({"active_threshold": 0.4})
    with pytest.raises(RangeError):
        RangeConfig.from_dict("nope")  # type: ignore[arg-type]
