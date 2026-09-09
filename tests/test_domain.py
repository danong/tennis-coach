"""Focused M1.1 tests for versioned media domain schemas.

Deterministic, offline, and independent of private footage.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from serve_review.domain import (
    DomainError,
    ExportPlan,
    ExportPlanError,
    MediaRange,
    RangeError,
    SchemaVersionError,
    SourceMetadata,
    SourceMetadataError,
)


def make_source(**overrides) -> SourceMetadata:
    fields = {
        "fingerprint": "sha256:abc123",
        "duration_seconds": 120.0,
        "width": 1080,
        "height": 1920,
        "frame_rate_num": 120,
        "frame_rate_den": 1,
        "video_codec": "hevc",
        "rotation_degrees": 0,
    }
    fields.update(overrides)
    return SourceMetadata(**fields)


# --- SourceMetadata -------------------------------------------------------


def test_source_metadata_is_immutable() -> None:
    source = make_source()
    with pytest.raises(dataclasses.FrozenInstanceError):
        source.width = 720  # type: ignore[misc]


def test_source_metadata_reports_nominal_fps() -> None:
    assert make_source(frame_rate_num=24000, frame_rate_den=1001).frames_per_second == (
        pytest.approx(24000 / 1001)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_seconds": 0.0},
        {"duration_seconds": -1.0},
        {"duration_seconds": float("nan")},
        {"duration_seconds": float("inf")},
        {"duration_seconds": "120"},
        {"duration_seconds": True},
        {"width": 0},
        {"width": -1080},
        {"width": 1080.0},
        {"width": True},
        {"height": 0},
        {"frame_rate_num": 0},
        {"frame_rate_den": 0},
        {"frame_rate_den": -1},
        {"fingerprint": ""},
        {"fingerprint": "   "},
        {"fingerprint": None},
        {"video_codec": ""},
        {"rotation_degrees": 45},
        {"rotation_degrees": -90},
        {"rotation_degrees": False},
        {"time_base_num": 1},  # den left null
        {"time_base_den": 90000},  # num left null
        {"time_base_num": 0, "time_base_den": 90000},
    ],
)
def test_source_metadata_rejects_invalid_values(kwargs) -> None:
    with pytest.raises((SourceMetadataError, SchemaVersionError)):
        make_source(**kwargs)


def test_source_metadata_accepts_optional_time_base() -> None:
    source = make_source(time_base_num=1, time_base_den=90000)
    assert source.time_base_num == 1
    assert source.time_base_den == 90000


def test_source_metadata_rejects_newer_schema_version() -> None:
    with pytest.raises(SchemaVersionError):
        make_source()
        SourceMetadata.from_dict({**make_source().to_dict(), "schema_version": 2})
    with pytest.raises(SchemaVersionError):
        SourceMetadata.from_dict({**make_source().to_dict(), "schema_version": 999})


def test_source_metadata_rejects_missing_and_unknown_keys() -> None:
    payload = make_source().to_dict()
    del payload["width"]
    with pytest.raises(SourceMetadataError):
        SourceMetadata.from_dict(payload)
    with pytest.raises(SourceMetadataError):
        SourceMetadata.from_dict({**make_source().to_dict(), "extra": 1})
    with pytest.raises(DomainError):
        SourceMetadata.from_dict({"schema_version": 1})


def test_source_metadata_json_round_trip_is_deterministic() -> None:
    source = make_source(time_base_num=1, time_base_den=90000)
    first = source.to_json()
    assert source.to_json() == first
    assert list(json.loads(first)) == sorted(json.loads(first))
    assert SourceMetadata.from_json(first) == source
    assert SourceMetadata.from_json(first.encode("utf-8")) == source
    with pytest.raises(DomainError):
        SourceMetadata.from_json("not json{")


# --- MediaRange validation -------------------------------------------------


def test_media_range_accepts_int_bounds_as_floats() -> None:
    assert MediaRange(start_seconds=1, end_seconds=3) == MediaRange(
        start_seconds=1.0, end_seconds=3.0
    )


@pytest.mark.parametrize(
    "start,end",
    [
        (1.0, 1.0),  # zero length
        (2.0, 1.0),  # inverted
        (-0.1, 1.0),  # negative start
        (0.0, -1.0),  # negative end
        (float("nan"), 1.0),
        (0.0, float("nan")),
        (float("inf"), 2.0),
        (0.0, float("inf")),
        (True, 1.0),
        ("0", 1.0),
        (0.0, None),
    ],
)
def test_media_range_rejects_invalid_bounds(start, end) -> None:
    with pytest.raises((RangeError, SchemaVersionError)):
        MediaRange(start_seconds=start, end_seconds=end)


def test_media_range_duration_and_half_open_contains() -> None:
    span = MediaRange(start_seconds=2.0, end_seconds=5.0)
    assert span.duration_seconds == pytest.approx(3.0)
    assert span.contains(2.0)  # closed at start
    assert span.contains(4.999999)
    assert not span.contains(5.0)  # open at end
    assert not span.contains(1.999999)
    assert not span.contains(float("nan"))
    assert not span.contains("2.0")


def test_media_range_overlaps_half_open_boundaries() -> None:
    base = MediaRange(start_seconds=10.0, end_seconds=20.0)
    assert base.overlaps(MediaRange(start_seconds=15.0, end_seconds=25.0))
    assert base.overlaps(base)
    assert base.overlaps(MediaRange(start_seconds=12.0, end_seconds=18.0))
    # Adjacent at exactly 20.0 shares no instant: no overlap.
    assert not base.overlaps(MediaRange(start_seconds=20.0, end_seconds=30.0))
    assert not base.overlaps(MediaRange(start_seconds=0.0, end_seconds=10.0))
    assert not base.overlaps(MediaRange(start_seconds=30.0, end_seconds=40.0))


def test_media_range_adjacency_is_directional_and_excludes_overlap() -> None:
    base = MediaRange(start_seconds=10.0, end_seconds=20.0)
    assert base.is_adjacent(MediaRange(start_seconds=20.0, end_seconds=30.0))
    assert MediaRange(start_seconds=0.0, end_seconds=10.0).is_adjacent(base)
    assert not base.is_adjacent(MediaRange(start_seconds=15.0, end_seconds=25.0))
    assert not base.is_adjacent(MediaRange(start_seconds=30.0, end_seconds=40.0))


def test_media_range_union_merges_overlap_and_adjacency() -> None:
    base = MediaRange(start_seconds=10.0, end_seconds=20.0)
    assert base.union(MediaRange(start_seconds=20.0, end_seconds=30.0)) == MediaRange(
        start_seconds=10.0, end_seconds=30.0
    )
    assert base.union(MediaRange(start_seconds=15.0, end_seconds=25.0)) == MediaRange(
        start_seconds=10.0, end_seconds=25.0
    )
    with pytest.raises(RangeError):
        base.union(MediaRange(start_seconds=30.0, end_seconds=40.0))


def test_media_range_ordering_sorts_by_start_then_end() -> None:
    ranges = [
        MediaRange(start_seconds=5.0, end_seconds=9.0),
        MediaRange(start_seconds=1.0, end_seconds=9.0),
        MediaRange(start_seconds=1.0, end_seconds=2.0),
        MediaRange(start_seconds=5.0, end_seconds=6.0),
    ]
    assert sorted(ranges) == [
        MediaRange(start_seconds=1.0, end_seconds=2.0),
        MediaRange(start_seconds=1.0, end_seconds=9.0),
        MediaRange(start_seconds=5.0, end_seconds=6.0),
        MediaRange(start_seconds=5.0, end_seconds=9.0),
    ]
    assert MediaRange(1.0, 2.0) < MediaRange(1.0, 3.0) <= MediaRange(1.0, 3.0)
    assert MediaRange(2.0, 3.0) > MediaRange(1.0, 9.0) >= MediaRange(1.0, 9.0)


def test_media_range_json_round_trip_and_version_rejection() -> None:
    span = MediaRange(start_seconds=0.5, end_seconds=2.5)
    encoded = span.to_json()
    assert span.to_json() == encoded
    assert MediaRange.from_json(encoded) == span
    assert MediaRange.from_dict(span.to_dict()) == span
    with pytest.raises(SchemaVersionError):
        MediaRange.from_dict({**span.to_dict(), "schema_version": 2})
    with pytest.raises(RangeError):
        MediaRange.from_dict({"schema_version": 1, "start_seconds": 0.0})
    with pytest.raises(DomainError):
        MediaRange.from_dict({**span.to_dict(), "bogus": True})


# --- Clamping and padding ---------------------------------------------------


def test_media_range_clamp_trims_end_to_duration() -> None:
    assert MediaRange(start_seconds=118.0, end_seconds=130.0).clamp(120.0) == MediaRange(
        start_seconds=118.0, end_seconds=120.0
    )


def test_media_range_clamp_leaves_inside_range_unchanged() -> None:
    span = MediaRange(start_seconds=10.0, end_seconds=20.0)
    assert span.clamp(120.0) == span


@pytest.mark.parametrize("duration", [0.0, -5.0, float("nan"), float("inf"), "120"])
def test_media_range_clamp_rejects_bad_duration(duration) -> None:
    with pytest.raises(RangeError):
        MediaRange(start_seconds=1.0, end_seconds=2.0).clamp(duration)


def test_media_range_clamp_rejects_range_beyond_duration() -> None:
    with pytest.raises(RangeError):
        MediaRange(start_seconds=120.0, end_seconds=130.0).clamp(120.0)


def test_media_range_with_padding_clamps_to_source_bounds() -> None:
    span = MediaRange(start_seconds=1.0, end_seconds=119.0)
    assert span.with_padding(1.0, 120.0) == MediaRange(
        start_seconds=0.0, end_seconds=120.0
    )
    assert span.with_padding(0.0, 120.0) == span
    with pytest.raises(RangeError):
        span.with_padding(-1.0, 120.0)
    with pytest.raises(RangeError):
        span.with_padding(float("nan"), 120.0)


# --- ExportPlan ------------------------------------------------------------


def test_export_plan_validates_order_overlap_and_bounds() -> None:
    source = make_source()
    plan = ExportPlan.for_source(
        source,
        [
            MediaRange(start_seconds=1.0, end_seconds=2.0),
            MediaRange(start_seconds=2.0, end_seconds=3.0),  # adjacent: allowed
            MediaRange(start_seconds=10.0, end_seconds=12.0),
        ],
    )
    assert plan.total_duration_seconds == pytest.approx(4.0)
    assert len(plan) == 3
    assert list(plan) == list(plan.ranges)
    assert plan[0] == MediaRange(start_seconds=1.0, end_seconds=2.0)

    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(
            source,
            [
                MediaRange(start_seconds=10.0, end_seconds=12.0),
                MediaRange(start_seconds=1.0, end_seconds=2.0),
            ],
        )
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(
            source,
            [
                MediaRange(start_seconds=1.0, end_seconds=3.0),
                MediaRange(start_seconds=2.0, end_seconds=4.0),
            ],
        )
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(source, [])
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(
            source, [MediaRange(start_seconds=119.0, end_seconds=121.0)]
        )


def test_export_plan_clamped_constructor_trims_to_source() -> None:
    source = make_source()
    plan = ExportPlan.clamped(
        source, [MediaRange(start_seconds=118.0, end_seconds=130.0)]
    )
    assert plan.ranges == (MediaRange(start_seconds=118.0, end_seconds=120.0),)
    with pytest.raises(RangeError):
        ExportPlan.clamped(
            source, [MediaRange(start_seconds=121.0, end_seconds=130.0)]
        )


def test_export_plan_json_round_trip_is_deterministic() -> None:
    source = make_source()
    plan = ExportPlan.for_source(
        source,
        [
            MediaRange(start_seconds=1.0, end_seconds=2.0),
            MediaRange(start_seconds=10.0, end_seconds=12.5),
        ],
    )
    encoded = plan.to_json()
    assert plan.to_json() == encoded
    assert list(json.loads(encoded)) == sorted(json.loads(encoded))
    decoded = ExportPlan.from_json(encoded)
    assert decoded == plan
    assert decoded.source_fingerprint == "sha256:abc123"
    assert decoded.total_duration_seconds == pytest.approx(3.5)


def test_export_plan_rejects_newer_versions_and_bad_payloads() -> None:
    plan = ExportPlan.for_source(
        make_source(), [MediaRange(start_seconds=1.0, end_seconds=2.0)]
    )
    with pytest.raises(SchemaVersionError):
        ExportPlan.from_dict({**plan.to_dict(), "schema_version": 2})
    nested = plan.to_dict()
    nested["ranges"] = [
        {**nested["ranges"][0], "schema_version": 99},
    ]
    with pytest.raises(ExportPlanError):
        ExportPlan.from_dict(nested)
    with pytest.raises(ExportPlanError):
        ExportPlan.from_dict({**plan.to_dict(), "ranges": []})
    with pytest.raises(ExportPlanError):
        ExportPlan.from_dict({**plan.to_dict(), "ranges": "nope"})
    with pytest.raises(DomainError):
        ExportPlan.from_dict({**plan.to_dict(), "unexpected": True})
    with pytest.raises(DomainError):
        ExportPlan.from_json("{oops")
