from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve_review.scene import Point2D, Racket2D
from serve_review.tracking import (
    RacketVisionCacheCorruptError,
    RacketVisionCacheError,
    RacketVisionCacheIdentity,
    RacketVisionCacheStaleError,
    RacketVisionFrameObservation,
    fingerprint_frame_times,
    load_racketvision_cache,
    write_racketvision_cache,
)
from serve_review.tracking import cache as cache_module

TIMES = (1.0, 1.25)


def _identity(**changes: object) -> RacketVisionCacheIdentity:
    values = {
        "source_fingerprint": "sha256:source",
        "attempt_start_seconds": 0.75,
        "attempt_end_seconds": 1.5,
        "timeline_fingerprint": fingerprint_frame_times(TIMES),
        "tracker_fingerprint": "sha256:tracker",
    }
    values.update(changes)
    return RacketVisionCacheIdentity(**values)


def _observations() -> tuple[RacketVisionFrameObservation, ...]:
    return (
        RacketVisionFrameObservation(
            time_seconds=1.0,
            ball_2d=Point2D(0.25, 0.5, 0.8),
            racket_2d=Racket2D(
                bbox=(0.1, 0.2, 0.4, 0.7),
                bbox_confidence=0.9,
                keypoints={
                    "Top": Point2D(0.2, 0.3, 0.75),
                    "Handle": None,
                },
            ),
        ),
        RacketVisionFrameObservation(
            time_seconds=1.25,
            ball_2d=None,
            racket_2d=None,
        ),
    )


def test_complete_cache_round_trip_preserves_raw_missingness(tmp_path: Path) -> None:
    path = tmp_path / "racketvision-track-v1.jsonl"
    identity = _identity()

    write_racketvision_cache(path, identity, _observations(), expected_times=TIMES)
    snapshot = load_racketvision_cache(path, identity, expected_times=TIMES)

    assert snapshot.complete is True
    assert snapshot.identity == identity
    assert snapshot.observations == _observations()
    assert snapshot.observations[0].racket_2d is not None
    assert snapshot.observations[0].racket_2d.keypoints["Handle"] is None
    assert snapshot.observations[1].ball_2d is None
    assert snapshot.observations[1].racket_2d is None


def test_serialization_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    identity = _identity()

    write_racketvision_cache(first, identity, _observations(), expected_times=TIMES)
    write_racketvision_cache(second, identity, _observations(), expected_times=TIMES)

    assert first.read_bytes() == second.read_bytes()


def test_load_rejects_stale_identity(tmp_path: Path) -> None:
    path = tmp_path / "track.jsonl"
    write_racketvision_cache(path, _identity(), _observations(), expected_times=TIMES)

    with pytest.raises(RacketVisionCacheStaleError):
        load_racketvision_cache(
            path,
            _identity(tracker_fingerprint="sha256:different"),
            expected_times=TIMES,
        )


def test_write_and_load_require_exact_canonical_pts(tmp_path: Path) -> None:
    path = tmp_path / "track.jsonl"
    identity = _identity()

    with pytest.raises(RacketVisionCacheError, match="exactly match"):
        write_racketvision_cache(
            path, identity, _observations(), expected_times=(1.0, 1.3)
        )

    write_racketvision_cache(path, identity, _observations(), expected_times=TIMES)
    with pytest.raises(RacketVisionCacheCorruptError, match="exactly match"):
        load_racketvision_cache(path, identity, expected_times=(1.0, 1.2))


def test_load_rejects_unknown_fields_and_incomplete_cache(tmp_path: Path) -> None:
    path = tmp_path / "track.jsonl"
    identity = _identity()
    write_racketvision_cache(path, identity, _observations(), expected_times=TIMES)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["unexpected"] = True
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(RacketVisionCacheCorruptError, match="unknown"):
        load_racketvision_cache(path, identity, expected_times=TIMES)

    path.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(RacketVisionCacheCorruptError, match="incomplete"):
        load_racketvision_cache(path, identity, expected_times=TIMES)


def test_failed_atomic_replace_preserves_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "track.jsonl"
    identity = _identity()
    write_racketvision_cache(path, identity, _observations(), expected_times=TIMES)
    original = path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(cache_module.os, "replace", fail_replace)
    with pytest.raises(RacketVisionCacheError, match="simulated replace failure"):
        write_racketvision_cache(path, identity, _observations(), expected_times=TIMES)

    assert path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_identity_and_timeline_validation_are_strict() -> None:
    with pytest.raises(RacketVisionCacheError, match="start < end"):
        _identity(attempt_end_seconds=0.75)
    with pytest.raises(RacketVisionCacheError, match="strictly increasing"):
        fingerprint_frame_times((1.0, 1.0))
