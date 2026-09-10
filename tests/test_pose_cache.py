"""Focused M2.2 tests for the fingerprinted atomic pose JSONL cache.

Deterministic, offline, and independent of private footage and model
execution. All caches live in temporary directories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve_review.pose.cache import (
    CACHE_SCHEMA_VERSION,
    CacheCorruptError,
    CacheError,
    CacheSnapshot,
    CacheStaleError,
    PoseCacheWriter,
    append_frames,
    finalize_cache,
    header_from_dict,
    header_to_dict,
    load_cache,
    open_cache_writer,
    quarantine_corrupt,
    require_matching_identity,
    write_complete_cache,
    write_partial_cache,
)
from serve_review.pose.schema import (
    POSE_SCHEMA_VERSION,
    BodyKeypoint,
    CacheIdentity,
    FrameObservation,
    PersonBox,
    PersonObservation,
)


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


def make_person(*, seed: int = 0) -> PersonObservation:
    joints = tuple(
        None
        if index % 7 == 0 and index != 1
        else BodyKeypoint(
            x=0.1 + 0.01 * ((index + seed) % 10),
            y=0.2 + 0.01 * ((index + seed) % 5),
            visibility=0.9,
        )
        for index in range(33)
    )
    # Guarantee at least one present joint.
    assert any(joint is not None for joint in joints)
    return PersonObservation(
        box=PersonBox(x_min=0.1, y_min=0.1, x_max=0.7, y_max=0.9),
        keypoints=joints,
        score=0.6,
    )


def make_frame(time_seconds: float, *, persons: bool = True) -> FrameObservation:
    return FrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(make_person(seed=int(round(time_seconds * 1000))),) if persons else (),
    )


def make_frames(count: int, *, start: float = 0.0, step: float = 1 / 30.0) -> list:
    return [make_frame(start + index * step) for index in range(count)]


# --- Round trips ----------------------------------------------------------


def test_cache_versions_are_pinned() -> None:
    assert CACHE_SCHEMA_VERSION == 1
    assert POSE_SCHEMA_VERSION == 1


def test_complete_round_trip_preserves_identity_order_and_times(tmp_path: Path) -> None:
    identity = make_identity()
    frames = make_frames(5)
    path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(path, identity, frames)
    snapshot = load_cache(path)
    assert isinstance(snapshot, CacheSnapshot)
    assert snapshot.complete is True
    assert snapshot.identity == identity
    assert list(snapshot.frames) == frames
    times = [frame.time_seconds for frame in snapshot.frames]
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    for stored, expected in zip(snapshot.frames, frames):
        assert stored.timestamp_ms == int(round(expected.time_seconds * 1000))


def test_complete_cache_file_is_deterministic_jsonl(tmp_path: Path) -> None:
    identity = make_identity()
    frames = make_frames(2)
    path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(path, identity, frames)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # header + 2 frames + footer
    for line in lines:
        assert json.loads(line)  # each line is standalone JSON
    assert json.loads(lines[0])["type"] == "header"
    assert json.loads(lines[-1]) == {
        "complete": True,
        "frame_count": 2,
        "schema_version": 1,
        "type": "footer",
    }
    assert write_complete_cache(tmp_path / "again.jsonl", identity, frames)
    assert (tmp_path / "again.jsonl").read_text() == path.read_text()


def test_empty_frames_round_trip_is_complete(tmp_path: Path) -> None:
    path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(path, make_identity(), [])
    snapshot = load_cache(path)
    assert snapshot.complete is True
    assert snapshot.frames == ()


def test_header_codec_round_trips_and_validates_type(tmp_path: Path) -> None:
    identity = make_identity()
    header = header_to_dict(identity)
    assert header["type"] == "header"
    assert header["pose_schema_version"] == POSE_SCHEMA_VERSION
    assert header_from_dict(header) == identity
    with pytest.raises(CacheCorruptError):
        header_from_dict({**header, "type": "frame"})
    with pytest.raises(CacheCorruptError):
        header_from_dict({**header, "schema_version": 999})
    with pytest.raises(CacheCorruptError):
        header_from_dict({**header, "pose_schema_version": 999})
    with pytest.raises(CacheCorruptError):
        header_from_dict({**header, "extra": 1})


# --- Stale identity -------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_fingerprint", "sha256:different"),
        ("model_name", "other-model"),
        ("model_version", "9.9.9"),
        ("sampling_rate_hz", 60.0),
        ("sampling_start_seconds", 1.0),
    ],
)
def test_stale_identity_fields_are_rejected(tmp_path: Path, field: str, value) -> None:
    stored = make_identity()
    requested = make_identity(**{field: value})
    with pytest.raises(CacheStaleError, match=field):
        require_matching_identity(stored, requested)
    # End to end: rows written under one identity are never valid for another.
    path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(path, stored, make_frames(2))
    snapshot = load_cache(path)
    with pytest.raises(CacheStaleError):
        require_matching_identity(snapshot.identity, requested)
    with pytest.raises(CacheStaleError):
        append_frames(path, requested, make_frames(1, start=5.0))
    with pytest.raises(CacheStaleError):
        finalize_cache(path, requested)
    # The stale attempt leaves the stored file untouched and complete.
    assert load_cache(path).complete is True


def test_matching_identity_passes_silently() -> None:
    require_matching_identity(make_identity(), make_identity())


# --- Partial resume -------------------------------------------------------


def test_interrupted_partial_cache_is_incomplete_but_resumable(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    first = make_frames(3)
    write_partial_cache(path, identity, first)
    snapshot = load_cache(path)
    assert snapshot.complete is False
    assert list(snapshot.frames) == first
    # Resume by appending, then finalize to complete.
    rest = make_frames(2, start=first[-1].time_seconds + 1 / 30.0)
    append_frames(path, identity, rest)
    resumed = load_cache(path)
    assert resumed.complete is False
    assert list(resumed.frames) == first + rest
    finalize_cache(path, identity)
    finished = load_cache(path)
    assert finished.complete is True
    assert list(finished.frames) == first + rest


def test_finalize_is_idempotent_on_complete_cache(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(path, identity, make_frames(2))
    before = path.read_text(encoding="utf-8")
    assert finalize_cache(path, identity) == path
    assert path.read_text(encoding="utf-8") == before
    assert load_cache(path).complete is True


def test_append_to_complete_cache_is_refused(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(path, identity, make_frames(2))
    with pytest.raises(CacheError, match="already complete"):
        append_frames(path, identity, make_frames(1, start=5.0))
    assert load_cache(path).complete is True


def test_append_rejects_out_of_order_times(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    write_partial_cache(path, identity, make_frames(3))
    with pytest.raises(CacheCorruptError):
        append_frames(path, identity, [make_frame(0.01)])
    # Untouched and still resumable.
    snapshot = load_cache(path)
    assert snapshot.complete is False
    assert len(snapshot.frames) == 3


# --- Corrupt quarantine ---------------------------------------------------


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _header_line(identity: CacheIdentity) -> str:
    return json.dumps(header_to_dict(identity), sort_keys=True) + "\n"


def _frame_line(frame: FrameObservation) -> str:
    return (
        json.dumps(
            {"observation": frame.to_dict(), "type": "frame"}, sort_keys=True
        )
        + "\n"
    )


def test_corrupt_caches_raise_and_never_present_as_complete(tmp_path: Path) -> None:
    identity = make_identity()
    good_frames = make_frames(2)
    header = _header_line(identity)
    frame_lines = [_frame_line(frame) for frame in good_frames]
    footer = (
        json.dumps(
            {"complete": True, "frame_count": 2, "schema_version": 1, "type": "footer"},
            sort_keys=True,
        )
        + "\n"
    )
    cases = {
        "empty file": [],
        "blank file": ["\n"],
        "missing header": frame_lines + [footer],
        "truncated json": [header, frame_lines[0][:20]],
        "blank middle line": [header, "\n", frame_lines[0]],
        "unknown record type": [header, json.dumps({"type": "pose"}) + "\n"],
        "frame after footer": [header, *frame_lines, footer, frame_lines[0]],
        "double footer": [header, *frame_lines, footer, footer],
        "count mismatch": [
            header,
            *frame_lines,
            json.dumps(
                {
                    "complete": True,
                    "frame_count": 99,
                    "schema_version": 1,
                    "type": "footer",
                },
                sort_keys=True,
            )
            + "\n",
        ],
        "non-boolean complete": [
            header,
            *frame_lines,
            json.dumps(
                {
                    "complete": "yes",
                    "frame_count": 2,
                    "schema_version": 1,
                    "type": "footer",
                },
                sort_keys=True,
            )
            + "\n",
        ],
        "out-of-order times": [header, frame_lines[1], frame_lines[0], footer],
        "duplicate times": [header, frame_lines[0], frame_lines[0]],
        "bad observation": [
            header,
            json.dumps({"observation": {"nope": 1}, "type": "frame"}) + "\n",
        ],
        "coordinate out of range": [
            header,
            json.dumps(
                {
                    "observation": {
                        **good_frames[0].to_dict(),
                        "persons": [
                            {
                                **good_frames[0].persons[0].to_dict(),
                                "box": {
                                    "schema_version": 1,
                                    "x_max": 5.0,
                                    "x_min": 0.1,
                                    "y_max": 0.9,
                                    "y_min": 0.1,
                                },
                            }
                        ],
                    },
                    "type": "frame",
                }
            )
            + "\n",
        ],
        "stale cache schema": [
            json.dumps({**json.loads(header), "schema_version": 999}) + "\n",
            *frame_lines,
        ],
        "stale pose schema": [
            json.dumps({**json.loads(header), "pose_schema_version": 999}) + "\n",
            *frame_lines,
        ],
    }
    for label, lines in cases.items():
        path = tmp_path / "pose-v1.jsonl"
        _write_lines(path, lines)
        try:
            load_cache(path)
        except (CacheCorruptError, CacheError):
            continue
        raise AssertionError(f"corrupt case did not raise: {label}")


def test_quarantine_moves_corrupt_file_aside(tmp_path: Path) -> None:
    path = tmp_path / "pose-v1.jsonl"
    path.write_text("this is { not json\n", encoding="utf-8")
    with pytest.raises(CacheCorruptError):
        load_cache(path)
    quarantined = quarantine_corrupt(path)
    assert quarantined == tmp_path / "pose-v1.jsonl.corrupt"
    assert quarantined.is_file()
    assert not path.exists()
    # A second quarantine takes a numeric suffix deterministically.
    path.write_text("still corrupt\n", encoding="utf-8")
    second = quarantine_corrupt(path)
    assert second == tmp_path / "pose-v1.jsonl.corrupt.1"
    assert not path.exists()
    # Fresh extraction can start at the original location.
    write_complete_cache(path, make_identity(), make_frames(1))
    assert load_cache(path).complete is True


def test_quarantine_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match="does not exist"):
        quarantine_corrupt(tmp_path / "pose-v1.jsonl")


def test_load_missing_file_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match="does not exist"):
        load_cache(tmp_path / "pose-v1.jsonl")


# --- Atomic writer behavior -------------------------------------------------


def test_writer_publishes_only_on_commit(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    frames = make_frames(3)
    with open_cache_writer(path, identity) as writer:
        assert writer.staged_count == 0
        # The final file must not exist before commit (atomic publish).
        assert not path.exists()
        for frame in frames:
            writer.append(frame)
        assert writer.staged_count == 3
        assert not path.exists()
        writer.commit()
    assert path.is_file()
    snapshot = load_cache(path)
    assert snapshot.complete is True
    assert list(snapshot.frames) == frames
    # No staging files leak beside the final cache.
    leftovers = [entry for entry in tmp_path.iterdir() if ".tmp-" in entry.name]
    assert leftovers == []


def test_interrupted_writer_never_presents_as_complete(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    with open_cache_writer(path, identity) as writer:
        writer.append(make_frame(0.0))
        writer.append(make_frame(1 / 30.0))
        # No commit: exiting the block must not publish a complete cache.
    assert not path.exists()
    leftovers = [entry for entry in tmp_path.iterdir() if ".tmp-" in entry.name]
    assert leftovers == []
    # Aborted staging can be restarted from scratch.
    write_complete_cache(path, identity, make_frames(1))
    assert load_cache(path).complete is True


def test_writer_resume_carries_forward_partial_rows(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    first = make_frames(2)
    write_partial_cache(path, identity, first)
    before = path.read_text(encoding="utf-8")
    with open_cache_writer(path, identity, resume=True) as writer:
        assert writer.staged_count == 2
        # The previous resumable file stays in place until commit.
        assert path.read_text(encoding="utf-8") == before
        extra = make_frames(2, start=first[-1].time_seconds + 1 / 30.0)
        for frame in extra:
            writer.append(frame)
        writer.commit()
    snapshot = load_cache(path)
    assert snapshot.complete is True
    assert list(snapshot.frames) == first + extra


def test_writer_abort_leaves_previous_cache_untouched(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    first = make_frames(2)
    write_partial_cache(path, identity, first)
    before = path.read_text(encoding="utf-8")
    writer = open_cache_writer(path, identity, resume=True)
    writer.append(make_frame(first[-1].time_seconds + 1 / 30.0))
    writer.abort()
    assert path.read_text(encoding="utf-8") == before
    assert load_cache(path).complete is False


def test_writer_rejects_existing_without_resume_complete_and_stale(
    tmp_path: Path,
) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    write_partial_cache(path, identity, make_frames(1))
    with pytest.raises(CacheError, match="resume=True"):
        PoseCacheWriter(path, identity)
    write_complete_cache(path, identity, make_frames(1))
    with pytest.raises(CacheError, match="already complete"):
        PoseCacheWriter(path, identity, resume=True)
    with pytest.raises(CacheStaleError):
        PoseCacheWriter(path, make_identity(source_fingerprint="sha256:other"), resume=True)
    # Untouched and still complete.
    assert load_cache(path).complete is True


def test_writer_rejects_out_of_order_and_bad_rows(tmp_path: Path) -> None:
    identity = make_identity()
    path = tmp_path / "pose-v1.jsonl"
    with open_cache_writer(path, identity) as writer:
        writer.append(make_frame(0.5))
        with pytest.raises(CacheError, match="strictly increasing"):
            writer.append(make_frame(0.5))
        with pytest.raises(CacheError, match="strictly increasing"):
            writer.append(make_frame(0.1))
        with pytest.raises(CacheError):
            writer.append("not-a-frame")  # type: ignore[arg-type]
        writer.commit()
    assert load_cache(path).complete is True
    with pytest.raises(CacheError):
        writer.append(make_frame(9.0))
    with pytest.raises(CacheError):
        writer.commit()
