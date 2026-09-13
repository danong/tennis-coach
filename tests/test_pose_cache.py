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


# --- Sampler orientation quarantine ----------------------------------------


def test_header_pins_sampler_orientation_version() -> None:
    from serve_review.media.frames import SAMPLER_ORIENTATION_VERSION

    assert SAMPLER_ORIENTATION_VERSION == 2
    header = header_to_dict(make_identity())
    assert header["sampler_orientation_version"] == 2
    assert header_from_dict(header) == make_identity()


def test_matching_orientation_header_hits_and_resumes(tmp_path: Path) -> None:
    identity = make_identity()
    frames = make_frames(3)
    path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(path, identity, frames)
    stored_header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert stored_header["sampler_orientation_version"] == 2
    snapshot = load_cache(path)
    assert snapshot.complete is True
    require_matching_identity(snapshot.identity, identity)
    # Partial cache with matching v2 resumes.
    partial = tmp_path / "partial.jsonl"
    write_partial_cache(partial, identity, frames[:2])
    rest = make_frames(1, start=frames[1].time_seconds + 1 / 30.0)
    append_frames(partial, identity, rest)
    assert load_cache(partial).complete is False
    finalize_cache(partial, identity)
    assert load_cache(partial).complete is True


def test_header_missing_orientation_quarantines_as_stale(tmp_path: Path) -> None:
    identity = make_identity()
    header = header_to_dict(identity)
    del header["sampler_orientation_version"]
    with pytest.raises(CacheStaleError, match="sampler_orientation_version"):
        header_from_dict(header)
    # End to end: a complete cache file without the field fails closed.
    path = tmp_path / "pose-v1.jsonl"
    lines = [json.dumps(header, sort_keys=True) + "\n"]
    for frame in make_frames(2):
        lines.append(_frame_line(frame))
    lines.append(
        json.dumps(
            {"complete": True, "frame_count": 2, "schema_version": 1, "type": "footer"},
            sort_keys=True,
        )
        + "\n"
    )
    _write_lines(path, lines)
    with pytest.raises(CacheStaleError):
        load_cache(path)
    # Partial cache without the field is also stale, never resumable.
    partial = tmp_path / "partial.jsonl"
    _write_lines(partial, lines[:2])
    with pytest.raises(CacheStaleError):
        load_cache(partial)
    with pytest.raises(CacheStaleError):
        append_frames(partial, identity, make_frames(1, start=5.0))


@pytest.mark.parametrize("bad", [1, 0, 3, 999, "2", 2.0, True, None])
def test_header_orientation_mismatch_quarantines_as_stale(tmp_path: Path, bad) -> None:
    identity = make_identity()
    header = {**header_to_dict(identity), "sampler_orientation_version": bad}
    with pytest.raises(CacheStaleError, match="sampler_orientation_version"):
        header_from_dict(header)
    path = tmp_path / "pose-v1.jsonl"
    lines = [json.dumps(header, sort_keys=True) + "\n"]
    for frame in make_frames(1):
        lines.append(_frame_line(frame))
    _write_lines(path, lines)
    with pytest.raises(CacheStaleError):
        load_cache(path)
    with pytest.raises(CacheStaleError):
        append_frames(path, identity, make_frames(1, start=5.0))
    with pytest.raises(CacheStaleError):
        finalize_cache(path, identity)


def test_orientation_header_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(CacheError):
        header_to_dict("not-an-identity")  # type: ignore[arg-type]
    with pytest.raises(CacheError):
        require_matching_identity("x", make_identity())  # type: ignore[arg-type]
    with pytest.raises(CacheError):
        require_matching_identity(make_identity(), "x")  # type: ignore[arg-type]
    header = header_to_dict(make_identity())
    with pytest.raises(CacheCorruptError):
        header_from_dict({**header, "unexpected": 1})


# --- Dense-world cache (M4.7 leaf 1) -------------------------------------------
# The frozen 2D cache tests above are untouched; everything below exercises
# the new dense-world JSONL representation only.

from serve_review.pose.schema import FrameObservation as _Frame2D  # noqa: E402
from serve_review.pose.world import (  # noqa: E402
    NUM_WORLD_LANDMARKS,
    WORLD_CACHE_SCHEMA_VERSION,
    WORLD_REPRESENTATION,
    WORLD_SCHEMA_VERSION,
    WorldCacheIdentity,
    WorldFrameObservation,
    WorldLandmark,
    append_world_frames,
    finalize_world_cache,
    load_world_cache,
    quarantine_world_cache,
    require_matching_world_identity,
    world_header_from_dict,
    world_header_to_dict,
    write_complete_world_cache,
    write_partial_world_cache,
)


def _make_world_identity(**overrides) -> WorldCacheIdentity:
    fields = {
        "source_fingerprint": "sha256:worldabc",
        "model_name": "mediapipe-pose-landmarker-heavy",
        "model_version": "heavy-test",
    }
    fields.update(overrides)
    return WorldCacheIdentity(**fields)


def _make_world_frame(time_seconds: float) -> WorldFrameObservation:
    joints = tuple(
        None
        if index % 5 == 0 and index not in (1, 2)
        else WorldLandmark(
            x=0.01 * index - 0.1, y=1.0 + 0.004 * index, z=-0.15 + 0.003 * index
        )
        for index in range(NUM_WORLD_LANDMARKS)
    )
    assert any(joint is not None for joint in joints)
    companion = _Frame2D(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        persons=(),
    )
    return WorldFrameObservation(
        time_seconds=time_seconds,
        timestamp_ms=int(round(time_seconds * 1000)),
        world_landmarks=joints,
        frame_2d=companion,
    )


def _make_world_frames(count: int, *, start: float = 0.0) -> list:
    step = 1 / 30.0
    return [_make_world_frame(start + index * step) for index in range(count)]


def _world_frame_line(frame: WorldFrameObservation) -> str:
    return (
        json.dumps(
            {"observation": frame.to_dict(), "type": "dense-world-frame"},
            sort_keys=True,
        )
        + "\n"
    )


def test_world_cache_versions_are_pinned() -> None:
    assert WORLD_CACHE_SCHEMA_VERSION == 1
    assert WORLD_SCHEMA_VERSION == 1
    assert WORLD_REPRESENTATION == "dense-world-v1"


def test_world_complete_round_trip_preserves_identity_and_times(tmp_path: Path) -> None:
    identity = _make_world_identity()
    frames = _make_world_frames(4)
    path = tmp_path / "world-v1.jsonl"
    write_complete_world_cache(path, identity, frames)
    snapshot = load_world_cache(path)
    assert snapshot.complete is True
    assert snapshot.identity == identity
    assert list(snapshot.frames) == frames
    for stored, expected in zip(snapshot.frames, frames):
        assert stored.time_seconds == expected.time_seconds
        assert stored.timestamp_ms == expected.timestamp_ms
        assert stored.frame_2d.time_seconds == stored.time_seconds
        assert stored.frame_2d.timestamp_ms == stored.timestamp_ms
    times = [frame.time_seconds for frame in snapshot.frames]
    assert all(later > earlier for earlier, later in zip(times, times[1:]))


def test_world_cache_file_is_deterministic_jsonl(tmp_path: Path) -> None:
    identity = _make_world_identity()
    frames = _make_world_frames(2)
    path = tmp_path / "world-v1.jsonl"
    write_complete_world_cache(path, identity, frames)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # header + 2 frames + footer
    for line in lines:
        assert json.loads(line)
    assert json.loads(lines[0])["type"] == "dense-world-header"
    assert json.loads(lines[0])["representation"] == WORLD_REPRESENTATION
    assert json.loads(lines[-1])["type"] == "dense-world-footer"
    assert json.loads(lines[-1]) == {
        "complete": True,
        "frame_count": 2,
        "schema_version": 1,
        "type": "dense-world-footer",
    }
    again = tmp_path / "again.jsonl"
    write_complete_world_cache(again, identity, frames)
    assert again.read_text(encoding="utf-8") == path.read_text(encoding="utf-8")


def test_world_empty_frames_round_trip_is_complete(tmp_path: Path) -> None:
    path = tmp_path / "world-v1.jsonl"
    write_complete_world_cache(path, _make_world_identity(), [])
    snapshot = load_world_cache(path)
    assert snapshot.complete is True
    assert snapshot.frames == ()


def test_world_header_codec_round_trips_and_validates(tmp_path: Path) -> None:
    identity = _make_world_identity()
    header = world_header_to_dict(identity)
    assert header["type"] == "dense-world-header"
    assert header["representation"] == WORLD_REPRESENTATION
    assert header["world_schema_version"] == WORLD_SCHEMA_VERSION
    assert world_header_from_dict(header) == identity
    with pytest.raises(CacheCorruptError):
        world_header_from_dict({**header, "type": "header"})
    with pytest.raises(CacheCorruptError):
        world_header_from_dict({**header, "schema_version": 999})
    with pytest.raises(CacheCorruptError):
        world_header_from_dict({**header, "world_schema_version": 999})
    with pytest.raises(CacheCorruptError):
        world_header_from_dict({**header, "representation": "pose-v1"})
    with pytest.raises(CacheCorruptError):
        world_header_from_dict({**header, "extra": 1})


def test_world_stale_identity_is_rejected_and_leaves_file_untouched(
    tmp_path: Path,
) -> None:
    stored = _make_world_identity()
    requested = _make_world_identity(source_fingerprint="sha256:other")
    with pytest.raises(CacheStaleError, match="source_fingerprint"):
        require_matching_world_identity(stored, requested)
    with pytest.raises(CacheStaleError, match="model_name"):
        require_matching_world_identity(
            stored, _make_world_identity(model_name="other-model")
        )
    path = tmp_path / "world-v1.jsonl"
    write_complete_world_cache(path, stored, _make_world_frames(2))
    with pytest.raises(CacheStaleError):
        append_world_frames(path, requested, _make_world_frames(1, start=5.0))
    with pytest.raises(CacheStaleError):
        finalize_world_cache(path, requested)
    assert load_world_cache(path).complete is True


def test_world_partial_cache_is_resumable_then_finalized(tmp_path: Path) -> None:
    identity = _make_world_identity()
    path = tmp_path / "world-v1.jsonl"
    first = _make_world_frames(3)
    write_partial_world_cache(path, identity, first)
    assert load_world_cache(path).complete is False
    rest = _make_world_frames(2, start=first[-1].time_seconds + 1 / 30.0)
    append_world_frames(path, identity, rest)
    resumed = load_world_cache(path)
    assert resumed.complete is False
    assert list(resumed.frames) == first + rest
    finalize_world_cache(path, identity)
    finished = load_world_cache(path)
    assert finished.complete is True
    assert list(finished.frames) == first + rest
    # Finalize is idempotent.
    before = path.read_text(encoding="utf-8")
    assert finalize_world_cache(path, identity) == path
    assert path.read_text(encoding="utf-8") == before


def test_world_append_to_complete_is_refused_and_order_enforced(
    tmp_path: Path,
) -> None:
    identity = _make_world_identity()
    path = tmp_path / "world-v1.jsonl"
    write_complete_world_cache(path, identity, _make_world_frames(2))
    with pytest.raises(CacheError, match="already complete"):
        append_world_frames(path, identity, _make_world_frames(1, start=5.0))
    assert load_world_cache(path).complete is True
    partial = tmp_path / "partial.jsonl"
    write_partial_world_cache(partial, identity, _make_world_frames(3))
    with pytest.raises(CacheCorruptError):
        append_world_frames(partial, identity, [_make_world_frame(0.01)])
    assert load_world_cache(partial).complete is False


def test_world_sparse_2d_caches_never_validate_as_dense(tmp_path: Path) -> None:
    # A sparse 2D pose-v1 file must fail as a dense-world cache.
    sparse_identity = make_identity()
    sparse_path = tmp_path / "pose-v1.jsonl"
    write_complete_cache(sparse_path, sparse_identity, make_frames(2))
    with pytest.raises(CacheCorruptError):
        load_world_cache(sparse_path)
    # And a dense-world file must fail as a sparse 2D cache.
    world_path = tmp_path / "world-v1.jsonl"
    write_complete_world_cache(
        world_path, _make_world_identity(), _make_world_frames(2)
    )
    with pytest.raises((CacheCorruptError, CacheError)):
        load_cache(world_path)


def test_world_corrupt_caches_raise_and_never_present_as_complete(
    tmp_path: Path,
) -> None:
    identity = _make_world_identity()
    good_frames = _make_world_frames(2)
    header = json.dumps(world_header_to_dict(identity), sort_keys=True) + "\n"
    frame_lines = [_world_frame_line(frame) for frame in good_frames]
    footer = (
        json.dumps(
            {
                "complete": True,
                "frame_count": 2,
                "schema_version": 1,
                "type": "dense-world-footer",
            },
            sort_keys=True,
        )
        + "\n"
    )
    cases = {
        "empty file": [],
        "missing header": frame_lines + [footer],
        "sparse 2D header": [
            json.dumps(header_to_dict(make_identity()), sort_keys=True) + "\n",
            *frame_lines,
        ],
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
                    "type": "dense-world-footer",
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
                    "type": "dense-world-footer",
                },
                sort_keys=True,
            )
            + "\n",
        ],
        "out-of-order times": [header, frame_lines[1], frame_lines[0], footer],
        "duplicate times": [header, frame_lines[0], frame_lines[0]],
        "bad observation": [
            header,
            json.dumps({"observation": {"nope": 1}, "type": "dense-world-frame"})
            + "\n",
        ],
        "non-finite world coordinate": [
            header,
            json.dumps(
                {
                    "observation": {
                        **good_frames[0].to_dict(),
                        "world_landmarks": [
                            {"x": float("inf"), "y": 0.0, "z": 0.0}
                        ]
                        + [None] * (NUM_WORLD_LANDMARKS - 1),
                    },
                    "type": "dense-world-frame",
                },
                sort_keys=True,
                allow_nan=True,
            )
            + "\n",
        ],
        "world/2D timestamp divergence": [
            header,
            json.dumps(
                {
                    "observation": {
                        **good_frames[0].to_dict(),
                        "time_seconds": good_frames[0].time_seconds + 1.0,
                    },
                    "type": "dense-world-frame",
                },
                sort_keys=True,
            )
            + "\n",
        ],
        "stale cache schema": [
            json.dumps({**json.loads(header), "schema_version": 999}) + "\n",
            *frame_lines,
        ],
        "stale world schema": [
            json.dumps({**json.loads(header), "world_schema_version": 999}) + "\n",
            *frame_lines,
        ],
    }
    for label, lines in cases.items():
        path = tmp_path / "world-v1.jsonl"
        path.write_text("".join(lines), encoding="utf-8")
        try:
            load_world_cache(path)
        except (CacheCorruptError, CacheError):
            continue
        raise AssertionError(f"corrupt world case did not raise: {label}")


def test_world_quarantine_moves_corrupt_file_aside(tmp_path: Path) -> None:
    path = tmp_path / "world-v1.jsonl"
    path.write_text("this is { not json\n", encoding="utf-8")
    with pytest.raises(CacheCorruptError):
        load_world_cache(path)
    quarantined = quarantine_world_cache(path)
    assert quarantined == tmp_path / "world-v1.jsonl.corrupt"
    assert quarantined.is_file()
    assert not path.exists()
    with pytest.raises(CacheError, match="does not exist"):
        quarantine_world_cache(path)
    with pytest.raises(CacheError, match="does not exist"):
        load_world_cache(path)
