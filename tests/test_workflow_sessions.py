from __future__ import annotations

import json
from pathlib import Path

import pytest

from serve_review.domain import SourceMetadata
from serve_review.workflow.records import (
    SessionRecord,
    SourceRecord,
    WorkflowRecordError,
    source_id_for_fingerprint,
)
from serve_review.workflow.sessions import (
    SessionStoreError,
    add_source,
    create_session,
    delete_session,
    list_sessions,
    load_session,
    remove_source,
    rename_session,
    resolve_session,
    session_record_path,
    set_session_tags,
)
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def _source(workspace: WorkspacePaths, digit: str = "a") -> str:
    fingerprint = "sha256:" + digit * 64
    source_id = source_id_for_fingerprint(fingerprint)
    metadata = SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=1.0,
        width=1,
        height=1,
        frame_rate_num=1,
        frame_rate_den=1,
        video_codec="x",
        rotation_degrees=0,
        time_base_num=1,
        time_base_den=1,
    )
    record = SourceRecord(
        1, source_id, fingerprint, metadata, (f"/source/{digit}.mov",)
    )
    write_json_atomic(source_record_path(workspace, source_id), record.to_dict())
    return source_id


def test_session_record_codec_is_strict_and_deterministic() -> None:
    record = SessionRecord(
        1,
        "session-0123456789abcdef",
        "  Name  ",
        ("source-0123456789abcdef",),
        ("a", "b"),
    )
    assert SessionRecord.from_json(record.to_json()) == record
    assert record.to_json() == record.to_json()
    assert record.to_json().endswith("\n")

    payload = record.to_dict()
    invalid = [
        {**payload, "schema_version": True},
        {**payload, "schema_version": 2},
        {**payload, "display_name": " "},
        {**payload, "session_id": "../bad"},
        {**payload, "source_ids": [payload["source_ids"][0]] * 2},
        {**payload, "tags": ["b", "a"]},
        {**payload, "tags": ["a", "a"]},
        {**payload, "extra": 1},
    ]
    missing = dict(payload)
    missing.pop("display_name")
    invalid.append(missing)
    for value in invalid:
        with pytest.raises(WorkflowRecordError):
            SessionRecord.from_dict(value)

    for text in (
        "[]",
        "{bad",
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
    ):
        with pytest.raises(WorkflowRecordError):
            SessionRecord.from_json(text)


def test_crud_order_resolve_and_source_survives(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "w")
    first_source = _source(workspace, "a")
    second_source = _source(workspace, "b")
    ids = iter(("session-ffffffffffffffff", "session-0000000000000001"))
    first = create_session(
        workspace,
        "same",
        id_factory=lambda: next(ids),
        source_ids=(first_source,),
        tags=("z", "a"),
    )
    second = create_session(workspace, "same", id_factory=lambda: next(ids))
    assert [record.session_id for record in list_sessions(workspace)] == [
        second.session_id,
        first.session_id,
    ]
    assert load_session(workspace, first.session_id) == first
    assert resolve_session(workspace, first.session_id) == first
    with pytest.raises(SessionStoreError, match="ambiguous"):
        resolve_session(workspace, "same")

    renamed = rename_session(workspace, first.session_id, "new")
    assert resolve_session(workspace, "new") == renamed
    tagged = set_session_tags(workspace, first.session_id, ("c", "b"))
    assert tagged.tags == ("b", "c")
    added = add_source(workspace, first.session_id, second_source)
    assert added.source_ids == (first_source, second_source)
    removed = remove_source(workspace, first.session_id, first_source)
    assert removed.source_ids == (second_source,)

    artifact = workspace.sources / first_source / "artifact.bin"
    artifact.write_bytes(b"keep")
    source_manifest = source_record_path(workspace, first_source)
    source_before = source_manifest.read_bytes()
    delete_session(workspace, first.session_id)
    assert artifact.read_bytes() == b"keep"
    assert source_manifest.read_bytes() == source_before
    with pytest.raises(SessionStoreError, match="missing|not found"):
        load_session(workspace, first.session_id)


def test_source_exclusivity_on_create_and_add(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "w")
    source_id = _source(workspace, "c")
    first_id = "session-0000000000000001"
    second_id = "session-0000000000000002"
    create_session(
        workspace,
        "one",
        id_factory=lambda: first_id,
        source_ids=(source_id,),
    )
    create_session(workspace, "two", id_factory=lambda: second_id)
    with pytest.raises(SessionStoreError) as create_error:
        create_session(
            workspace,
            "three",
            id_factory=lambda: "session-0000000000000003",
            source_ids=(source_id,),
        )
    assert source_id in str(create_error.value)
    assert first_id in str(create_error.value)
    assert "session-0000000000000003" in str(create_error.value)

    with pytest.raises(SessionStoreError) as add_error:
        add_source(workspace, second_id, source_id)
    assert source_id in str(add_error.value)
    assert first_id in str(add_error.value)
    assert second_id in str(add_error.value)


def test_duplicate_missing_and_invalid_mutations_are_explicit(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "w")
    source_id = _source(workspace, "d")
    session = create_session(
        workspace,
        "one",
        id_factory=lambda: "session-0000000000000001",
        source_ids=(source_id,),
    )
    with pytest.raises(SessionStoreError, match="already in"):
        add_source(workspace, session.session_id, source_id)
    with pytest.raises(SessionStoreError, match="not in"):
        remove_source(
            workspace, session.session_id, "source-0123456789abcdef"
        )
    with pytest.raises(SessionStoreError, match="does not exist"):
        add_source(
            workspace, session.session_id, "source-1111111111111111"
        )
    with pytest.raises(SessionStoreError, match="tags"):
        set_session_tags(workspace, session.session_id, "tag")
    with pytest.raises(SessionStoreError, match="not found"):
        resolve_session(workspace, "missing")


def test_id_factory_once_and_collision_without_overwrite(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "w")
    calls = 0

    def make_id() -> str:
        nonlocal calls
        calls += 1
        return "session-0000000000000001"

    original = create_session(workspace, "one", id_factory=make_id)
    assert calls == 1
    before = session_record_path(workspace, original.session_id).read_bytes()
    with pytest.raises(SessionStoreError, match="collision"):
        create_session(workspace, "replacement", id_factory=make_id)
    assert calls == 2
    assert session_record_path(workspace, original.session_id).read_bytes() == before


def test_manifest_filename_must_match_embedded_id(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "w")
    filename_id = "session-0000000000000001"
    embedded_id = "session-0000000000000002"
    payload = SessionRecord(1, embedded_id, "wrong", (), ()).to_dict()
    write_json_atomic(session_record_path(workspace, filename_id), payload)
    with pytest.raises(SessionStoreError, match=f"contains ID {embedded_id}"):
        list_sessions(workspace)
    with pytest.raises(SessionStoreError, match=f"expected {filename_id}"):
        load_session(workspace, filename_id)


def test_malformed_or_non_file_session_entries_fail(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "w")
    workspace.sessions.mkdir(parents=True)
    (workspace.sessions / "notes.txt").write_text("ignore? no", encoding="utf-8")
    with pytest.raises(SessionStoreError, match="malformed/non-file"):
        list_sessions(workspace)

    (workspace.sessions / "notes.txt").unlink()
    malformed = workspace.sessions / "session-0000000000000001.json"
    malformed.write_text(json.dumps({"bad": True}), encoding="utf-8")
    with pytest.raises(SessionStoreError, match="could not load"):
        list_sessions(workspace)
