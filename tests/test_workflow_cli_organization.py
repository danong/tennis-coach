from __future__ import annotations

from pathlib import Path

import pytest

from serve_review.cli import build_parser
from serve_review.domain import SourceMetadata
from serve_review.workflow.cli import organization_command
from serve_review.workflow.collections import (
    create_collection,
    load_collection,
)
from serve_review.workflow.records import SourceRecord, source_id_for_fingerprint
from serve_review.workflow.sessions import create_session, load_session
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


SESSION_ID = "session-0123456789abcdef"
OTHER_SESSION_ID = "session-fedcba9876543210"
COLLECTION_ID = "collection-0123456789abcdef"
OTHER_COLLECTION_ID = "collection-fedcba9876543210"
ATTEMPT_ID = "attempt-0123456789abcdef01234567"


def parse(*arguments: str):
    return build_parser().parse_args(list(arguments))


def make_source(workspace: WorkspacePaths, path: Path, digit: str = "a") -> SourceRecord:
    path.write_bytes(b"source")
    fingerprint = "sha256:" + digit * 64
    metadata = SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=1,
        width=1,
        height=1,
        frame_rate_num=1,
        video_codec="h264",
    )
    record = SourceRecord(
        1,
        source_id_for_fingerprint(fingerprint),
        fingerprint,
        metadata,
        (str(path.resolve()),),
    )
    write_json_atomic(source_record_path(workspace, record.source_id), record.to_dict())
    return record


def run(arguments: tuple[str, ...], services=None) -> int:
    return organization_command(parse(*arguments), services=services)


def test_parser_exposes_documented_organization_commands(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    forms = [
        ("session", "list", "--workspace", workspace),
        ("session", "show", "name", "--workspace", workspace),
        ("session", "create", "name", "--workspace", workspace),
        ("session", "add", "name", "video.mov", "--workspace", workspace),
        ("session", "remove", "name", "--source", "source-0123456789abcdef", "--workspace", workspace),
        ("session", "tag", "name", "--tag", "drill:serve", "--workspace", workspace),
        ("collection", "list", "--workspace", workspace),
        ("collection", "show", "name", "--workspace", workspace),
        ("collection", "create", "name", "--workspace", workspace),
        ("collection", "add", "name", "--attempt", ATTEMPT_ID, "--workspace", workspace),
        ("collection", "remove", "name", "--attempt", ATTEMPT_ID, "--workspace", workspace),
        ("collection", "tag", "name", "--tag", "favorite", "--workspace", workspace),
        ("collection", "delete", "name", "--workspace", workspace),
    ]
    for form in forms:
        args = parse(*form)
        assert args.handler is organization_command


def test_session_create_list_show_by_name_and_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    services = {"session_id_factory": lambda: SESSION_ID}
    assert run(("session", "create", "Practice", "--workspace", str(workspace)), services) == 0
    output = capsys.readouterr().out
    assert SESSION_ID in output
    assert str(workspace / "sessions" / f"{SESSION_ID}.json") in output

    assert run(("session", "list", "--workspace", str(workspace))) == 0
    output = capsys.readouterr().out
    assert SESSION_ID in output and "Practice" in output and "members: 0" in output

    for selector in (SESSION_ID, "Practice"):
        assert run(("session", "show", selector, "--workspace", str(workspace))) == 0
        detail = capsys.readouterr().out
        assert f"id: {SESSION_ID}" in detail
        assert "name: Practice" in detail
        assert "sources: -" in detail


def test_ambiguous_name_exits_two_and_lists_matching_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    create_session(workspace, "same", id_factory=lambda: SESSION_ID)
    create_session(workspace, "same", id_factory=lambda: OTHER_SESSION_ID)
    result = run(("session", "show", "same", "--workspace", str(workspace.root)))
    assert result == 2
    error = capsys.readouterr().err
    assert "ambiguous" in error
    assert SESSION_ID in error and OTHER_SESSION_ID in error


def test_session_add_registers_without_processing_and_remove_preserves_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = make_source(workspace, tmp_path / "video.mov")
    create_session(workspace, "Practice", id_factory=lambda: SESSION_ID)
    artifact = workspace.sources / source.source_id / "artifact.bin"
    artifact.write_bytes(b"keep")
    processed = False

    def forbidden(*args, **kwargs):
        nonlocal processed
        processed = True
        raise AssertionError("organization command processed media")

    services = {
        "register_source": lambda *args, **kwargs: source,
        "process_source": forbidden,
    }
    assert run(
        (
            "session", "add", "Practice", str(tmp_path / "video.mov"),
            "--workspace", str(workspace.root),
        ),
        services,
    ) == 0
    assert load_session(workspace, SESSION_ID).source_ids == (source.source_id,)
    assert processed is False
    assert "Manifest:" in capsys.readouterr().out

    assert run(
        (
            "session", "remove", SESSION_ID, "--source", source.source_id,
            "--workspace", str(workspace.root),
        )
    ) == 0
    assert load_session(workspace, SESSION_ID).source_ids == ()
    assert artifact.read_bytes() == b"keep"
    assert source_record_path(workspace, source.source_id).is_file()


def test_session_exclusivity_error_does_not_move_source(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = make_source(workspace, tmp_path / "video.mov")
    first = create_session(
        workspace, "one", id_factory=lambda: SESSION_ID, source_ids=(source.source_id,)
    )
    create_session(workspace, "two", id_factory=lambda: OTHER_SESSION_ID)
    result = run(
        (
            "session", "add", "two", str(tmp_path / "video.mov"),
            "--workspace", str(workspace.root),
        ),
        {"register_source": lambda *args, **kwargs: source},
    )
    assert result == 2
    assert load_session(workspace, first.session_id).source_ids == (source.source_id,)
    assert load_session(workspace, OTHER_SESSION_ID).source_ids == ()


def test_tag_commands_preserve_existing_tags(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    create_session(
        workspace, "Practice", id_factory=lambda: SESSION_ID, tags=("existing",)
    )
    assert run(
        (
            "session", "tag", "Practice", "--tag", "new", "--tag", "alpha",
            "--workspace", str(workspace.root),
        )
    ) == 0
    assert load_session(workspace, SESSION_ID).tags == ("alpha", "existing", "new")

    create_collection(
        workspace, "Favorites", id_factory=lambda: COLLECTION_ID, tags=("existing",)
    )
    assert run(
        (
            "collection", "tag", COLLECTION_ID, "--tag", "new",
            "--workspace", str(workspace.root),
        )
    ) == 0
    assert load_collection(workspace, COLLECTION_ID).tags == ("existing", "new")


def test_collection_references_are_nonexclusive_and_delete_is_manifest_only(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = make_source(workspace, tmp_path / "video.mov")
    create_session(
        workspace, "session", id_factory=lambda: SESSION_ID, source_ids=(source.source_id,)
    )
    create_collection(workspace, "one", id_factory=lambda: COLLECTION_ID)
    create_collection(workspace, "two", id_factory=lambda: OTHER_COLLECTION_ID)
    registration = {"register_source": lambda *args, **kwargs: source}

    for collection in (COLLECTION_ID, OTHER_COLLECTION_ID):
        assert run(
            (
                "collection", "add", collection, str(tmp_path / "video.mov"),
                "--attempt", ATTEMPT_ID, "--workspace", str(workspace.root),
            ),
            registration,
        ) == 0
        record = load_collection(workspace, collection)
        assert record.source_ids == (source.source_id,)
        assert record.attempt_ids == (ATTEMPT_ID,)

    assert run(
        (
            "collection", "remove", "one", "--source", source.source_id,
            "--attempt", ATTEMPT_ID, "--workspace", str(workspace.root),
        )
    ) == 0
    assert load_collection(workspace, COLLECTION_ID).source_ids == ()
    assert load_collection(workspace, COLLECTION_ID).attempt_ids == ()

    artifact = workspace.sources / source.source_id / "analysis.json"
    artifact.write_text("keep", encoding="utf-8")
    assert run(
        ("collection", "delete", "one", "--workspace", str(workspace.root))
    ) == 0
    assert artifact.read_text(encoding="utf-8") == "keep"
    assert source_record_path(workspace, source.source_id).is_file()
    assert load_session(workspace, SESSION_ID).source_ids == (source.source_id,)
    assert load_collection(workspace, OTHER_COLLECTION_ID).source_ids == (source.source_id,)


def test_collection_add_remove_require_references(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    create_collection(workspace, "one", id_factory=lambda: COLLECTION_ID)
    assert run(
        ("collection", "add", "one", "--workspace", str(workspace.root))
    ) == 2
    assert run(
        ("collection", "remove", "one", "--workspace", str(workspace.root))
    ) == 2
    assert run(
        (
            "collection", "add", "one", "--attempt", "bad-attempt",
            "--workspace", str(workspace.root),
        )
    ) == 2
