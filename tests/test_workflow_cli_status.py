from __future__ import annotations

from pathlib import Path

import pytest

from serve_review.cli import build_parser
from serve_review.domain import SourceMetadata
from serve_review.media.probe import fingerprint_for_path
from serve_review.workflow.cli import status_command
from serve_review.workflow.records import SourceRecord, StatusDocument, source_id_for_fingerprint
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def parse(*arguments: str):
    return build_parser().parse_args(["status", *arguments])


def register(workspace: WorkspacePaths, video: Path) -> SourceRecord:
    video.write_bytes(b"source")
    fingerprint = fingerprint_for_path(video)
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
        (str(video.resolve()),),
    )
    write_json_atomic(source_record_path(workspace, record.source_id), record.to_dict())
    return record


def test_status_parser_forms_and_mutual_selector_validation(tmp_path: Path) -> None:
    workspace = str(tmp_path / "workspace")
    assert parse("--workspace", workspace).source is None
    assert parse("source-0123456789abcdef", "--workspace", workspace).source
    assert parse("--session", "Practice", "--workspace", workspace).session
    assert parse("--collection", "Favorites", "--workspace", workspace).collection
    with pytest.raises(SystemExit) as error:
        parse("--session", "a", "--collection", "b")
    assert error.value.code == 2


def test_human_and_json_status_outputs(tmp_path: Path, capsys) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")

    assert status_command(parse("--workspace", str(workspace.root))) == 0
    human = capsys.readouterr().out
    assert f"Workspace: {workspace.root}" in human
    assert source.source_id in human
    assert "cache=unknown" in human

    assert status_command(
        parse(source.source_id, "--workspace", str(workspace.root), "--json")
    ) == 0
    payload = capsys.readouterr().out
    document = StatusDocument.from_json(payload)
    assert [item.source_id for item in document.sources] == [source.source_id]

    assert status_command(
        parse(str(tmp_path / "video.mov"), "--workspace", str(workspace.root))
    ) == 0
    assert source.source_id in capsys.readouterr().out


def test_status_missing_workspace_is_read_only(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "missing"
    assert status_command(parse("--workspace", str(workspace))) == 0
    assert not workspace.exists()
    assert "Workspace:" in capsys.readouterr().out


def test_invalid_or_combined_selector_exits_two(tmp_path: Path, capsys) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    assert status_command(
        parse("source-does-not-exist", "--workspace", str(workspace.root))
    ) == 2
    assert "ERROR:" in capsys.readouterr().err

    args = parse(source.source_id, "--workspace", str(workspace.root))
    args.session = "Practice"
    assert status_command(args) == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_malformed_registry_exits_two_without_repair(tmp_path: Path, capsys) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    workspace.sources.mkdir(parents=True)
    malformed = workspace.sources / "junk"
    malformed.mkdir()
    before = set(workspace.root.rglob("*"))
    assert status_command(parse("--workspace", str(workspace.root))) == 2
    assert "malformed source entry" in capsys.readouterr().out
    assert set(workspace.root.rglob("*")) == before
