from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from serve_review.cli import build_parser
from serve_review.domain import SourceMetadata
from serve_review.media.probe import fingerprint_for_path
from serve_review.workflow.cli import cleanup_command
from serve_review.workflow.records import SourceRecord, source_id_for_fingerprint
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def parse(*arguments: str):
    return build_parser().parse_args(["clean", *arguments])


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


def test_clean_parser_exposes_only_dry_run_inventory_options(tmp_path: Path) -> None:
    args = parse(
        "--dry-run",
        "--workspace",
        str(tmp_path / "workspace"),
        "--source",
        "source-0123456789abcdef",
        "--stale",
        "--failed-runs",
        "--temporary",
    )
    assert args.handler is cleanup_command
    assert args.dry_run and args.stale and args.failed_runs and args.temporary
    with pytest.raises(SystemExit) as error:
        parse("--dry-run", "--confirm")
    assert error.value.code == 2


def test_clean_without_dry_run_is_rejected_without_creating_workspace(
    tmp_path: Path, capsys
) -> None:
    workspace = tmp_path / "missing"
    assert cleanup_command(parse("--workspace", str(workspace))) == 2
    assert "requires --dry-run" in capsys.readouterr().err
    assert not workspace.exists()


def test_clean_dry_run_prints_candidates_protected_and_totals(
    tmp_path: Path, capsys
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    cache = workspace.sources / source.source_id / "cache" / "track.jsonl"
    cache.parent.mkdir(parents=True)
    cache.write_text("cache", encoding="utf-8")
    temporary = workspace.temporary / "index.html"
    temporary.parent.mkdir(parents=True)
    temporary.write_text("page", encoding="utf-8")

    result = cleanup_command(
        parse("--dry-run", "--workspace", str(workspace.root))
    )
    assert result == 0
    output = capsys.readouterr().out
    assert f"CANDIDATE recomputable-cache {cache}" in output
    assert "PROTECTED manifest-provenance" in output
    assert "PROTECTED source-media" in output
    assert "Totals:" in output
    assert "candidates:" in output and "protected/excluded:" in output


def test_clean_passes_selectors_to_injected_inventory(tmp_path: Path) -> None:
    observed = []

    class Result:
        items = ()
        candidates = ()
        protected = ()
        totals = {}

    def inventory(workspace, **kwargs):
        observed.append((workspace, kwargs))
        return Result()

    args = parse(
        "--dry-run",
        "--workspace",
        str(tmp_path / "workspace"),
        "--source",
        "source-0123456789abcdef",
        "--stale",
        "--temporary",
    )
    assert cleanup_command(args, services={"inventory_cleanup": inventory}) == 0
    assert observed[0][1] == {
        "source": ["source-0123456789abcdef"],
        "stale": True,
        "failed_runs": False,
        "temporary": True,
    }


def test_clean_never_calls_filesystem_mutation_primitives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    artifact = workspace.temporary / "index.html"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("page", encoding="utf-8")
    before = {
        path: path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("cleanup attempted filesystem mutation")

    monkeypatch.setattr(Path, "unlink", forbidden)
    monkeypatch.setattr(os, "remove", forbidden)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(shutil, "rmtree", forbidden)
    assert cleanup_command(
        parse("--dry-run", "--workspace", str(workspace.root))
    ) == 0
    assert {
        path: path.read_bytes()
        for path in workspace.root.rglob("*")
        if path.is_file()
    } == before
    assert Path(source.known_paths[0]).read_bytes() == b"source"
