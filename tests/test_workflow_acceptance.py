"""Public delivery-stage-2 workflow acceptance tests (offline and boundary-injected)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from serve_review.cli import build_parser
from serve_review.domain import SourceMetadata
from serve_review.workflow.cli import cleanup_command, compare_command, process_command, status_command
from serve_review.workflow.collections import list_collections
from serve_review.workflow.records import (
    SourceRecord,
    StatusDocument,
    source_id_for_fingerprint,
)
from serve_review.workflow.sessions import list_sessions
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def fp(char: str = "a") -> str:
    return "sha256:" + char * 64


def record(path: Path, char: str = "a") -> SourceRecord:
    fingerprint = fp(char)
    metadata = SourceMetadata(fingerprint=fingerprint, duration_seconds=2.0, width=1, height=1, frame_rate_num=1, video_codec="h264")
    return SourceRecord(1, source_id_for_fingerprint(fingerprint), fingerprint, metadata, (str(path.resolve()),))


def parse(*args: str):
    return build_parser().parse_args(list(args))


def render(_page, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("landing", encoding="utf-8")
    return destination


def batch(_pages, destination):
    return render(None, destination)


def services(source: SourceRecord, *, processed=None):
    def register(_path, workspace, **_kwargs):
        destination = source_record_path(workspace, source.source_id)
        write_json_atomic(destination, source.to_dict())
        return source
    return {
        "fingerprint": lambda _path: source.source_fingerprint,
        "source_lookup": lambda _value: None,
        "register_source": register,
        "process_source": lambda *_args, **_kwargs: processed or SimpleNamespace(overall_status="complete", attempts=()),
        "render": render,
        "render_batch": batch,
    }


def test_process_dry_run_has_no_writes(tmp_path, capsys):
    video = tmp_path / "serve.mov"
    video.write_bytes(b"fixture")
    workspace = tmp_path / "workspace"
    assert process_command(parse("process", str(video), "--workspace", str(workspace), "--dry-run"), services=services(record(video))) == 0
    assert not workspace.exists()
    assert "Mode: normal" in capsys.readouterr().out


def test_one_source_process_publishes_landing(tmp_path, capsys):
    video = tmp_path / "serve.mov"; video.write_bytes(b"fixture")
    workspace = tmp_path / "workspace"; source = record(video)
    assert process_command(parse("process", str(video), "--workspace", str(workspace)), services=services(source)) == 0
    assert "Primary landing:" in capsys.readouterr().out
    assert list(workspace.rglob("index.html"))


def test_partial_two_source_batch_returns_three(tmp_path, capsys):
    first = tmp_path / "one.mov"; second = tmp_path / "two.mov"
    first.write_bytes(b"one"); second.write_bytes(b"two")
    records = {first: record(first, "a"), second: record(second, "b")}
    def register(path, workspace, **_kwargs):
        item = records[path]
        write_json_atomic(source_record_path(workspace, item.source_id), item.to_dict())
        return item
    def process(source, *_args, **_kwargs):
        if source is records[second]: raise RuntimeError("injected failure")
        return SimpleNamespace(overall_status="complete", attempts=())
    svc = services(records[first]); svc.update({"fingerprint": lambda path: records[Path(path)].source_fingerprint, "register_source": register, "process_source": process})
    assert process_command(parse("process", str(first), str(second), "--workspace", str(tmp_path / "w")), services=svc) == 3
    assert "processing failed" in capsys.readouterr().err


def test_named_session_after_registration(tmp_path):
    video = tmp_path / "serve.mov"; video.write_bytes(b"fixture")
    workspace = tmp_path / "workspace"; source = record(video)
    assert process_command(parse("process", str(video), "--session", "Practice", "--workspace", str(workspace)), services=services(source)) == 0
    sessions = list_sessions(WorkspacePaths(workspace))
    assert len(sessions) == 1
    assert sessions[0].display_name == "Practice"
    assert sessions[0].source_ids == (source.source_id,)


def test_unnamed_compare_uses_temporary_only(tmp_path, capsys):
    video = tmp_path / "serve.mov"; video.write_bytes(b"fixture")
    workspace = tmp_path / "workspace"; source = record(video)
    process_command(parse("process", str(video), "--workspace", str(workspace)), services=services(source))
    run_files_before = tuple(sorted((workspace / "runs").glob("*.json")))
    assert compare_command(parse("compare", source.source_id, "--workspace", str(workspace)), services={"render": render, "render_batch": batch}) == 0
    capsys.readouterr()
    assert not list((workspace / "sessions").glob("*.json"))
    assert not list((workspace / "collections").glob("*.json"))
    assert tuple(sorted((workspace / "runs").glob("*.json"))) == run_files_before
    assert (workspace / "temporary").exists()


def test_compare_save_as_records_explicit_collection(tmp_path, capsys):
    video = tmp_path / "serve.mov"; video.write_bytes(b"fixture")
    workspace = tmp_path / "workspace"; source = record(video)
    process_command(parse("process", str(video), "--workspace", str(workspace)), services=services(source))
    assert compare_command(parse("compare", source.source_id, "--save-as", "Saved", "--workspace", str(workspace)), services={"render": render, "render_batch": batch}) == 0
    collections = list_collections(WorkspacePaths(workspace))
    assert len(collections) == 1
    assert collections[0].display_name == "Saved"
    assert collections[0].source_ids == (source.source_id,)
    assert collections[0].attempt_ids == ()
    assert "Collection:" in capsys.readouterr().out


def test_status_json_is_strict_and_read_only(tmp_path, capsys):
    video = tmp_path / "serve.mov"; video.write_bytes(b"fixture")
    workspace = tmp_path / "workspace"; source = record(video)
    process_command(parse("process", str(video), "--workspace", str(workspace)), services=services(source))
    before = sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*"))
    capsys.readouterr()
    assert status_command(parse("status", "--json", "--workspace", str(workspace))) == 0
    document = StatusDocument.from_json(capsys.readouterr().out)
    assert document.schema_version == 1 and document.workspace
    assert before == sorted(str(p.relative_to(workspace)) for p in workspace.rglob("*"))


def test_clean_dry_run_does_not_mutate(tmp_path, capsys):
    workspace = tmp_path / "workspace"; (workspace / "temporary").mkdir(parents=True)
    marker = workspace / "temporary" / "old.html"; marker.write_text("x")
    before = {path: path.read_bytes() for path in workspace.rglob("*") if path.is_file()}
    assert cleanup_command(parse("clean", "--dry-run", "--temporary", "--workspace", str(workspace))) == 0
    after = {path: path.read_bytes() for path in workspace.rglob("*") if path.is_file()}
    assert after == before
    assert "Totals:" in capsys.readouterr().out
