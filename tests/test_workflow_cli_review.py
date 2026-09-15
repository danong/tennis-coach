from __future__ import annotations

from pathlib import Path

import pytest

from serve_review.cli import build_parser
from serve_review.domain import SourceMetadata
from serve_review.media.probe import fingerprint_for_path
from serve_review.workflow.cli import compare_command, review_command
from serve_review.workflow.collections import create_collection, list_collections
from serve_review.workflow.records import (
    AttemptOutcome,
    SourceRecord,
    WorkflowRunRecord,
    attempt_id_for,
    source_id_for_fingerprint,
)
from serve_review.workflow.sessions import create_session
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def parse(*arguments: str):
    return build_parser().parse_args(list(arguments))


def register(workspace: WorkspacePaths, path: Path, data: bytes) -> SourceRecord:
    path.write_bytes(data)
    fingerprint = fingerprint_for_path(path)
    metadata = SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=2,
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


def add_run(workspace: WorkspacePaths, source: SourceRecord, digit: str) -> str:
    attempt_id = attempt_id_for(source.source_id, "method", "config", 0, 1)
    artifact = str(
        (
            workspace.sources
            / source.source_id
            / "attempts"
            / attempt_id
            / "diagnostics.json"
        ).resolve()
    )
    outcome = AttemptOutcome(attempt_id, 0, 1, "complete", None, (artifact,))
    run = WorkflowRunRecord(
        1,
        f"run-{digit * 16}",
        source.source_id,
        source.source_fingerprint,
        "normal",
        "method",
        "config",
        "complete",
        None,
        (outcome,),
        "complete",
    )
    write_json_atomic(workspace.runs / f"{run.run_id}.json", run.to_dict())
    return attempt_id


def render_capture(captured):
    def render(page, destination):
        captured.append(page)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("page", encoding="utf-8")
        return path

    return render


def batch_render(entries, destination):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("index", encoding="utf-8")
    return path


def test_parser_exposes_review_and_compare_forms() -> None:
    review = parse("review", "source-0123456789abcdef", "--open")
    assert review.handler is review_command
    assert review.workspace is None
    compare = parse(
        "compare",
        "one.mov",
        "--source",
        "source-0123456789abcdef",
        "--session",
        "Practice",
        "--collection",
        "Favorites",
        "--stage",
        "contact",
        "--stages",
        "loading,finish",
        "--save-as",
        "Saved",
        "--open",
    )
    assert compare.handler is compare_command
    assert compare.workspace is None


def test_review_requires_exactly_one_selector_and_resolves_source(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov", b"one")
    captured = []
    args = parse(
        "review", source.source_id, "--workspace", str(workspace.root)
    )
    assert review_command(args, services={"render": render_capture(captured)}) == 0
    assert captured[0].source == source
    missing = captured[0].source_artifacts[0]
    assert missing.status == "missing"
    assert missing.remediation == (
        f"serve-review process {source.known_paths[0]} --workspace {workspace.root}"
    )
    assert str(workspace.temporary) in str(args.workspace.parent / "workspace" / "temporary")

    invalid = parse("review", "--workspace", str(workspace.root))
    assert review_command(invalid, services={"render": pytest.fail}) == 2
    combined = parse(
        "review",
        source.source_id,
        "--session",
        "Practice",
        "--workspace",
        str(workspace.root),
    )
    assert review_command(combined, services={"render": pytest.fail}) == 2


def test_review_session_collection_and_ambiguity(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    first = register(workspace, tmp_path / "one.mov", b"one")
    second = register(workspace, tmp_path / "two.mov", b"two")
    create_session(
        workspace,
        "Practice",
        id_factory=lambda: "session-0123456789abcdef",
        source_ids=(first.source_id, second.source_id),
    )
    create_collection(
        workspace,
        "Favorites",
        id_factory=lambda: "collection-0123456789abcdef",
        source_ids=(second.source_id,),
    )
    pages = []
    services = {"render": render_capture(pages), "render_batch": batch_render}
    assert review_command(
        parse("review", "--session", "Practice", "--workspace", str(workspace.root)),
        services=services,
    ) == 0
    assert {page.source.source_id for page in pages} == {
        first.source_id,
        second.source_id,
    }

    pages.clear()
    assert review_command(
        parse("review", "--collection", "Favorites", "--workspace", str(workspace.root)),
        services=services,
    ) == 0
    assert [page.source.source_id for page in pages] == [second.source_id]

    create_collection(
        workspace,
        "Favorites",
        id_factory=lambda: "collection-fedcba9876543210",
    )
    assert review_command(
        parse("review", "--collection", "Favorites", "--workspace", str(workspace.root)),
        services=services,
    ) == 2


def test_compare_selector_or_within_type_and_and_across_types(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    first = register(workspace, tmp_path / "one.mov", b"one")
    second = register(workspace, tmp_path / "two.mov", b"two")
    third = register(workspace, tmp_path / "three.mov", b"three")
    create_session(
        workspace,
        "session-one",
        id_factory=lambda: "session-0123456789abcdef",
        source_ids=(first.source_id, second.source_id),
    )
    create_session(
        workspace,
        "session-two",
        id_factory=lambda: "session-fedcba9876543210",
        source_ids=(third.source_id,),
    )
    create_collection(
        workspace,
        "collection-one",
        id_factory=lambda: "collection-0123456789abcdef",
        source_ids=(second.source_id, third.source_id),
    )
    pages = []
    services = {"render": render_capture(pages), "render_batch": batch_render}
    result = compare_command(
        parse(
            "compare",
            "--session",
            "session-one",
            "--session",
            "session-two",
            "--collection",
            "collection-one",
            "--workspace",
            str(workspace.root),
        ),
        services=services,
    )
    assert result == 0
    assert {page.source.source_id for page in pages} == {
        second.source_id,
        third.source_id,
    }

    pages.clear()
    result = compare_command(
        parse(
            "compare",
            str(tmp_path / "one.mov"),
            "--source",
            third.source_id,
            "--session",
            "session-one",
            "--workspace",
            str(workspace.root),
        ),
        services=services,
    )
    assert result == 0
    assert [page.source.source_id for page in pages] == [first.source_id]


def test_compare_requires_selector_and_stage_filter_is_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serve_review.workflow.cli as module

    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov", b"one")
    add_run(workspace, source, "1")
    assert compare_command(
        parse("compare", "--workspace", str(workspace.root)),
        services={"render": pytest.fail},
    ) == 2

    monkeypatch.setattr(
        module,
        "_stage_names",
        lambda path, workspace: {"contact"},
    )
    pages = []
    assert compare_command(
        parse(
            "compare", source.source_id, "--stage", "contact",
            "--stages", "loading,contact", "--workspace", str(workspace.root),
        ),
        services={"render": render_capture(pages)},
    ) == 0
    assert len(pages[0].attempts) == 1

    assert compare_command(
        parse(
            "compare", source.source_id, "--stage", "finish",
            "--workspace", str(workspace.root),
        ),
        services={"render": pytest.fail},
    ) == 2


def test_compare_save_as_stores_only_explicit_references(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov", b"one")
    attempt_id = add_run(workspace, source, "1")
    source_before = source_record_path(workspace, source.source_id).read_bytes()
    result = compare_command(
        parse(
            "compare", source.source_id, "--save-as", "Saved",
            "--workspace", str(workspace.root),
        ),
        services={
            "render": render_capture([]),
            "collection_id_factory": lambda: "collection-0123456789abcdef",
        },
    )
    assert result == 0
    collections = list_collections(workspace)
    assert len(collections) == 1
    assert collections[0].source_ids == (source.source_id,)
    assert collections[0].attempt_ids == (attempt_id,)
    assert source_record_path(workspace, source.source_id).read_bytes() == source_before
    assert len(list(workspace.runs.glob("*.json"))) == 1


def test_unnamed_compare_only_publishes_temporary_and_open_is_gated(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov", b"one")
    opened = []
    services = {
        "render": render_capture([]),
        "opener": lambda path: opened.append(path),
        "process_source": lambda *args, **kwargs: pytest.fail("must not process"),
        "register_source": lambda *args, **kwargs: pytest.fail("must not register"),
    }
    args = parse("compare", source.source_id, "--workspace", str(workspace.root))
    assert compare_command(args, services=services) == 0
    assert opened == []
    assert list_collections(workspace) == []
    assert not workspace.runs.exists()

    args.open_page = True
    assert compare_command(args, services=services) == 0
    assert len(opened) == 1
    assert str(workspace.temporary) in str(opened[0])
