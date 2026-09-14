from __future__ import annotations

from pathlib import Path

import pytest

from serve_review.domain import AttemptDocument, MediaRange, PhaseDocument, SourceMetadata
from serve_review.media.probe import fingerprint_for_path
from serve_review.workflow.collections import create_collection
from serve_review.workflow.records import (
    AttemptOutcome,
    SourceRecord,
    StatusDocument,
    WorkflowRecordError,
    WorkflowRunRecord,
    attempt_id_for,
    source_id_for_fingerprint,
)
from serve_review.workflow.sessions import create_session
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.status import project_status, render_human
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def register(
    workspace: WorkspacePaths, video: Path, data: bytes = b"source bytes"
) -> SourceRecord:
    video.write_bytes(data)
    fingerprint = fingerprint_for_path(video)
    metadata = SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=10,
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


def write_run(
    workspace: WorkspacePaths,
    source: SourceRecord,
    run_id: str,
    checkpoint: Path | None = None,
) -> WorkflowRunRecord:
    if checkpoint is None:
        attempts = ()
        detection_status = "empty"
        overall = "empty"
    else:
        attempt_id = attempt_id_for(source.source_id, "method", "config", 1, 2)
        attempts = (
            AttemptOutcome(
                attempt_id,
                1,
                2,
                "complete",
                None,
                (str(checkpoint.resolve()),),
            ),
        )
        detection_status = "complete"
        overall = "complete"
    record = WorkflowRunRecord(
        1,
        run_id,
        source.source_id,
        source.source_fingerprint,
        "normal",
        "method",
        "config",
        detection_status,
        None,
        attempts,
        overall,
    )
    write_json_atomic(workspace.runs / f"{run_id}.json", record.to_dict())
    return record


def write_attempts(workspace: WorkspacePaths, source: SourceRecord, fingerprint: str | None = None) -> Path:
    destination = (
        workspace.sources
        / source.source_id
        / "compatibility"
        / Path(source.known_paths[0]).stem
        / "attempts.json"
    )
    document = AttemptDocument(
        source_fingerprint=fingerprint or source.source_fingerprint,
        source_duration_seconds=10,
        padding_seconds=1,
        attempts=(),
        export_ranges=(),
    )
    write_json_atomic(destination, document.to_dict())
    return destination


def test_status_reports_paths_memberships_artifacts_and_honest_unknown(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    write_attempts(workspace, source)
    create_session(
        workspace,
        "Practice",
        id_factory=lambda: "session-0123456789abcdef",
        source_ids=(source.source_id,),
    )
    create_collection(
        workspace,
        "Favorites",
        id_factory=lambda: "collection-0123456789abcdef",
        source_ids=(source.source_id,),
    )
    landing = workspace.sources / source.source_id / "index.html"
    landing.write_text("page", encoding="utf-8")
    write_run(workspace, source, "run-0123456789abcdef")

    document = project_status(workspace)
    item = document.sources[0]
    assert item.source.state == "present"
    assert item.sessions == ("session-0123456789abcdef",)
    assert item.collections == ("collection-0123456789abcdef",)
    assert item.runs.state == item.latest_run.state == "present"
    assert item.latest_run.value == "empty"
    assert item.compatibility.state == "present"
    assert item.checkpoints.state == "missing"
    assert item.cache.state == "unknown"
    assert "not provable" in item.cache.reason
    assert item.attempt_count == 0
    assert item.landing_paths == (str(landing),)
    assert "accepted" not in render_human(document)
    assert StatusDocument.from_json(document.to_json()) == document


def test_missing_stale_and_corrupt_are_distinct(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    video = tmp_path / "video.mov"
    source = register(workspace, video)
    assert project_status(workspace).sources[0].compatibility.state == "missing"

    path = write_attempts(workspace, source, "sha256:" + "f" * 64)
    assert project_status(workspace).sources[0].compatibility.state == "stale"

    path.write_text("{bad", encoding="utf-8")
    assert project_status(workspace).sources[0].compatibility.state == "corrupt"

    video.unlink()
    assert project_status(workspace).sources[0].source.state == "missing"


def test_checkpoint_presence_corruption_and_staleness(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    checkpoint = workspace.sources / source.source_id / "attempts" / "x" / "checkpoints.json"
    checkpoint.parent.mkdir(parents=True)
    valid = PhaseDocument(
        source_fingerprint=source.source_fingerprint,
        source_duration_seconds=10,
        attempts=[],
    )
    checkpoint.write_text(valid.to_json(), encoding="utf-8")
    write_run(workspace, source, "run-0123456789abcdef", checkpoint)
    assert project_status(workspace).sources[0].checkpoints.state == "present"

    checkpoint.write_text("{bad", encoding="utf-8")
    assert project_status(workspace).sources[0].checkpoints.state == "corrupt"

    stale = PhaseDocument(
        source_fingerprint="sha256:stale",
        source_duration_seconds=10,
        attempts=[],
    )
    checkpoint.write_text(stale.to_json(), encoding="utf-8")
    assert project_status(workspace).sources[0].checkpoints.state == "stale"


def test_multiple_runs_do_not_invent_latest_and_attempts_are_deduplicated(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    write_run(workspace, source, "run-0000000000000001")
    write_run(workspace, source, "run-0000000000000002")
    item = project_status(workspace).sources[0]
    assert item.latest_run.state == "unknown"
    assert item.latest_run.reason == "run ordering is not recorded"


def test_source_session_collection_and_path_selectors(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    first_path = tmp_path / "one.mov"
    second_path = tmp_path / "two.mov"
    first = register(workspace, first_path)
    second = register(workspace, second_path, b"different bytes")
    create_session(
        workspace,
        "Practice",
        id_factory=lambda: "session-0123456789abcdef",
        source_ids=(first.source_id,),
    )
    create_collection(
        workspace,
        "Favorites",
        id_factory=lambda: "collection-0123456789abcdef",
        source_ids=(second.source_id,),
    )
    selectors = [
        ("source_id", first.source_id, first.source_id),
        ("source_path", str(first_path), first.source_id),
        ("session", "Practice", first.source_id),
        ("collection", "Favorites", second.source_id),
    ]
    for kind, value, expected in selectors:
        assert project_status(workspace, kind, value).sources[0].source_id == expected
    with pytest.raises(ValueError):
        project_status(workspace, "source_id", "../../escape")


def test_malformed_registry_and_symlink_escape_are_reported_without_mutation(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    before = set(workspace.root.rglob("*"))
    outside = tmp_path / "outside"
    outside.mkdir()
    compatibility = workspace.sources / source.source_id / "compatibility"
    try:
        compatibility.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    item = project_status(workspace).sources[0]
    assert item.compatibility.state == "unavailable"
    assert set(workspace.root.rglob("*")) == before | {compatibility}


def test_status_document_rejects_noncanonical_json(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    document = project_status(workspace)
    with pytest.raises(WorkflowRecordError):
        StatusDocument.from_dict({**document.to_dict(), "schema_version": True})
    with pytest.raises(WorkflowRecordError):
        StatusDocument.from_json('{"schema_version":1,"schema_version":1}')
