from __future__ import annotations

from pathlib import Path

import pytest

from serve_review.domain import AttemptDocument, SourceMetadata
from serve_review.media.probe import fingerprint_for_path
from serve_review.workflow.cleanup import (
    CLASSES,
    CleanupInventoryError,
    CleanupItem,
    inventory_cleanup,
)
from serve_review.workflow.records import SourceRecord, WorkflowRunRecord, source_id_for_fingerprint
from serve_review.workflow.sessions import create_session
from serve_review.workflow.sources import source_record_path
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def register(workspace: WorkspacePaths, video: Path, data: bytes = b"source") -> SourceRecord:
    video.write_bytes(data)
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


def failed_run(workspace: WorkspacePaths, source: SourceRecord) -> WorkflowRunRecord:
    record = WorkflowRunRecord(
        1,
        "run-0123456789abcdef",
        source.source_id,
        source.source_fingerprint,
        "normal",
        "method",
        "config",
        "failed",
        "detection failed",
        (),
        "failed",
    )
    write_json_atomic(workspace.runs / f"{record.run_id}.json", record.to_dict())
    return record


def stale_attempts(workspace: WorkspacePaths, source: SourceRecord) -> Path:
    path = (
        workspace.sources
        / source.source_id
        / "compatibility"
        / Path(source.known_paths[0]).stem
        / "attempts.json"
    )
    document = AttemptDocument(
        source_fingerprint="sha256:" + "f" * 64,
        source_duration_seconds=1,
        padding_seconds=0,
        attempts=(),
        export_ranges=(),
    )
    write_json_atomic(path, document.to_dict())
    return path


def test_inventory_classifies_candidates_and_protected_paths(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    create_session(
        workspace, "Practice", id_factory=lambda: "session-0123456789abcdef"
    )
    source_root = workspace.sources / source.source_id
    cache = source_root / "attempts" / "a" / "cache" / "track.jsonl"
    review = source_root / "attempts" / "a" / "index.html"
    annotation = source_root / "attempts" / "a" / "annotation.json"
    unknown = source_root / "private.blob"
    temporary = workspace.temporary / "index.html"
    for path in (cache, review, annotation, unknown, temporary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")
    run = failed_run(workspace, source)
    failed_artifact = workspace.runs / run.run_id / "partial.tmp"
    failed_artifact.parent.mkdir()
    failed_artifact.write_text("partial", encoding="utf-8")
    stale = stale_attempts(workspace, source)

    result = inventory_cleanup(workspace)
    by_path = {item.path: item for item in result.items}
    assert by_path[cache].classification == "recomputable-cache"
    assert by_path[review].classification == "generated-review-media"
    assert by_path[temporary].classification == "generated-review-media"
    assert by_path[failed_artifact].classification == "failed-run-artifact"
    assert by_path[stale].classification == "stale-artifact"
    assert by_path[annotation].classification == "user-annotation"
    assert by_path[unknown].classification == "manifest-provenance"
    assert by_path[source_record_path(workspace, source.source_id)].classification == "manifest-provenance"
    assert by_path[workspace.runs / f"{run.run_id}.json"].classification == "manifest-provenance"
    assert by_path[Path(source.known_paths[0])].classification == "source-media"
    assert set(item.classification for item in result.items) == set(CLASSES)
    assert all(item.classification in CLASSES[:4] for item in result.candidates)
    assert all(not item.future_deletable for item in result.protected if not item.excluded or item.classification not in CLASSES[:4])


def test_category_filters_are_union_and_never_promote_protected_files(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    stale = stale_attempts(workspace, source)
    temporary = workspace.temporary / "index.html"
    temporary.parent.mkdir(parents=True)
    temporary.write_text("temp", encoding="utf-8")
    unknown = workspace.sources / source.source_id / "unknown.bin"
    unknown.write_bytes(b"unknown")

    result = inventory_cleanup(workspace, stale=True, temporary=True)
    assert {item.path for item in result.candidates} == {stale, temporary}
    assert unknown in {item.path for item in result.protected}


def test_source_selector_filters_candidates_by_id_or_path(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    first = register(workspace, tmp_path / "one.mov", b"one")
    second = register(workspace, tmp_path / "two.mov", b"two")
    first_cache = workspace.sources / first.source_id / "cache" / "one.jsonl"
    second_cache = workspace.sources / second.source_id / "cache" / "two.jsonl"
    for path in (first_cache, second_cache):
        path.parent.mkdir(parents=True)
        path.write_text("cache", encoding="utf-8")

    by_id = inventory_cleanup(workspace, source=[first.source_id])
    assert first_cache in {item.path for item in by_id.candidates}
    assert second_cache not in {item.path for item in by_id.candidates}
    by_path = inventory_cleanup(workspace, source=[second.known_paths[0]])
    assert second_cache in {item.path for item in by_path.candidates}
    with pytest.raises(CleanupInventoryError, match="did not match"):
        inventory_cleanup(workspace, source=["../../escape"])


def test_symlink_is_protected_and_target_is_not_followed(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.html"
    secret.write_text("secret", encoding="utf-8")
    link = workspace.sources / source.source_id / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    result = inventory_cleanup(workspace)
    item = next(item for item in result.items if item.path == link)
    assert item.classification == "manifest-provenance"
    assert item.excluded
    assert secret not in {entry.path for entry in result.items}


def test_inventory_is_read_only_and_deterministic(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    artifact = workspace.temporary / "z.html"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("page", encoding="utf-8")
    paths = [path for path in workspace.root.rglob("*") if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    first = inventory_cleanup(workspace)
    second = inventory_cleanup(workspace)
    assert first == second
    assert [str(item.path) for item in first.items] == sorted(str(item.path) for item in first.items)
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before
    assert Path(source.known_paths[0]).read_bytes() == b"source"


def test_malformed_manifests_fail_without_repair(tmp_path: Path) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    source = register(workspace, tmp_path / "video.mov")
    manifest = source_record_path(workspace, source.source_id)
    manifest.write_text("{bad", encoding="utf-8")
    with pytest.raises(CleanupInventoryError, match="malformed source manifest"):
        inventory_cleanup(workspace)
    assert manifest.read_text(encoding="utf-8") == "{bad"


def test_cleanup_item_enforces_deletability_invariant(tmp_path: Path) -> None:
    with pytest.raises(CleanupInventoryError, match="invariant"):
        CleanupItem(tmp_path / "x", "source-media", True, "wrong")
    with pytest.raises(CleanupInventoryError, match="invariant"):
        CleanupItem(tmp_path / "x", "recomputable-cache", False, "wrong")
