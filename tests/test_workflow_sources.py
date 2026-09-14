from __future__ import annotations

from pathlib import Path

import pytest

from serve_review.domain import SourceMetadata
from serve_review.workflow import write_json_atomic
from serve_review.workflow.records import SourceRecord, source_id_for_fingerprint
from serve_review.workflow.sources import (
    SOURCE_RECORD_FILENAME,
    SourceRegistryError,
    load_source,
    register_source,
    source_record_path,
)
from serve_review.workflow.workspace import WorkspacePaths


def metadata(fingerprint: str, *, duration: float = 1.0) -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=duration,
        width=1920,
        height=1080,
        frame_rate_num=30000,
        frame_rate_den=1001,
        video_codec="h264",
        rotation_degrees=90,
        time_base_num=1,
        time_base_den=90000,
    )


def test_register_source_stores_reference_and_is_idempotent(tmp_path: Path) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"tiny source bytes")
    video.chmod(0o640)
    before_bytes = video.read_bytes()
    before_mode = video.stat().st_mode
    fingerprint = "sha256:" + "b" * 64
    probed = metadata(fingerprint)
    calls: list[Path] = []

    def probe(path: Path) -> SourceMetadata:
        calls.append(path)
        return probed

    workspace = WorkspacePaths(tmp_path / "workspace")
    first = register_source(video, workspace, probe_fn=probe)
    second = register_source(video, workspace, probe_fn=probe)

    assert first == second == load_source(workspace, first.source_id)
    assert calls == [video.resolve(), video.resolve()]
    assert first.source_fingerprint == fingerprint
    assert first.metadata == probed
    assert first.known_paths == (str(video.resolve()),)
    assert source_record_path(workspace, first.source_id) == (
        workspace.sources.resolve() / first.source_id / SOURCE_RECORD_FILENAME
    )
    assert video.read_bytes() == before_bytes
    assert video.stat().st_mode == before_mode
    assert set(workspace.sources.rglob("*")) == {
        workspace.sources / first.source_id,
        workspace.sources / first.source_id / SOURCE_RECORD_FILENAME,
    }


def test_reregister_same_bytes_at_another_path_adds_sorted_known_path(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "z.mov"
    second_path = tmp_path / "a.mov"
    first_path.write_bytes(b"same")
    second_path.write_bytes(b"same")
    fingerprint = "sha256:" + "c" * 64
    probed = metadata(fingerprint)
    workspace = WorkspacePaths(tmp_path / "workspace")

    register_source(first_path, workspace, probe_fn=lambda _: probed)
    record = register_source(second_path, workspace, probe_fn=lambda _: probed)
    again = register_source(first_path, workspace, probe_fn=lambda _: probed)

    expected = tuple(sorted((str(first_path.resolve()), str(second_path.resolve()))))
    assert record.known_paths == expected
    assert again == record
    assert load_source(workspace, record.source_id) == record
    assert first_path.read_bytes() == second_path.read_bytes() == b"same"


def test_optional_fingerprint_is_called_once_and_compared_case_insensitively(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    fingerprint = "sha256:" + "d" * 64
    calls: list[Path] = []

    def fingerprint_fn(path: Path) -> str:
        calls.append(path)
        return fingerprint.upper().replace("SHA256", "sha256")

    record = register_source(
        video,
        WorkspacePaths(tmp_path / "workspace"),
        probe_fn=lambda _: metadata(fingerprint),
        fingerprint_fn=fingerprint_fn,
    )
    assert calls == [video.resolve()]
    assert record.source_fingerprint == fingerprint


def test_optional_fingerprint_mismatch_does_not_create_record(tmp_path: Path) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    workspace = WorkspacePaths(tmp_path / "workspace")
    with pytest.raises(SourceRegistryError, match="fingerprint mismatch"):
        register_source(
            video,
            workspace,
            probe_fn=lambda _: metadata("sha256:" + "e" * 64),
            fingerprint_fn=lambda _: "sha256:" + "f" * 64,
        )
    assert not workspace.sources.exists()


def test_short_id_collision_is_rejected_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serve_review.workflow.sources as sources_module

    video = tmp_path / "new.mov"
    video.write_bytes(b"new")
    first_fingerprint = "sha256:" + "1" * 64
    second_fingerprint = "sha256:" + "2" * 64
    source_id = source_id_for_fingerprint(first_fingerprint)
    workspace = WorkspacePaths(tmp_path / "workspace")
    existing = SourceRecord(
        1,
        source_id,
        first_fingerprint,
        metadata(first_fingerprint),
        (str(tmp_path / "old.mov"),),
    )
    destination = source_record_path(workspace, source_id)
    write_json_atomic(destination, existing.to_dict())
    before = destination.read_bytes()
    monkeypatch.setattr(
        sources_module, "source_id_for_fingerprint", lambda _: source_id
    )

    with pytest.raises(SourceRegistryError, match="collision"):
        register_source(
            video, workspace, probe_fn=lambda _: metadata(second_fingerprint)
        )
    assert destination.read_bytes() == before


def test_existing_metadata_mismatch_is_rejected_without_overwrite(
    tmp_path: Path,
) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"same")
    fingerprint = "sha256:" + "2" * 64
    workspace = WorkspacePaths(tmp_path / "workspace")
    original = register_source(
        video, workspace, probe_fn=lambda _: metadata(fingerprint, duration=1)
    )
    destination = source_record_path(workspace, original.source_id)
    before = destination.read_bytes()

    with pytest.raises(SourceRegistryError, match="metadata mismatch"):
        register_source(
            video, workspace, probe_fn=lambda _: metadata(fingerprint, duration=2)
        )
    assert destination.read_bytes() == before


def test_existing_malformed_record_is_rejected_without_overwrite(tmp_path: Path) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    fingerprint = "sha256:" + "3" * 64
    workspace = WorkspacePaths(tmp_path / "workspace")
    destination = source_record_path(
        workspace, source_id_for_fingerprint(fingerprint)
    )
    destination.parent.mkdir(parents=True)
    destination.write_text("{bad", encoding="utf-8")

    with pytest.raises(SourceRegistryError, match="could not load"):
        register_source(video, workspace, probe_fn=lambda _: metadata(fingerprint))
    assert destination.read_text(encoding="utf-8") == "{bad"


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_registration_rejects_missing_or_non_file_source(
    tmp_path: Path, kind: str
) -> None:
    source = tmp_path / kind
    if kind == "directory":
        source.mkdir()
    called = False

    def probe(_: Path) -> SourceMetadata:
        nonlocal called
        called = True
        return metadata("sha256:" + "4" * 64)

    with pytest.raises(SourceRegistryError, match="does not exist|regular file"):
        register_source(source, WorkspacePaths(tmp_path / "workspace"), probe_fn=probe)
    assert called is False


def test_registration_wraps_bad_probe_results_and_failures(tmp_path: Path) -> None:
    video = tmp_path / "video.mov"
    video.write_bytes(b"source")
    workspace = WorkspacePaths(tmp_path / "workspace")

    with pytest.raises(SourceRegistryError, match="did not return SourceMetadata"):
        register_source(video, workspace, probe_fn=lambda _: object())  # type: ignore[arg-type]

    def fail_probe(_: Path) -> SourceMetadata:
        raise OSError("probe failed")

    with pytest.raises(SourceRegistryError, match="probe failed"):
        register_source(video, workspace, probe_fn=fail_probe)


def test_source_record_path_rejects_invalid_ids_and_symlink_escape(
    tmp_path: Path,
) -> None:
    workspace = WorkspacePaths(tmp_path / "workspace")
    for source_id in ("", "../outside", "source-ABCDEF0123456789", "source-short"):
        with pytest.raises(SourceRegistryError, match="source_id"):
            source_record_path(workspace, source_id)

    workspace.sources.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    source_id = "source-0123456789abcdef"
    try:
        (workspace.sources / source_id).symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(SourceRegistryError, match="escapes"):
        source_record_path(workspace, source_id)


def test_load_source_wraps_missing_record(tmp_path: Path) -> None:
    with pytest.raises(SourceRegistryError, match="could not load"):
        load_source(
            WorkspacePaths(tmp_path / "workspace"),
            "source-0123456789abcdef",
        )
