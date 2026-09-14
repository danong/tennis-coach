from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from serve_review.domain import Attempt, AttemptDocument, MediaRange, SourceMetadata
from serve_review.workflow.processing import process_registered_source
from serve_review.workflow.records import (
    AttemptOutcome,
    SourceRecord,
    WorkflowRecordError,
    WorkflowRunRecord,
    attempt_id_for,
    source_id_for_fingerprint,
)
from serve_review.workflow.workspace import WorkspacePaths, read_json


def source_fixture(tmp_path: Path, *, aliases: int = 1) -> tuple[SourceRecord, Path]:
    fingerprint = "sha256:" + "a" * 64
    paths = []
    for index in range(aliases):
        path = tmp_path / f"source-{index}.mov"
        path.write_bytes(b"source")
        paths.append(str(path.resolve()))
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
        tuple(sorted(paths)),
    )
    return record, Path(sorted(paths)[0])


def detected_document(source: SourceRecord, ranges: tuple[MediaRange, ...]):
    attempts = tuple(
        Attempt(
            attempt_id=f"serve-{index:03d}",
            detected_range=value,
            effective_range=value,
        )
        for index, value in enumerate(ranges, 1)
    )
    return AttemptDocument(
        source_fingerprint=source.source_fingerprint,
        source_duration_seconds=source.metadata.duration_seconds,
        padding_seconds=0,
        method_version="compat-v1",
        attempts=attempts,
        export_ranges=ranges,
    )


def analyzer_result(
    source: SourceRecord, media_range: MediaRange, output_dir: Path
):
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "checkpoints_path": output_dir / "checkpoints.json",
        "diagnostics_path": output_dir / "diagnostics.json",
        "index_html": review_dir / "index.html",
        "review_json": review_dir / "review.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        source_metadata=source.metadata,
        requested_range=media_range,
        attempt_range=media_range,
        review_dir=review_dir,
        **paths,
    )


def test_normal_mode_calls_cut_and_scopes_each_attempt(tmp_path: Path) -> None:
    source, video = source_fixture(tmp_path, aliases=2)
    workspace = WorkspacePaths(tmp_path / "workspace")
    ranges = (MediaRange(1, 2), MediaRange(4, 5))
    cut_calls = []
    analyze_calls = []

    def cut(path, **kwargs):
        cut_calls.append((path, kwargs))
        return SimpleNamespace(
            source_metadata=source.metadata,
            attempts_document=detected_document(source, ranges),
        )

    def analyze(path, **kwargs):
        analyze_calls.append((path, kwargs))
        media_range = MediaRange(kwargs["start_seconds"], kwargs["end_seconds"])
        return analyzer_result(source, media_range, Path(kwargs["output_dir"]))

    record = process_registered_source(
        source,
        workspace,
        "normal",
        run_id_factory=lambda: "run-0123456789abcdef",
        producer_method="detector:v1",
        producer_config="config:abc",
        run_cut_fn=cut,
        run_analyze_fn=analyze,
    )

    assert cut_calls == [
        (
            video,
            {
                "output_dir": workspace.sources
                / source.source_id
                / "compatibility",
                "mode": "both",
                "overwrite": False,
            },
        )
    ]
    assert record.overall_status == "complete"
    assert len(record.attempts) == 2
    assert [outcome.attempt_id for outcome in record.attempts] == [
        attempt_id_for(
            source.source_id,
            "detector:v1",
            "config:abc",
            value.start_seconds,
            value.end_seconds,
        )
        for value in ranges
    ]
    output_dirs = [Path(call[1]["output_dir"]) for call in analyze_calls]
    assert len(set(output_dirs)) == 2
    for call, outcome in zip(analyze_calls, record.attempts):
        kwargs = call[1]
        assert kwargs["overwrite"] is False
        assert kwargs["cache_path"] == (
            Path(kwargs["output_dir"]) / "cache" / "kinematic-track-v1.jsonl"
        )
        assert outcome.artifacts
    run_path = workspace.runs / f"{record.run_id}.json"
    assert WorkflowRunRecord.from_dict(read_json(run_path)) == record


def test_explicit_modes_bypass_cut_and_use_exact_ranges(tmp_path: Path) -> None:
    source, _ = source_fixture(tmp_path)
    workspace = WorkspacePaths(tmp_path / "workspace")
    calls = []

    def forbidden_cut(*args, **kwargs):
        raise AssertionError("cut must not run")

    def analyze(path, **kwargs):
        calls.append(kwargs)
        value = MediaRange(kwargs["start_seconds"], kwargs["end_seconds"])
        return analyzer_result(source, value, Path(kwargs["output_dir"]))

    single = process_registered_source(
        source,
        workspace,
        "single-attempt",
        run_id_factory=lambda: "run-0000000000000001",
        producer_method="manual:v1",
        producer_config="whole",
        run_cut_fn=forbidden_cut,
        run_analyze_fn=analyze,
    )
    explicit_range = MediaRange(2.25, 3.75)
    explicit = process_registered_source(
        source,
        workspace,
        "explicit-range",
        explicit_range=explicit_range,
        run_id_factory=lambda: "run-0000000000000002",
        producer_method="manual:v1",
        producer_config="range",
        run_cut_fn=forbidden_cut,
        run_analyze_fn=analyze,
    )
    assert (single.attempts[0].start_seconds, single.attempts[0].end_seconds) == (
        0,
        10,
    )
    assert (
        explicit.attempts[0].start_seconds,
        explicit.attempts[0].end_seconds,
    ) == (2.25, 3.75)
    assert all(call["overwrite"] is False for call in calls)


def test_empty_detection_is_honest(tmp_path: Path) -> None:
    source, _ = source_fixture(tmp_path)
    analyzed = False

    def analyze(*args, **kwargs):
        nonlocal analyzed
        analyzed = True

    result = process_registered_source(
        source,
        WorkspacePaths(tmp_path / "workspace"),
        "normal",
        run_id_factory=lambda: "run-0000000000000001",
        producer_method="detector:v1",
        producer_config="config",
        run_cut_fn=lambda *_args, **_kwargs: SimpleNamespace(
            source_metadata=source.metadata,
            attempts_document=detected_document(source, ()),
        ),
        run_analyze_fn=analyze,
    )
    assert result.detection_status == result.overall_status == "empty"
    assert result.attempts == ()
    assert analyzed is False


def test_detection_failure_writes_failed_provenance(tmp_path: Path) -> None:
    source, _ = source_fixture(tmp_path)
    workspace = WorkspacePaths(tmp_path / "workspace")

    def fail(*args, **kwargs):
        raise RuntimeError("detection exploded")

    result = process_registered_source(
        source,
        workspace,
        "normal",
        run_id_factory=lambda: "run-0000000000000001",
        producer_method="detector:v1",
        producer_config="config",
        run_cut_fn=fail,
    )
    assert result.detection_status == result.overall_status == "failed"
    assert "detection exploded" in result.detection_error
    assert result.attempts == ()
    assert (workspace.runs / f"{result.run_id}.json").is_file()


@pytest.mark.parametrize(
    ("fail_indices", "expected"),
    [({1}, "partial"), ({0, 1}, "failed")],
)
def test_analysis_failures_are_isolated(
    tmp_path: Path, fail_indices: set[int], expected: str
) -> None:
    source, _ = source_fixture(tmp_path)
    ranges = (MediaRange(1, 2), MediaRange(3, 4))
    index = 0

    def analyze(path, **kwargs):
        nonlocal index
        current = index
        index += 1
        if current in fail_indices:
            raise RuntimeError(f"attempt {current} failed")
        value = MediaRange(kwargs["start_seconds"], kwargs["end_seconds"])
        return analyzer_result(source, value, Path(kwargs["output_dir"]))

    result = process_registered_source(
        source,
        WorkspacePaths(tmp_path / "workspace"),
        "normal",
        run_id_factory=lambda: "run-0000000000000001",
        producer_method="detector:v1",
        producer_config="config",
        run_cut_fn=lambda *_args, **_kwargs: SimpleNamespace(
            source_metadata=source.metadata,
            attempts_document=detected_document(source, ranges),
        ),
        run_analyze_fn=analyze,
    )
    assert result.overall_status == expected
    assert [outcome.status for outcome in result.attempts] == [
        "failed" if value in fail_indices else "complete" for value in range(2)
    ]


def test_duplicate_detected_range_records_collision_failure(tmp_path: Path) -> None:
    source, _ = source_fixture(tmp_path)
    repeated = MediaRange(1, 2)
    analyzed = False

    def analyze(*args, **kwargs):
        nonlocal analyzed
        analyzed = True

    result = process_registered_source(
        source,
        WorkspacePaths(tmp_path / "workspace"),
        "normal",
        run_id_factory=lambda: "run-0000000000000001",
        producer_method="detector:v1",
        producer_config="config",
        run_cut_fn=lambda *_args, **_kwargs: SimpleNamespace(
            source_metadata=source.metadata,
            attempts_document=SimpleNamespace(
                source_fingerprint=source.source_fingerprint,
                attempts=(
                    SimpleNamespace(detected_range=repeated),
                    SimpleNamespace(detected_range=repeated),
                ),
            ),
        ),
        run_analyze_fn=analyze,
    )
    assert result.overall_status == result.detection_status == "failed"
    assert "stable attempt ID collision" in result.detection_error
    assert result.attempts == ()
    assert analyzed is False


def test_malformed_analyze_result_becomes_failed_outcome(tmp_path: Path) -> None:
    source, _ = source_fixture(tmp_path)
    result = process_registered_source(
        source,
        WorkspacePaths(tmp_path / "workspace"),
        "explicit-range",
        explicit_range=MediaRange(1, 2),
        run_id_factory=lambda: "run-0000000000000001",
        producer_method="manual:v1",
        producer_config="config",
        run_analyze_fn=lambda *_args, **_kwargs: SimpleNamespace(
            source_metadata=source.metadata,
            requested_range=MediaRange(1, 2),
            attempt_range=MediaRange(1, 2),
        ),
    )
    assert result.overall_status == "failed"
    assert "missing required artifact" in result.attempts[0].error


def test_unavailable_source_and_run_collision_are_safe(tmp_path: Path) -> None:
    source, video = source_fixture(tmp_path)
    workspace = WorkspacePaths(tmp_path / "workspace")
    video.unlink()
    with pytest.raises(FileNotFoundError, match="source unavailable"):
        process_registered_source(
            source,
            workspace,
            "single-attempt",
            run_id_factory=lambda: "run-0000000000000001",
            producer_method="manual:v1",
            producer_config="config",
        )

    video.write_bytes(b"source")
    collision = workspace.runs / "run-0000000000000001.json"
    collision.parent.mkdir(parents=True)
    collision.write_text("keep", encoding="utf-8")
    with pytest.raises(WorkflowRecordError, match="collision"):
        process_registered_source(
            source,
            workspace,
            "single-attempt",
            run_id_factory=lambda: "run-0000000000000001",
            producer_method="manual:v1",
            producer_config="config",
        )
    assert collision.read_text(encoding="utf-8") == "keep"


def test_run_record_codec_rejects_inconsistent_statuses(tmp_path: Path) -> None:
    source, _ = source_fixture(tmp_path)
    artifact = str((tmp_path / "artifact.json").resolve())
    outcome = AttemptOutcome(
        attempt_id_for(source.source_id, "method", "config", 1, 2),
        1,
        2,
        "complete",
        None,
        (artifact,),
    )
    record = WorkflowRunRecord(
        1,
        "run-0000000000000001",
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
    assert WorkflowRunRecord.from_json(record.to_json()) == record
    with pytest.raises(WorkflowRecordError):
        WorkflowRunRecord.from_dict({**record.to_dict(), "schema_version": True})
    with pytest.raises(WorkflowRecordError, match="overall status"):
        WorkflowRunRecord.from_dict({**record.to_dict(), "overall_status": "partial"})
    with pytest.raises(WorkflowRecordError):
        WorkflowRunRecord.from_json('{"schema_version":1,"schema_version":1}')
