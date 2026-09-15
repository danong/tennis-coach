import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from serve_review.domain import Attempt, AttemptDocument, MediaRange, SourceMetadata
from serve_review.process import FingerprintMismatch, ProcessError, discover, process


def metadata(fingerprint: str = "sha256:test") -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=10,
        width=1,
        height=1,
        frame_rate_num=1,
        frame_rate_den=1,
        video_codec="test",
    )


def attempts(fingerprint: str = "sha256:test", count: int = 1) -> AttemptDocument:
    values = tuple(
        Attempt(
            attempt_id=f"serve-{index:03d}",
            detected_range=MediaRange(index, index + 0.5),
            effective_range=MediaRange(index - 0.5, index + 1),
        )
        for index in range(1, count + 1)
    )
    return AttemptDocument(
        source_fingerprint=fingerprint,
        source_duration_seconds=10,
        padding_seconds=1,
        attempts=values,
        export_ranges=(MediaRange(0.5, count + 1),) if values else (),
    )


def write_cut(video: Path, document: AttemptDocument) -> None:
    root = video.parent / "metadata" / video.stem
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.json").write_text(metadata(document.source_fingerprint).to_json())
    (root / "attempts.json").write_text(document.to_json())
    (root / "run.json").write_text(json.dumps({"status": "ok"}))
    if document.attempts:
        export = video.parent / "exports" / video.stem
        export.mkdir(parents=True, exist_ok=True)
        (export / "serves.mov").write_bytes(b"movie")


def write_analysis(destination: Path) -> None:
    review = destination / "review-serve-3d"
    review.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoints.json").write_text("{}")
    (destination / "serve-3d-diagnostics.json").write_text("{}")
    (review / "review.json").write_text(json.dumps({"entries": []}))
    (review / "index.html").write_text("<html>review</html>")


def test_discovery_is_immediate_sorted_and_rejects_duplicate_stems(tmp_path: Path) -> None:
    (tmp_path / "z.MP4").write_bytes(b"z")
    (tmp_path / "A.mov").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.mov").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [path.name for path in discover(tmp_path)] == ["A.mov", "z.MP4"]

    (tmp_path / "A.mp4").write_bytes(b"duplicate")
    with pytest.raises(ProcessError, match="duplicate"):
        discover(tmp_path)


def test_empty_and_unsupported_targets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProcessError, match="no MOV"):
        discover(tmp_path)
    text = tmp_path / "notes.txt"
    text.write_text("x")
    with pytest.raises(ProcessError, match="unsupported"):
        discover(text)


def test_complete_empty_cut_is_skipped_without_producers(tmp_path: Path) -> None:
    video = tmp_path / "serve.mov"
    video.write_bytes(b"video")
    write_cut(video, attempts(count=0))

    def unexpected(*args, **kwargs):
        raise AssertionError("producer must not run")

    result = process(video, probe_fn=lambda path: metadata(), cut_fn=unexpected, analyze_fn=unexpected)
    assert [(action.attempt, action.action) for action in result.actions] == [(None, "skip")]
    assert result.failures == ()


def test_dry_run_reports_clear_without_mutation_or_producers(tmp_path: Path) -> None:
    video = tmp_path / "serve.mov"
    video.write_bytes(b"video")
    root = tmp_path / "metadata" / "serve"
    root.mkdir(parents=True)
    marker = root / "partial"
    marker.write_text("keep")

    def unexpected(*args, **kwargs):
        raise AssertionError("producer must not run")

    result = process(
        video,
        probe_fn=lambda path: metadata(),
        cut_fn=unexpected,
        analyze_fn=unexpected,
        dry_run=True,
        force=True,
    )
    assert result.actions[0].action == "clear"
    assert marker.read_text() == "keep"


def test_fingerprint_mismatch_requires_force_before_mutation(tmp_path: Path) -> None:
    video = tmp_path / "serve.mov"
    video.write_bytes(b"video")
    write_cut(video, attempts("sha256:old", count=0))
    with pytest.raises(FingerprintMismatch, match="--force"):
        process(video, probe_fn=lambda path: metadata("sha256:new"))
    assert (tmp_path / "metadata" / "serve" / "attempts.json").is_file()


def test_incomplete_cut_clears_source_tree_and_uses_unpadded_range(tmp_path: Path) -> None:
    video = tmp_path / "serve.mov"
    video.write_bytes(b"video")
    old = tmp_path / "metadata" / "serve"
    (old / "attempts" / "serve-001").mkdir(parents=True)
    (old / "attempts" / "serve-001" / "old").write_text("old")
    export = tmp_path / "exports" / "serve"
    export.mkdir(parents=True)
    (export / "old").write_text("old")
    document = attempts()
    calls: list[tuple] = []

    def cut(video_path: Path, **kwargs):
        calls.append(("cut", kwargs))
        write_cut(video_path, document)
        return SimpleNamespace(attempts_document=document)

    def analyze(video_path: Path, **kwargs):
        calls.append(("analyze", kwargs))
        write_analysis(kwargs["output_dir"])

    result = process(video, probe_fn=lambda path: metadata(), cut_fn=cut, analyze_fn=analyze)
    assert result.failures == ()
    assert not (old / "attempts" / "serve-001" / "old").exists()
    cut_call, analyze_call = calls
    assert cut_call[1]["padding_seconds"] == 1
    assert cut_call[1]["mode"] == "compilation"
    assert analyze_call[1]["start_seconds"] == 1
    assert analyze_call[1]["end_seconds"] == 1.5
    assert analyze_call[1]["output_dir"] == old / "attempts" / "serve-001"


def test_incomplete_attempt_preserves_complete_sibling(tmp_path: Path) -> None:
    video = tmp_path / "serve.mov"
    video.write_bytes(b"video")
    document = attempts(count=2)
    write_cut(video, document)
    first = tmp_path / "metadata" / "serve" / "attempts" / "serve-001"
    second = first.parent / "serve-002"
    write_analysis(first)
    write_analysis(second)
    (second / "serve-3d-diagnostics.json").write_text("{")
    (second / "partial").write_text("old")
    marker = first / "keep"
    marker.write_text("keep")

    def analyze(video_path: Path, **kwargs):
        write_analysis(kwargs["output_dir"])

    result = process(
        video,
        probe_fn=lambda path: metadata(),
        cut_fn=lambda *args, **kwargs: pytest.fail("cut must be skipped"),
        analyze_fn=analyze,
    )
    assert marker.read_text() == "keep"
    assert not (second / "partial").exists()
    assert _actions(result, "serve-001") == ["skip"]
    assert _actions(result, "serve-002") == ["clear"]


def test_failures_keep_stage_and_do_not_stop_later_sources(tmp_path: Path) -> None:
    first = tmp_path / "a.mov"
    second = tmp_path / "b.mov"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    document = attempts()

    class Failed(Exception):
        stage = "decode"

    def cut(video: Path, **kwargs):
        if video == first:
            raise Failed("no candidate")
        write_cut(video, document)
        return SimpleNamespace(attempts_document=document)

    analyzed: list[str] = []

    def analyze(video: Path, **kwargs):
        analyzed.append(video.name)
        write_analysis(kwargs["output_dir"])

    result = process(tmp_path, probe_fn=lambda path: metadata(), cut_fn=cut, analyze_fn=analyze)
    assert analyzed == ["b.mov"]
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert (failure.filename, failure.step, failure.error) == ("a.mov", "decode", "no candidate")


def _actions(result, attempt: str) -> list[str]:
    return [action.action for action in result.actions if action.attempt == attempt]
