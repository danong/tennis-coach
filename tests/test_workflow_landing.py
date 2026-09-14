from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import pytest

from serve_review.domain import SourceMetadata
from serve_review.workflow.landing import LandingRenderError, render_landing_page
from serve_review.workflow.records import (
    ArtifactRecord,
    AttemptRecord,
    LandingPage,
    SourceRecord,
    WorkflowRecordError,
    source_id_for_fingerprint,
)


def source_record(path: Path) -> SourceRecord:
    fingerprint = "sha256:" + "a" * 64
    metadata = SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=1,
        width=1,
        height=1,
        frame_rate_num=1,
        video_codec="h264",
    )
    return SourceRecord(
        1,
        source_id_for_fingerprint(fingerprint),
        fingerprint,
        metadata,
        (str(path.resolve()),),
    )


def artifact(
    root: Path,
    name: str,
    kind: str,
    *,
    status: str = "available",
    label: str | None = None,
    remediation: str | None = None,
    stage: str | None = None,
    time_seconds: float | None = None,
    attempt_label: str | None = None,
) -> ArtifactRecord:
    return ArtifactRecord(
        label or name,
        str((root / name).resolve()),
        kind,
        status,
        remediation,
        stage,
        time_seconds,
        attempt_label,
    )


def test_renderer_escapes_text_encodes_urls_and_exposes_all_artifacts(
    tmp_path: Path,
) -> None:
    media = artifact(
        tmp_path,
        "clip #1 ? 50% é.mov",
        "clip",
        label='<clip "one">',
        remediation="run <unsafe> & retry",
    )
    compilation = artifact(tmp_path, "all serves.mp4", "compilation")
    source_json = artifact(
        tmp_path, "source.json", "json", status="stale", remediation="process --force"
    )
    checkpoint = artifact(
        tmp_path,
        "contact image.jpg",
        "checkpoint",
        label="contact <image>",
        stage="racket <contact>",
        time_seconds=1.25,
        attempt_label='Attempt "A"',
    )
    review = artifact(tmp_path, "review page.html", "review")
    diagnostic = artifact(
        tmp_path, "diagnostics.json", "diagnostic", status="failed"
    )
    other = artifact(tmp_path, "extra.txt", "other", status="missing")
    attempt = AttemptRecord(
        "attempt-0123456789abcdef01234567",
        "Attempt <A>",
        "failed",
        "rerun & inspect",
        (review, checkpoint, diagnostic, other),
    )
    page = LandingPage(
        source_record(tmp_path / 'source <bad> "video".mov'),
        (source_json, compilation, media),
        (attempt,),
    )
    destination = tmp_path / "output" / "index.html"

    assert render_landing_page(page, destination) == destination
    html = destination.read_text(encoding="utf-8")

    assert '<clip "one">' not in html
    assert "&lt;clip &quot;one&quot;&gt;" in html
    assert "run &lt;unsafe&gt; &amp; retry" in html
    assert "Attempt &lt;A&gt;" in html
    assert "racket &lt;contact&gt;" in html
    assert "source &lt;bad&gt; &quot;" in html
    assert "clip%20%231%20%3F%2050%25%20%C3%A9.mov" in html
    for supplied in (
        media,
        compilation,
        source_json,
        checkpoint,
        review,
        diagnostic,
        other,
    ):
        relative = os.path.relpath(supplied.path, destination.parent)
        assert quote(relative, safe="/") in html
    assert html.count("<video controls") == 2
    assert "Direct media link" in html
    assert "cannot play every source codec" in html
    assert '<img src="../contact%20image.jpg"' in html
    assert "checkpoint 1.25 seconds" in html
    assert "stale" in html and "failed" in html and "missing" in html
    assert "process --force" in html and "rerun &amp; inspect" in html
    assert "<script" not in html and "javascript:" not in html


def test_output_is_deterministic_independent_of_input_order(tmp_path: Path) -> None:
    first = artifact(tmp_path, "z.json", "json")
    second = artifact(tmp_path, "a.json", "json")
    attempt_a = AttemptRecord(
        "attempt-000000000000000000000001", "A", "available", None, ()
    )
    attempt_b = AttemptRecord(
        "attempt-000000000000000000000002", "B", "available", None, ()
    )
    source = source_record(tmp_path / "video.mov")
    one = LandingPage(source, (first, second), (attempt_b, attempt_a))
    two = LandingPage(source, (second, first), (attempt_a, attempt_b))
    first_output = tmp_path / "one" / "index.html"
    second_output = tmp_path / "two" / "index.html"
    render_landing_page(one, first_output)
    render_landing_page(two, second_output)
    first_text = first_output.read_text(encoding="utf-8").replace("../../", "../")
    second_text = second_output.read_text(encoding="utf-8").replace("../../", "../")
    assert first_text == second_text
    assert first_output.read_bytes().endswith(b"\n")


def test_landing_records_have_strict_roundtrip_codec(tmp_path: Path) -> None:
    item = artifact(
        tmp_path,
        "checkpoint.jpg",
        "checkpoint",
        stage="maximum knee bend",
        time_seconds=0.5,
        attempt_label="Attempt 1",
    )
    attempt = AttemptRecord(
        "attempt-0123456789abcdef01234567", "Attempt 1", "available", None, (item,)
    )
    page = LandingPage(source_record(tmp_path / "video.mov"), (), (attempt,))
    assert LandingPage.from_json(page.to_json()) == page
    with pytest.raises(WorkflowRecordError):
        LandingPage.from_dict({**page.to_dict(), "extra": True})
    with pytest.raises(WorkflowRecordError):
        LandingPage.from_json('{"source":{},"source":{}}')
    with pytest.raises(WorkflowRecordError):
        ArtifactRecord("x", "relative", "other", "available")
    with pytest.raises(WorkflowRecordError):
        ArtifactRecord(
            "x", str((tmp_path / "x").resolve()), "checkpoint", "available"
        )
    with pytest.raises(WorkflowRecordError):
        ArtifactRecord(
            "x",
            str((tmp_path / "x").resolve()),
            "json",
            "available",
            stage="wrong field",
        )


def test_renderer_does_not_read_or_require_artifacts(tmp_path: Path) -> None:
    missing = artifact(tmp_path, "does-not-exist.mp4", "clip", status="missing")
    page = LandingPage(source_record(tmp_path / "also-missing.mov"), (missing,), ())
    output = tmp_path / "index.html"
    render_landing_page(page, output)
    assert output.is_file()
    assert not Path(missing.path).exists()


def test_actual_write_failure_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serve_review.workflow.landing as module

    real_named_temporary = module.tempfile.NamedTemporaryFile

    class FailingHandle:
        def __init__(self, handle):
            self.handle = handle
            self.name = handle.name

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, text):
            self.handle.write(text)
            raise OSError("write failed")

        def flush(self):
            self.handle.flush()

        def fileno(self):
            return self.handle.fileno()

    monkeypatch.setattr(
        module.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: FailingHandle(
            real_named_temporary(*args, **kwargs)
        ),
    )
    output = tmp_path / "index.html"
    with pytest.raises(LandingRenderError, match="write failed"):
        render_landing_page(
            LandingPage(source_record(tmp_path / "video.mov"), (), ()), output
        )
    assert not list(tmp_path.glob("index.html.tmp-*"))


def test_replace_failure_removes_unique_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serve_review.workflow.landing as module

    observed: list[Path] = []

    def fail_replace(source, destination):
        observed.append(Path(source))
        raise OSError("replace failed")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    output = tmp_path / "index.html"
    with pytest.raises(LandingRenderError, match="replace failed"):
        render_landing_page(
            LandingPage(source_record(tmp_path / "video.mov"), (), ()), output
        )
    assert len(observed) == 1
    assert observed[0].parent == output.parent
    assert observed[0].name.startswith("index.html.tmp-")
    assert not observed[0].exists()
