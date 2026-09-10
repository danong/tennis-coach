"""Tests for the synthetic media fixture generator (M1.3).

Generation tests use the real local FFmpeg and write only into the
``tmp_path`` temporary directory. A test is skipped only when FFmpeg is
not installed; nothing is skipped when FFmpeg is available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from serve_review.media.probe import probe_source
from media_factory import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_FPS,
    FixtureError,
    FixtureSpec,
    LANDSCAPE_SIZE,
    MAX_DIMENSION,
    MAX_DURATION_SECONDS,
    PORTRAIT_SIZE,
    SEGMENT_COLORS,
    build_ffmpeg_args,
    extract_frame_bytes,
    ffmpeg_available,
    filter_graph,
    generate_fixture,
    landscape_spec,
    make_fixture,
    portrait_spec,
    segment_color,
    segment_hue_degrees,
    source_input_spec,
    spec_for_orientation,
)

FFMPEG_MISSING = shutil.which("ffmpeg") is None
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG_MISSING, reason="FFmpeg is required to generate fixtures"
)


def test_presets_are_tiny_portrait_and_landscape() -> None:
    portrait = portrait_spec()
    landscape = landscape_spec()
    assert (portrait.width, portrait.height) == PORTRAIT_SIZE
    assert (landscape.width, landscape.height) == LANDSCAPE_SIZE
    assert portrait.height > portrait.width
    assert landscape.width > landscape.height


def test_defaults_are_bounded() -> None:
    for spec in (portrait_spec(), landscape_spec()):
        assert 0 < spec.duration_seconds <= MAX_DURATION_SECONDS
        assert spec.width <= MAX_DIMENSION
        assert spec.height <= MAX_DIMENSION
        assert spec.width * spec.height <= MAX_DIMENSION * MAX_DIMENSION
    assert DEFAULT_DURATION_SECONDS <= MAX_DURATION_SECONDS
    assert DEFAULT_FPS <= 30


def test_spec_for_orientation_rejects_unknown() -> None:
    with pytest.raises(FixtureError, match="orientation"):
        spec_for_orientation("square")


def test_spec_rejects_out_of_bounds() -> None:
    with pytest.raises(FixtureError, match="[Dd]imension|pixels"):
        FixtureSpec(width=640, height=480)
    with pytest.raises(FixtureError, match="[Dd]imension|pixels"):
        FixtureSpec(width=320, height=321)
    with pytest.raises(FixtureError, match="duration"):
        FixtureSpec(width=64, height=48, duration_seconds=10.0)
    with pytest.raises(FixtureError, match="duration"):
        FixtureSpec(width=64, height=48, duration_seconds=0.0)
    with pytest.raises(FixtureError, match="fps"):
        FixtureSpec(width=64, height=48, fps=120)
    with pytest.raises(FixtureError, match="segment_index"):
        FixtureSpec(width=64, height=48, segment_index=-1)
    with pytest.raises(FixtureError, match="even"):
        FixtureSpec(width=65, height=48)


def test_segment_identity_is_deterministic() -> None:
    assert segment_hue_degrees(0) == 0
    assert segment_hue_degrees(1) == 90
    assert segment_hue_degrees(4) == 0
    assert segment_color(0) == SEGMENT_COLORS[0]
    assert segment_color(len(SEGMENT_COLORS)) == SEGMENT_COLORS[0]
    assert segment_color(0) != segment_color(1)
    with pytest.raises(FixtureError, match="segment_index"):
        segment_hue_degrees(-1)
    with pytest.raises(FixtureError, match="segment_index"):
        segment_color(-2)


def test_source_input_spec_is_deterministic() -> None:
    spec = portrait_spec(segment_index=2)
    first = source_input_spec(spec)
    assert source_input_spec(spec) == first
    assert f"{spec.width}x{spec.height}" in first
    assert "testsrc2" in first
    other = portrait_spec(segment_index=3)
    # Size/rate/duration are shared; per-segment identity lives in the filter.
    assert source_input_spec(other) == first
    assert filter_graph(other) != filter_graph(spec)


def test_filter_graph_distinguishes_segments() -> None:
    graphs = {filter_graph(portrait_spec(i)) for i in range(4)}
    assert len(graphs) == 4
    assert filter_graph(portrait_spec(0)) == filter_graph(portrait_spec(0))
    assert "drawbox" in filter_graph(portrait_spec(0))
    assert "hue" in filter_graph(portrait_spec(0))


def test_build_ffmpeg_args_is_deterministic_array(tmp_path: Path) -> None:
    output = tmp_path / "fixture.mov"
    spec = portrait_spec(segment_index=1)
    first = build_ffmpeg_args(output, spec)
    second = build_ffmpeg_args(output, spec)
    assert first == second
    assert isinstance(first, list)
    assert all(isinstance(part, str) for part in first)
    assert first[0] == "ffmpeg"
    assert "-f" in first and "lavfi" in first
    assert "testsrc2" in " ".join(first)
    assert str(output) in first
    assert "-pix_fmt" in first and "yuv420p" in first
    assert "-c:v" in first and "libx264" in first


def test_build_ffmpeg_args_distinguishes_orientation_and_segment(
    tmp_path: Path,
) -> None:
    portrait = build_ffmpeg_args(tmp_path / "p.mov", portrait_spec(0))
    landscape = build_ffmpeg_args(tmp_path / "l.mov", landscape_spec(0))
    assert portrait != landscape
    first_segment = build_ffmpeg_args(tmp_path / "s0.mov", portrait_spec(0))
    second_segment = build_ffmpeg_args(tmp_path / "s1.mov", portrait_spec(1))
    assert first_segment != second_segment


def test_build_ffmpeg_args_rejects_bad_spec(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="[Ss]pec"):
        build_ffmpeg_args(tmp_path / "o.mov", "not-a-spec")  # type: ignore[arg-type]
    with pytest.raises(FixtureError, match="executable"):
        build_ffmpeg_args(tmp_path / "o.mov", portrait_spec(), ffmpeg="")


def test_generate_uses_argument_array_without_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import media_factory as factory

    captured: dict = {}

    def _fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        Path(args[-1]).write_bytes(b"fake-fixture")
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(factory.subprocess, "run", _fake_run)
    output = tmp_path / "nested" / "clip.mov"
    assert generate_fixture(output, portrait_spec()) == output
    assert output.is_file()
    assert isinstance(captured["args"], list)
    assert captured["args"][0] == "ffmpeg"
    assert "shell" not in captured["kwargs"]


def test_generate_missing_tool_raises_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import media_factory as factory

    monkeypatch.setattr(factory.shutil, "which", lambda _exe: None)
    with pytest.raises(FixtureError, match="not found.*install FFmpeg"):
        generate_fixture(tmp_path / "clip.mov", portrait_spec())


def test_generate_failure_removes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import media_factory as factory

    output = tmp_path / "clip.mov"

    def _fake_fail(args, **kwargs):
        Path(args[-1]).write_bytes(b"partial-bytes")
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr(factory.subprocess, "run", _fake_fail)
    with pytest.raises(FixtureError, match="exit 1.*boom"):
        generate_fixture(output, portrait_spec())
    assert not output.exists()


def test_extract_frame_rejects_missing_video(tmp_path: Path) -> None:
    with pytest.raises(FixtureError, match="does not exist"):
        extract_frame_bytes(tmp_path / "missing.mov", 0.1)


# --- Integration: real FFmpeg + ffprobe in temporary directories ---


@needs_ffmpeg
def test_generate_portrait_and_probe(tmp_path: Path) -> None:
    assert ffmpeg_available()
    output = tmp_path / "portrait.mov"
    spec = portrait_spec(segment_index=0)
    assert generate_fixture(output, spec) == output
    assert output.is_file() and output.stat().st_size > 0
    meta = probe_source(output)
    assert (meta.width, meta.height) == PORTRAIT_SIZE
    assert meta.height > meta.width
    assert meta.duration_seconds == pytest.approx(
        DEFAULT_DURATION_SECONDS, abs=0.15
    )
    assert meta.video_codec == "h264"


@needs_ffmpeg
def test_generate_landscape_and_probe(tmp_path: Path) -> None:
    output = tmp_path / "landscape.mov"
    spec = landscape_spec(segment_index=1)
    generate_fixture(output, spec)
    meta = probe_source(output)
    assert (meta.width, meta.height) == LANDSCAPE_SIZE
    assert meta.width > meta.height
    assert meta.duration_seconds == pytest.approx(
        DEFAULT_DURATION_SECONDS, abs=0.15
    )


@needs_ffmpeg
def test_make_fixture_both_orientations(tmp_path: Path) -> None:
    portrait = make_fixture(tmp_path / "p.mov", "portrait", segment_index=0)
    landscape = make_fixture(tmp_path / "l.mov", "landscape", segment_index=0)
    assert probe_source(portrait).height > probe_source(portrait).width
    assert probe_source(landscape).width > probe_source(landscape).height
    with pytest.raises(FixtureError, match="orientation"):
        make_fixture(tmp_path / "bad.mov", "square")


@needs_ffmpeg
def test_timestamp_identity_frames_differ(tmp_path: Path) -> None:
    output = tmp_path / "timed.mov"
    generate_fixture(output, portrait_spec())
    early = extract_frame_bytes(output, 0.1)
    late = extract_frame_bytes(output, 0.8)
    assert len(early) > 0 and len(early) == len(late)
    assert early != late


@needs_ffmpeg
def test_segment_identity_frames_differ(tmp_path: Path) -> None:
    first = tmp_path / "seg0.mov"
    second = tmp_path / "seg1.mov"
    generate_fixture(first, portrait_spec(segment_index=0))
    generate_fixture(second, portrait_spec(segment_index=1))
    assert first.read_bytes() != second.read_bytes()
    frame_first = extract_frame_bytes(first, 0.5)
    frame_second = extract_frame_bytes(second, 0.5)
    assert frame_first != frame_second
