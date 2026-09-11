"""Tests for the compilation exporter and multi-mode export (M1.5).

All media is generated synthetically in temporary directories with
``tests/media_factory.py``; no private footage or network access is used.
Real FFmpeg/ffprobe run the integration paths; unit paths check command
shape and failure cleanup without media.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from serve_review.domain import ExportPlan, ExportPlanError, MediaRange, SourceMetadata
from serve_review.media import export as export_module
from serve_review.media.export import (
    CLIPS_SUBDIR,
    COMPILATION_FILENAME,
    DEFAULT_VIDEO_BITRATE,
    DEFAULT_VIDEO_BITRATE_BPS,
    DEFAULT_VIDEO_ENCODER,
    EXPORT_MODES,
    SOFTWARE_VIDEO_ENCODER,
    ExportCancelled,
    ExportCollisionError,
    ExportError,
    build_compilation_ffmpeg_args,
    export_compilation,
    export_outputs,
)
from serve_review.media.probe import probe_source
from media_factory import (
    extract_frame_bytes,
    ffmpeg_available,
    generate_fixture,
    landscape_spec,
    portrait_spec,
)

NEEDS_TOOLS = pytest.mark.skipif(
    not ffmpeg_available() or shutil.which("ffprobe") is None,
    reason="FFmpeg and ffprobe are required for export integration tests",
)


def _make_source(
    tmp_path: Path,
    *,
    orientation: str = "landscape",
    duration_seconds: float = 2.0,
    fps: int = 30,
    name: str = "source.mov",
) -> tuple[Path, SourceMetadata]:
    spec = (
        landscape_spec(duration_seconds=duration_seconds, fps=fps)
        if orientation == "landscape"
        else portrait_spec(duration_seconds=duration_seconds, fps=fps)
    )
    video = generate_fixture(tmp_path / name, spec)
    return video, probe_source(video)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_tolerance(meta: SourceMetadata) -> float:
    return 1.0 / meta.frames_per_second + 0.02


def _tmp_leftovers(directory: Path) -> list[Path]:
    found: list[Path] = []
    if directory.is_dir():
        for child in directory.rglob("*"):
            if ".tmp-" in child.name:
                found.append(child)
    return sorted(found)


def _mean_abs_diff(first: bytes, second: bytes) -> float:
    count = min(len(first), len(second))
    assert count > 0
    total = sum(abs(a - b) for a, b in zip(first[:count], second[:count]))
    return total / count


# --- Constants and command construction (no media required) ---


def test_export_modes_and_filenames() -> None:
    assert EXPORT_MODES == ("compilation", "clips", "both")
    assert COMPILATION_FILENAME == "serves.mov"
    assert CLIPS_SUBDIR == "clips"


def test_build_compilation_args_concatenates_in_order_without_gaps() -> None:
    args = build_compilation_ffmpeg_args(
        "src.mov",
        [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)],
        "out.mov",
    )
    assert isinstance(args, list)
    assert all(isinstance(part, str) for part in args)
    assert args[0] == "ffmpeg"
    assert "-filter_complex" in args
    script = args[args.index("-filter_complex") + 1]
    assert "trim=start=0.200000:end=0.700000" in script
    assert "trim=start=1.200000:end=1.800000" in script
    # First segment appears before the second: source order preserved.
    assert script.index("0.200000") < script.index("1.200000")
    assert "setpts=PTS-STARTPTS" in script
    assert "concat=n=2:v=1:a=0" in script
    # No seek flags, no artificial gaps/padding, no stream copy.
    assert "-ss" not in args
    assert "copy" not in args
    assert "adelay" not in script and "aevalsrc" not in script
    assert args[args.index("-c:v") + 1] == DEFAULT_VIDEO_ENCODER
    assert DEFAULT_VIDEO_ENCODER == "h264_videotoolbox"
    assert args[args.index("-b:v") + 1] == DEFAULT_VIDEO_BITRATE
    assert "-preset" not in args and "-crf" not in args
    assert "-an" in args
    assert args[-1] == "out.mov"


def test_default_compilation_bitrate_is_review_band() -> None:
    assert DEFAULT_VIDEO_BITRATE_BPS == 6_000_000
    from serve_review.media import export as export_mod

    assert (
        export_mod.REVIEW_VIDEO_BITRATE_MIN_BPS
        <= DEFAULT_VIDEO_BITRATE_BPS
        <= export_mod.REVIEW_VIDEO_BITRATE_MAX_BPS
    )
    assert 4_000_000 <= export_mod._parse_video_bitrate_to_bps(
        DEFAULT_VIDEO_BITRATE
    ) <= 8_000_000


def test_compilation_libx264_override_keeps_software_options() -> None:
    args = build_compilation_ffmpeg_args(
        "a.mov", [MediaRange(0.0, 0.5)], "t.mov", video_encoder="libx264"
    )
    assert args[args.index("-c:v") + 1] == "libx264"
    assert "-b:v" not in args
    assert args[args.index("-preset") + 1] == "veryfast"
    assert args[args.index("-crf") + 1] == "18"
    assert SOFTWARE_VIDEO_ENCODER == "libx264"


def test_compilation_args_reject_bad_bitrate() -> None:
    with pytest.raises(ExportError, match="video_bitrate"):
        build_compilation_ffmpeg_args(
            "a.mov", [MediaRange(0.0, 0.5)], "t.mov", video_bitrate="nope"
        )


def test_build_compilation_args_is_deterministic() -> None:
    ranges = [MediaRange(0.0, 0.4), MediaRange(1.0, 1.4)]
    first = build_compilation_ffmpeg_args("a.mov", ranges, "t.mov")
    second = build_compilation_ffmpeg_args("a.mov", ranges, "t.mov")
    assert first == second
    other = build_compilation_ffmpeg_args("a.mov", [MediaRange(0.0, 0.5)], "t.mov")
    assert other != first


def test_build_compilation_args_audio_variant() -> None:
    args = build_compilation_ffmpeg_args(
        "a.mov",
        [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)],
        "t.mov",
        include_audio=True,
    )
    script = args[args.index("-filter_complex") + 1]
    assert "atrim=start=0.000000:end=0.500000" in script
    assert "asetpts=PTS-STARTPTS" in script
    assert "concat=n=2:v=0:a=1" in script
    assert "-an" not in args
    assert args[args.index("-c:a") + 1] == "aac"


def test_build_compilation_args_rejects_stream_copy_and_blank_encoder() -> None:
    with pytest.raises(ExportError, match="stream copy"):
        build_compilation_ffmpeg_args(
            "a.mov", [MediaRange(0.0, 0.5)], "t.mov", video_encoder="copy"
        )
    with pytest.raises(ExportError, match="video_encoder"):
        build_compilation_ffmpeg_args(
            "a.mov", [MediaRange(0.0, 0.5)], "t.mov", video_encoder="  "
        )


def test_build_compilation_args_rejects_empty_and_invalid_ranges() -> None:
    with pytest.raises(ExportError, match="at least one range"):
        build_compilation_ffmpeg_args("a.mov", [], "t.mov")
    with pytest.raises(ExportError, match="MediaRange"):
        build_compilation_ffmpeg_args("a.mov", ["x"], "t.mov")  # type: ignore[list-item]


def test_export_compilation_rejects_non_plan_and_missing_source(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="ExportPlan"):
        export_compilation(tmp_path / "missing.mov", "nope", tmp_path / "out.mov")  # type: ignore[arg-type]
    meta = SourceMetadata(
        fingerprint="sha256:" + "0" * 64,
        duration_seconds=1.0,
        width=64,
        height=48,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
    )
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    with pytest.raises(ExportError, match="does not exist"):
        export_compilation(tmp_path / "missing.mov", plan, tmp_path / "out.mov")


def test_export_outputs_rejects_bad_mode(tmp_path: Path) -> None:
    meta = SourceMetadata(
        fingerprint="sha256:" + "0" * 64,
        duration_seconds=1.0,
        width=64,
        height=48,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
    )
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    with pytest.raises(ExportError, match="output mode"):
        export_outputs(tmp_path / "missing.mov", plan, tmp_path / "out", mode="dvd")  # type: ignore[arg-type]


# --- Integration: real FFmpeg on generated fixtures ---


@NEEDS_TOOLS
def test_compilation_duration_is_sum_of_ranges_with_no_gaps(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)]
    plan = ExportPlan.for_source(meta, ranges)
    out = export_compilation(video, plan, tmp_path / "serves.mov", source=meta)
    assert out.is_file()
    probed = probe_source(out)
    expected = sum(r.duration_seconds for r in ranges)
    assert probed.duration_seconds == pytest.approx(expected, abs=_sample_tolerance(meta))
    assert plan.total_duration_seconds == pytest.approx(expected)


@NEEDS_TOOLS
def test_compilation_preserves_segment_identity_and_order(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    # Two non-adjacent ranges from distinct source times.
    plan = ExportPlan.for_source(meta, [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)])
    out = export_compilation(video, plan, tmp_path / "serves.mov", source=meta)
    src_early = extract_frame_bytes(video, 0.3)
    src_late = extract_frame_bytes(video, 1.5)
    # Sanity: source times are visibly distinct.
    assert _mean_abs_diff(src_early, src_late) > 1.0
    comp_first = extract_frame_bytes(out, 0.1)
    comp_second = extract_frame_bytes(out, 0.8)
    assert _mean_abs_diff(comp_first, comp_second) > 1.0
    # Nearest-neighbor identity: each compilation position matches its source range.
    assert _mean_abs_diff(comp_first, src_early) < _mean_abs_diff(comp_first, src_late)
    assert _mean_abs_diff(comp_second, src_late) < _mean_abs_diff(comp_second, src_early)


@NEEDS_TOOLS
def test_compilation_preserves_dimensions(tmp_path: Path) -> None:
    for orientation in ("landscape", "portrait"):
        video, meta = _make_source(
            tmp_path, orientation=orientation, name=f"{orientation}.mov"
        )
        plan = ExportPlan.for_source(meta, [MediaRange(0.1, 0.6), MediaRange(1.0, 1.5)])
        out = export_compilation(
            video, plan, tmp_path / f"serves-{orientation}.mov", source=meta
        )
        probed = probe_source(out)
        assert (probed.width, probed.height) == (meta.width, meta.height)
        assert probed.rotation_degrees == meta.rotation_degrees


@NEEDS_TOOLS
def test_compilation_adjacent_ranges_concatenate(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.0, 0.5), MediaRange(0.5, 1.0)]
    plan = ExportPlan.for_source(meta, ranges)  # adjacency is allowed
    out = export_compilation(video, plan, tmp_path / "serves.mov", source=meta)
    assert probe_source(out).duration_seconds == pytest.approx(
        1.0, abs=_sample_tolerance(meta)
    )


@NEEDS_TOOLS
def test_export_outputs_all_modes(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)]
    plan = ExportPlan.for_source(meta, ranges)

    only_comp = export_outputs(
        video, plan, tmp_path / "only-comp", mode="compilation", source=meta
    )
    assert isinstance(only_comp["compilation"], Path)
    assert only_comp["compilation"].is_file()
    assert only_comp["clips"] == []
    assert not (tmp_path / "only-comp" / CLIPS_SUBDIR).exists()
    assert probe_source(only_comp["compilation"]).duration_seconds == pytest.approx(
        plan.total_duration_seconds, abs=_sample_tolerance(meta)
    )

    only_clips = export_outputs(
        video, plan, tmp_path / "only-clips", mode="clips", source=meta
    )
    assert only_clips["compilation"] is None
    assert [p.name for p in only_clips["clips"]] == ["serve-001.mov", "serve-002.mov"]
    assert all(p.is_file() for p in only_clips["clips"])
    assert not (tmp_path / "only-clips" / COMPILATION_FILENAME).exists()

    both = export_outputs(video, plan, tmp_path / "both", mode="both", source=meta)
    assert isinstance(both["compilation"], Path) and both["compilation"].is_file()
    assert [p.name for p in both["clips"]] == ["serve-001.mov", "serve-002.mov"]
    assert both["compilation"].parent == tmp_path / "both"
    assert both["clips"][0].parent == tmp_path / "both" / CLIPS_SUBDIR


@NEEDS_TOOLS
def test_source_file_is_not_modified_by_compilation(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    before_hash = _sha256(video)
    before_mtime = video.stat().st_mtime_ns
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    export_outputs(video, plan, tmp_path / "out", mode="both", source=meta)
    assert _sha256(video) == before_hash
    assert video.stat().st_mtime_ns == before_mtime


@NEEDS_TOOLS
def test_no_tmp_leftovers_after_compilation_success(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "out"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    export_outputs(video, plan, out_dir, mode="both", source=meta)
    assert _tmp_leftovers(out_dir) == []
    assert _tmp_leftovers(tmp_path) == []


@NEEDS_TOOLS
def test_invalid_and_empty_plans_rejected_without_ffmpeg(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path, duration_seconds=2.0)
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(meta, [])
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(meta, [MediaRange(0.0, 0.6), MediaRange(0.4, 1.0)])
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(meta, [MediaRange(1.5, 5.0)])
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(meta, [MediaRange(0.8, 1.0), MediaRange(0.1, 0.3)])


@NEEDS_TOOLS
def test_fingerprint_mismatch_rejected_for_compilation(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    foreign = ExportPlan(
        source_fingerprint="sha256:" + "f" * 64,
        source_duration_seconds=meta.duration_seconds,
        ranges=(MediaRange(0.0, 0.5),),
    )
    with pytest.raises(ExportError, match="fingerprint"):
        export_compilation(video, foreign, tmp_path / "serves.mov", source=meta)
    with pytest.raises(ExportError, match="fingerprint"):
        export_outputs(video, foreign, tmp_path / "out", mode="both", source=meta)


@NEEDS_TOOLS
def test_collision_without_overwrite_raises_before_any_work(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = out_dir / COMPILATION_FILENAME
    sentinel.write_bytes(b"sentinel-do-not-touch")
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    with pytest.raises(ExportCollisionError, match="collision"):
        export_outputs(video, plan, out_dir, mode="both", source=meta)
    assert sentinel.read_bytes() == b"sentinel-do-not-touch"
    assert not (out_dir / CLIPS_SUBDIR / "serve-001.mov").exists()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_overwrite_replaces_compilation_explicitly(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / COMPILATION_FILENAME).write_bytes(b"stale")
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    out = export_compilation(video, plan, out_dir / COMPILATION_FILENAME, source=meta, overwrite=True)
    assert out.read_bytes() != b"stale"
    assert probe_source(out).duration_seconds == pytest.approx(
        0.5, abs=_sample_tolerance(meta)
    )


@NEEDS_TOOLS
def test_no_partial_final_file_when_compilation_ffmpeg_fails(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "out"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    real_run = subprocess.run

    def _flaky(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr(export_module.subprocess, "run", _flaky)
    with pytest.raises(ExportError, match="ffmpeg failed.*compilation"):
        export_compilation(video, plan, out_dir / COMPILATION_FILENAME, source=meta)
    assert not (out_dir / COMPILATION_FILENAME).exists()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_no_partial_files_when_both_mode_ffmpeg_fails(
    tmp_path: Path, monkeypatch
) -> None:
    # Both-mode encodes clips first, then joins serves.mov losslessly.
    # The first FFmpeg call (clip) succeeds; the join copy and the
    # fallback re-encode both fail, so the finished clip remains while
    # no partial compilation or temp files are left behind.
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "out"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    real_run = subprocess.run
    calls = {"count": 0}

    def _flaky(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 1:
            return real_run(args, **kwargs)
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr(export_module.subprocess, "run", _flaky)
    with pytest.raises(ExportError, match="ffmpeg failed"):
        export_outputs(video, plan, out_dir, mode="both", source=meta)
    assert not (out_dir / COMPILATION_FILENAME).exists()
    assert (out_dir / CLIPS_SUBDIR / "serve-001.mov").is_file()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_cancellation_before_compilation_cleans_up(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    with pytest.raises(ExportCancelled, match="cancelled"):
        export_compilation(
            video,
            plan,
            tmp_path / "out" / COMPILATION_FILENAME,
            source=meta,
            is_cancelled=lambda: True,
        )
    assert not (tmp_path / "out" / COMPILATION_FILENAME).exists()
    assert _tmp_leftovers(tmp_path) == []


@NEEDS_TOOLS
def test_keyboard_interrupt_cleans_compilation_temp(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])

    def _interrupted(args, **kwargs):
        raise KeyboardInterrupt("simulated cancel")

    monkeypatch.setattr(export_module.subprocess, "run", _interrupted)
    with pytest.raises(KeyboardInterrupt):
        export_compilation(video, plan, tmp_path / "out" / COMPILATION_FILENAME, source=meta)
    assert not (tmp_path / "out" / COMPILATION_FILENAME).exists()
    assert _tmp_leftovers(tmp_path) == []


@NEEDS_TOOLS
def test_explicit_encoder_selection_reaches_compilation_ffmpeg(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path, duration_seconds=1.0)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.4)])
    captured: dict = {}
    real_run = subprocess.run

    def _spy(args, **kwargs):
        captured["args"] = list(args)
        return real_run(args, **kwargs)

    monkeypatch.setattr(export_module.subprocess, "run", _spy)
    export_compilation(
        video, plan, tmp_path / "serves.mov", source=meta, video_encoder="libx264"
    )
    args = captured["args"]
    assert args[args.index("-c:v") + 1] == "libx264"
    assert "copy" not in args
    assert "-filter_complex" in args
    script = args[args.index("-filter_complex") + 1]
    assert "concat=" in script


def _hardware_available() -> bool:
    try:
        done = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "encoder=h264_videotoolbox"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError:
        return False
    return done.returncode == 0


NEEDS_VIDEOTOOLBOX = pytest.mark.skipif(
    not ffmpeg_available() or not _hardware_available(),
    reason="h264_videotoolbox encoder is required for hardware export tests",
)


def _probe_video_codec(path: Path) -> str:
    done = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of",
            "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@NEEDS_VIDEOTOOLBOX
def test_hardware_compilation_is_valid_portrait_and_landscape(
    tmp_path: Path,
) -> None:
    for orientation in ("landscape", "portrait"):
        video, meta = _make_source(
            tmp_path, orientation=orientation, name=f"hw-comp-{orientation}.mov"
        )
        ranges = [MediaRange(0.1, 0.6), MediaRange(1.0, 1.5)]
        plan = ExportPlan.for_source(meta, ranges)
        out = export_compilation(
            video, plan, tmp_path / f"hw-serves-{orientation}.mov", source=meta
        )
        assert out.is_file() and out.stat().st_size > 0
        probed = probe_source(out)
        assert _probe_video_codec(out) == "h264"
        assert (probed.width, probed.height) == (meta.width, meta.height)
        assert probed.rotation_degrees == meta.rotation_degrees
        expected = sum(r.duration_seconds for r in ranges)
        assert probed.duration_seconds == pytest.approx(
            expected, abs=_sample_tolerance(meta)
        )


@NEEDS_VIDEOTOOLBOX
def test_hardware_compilation_bitrate_target_within_band(
    tmp_path: Path,
) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    out = export_compilation(video, plan, tmp_path / "hw-rate.mov", source=meta)
    from serve_review.media import export as export_mod

    target_bps = export_mod._parse_video_bitrate_to_bps(DEFAULT_VIDEO_BITRATE)
    assert 4_000_000 <= target_bps <= 8_000_000
    measured_bps = 8 * out.stat().st_size / probe_source(out).duration_seconds
    assert measured_bps > 10_000
    assert measured_bps < 12_000_000


@NEEDS_VIDEOTOOLBOX
def test_hardware_compilation_trim_accuracy_matches_software(
    tmp_path: Path,
) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)]
    plan = ExportPlan.for_source(meta, ranges)
    tolerance = _sample_tolerance(meta)
    expected = sum(r.duration_seconds for r in ranges)
    hw_out = export_compilation(video, plan, tmp_path / "hw-serves.mov", source=meta)
    sw_out = export_compilation(
        video, plan, tmp_path / "sw-serves.mov", source=meta,
        video_encoder="libx264",
    )
    hw_dur = probe_source(hw_out).duration_seconds
    sw_dur = probe_source(sw_out).duration_seconds
    assert hw_dur == pytest.approx(expected, abs=tolerance)
    assert sw_dur == pytest.approx(expected, abs=tolerance)
    assert hw_dur == pytest.approx(sw_dur, abs=tolerance)


@NEEDS_TOOLS
def test_hardware_compilation_failure_names_encoder_and_cleans_up(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "out"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    real_run = subprocess.run

    def _fail(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr=(
                "[h264_videotoolbox @ 0xabc] Error encoding frame: "
                "hardware unavailable (-12908)"
            ),
        )

    monkeypatch.setattr(export_module.subprocess, "run", _fail)
    with pytest.raises(ExportError) as excinfo:
        export_compilation(video, plan, out_dir / COMPILATION_FILENAME, source=meta)
    message = str(excinfo.value)
    assert "h264_videotoolbox" in message
    assert "hardware unavailable" in message
    assert not (out_dir / COMPILATION_FILENAME).exists()
    assert _tmp_leftovers(out_dir) == []


def test_build_compilation_args_places_hwaccel_videotoolbox_before_input() -> None:
    args = build_compilation_ffmpeg_args(
        "a.mov", [MediaRange(0.0, 0.5)], "t.mov"
    )
    assert "-hwaccel" in args
    assert args[args.index("-hwaccel") + 1] == "videotoolbox"
    assert args.index("-hwaccel") < args.index("-i")
    for index, part in enumerate(args):
        if part == "-i":
            assert args[index - 2] == "-hwaccel"
            assert args[index - 1] == "videotoolbox"
    # Deterministic construction still holds with the decode flags present.
    again = build_compilation_ffmpeg_args(
        "a.mov", [MediaRange(0.0, 0.5)], "t.mov"
    )
    assert again == args


@NEEDS_TOOLS
def test_export_compilation_passes_hwaccel_decode_to_ffmpeg(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path, duration_seconds=1.0)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.4)])
    captured: dict = {}
    real_run = subprocess.run

    def _spy(args, **kwargs):
        if not str(args[0]).endswith("ffprobe"):
            captured["args"] = list(args)
        return real_run(args, **kwargs)

    monkeypatch.setattr(export_module.subprocess, "run", _spy)
    export_compilation(
        video, plan, tmp_path / "serves.mov", source=meta
    )
    args = captured["args"]
    assert "-hwaccel" in args
    assert args[args.index("-hwaccel") + 1] == "videotoolbox"
    assert args.index("-hwaccel") < args.index("-i")


@NEEDS_TOOLS
def test_hwaccel_decode_compilation_failure_names_videotoolbox(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "out"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    real_run = subprocess.run
    seen: list[list[str]] = []

    def _fail(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        seen.append(list(args))
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr=(
                "[vist#0:0/h264 @ 0xabc] No device available for decoder: "
                "device type videotoolbox needed for codec h264. "
                "Hardware device setup failed for decoder."
            ),
        )

    monkeypatch.setattr(export_module.subprocess, "run", _fail)
    with pytest.raises(ExportError) as excinfo:
        export_compilation(video, plan, out_dir / COMPILATION_FILENAME, source=meta)
    message = str(excinfo.value)
    assert "VideoToolbox" in message
    assert "-hwaccel videotoolbox" in message
    assert "No software fallback" in message
    assert seen and "-hwaccel" in seen[0]
    assert seen[0][seen[0].index("-hwaccel") + 1] == "videotoolbox"
    assert not (out_dir / COMPILATION_FILENAME).exists()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_VIDEOTOOLBOX
def test_hwaccel_decode_compilation_matches_software_decode(
    tmp_path: Path,
) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)]
    plan = ExportPlan.for_source(meta, ranges)
    tolerance = _sample_tolerance(meta)
    expected = sum(r.duration_seconds for r in ranges)
    hw_out = export_compilation(video, plan, tmp_path / "hwdec-serves.mov", source=meta)
    hw_args = build_compilation_ffmpeg_args(
        video, ranges, tmp_path / "probe-mov"
    )
    assert hw_args[hw_args.index("-hwaccel") + 1] == "videotoolbox"
    sw_args = [part for part in hw_args if part not in ("-hwaccel", "videotoolbox")]
    sw_args[-1] = str(tmp_path / "swdec-serves.mov")
    assert "-hwaccel" not in sw_args
    done = subprocess.run(sw_args, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    hw_probed = probe_source(hw_out)
    sw_probed = probe_source(tmp_path / "swdec-serves.mov")
    assert hw_probed.duration_seconds == pytest.approx(expected, abs=tolerance)
    assert sw_probed.duration_seconds == pytest.approx(expected, abs=tolerance)
    assert hw_probed.duration_seconds == pytest.approx(
        sw_probed.duration_seconds, abs=tolerance
    )
    assert (hw_probed.width, hw_probed.height) == (
        sw_probed.width,
        sw_probed.height,
    ) == (meta.width, meta.height)


@NEEDS_VIDEOTOOLBOX
def test_hwaccel_decode_compilation_valid_portrait_and_landscape(
    tmp_path: Path,
) -> None:
    for orientation in ("landscape", "portrait"):
        video, meta = _make_source(
            tmp_path, orientation=orientation, name=f"hwdec-comp-{orientation}.mov"
        )
        ranges = [MediaRange(0.1, 0.6), MediaRange(1.0, 1.5)]
        plan = ExportPlan.for_source(meta, ranges)
        out = export_compilation(
            video, plan, tmp_path / f"hwdec-serves-{orientation}.mov", source=meta
        )
        assert out.is_file() and out.stat().st_size > 0
        probed = probe_source(out)
        assert _probe_video_codec(out) == "h264"
        assert (probed.width, probed.height) == (meta.width, meta.height)
        assert probed.rotation_degrees == meta.rotation_degrees
        expected = sum(r.duration_seconds for r in ranges)
        assert probed.duration_seconds == pytest.approx(
            expected, abs=_sample_tolerance(meta)
        )


# --- Lossless stream-copy concat join for both-mode (same-run clips) ---


def test_build_concat_copy_args_is_stream_copy_without_reencode() -> None:
    args = export_module.build_concat_copy_args("list.txt", "out.mov")
    assert args[0] == "ffmpeg"
    assert "-f" in args and args[args.index("-f") + 1] == "concat"
    assert args[args.index("-safe") + 1] == "0"
    assert args[args.index("-i") + 1] == "list.txt"
    assert args[args.index("-c") + 1] == "copy"
    assert "-c:v" not in args and "-c:a" not in args
    assert "-filter_complex" not in args and "-vf" not in args
    assert "-preset" not in args and "-crf" not in args and "-b:v" not in args
    assert "-hwaccel" not in args
    assert args[-1] == "out.mov"
    again = export_module.build_concat_copy_args("list.txt", "out.mov")
    assert again == args


def test_concat_join_readiness_rejects_missing_and_mismatched(tmp_path: Path) -> None:
    missing = export_module.concat_join_readiness([tmp_path / "nope.mov"])
    assert isinstance(missing, str) and "missing" in missing.lower()
    assert "re-encode" in missing
    empty = tmp_path / "empty.mov"
    empty.write_bytes(b"")
    reason_empty = export_module.concat_join_readiness([empty])
    assert isinstance(reason_empty, str) and "empty" in reason_empty.lower()
    assert export_module.concat_join_readiness([]) is not None
    # Mismatched codec/resolution inputs fail closed with an explicit reason.
    land, _ = _make_source(tmp_path, orientation="landscape", name="land.mov")
    port, _ = _make_source(tmp_path, orientation="portrait", name="port.mov")
    reason = export_module.concat_join_readiness([land, port])
    assert isinstance(reason, str)
    assert "differing" in reason.lower() or "differ" in reason.lower()
    assert "re-encode" in reason


@NEEDS_TOOLS
def test_both_mode_join_matches_clips_losslessly_and_reencode(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)]
    plan = ExportPlan.for_source(meta, ranges)
    both = export_outputs(video, plan, tmp_path / "both", mode="both", source=meta)
    clips = list(both["clips"])
    comp = both["compilation"]
    assert isinstance(comp, Path) and comp.is_file()
    assert [p.name for p in clips] == ["serve-001.mov", "serve-002.mov"]
    # Duration equals the sum of ranges (no dead-time gaps).
    expected = sum(r.duration_seconds for r in ranges)
    tolerance = _sample_tolerance(meta)
    assert probe_source(comp).duration_seconds == pytest.approx(expected, abs=tolerance)
    clip_total = sum(probe_source(c).duration_seconds for c in clips)
    assert probe_source(comp).duration_seconds == pytest.approx(clip_total, abs=tolerance)
    # Lossless: first compilation frames decode identically to clip 1 frames.
    comp_frame = extract_frame_bytes(comp, 0.1)
    clip_frame = extract_frame_bytes(clips[0], 0.1)
    assert _mean_abs_diff(comp_frame, clip_frame) < 1.0
    # Byte-comparable to the standalone re-encode path in frames/duration.
    re_out = export_compilation(video, plan, tmp_path / "re-serves.mov", source=meta)
    assert probe_source(re_out).duration_seconds == pytest.approx(expected, abs=tolerance)
    re_frame = extract_frame_bytes(re_out, 0.1)
    assert _mean_abs_diff(comp_frame, re_frame) < 12.0
    assert probe_source(comp).duration_seconds == pytest.approx(
        probe_source(re_out).duration_seconds, abs=tolerance
    )


@NEEDS_TOOLS
def test_both_mode_join_valid_portrait_and_landscape(tmp_path: Path) -> None:
    for orientation in ("landscape", "portrait"):
        video, meta = _make_source(
            tmp_path, orientation=orientation, name=f"join-{orientation}.mov"
        )
        ranges = [MediaRange(0.1, 0.6), MediaRange(1.0, 1.5)]
        plan = ExportPlan.for_source(meta, ranges)
        both = export_outputs(
            video, plan, tmp_path / f"join-{orientation}-out",
            mode="both", source=meta,
        )
        comp = both["compilation"]
        assert isinstance(comp, Path) and comp.is_file() and comp.stat().st_size > 0
        probed = probe_source(comp)
        assert (probed.width, probed.height) == (meta.width, meta.height)
        assert probed.rotation_degrees == meta.rotation_degrees
        expected = sum(r.duration_seconds for r in ranges)
        assert probed.duration_seconds == pytest.approx(
            expected, abs=_sample_tolerance(meta)
        )
        # Order preserved: each half matches its source range, not the other.
        src_early = extract_frame_bytes(video, 0.2)
        src_late = extract_frame_bytes(video, 1.2)
        assert _mean_abs_diff(src_early, src_late) > 1.0
        first = extract_frame_bytes(comp, 0.1)
        second = extract_frame_bytes(comp, 0.7)
        assert _mean_abs_diff(first, second) > 1.0
        assert _mean_abs_diff(first, src_early) < _mean_abs_diff(first, src_late)
        assert _mean_abs_diff(second, src_late) < _mean_abs_diff(second, src_early)


@NEEDS_TOOLS
def test_both_mode_falls_back_to_reencode_on_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    monkeypatch.setattr(
        export_module,
        "concat_join_readiness",
        lambda clips, **kwargs: "simulated mismatch; falling back to filter re-encode.",
    )
    out_dir = tmp_path / "out"
    result = export_outputs(video, plan, out_dir, mode="both", source=meta)
    comp = result["compilation"]
    assert isinstance(comp, Path) and comp.is_file()
    assert [p.name for p in result["clips"]] == ["serve-001.mov", "serve-002.mov"]
    expected = plan.total_duration_seconds
    assert probe_source(comp).duration_seconds == pytest.approx(
        expected, abs=_sample_tolerance(meta)
    )
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_both_mode_copy_failure_falls_back_to_reencode(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    real_run = subprocess.run

    def _flaky(args, **kwargs):
        if "-f" in args and "concat" in args:
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout="", stderr="concat boom"
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(export_module.subprocess, "run", _flaky)
    out_dir = tmp_path / "out"
    result = export_outputs(video, plan, out_dir, mode="both", source=meta)
    assert isinstance(result["compilation"], Path)
    assert result["compilation"].is_file()
    assert probe_source(result["compilation"]).duration_seconds == pytest.approx(
        plan.total_duration_seconds, abs=_sample_tolerance(meta)
    )
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_both_mode_join_cancellation_cleans_temp_but_keeps_clips(
    tmp_path: Path,
) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    calls = {"count": 0}

    def _cancel():
        calls["count"] += 1
        # export_clips polls once at start + once per clip (3 polls);
        # cancel at the join stage that follows.
        return calls["count"] > 3

    out_dir = tmp_path / "out"
    with pytest.raises(ExportCancelled, match="cancelled"):
        export_outputs(
            video, plan, out_dir, mode="both", source=meta, is_cancelled=_cancel
        )
    assert not (out_dir / COMPILATION_FILENAME).exists()
    assert (out_dir / CLIPS_SUBDIR / "serve-001.mov").is_file()
    assert _tmp_leftovers(out_dir) == []
    assert _tmp_leftovers(tmp_path) == []


@NEEDS_TOOLS
def test_both_mode_join_keyboard_interrupt_cleans_temp(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    real_run = subprocess.run

    def _interrupted(args, **kwargs):
        if "-f" in args and "concat" in args:
            raise KeyboardInterrupt("simulated cancel during join")
        return real_run(args, **kwargs)

    monkeypatch.setattr(export_module.subprocess, "run", _interrupted)
    with pytest.raises(KeyboardInterrupt):
        export_outputs(video, plan, tmp_path / "out", mode="both", source=meta)
    assert not (tmp_path / "out" / COMPILATION_FILENAME).exists()
    assert _tmp_leftovers(tmp_path) == []
