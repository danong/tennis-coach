"""Tests for the individual clip exporter (M1.4).

All media is generated synthetically in temporary directories with
``tests/media_factory.py``; no private footage or network access is used.
Real FFmpeg/ffprobe runs the integration paths below; a few unit paths
fake the subprocess layer to prove command shape and failure cleanup.
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
    DEFAULT_VIDEO_BITRATE,
    DEFAULT_VIDEO_BITRATE_BPS,
    DEFAULT_VIDEO_ENCODER,
    SOFTWARE_VIDEO_ENCODER,
    ExportCancelled,
    ExportCollisionError,
    ExportError,
    build_ffmpeg_args,
    clip_filename,
    export_clips,
)
from serve_review.media.probe import probe_source
from media_factory import (
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
    return sorted(directory.glob("*.tmp-*")) + sorted(directory.glob("*.tmp-*.mov"))


# --- Naming and command construction (no media required) ---


def test_clip_filename_is_deterministic_and_collision_safe() -> None:
    assert clip_filename(1) == "serve-001.mov"
    assert clip_filename(2) == "serve-002.mov"
    assert clip_filename(12) == "serve-012.mov"
    assert len({clip_filename(i) for i in range(1, 50)}) == 49


def test_clip_filename_rejects_invalid_index() -> None:
    for bad in (0, -1, True, "1", 1.0, None):
        with pytest.raises(ExportError, match="clip index"):
            clip_filename(bad)  # type: ignore[arg-type]


def test_build_ffmpeg_args_uses_exact_filter_trim_and_explicit_encoder() -> None:
    args = build_ffmpeg_args(
        "/tmp/src.mov", 0.5, 1.25, "/tmp/out.mov", video_encoder="libx264"
    )
    assert isinstance(args, list)
    assert all(isinstance(part, str) for part in args)
    assert args[0] == "ffmpeg"
    assert "-c:v" in args
    assert args[args.index("-c:v") + 1] == "libx264"
    assert "copy" not in args
    assert "-ss" not in args
    vf = args[args.index("-vf") + 1]
    assert "trim=" in vf and "setpts=PTS-STARTPTS" in vf
    assert "start=0.500000" in vf and "end=1.250000" in vf
    assert args[-1] == "/tmp/out.mov"


def test_build_ffmpeg_args_is_deterministic() -> None:
    first = build_ffmpeg_args("a.mov", 0.1, 0.4, "t.mov")
    second = build_ffmpeg_args("a.mov", 0.1, 0.4, "t.mov")
    assert first == second
    other = build_ffmpeg_args("a.mov", 0.1, 0.5, "t.mov")
    assert other != first


def test_build_ffmpeg_args_rejects_stream_copy_encoder() -> None:
    with pytest.raises(ExportError, match="stream copy"):
        build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov", video_encoder="copy")


def test_build_ffmpeg_args_rejects_blank_encoder() -> None:
    with pytest.raises(ExportError, match="video_encoder"):
        build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov", video_encoder="  ")


def test_build_ffmpeg_args_rejects_invalid_range() -> None:
    with pytest.raises(ExportError, match="trim range"):
        build_ffmpeg_args("a.mov", 0.5, 0.5, "t.mov")
    with pytest.raises(ExportError, match="trim range"):
        build_ffmpeg_args("a.mov", 0.9, 0.4, "t.mov")


def test_build_ffmpeg_args_audio_filter_only_when_requested() -> None:
    video_only = build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov", include_audio=False)
    assert "-an" in video_only
    assert "-af" not in video_only
    with_audio = build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov", include_audio=True)
    assert "-an" not in with_audio
    af = with_audio[with_audio.index("-af") + 1]
    assert "atrim=" in af and "asetpts=PTS-STARTPTS" in af


def test_export_rejects_non_plan_and_bad_options(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="ExportPlan"):
        export_clips(tmp_path / "missing.mov", "not-a-plan", tmp_path)  # type: ignore[arg-type]


def test_export_rejects_missing_source(tmp_path: Path) -> None:
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
        export_clips(tmp_path / "missing.mov", plan, tmp_path / "out")


def test_export_rejects_missing_ffmpeg_tool(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "src.mov"
    video.write_bytes(b"bytes")
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
    monkeypatch.setattr(export_module.shutil, "which", lambda _exe: None)
    with pytest.raises(ExportError, match="not found.*FFmpeg"):
        export_clips(
            video, plan, tmp_path / "out", source=meta, ffmpeg="ffmpeg-missing"
        )


# --- Integration: real FFmpeg on generated fixtures ---


@NEEDS_TOOLS
def test_exports_clip_count_and_durations_within_one_sample(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)]
    plan = ExportPlan.for_source(meta, ranges)
    clips = export_clips(video, plan, tmp_path / "clips", source=meta)
    assert [path.name for path in clips] == ["serve-001.mov", "serve-002.mov"]
    assert all(path.is_file() for path in clips)
    tolerance = _sample_tolerance(meta)
    for clip, expected in zip(clips, ranges):
        probed = probe_source(clip)
        assert probed.duration_seconds == pytest.approx(
            expected.duration_seconds, abs=tolerance
        )


@NEEDS_TOOLS
def test_preserves_dimensions_and_orientation(tmp_path: Path) -> None:
    for orientation in ("landscape", "portrait"):
        video, meta = _make_source(
            tmp_path, orientation=orientation, name=f"{orientation}.mov"
        )
        plan = ExportPlan.for_source(meta, [MediaRange(0.1, 0.6)])
        (clip,) = export_clips(
            video, plan, tmp_path / f"clips-{orientation}", source=meta
        )
        probed = probe_source(clip)
        assert (probed.width, probed.height) == (meta.width, meta.height)
        assert probed.rotation_degrees == meta.rotation_degrees


@NEEDS_TOOLS
def test_clip_ordering_matches_plan(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.0, 0.3), MediaRange(0.5, 1.2), MediaRange(1.5, 1.9)]
    plan = ExportPlan.for_source(meta, ranges)
    clips = export_clips(video, plan, tmp_path / "clips", source=meta)
    assert clips == sorted(clips)
    tolerance = _sample_tolerance(meta)
    for clip, expected in zip(clips, ranges):
        assert probe_source(clip).duration_seconds == pytest.approx(
            expected.duration_seconds, abs=tolerance
        )


@NEEDS_TOOLS
def test_probes_source_when_metadata_not_given(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.2, 0.9)])
    clips = export_clips(video, plan, tmp_path / "clips")
    assert len(clips) == 1 and clips[0].is_file()
    assert probe_source(clips[0]).duration_seconds == pytest.approx(
        0.7, abs=_sample_tolerance(meta)
    )


@NEEDS_TOOLS
def test_source_file_is_not_modified(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    before_hash = _sha256(video)
    before_mtime = video.stat().st_mtime_ns
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    export_clips(video, plan, tmp_path / "clips", source=meta)
    assert _sha256(video) == before_hash
    assert video.stat().st_mtime_ns == before_mtime


@NEEDS_TOOLS
def test_no_tmp_leftovers_after_success(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "clips"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    export_clips(video, plan, out_dir, source=meta)
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_fingerprint_mismatch_rejected(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    foreign = ExportPlan(
        source_fingerprint="sha256:" + "f" * 64,
        source_duration_seconds=meta.duration_seconds,
        ranges=(MediaRange(0.0, 0.5),),
    )
    assert foreign.source_fingerprint != meta.fingerprint
    with pytest.raises(ExportError, match="fingerprint"):
        export_clips(video, foreign, tmp_path / "clips", source=meta)


@NEEDS_TOOLS
def test_invalid_and_empty_plans_rejected(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path, duration_seconds=2.0)
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(meta, [])
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(
            meta, [MediaRange(0.0, 0.6), MediaRange(0.4, 1.0)]
        )
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(meta, [MediaRange(1.5, 5.0)])
    with pytest.raises(ExportPlanError):
        ExportPlan.for_source(meta, [MediaRange(0.8, 1.0), MediaRange(0.1, 0.3)])


@NEEDS_TOOLS
def test_collision_without_overwrite_raises_before_any_work(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    sentinel = out_dir / "serve-001.mov"
    sentinel.write_bytes(b"sentinel-do-not-touch")
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    with pytest.raises(ExportCollisionError, match="collision"):
        export_clips(video, plan, out_dir, source=meta)
    assert sentinel.read_bytes() == b"sentinel-do-not-touch"
    assert not (out_dir / "serve-002.mov").exists()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_overwrite_replaces_explicitly(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    (out_dir / "serve-001.mov").write_bytes(b"stale")
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    (clip,) = export_clips(video, plan, out_dir, source=meta, overwrite=True)
    assert clip.read_bytes() != b"stale"
    assert probe_source(clip).duration_seconds == pytest.approx(
        0.5, abs=_sample_tolerance(meta)
    )


@NEEDS_TOOLS
def test_no_partial_final_file_when_ffmpeg_fails(tmp_path: Path, monkeypatch) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "clips"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    real_run = subprocess.run
    calls = {"count": 0}

    def _flaky(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 2:
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout="", stderr="boom"
            )
        return real_run(args, **kwargs)

    monkeypatch.setattr(export_module.subprocess, "run", _flaky)
    with pytest.raises(ExportError, match="ffmpeg failed.*clip 2 of 2"):
        export_clips(video, plan, out_dir, source=meta)
    assert (out_dir / "serve-001.mov").is_file()
    assert not (out_dir / "serve-002.mov").exists()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_cancellation_hook_cleans_up(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "clips"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5), MediaRange(1.0, 1.5)])
    polls = {"count": 0}

    def _cancel_on_second():
        polls["count"] += 1
        return polls["count"] >= 3

    with pytest.raises(ExportCancelled, match="cancelled"):
        export_clips(video, plan, out_dir, source=meta, is_cancelled=_cancel_on_second)
    assert (out_dir / "serve-001.mov").is_file()
    assert not (out_dir / "serve-002.mov").exists()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_keyboard_interrupt_cleans_up_temp_and_final(tmp_path: Path, monkeypatch) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "clips"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])

    def _interrupted(args, **kwargs):
        raise KeyboardInterrupt("simulated cancel")

    monkeypatch.setattr(export_module.subprocess, "run", _interrupted)
    with pytest.raises(KeyboardInterrupt):
        export_clips(video, plan, out_dir, source=meta)
    assert not (out_dir / "serve-001.mov").exists()
    assert _tmp_leftovers(out_dir) == []


@NEEDS_TOOLS
def test_explicit_encoder_selection_reaches_ffmpeg(
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
    export_clips(
        video, plan, tmp_path / "clips", source=meta, video_encoder="libx264"
    )
    args = captured["args"]
    assert args[args.index("-c:v") + 1] == "libx264"
    assert "copy" not in args
    assert any("trim=" in part for part in args)


@NEEDS_TOOLS
def test_default_encoder_is_explicit_and_never_stream_copy() -> None:
    assert DEFAULT_VIDEO_ENCODER == "h264_videotoolbox"
    assert SOFTWARE_VIDEO_ENCODER == "libx264"
    args = build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov")
    assert args[args.index("-c:v") + 1] == DEFAULT_VIDEO_ENCODER
    assert "copy" not in args
    # Hardware default carries an explicit review-band bitrate target;
    # libx264-specific preset/CRF flags are never emitted for VideoToolbox.
    assert args[args.index("-b:v") + 1] == DEFAULT_VIDEO_BITRATE
    assert "-preset" not in args and "-crf" not in args


def test_default_bitrate_is_documented_review_band_target() -> None:
    assert DEFAULT_VIDEO_BITRATE_BPS == 6_000_000
    assert DEFAULT_VIDEO_BITRATE == "6M"
    from serve_review.media import export as export_mod

    assert (
        export_mod.REVIEW_VIDEO_BITRATE_MIN_BPS
        <= DEFAULT_VIDEO_BITRATE_BPS
        <= export_mod.REVIEW_VIDEO_BITRATE_MAX_BPS
    )
    assert 4_000_000 <= export_mod._parse_video_bitrate_to_bps(
        DEFAULT_VIDEO_BITRATE
    ) <= 8_000_000


def test_libx264_override_keeps_software_options() -> None:
    args = build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov", video_encoder="libx264")
    assert args[args.index("-c:v") + 1] == "libx264"
    assert "-b:v" not in args
    assert args[args.index("-preset") + 1] == "veryfast"
    assert args[args.index("-crf") + 1] == "18"


def test_build_ffmpeg_args_rejects_bad_bitrate() -> None:
    with pytest.raises(ExportError, match="video_bitrate"):
        build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov", video_bitrate="  ")
    with pytest.raises(ExportError, match="video_bitrate"):
        build_ffmpeg_args("a.mov", 0.0, 0.5, "t.mov", video_bitrate="bogus")


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
def test_hardware_default_clip_is_valid_portrait_and_landscape(
    tmp_path: Path,
) -> None:
    for orientation in ("landscape", "portrait"):
        video, meta = _make_source(
            tmp_path, orientation=orientation, name=f"hw-{orientation}.mov"
        )
        plan = ExportPlan.for_source(meta, [MediaRange(0.2, 0.7)])
        (clip,) = export_clips(
            video, plan, tmp_path / f"hw-clips-{orientation}", source=meta
        )
        assert clip.is_file() and clip.stat().st_size > 0
        probed = probe_source(clip)
        assert _probe_video_codec(clip) == "h264"
        assert (probed.width, probed.height) == (meta.width, meta.height)
        assert probed.rotation_degrees == meta.rotation_degrees
        assert probed.duration_seconds == pytest.approx(
            0.5, abs=_sample_tolerance(meta)
        )


@NEEDS_VIDEOTOOLBOX
def test_hardware_bitrate_target_within_review_band(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    (clip,) = export_clips(video, plan, tmp_path / "hw-rate", source=meta)
    from serve_review.media import export as export_mod

    target_bps = export_mod._parse_video_bitrate_to_bps(DEFAULT_VIDEO_BITRATE)
    assert 4_000_000 <= target_bps <= 8_000_000
    # Tiny synthetic fixtures cannot saturate a 6 Mbps CBR target; the
    # measured average must simply be sane and below a generous ceiling.
    measured_bps = 8 * clip.stat().st_size / probe_source(clip).duration_seconds
    assert measured_bps > 10_000
    assert measured_bps < 12_000_000


@NEEDS_VIDEOTOOLBOX
def test_hardware_trim_accuracy_matches_software_path(tmp_path: Path) -> None:
    video, meta = _make_source(tmp_path)
    ranges = [MediaRange(0.2, 0.7), MediaRange(1.2, 1.8)]
    plan = ExportPlan.for_source(meta, ranges)
    tolerance = _sample_tolerance(meta)
    hw_clips = export_clips(video, plan, tmp_path / "hw-trim", source=meta)
    sw_clips = export_clips(
        video, plan, tmp_path / "sw-trim", source=meta,
        video_encoder="libx264",
    )
    assert [p.name for p in hw_clips] == [p.name for p in sw_clips]
    for hw, sw, expected in zip(hw_clips, sw_clips, ranges):
        hw_dur = probe_source(hw).duration_seconds
        sw_dur = probe_source(sw).duration_seconds
        assert hw_dur == pytest.approx(expected.duration_seconds, abs=tolerance)
        assert sw_dur == pytest.approx(expected.duration_seconds, abs=tolerance)
        assert hw_dur == pytest.approx(sw_dur, abs=tolerance)


@NEEDS_TOOLS
def test_hardware_failure_names_encoder_and_cleans_up(
    tmp_path: Path, monkeypatch
) -> None:
    video, meta = _make_source(tmp_path)
    out_dir = tmp_path / "hw-fail"
    plan = ExportPlan.for_source(meta, [MediaRange(0.0, 0.5)])
    real_run = subprocess.run

    def _fail(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr=(
                "[h264_videotoolbox @ 0x123] Error encoding frame: "
                "hardware unavailable (-12908)"
            ),
        )

    monkeypatch.setattr(export_module.subprocess, "run", _fail)
    with pytest.raises(ExportError) as excinfo:
        export_clips(video, plan, out_dir, source=meta)
    message = str(excinfo.value)
    assert "h264_videotoolbox" in message
    assert "hardware unavailable" in message
    assert not (out_dir / "serve-001.mov").exists()
    assert _tmp_leftovers(out_dir) == []
