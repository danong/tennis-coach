from pathlib import Path

import pytest

from serve_review.cli import build_parser, cut, export_cmd, probe


def test_cut_defaults() -> None:
    args = build_parser().parse_args(["cut", "session.mov"])

    assert args.video == Path("session.mov")
    assert args.padding == 1.0
    assert args.output == "compilation"
    assert args.output_dir == Path("output")


def test_cut_rejects_missing_input(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(["cut", str(tmp_path / "missing.mov")])

    assert cut(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_cut_rejects_negative_padding(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.mov"
    source.touch()
    args = build_parser().parse_args(["cut", str(source), "--padding", "-0.1"])

    assert cut(args) == 2
    assert "zero or greater" in capsys.readouterr().err


def test_cut_truthfully_reports_unimplemented_pipeline(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.mov"
    source.touch()
    args = build_parser().parse_args(["cut", str(source)])

    assert cut(args) == 3
    assert "not implemented yet" in capsys.readouterr().err


def test_probe_help_documents_command(capsys) -> None:
    top_help = build_parser().format_help()
    assert "probe" in top_help
    assert "source.json" in top_help

    parser = build_parser()
    with __import__("pytest").raises(SystemExit) as excinfo:
        parser.parse_args(["probe", "--help"])
    assert excinfo.value.code == 0
    probe_help = capsys.readouterr().out
    assert "ffprobe" in probe_help.lower()
    assert "--output" in probe_help


def test_probe_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(["probe", "session.mov"])

    assert args.video == Path("session.mov")
    assert args.output is None
    assert args.ffprobe == "ffprobe"


def test_probe_rejects_missing_input(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(["probe", str(tmp_path / "missing.mov")])

    assert probe(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_probe_success_writes_source_json(tmp_path, monkeypatch, capsys) -> None:
    import json

    from serve_review.domain import SourceMetadata
    from serve_review.media import probe as probe_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    dest = tmp_path / "out" / "source.json"
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "120/1",
                "r_frame_rate": "120/1",
                "time_base": "1/90000",
                "duration": "4.0",
                "side_data_list": [
                    {"side_data_type": "Display Matrix", "rotation": -90}
                ],
            }
        ],
        "format": {"duration": "4.0"},
    }

    import subprocess

    def _fake_run(argv, **kwargs):
        assert isinstance(argv, list)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(probe_module.subprocess, "run", _fake_run)
    args = build_parser().parse_args(["probe", str(source), "--output", str(dest)])

    assert probe(args) == 0
    assert dest.is_file()
    stored = SourceMetadata.from_json(dest.read_text(encoding="utf-8"))
    assert stored.rotation_degrees == 270
    assert (stored.frame_rate_num, stored.frame_rate_den) == (120, 1)
    assert str(dest) in capsys.readouterr().out


def test_probe_reports_probe_failure(tmp_path, monkeypatch, capsys) -> None:
    import subprocess

    from serve_review.media import probe as probe_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    dest = tmp_path / "source.json"

    def _fake_fail(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv, returncode=1, stdout="", stderr="Invalid data"
        )

    monkeypatch.setattr(probe_module.subprocess, "run", _fake_fail)
    args = build_parser().parse_args(["probe", str(source), "--output", str(dest)])

    assert probe(args) == 1
    assert "ERROR" in capsys.readouterr().err
    assert not dest.exists()


def test_export_help_documents_command(capsys) -> None:
    top_help = build_parser().format_help()
    assert "export" in top_help

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["export", "--help"])
    assert excinfo.value.code == 0
    export_help = capsys.readouterr().out
    assert "--ranges" in export_help
    assert "compilation" in export_help


def test_export_defaults() -> None:
    args = build_parser().parse_args(
        ["export", "session.mov", "--ranges", "ranges.json"]
    )

    assert args.video == Path("session.mov")
    assert args.ranges == Path("ranges.json")
    assert args.output == "compilation"
    assert args.output_dir == Path("output")
    assert args.overwrite is False
    assert args.ffmpeg == "ffmpeg"
    assert args.ffprobe == "ffprobe"


def test_export_requires_ranges_option() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["export", "session.mov"])


def test_export_rejects_missing_video(tmp_path: Path, capsys) -> None:
    ranges = tmp_path / "ranges.json"
    ranges.write_text("[]", encoding="utf-8")
    args = build_parser().parse_args(
        ["export", str(tmp_path / "missing.mov"), "--ranges", str(ranges)]
    )

    assert export_cmd(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_export_rejects_missing_ranges_file(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.mov"
    source.touch()
    args = build_parser().parse_args(
        ["export", str(source), "--ranges", str(tmp_path / "missing.json")]
    )

    assert export_cmd(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_export_rejects_malformed_ranges_json(tmp_path: Path, capsys) -> None:
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=1.0)
    )
    bad = tmp_path / "ranges.json"
    bad.write_text("{not json", encoding="utf-8")
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(bad), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "invalid ranges JSON" in capsys.readouterr().err


def test_export_rejects_empty_ranges(tmp_path: Path, capsys) -> None:
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=1.0)
    )
    ranges = tmp_path / "ranges.json"
    ranges.write_text("[]", encoding="utf-8")
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "at least one range" in capsys.readouterr().err


def test_export_rejects_overlapping_ranges(tmp_path: Path, capsys) -> None:
    import json
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=2.0)
    )
    ranges = tmp_path / "ranges.json"
    ranges.write_text(
        json.dumps(
            [
                {"start_seconds": 0.0, "end_seconds": 0.6},
                {"start_seconds": 0.4, "end_seconds": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "invalid ranges" in capsys.readouterr().err


def test_export_rejects_out_of_bounds_ranges(tmp_path: Path, capsys) -> None:
    import json
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=1.0)
    )
    ranges = tmp_path / "ranges.json"
    ranges.write_text(
        json.dumps([{"start_seconds": 0.5, "end_seconds": 5.0}]),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "invalid ranges" in capsys.readouterr().err


def _make_export_fixture(tmp_path: Path):
    from media_factory import generate_fixture, landscape_spec
    from serve_review.media.probe import probe_source

    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=2.0)
    )
    meta = probe_source(video)
    return video, meta


def _write_ranges(tmp_path: Path, entries) -> Path:
    import json

    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_export_compilation_mode_reports_output(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    from serve_review.media.probe import probe_source

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path,
        [
            {"start_seconds": 0.2, "end_seconds": 0.7},
            {"start_seconds": 1.2, "end_seconds": 1.8},
        ],
    )
    out_base = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output",
            "compilation",
            "--output-dir",
            str(out_base),
        ]
    )
    before = video.read_bytes()

    assert export_cmd(args) == 0
    comp = out_base / video.stem / "serves.mov"
    assert comp.is_file()
    assert str(comp) in capsys.readouterr().out
    probed = probe_source(comp)
    assert probed.duration_seconds == _pytest.approx(
        1.1, abs=1.0 / meta.frames_per_second + 0.02
    )
    assert video.read_bytes() == before


def test_export_clips_and_both_modes(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    from serve_review.media.probe import probe_source

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path, [{"start_seconds": 0.2, "end_seconds": 0.7}]
    )

    clips_base = tmp_path / "out-clips"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output",
            "clips",
            "--output-dir",
            str(clips_base),
        ]
    )
    assert export_cmd(args) == 0
    clip = clips_base / video.stem / "clips" / "serve-001.mov"
    assert clip.is_file()
    assert str(clip) in capsys.readouterr().out
    assert not (clips_base / video.stem / "serves.mov").exists()

    both_base = tmp_path / "out-both"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output",
            "both",
            "--output-dir",
            str(both_base),
        ]
    )
    assert export_cmd(args) == 0
    assert (both_base / video.stem / "serves.mov").is_file()
    assert (both_base / video.stem / "clips" / "serve-001.mov").is_file()
    out = capsys.readouterr().out
    assert "serves.mov" in out and "serve-001.mov" in out
    probed = probe_source(both_base / video.stem / "serves.mov")
    assert probed.duration_seconds == _pytest.approx(
        0.5, abs=1.0 / meta.frames_per_second + 0.02
    )


def test_export_accepts_full_plan_document(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    from serve_review.domain import ExportPlan, MediaRange

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.2, 0.7)])
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.to_json(), encoding="utf-8")
    out_base = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(plan_path),
            "--output-dir",
            str(out_base),
        ]
    )

    assert export_cmd(args) == 0
    assert (out_base / video.stem / "serves.mov").is_file()


def test_export_rejects_plan_fingerprint_mismatch(tmp_path: Path, capsys) -> None:
    import json
    import shutil

    from media_factory import generate_fixture, landscape_spec
    from serve_review.domain import ExportPlan, MediaRange
    from serve_review.media.probe import probe_source

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=2.0)
    )
    meta = probe_source(video)
    plan = ExportPlan.for_source(meta, [MediaRange(0.2, 0.7)])
    payload = plan.to_dict()
    payload["source_fingerprint"] = "sha256:" + "f" * 64
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(plan_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert export_cmd(args) == 2
    assert "fingerprint" in capsys.readouterr().err


def test_export_collision_reports_actionable_error(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path, [{"start_seconds": 0.2, "end_seconds": 0.7}]
    )
    out_base = tmp_path / "out"
    first = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(out_base)]
    )
    assert export_cmd(first) == 0
    capsys.readouterr()

    second = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(out_base)]
    )
    assert export_cmd(second) == 1
    assert "collision" in capsys.readouterr().err.lower()

    retry = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output-dir",
            str(out_base),
            "--overwrite",
        ]
    )
    assert export_cmd(retry) == 0


def test_export_reports_ffmpeg_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    import shutil
    import subprocess

    from serve_review.media import export as export_module

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path, [{"start_seconds": 0.2, "end_seconds": 0.7}]
    )
    real_run = subprocess.run

    def _fail_ffmpeg_only(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr(export_module.subprocess, "run", _fail_ffmpeg_only)
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 1
    assert "ERROR" in capsys.readouterr().err
    assert not (tmp_path / "out" / video.stem / "serves.mov").exists()
