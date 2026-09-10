from pathlib import Path

from serve_review.cli import build_parser, cut, probe


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
