from pathlib import Path

from serve_review.cli import build_parser, cut


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
