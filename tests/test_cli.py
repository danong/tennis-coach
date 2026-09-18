"""Focused public command-surface checks."""
from __future__ import annotations

from click.testing import CliRunner

from serve_review.cli.main import cli


def test_root_help_lists_existing_root_and_dev_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("process", "cut", "analyze-serve", "doctor", "probe", "export", "extract-poses", "dev"):
        assert command in result.output


def test_development_help_lists_development_commands() -> None:
    result = CliRunner().invoke(cli, ["dev", "--help"])
    assert result.exit_code == 0
    assert "anchor2-comparison" in result.output
    assert "phase-annotate" in result.output
    assert "phase-evaluate" in result.output


def test_analyze_rejects_removed_anchor2_option() -> None:
    result = CliRunner().invoke(cli, ["analyze-serve", "session.mov", "--anchor2comparison"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_process_missing_target_keeps_validation_exit_code() -> None:
    result = CliRunner().invoke(cli, ["process", "missing.mov"])
    assert result.exit_code == 2
    assert "does not exist" in result.output
