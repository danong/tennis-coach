"""Serve-review command-line entry point."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import click

from . import commands
from .main import cli, main


# Direct adapters remain importable for focused workflow tests; command parsing is
# owned exclusively by ``cli`` in main.py.
def _values(args: object) -> dict[str, object]:
    return {key: value for key, value in vars(args).items() if key != "handler"}


def cut(args: object) -> int:
    return commands.cut(**_values(args))


def probe(args: object) -> int:
    return commands.probe(**_values(args))


def export_cmd(args: object) -> int:
    return commands.export_cmd(**_values(args))


def extract_poses_cmd(args: object) -> int:
    return commands.extract_poses_cmd(**_values(args))


def process_cmd(args: object) -> int:
    return commands.process_cmd(**_values(args))


def analyze_serve(args: object) -> int:
    return commands.analyze_serve(**_values(args))


class _ClickTestParser:
    def format_help(self) -> str:
        return cli.get_help(click.Context(cli, info_name="serve-review"))

    def parse_args(self, argv: Sequence[str]) -> SimpleNamespace:
        context = click.Context(cli, info_name="serve-review")
        try:
            name, command, remaining = cli.resolve_command(context, list(argv))
            parsed = command.make_context(name, remaining, parent=context)
        except click.exceptions.Exit as exc:
            raise SystemExit(exc.exit_code) from exc
        except click.ClickException as exc:
            exc.show()
            raise SystemExit(exc.exit_code) from exc
        values = dict(parsed.params)
        handlers = {"cut": cut, "probe": probe, "export": export_cmd,
                    "extract-poses": extract_poses_cmd, "process": process_cmd,
                    "analyze-serve": analyze_serve}
        if name in handlers:
            values["handler"] = handlers[name]
        if name == "analyze-serve":
            values["anchor2comparison"] = False
        return SimpleNamespace(**values)


def build_parser() -> _ClickTestParser:
    """Test-only Click declaration adapter; not used by the CLI."""
    return _ClickTestParser()


__all__ = ["main"]
