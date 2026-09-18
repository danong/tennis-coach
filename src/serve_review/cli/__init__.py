"""Click CLI entry point and transitional direct-adapter exports for tests."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import click

from serve_review import cli_legacy as _legacy
from .main import cli, main

cut = _legacy.cut
probe = _legacy.probe
export_cmd = _legacy.export_cmd
extract_poses_cmd = _legacy.extract_poses_cmd
process_cmd = _legacy.process_cmd
analyze_serve = _legacy.analyze_serve


class _ClickCommandParser:
    """Expose Click-declared options to legacy direct-adapter tests.

    This is not an argparse parser; production command parsing and help remain
    entirely Click-owned.
    """

    def format_help(self) -> str:
        context = click.Context(cli, info_name="serve-review")
        return cli.get_help(context)

    def parse_args(self, argv: Sequence[str]) -> SimpleNamespace:
        context = click.Context(cli, info_name="serve-review")
        try:
            command_name, command, remaining = cli.resolve_command(context, list(argv))
            command_context = command.make_context(command_name, remaining, parent=context)
        except click.exceptions.Exit as exc:
            raise SystemExit(exc.exit_code) from exc
        except click.ClickException as exc:
            exc.show()
            raise SystemExit(exc.exit_code) from exc
        values = dict(command_context.params)
        handlers = {
            "cut": cut,
            "probe": probe,
            "export": export_cmd,
            "extract-poses": extract_poses_cmd,
            "process": process_cmd,
            "analyze-serve": analyze_serve,
        }
        if command_name in handlers:
            values["handler"] = handlers[command_name]
        if command_name == "analyze-serve":
            values["anchor2comparison"] = False
        return SimpleNamespace(**values)


def build_parser() -> _ClickCommandParser:
    """Compatibility test helper backed by the Click command declarations."""
    return _ClickCommandParser()


__all__ = ["main"]
