"""Click command surface for serve-review workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from . import commands


def _run(handler: Any, **values: Any) -> None:
    result = handler(**values)
    if result:
        raise click.exceptions.Exit(result)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Offline tennis serve processing and diagnostic tools."""


@cli.command()
@click.argument("target", type=click.Path(path_type=Path))
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
def process(target: Path, dry_run: bool, force: bool) -> None:
    """Process a recording directory or one video beside its source."""
    _run(commands.process_cmd, target=target, dry_run=dry_run, force=force)


@cli.command("cut")
@click.argument("video", type=click.Path(path_type=Path))
@click.option("--padding", type=float, default=1.0, show_default=True)
@click.option("--output", type=click.Choice(["compilation", "clips", "both"]), default="compilation")
@click.option("--metadata-dir", type=click.Path(path_type=Path), default=None)
@click.option("--export-dir", type=click.Path(path_type=Path), default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--ffmpeg", default="ffmpeg")
@click.option("--ffprobe", default="ffprobe")
def cut(**values: Any) -> None:
    """Detect and export serves."""
    _run(commands.cut, **values)


@cli.command("analyze-serve")
@click.argument("video", type=click.Path(path_type=Path))
@click.option("--start-seconds", type=float, default=None)
@click.option("--end-seconds", type=float, default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--cache", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=click.Path(path_type=Path), default=Path("models/pose_landmarker_heavy.task"))
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--ffmpeg", default="ffmpeg")
@click.option("--ffprobe", default="ffprobe")
def analyze_serve(**values: Any) -> None:
    """Analyze one explicit serve range."""
    _run(commands.analyze_serve, anchor2comparison=False, **values)


@cli.command()
def doctor() -> None:
    """Check local prerequisites."""
    result = commands.doctor()
    if result:
        raise click.exceptions.Exit(result)


@cli.command()
@click.argument("video", type=click.Path(path_type=Path))
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--ffprobe", default="ffprobe")
def probe(**values: Any) -> None:
    """Inspect a video and emit normalized source.json metadata."""
    _run(commands.probe, **values)


@cli.command("export")
@click.argument("video", type=click.Path(path_type=Path))
@click.option("--ranges", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Choice(["compilation", "clips", "both"]), default="compilation")
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("output"))
@click.option("--overwrite", is_flag=True)
@click.option("--ffmpeg", default="ffmpeg")
@click.option("--ffprobe", default="ffprobe")
def export(**values: Any) -> None:
    """Export manual ranges."""
    _run(commands.export_cmd, **values)


@cli.command("extract-poses")
@click.argument("video", type=click.Path(path_type=Path))
@click.option("--model", type=click.Path(path_type=Path), default=Path("models/pose_landmarker_heavy.task"))
@click.option("--sample-rate", type=float, default=30.0)
@click.option("--cache", type=click.Path(path_type=Path), default=None)
@click.option("--overlay", type=click.Path(path_type=Path), default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("output"))
@click.option("--overwrite", is_flag=True)
@click.option("--ffmpeg", default="ffmpeg")
@click.option("--ffprobe", default="ffprobe")
def extract_poses(**values: Any) -> None:
    """Extract cached body-pose observations for diagnostics."""
    _run(commands.extract_poses_cmd, **values)


@cli.group()
def dev() -> None:
    """Development and evaluation commands."""


@dev.command("anchor2-comparison")
@click.argument("video", type=click.Path(path_type=Path))
@click.option("--start-seconds", type=float, default=None)
@click.option("--end-seconds", type=float, default=None)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--cache", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=click.Path(path_type=Path), default=Path("models/pose_landmarker_heavy.task"))
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--ffmpeg", default="ffmpeg")
@click.option("--ffprobe", default="ffprobe")
def anchor2_comparison(**values: Any) -> None:
    """Analyze the fixed anchor-2 development comparison."""
    _run(commands.analyze_serve, anchor2comparison=True, **values)


@dev.command("phase-annotate")
@click.argument("video", type=click.Path(path_type=Path))
@click.option("--attempts", type=click.Path(path_type=Path), required=True)
@click.option("--labels", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--ffprobe", default="ffprobe")
@click.option("--overwrite", is_flag=True)
def phase_annotate(video: Path, attempts: Path, labels: Path, output: Path, ffprobe: str, overwrite: bool) -> None:
    from tools.phase_annotations import AnnotationToolError, annotate
    try: click.echo(annotate(video, attempts, labels, output, ffprobe=ffprobe, overwrite=overwrite))
    except AnnotationToolError as exc: raise click.ClickException(str(exc)) from exc


@dev.command("phase-evaluate")
@click.option("--checkpoints", type=click.Path(path_type=Path), required=True)
@click.option("--annotations", type=click.Path(path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--overwrite", is_flag=True)
def phase_evaluate(checkpoints: Path, annotations: Path, output: Path, overwrite: bool) -> None:
    from tools.evaluate_checkpoints import evaluate
    try: click.echo(evaluate(checkpoints, annotations, output, overwrite=overwrite))
    except ValueError as exc: raise click.ClickException(str(exc)) from exc


def main() -> int:
    cli(standalone_mode=False)
    return 0
