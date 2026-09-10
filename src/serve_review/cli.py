from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _tool_version(executable: str) -> str | None:
    path = shutil.which(executable)
    if path is None:
        return None
    result = subprocess.run(
        [path, "-version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0] if first_line else path


def doctor() -> int:
    failures: list[str] = []
    print(f"Python {sys.version.split()[0]} ({sys.executable})")

    if sys.version_info[:2] != (3, 11):
        failures.append("Python 3.11 is required; run `mise install` then `mise run setup`.")

    for executable in ("ffmpeg", "ffprobe"):
        version = _tool_version(executable)
        if version is None:
            failures.append(
                f"{executable} was not found or could not run; install FFmpeg and ensure it is on PATH."
            )
        else:
            print(version)

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("Offline pipeline prerequisites are available.")
    return 0


def probe(args: argparse.Namespace) -> int:
    from serve_review.media.probe import ProbeError, probe_source, write_source_json

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    if args.output is not None:
        dest = args.output.expanduser()
    else:
        dest = Path("output") / source.stem / "source.json"
    try:
        metadata = probe_source(source, ffprobe=args.ffprobe)
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        written = write_source_json(metadata, dest)
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(str(written))
    return 0


def cut(args: argparse.Namespace) -> int:
    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    if args.padding < 0:
        print("ERROR: --padding must be zero or greater.", file=sys.stderr)
        return 2

    print(
        "Serve detection/export is not implemented yet. "
        "The next milestone adds manual-range export before model inference.",
        file=sys.stderr,
    )
    return 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serve-review",
        description="Detect tennis serves in a recording and export useful ranges.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="check local prerequisites")
    doctor_parser.set_defaults(handler=lambda _args: doctor())

    probe_parser = subparsers.add_parser(
        "probe",
        help="inspect a video with ffprobe and emit normalized source.json",
        description=(
            "Run ffprobe on a source MOV/MP4 video and write normalized "
            "source.json metadata (dimensions, rational frame rate/time base, "
            "duration, codec, rotation)."
        ),
    )
    probe_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    probe_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="destination source.json file (default: output/<source-stem>/source.json)",
    )
    probe_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    probe_parser.set_defaults(handler=probe)

    cut_parser = subparsers.add_parser("cut", help="detect and export serves")
    cut_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    cut_parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="seconds to retain before and after each detected serve (default: 1)",
    )
    cut_parser.add_argument(
        "--output",
        choices=("compilation", "clips", "both"),
        default="compilation",
        help="output form (default: compilation)",
    )
    cut_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="generated output directory (default: output)",
    )
    cut_parser.set_defaults(handler=cut)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
