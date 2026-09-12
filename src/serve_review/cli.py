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


def export_cmd(args: argparse.Namespace) -> int:
    import json

    from serve_review.domain import DomainError, ExportPlan, MediaRange
    from serve_review.media import export as export_module
    from serve_review.media.probe import ProbeError, probe_source

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    ranges_path = args.ranges.expanduser()
    if not ranges_path.is_file():
        print(f"ERROR: ranges file does not exist: {ranges_path}", file=sys.stderr)
        return 2
    try:
        metadata = probe_source(source, ffprobe=args.ffprobe)
    except ProbeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        raw_text = ranges_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: could not read ranges file {ranges_path}: {exc}.", file=sys.stderr)
        return 2
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid ranges JSON in {ranges_path}: {exc}.", file=sys.stderr)
        return 2
    try:
        plan = _parse_manual_ranges(payload, metadata)  # type: ignore[arg-type]
    except DomainError as exc:
        print(f"ERROR: invalid ranges: {exc}.", file=sys.stderr)
        return 2
    base_dir = args.output_dir.expanduser() / source.stem
    try:
        results = export_module.export_outputs(
            source,
            plan,
            base_dir,
            mode=args.output,
            source=metadata,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            overwrite=args.overwrite,
        )
    except export_module.ExportCollisionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except export_module.ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    compilation = results.get("compilation")
    clips = results.get("clips", [])
    assert isinstance(clips, list)
    if compilation is not None:
        print(str(compilation))
    for clip in clips:
        print(str(clip))
    return 0


def _parse_manual_ranges(payload: object, metadata) -> object:
    """Parse a manual ranges JSON payload into a validated ExportPlan."""
    from serve_review.domain import ExportPlan, MediaRange

    # Full export-plan document: validate strictly, then check identity.
    if isinstance(payload, dict) and {"ranges", "source_fingerprint"} <= set(payload):
        plan = ExportPlan.from_dict(payload)  # type: ignore[arg-type]
        if plan.source_fingerprint != metadata.fingerprint:
            raise plan_error(
                "export plan fingerprint does not match the source video; "
                "re-probe the source and rebuild the plan from its metadata."
            )
        import math as _math

        if abs(plan.source_duration_seconds - metadata.duration_seconds) > 1e-6:
            raise plan_error(
                "export plan duration does not match the source video; "
                "re-probe the source and rebuild the plan from its metadata."
            )
        void = _math  # keep import local and explicit
        del void
        return plan
    if isinstance(payload, dict) and "ranges" in payload:
        raw_ranges = payload["ranges"]
    elif isinstance(payload, list):
        raw_ranges = payload
    else:
        raise plan_error(
            "ranges file must be a JSON list of ranges or an object "
            'with a "ranges" list.'
        )
    if not isinstance(raw_ranges, list):
        raise plan_error('"ranges" must be a JSON list.')
    if not raw_ranges:
        raise plan_error("at least one range is required; an empty plan would produce no output.")
    parsed: list[MediaRange] = []
    for entry in raw_ranges:
        parsed.append(_parse_manual_range(entry))
    return ExportPlan.for_source(metadata, parsed)


def _parse_manual_range(entry: object) -> object:
    from serve_review.domain import MediaRange

    if not isinstance(entry, dict):
        raise plan_error(f"invalid range entry {entry!r}; expected an object.")
    if "start_seconds" not in entry or "end_seconds" not in entry:
        raise plan_error(f"invalid range entry {entry!r}; expected start_seconds/end_seconds.")
    known = {"start_seconds", "end_seconds", "schema_version"}
    unknown = sorted(set(entry) - known)
    if unknown:
        raise plan_error(f"invalid range entry {entry!r}; unknown keys {unknown!r}.")
    values: dict[str, object] = {
        "start_seconds": entry["start_seconds"],
        "end_seconds": entry["end_seconds"],
        "schema_version": entry.get("schema_version", 1),
    }
    return MediaRange.from_dict(values)  # type: ignore[arg-type]


def plan_error(message: str) -> Exception:
    from serve_review.domain import ExportPlanError

    return ExportPlanError(message)


def extract_poses_cmd(args: argparse.Namespace) -> int:
    from serve_review.pose import extract as extract_module

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    try:
        rate = float(args.sample_rate)
    except (TypeError, ValueError):
        print(
            f"ERROR: invalid --sample-rate {args.sample_rate!r}; expected a number in (0, 120].",
            file=sys.stderr,
        )
        return 2
    if not (0 < rate <= 120.0):
        print(
            f"ERROR: invalid --sample-rate {args.sample_rate!r}; expected a number in (0, 120].",
            file=sys.stderr,
        )
        return 2
    if args.cache is not None:
        cache_path = args.cache.expanduser()
    else:
        cache_path = extract_module.default_cache_path_for(source, args.output_dir)
    model_path = args.model.expanduser()
    overlay = args.overlay.expanduser() if args.overlay is not None else None

    def _progress(done: int, total: int, observation: object) -> None:
        print(f"extract-poses: {done}/{total} frames", file=sys.stderr)

    def _overlay_progress(done: int, total: int | None) -> None:
        if total is None:
            print(f"overlay: {done} frames", file=sys.stderr)
        else:
            print(f"overlay: {done}/{total} frames", file=sys.stderr)

    try:
        result = extract_module.extract_poses(
            source,
            cache_path,
            sample_rate_hz=rate,
            model_path=model_path,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            overwrite=args.overwrite,
            progress_callback=_progress,
            overlay_path=overlay,
            overlay_progress_callback=_overlay_progress if overlay is not None else None,
        )
    except extract_module.ExtractionCancelled as exc:
        print(f"ERROR: pose extraction was cancelled: {exc}.", file=sys.stderr)
        return 1
    except extract_module.ExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.cache_hit:
        print(f"cache hit: {result.cache_path} ({result.frame_count} frames)")
    else:
        print(
            f"extracted {result.frame_count} frames "
            f"({result.inferred_frames} inferred, {result.cached_frames} cached) "
            f"to {result.cache_path}"
        )
    print(str(result.cache_path))
    if result.overlay_path is not None:
        print(f"overlay: {result.overlay_path}")
    return 0


def cut(args: argparse.Namespace) -> int:
    from serve_review.pipeline import CutCancelled, CutError, run_cut

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    try:
        padding = float(args.padding)
    except (TypeError, ValueError):
        print(
            f"ERROR: invalid --padding {args.padding!r}; "
            "expected seconds >= 0.",
            file=sys.stderr,
        )
        return 2
    if not (padding >= 0):
        print("ERROR: --padding must be zero or greater.", file=sys.stderr)
        return 2

    def _progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        result = run_cut(
            source,
            output_dir=args.output_dir,
            padding_seconds=padding,
            mode=args.output,
            overwrite=args.overwrite,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            progress_callback=_progress,
        )
    except CutCancelled as exc:
        print(f"ERROR: cut was cancelled at {exc.stage}: {exc.message}.", file=sys.stderr)
        return 1
    except CutError as exc:
        print(f"ERROR: cut failed at {exc.stage}: {exc.message}.", file=sys.stderr)
        return 1
    print(str(result.attempts_path))
    if result.shadows_path is not None:
        print(str(result.shadows_path))
    if result.empty:
        print("no serves detected; wrote empty attempts only (no media output).")
        return 0
    if result.compilation is not None:
        print(str(result.compilation))
    for clip in result.clips:
        print(str(clip))
    return 0


def analyze(args: argparse.Namespace) -> int:
    from serve_review.analysis_pipeline import (
        AnalyzeCancelled,
        AnalyzeError,
        run_analyze,
    )

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    attempts = args.attempts.expanduser() if args.attempts is not None else None
    if attempts is not None and not attempts.is_file():
        print(f"ERROR: attempts file does not exist: {attempts}", file=sys.stderr)
        return 2

    def _progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        result = run_analyze(
            source,
            output_dir=args.output_dir,
            attempts_path=attempts,
            overwrite=args.overwrite,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            progress_callback=_progress,
        )
    except AnalyzeCancelled as exc:
        print(
            f"ERROR: analyze was cancelled at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 1
    except AnalyzeError as exc:
        print(
            f"ERROR: analyze failed at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 1
    print(str(result.checkpoints_path))
    if result.empty:
        print("no attempts to analyze; wrote empty checkpoints.")
        return 0
    failures = len(result.attempt_failures)
    if failures:
        print(
            f"analyzed {len(result.phase_document)} attempt(s) with "
            f"{failures} isolated phase failure(s): "
            f"{result.checkpoints_path}",
        )
    else:
        print(
            f"analyzed {len(result.phase_document)} attempt(s): "
            f"{result.checkpoints_path}",
        )
    return 0


def review_phases(args: argparse.Namespace) -> int:
    from serve_review.phase_review import (
        ReviewCancelled,
        ReviewError,
        run_review,
    )

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    checkpoints = args.checkpoints.expanduser() if args.checkpoints is not None else None
    if checkpoints is not None and not checkpoints.is_file():
        print(f"ERROR: checkpoints file does not exist: {checkpoints}", file=sys.stderr)
        return 2
    cache = args.cache.expanduser() if args.cache is not None else None
    output = args.output.expanduser() if args.output is not None else None

    def _progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        result = run_review(
            source,
            output_dir=args.output_dir,
            checkpoints_path=checkpoints,
            output=output,
            cache_path=cache,
            overwrite=args.overwrite,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            progress_callback=_progress,
        )
    except ReviewCancelled as exc:
        print(
            f"ERROR: phase review was cancelled at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 1
    except ReviewError as exc:
        print(
            f"ERROR: phase review failed at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 1
    print(str(result.review_json))
    print(str(result.index_html))
    if result.empty:
        print("no phases to review; wrote empty review index.")
        return 0
    print(
        f"rendered {result.image_count} image(s) for "
        f"{result.entry_count} phase(s): {result.review_dir}",
    )
    return 0


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
    cut_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing outputs (default: fail on collision)",
    )
    cut_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    cut_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    cut_parser.set_defaults(handler=cut)

    export_parser = subparsers.add_parser(
        "export",
        help="export manual ranges as clips and/or a compilation",
        description=(
            "Concatenate validated manual source ranges without dead-time gaps. "
            "Ranges are read from a JSON file (a list of "
            "{start_seconds, end_seconds} objects or a full export-plan document) "
            "and exported from the original source samples with filter-based "
            "re-encoding."
        ),
    )
    export_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    export_parser.add_argument(
        "--ranges",
        type=Path,
        required=True,
        metavar="RANGES_JSON",
        help="JSON file with manual ranges to export",
    )
    export_parser.add_argument(
        "--output",
        choices=("compilation", "clips", "both"),
        default="compilation",
        help="output form (default: compilation)",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="generated output directory (default: output)",
    )
    export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing outputs (default: fail on collision)",
    )
    export_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    export_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    export_parser.set_defaults(handler=export_cmd)

    poses_parser = subparsers.add_parser(
        "extract-poses",
        help="extract cached body-pose observations for diagnostics",
        description=(
            "Sample timestamped frames at --sample-rate Hz, run the approved "
            "Heavy Pose Landmarker in serialized VIDEO mode, and write a "
            "resumable versioned pose cache. Complete caches with matching "
            "identity are reused without inference; interrupted caches resume."
        ),
    )
    poses_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    poses_parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_heavy.task"),
        metavar="PATH",
        help="approved Pose Landmarker .task artifact (default: models/pose_landmarker_heavy.task)",
    )
    poses_parser.add_argument(
        "--sample-rate",
        type=float,
        default=30.0,
        metavar="HZ",
        help="uniform sampling rate in Hz (default: 30)",
    )
    poses_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        metavar="PATH",
        help="destination pose cache file (default: <output-dir>/<source-stem>/cache/pose-v1.jsonl)",
    )
    poses_parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "optional diagnostic .mp4: render the sampled upright frames "
            "with pose landmarks and skeleton lines via the system ffmpeg "
            "(default: off)"
        ),
    )
    poses_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="generated output directory (default: output)",
    )
    poses_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace any existing cache instead of resuming it",
    )
    poses_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    poses_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    poses_parser.set_defaults(handler=extract_poses_cmd)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="analyze accepted attempts into phase checkpoints",
        description=(
            "Load accepted serve attempts (an explicit --attempts file or "
            "the prior cut output for the source), reuse the compatible "
            "cached pose observations, demux raw-source audio once, run "
            "phase features/evidence/solver per attempt, and atomically "
            "write checkpoints.json. Never modifies attempts.json, clips, "
            "or the compilation."
        ),
    )
    analyze_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    analyze_parser.add_argument(
        "--attempts",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "explicit attempts JSON file (default: "
            "output/<source-stem>/attempts.json from a prior cut)"
        ),
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="generated output directory (default: output)",
    )
    analyze_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing checkpoints.json (default: fail on collision)",
    )
    analyze_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    analyze_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    analyze_parser.set_defaults(handler=analyze)

    review_parser = subparsers.add_parser(
        "review-phases",
        help="render phase keyframes with skeleton overlays for review",
        description=(
            "Sample the original upright source at each available/partial "
            "phase keyframe, pair it with the nearest cached pose "
            "observation within a bounded tolerance, render the existing "
            "skeleton overlay, burn a deterministic raster caption into a "
            "JPEG (attempt ID, stage, source times, confidence/provenance, "
            "availability, anomalies), and write a deterministic review "
            "index (review.json, index.html, MANIFEST.txt). Unavailable or "
            "unsupported phases produce no image but appear in the index. "
            "Contact is labelled with provenance and source time and is "
            "never claimed as exact visual observation. Never modifies "
            "checkpoints.json, attempts.json, clips, or the compilation."
        ),
    )
    review_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    review_parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "explicit checkpoints JSON file (default: "
            "output/<source-stem>/checkpoints.json from analyze)"
        ),
    )
    review_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="generated output directory (default: output)",
    )
    review_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "explicit final review directory (default: "
            "output/<source-stem>/review-phases)"
        ),
    )
    review_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "explicit pose cache file (default: "
            "<output-dir>/<source-stem>/cache/pose-v1.jsonl); must be a "
            "complete cache with matching identity, never re-inferred"
        ),
    )
    review_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing review directory (default: fail on collision)",
    )
    review_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    review_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    review_parser.set_defaults(handler=review_phases)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
