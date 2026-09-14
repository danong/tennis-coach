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


def analyze_serve(args: argparse.Namespace) -> int:
    from serve_review.analyze_serve import (
        AnalyzeServeCancelled,
        AnalyzeServeError,
        run_analyze_serve,
    )

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2

    def _progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        result = run_analyze_serve(
            source,
            start_seconds=args.start_seconds,
            end_seconds=args.end_seconds,
            output_dir=args.output_dir,
            cache_path=args.cache,
            model_path=args.model,
            overwrite=args.overwrite,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            anchor2comparison=args.anchor2comparison,
            progress_callback=_progress,
        )
    except AnalyzeServeCancelled as exc:
        print(
            f"ERROR: analyze-serve was cancelled at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 1
    except AnalyzeServeError as exc:
        if exc.stage == "validate":
            print(f"ERROR: invalid serve range: {exc.message}.", file=sys.stderr)
            return 2
        print(
            f"ERROR: analyze-serve failed at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 1
    print(str(result.checkpoints_path))
    print(str(result.diagnostics_path))
    print(str(result.index_html))
    print(
        f"analyzed {result.attempt_phase.attempt_id} "
        f"[{result.attempt_range.start_seconds}, "
        f"{result.attempt_range.end_seconds}): "
        f"{result.image_count} image(s), "
        f"structural={result.attempt_phase.structural_status}.",
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


def phase_debug_cmd(args: argparse.Namespace) -> int:
    from serve_review.phase_debug_report import (
        PhaseDebugReportCollisionError,
        PhaseDebugReportError,
        PhaseDebugReportInputError,
        run_phase_debug_report,
    )

    grid = args.grid.expanduser()
    if not grid.is_file():
        print(f"ERROR: grid file does not exist: {grid}", file=sys.stderr)
        return 2
    evidence = args.evidence.expanduser()
    if not evidence.is_file():
        print(f"ERROR: evidence file does not exist: {evidence}", file=sys.stderr)
        return 2
    solver_result = args.solver_result.expanduser()
    if not solver_result.is_file():
        print(
            f"ERROR: solver-result file does not exist: {solver_result}",
            file=sys.stderr,
        )
        return 2
    solver_config = args.solver_config.expanduser()
    if not solver_config.is_file():
        print(
            f"ERROR: solver-config file does not exist: {solver_config}",
            file=sys.stderr,
        )
        return 2
    manual = args.manual.expanduser() if args.manual is not None else None
    if manual is not None and not manual.is_file():
        print(f"ERROR: manual file does not exist: {manual}", file=sys.stderr)
        return 2
    output = args.output.expanduser()
    html = args.html.expanduser() if args.html is not None else None
    try:
        result = run_phase_debug_report(
            grid,
            evidence,
            solver_result,
            solver_config,
            output,
            html,
            manual_path=manual,
            overwrite=args.overwrite,
        )
    except PhaseDebugReportInputError as exc:
        print(f"ERROR: phase-debug failed at {exc.stage}: {exc.message}.", file=sys.stderr)
        return 2
    except PhaseDebugReportCollisionError as exc:
        print(f"ERROR: {exc.message}", file=sys.stderr)
        return 1
    except PhaseDebugReportError as exc:
        print(f"ERROR: phase-debug failed at {exc.stage}: {exc.message}.", file=sys.stderr)
        return 1
    print(str(result.output_json))
    print(str(result.output_html))
    print(f"phase-debug {result.attempt_id}: wrote inspection report.")
    return 0


def phase_debug_dev_cmd(args: argparse.Namespace) -> int:
    from serve_review.phase_debug_dev import (
        PhaseDebugDevCollisionError,
        PhaseDebugDevError,
        PhaseDebugDevInputError,
        run_phase_debug_dev,
    )

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    attempts = args.attempts.expanduser()
    if not attempts.is_file():
        print(f"ERROR: attempts file does not exist: {attempts}", file=sys.stderr)
        return 2
    cache = args.cache.expanduser()
    if not cache.is_file():
        print(f"ERROR: cache file does not exist: {cache}", file=sys.stderr)
        return 2
    annotations = args.annotations.expanduser()
    if not annotations.is_file():
        print(
            f"ERROR: annotations file does not exist: {annotations}",
            file=sys.stderr,
        )
        return 2
    output_dir = args.output_dir.expanduser()

    def _progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        result = run_phase_debug_dev(
            source,
            attempts_path=attempts,
            cache_path=cache,
            annotations_path=annotations,
            attempt_id=args.attempt_id,
            output_dir=output_dir,
            overwrite=args.overwrite,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            progress_callback=_progress,
        )
    except PhaseDebugDevInputError as exc:
        print(
            f"ERROR: phase-debug-dev failed at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 2
    except PhaseDebugDevCollisionError as exc:
        print(f"ERROR: {exc.message}", file=sys.stderr)
        return 1
    except PhaseDebugDevError as exc:
        print(
            f"ERROR: phase-debug-dev failed at {exc.stage}: {exc.message}.",
            file=sys.stderr,
        )
        return 1
    print(str(result.output_json))
    print(str(result.output_html))
    print(f"phase-debug-dev {result.attempt_id}: wrote inspection report.")
    return 0


def extract_world_cmd(args: argparse.Namespace) -> int:
    from serve_review.pose import world_extract as world_extract_module

    source = args.video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    attempts = args.attempts.expanduser()
    if not attempts.is_file():
        print(f"ERROR: attempts file does not exist: {attempts}", file=sys.stderr)
        return 2
    cache_path = args.cache.expanduser()
    model_path = args.model.expanduser()

    def _progress(done: int, total: int, observation: object) -> None:
        print(f"extract-world: {done}/{total} frames", file=sys.stderr)

    try:
        result = world_extract_module.extract_attempt_world(
            source,
            attempts_path=attempts,
            attempt_id=args.attempt_id,
            cache_path=cache_path,
            model_path=model_path,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            overwrite=args.overwrite,
            no_resume=args.no_resume,
            progress_callback=_progress,
        )
    except world_extract_module.WorldExtractionCancelled as exc:
        print(f"ERROR: dense-world extraction was cancelled: {exc}.", file=sys.stderr)
        return 1
    except world_extract_module.WorldExtractionInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except world_extract_module.WorldExtractionCollisionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except world_extract_module.WorldExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if result.cache_hit:
        print(f"cache hit: {result.cache_path} ({result.frame_count} frames)")
    else:
        print(
            f"extracted {result.frame_count} frames "
            f"({result.inferred_frames} inferred, {result.cached_frames} cached) "
            f"for {result.attempt_id} "
            f"[{result.attempt_start_seconds}, {result.attempt_end_seconds}) "
            f"to {result.cache_path}"
        )
    print(str(result.cache_path))
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

    serve_parser = subparsers.add_parser(
        "analyze-serve",
        help="analyze one serve range through the 3D waveform path",
        description=(
            "Analyze one explicit serve video/range using the 3D "
            "kinematic path (native-PTS world track, segment-safe filter, "
            "3D waveforms with aligned raw-audio transient channels when "
            "present, six composite anchor candidates, DP chronology "
            "search, two derived midpoint stages) and atomically write "
            "checkpoints.json, a 3D diagnostic JSON, and a source-frame "
            "review page. Defaults to the entire source timeline; no "
            "attempts.json, attempt id, or phase JSON is required. A "
            "synthetic in-memory serve-001 range exists solely for output "
            "linkage. Raw-source audio is demuxed once and aligned to the "
            "exact kinematic PTS; demux/alignment failure or an absent "
            "stream is nonfatal and yields unavailable audio channels. "
            "Contact uses body_pose_audio provenance when the selected "
            "contact carries an available supporting audio cue, otherwise "
            "body_pose."
        ),
    )
    serve_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    serve_parser.add_argument(
        "--start-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="start_seconds",
        help="serve range start in source seconds (default: 0.0)",
    )
    serve_parser.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        dest="end_seconds",
        help="serve range end in source seconds (default: source duration)",
    )
    serve_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="generated output directory (default: output)",
    )
    serve_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "reusable kinematic-track cache file (default: "
            "<output-dir>/<source-stem>/cache/kinematic-track-v1.jsonl)"
        ),
    )
    serve_parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_heavy.task"),
        metavar="PATH",
        help="approved Pose Landmarker .task artifact (default: models/pose_landmarker_heavy.task)",
    )
    serve_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing outputs/cache (default: fail on collision)",
    )
    serve_parser.add_argument(
        "--anchor2comparison",
        "--anchor2-comparison",
        dest="anchor2comparison",
        action="store_true",
        default=False,
        help=(
            "compare against the fixed manual anchor for exactly "
            "refs/anchors/single-serve-02.mov "
            "(refs/annotations/dev/single-serve-02.anchor2.json)"
        ),
    )
    serve_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    serve_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    serve_parser.set_defaults(handler=analyze_serve)

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
            "Contact is labelled with provenance and source time. Never modifies "
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

    debug_parser = subparsers.add_parser(
        "phase-debug",
        help="render a private deterministic phase-debug inspection report",
        description=(
            "Read one PhaseFeatureGrid, PhaseEvidence, PhaseSolverResult, "
            "and PhaseSolverConfig JSON plus an optional manual stage-time "
            "mapping, invoke the accepted phase-debug builder/reconciliation, "
            "and atomically write deterministic phase-debug JSON plus a "
            "self-contained deterministic HTML inspection page. Private "
            "diagnostic presentation only: no media decoding, pose "
            "inference, frame rendering, or checkpoints mutation."
        ),
    )
    debug_parser.add_argument(
        "--grid",
        type=Path,
        required=True,
        metavar="GRID_JSON",
        help="PhaseFeatureGrid JSON file",
    )
    debug_parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        metavar="EVIDENCE_JSON",
        help="PhaseEvidence JSON file",
    )
    debug_parser.add_argument(
        "--solver-result",
        type=Path,
        required=True,
        metavar="RESULT_JSON",
        dest="solver_result",
        help="PhaseSolverResult JSON file",
    )
    debug_parser.add_argument(
        "--solver-config",
        type=Path,
        required=True,
        metavar="CONFIG_JSON",
        dest="solver_config",
        help="PhaseSolverConfig JSON file",
    )
    debug_parser.add_argument(
        "--manual",
        type=Path,
        default=None,
        metavar="MANUAL_JSON",
        help=(
            "optional manual stage-time mapping JSON (object of stage to "
            "seconds or null; missing keys read as null)"
        ),
    )
    debug_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="destination phase-debug JSON file",
    )
    debug_parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "destination inspection HTML file (default: sibling of --output "
            "with .html extension)"
        ),
    )
    debug_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing outputs (default: fail on collision)",
    )
    debug_parser.set_defaults(handler=phase_debug_cmd)

    dev_parser = subparsers.add_parser(
        "phase-debug-dev",
        help="render a private dev-only phase-debug report for one frozen attempt",
        description=(
            "Dev-only bridge for frozen private development attempts: "
            "validate explicit source video, attempts JSON, complete "
            "compatible pose cache, and dev phase-annotation manifest for "
            "one attempt id; run the exact current sparse M4 path "
            "(raw-source audio demux/alignment, build_phase_feature_grid, "
            "build_phase_evidence, solve_with_diagnostics) with default "
            "configs; serialize grid/evidence/solver-result/solver-config "
            "as provenance inputs; invoke the accepted phase-debug "
            "reporting to write deterministic phase-debug JSON plus a "
            "self-contained HTML page into a new explicit private output "
            "directory. Never modifies attempts.json, checkpoints.json, "
            "cache, clips, source, annotations, or model artifacts. No "
            "dense re-inference, solver/evidence/config tuning, held-out "
            "support, or media/frame rendering. Rejects non-dev manifests, "
            "attempt ambiguity, and unavailable/incompatible cache/input "
            "identities rather than guessing."
        ),
    )
    dev_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    dev_parser.add_argument(
        "--attempts",
        "--attempts-path",
        type=Path,
        required=True,
        metavar="ATTEMPTS_JSON",
        dest="attempts",
        help="explicit attempts JSON file",
    )
    dev_parser.add_argument(
        "--cache",
        "--cache-path",
        type=Path,
        required=True,
        metavar="CACHE_JSONL",
        dest="cache",
        help="explicit complete compatible pose cache file",
    )
    dev_parser.add_argument(
        "--annotations",
        "--manifest",
        "--annotations-path",
        "--manifest-path",
        type=Path,
        required=True,
        metavar="ANNOTATIONS_JSON",
        dest="annotations",
        help="explicit dev phase-annotation manifest JSON",
    )
    dev_parser.add_argument(
        "--attempt-id",
        "--attempt_id",
        type=str,
        required=True,
        metavar="ATTEMPT_ID",
        dest="attempt_id",
        help="attempt id to debug (for example serve-001)",
    )
    dev_parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--output",
        type=Path,
        required=True,
        metavar="OUTPUT_DIR",
        dest="output_dir",
        help="new explicit private output directory for this attempt",
    )
    dev_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing outputs (default: fail on collision)",
    )
    dev_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    dev_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    dev_parser.set_defaults(handler=phase_debug_dev_cmd)

    world_parser = subparsers.add_parser(
        "extract-world",
        help="extract dense native-frame world poses for one attempt",
        description=(
            "Private single-attempt diagnostic: decode every native source "
            "frame inside one accepted unpadded attempt range "
            "(detected_range, never padded effective_range), run the "
            "approved Heavy Pose Landmarker in serialized VIDEO mode, and "
            "write a resumable versioned dense-world-v1 cache with exact "
            "ffprobe PTS. Complete caches with matching identity and exact "
            "native times are reused without inference; interrupted caches "
            "resume. Never modifies the source video or the attempts file."
        ),
    )
    world_parser.add_argument("video", type=Path, help="source MOV/MP4 video")
    world_parser.add_argument(
        "--attempts",
        type=Path,
        required=True,
        metavar="ATTEMPTS_JSON",
        help="explicit attempts JSON file",
    )
    world_parser.add_argument(
        "--attempt-id",
        type=str,
        required=True,
        metavar="ATTEMPT_ID",
        dest="attempt_id",
        help="attempt id to extract (for example serve-001)",
    )
    world_parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_heavy.task"),
        metavar="PATH",
        help="approved Pose Landmarker .task artifact (default: models/pose_landmarker_heavy.task)",
    )
    world_parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        metavar="PATH",
        help="destination dense-world cache file (dense-world-v1 JSONL)",
    )
    world_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace any existing cache instead of hitting/resuming it",
    )
    world_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="fail on an existing incomplete cache instead of resuming it",
    )
    world_parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        metavar="EXE",
        help="ffmpeg executable (default: ffmpeg)",
    )
    world_parser.add_argument(
        "--ffprobe",
        default="ffprobe",
        metavar="EXE",
        help="ffprobe executable (default: ffprobe)",
    )
    world_parser.set_defaults(handler=extract_world_cmd)

    # Keep workflow commands isolated from this long-standing parser.
    from serve_review.workflow.cli import add_workflow_parsers
    add_workflow_parsers(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
