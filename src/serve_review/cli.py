from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Sequence


@contextmanager
def _silence_model_output():
    """Hide noisy in-process/native model output during the primary workflow."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


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
            metadata_dir=args.metadata_dir,
            export_dir=args.export_dir,
            padding_seconds=padding,
            mode=args.output,
            dry_run=args.dry_run,
            force=args.force,
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
    if result is None:
        print("cut dry-run: validated request; no writes performed.")
        return 0
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
            dry_run=args.dry_run,
            force=args.force,
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
    if result is None:
        print("analyze-serve dry-run: validated request; no writes performed.")
        return 0
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


def process_cmd(args: argparse.Namespace) -> int:
    from serve_review.process import FingerprintMismatch, ProcessError
    from serve_review.process import process as run_process

    target = args.target.expanduser()

    progress_stream = os.fdopen(os.dup(2), "w", encoding="utf-8")

    def _progress(message: str) -> None:
        print(message, file=progress_stream, flush=True)

    try:
        with progress_stream, _silence_model_output():
            result = run_process(
                target,
                force=args.force,
                dry_run=args.dry_run,
                progress_callback=_progress,
            )
    except FingerprintMismatch as exc:
        print(f"ERROR: {exc}.", file=sys.stderr)
        return 2
    except ProcessError as exc:
        print(f"ERROR: {exc}.", file=sys.stderr)
        return 2
    if args.dry_run:
        for action in result.actions:
            detail = action.filename
            if action.attempt is not None:
                detail += f" {action.attempt}"
            print(f"dry-run {action.action}: {detail}")
        for failure in result.failures:
            detail = failure.filename
            if failure.attempt is not None:
                detail += f" {failure.attempt}"
            print(f"dry-run failed: {detail}: {failure.step}: {failure.error}")
        return 1 if result.failures else 0
    print(f"Complete: {result.complete_sources} videos, {result.complete_attempts} attempts")
    if result.failures:
        print(f"Failed: {len(result.failures)} failure(s)")
        for failure in result.failures:
            detail = failure.filename
            if failure.attempt is not None:
                detail += f" {failure.attempt}"
            print(f"failed: {detail}: {failure.step}: {failure.error}")
    if result.summary_path is not None:
        print(f"Review: {result.summary_path.resolve().as_uri()}")
    return 1 if result.failures else 0


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
        "--metadata-dir",
        type=Path,
        default=None,
        metavar="PATH",
        dest="metadata_dir",
        help=(
            "exact metadata/cache destination override "
            "(default: VIDEO.parent/metadata/VIDEO.stem)"
        ),
    )
    cut_parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        metavar="PATH",
        dest="export_dir",
        help=(
            "exact media destination override "
            "(default: VIDEO.parent/exports/VIDEO.stem)"
        ),
    )
    cut_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="validate and report destinations without writes (default: off)",
    )
    cut_parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="replace only this command's exact destinations (default: fail on collision)",
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
        default=None,
        metavar="PATH",
        dest="output_dir",
        help=(
            "exact output destination override "
            "(default: VIDEO.parent/metadata/VIDEO.stem/manual-analysis)"
        ),
    )
    serve_parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "reusable kinematic-track cache file (default: "
            "<exact-output>/cache/kinematic-track-v1.jsonl)"
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
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="validate and report destinations without writes (default: off)",
    )
    serve_parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="replace only this command's exact destination (default: fail on collision)",
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

    process_parser = subparsers.add_parser(
        "process",
        help="process a recording directory or single video beside its sources",
    )
    process_parser.add_argument("target", type=Path, help="recording directory or MOV/MP4 video")
    process_parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="report planned skip/clear/run actions without writes (default: off)",
    )
    process_parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="replace selected generated trees before processing (default: off)",
    )
    process_parser.set_defaults(handler=process_cmd)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
