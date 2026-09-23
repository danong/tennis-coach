from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Sequence


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


def _comparison_policy(policy_id: str, capture_context: str) -> Any:
    from serve_review.comparison import ComparisonPolicy

    return ComparisonPolicy(policy_id=policy_id, capture_context=capture_context)


def compare_pairwise_cmd(
    *,
    candidate: Path,
    reference: Path,
    policy_id: str,
    capture_context: str,
) -> int:
    """Emit one explicit ServeComparisonV1 pairwise result as JSON."""

    from serve_review.comparison import (
        ComparisonError,
        compare_pairwise,
        load_fingerprint,
    )

    try:
        result = compare_pairwise(
            load_fingerprint(candidate),
            load_fingerprint(reference),
            policy=_comparison_policy(policy_id, capture_context),
        )
    except (ComparisonError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(result.to_json(), end="")
    return 0


def compare_baseline_cmd(
    *,
    candidate: Path,
    cohort: Sequence[Path],
    cohort_id: str,
    policy_id: str,
    capture_context: str,
) -> int:
    """Emit one explicit ServeComparisonV1 baseline result as JSON."""

    from serve_review.comparison import (
        ComparisonError,
        compare_to_baseline,
        load_fingerprint,
        load_fingerprints,
    )

    try:
        result = compare_to_baseline(
            load_fingerprint(candidate),
            load_fingerprints(cohort),
            cohort_id=cohort_id,
            policy=_comparison_policy(policy_id, capture_context),
        )
    except (ComparisonError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(result.to_json(), end="")
    return 0


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


def doctor(*, device: str = "auto") -> int:
    failures: list[str] = []
    print(f"Python {sys.version.split()[0]} ({sys.executable})")

    from serve_review.tracking.racketvision import (
        RacketVisionError,
        racketvision_cuda_available,
        resolve_racketvision_device,
    )

    cuda_available = racketvision_cuda_available()
    print(f"RacketVision CUDA: {'available' if cuda_available else 'unavailable'}")
    try:
        selected_device = resolve_racketvision_device(device)
        print(f"RacketVision device: {selected_device}")
    except RacketVisionError as exc:
        failures.append(str(exc))

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


def probe(*, video: Path, output: Path | None, ffprobe: str) -> int:
    from serve_review.media.probe import ProbeError, probe_source, write_source_json

    source = video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    if output is not None:
        dest = output.expanduser()
    else:
        dest = Path("output") / source.stem / "source.json"
    try:
        metadata = probe_source(source, ffprobe=ffprobe)
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


def export_cmd(*, video: Path, ranges: Path, output: str, output_dir: Path, overwrite: bool, ffmpeg: str, ffprobe: str) -> int:
    import json

    from serve_review.domain import DomainError, ExportPlan, MediaRange
    from serve_review.media import export as export_module
    from serve_review.media.probe import ProbeError, probe_source

    source = video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    ranges_path = ranges.expanduser()
    if not ranges_path.is_file():
        print(f"ERROR: ranges file does not exist: {ranges_path}", file=sys.stderr)
        return 2
    try:
        metadata = probe_source(source, ffprobe=ffprobe)
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
    base_dir = output_dir.expanduser() / source.stem
    try:
        results = export_module.export_outputs(
            source,
            plan,
            base_dir,
            mode=output,
            source=metadata,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            overwrite=overwrite,
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


def extract_poses_cmd(*, video: Path, model: Path, sample_rate: float, cache: Path | None, overlay: Path | None, output_dir: Path, overwrite: bool, ffmpeg: str, ffprobe: str) -> int:
    from serve_review.pose import extract as extract_module

    source = video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    try:
        rate = float(sample_rate)
    except (TypeError, ValueError):
        print(
            f"ERROR: invalid --sample-rate {sample_rate!r}; expected a number in (0, 120].",
            file=sys.stderr,
        )
        return 2
    if not (0 < rate <= 120.0):
        print(
            f"ERROR: invalid --sample-rate {sample_rate!r}; expected a number in (0, 120].",
            file=sys.stderr,
        )
        return 2
    if cache is not None:
        cache_path = cache.expanduser()
    else:
        cache_path = extract_module.default_cache_path_for(source, output_dir)
    model_path = model.expanduser()
    overlay = overlay.expanduser() if overlay is not None else None

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
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            overwrite=overwrite,
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


def cut(*, video: Path, padding: float, output: str, metadata_dir: Path | None, export_dir: Path | None, dry_run: bool, force: bool, ffmpeg: str, ffprobe: str) -> int:
    from serve_review.pipeline import CutCancelled, CutError, run_cut

    source = video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2
    try:
        padding = float(padding)
    except (TypeError, ValueError):
        print(
            f"ERROR: invalid --padding {padding!r}; "
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
            metadata_dir=metadata_dir,
            export_dir=export_dir,
            padding_seconds=padding,
            mode=output,
            dry_run=dry_run,
            force=force,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
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


def analyze_serve(*, video: Path, start_seconds: float | None, end_seconds: float | None, output_dir: Path | None, cache: Path | None, model: Path, device: str = "auto", dry_run: bool, force: bool, ffmpeg: str, ffprobe: str, anchor2comparison: bool = False) -> int:
    from serve_review.analyze_serve import (
        AnalyzeServeCancelled,
        AnalyzeServeError,
        run_analyze_serve,
    )
    from serve_review.tracking.racketvision import RacketVisionConfig

    source = video.expanduser()
    if not source.is_file():
        print(f"ERROR: input video does not exist: {source}", file=sys.stderr)
        return 2

    def _progress(message: str) -> None:
        print(message, file=sys.stderr)

    try:
        result = run_analyze_serve(
            source,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            output_dir=output_dir,
            cache_path=cache,
            racketvision_config=RacketVisionConfig(device=device),
            model_path=model,
            dry_run=dry_run,
            force=force,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            anchor2comparison=anchor2comparison,
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


def process_cmd(
    *, target: Path, device: str = "auto", cut_workers: int = 1, dry_run: bool, force: bool
) -> int:
    from serve_review.process import FingerprintMismatch, ProcessError
    from serve_review.process import process as run_process

    target = target.expanduser()

    progress_stream = os.fdopen(os.dup(2), "w", encoding="utf-8")

    def _progress(message: str) -> None:
        print(message, file=progress_stream, flush=True)

    try:
        with progress_stream, _silence_model_output():
            result = run_process(
                target,
                device=device,
                force=force,
                dry_run=dry_run,
                cut_workers=cut_workers,
                progress_callback=_progress,
            )
    except FingerprintMismatch as exc:
        print(f"ERROR: {exc}.", file=sys.stderr)
        return 2
    except ProcessError as exc:
        print(f"ERROR: {exc}.", file=sys.stderr)
        return 2
    if dry_run:
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
