"""Single-source compatibility orchestration.

This module is intentionally a thin adapter: all expensive work is delegated to
injected (or existing) Python services and every output is source/attempt scoped.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Any

from serve_review.domain import MediaRange
from serve_review.workflow.records import (
    AttemptOutcome, SourceRecord, WorkflowRecordError, WorkflowRunRecord,
    RUN_RECORD_SCHEMA_VERSION, attempt_id_for,
)
from serve_review.workflow.workspace import WorkspacePaths, write_json_atomic


def _default_cut(*args: Any, **kwargs: Any) -> Any:
    from serve_review.pipeline import run_cut
    return run_cut(*args, **kwargs)


def _default_analyze(*args: Any, **kwargs: Any) -> Any:
    from serve_review.analyze_serve import run_analyze_serve
    return run_analyze_serve(*args, **kwargs)


def _error(value: BaseException) -> str:
    return f"{type(value).__name__}: {value}"


def _artifact_paths(result: Any, directory: Path) -> tuple[str, ...]:
    names = ("checkpoints_path", "diagnostics_path", "review_dir", "index_html", "review_json")
    paths: list[str] = []
    root = directory.resolve()
    for name in names:
        value = getattr(result, name, None)
        if value is None:
            continue
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkflowRecordError(
                f"collaborator artifact {name} is missing or outside attempt output."
            ) from exc
        paths.append(str(resolved))
    if len(paths) != len(names):
        raise WorkflowRecordError("analyze result is missing required artifact paths.")
    return tuple(sorted(paths))


def process_registered_source(
    source: SourceRecord,
    workspace: WorkspacePaths,
    mode: str,
    *,
    run_id_factory: Callable[[], str],
    producer_method: str,
    producer_config: str,
    explicit_range: MediaRange | None = None,
    padding_seconds: float | None = None,
    run_cut_fn: Callable[..., Any] = _default_cut,
    run_analyze_fn: Callable[..., Any] = _default_analyze,
) -> WorkflowRunRecord:
    """Process exactly one registered source and persist its run provenance."""
    if type(source) is not SourceRecord:
        raise TypeError("source must be exactly a SourceRecord")
    if type(workspace) is not WorkspacePaths:
        raise TypeError("workspace must be exactly a WorkspacePaths")
    if mode not in {"normal", "single-attempt", "explicit-range"}:
        raise ValueError("mode must be normal, single-attempt, or explicit-range")
    if not isinstance(producer_method, str) or not producer_method.strip() or not isinstance(producer_config, str) or not producer_config.strip():
        raise ValueError("producer method and config identities must be nonblank")
    if mode == "explicit-range" and not isinstance(explicit_range, MediaRange):
        raise ValueError("explicit-range mode requires explicit_range")
    if mode != "explicit-range" and explicit_range is not None:
        raise ValueError("explicit_range is only valid in explicit-range mode")
    if padding_seconds is not None and (
        isinstance(padding_seconds, bool)
        or not isinstance(padding_seconds, (int, float))
        or not math.isfinite(padding_seconds)
        or padding_seconds < 0
    ):
        raise ValueError("padding_seconds must be finite and nonnegative")

    try:
        run_id = run_id_factory()
    except Exception as exc:
        raise WorkflowRecordError(f"could not create run ID: {exc}") from exc
    if not isinstance(run_id, str) or not run_id.startswith("run-") or len(run_id) != 20 or any(c not in "0123456789abcdef" for c in run_id[4:]):
        raise WorkflowRecordError("run_id_factory returned an invalid run ID")
    try:
        runs_root = workspace.runs.resolve()
        run_path = (workspace.runs / f"{run_id}.json").resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise WorkflowRecordError(f"could not resolve run path: {exc}") from exc
    if run_path.parent != runs_root:
        raise WorkflowRecordError("run record path escapes workspace runs")
    if os.path.lexists(run_path):
        raise WorkflowRecordError(f"run ID collision: {run_id}")
    existing = sorted(
        (Path(path).resolve() for path in source.known_paths if Path(path).is_file()),
        key=str,
    )
    if not existing:
        raise FileNotFoundError("source unavailable: no known path is a regular file")
    video = existing[0]
    source_dir = workspace.sources / source.source_id
    compatibility_dir = source_dir / "compatibility"
    detection_status, detection_error = "complete", None
    ranges: list[MediaRange] = []
    try:
        if mode == "normal":
            cut_kwargs = {"output_dir": compatibility_dir, "mode": "both", "overwrite": False}
            if padding_seconds is not None:
                cut_kwargs["padding_seconds"] = padding_seconds
            cut = run_cut_fn(video, **cut_kwargs)
            metadata = getattr(cut, "source_metadata", None)
            document = getattr(cut, "attempts_document", None)
            if metadata is None or getattr(metadata, "fingerprint", None) != source.source_fingerprint:
                raise WorkflowRecordError("cut result source fingerprint mismatch")
            if getattr(document, "source_fingerprint", None) != source.source_fingerprint:
                raise WorkflowRecordError("cut attempts source fingerprint mismatch")
            for attempt in getattr(document, "attempts", ()):
                ranges.append(attempt.detected_range)
            if not ranges:
                detection_status = "empty"
        elif mode == "single-attempt":
            ranges = [MediaRange(0.0, source.metadata.duration_seconds)]
        else:
            ranges = [explicit_range]  # type: ignore[list-item]
            if ranges[0].end_seconds > source.metadata.duration_seconds:
                raise WorkflowRecordError("explicit range exceeds source duration")
    except Exception as exc:
        detection_status, detection_error = "failed", _error(exc)

    stable_ids: list[str] = []
    if detection_status == "complete":
        try:
            stable_ids = [
                attempt_id_for(
                    source.source_id,
                    producer_method,
                    producer_config,
                    media_range.start_seconds,
                    media_range.end_seconds,
                )
                for media_range in ranges
            ]
            if len(stable_ids) != len(set(stable_ids)):
                raise WorkflowRecordError("stable attempt ID collision")
        except WorkflowRecordError as exc:
            detection_status = "failed"
            detection_error = _error(exc)
            stable_ids = []

    outcomes: list[AttemptOutcome] = []
    if detection_status == "complete":
        for media_range, attempt_id in zip(ranges, stable_ids):
            try:
                attempt_dir = source_dir / "attempts" / attempt_id
                cache_path = attempt_dir / "cache" / "kinematic-track-v1.jsonl"
                result = run_analyze_fn(video, start_seconds=media_range.start_seconds, end_seconds=media_range.end_seconds, output_dir=attempt_dir, cache_path=cache_path, overwrite=False)
                metadata = getattr(result, "source_metadata", None)
                if getattr(metadata, "fingerprint", None) != source.source_fingerprint:
                    raise WorkflowRecordError("analyze result source fingerprint mismatch")
                for attr, expected in (("requested_range", media_range), ("attempt_range", media_range)):
                    got = getattr(result, attr, None)
                    if not isinstance(got, MediaRange) or got.start_seconds != expected.start_seconds or got.end_seconds != expected.end_seconds:
                        raise WorkflowRecordError(f"analyze result {attr} mismatch")
                artifacts = _artifact_paths(result, attempt_dir)
                outcomes.append(AttemptOutcome(attempt_id, media_range.start_seconds, media_range.end_seconds, "complete", None, artifacts))
            except Exception as exc:
                outcomes.append(AttemptOutcome(attempt_id, media_range.start_seconds, media_range.end_seconds, "failed", _error(exc), ()))

    if detection_status == "failed":
        overall = "failed"
    elif not outcomes:
        overall = "empty"
    elif all(a.status == "failed" for a in outcomes):
        overall = "failed"
    elif any(a.status == "failed" for a in outcomes):
        overall = "partial"
    else:
        overall = "complete"
    record = WorkflowRunRecord(RUN_RECORD_SCHEMA_VERSION, run_id, source.source_id, source.source_fingerprint, mode, producer_method, producer_config, detection_status, detection_error, tuple(outcomes), overall)
    write_json_atomic(run_path, record.to_dict())
    return record
