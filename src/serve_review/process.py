"""Coordinate cutting and attempt analysis beside source videos."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from serve_review.analysis_artifacts import AttemptAnalysisArtifacts
from serve_review.analyze_serve import run_analyze_serve
from serve_review.domain import AttemptDocument, SourceMetadata
from serve_review.media.probe import probe_source
from serve_review.pipeline import run_cut
from serve_review.tracking.racketvision import RacketVisionConfig, RacketVisionTracker

_VIDEO_SUFFIXES = {".mov", ".mp4"}


@dataclass(frozen=True, slots=True)
class Failure:
    filename: str
    attempt: str | None
    step: str
    error: str


@dataclass(frozen=True, slots=True)
class Action:
    filename: str
    attempt: str | None
    action: str


@dataclass(frozen=True, slots=True)
class ProcessResult:
    sources: tuple[Path, ...]
    actions: tuple[Action, ...]
    failures: tuple[Failure, ...]
    summary_path: Path | None = None
    complete_sources: int = 0
    complete_attempts: int = 0


class ProcessError(ValueError):
    """The target or its owned artifacts cannot be processed safely."""


class FingerprintMismatch(ProcessError):
    """A source path now identifies different media."""


def discover(target: Path | str) -> tuple[Path, ...]:
    path = Path(target).expanduser()
    if path.is_file():
        if path.suffix.lower() not in _VIDEO_SUFFIXES:
            raise ProcessError(f"unsupported video target: {path}")
        return (path,)
    if not path.is_dir():
        raise ProcessError(f"target does not exist or is not a directory: {path}")

    videos = tuple(
        sorted(
            (
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in _VIDEO_SUFFIXES
            ),
            key=lambda child: (child.name.casefold(), child.name),
        )
    )
    if not videos:
        raise ProcessError(f"directory contains no MOV or MP4 videos: {path}")
    stems = [video.stem.casefold() for video in videos]
    if len(stems) != len(set(stems)):
        raise ProcessError("duplicate video filename stems; rename the sources")
    return videos


def generated_paths(video: Path) -> tuple[Path, Path]:
    return (
        video.parent / "metadata" / video.stem,
        video.parent / "exports" / video.stem,
    )


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def _source_metadata(path: Path) -> SourceMetadata | None:
    value = _json_object(path)
    if value is None:
        return None
    try:
        return SourceMetadata.from_dict(value)
    except (TypeError, ValueError):
        return None


def _attempt_document(path: Path) -> AttemptDocument | None:
    try:
        return AttemptDocument.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None


def _cut_complete(
    metadata_dir: Path, export_dir: Path
) -> tuple[bool, AttemptDocument | None]:
    source = _source_metadata(metadata_dir / "source.json")
    attempts = _attempt_document(metadata_dir / "attempts.json")
    run = _json_object(metadata_dir / "run.json")
    complete = (
        source is not None
        and attempts is not None
        and run is not None
        and run.get("status") in {"ok", "empty"}
        and (not attempts.attempts or (export_dir / "serves.mov").is_file())
    )
    return complete, attempts


def _clear(*directories: Path) -> None:
    for directory in directories:
        if directory.is_dir():
            shutil.rmtree(directory)
        elif directory.exists():
            directory.unlink()


def _failure(video: Path, attempt: str | None, fallback: str, error: Exception) -> Failure:
    step = str(getattr(error, "stage", fallback))
    detail = str(getattr(error, "message", error))
    return Failure(video.name, attempt, step, detail)


def _format_moment(value: float) -> str:
    total = max(0.0, float(value))
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def _format_range(start: float, end: float) -> str:
    return f"{_format_moment(start)}\u2013{_format_moment(end)}"


def _quote(name: str) -> str:
    return quote(name, safe="/")


def _source_status(
    video: Path,
    attempts: AttemptDocument | None,
    source_failures: list[Failure],
    attempt_failures: dict[str, list[Failure]],
) -> str:
    if source_failures or attempts is None:
        return "failed"
    if any(attempt_failures.values()):
        return "failed"
    if not attempts.attempts:
        return "empty"
    metadata_dir, _ = generated_paths(video)
    for attempt in attempts.attempts:
        if not AttemptAnalysisArtifacts(
            metadata_dir / "attempts" / attempt.attempt_id
        ).is_complete():
            return "incomplete"
    return "complete"


def _render_index(
    recording_dir: Path,
    videos: tuple[Path, ...],
    documents: dict[Path, AttemptDocument],
    failures: tuple[Failure, ...],
) -> str:
    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head><meta charset=\"utf-8\">",
        f"<title>{html.escape(recording_dir.name)} \u2014 serve review</title>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(recording_dir.name)}</h1>",
    ]
    by_source: dict[str, list[Failure]] = {}
    for failure in failures:
        by_source.setdefault(failure.filename, []).append(failure)
    for video in videos:
        metadata_dir, export_dir = generated_paths(video)
        attempts = documents.get(video)
        if attempts is None:
            attempts = _attempt_document(metadata_dir / "attempts.json")
        source_failures = [
            item for item in by_source.get(video.name, []) if item.attempt is None
        ]
        attempt_failures: dict[str, list[Failure]] = {}
        for item in by_source.get(video.name, []):
            if item.attempt is not None:
                attempt_failures.setdefault(item.attempt, []).append(item)
        status = _source_status(video, attempts, source_failures, attempt_failures)
        count = len(attempts.attempts) if attempts is not None else 0
        lines.append("<section>")
        lines.append(
            f"<h2>{html.escape(video.name)} \u2014 {count} serves \u2014 {status}</h2>"
        )
        source_link = _quote("../" + video.name)
        lines.append(f"<p><a href=\"{source_link}\">[Source]</a>")
        lines[-1] += "</p>"
        lines.append(
            f"<details><summary>Play source</summary><video controls "
            f"preload=\"metadata\" src=\"{source_link}\"></video></details>"
        )
        if (export_dir / "serves.mov").is_file():
            link = _quote(f"../exports/{video.stem}/serves.mov")
            lines.append(f"<p><a href=\"{link}\">[Compilation]</a></p>")
            lines.append(
                f"<details><summary>Play compilation</summary><video controls "
                f"preload=\"metadata\" src=\"{link}\"></video></details>"
            )
        if attempts is not None:
            lines.append("<ul>")
            for attempt in attempts.attempts:
                destination = metadata_dir / "attempts" / attempt.attempt_id
                artifacts = AttemptAnalysisArtifacts(destination)
                done = artifacts.is_complete()
                span = _format_range(
                    attempt.detected_range.start_seconds,
                    attempt.detected_range.end_seconds,
                )
                row = (
                    f"<li>{html.escape(attempt.attempt_id)} \u2014 "
                    f"{html.escape(span)} \u2014 "
                )
                if done:
                    link = _quote(
                        f"{video.stem}/attempts/{attempt.attempt_id}/"
                        f"{artifacts.review_dir.name}/{artifacts.index_html_path.name}"
                    )
                    row += f"<a href=\"{link}\">[Checkpoint review]</a>"
                else:
                    details = attempt_failures.get(attempt.attempt_id, [])
                    if details:
                        row += "failed: " + "; ".join(
                            html.escape(f"{item.step}: {item.error}")
                            for item in details
                        )
                    else:
                        row += "incomplete"
                row += "</li>"
                lines.append(row)
            lines.append("</ul>")
        for item in source_failures:
            lines.append(
                f"<p>failed: {html.escape(item.step)}: {html.escape(item.error)}</p>"
            )
        lines.append("</section>")
    lines.extend(["</body>", "</html>", ""])
    return "\n".join(lines)


def _write_index(
    recording_dir: Path,
    videos: tuple[Path, ...],
    documents: dict[Path, AttemptDocument],
    failures: tuple[Failure, ...],
) -> Path:
    target = recording_dir / "metadata" / "index.html"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _render_index(recording_dir, videos, documents, failures), encoding="utf-8"
        )
    except OSError as error:
        raise ProcessError(f"could not write summary {target}: {error}") from error
    return target


def process(
    target: Path | str,
    *,
    force: bool = False,
    dry_run: bool = False,
    probe_fn: Callable[[Path], SourceMetadata] | None = None,
    cut_fn: Callable[..., Any] | None = None,
    analyze_fn: Callable[..., Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ProcessResult:
    """Process one video or each immediate video in its directory."""
    videos = discover(target)
    probe = probe_fn or probe_source
    cut = cut_fn or run_cut
    analyze = analyze_fn or run_analyze_serve
    actions: list[Action] = []
    failures: list[Failure] = []
    current_sources: dict[Path, SourceMetadata] = {}
    tracker: RacketVisionTracker | None = None

    def racketvision_tracker_factory(config: RacketVisionConfig) -> RacketVisionTracker:
        nonlocal tracker
        if tracker is None:
            tracker = RacketVisionTracker(config)
        return tracker

    def _emit(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    total = len(videos)
    for video in videos:
        try:
            current_sources[video] = probe(video)
            metadata_dir, _ = generated_paths(video)
            stored = _source_metadata(metadata_dir / "source.json")
            if stored is not None and stored.fingerprint != current_sources[video].fingerprint:
                if not force:
                    raise FingerprintMismatch(
                        f"{video.name}: source fingerprint mismatch; pass --force"
                    )
        except FingerprintMismatch:
            raise
        except Exception as error:
            failures.append(_failure(video, None, "probe", error))

    documents: dict[Path, AttemptDocument] = {}
    for index, video in enumerate(videos, start=1):
        _emit(f"[{index}/{total} videos] {video.name}")
        if video not in current_sources:
            continue
        metadata_dir, export_dir = generated_paths(video)
        complete, attempts = _cut_complete(metadata_dir, export_dir)
        needs_cut = force or not complete
        action = "skip"
        if force:
            action = "clear"
        elif needs_cut:
            action = "clear" if metadata_dir.exists() or export_dir.exists() else "run"
        actions.append(Action(video.name, None, action))

        if dry_run:
            if complete and not force and attempts is not None:
                documents[video] = attempts
            continue
        try:
            if needs_cut:
                if not dry_run:
                    _emit("  detecting serves\u2026")
                _clear(metadata_dir, export_dir)
                result = cut(
                    video,
                    metadata_dir=metadata_dir,
                    export_dir=export_dir,
                    padding_seconds=1,
                    mode="compilation",
                    force=False,
                )
                attempts = getattr(result, "attempts_document", None)
                if attempts is None:
                    attempts = _attempt_document(metadata_dir / "attempts.json")
            if attempts is None:
                raise ProcessError("cut did not produce a readable attempts document")
            documents[video] = attempts
            if needs_cut and not dry_run and attempts is not None:
                _emit(f"  found {len(attempts.attempts)} serves")
        except Exception as error:
            failures.append(_failure(video, None, "cut", error))

    for video, attempts in documents.items():
        metadata_dir, _ = generated_paths(video)
        total_attempts = len(attempts.attempts)
        for position, attempt in enumerate(attempts.attempts, start=1):
            destination = metadata_dir / "attempts" / attempt.attempt_id
            complete = AttemptAnalysisArtifacts(destination).is_complete()
            if complete:
                actions.append(Action(video.name, attempt.attempt_id, "skip"))
                continue
            actions.append(
                Action(
                    video.name,
                    attempt.attempt_id,
                    "clear" if destination.exists() else "run",
                )
            )
            if dry_run:
                continue
            _emit(
                f"  [{video.name} {position}/{total_attempts} attempts] analyzing "
                f"{_format_range(attempt.detected_range.start_seconds, attempt.detected_range.end_seconds)}\u2026"
            )
            try:
                _clear(destination)
                analyze(
                    video,
                    start_seconds=attempt.detected_range.start_seconds,
                    end_seconds=attempt.detected_range.end_seconds,
                    output_dir=destination,
                    force=force,
                    racketvision_tracker_factory=racketvision_tracker_factory,
                )
            except Exception as error:
                failures.append(
                    _failure(video, attempt.attempt_id, "analysis", error)
                )

    if dry_run:
        return ProcessResult(videos, tuple(actions), tuple(failures))
    recording_dir = videos[0].parent
    summary_videos = discover(recording_dir)
    summary_path = _write_index(
        recording_dir, summary_videos, documents, tuple(failures)
    )
    complete_sources = 0
    complete_attempts = 0
    failed_names = {item.filename for item in failures}
    for video in videos:
        if video.name not in failed_names:
            metadata_dir, _ = generated_paths(video)
            known = documents.get(video) or _attempt_document(
                metadata_dir / "attempts.json"
            )
            if known is not None:
                complete_sources += 1
                for attempt in known.attempts:
                    if AttemptAnalysisArtifacts(
                        metadata_dir / "attempts" / attempt.attempt_id
                    ).is_complete():
                        complete_attempts += 1
    return ProcessResult(
        videos,
        tuple(actions),
        tuple(failures),
        summary_path=summary_path,
        complete_sources=complete_sources,
        complete_attempts=complete_attempts,
    )
