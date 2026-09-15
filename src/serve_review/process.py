"""Coordinate cutting and attempt analysis beside source videos."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from serve_review.analyze_serve import (
    DIAGNOSTICS_FILENAME,
    INDEX_HTML_FILENAME,
    REVIEW_DIRNAME,
    REVIEW_JSON_FILENAME,
    run_analyze_serve,
)
from serve_review.domain import AttemptDocument, SourceMetadata
from serve_review.media.probe import probe_source
from serve_review.pipeline import run_cut

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


def _analysis_complete(directory: Path) -> bool:
    review_dir = directory / REVIEW_DIRNAME
    checkpoints = _json_object(directory / "checkpoints.json")
    diagnostics = _json_object(directory / DIAGNOSTICS_FILENAME)
    review = _json_object(review_dir / REVIEW_JSON_FILENAME)
    try:
        html = (review_dir / INDEX_HTML_FILENAME).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if checkpoints is None or diagnostics is None or review is None or not html.strip():
        return False

    entries = review.get("entries")
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        for key in ("image", "manual_image"):
            image = entry.get(key)
            if image and not (review_dir / Path(str(image)).name).is_file():
                return False
    return True


def _clear(*directories: Path) -> None:
    for directory in directories:
        if directory.exists():
            shutil.rmtree(directory)


def _failure(video: Path, attempt: str | None, fallback: str, error: Exception) -> Failure:
    return Failure(video.name, attempt, str(getattr(error, "stage", fallback)), str(error))


def process(
    target: Path | str,
    *,
    force: bool = False,
    dry_run: bool = False,
    probe_fn: Callable[[Path], SourceMetadata] | None = None,
    cut_fn: Callable[..., Any] | None = None,
    analyze_fn: Callable[..., Any] | None = None,
) -> ProcessResult:
    """Process one video or each immediate video in its directory."""
    videos = discover(target)
    probe = probe_fn or probe_source
    cut = cut_fn or run_cut
    analyze = analyze_fn or run_analyze_serve
    actions: list[Action] = []
    failures: list[Failure] = []
    current_sources: dict[Path, SourceMetadata] = {}

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
    for video in videos:
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
        except Exception as error:
            failures.append(_failure(video, None, "cut", error))

    for video, attempts in documents.items():
        metadata_dir, _ = generated_paths(video)
        for attempt in attempts.attempts:
            destination = metadata_dir / "attempts" / attempt.attempt_id
            complete = _analysis_complete(destination)
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
            try:
                _clear(destination)
                analyze(
                    video,
                    start_seconds=attempt.detected_range.start_seconds,
                    end_seconds=attempt.detected_range.end_seconds,
                    output_dir=destination,
                    force=False,
                )
            except Exception as error:
                failures.append(
                    _failure(video, attempt.attempt_id, "analysis", error)
                )

    return ProcessResult(videos, tuple(actions), tuple(failures))
