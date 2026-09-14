"""Read-only projection over workflow records and compatibility artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from serve_review.domain import AttemptDocument, PhaseDocument
from serve_review.media.probe import fingerprint_for_path
from serve_review.workflow.collections import list_collections, resolve_collection
from serve_review.workflow.records import (
    SourceRecord,
    SourceStatus,
    StatusDocument,
    StatusValue,
    WorkflowRunRecord,
)
from serve_review.workflow.sessions import list_sessions, resolve_session
from serve_review.workflow.sources import load_source, source_record_path
from serve_review.workflow.workspace import WorkspacePaths

_SOURCE_ID = re.compile(r"^source-[0-9a-f]{16}$")
_RUN_FILE = re.compile(r"^(run-[0-9a-f]{16})\.json$")


def _contained(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _load_sources(workspace: WorkspacePaths) -> tuple[list[SourceRecord], list[str]]:
    root = workspace.sources
    if not root.exists():
        return [], []
    if not root.is_dir():
        return [], [f"sources path is not a directory: {root}"]
    records: list[SourceRecord] = []
    errors: list[str] = []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        return [], [f"cannot inspect sources: {exc}"]
    for entry in entries:
        if (
            _SOURCE_ID.fullmatch(entry.name) is None
            or not entry.is_dir()
            or not _contained(entry, root)
        ):
            errors.append(f"malformed source entry: {entry}")
            continue
        try:
            record = load_source(workspace, entry.name)
        except Exception as exc:
            errors.append(f"malformed source manifest {entry}: {exc}")
            continue
        if record.source_id != entry.name:
            errors.append(
                f"source manifest ID {record.source_id} does not match {entry.name}"
            )
            continue
        records.append(record)
    return records, errors


def _runs(
    workspace: WorkspacePaths, source: SourceRecord
) -> tuple[list[WorkflowRunRecord], list[str]]:
    root = workspace.runs
    if not root.exists():
        return [], []
    if not root.is_dir():
        return [], [f"runs path is not a directory: {root}"]
    records: list[WorkflowRunRecord] = []
    errors: list[str] = []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as exc:
        return [], [f"cannot inspect runs: {exc}"]
    for entry in entries:
        match = _RUN_FILE.fullmatch(entry.name)
        if match is None or not entry.is_file() or not _contained(entry, root):
            errors.append(f"malformed run entry: {entry}")
            continue
        try:
            record = WorkflowRunRecord.from_json(entry.read_bytes())
        except Exception as exc:
            errors.append(f"malformed run manifest {entry}: {exc}")
            continue
        if record.run_id != match.group(1):
            errors.append(
                f"run manifest ID {record.run_id} does not match {entry.name}"
            )
            continue
        if record.source_id == source.source_id:
            if record.source_fingerprint != source.source_fingerprint:
                errors.append(f"run {entry} fingerprint does not match source")
            else:
                records.append(record)
    return records, errors


def _source_availability(source: SourceRecord) -> StatusValue:
    existing: list[Path] = []
    for value in source.known_paths:
        path = Path(value)
        try:
            if path.is_file():
                existing.append(path)
        except OSError:
            continue
    if not existing:
        return StatusValue(
            "missing", "none of the registered source paths is present"
        )
    try:
        mismatched = [
            str(path)
            for path in existing
            if fingerprint_for_path(path) != source.source_fingerprint
        ]
    except Exception as exc:
        return StatusValue(
            "unavailable", f"cannot fingerprint registered source: {exc}"
        )
    if mismatched:
        return StatusValue(
            "stale", "present source fingerprint does not match registry"
        )
    return StatusValue("present", value=len(existing))


def _compatibility(
    workspace: WorkspacePaths, source: SourceRecord
) -> tuple[StatusValue, AttemptDocument | None]:
    source_root = workspace.sources / source.source_id
    root = source_root / "compatibility"
    if not _contained(root, source_root):
        return StatusValue("unavailable", "compatibility path escapes source"), None
    candidates = tuple(
        sorted(
            {
                root / Path(known_path).stem / "attempts.json"
                for known_path in source.known_paths
            },
            key=str,
        )
    )
    existing: list[Path] = []
    for candidate in candidates:
        if not _contained(candidate, root):
            return StatusValue("unavailable", "compatibility path escapes source"), None
        try:
            if candidate.is_file():
                existing.append(candidate)
        except OSError as exc:
            return StatusValue("unavailable", f"cannot inspect compatibility: {exc}"), None
    if not existing:
        if root.exists():
            return (
                StatusValue(
                    "unknown",
                    "compatibility directory exists but no expected attempts document is present",
                ),
                None,
            )
        return StatusValue("missing", "compatibility artifacts are not present"), None
    if len(existing) > 1:
        return StatusValue("corrupt", "multiple attempts documents are ambiguous"), None
    try:
        document = AttemptDocument.from_json(existing[0].read_bytes())
    except Exception as exc:
        return StatusValue("corrupt", f"attempts document is not valid: {exc}"), None
    if document.source_fingerprint != source.source_fingerprint:
        return (
            StatusValue(
                "stale",
                "compatibility fingerprint does not match registered source",
            ),
            document,
        )
    return StatusValue("present", value=len(document.attempts)), document


def _checkpoints(
    workspace: WorkspacePaths,
    source: SourceRecord,
    runs: list[WorkflowRunRecord],
) -> StatusValue:
    paths = sorted(
        {
            Path(path)
            for run in runs
            for attempt in run.attempts
            for path in attempt.artifacts
            if Path(path).name == "checkpoints.json"
        },
        key=str,
    )
    if not paths:
        return StatusValue(
            "missing", "no workflow run proves a checkpoint artifact path"
        )
    states: list[str] = []
    for path in paths:
        if not _contained(path, workspace.root):
            return StatusValue("unavailable", "checkpoint path escapes workspace")
        if not path.is_file():
            states.append("missing")
            continue
        try:
            document = PhaseDocument.from_json(path.read_bytes())
        except Exception as exc:
            return StatusValue(
                "corrupt", f"checkpoint document is not valid: {exc}"
            )
        if document.source_fingerprint != source.source_fingerprint:
            return StatusValue(
                "stale", "checkpoint fingerprint does not match registered source"
            )
        states.append("present")
    if "missing" in states:
        return StatusValue("missing", "a recorded checkpoint artifact is missing")
    return StatusValue("present", value=len(paths))


def _landing_paths(
    workspace: WorkspacePaths, source: SourceRecord, session_ids: tuple[str, ...]
) -> tuple[str, ...]:
    candidates = {
        workspace.sources / source.source_id / "index.html",
        workspace.temporary / f"{source.source_id}.html",
        *(workspace.sessions / f"{session_id}-{source.source_id}.html" for session_id in session_ids),
    }
    return tuple(
        sorted(
            str(path)
            for path in candidates
            if _contained(path, workspace.root) and path.is_file()
        )
    )


def _source_status(
    workspace: WorkspacePaths,
    source: SourceRecord,
    sessions: Any,
    collections: Any,
) -> tuple[SourceStatus, list[str]]:
    runs, errors = _runs(workspace, source)
    run_state = (
        StatusValue("present", value=[run.overall_status for run in runs])
        if runs
        else StatusValue("missing", "no workflow run manifest is recorded")
    )
    latest = (
        StatusValue("present", value=runs[0].overall_status)
        if len(runs) == 1
        else StatusValue(
            "unknown",
            "no workflow run exists to establish a latest outcome"
            if not runs
            else "run ordering is not recorded",
        )
    )
    compatibility, attempts_document = _compatibility(workspace, source)
    attempt_ids = {attempt.attempt_id for run in runs for attempt in run.attempts}
    session_ids = tuple(
        sorted(
            record.session_id
            for record in sessions
            if source.source_id in record.source_ids
        )
    )
    collection_ids = tuple(
        sorted(
            record.collection_id
            for record in collections
            if source.source_id in record.source_ids
            or any(attempt_id in attempt_ids for attempt_id in record.attempt_ids)
        )
    )
    attempt_count = (
        len(attempts_document.attempts)
        if attempts_document is not None and compatibility.state == "present"
        else len(attempt_ids)
    )
    return (
        SourceStatus(
            source.source_id,
            source.source_fingerprint,
            source.known_paths,
            _source_availability(source),
            session_ids,
            collection_ids,
            run_state,
            latest,
            compatibility,
            _checkpoints(workspace, source, runs),
            StatusValue(
                "unknown",
                "cache eligibility is not provable from workflow provenance",
            ),
            _landing_paths(workspace, source, session_ids),
            attempt_count,
        ),
        errors,
    )


def project_status(
    workspace: WorkspacePaths,
    selector_kind: str | None = None,
    selector_value: str | None = None,
) -> StatusDocument:
    """Project deterministic status without modifying any path."""

    if not isinstance(workspace, WorkspacePaths):
        raise ValueError("workspace must be WorkspacePaths")
    sources, errors = _load_sources(workspace)
    try:
        sessions = list_sessions(workspace)
    except Exception as exc:
        sessions = []
        errors.append(f"malformed session registry: {exc}")
    try:
        collections = list_collections(workspace)
    except Exception as exc:
        collections = []
        errors.append(f"malformed collection registry: {exc}")

    if selector_kind is not None:
        if (
            selector_kind
            not in {"source_id", "source_path", "session", "collection"}
            or not isinstance(selector_value, str)
            or not selector_value
        ):
            raise ValueError("invalid status selector")
        if selector_kind == "source_id":
            if _SOURCE_ID.fullmatch(selector_value) is None:
                raise ValueError("invalid source ID selector")
            sources = [source for source in sources if source.source_id == selector_value]
        elif selector_kind == "source_path":
            try:
                selected_path = str(Path(selector_value).expanduser().resolve(strict=False))
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(f"invalid source path selector: {exc}") from exc
            sources = [
                source for source in sources if selected_path in source.known_paths
            ]
        elif selector_kind == "session":
            selected = resolve_session(workspace, selector_value)
            sources = [
                source for source in sources if source.source_id in selected.source_ids
            ]
        else:
            selected = resolve_collection(workspace, selector_value)
            selected_source_ids = set(selected.source_ids)
            run_attempt_sources = {
                source.source_id
                for source in sources
                for run in _runs(workspace, source)[0]
                if any(
                    attempt.attempt_id in selected.attempt_ids
                    for attempt in run.attempts
                )
            }
            selected_source_ids.update(run_attempt_sources)
            sources = [
                source for source in sources if source.source_id in selected_source_ids
            ]
        if not sources:
            raise ValueError("status selector did not match a registered source")

    projected: list[SourceStatus] = []
    for source in sorted(sources, key=lambda item: item.source_id):
        item, source_errors = _source_status(
            workspace, source, sessions, collections
        )
        projected.append(item)
        errors.extend(source_errors)
    return StatusDocument(
        1,
        str(workspace.root.resolve()),
        tuple(projected),
        tuple(sorted(set(errors))),
    )


def render_human(document: StatusDocument) -> str:
    """Render concise deterministic status text."""

    lines = [f"Workspace: {document.workspace}"]
    for source in document.sources:
        lines.append(
            f"{source.source_id}: source={source.source.state}; "
            f"runs={source.runs.state}; latest={source.latest_run.state}; "
            f"compatibility={source.compatibility.state}; "
            f"checkpoints={source.checkpoints.state}; "
            f"attempts={source.attempt_count}; cache={source.cache.state}"
        )
        if source.sessions:
            lines.append("  sessions: " + ", ".join(source.sessions))
        if source.collections:
            lines.append("  collections: " + ", ".join(source.collections))
        for path in source.landing_paths:
            lines.append(f"  landing: {path}")
    for error in document.errors:
        lines.append(f"Error: {error}")
    return "\n".join(lines)
