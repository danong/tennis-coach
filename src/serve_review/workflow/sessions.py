"""Atomic storage for recording-session manifests."""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from serve_review.workflow.records import SessionRecord, WorkflowRecordError
from serve_review.workflow.sources import SourceRegistryError, load_source
from serve_review.workflow.workspace import WorkspaceError, WorkspacePaths, write_json_atomic

SESSION_RECORD_SUFFIX = ".json"
_SESSION_ID = re.compile(r"^session-[0-9a-f]{16}$")
_SOURCE_ID = re.compile(r"^source-[0-9a-f]{16}$")


class SessionStoreError(Exception):
    """Raised when a session manifest operation cannot be completed safely."""


def session_record_path(workspace: WorkspacePaths, session_id: str) -> Path:
    if not isinstance(workspace, WorkspacePaths):
        raise SessionStoreError("session path requires WorkspacePaths.")
    if not isinstance(session_id, str) or _SESSION_ID.fullmatch(session_id) is None:
        raise SessionStoreError("session_id must have the form session- plus 16 lowercase hex digits.")
    try:
        root = workspace.sessions.resolve()
        directory = (workspace.sessions / (session_id + SESSION_RECORD_SUFFIX)).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SessionStoreError(f"could not resolve session path: {exc}.") from exc
    if directory.parent != root:
        raise SessionStoreError("session record path escapes workspace sessions.")
    return directory


def _read(path: Path, expected_id: str) -> SessionRecord:
    try:
        if not path.exists() or not path.is_file():
            raise SessionStoreError(
                f"session manifest {path} is missing or not a regular file."
            )
        record = SessionRecord.from_json(path.read_bytes())
    except SessionStoreError:
        raise
    except (OSError, WorkflowRecordError) as exc:
        raise SessionStoreError(
            f"could not load session manifest {path}: {exc}"
        ) from exc
    if record.session_id != expected_id:
        raise SessionStoreError(
            f"session manifest {path} contains ID {record.session_id}, "
            f"expected {expected_id}."
        )
    return record


def load_session(workspace: WorkspacePaths, session_id: str) -> SessionRecord:
    return _read(session_record_path(workspace, session_id), session_id)


def _all(workspace: WorkspacePaths) -> list[SessionRecord]:
    if not isinstance(workspace, WorkspacePaths):
        raise SessionStoreError("session operation requires WorkspacePaths.")
    root = workspace.sessions
    if not root.exists():
        return []
    if not root.is_dir():
        raise SessionStoreError(f"sessions path {root} is not a directory.")
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
        resolved_root = root.resolve()
    except (OSError, RuntimeError) as exc:
        raise SessionStoreError(f"could not inspect sessions path {root}: {exc}") from exc
    records: list[SessionRecord] = []
    for entry in entries:
        try:
            contained = entry.resolve(strict=False).parent == resolved_root
        except (OSError, RuntimeError) as exc:
            raise SessionStoreError(f"could not validate session entry {entry}: {exc}") from exc
        entry_id = entry.name[: -len(SESSION_RECORD_SUFFIX)]
        if (
            not entry.is_file()
            or not contained
            or not entry.name.endswith(SESSION_RECORD_SUFFIX)
            or _SESSION_ID.fullmatch(entry_id) is None
        ):
            raise SessionStoreError(f"malformed/non-file session entry {entry}.")
        records.append(_read(entry, entry_id))
    return sorted(records, key=lambda record: record.session_id)


def list_sessions(workspace: WorkspacePaths) -> list[SessionRecord]:
    return _all(workspace)


def _canonical_tags(tags: Iterable[str]) -> tuple[str, ...]:
    if isinstance(tags, (str, bytes)):
        raise SessionStoreError("tags must be an iterable of strings, not text.")
    try:
        values = tuple(tags)
    except (TypeError, ValueError) as exc:
        raise SessionStoreError(f"tags must be an iterable of strings: {exc}") from exc
    if any(not isinstance(tag, str) or not tag.strip() for tag in values):
        raise SessionStoreError("tags must contain nonblank strings.")
    if len(set(values)) != len(values):
        raise SessionStoreError("tags must be unique.")
    return tuple(sorted(values))


def _sources_exist(workspace: WorkspacePaths, source_ids: Iterable[str]) -> None:
    for source_id in source_ids:
        if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
            raise SessionStoreError(f"invalid source ID {source_id!r}.")
        try:
            load_source(workspace, source_id)
        except SourceRegistryError as exc:
            raise SessionStoreError(f"source {source_id} does not exist or is invalid: {exc}") from exc


def _owners(source_id: str, records: list[SessionRecord]) -> tuple[str, ...]:
    return tuple(record.session_id for record in records if source_id in record.source_ids)


def _write(workspace: WorkspacePaths, record: SessionRecord) -> SessionRecord:
    try:
        write_json_atomic(session_record_path(workspace, record.session_id), record.to_dict())
    except (SessionStoreError, WorkspaceError, WorkflowRecordError, OSError) as exc:
        raise SessionStoreError(f"could not write session {record.session_id}: {exc}") from exc
    return record


def create_session(workspace: WorkspacePaths, display_name: str, *, id_factory: Callable[[], str], source_ids: Iterable[str] = (), tags: Iterable[str] = ()) -> SessionRecord:
    try:
        session_id = id_factory()  # deliberately exactly one call
    except Exception as exc:
        raise SessionStoreError(f"could not create session ID: {exc}") from exc
    try:
        record = SessionRecord(1, session_id, display_name, tuple(source_ids), _canonical_tags(tags))
        path = session_record_path(workspace, session_id)
        if path.exists():
            raise SessionStoreError(f"session ID collision: {session_id}.")
        _sources_exist(workspace, record.source_ids)
        existing = _all(workspace)
        for source_id in record.source_ids:
            owners = _owners(source_id, existing)
            if owners:
                raise SessionStoreError(f"source {source_id} already belongs to sessions {', '.join(owners)} and {session_id}.")
        return _write(workspace, record)
    except SessionStoreError:
        raise
    except (WorkflowRecordError, TypeError, ValueError) as exc:
        raise SessionStoreError(f"could not create session: {exc}") from exc


def resolve_session(workspace: WorkspacePaths, identifier: str) -> SessionRecord:
    if isinstance(identifier, str) and _SESSION_ID.fullmatch(identifier):
        try:
            return load_session(workspace, identifier)
        except SessionStoreError as exc:
            raise SessionStoreError(f"session ID {identifier} not found: {exc}") from exc
    matches = [r for r in _all(workspace) if r.display_name == identifier]
    if not matches:
        raise SessionStoreError(f"session display name {identifier!r} not found.")
    if len(matches) > 1:
        raise SessionStoreError(f"session display name {identifier!r} is ambiguous: " + ", ".join(r.session_id for r in matches))
    return matches[0]


def rename_session(workspace: WorkspacePaths, session_id: str, display_name: str) -> SessionRecord:
    record = load_session(workspace, session_id)
    try:
        return _write(workspace, SessionRecord(1, record.session_id, display_name, record.source_ids, record.tags))
    except (WorkflowRecordError, TypeError, ValueError) as exc:
        raise SessionStoreError(f"could not rename session {session_id}: {exc}") from exc


def set_session_tags(workspace: WorkspacePaths, session_id: str, tags: Iterable[str]) -> SessionRecord:
    record = load_session(workspace, session_id)
    try:
        return _write(workspace, SessionRecord(1, record.session_id, record.display_name, record.source_ids, _canonical_tags(tags)))
    except (WorkflowRecordError, TypeError, ValueError) as exc:
        raise SessionStoreError(f"could not set tags for session {session_id}: {exc}") from exc


def add_source(workspace: WorkspacePaths, session_id: str, source_id: str) -> SessionRecord:
    record = load_session(workspace, session_id)
    if source_id in record.source_ids:
        raise SessionStoreError(f"source {source_id} is already in session {session_id}.")
    _sources_exist(workspace, (source_id,))
    owners = _owners(source_id, _all(workspace))
    if owners:
        raise SessionStoreError(f"source {source_id} already belongs to sessions {', '.join(owners)} and {session_id}.")
    return _write(workspace, SessionRecord(1, record.session_id, record.display_name, record.source_ids + (source_id,), record.tags))


def remove_source(workspace: WorkspacePaths, session_id: str, source_id: str) -> SessionRecord:
    record = load_session(workspace, session_id)
    if source_id not in record.source_ids:
        raise SessionStoreError(f"source {source_id} is not in session {session_id}.")
    remaining = tuple(item for item in record.source_ids if item != source_id)
    return _write(workspace, SessionRecord(1, record.session_id, record.display_name, remaining, record.tags))


def delete_session(workspace: WorkspacePaths, session_id: str) -> None:
    path = session_record_path(workspace, session_id)
    _read(path, session_id)
    try:
        path.unlink()
    except OSError as exc:
        raise SessionStoreError(f"could not delete session {session_id}: {exc}") from exc
