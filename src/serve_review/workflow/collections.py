"""Atomic storage for explicit, non-exclusive collection manifests."""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from serve_review.workflow.records import CollectionRecord, WorkflowRecordError
from serve_review.workflow.sources import SourceRegistryError, load_source
from serve_review.workflow.workspace import WorkspaceError, WorkspacePaths, write_json_atomic

COLLECTION_RECORD_SUFFIX = ".json"
_COLLECTION_ID = re.compile(r"^collection-[0-9a-f]{16}$")
_SOURCE_ID = re.compile(r"^source-[0-9a-f]{16}$")
_ATTEMPT_ID = re.compile(r"^attempt-[0-9a-f]{24}$")


class CollectionStoreError(Exception):
    """Raised when a collection manifest operation cannot be completed."""


def collection_record_path(workspace: WorkspacePaths, collection_id: str) -> Path:
    if not isinstance(workspace, WorkspacePaths):
        raise CollectionStoreError("collection path requires WorkspacePaths.")
    if not isinstance(collection_id, str) or _COLLECTION_ID.fullmatch(collection_id) is None:
        raise CollectionStoreError("collection_id must have the form collection- plus 16 lowercase hex digits.")
    try:
        root = workspace.collections.resolve()
        path = (workspace.collections / (collection_id + COLLECTION_RECORD_SUFFIX)).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise CollectionStoreError(f"could not resolve collection path: {exc}") from exc
    if path.parent != root:
        raise CollectionStoreError("collection record path escapes workspace collections.")
    return path


def _read(path: Path, expected_id: str) -> CollectionRecord:
    try:
        if not path.exists() or not path.is_file():
            raise CollectionStoreError(f"collection manifest {path} is missing or not a regular file.")
        record = CollectionRecord.from_json(path.read_bytes())
    except CollectionStoreError:
        raise
    except (OSError, WorkflowRecordError) as exc:
        raise CollectionStoreError(f"could not load collection manifest {path}: {exc}") from exc
    if record.collection_id != expected_id:
        raise CollectionStoreError(f"collection manifest {path} contains ID {record.collection_id}, expected {expected_id}.")
    return record


def load_collection(workspace: WorkspacePaths, collection_id: str) -> CollectionRecord:
    return _read(collection_record_path(workspace, collection_id), collection_id)


def _all(workspace: WorkspacePaths) -> list[CollectionRecord]:
    if not isinstance(workspace, WorkspacePaths):
        raise CollectionStoreError("collection operation requires WorkspacePaths.")
    root = workspace.collections
    if not root.exists():
        return []
    if not root.is_dir():
        raise CollectionStoreError(f"collections path {root} is not a directory.")
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name)
        resolved_root = root.resolve()
    except (OSError, RuntimeError) as exc:
        raise CollectionStoreError(f"could not inspect collections path {root}: {exc}") from exc
    records: list[CollectionRecord] = []
    for entry in entries:
        entry_id = entry.name[:-len(COLLECTION_RECORD_SUFFIX)] if entry.name.endswith(COLLECTION_RECORD_SUFFIX) else ""
        try:
            contained = entry.resolve(strict=False).parent == resolved_root
        except (OSError, RuntimeError) as exc:
            raise CollectionStoreError(f"could not validate collection entry {entry}: {exc}") from exc
        if (not entry.is_file() or not contained or not entry.name.endswith(COLLECTION_RECORD_SUFFIX)
                or _COLLECTION_ID.fullmatch(entry_id) is None):
            raise CollectionStoreError(f"malformed/non-file collection entry {entry}.")
        records.append(_read(entry, entry_id))
    return sorted(records, key=lambda record: record.collection_id)


def list_collections(workspace: WorkspacePaths) -> list[CollectionRecord]:
    return _all(workspace)


def _tags(tags: Iterable[str]) -> tuple[str, ...]:
    if isinstance(tags, (str, bytes)):
        raise CollectionStoreError("tags must be an iterable of strings, not text.")
    try:
        values = tuple(tags)
    except (TypeError, ValueError) as exc:
        raise CollectionStoreError(f"tags must be an iterable of strings: {exc}") from exc
    if any(not isinstance(tag, str) or not tag.strip() for tag in values):
        raise CollectionStoreError("tags must contain nonblank strings.")
    if len(set(values)) != len(values):
        raise CollectionStoreError("tags must be unique.")
    return tuple(sorted(values))


def _sources_exist(workspace: WorkspacePaths, source_ids: Iterable[str]) -> None:
    for source_id in source_ids:
        if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
            raise CollectionStoreError(f"invalid source ID {source_id!r}.")
        try:
            load_source(workspace, source_id)
        except SourceRegistryError as exc:
            raise CollectionStoreError(f"source {source_id} does not exist or is invalid: {exc}") from exc


def _write(workspace: WorkspacePaths, record: CollectionRecord) -> CollectionRecord:
    try:
        write_json_atomic(collection_record_path(workspace, record.collection_id), record.to_dict())
    except (CollectionStoreError, WorkspaceError, WorkflowRecordError, OSError) as exc:
        raise CollectionStoreError(f"could not write collection {record.collection_id}: {exc}") from exc
    return record


def create_collection(workspace: WorkspacePaths, name: str, *, id_factory: Callable[[], str],
                      source_ids: Iterable[str] = (), attempt_ids: Iterable[str] = (),
                      tags: Iterable[str] = ()) -> CollectionRecord:
    try:
        collection_id = id_factory()  # exactly one call, even when validation fails
    except Exception as exc:
        raise CollectionStoreError(f"could not create collection ID: {exc}") from exc
    try:
        record = CollectionRecord(1, collection_id, name, tuple(source_ids), tuple(attempt_ids), _tags(tags))
        path = collection_record_path(workspace, collection_id)
        if os.path.lexists(path):
            raise CollectionStoreError(f"collection ID collision: {collection_id}.")
        _sources_exist(workspace, record.source_ids)
        return _write(workspace, record)
    except CollectionStoreError:
        raise
    except (TypeError, ValueError, WorkflowRecordError) as exc:
        raise CollectionStoreError(f"could not create collection: {exc}") from exc


def resolve_collection(workspace: WorkspacePaths, identifier: str) -> CollectionRecord:
    if isinstance(identifier, str) and _COLLECTION_ID.fullmatch(identifier):
        try:
            return load_collection(workspace, identifier)
        except CollectionStoreError as exc:
            raise CollectionStoreError(f"collection ID {identifier} not found: {exc}") from exc
    matches = [record for record in _all(workspace) if record.display_name == identifier]
    if not matches:
        raise CollectionStoreError(f"collection display name {identifier!r} not found.")
    if len(matches) > 1:
        raise CollectionStoreError(f"collection display name {identifier!r} is ambiguous: " + ", ".join(r.collection_id for r in matches))
    return matches[0]


def rename_collection(workspace: WorkspacePaths, collection_id: str, name: str) -> CollectionRecord:
    record = load_collection(workspace, collection_id)
    try:
        return _write(workspace, CollectionRecord(1, record.collection_id, name, record.source_ids, record.attempt_ids, record.tags))
    except (TypeError, ValueError, WorkflowRecordError) as exc:
        raise CollectionStoreError(f"could not rename collection {collection_id}: {exc}") from exc


def set_collection_tags(workspace: WorkspacePaths, collection_id: str, tags: Iterable[str]) -> CollectionRecord:
    record = load_collection(workspace, collection_id)
    try:
        return _write(workspace, CollectionRecord(1, record.collection_id, record.display_name, record.source_ids, record.attempt_ids, _tags(tags)))
    except (TypeError, ValueError, WorkflowRecordError) as exc:
        raise CollectionStoreError(f"could not set tags for collection {collection_id}: {exc}") from exc


def add_source(workspace: WorkspacePaths, collection_id: str, source_id: str) -> CollectionRecord:
    record = load_collection(workspace, collection_id)
    if source_id in record.source_ids:
        raise CollectionStoreError(f"source {source_id} is already in collection {collection_id}.")
    _sources_exist(workspace, (source_id,))
    return _write(workspace, CollectionRecord(1, record.collection_id, record.display_name, record.source_ids + (source_id,), record.attempt_ids, record.tags))


def remove_source(workspace: WorkspacePaths, collection_id: str, source_id: str) -> CollectionRecord:
    record = load_collection(workspace, collection_id)
    if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
        raise CollectionStoreError(f"invalid source ID {source_id!r}.")
    if source_id not in record.source_ids:
        raise CollectionStoreError(f"source {source_id} is not in collection {collection_id}.")
    return _write(workspace, CollectionRecord(1, record.collection_id, record.display_name, tuple(x for x in record.source_ids if x != source_id), record.attempt_ids, record.tags))


def add_attempt(workspace: WorkspacePaths, collection_id: str, attempt_id: str) -> CollectionRecord:
    record = load_collection(workspace, collection_id)
    if attempt_id in record.attempt_ids:
        raise CollectionStoreError(f"attempt {attempt_id} is already in collection {collection_id}.")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise CollectionStoreError(f"invalid attempt ID {attempt_id!r}.")
    return _write(workspace, CollectionRecord(1, record.collection_id, record.display_name, record.source_ids, record.attempt_ids + (attempt_id,), record.tags))


def remove_attempt(workspace: WorkspacePaths, collection_id: str, attempt_id: str) -> CollectionRecord:
    record = load_collection(workspace, collection_id)
    if not isinstance(attempt_id, str) or _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise CollectionStoreError(f"invalid attempt ID {attempt_id!r}.")
    if attempt_id not in record.attempt_ids:
        raise CollectionStoreError(f"attempt {attempt_id} is not in collection {collection_id}.")
    return _write(workspace, CollectionRecord(1, record.collection_id, record.display_name, record.source_ids, tuple(x for x in record.attempt_ids if x != attempt_id), record.tags))


def delete_collection(workspace: WorkspacePaths, collection_id: str) -> None:
    path = collection_record_path(workspace, collection_id)
    _read(path, collection_id)
    try:
        path.unlink()
    except OSError as exc:
        raise CollectionStoreError(f"could not delete collection {collection_id}: {exc}") from exc
