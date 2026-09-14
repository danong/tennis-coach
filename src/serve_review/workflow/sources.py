"""Registration of immutable source references in a managed workspace."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from serve_review.domain import SourceMetadata
from serve_review.media.probe import probe_source
from serve_review.workflow.records import (
    WORKFLOW_RECORD_SCHEMA_VERSION,
    SourceRecord,
    WorkflowRecordError,
    source_id_for_fingerprint,
)
from serve_review.workflow.workspace import (
    WorkspaceError,
    WorkspacePaths,
    read_json,
    write_json_atomic,
)

SOURCE_RECORD_FILENAME = "record.json"
_SOURCE_ID = re.compile(r"^source-[0-9a-f]{16}$")


class SourceRegistryError(Exception):
    """Raised when source registration or loading cannot be completed safely."""


def source_record_path(workspace: WorkspacePaths, source_id: str) -> Path:
    """Return the contained manifest path for a validated source ID."""

    if not isinstance(workspace, WorkspacePaths):
        raise SourceRegistryError("source record path requires WorkspacePaths.")
    if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
        raise SourceRegistryError(
            "source_id must have the form source- plus 16 lowercase hex digits."
        )
    try:
        sources_root = workspace.sources.resolve()
        source_directory = (workspace.sources / source_id).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SourceRegistryError(
            f"could not resolve source record directory for {source_id}: {exc}."
        ) from exc
    if source_directory.parent != sources_root:
        raise SourceRegistryError(
            f"source record directory escapes workspace sources: {source_directory}."
        )
    return source_directory / SOURCE_RECORD_FILENAME


def load_source(workspace: WorkspacePaths, source_id: str) -> SourceRecord:
    """Load and strictly validate one registered source record."""

    path = source_record_path(workspace, source_id)
    try:
        return SourceRecord.from_dict(read_json(path))
    except (WorkspaceError, WorkflowRecordError) as exc:
        raise SourceRegistryError(
            f"could not load source record {path}: {exc}"
        ) from exc


def _known_path(video: object) -> Path:
    try:
        path = Path(video).expanduser()  # type: ignore[arg-type]
        if not path.exists():
            raise SourceRegistryError(f"source video {path} does not exist.")
        if not path.is_file():
            raise SourceRegistryError(f"source video {path} is not a regular file.")
        return path.resolve()
    except SourceRegistryError:
        raise
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise SourceRegistryError(
            f"could not inspect source video {video!r}: {exc}"
        ) from exc


def register_source(
    video: Path | str,
    workspace: WorkspacePaths,
    *,
    probe_fn: Callable[[Path | str], SourceMetadata] = probe_source,
    fingerprint_fn: Callable[[Path | str], str] | None = None,
) -> SourceRecord:
    """Probe and register a source by reference without changing its media."""

    if not isinstance(workspace, WorkspacePaths):
        raise SourceRegistryError("register_source requires WorkspacePaths.")
    path = _known_path(video)
    try:
        metadata = probe_fn(path)
    except Exception as exc:
        raise SourceRegistryError(
            f"could not probe source video {path}: {exc}"
        ) from exc
    if not isinstance(metadata, SourceMetadata):
        raise SourceRegistryError(
            f"probe for {path} did not return SourceMetadata."
        )
    normalized_metadata = replace(
        metadata,
        fingerprint=metadata.fingerprint.lower(),
    )

    if fingerprint_fn is not None:
        try:
            checked = fingerprint_fn(path)
        except Exception as exc:
            raise SourceRegistryError(
                f"could not fingerprint source video {path}: {exc}"
            ) from exc
        try:
            source_id_for_fingerprint(checked)
            normalized_checked = "sha256:" + checked.split(":", 1)[1].lower()
        except (AttributeError, IndexError, WorkflowRecordError) as exc:
            raise SourceRegistryError(
                f"invalid fingerprint for source video {path}: {exc}"
            ) from exc
        if normalized_checked != normalized_metadata.fingerprint:
            raise SourceRegistryError(
                f"fingerprint mismatch for source video {path}."
            )

    try:
        source_id = source_id_for_fingerprint(normalized_metadata.fingerprint)
        destination = source_record_path(workspace, source_id)
        canonical_path = os.fspath(path)
        if destination.exists():
            existing = load_source(workspace, source_id)
            if existing.source_fingerprint != normalized_metadata.fingerprint:
                raise SourceRegistryError(
                    f"source ID collision at {destination.parent}."
                )
            if existing.metadata != normalized_metadata:
                raise SourceRegistryError(
                    f"metadata mismatch for existing source {source_id}."
                )
            known_paths = tuple(
                sorted(set(existing.known_paths + (canonical_path,)))
            )
            if known_paths == existing.known_paths:
                return existing
            updated = SourceRecord(
                schema_version=WORKFLOW_RECORD_SCHEMA_VERSION,
                source_id=source_id,
                source_fingerprint=existing.source_fingerprint,
                metadata=existing.metadata,
                known_paths=known_paths,
            )
        else:
            updated = SourceRecord(
                schema_version=WORKFLOW_RECORD_SCHEMA_VERSION,
                source_id=source_id,
                source_fingerprint=normalized_metadata.fingerprint,
                metadata=normalized_metadata,
                known_paths=(canonical_path,),
            )
        write_json_atomic(destination, updated.to_dict())
        return updated
    except SourceRegistryError:
        raise
    except (WorkspaceError, WorkflowRecordError, OSError, ValueError) as exc:
        raise SourceRegistryError(
            f"could not register source video {path}: {exc}"
        ) from exc
