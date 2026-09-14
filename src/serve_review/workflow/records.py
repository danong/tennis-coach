"""Strict, versioned workflow records and stable identity helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from typing import Any

from serve_review.domain import SourceMetadata, SourceMetadataError

WORKFLOW_RECORD_SCHEMA_VERSION = 1
SESSION_RECORD_SCHEMA_VERSION = 1
COLLECTION_RECORD_SCHEMA_VERSION = 1
_SOURCE_FINGERPRINT = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
_SOURCE_ID = re.compile(r"^source-[0-9a-f]{16}$")
_SESSION_ID = re.compile(r"^session-[0-9a-f]{16}$")
_ATTEMPT_ID = re.compile(r"^attempt-[0-9a-f]{24}$")
_COLLECTION_ID = re.compile(r"^collection-[0-9a-f]{16}$")


class WorkflowRecordError(ValueError):
    """Raised when a workflow record or stable identity is invalid."""


def _fingerprint(value: Any) -> str:
    if not isinstance(value, str):
        raise WorkflowRecordError("source_fingerprint must be a string.")
    match = _SOURCE_FINGERPRINT.fullmatch(value)
    if not match:
        raise WorkflowRecordError(
            "source_fingerprint must be sha256: plus 64 hex digits."
        )
    return "sha256:" + match.group(1).lower()


def source_id_for_fingerprint(fingerprint: str) -> str:
    """Derive a short stable ID by hashing the complete normalized fingerprint."""

    normalized = _fingerprint(fingerprint)
    digest = hashlib.sha256(normalized.encode("ascii")).hexdigest()
    return "source-" + digest[:16]


def _known_path(value: Any) -> str:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise WorkflowRecordError("known_paths must contain absolute strings.")
    normalized = os.path.normpath(value)
    if value != normalized:
        raise WorkflowRecordError(f"known path is not normalized: {value!r}.")
    return value


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Versioned source identity, normalized metadata, and known file paths."""

    schema_version: int
    source_id: str
    source_fingerprint: str
    metadata: SourceMetadata
    known_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise WorkflowRecordError("schema_version must be an integer.")
        if self.schema_version != WORKFLOW_RECORD_SCHEMA_VERSION:
            raise WorkflowRecordError(
                f"unsupported schema_version {self.schema_version!r}."
            )

        fingerprint = _fingerprint(self.source_fingerprint)
        if (
            not isinstance(self.source_id, str)
            or self.source_id != source_id_for_fingerprint(fingerprint)
        ):
            raise WorkflowRecordError(
                "source_id does not match source_fingerprint."
            )
        if not isinstance(self.metadata, SourceMetadata):
            raise WorkflowRecordError("metadata must be SourceMetadata.")
        metadata_fingerprint = _fingerprint(self.metadata.fingerprint)
        if metadata_fingerprint != fingerprint:
            raise WorkflowRecordError(
                "metadata fingerprint does not match source_fingerprint."
            )
        if metadata_fingerprint != self.metadata.fingerprint:
            object.__setattr__(
                self,
                "metadata",
                replace(self.metadata, fingerprint=metadata_fingerprint),
            )

        if not isinstance(self.known_paths, tuple) or not self.known_paths:
            raise WorkflowRecordError("known_paths must be a nonempty tuple.")
        paths = tuple(_known_path(path) for path in self.known_paths)
        if paths != tuple(sorted(set(paths))):
            raise WorkflowRecordError(
                "known_paths must be sorted and deduplicated."
            )
        object.__setattr__(self, "source_fingerprint", fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "known_paths": list(self.known_paths),
            "metadata": self.metadata.to_dict(),
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "source_id": self.source_id,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> SourceRecord:
        if not isinstance(values, dict):
            raise WorkflowRecordError("source record must be an object.")
        keys = {
            "schema_version",
            "source_id",
            "source_fingerprint",
            "metadata",
            "known_paths",
        }
        missing = keys - set(values)
        unknown = set(values) - keys
        if missing or unknown:
            raise WorkflowRecordError(
                "source record keys invalid; "
                f"missing={sorted(missing)!r}, unknown={sorted(unknown)!r}."
            )
        if not isinstance(values["known_paths"], list):
            raise WorkflowRecordError("known_paths must be a list in JSON.")
        try:
            metadata = SourceMetadata.from_dict(values["metadata"])
        except SourceMetadataError as exc:
            raise WorkflowRecordError(f"malformed source metadata: {exc}") from exc
        return cls(
            schema_version=values["schema_version"],
            source_id=values["source_id"],
            source_fingerprint=values["source_fingerprint"],
            metadata=metadata,
            known_paths=tuple(values["known_paths"]),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> SourceRecord:
        def object_without_duplicates(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate object key {key!r}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant {value!r}")

        try:
            if isinstance(data, (bytes, bytearray)):
                data = bytes(data).decode("utf-8")
            value = json.loads(
                data,
                parse_constant=reject_constant,
                object_pairs_hook=object_without_duplicates,
            )
        except (TypeError, UnicodeError, ValueError) as exc:
            raise WorkflowRecordError(
                f"invalid source record JSON: {exc}."
            ) from exc
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class CollectionRecord:
    """Versioned explicit collection membership and presentation metadata."""

    schema_version: int
    collection_id: str
    display_name: str
    source_ids: tuple[str, ...]
    attempt_ids: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise WorkflowRecordError("schema_version must be an integer.")
        if self.schema_version != COLLECTION_RECORD_SCHEMA_VERSION:
            raise WorkflowRecordError(f"unsupported schema_version {self.schema_version!r}.")
        if not isinstance(self.collection_id, str) or _COLLECTION_ID.fullmatch(self.collection_id) is None:
            raise WorkflowRecordError("collection_id must have the form collection- plus 16 lowercase hex digits.")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise WorkflowRecordError("display_name must be a nonblank string.")
        for name, values, pattern in (("source_ids", self.source_ids, _SOURCE_ID), ("attempt_ids", self.attempt_ids, _ATTEMPT_ID)):
            if not isinstance(values, tuple):
                raise WorkflowRecordError(f"{name} must be a tuple.")
            if any(not isinstance(value, str) or pattern.fullmatch(value) is None for value in values):
                raise WorkflowRecordError(f"{name} must contain valid IDs.")
            if len(set(values)) != len(values):
                raise WorkflowRecordError(f"{name} must be ordered and unique.")
        if not isinstance(self.tags, tuple):
            raise WorkflowRecordError("tags must be a tuple.")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise WorkflowRecordError("tags must contain nonblank strings.")
        if self.tags != tuple(sorted(set(self.tags))):
            raise WorkflowRecordError("tags must be in canonical sorted unique order.")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "collection_id": self.collection_id,
                "display_name": self.display_name, "source_ids": list(self.source_ids),
                "attempt_ids": list(self.attempt_ids), "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CollectionRecord":
        if not isinstance(values, dict):
            raise WorkflowRecordError("collection record must be an object.")
        keys = {"schema_version", "collection_id", "display_name", "source_ids", "attempt_ids", "tags"}
        missing, unknown = keys - set(values), set(values) - keys
        if missing or unknown:
            raise WorkflowRecordError(f"collection record keys invalid; missing={sorted(missing, key=repr)!r}, unknown={sorted(unknown, key=repr)!r}.")
        if any(not isinstance(values[name], list) for name in ("source_ids", "attempt_ids", "tags")):
            raise WorkflowRecordError("source_ids, attempt_ids, and tags must be lists in JSON.")
        return cls(values["schema_version"], values["collection_id"], values["display_name"],
                   tuple(values["source_ids"]), tuple(values["attempt_ids"]), tuple(values["tags"]))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> "CollectionRecord":
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate object key {key!r}")
                result[key] = value
            return result
        def constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant {value!r}")
        try:
            if isinstance(data, (bytes, bytearray)):
                data = bytes(data).decode("utf-8")
            value = json.loads(data, parse_constant=constant, object_pairs_hook=pairs)
        except (TypeError, UnicodeError, ValueError) as exc:
            raise WorkflowRecordError(f"invalid collection record JSON: {exc}.") from exc
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """The complete, deliberately small persisted recording-session record."""

    schema_version: int
    session_id: str
    display_name: str
    source_ids: tuple[str, ...]
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise WorkflowRecordError("schema_version must be an integer.")
        if self.schema_version != WORKFLOW_RECORD_SCHEMA_VERSION:
            raise WorkflowRecordError(f"unsupported schema_version {self.schema_version!r}.")
        if not isinstance(self.session_id, str) or _SESSION_ID.fullmatch(self.session_id) is None:
            raise WorkflowRecordError("session_id must have the form session- plus 16 lowercase hex digits.")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise WorkflowRecordError("display_name must be a nonblank string.")
        if not isinstance(self.source_ids, tuple):
            raise WorkflowRecordError("source_ids must be a tuple.")
        for source_id in self.source_ids:
            if not isinstance(source_id, str) or _SOURCE_ID.fullmatch(source_id) is None:
                raise WorkflowRecordError("source_ids must contain valid source IDs.")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise WorkflowRecordError("source_ids must be ordered and unique.")
        if not isinstance(self.tags, tuple):
            raise WorkflowRecordError("tags must be a tuple.")
        for tag in self.tags:
            if not isinstance(tag, str) or not tag.strip():
                raise WorkflowRecordError("tags must contain nonblank strings.")
        if self.tags != tuple(sorted(set(self.tags))):
            raise WorkflowRecordError("tags must be in canonical sorted unique order.")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "session_id": self.session_id,
                "display_name": self.display_name, "source_ids": list(self.source_ids),
                "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SessionRecord":
        if not isinstance(values, dict):
            raise WorkflowRecordError("session record must be an object.")
        keys = {"schema_version", "session_id", "display_name", "source_ids", "tags"}
        missing, unknown = keys - set(values), set(values) - keys
        if missing or unknown:
            raise WorkflowRecordError(f"session record keys invalid; missing={sorted(missing, key=repr)!r}, unknown={sorted(unknown, key=repr)!r}.")
        if not isinstance(values["source_ids"], list) or not isinstance(values["tags"], list):
            raise WorkflowRecordError("source_ids and tags must be lists in JSON.")
        return cls(values["schema_version"], values["session_id"], values["display_name"],
                   tuple(values["source_ids"]), tuple(values["tags"]))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> "SessionRecord":
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate object key {key!r}")
                result[key] = value
            return result
        def constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant {value!r}")
        try:
            if isinstance(data, (bytes, bytearray)):
                data = bytes(data).decode("utf-8")
            value = json.loads(data, parse_constant=constant, object_pairs_hook=pairs)
        except (TypeError, UnicodeError, ValueError) as exc:
            raise WorkflowRecordError(f"invalid session record JSON: {exc}.") from exc
        return cls.from_dict(value)


def _identity(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowRecordError(f"{name} must be a nonblank string.")
    return value


def attempt_id_for(
    source_identity: str,
    method_identity: str,
    config_identity: str,
    start: float,
    end: float,
) -> str:
    """Hash exact producer identities and unpadded range bounds."""

    source = _identity("source identity", source_identity)
    if _SOURCE_ID.fullmatch(source) is None:
        raise WorkflowRecordError(
            "source identity must have the form source- plus 16 lowercase hex digits."
        )
    method = _identity("method identity", method_identity)
    config = _identity("config identity", config_identity)
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
    ):
        raise WorkflowRecordError("attempt bounds must be numeric, not bool.")
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or start < 0
        or start >= end
    ):
        raise WorkflowRecordError(
            "attempt bounds must be finite and satisfy 0 <= start < end."
        )
    canonical = json.dumps(
        [source, method, config, float(start).hex(), float(end).hex()],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return "attempt-" + digest[:24]
