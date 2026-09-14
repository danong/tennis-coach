"""Strict, versioned workflow records and stable identity helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, replace
from typing import Any

from serve_review.domain import MediaRange, SourceMetadata, SourceMetadataError

WORKFLOW_RECORD_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
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


@dataclass(frozen=True, slots=True)
class DiscoveredSource:
    """A fingerprinted source found during planning, with stable path aliases."""

    source_fingerprint: str
    representative_path: str
    known_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        fingerprint = _fingerprint(self.source_fingerprint)
        paths = tuple(_known_path(p) for p in self.known_paths)
        if not paths or paths != tuple(sorted(set(paths))):
            raise WorkflowRecordError("known_paths must be sorted, unique, and nonempty.")
        if self.representative_path != paths[0]:
            raise WorkflowRecordError(
                "representative_path must be the first deterministic known path."
            )
        object.__setattr__(self, "source_fingerprint", fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {"known_paths": list(self.known_paths), "representative_path": self.representative_path,
                "source_fingerprint": self.source_fingerprint}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DiscoveredSource":
        if not isinstance(values, dict):
            raise WorkflowRecordError("discovered source must be an object.")
        keys = {"source_fingerprint", "representative_path", "known_paths"}
        if set(values) != keys:
            raise WorkflowRecordError("discovered source keys are invalid.")
        if not isinstance(values["known_paths"], list):
            raise WorkflowRecordError("known_paths must be a list in JSON.")
        return cls(values["source_fingerprint"], values["representative_path"], tuple(values["known_paths"]))


@dataclass(frozen=True, slots=True)
class StepPlan:
    """A conservative plan state: exactly ``required``, ``reusable``, or ``unknown``."""
    state: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or self.state not in {"required", "reusable", "unknown"}:
            raise WorkflowRecordError("step state must be required, reusable, or unknown.")

    def to_dict(self) -> dict[str, str]:
        return {"state": self.state}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StepPlan":
        if not isinstance(values, dict) or set(values) != {"state"}:
            raise WorkflowRecordError("step plan keys are invalid.")
        return cls(values["state"])


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """Per-source read-only work states."""
    source: DiscoveredSource
    registered: SourceRecord | None
    probe: StepPlan
    detection: StepPlan
    checkpoint_analysis: StepPlan
    landing_page: StepPlan

    def to_dict(self) -> dict[str, Any]:
        return {"checkpoint_analysis": self.checkpoint_analysis.to_dict(), "detection": self.detection.to_dict(),
                "landing_page": self.landing_page.to_dict(), "probe": self.probe.to_dict(),
                "registered": self.registered.to_dict() if self.registered else None, "source": self.source.to_dict()}

    def __post_init__(self) -> None:
        if not isinstance(self.source, DiscoveredSource) or (self.registered is not None and not isinstance(self.registered, SourceRecord)):
            raise WorkflowRecordError("source plan has invalid source record types")
        if any(not isinstance(getattr(self, name), StepPlan) for name in ("probe", "detection", "checkpoint_analysis", "landing_page")):
            raise WorkflowRecordError("source plan has invalid step types")
        if (
            self.registered is not None
            and self.registered.source_fingerprint != self.source.source_fingerprint
        ):
            raise WorkflowRecordError(
                "registered source fingerprint does not match discovered source."
            )

    @classmethod
    def from_dict(cls, v: dict[str, Any]) -> "SourcePlan":
        keys = {"checkpoint_analysis", "detection", "landing_page", "probe", "registered", "source"}
        if not isinstance(v, dict) or set(v) != keys: raise WorkflowRecordError("source plan keys are invalid")
        try:
            reg = None if v["registered"] is None else SourceRecord.from_dict(v["registered"])
            return cls(DiscoveredSource.from_dict(v["source"]), reg, StepPlan.from_dict(v["probe"]), StepPlan.from_dict(v["detection"]), StepPlan.from_dict(v["checkpoint_analysis"]), StepPlan.from_dict(v["landing_page"]))
        except (WorkflowRecordError, TypeError, KeyError) as exc: raise WorkflowRecordError(f"invalid source plan: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ProcessPlan:
    """Strict, versioned, deterministic description of work; it performs none."""
    schema_version: int
    mode: str
    explicit_range: MediaRange | None
    sources: tuple[SourcePlan, ...]
    unsupported_entries: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int) or self.schema_version != PLAN_SCHEMA_VERSION: raise WorkflowRecordError("unsupported plan schema version")
        if self.mode not in {"normal", "single-attempt", "explicit-range"}: raise WorkflowRecordError("invalid planning mode")
        if self.mode == "explicit-range" and not isinstance(self.explicit_range, MediaRange): raise WorkflowRecordError("explicit-range mode requires a MediaRange")
        if self.mode != "explicit-range" and self.explicit_range is not None: raise WorkflowRecordError("range is only valid in explicit-range mode")
        if not isinstance(self.sources, tuple) or any(not isinstance(s, SourcePlan) for s in self.sources) or tuple(s.source.representative_path for s in self.sources) != tuple(sorted(s.source.representative_path for s in self.sources)): raise WorkflowRecordError("sources must be deterministically ordered")
        fingerprints = tuple(source.source.source_fingerprint for source in self.sources)
        if len(set(fingerprints)) != len(fingerprints):
            raise WorkflowRecordError("sources must have unique fingerprints")
        if self.mode != "normal" and len(self.sources) != 1:
            raise WorkflowRecordError(
                "single-attempt and explicit-range plans require exactly one source"
            )
        if not isinstance(self.unsupported_entries, tuple) or any(not isinstance(p, str) or not os.path.isabs(p) or os.path.normpath(p) != p for p in self.unsupported_entries) or self.unsupported_entries != tuple(sorted(set(self.unsupported_entries))): raise WorkflowRecordError("unsupported entries must be normalized absolute sorted unique paths")

    @property
    def discovered(self) -> tuple[DiscoveredSource, ...]: return tuple(s.source for s in self.sources)
    @property
    def discovered_sources(self) -> tuple[DiscoveredSource, ...]: return self.discovered
    @property
    def range_detection(self) -> tuple[StepPlan, ...]: return tuple(s.detection for s in self.sources)
    @property
    def registered(self) -> tuple[SourceRecord, ...]: return tuple(s.registered for s in self.sources if s.registered is not None)
    @property
    def new(self) -> tuple[DiscoveredSource, ...]: return tuple(s.source for s in self.sources if s.registered is None)

    @property
    def new_sources(self) -> tuple[DiscoveredSource, ...]: return self.new

    def to_dict(self) -> dict[str, Any]:
        return {"explicit_range": self.explicit_range.to_dict() if self.explicit_range else None, "mode": self.mode, "schema_version": self.schema_version, "sources": [s.to_dict() for s in self.sources], "unsupported_entries": list(self.unsupported_entries)}
    @classmethod
    def from_dict(cls, v: dict[str, Any]) -> "ProcessPlan":
        keys = {"explicit_range", "mode", "schema_version", "sources", "unsupported_entries"}
        if not isinstance(v, dict) or set(v) != keys: raise WorkflowRecordError("process plan keys are invalid")
        if not isinstance(v["sources"], list) or not isinstance(v["unsupported_entries"], list): raise WorkflowRecordError("process plan arrays are required")
        try: return cls(v["schema_version"], v["mode"], None if v["explicit_range"] is None else MediaRange.from_dict(v["explicit_range"]), tuple(SourcePlan.from_dict(x) for x in v["sources"]), tuple(v["unsupported_entries"]))
        except Exception as exc:
            if isinstance(exc, WorkflowRecordError): raise
            raise WorkflowRecordError(f"invalid process plan: {exc}") from exc
    def to_json(self) -> str: return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> "ProcessPlan":
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result: raise ValueError(f"duplicate object key {key!r}")
                result[key] = value
            return result
        try:
            if isinstance(data, (bytes, bytearray)): data = bytes(data).decode("utf8")
            return cls.from_dict(json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)), object_pairs_hook=pairs))
        except Exception as exc:
            if isinstance(exc, WorkflowRecordError): raise
            raise WorkflowRecordError(f"invalid process plan JSON: {exc}") from exc
    def human(self) -> str:
        lines = [f"Mode: {self.mode}", f"Sources: {len(self.sources)} discovered, {len(self.new)} new, {len(self.registered)} already registered"]
        for label, attr in (("Probe", "probe"), ("Range detection", "detection"), ("Stage-checkpoint analysis", "checkpoint_analysis"), ("Review index", "landing_page")):
            counts = {x: sum(getattr(s, attr).state == x for s in self.sources) for x in ("reusable", "required", "unknown")}
            lines.append(f"{label}: {counts['reusable']} reusable, {counts['required']} required, {counts['unknown']} unknown")
        if self.unsupported_entries: lines.append(f"Unsupported entries: {len(self.unsupported_entries)}")
        return "\n".join(lines)
    render_human = human

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


RUN_RECORD_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"^run-[0-9a-f]{16}$")


def _strict_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowRecordError(f"{name} must be a nonblank string.")
    return value


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    """The deliberately small, provenance-only result of one analysis."""
    attempt_id: str
    start_seconds: float
    end_seconds: float
    status: str
    error: str | None = None
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, str) or _ATTEMPT_ID.fullmatch(self.attempt_id) is None:
            raise WorkflowRecordError("attempt outcome has an invalid attempt_id.")
        try:
            MediaRange(self.start_seconds, self.end_seconds)
        except Exception as exc:
            raise WorkflowRecordError(f"invalid attempt outcome range: {exc}") from exc
        if self.status not in {"complete", "failed"}:
            raise WorkflowRecordError("attempt outcome status must be complete or failed.")
        if (
            self.status == "failed"
            and (not isinstance(self.error, str) or not self.error.strip())
        ) or (self.status == "complete" and self.error is not None):
            raise WorkflowRecordError("attempt outcome error does not match status.")
        if not isinstance(self.artifacts, tuple) or any(
            not isinstance(path, str)
            or not os.path.isabs(path)
            or os.path.normpath(path) != path
            for path in self.artifacts
        ):
            raise WorkflowRecordError(
                "attempt outcome artifacts must be normalized absolute paths."
            )
        if self.artifacts != tuple(sorted(set(self.artifacts))):
            raise WorkflowRecordError(
                "attempt outcome artifacts must be sorted and unique."
            )
        if self.status == "complete" and not self.artifacts:
            raise WorkflowRecordError("complete attempt outcome requires artifacts.")
        if self.status == "failed" and self.artifacts:
            raise WorkflowRecordError("failed attempt outcome cannot claim artifacts.")

    def to_dict(self) -> dict[str, Any]:
        return {"artifacts": list(self.artifacts), "attempt_id": self.attempt_id, "end_seconds": self.end_seconds, "error": self.error, "start_seconds": self.start_seconds, "status": self.status}

    @classmethod
    def from_dict(cls, v: dict[str, Any]) -> "AttemptOutcome":
        keys = {"artifacts", "attempt_id", "end_seconds", "error", "start_seconds", "status"}
        if not isinstance(v, dict) or set(v) != keys or not isinstance(v["artifacts"], list):
            raise WorkflowRecordError("attempt outcome keys are invalid.")
        return cls(v["attempt_id"], v["start_seconds"], v["end_seconds"], v["status"], v["error"], tuple(v["artifacts"]))


@dataclass(frozen=True, slots=True)
class WorkflowRunRecord:
    """Strict persisted provenance for one source processing invocation."""
    schema_version: int
    run_id: str
    source_id: str
    source_fingerprint: str
    mode: str
    producer_method: str
    producer_config: str
    detection_status: str
    detection_error: str | None
    attempts: tuple[AttemptOutcome, ...]
    overall_status: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != RUN_RECORD_SCHEMA_VERSION
        ):
            raise WorkflowRecordError("unsupported workflow run schema version.")
        if not isinstance(self.run_id, str) or _RUN_ID.fullmatch(self.run_id) is None:
            raise WorkflowRecordError("run_id must have the form run- plus 16 lowercase hex digits.")
        if not isinstance(self.source_id, str) or _SOURCE_ID.fullmatch(self.source_id) is None:
            raise WorkflowRecordError("invalid source_id.")
        fingerprint = _fingerprint(self.source_fingerprint)
        if self.source_id != source_id_for_fingerprint(fingerprint):
            raise WorkflowRecordError("source_id does not match source_fingerprint.")
        if self.mode not in {"normal", "single-attempt", "explicit-range"}:
            raise WorkflowRecordError("invalid processing mode.")
        _strict_string("producer_method", self.producer_method); _strict_string("producer_config", self.producer_config)
        if self.detection_status not in {"complete", "failed", "empty"}:
            raise WorkflowRecordError("invalid detection status.")
        if self.detection_status == "failed":
            if not isinstance(self.detection_error, str) or not self.detection_error.strip():
                raise WorkflowRecordError("failed detection requires an error.")
        elif self.detection_error is not None:
            raise WorkflowRecordError("successful or empty detection cannot have an error.")
        if not isinstance(self.attempts, tuple) or any(not isinstance(a, AttemptOutcome) for a in self.attempts):
            raise WorkflowRecordError("attempts must be a tuple of outcomes.")
        ids = [a.attempt_id for a in self.attempts]
        if len(ids) != len(set(ids)):
            raise WorkflowRecordError("attempt outcome IDs collide.")
        if self.overall_status not in {"complete", "partial", "failed", "empty"}:
            raise WorkflowRecordError("invalid overall status.")
        expected_status = (
            "failed"
            if self.detection_status == "failed"
            else "empty"
            if not self.attempts
            else "failed"
            if all(attempt.status == "failed" for attempt in self.attempts)
            else "partial"
            if any(attempt.status == "failed" for attempt in self.attempts)
            else "complete"
        )
        if self.overall_status != expected_status:
            raise WorkflowRecordError(
                f"overall status must be {expected_status} for recorded outcomes."
            )
        if self.detection_status in {"failed", "empty"} and self.attempts:
            raise WorkflowRecordError(
                "failed or empty detection cannot contain attempt outcomes."
            )
        object.__setattr__(self, "source_fingerprint", fingerprint)

    def to_dict(self) -> dict[str, Any]:
        return {"attempts": [a.to_dict() for a in self.attempts], "detection_error": self.detection_error, "detection_status": self.detection_status, "mode": self.mode, "overall_status": self.overall_status, "producer_config": self.producer_config, "producer_method": self.producer_method, "run_id": self.run_id, "schema_version": self.schema_version, "source_fingerprint": self.source_fingerprint, "source_id": self.source_id}

    @classmethod
    def from_dict(cls, v: dict[str, Any]) -> "WorkflowRunRecord":
        keys = {"attempts", "detection_error", "detection_status", "mode", "overall_status", "producer_config", "producer_method", "run_id", "schema_version", "source_fingerprint", "source_id"}
        if not isinstance(v, dict) or set(v) != keys or not isinstance(v["attempts"], list):
            raise WorkflowRecordError("workflow run record keys are invalid.")
        return cls(v["schema_version"], v["run_id"], v["source_id"], v["source_fingerprint"], v["mode"], v["producer_method"], v["producer_config"], v["detection_status"], v["detection_error"], tuple(AttemptOutcome.from_dict(x) for x in v["attempts"]), v["overall_status"])

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> "WorkflowRunRecord":
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError(f"duplicate object key {key!r}")
                result[key] = value
            return result
        try:
            if isinstance(data, (bytes, bytearray)): data = bytes(data).decode("utf-8")
            return cls.from_dict(json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)), object_pairs_hook=pairs))
        except (TypeError, UnicodeError, ValueError, WorkflowRecordError) as exc:
            if isinstance(exc, WorkflowRecordError): raise
            raise WorkflowRecordError(f"invalid workflow run JSON: {exc}.") from exc
