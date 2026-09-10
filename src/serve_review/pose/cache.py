"""Fingerprinted, versioned, resumable pose JSONL cache (M2.2).

This module owns all pose cache file I/O. Observation validation lives in
:mod:`serve_review.pose.schema`; this module never runs models and never
classifies serves.

File format (JSONL, one object per line, UTF-8):

- line 1: ``{"type": "header", ...identity..., "schema_version": 1,
  "pose_schema_version": 1}`` pinning the source fingerprint, model, and
  sampling schedule;
- lines 2..N+1: ``{"type": "frame", "observation": {...}}`` with strictly
  increasing canonical ``time_seconds``;
- last line (complete caches only): ``{"type": "footer",
  "schema_version": 1, "frame_count": N, "complete": true}``.

Completeness rule: a cache is complete only when its footer is present,
``complete`` is true, and ``frame_count`` equals the number of frame
lines. An interrupted write has no footer and loads as incomplete but
resumable (never complete). Any malformed line, version mismatch, unknown
record type, out-of-order time, or count mismatch raises
:class:`CacheCorruptError` (never complete); the caller may quarantine the
file with :func:`quarantine_corrupt` and re-extract.

Identity rule: :func:`require_matching_identity` raises
:class:`CacheStaleError` when any identity field differs; stale rows are
never treated as valid.

Atomicity: every mutation writes a temporary file beside the final cache
and moves it into place with :func:`os.replace` after flushing and
fsyncing. A crash therefore leaves either the previous resumable cache or
no final file -- never a partial file masquerading as complete.
Temporary files are removed on failure.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve_review.pose.schema import (
    POSE_SCHEMA_VERSION,
    CacheIdentity,
    FrameObservation,
    PoseError,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheError",
    "CacheCorruptError",
    "CacheStaleError",
    "CacheSnapshot",
    "PoseCacheWriter",
    "header_to_dict",
    "header_from_dict",
    "require_matching_identity",
    "load_cache",
    "write_complete_cache",
    "write_partial_cache",
    "append_frames",
    "finalize_cache",
    "quarantine_corrupt",
    "open_cache_writer",
]

#: Version of the JSONL envelope (header/footer framing).
CACHE_SCHEMA_VERSION = 1


class CacheError(Exception):
    """Base error for pose cache failures."""


class CacheCorruptError(CacheError):
    """Raised when a cache file is malformed; it is never valid/complete."""


class CacheStaleError(CacheError):
    """Raised when a cache identity does not match the requested identity."""


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    """Result of :func:`load_cache`.

    ``identity`` is the stored header identity, ``frames`` the stored
    observations in increasing-time order, and ``complete`` is true only
    when a valid complete footer is present.
    """

    identity: CacheIdentity
    frames: tuple[FrameObservation, ...]
    complete: bool


def header_to_dict(identity: CacheIdentity) -> dict[str, Any]:
    """Serialize ``identity`` as a cache header record."""
    if not isinstance(identity, CacheIdentity):
        raise CacheError(
            "invalid cache identity: expected CacheIdentity, "
            f"got {type(identity).__name__}."
        )
    return {
        "model_name": identity.model_name,
        "model_version": identity.model_version,
        "pose_schema_version": POSE_SCHEMA_VERSION,
        "sampling_rate_hz": identity.sampling_rate_hz,
        "sampling_start_seconds": identity.sampling_start_seconds,
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint": identity.source_fingerprint,
        "type": "header",
    }


def header_from_dict(values: dict[str, Any]) -> CacheIdentity:
    """Parse and version-check a header record into a :class:`CacheIdentity`."""
    name = "pose_cache/header"
    if not isinstance(values, dict):
        raise CacheCorruptError(
            f"{name}: header record must be a JSON object, "
            f"got {type(values).__name__}."
        )
    known = {
        "model_name",
        "model_version",
        "pose_schema_version",
        "sampling_rate_hz",
        "sampling_start_seconds",
        "schema_version",
        "source_fingerprint",
        "type",
    }
    unknown = sorted(set(values) - known)
    if unknown:
        raise CacheCorruptError(f"{name}: unknown keys {unknown!r}.")
    if values.get("type") != "header":
        raise CacheCorruptError(
            f"{name}: first record must be a header "
            f"({{'type': 'header', ...}}), got {values.get('type')!r}."
        )
    version = values.get("schema_version")
    if version != CACHE_SCHEMA_VERSION:
        raise CacheCorruptError(
            f"{name}: unsupported cache schema_version {version!r}; "
            f"this build supports version {CACHE_SCHEMA_VERSION}. "
            "The cache is stale: re-extract from the source."
        )
    pose_version = values.get("pose_schema_version")
    if pose_version != POSE_SCHEMA_VERSION:
        raise CacheCorruptError(
            f"{name}: unsupported pose_schema_version {pose_version!r}; "
            f"this build supports version {POSE_SCHEMA_VERSION}. "
            "The cache is stale: re-extract from the source."
        )
    try:
        return CacheIdentity(
            source_fingerprint=values["source_fingerprint"],
            model_name=values["model_name"],
            model_version=values["model_version"],
            sampling_rate_hz=values["sampling_rate_hz"],
            sampling_start_seconds=values["sampling_start_seconds"],
        )
    except (PoseError, KeyError, TypeError) as exc:
        raise CacheCorruptError(f"{name}: invalid identity: {exc}.") from exc


def _footer_to_dict(frame_count: int, *, complete: bool) -> dict[str, Any]:
    return {
        "complete": bool(complete),
        "frame_count": int(frame_count),
        "schema_version": CACHE_SCHEMA_VERSION,
        "type": "footer",
    }


def _frame_to_dict(observation: FrameObservation) -> dict[str, Any]:
    if not isinstance(observation, FrameObservation):
        raise CacheError(
            "invalid frame observation: expected FrameObservation, "
            f"got {type(observation).__name__}."
        )
    return {"observation": observation.to_dict(), "type": "frame"}


def _dumps_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True) + "\n"


def require_matching_identity(
    stored: CacheIdentity, expected: CacheIdentity
) -> None:
    """Raise :class:`CacheStaleError` when any identity field differs."""
    if not isinstance(stored, CacheIdentity):
        raise CacheError(
            "invalid stored identity: expected CacheIdentity, "
            f"got {type(stored).__name__}."
        )
    if not isinstance(expected, CacheIdentity):
        raise CacheError(
            "invalid expected identity: expected CacheIdentity, "
            f"got {type(expected).__name__}."
        )
    fields = (
        "source_fingerprint",
        "model_name",
        "model_version",
        "sampling_rate_hz",
        "sampling_start_seconds",
    )
    mismatched = [
        field
        for field in fields
        if getattr(stored, field) != getattr(expected, field)
    ]
    if mismatched:
        details = ", ".join(
            f"{field} (cached={getattr(stored, field)!r} "
            f"!= requested={getattr(expected, field)!r})"
            for field in mismatched
        )
        raise CacheStaleError(
            "pose cache identity is stale: "
            f"{details}. Re-extract rather than reusing cached rows."
        )


def _check_frame_order(frames: list[FrameObservation], name: str) -> None:
    for earlier, later in zip(frames, frames[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise CacheCorruptError(
                f"{name}: frame times must be strictly increasing, got "
                f"{earlier.time_seconds!r} followed by {later.time_seconds!r}."
            )


def load_cache(path: Path | str) -> CacheSnapshot:
    """Load and validate the cache at ``path``.

    Returns a :class:`CacheSnapshot` with ``complete`` true only for a
    fully validated cache ending in a ``complete: true`` footer whose
    ``frame_count`` matches. A cleanly interrupted cache (valid prefix,
    no footer, or ``complete: false`` footer) returns ``complete`` False
    for resumption. Anything malformed raises :class:`CacheCorruptError`
    and must never be treated as complete.
    """
    target = Path(path).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CacheError(f"pose cache does not exist: {target}.") from exc
    except OSError as exc:
        raise CacheError(
            f"could not read pose cache {target}: {exc}."
        ) from exc
    name = f"pose_cache/{target.name}"
    if not text.strip():
        raise CacheCorruptError(
            f"{name}: cache file is empty; no header record found."
        )
    # A torn final line (interrupted mid-write without a trailing newline
    # would still parse, but a torn write usually leaves invalid JSON).
    # Any unparseable line is corrupt, never silently complete.
    raw_lines = text.split("\n")
    # Drop the single trailing empty string produced by the final newline.
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(raw_lines, start=1):
        if not line.strip():
            raise CacheCorruptError(
                f"{name}: line {lineno} is blank; cache records must be "
                "dense JSONL with no blank lines."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CacheCorruptError(
                f"{name}: line {lineno} is not valid JSON ({exc}); "
                "the cache is corrupt: quarantine and re-extract."
            ) from exc
        if not isinstance(record, dict):
            raise CacheCorruptError(
                f"{name}: line {lineno} must be a JSON object, "
                f"got {type(record).__name__}."
            )
        records.append(record)
    if not records or records[0].get("type") != "header":
        raise CacheCorruptError(
            f"{name}: first record must be a header "
            f"({{'type': 'header', ...}}); the cache is corrupt."
        )
    identity = header_from_dict(records[0])
    frames: list[FrameObservation] = []
    footer: dict[str, Any] | None = None
    for lineno, record in enumerate(records[1:], start=2):
        kind = record.get("type")
        if kind == "footer":
            if footer is not None:
                raise CacheCorruptError(
                    f"{name}: line {lineno} carries a second footer; "
                    "the cache is corrupt."
                )
            footer = record
            continue
        if footer is not None:
            raise CacheCorruptError(
                f"{name}: line {lineno} follows the footer; records must "
                "end at the footer."
            )
        if kind != "frame":
            raise CacheCorruptError(
                f"{name}: line {lineno} has unknown record type {kind!r}; "
                "expected 'frame' or 'footer'."
            )
        if set(record) - {"observation", "type"}:
            raise CacheCorruptError(
                f"{name}: line {lineno} has unknown frame keys "
                f"{sorted(set(record) - {'observation', 'type'})!r}."
            )
        if "observation" not in record:
            raise CacheCorruptError(
                f"{name}: line {lineno} frame record is missing "
                "'observation'."
            )
        observation_payload = record["observation"]
        if not isinstance(observation_payload, dict):
            raise CacheCorruptError(
                f"{name}: line {lineno} observation must be a JSON object."
            )
        try:
            frames.append(FrameObservation.from_dict(observation_payload))
        except PoseError as exc:
            raise CacheCorruptError(
                f"{name}: line {lineno} carries an invalid frame "
                f"observation: {exc}."
            ) from exc
    _check_frame_order(frames, name)
    if footer is None:
        return CacheSnapshot(identity=identity, frames=tuple(frames), complete=False)
    known_footer = {"complete", "frame_count", "schema_version", "type"}
    unknown_footer = sorted(set(footer) - known_footer)
    if unknown_footer:
        raise CacheCorruptError(
            f"{name}: footer has unknown keys {unknown_footer!r}."
        )
    if footer.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheCorruptError(
            f"{name}: unsupported footer schema_version "
            f"{footer.get('schema_version')!r}."
        )
    frame_count = footer.get("frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 0
    ):
        raise CacheCorruptError(
            f"{name}: footer 'frame_count' must be an integer >= 0, "
            f"got {frame_count!r}."
        )
    if frame_count != len(frames):
        raise CacheCorruptError(
            f"{name}: footer frame_count ({frame_count!r}) does not match "
            f"the {len(frames)} stored frame(s); the cache is corrupt."
        )
    if footer.get("complete") is True:
        return CacheSnapshot(identity=identity, frames=tuple(frames), complete=True)
    if footer.get("complete") is False:
        return CacheSnapshot(identity=identity, frames=tuple(frames), complete=False)
    raise CacheCorruptError(
        f"{name}: footer 'complete' must be true or false, "
        f"got {footer.get('complete')!r}."
    )


def _write_snapshot_atomic(
    target: Path, identity: CacheIdentity, frames: list[FrameObservation], *, complete: bool
) -> None:
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CacheError(
                f"could not create pose cache directory {parent}: {exc}."
            ) from exc
    _check_frame_order(list(frames), f"pose_cache/{target.name}")
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=target.name + ".tmp-",
            suffix=".jsonl",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(_dumps_line(header_to_dict(identity)))
            for observation in frames:
                handle.write(_dumps_line(_frame_to_dict(observation)))
            if complete:
                handle.write(
                    _dumps_line(_footer_to_dict(len(frames), complete=True))
                )
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
    except CacheError:
        if tmp_path is not None:
            _remove_quietly(tmp_path)
        raise
    except (OSError, TypeError, ValueError) as exc:
        if tmp_path is not None:
            _remove_quietly(tmp_path)
        raise CacheError(
            f"could not write pose cache to {target}: {exc}."
        ) from exc


def _remove_quietly(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def write_complete_cache(
    path: Path | str,
    identity: CacheIdentity,
    frames: list[FrameObservation] | tuple[FrameObservation, ...],
) -> Path:
    """Atomically write a complete cache with ``frames`` plus a footer."""
    target = Path(path).expanduser()
    items = list(frames)
    for entry in items:
        if not isinstance(entry, FrameObservation):
            raise CacheError(
                "invalid frames: expected FrameObservation entries, "
                f"got {type(entry).__name__}."
            )
    _write_snapshot_atomic(target, identity, items, complete=True)
    return target


def write_partial_cache(
    path: Path | str,
    identity: CacheIdentity,
    frames: list[FrameObservation] | tuple[FrameObservation, ...],
) -> Path:
    """Atomically write an incomplete (resumable, footerless) cache."""
    target = Path(path).expanduser()
    items = list(frames)
    for entry in items:
        if not isinstance(entry, FrameObservation):
            raise CacheError(
                "invalid frames: expected FrameObservation entries, "
                f"got {type(entry).__name__}."
            )
    _write_snapshot_atomic(target, identity, items, complete=False)
    return target


def append_frames(
    path: Path | str,
    identity: CacheIdentity,
    new_frames: list[FrameObservation] | tuple[FrameObservation, ...],
) -> Path:
    """Resume an incomplete cache by atomically appending ``new_frames``.

    The stored header identity must match ``identity`` exactly, else
    :class:`CacheStaleError` is raised and the file is left untouched.
    Appending to a complete cache raises :class:`CacheError`. The result
    stays incomplete (footerless) until :func:`finalize_cache`.
    """
    target = Path(path).expanduser()
    additions = list(new_frames)
    for entry in additions:
        if not isinstance(entry, FrameObservation):
            raise CacheError(
                "invalid new_frames: expected FrameObservation entries, "
                f"got {type(entry).__name__}."
            )
    snapshot = load_cache(target)
    require_matching_identity(snapshot.identity, identity)
    if snapshot.complete:
        raise CacheError(
            f"pose cache {target} is already complete with "
            f"{len(snapshot.frames)} frame(s); refusing to append."
        )
    combined = list(snapshot.frames) + additions
    _check_frame_order(combined, f"pose_cache/{target.name}")
    _write_snapshot_atomic(target, identity, combined, complete=False)
    return target


def finalize_cache(path: Path | str, identity: CacheIdentity) -> Path:
    """Atomically mark a resumable cache complete without adding frames.

    The stored identity must match ``identity`` (:class:`CacheStaleError`
    otherwise). Finalizing an already-complete cache with matching
    identity is a no-op returning the path.
    """
    target = Path(path).expanduser()
    snapshot = load_cache(target)
    require_matching_identity(snapshot.identity, identity)
    if snapshot.complete:
        return target
    _write_snapshot_atomic(target, identity, list(snapshot.frames), complete=True)
    return target


def quarantine_corrupt(path: Path | str) -> Path:
    """Move a corrupt/incompatible cache aside; never delete observations silently.

    The file is renamed beside itself to ``<name>.corrupt`` (with a
    numeric ``.corrupt.N`` suffix when taken) via :func:`os.replace`, so a
    fresh extraction can start. Returns the quarantine path.
    """
    target = Path(path).expanduser()
    if not target.is_file():
        raise CacheError(f"pose cache does not exist: {target}.")
    candidate = target.with_name(target.name + ".corrupt")
    counter = 0
    while candidate.exists():
        counter += 1
        candidate = target.with_name(f"{target.name}.corrupt.{counter}")
    try:
        os.replace(target, candidate)
    except OSError as exc:
        raise CacheError(
            f"could not quarantine corrupt pose cache {target}: {exc}."
        ) from exc
    return candidate


class PoseCacheWriter:
    """Incremental resumable writer with atomic commit.

    Rows are staged in a temporary sibling file; the final cache appears
    (or is replaced) only on :meth:`commit`, which appends the complete
    footer and atomically renames into place. Dropping the writer without
    committing -- or a crash -- leaves the previous final file untouched
    (still incomplete/resumable, never complete) and removes the staging
    file. Use ``resume=True`` to carry forward rows from an existing
    incomplete cache with matching identity.
    """

    def __init__(
        self, path: Path | str, identity: CacheIdentity, *, resume: bool = False
    ) -> None:
        if not isinstance(identity, CacheIdentity):
            raise CacheError(
                "invalid cache identity: expected CacheIdentity, "
                f"got {type(identity).__name__}."
            )
        if not isinstance(resume, bool):
            raise CacheError(
                f"invalid resume: {resume!r}; expected True or False."
            )
        self._target = Path(path).expanduser()
        self._identity = identity
        self._resume = resume
        self._staged: list[FrameObservation] = []
        self._handle: Any = None
        self._tmp_path: str | None = None
        self._committed = False
        self._closed = False
        existing: list[FrameObservation] = []
        if self._target.is_file():
            if not resume:
                raise CacheError(
                    f"pose cache {self._target} already exists; pass "
                    "resume=True to resume a partial cache or remove it first."
                )
            snapshot = load_cache(self._target)
            require_matching_identity(snapshot.identity, identity)
            if snapshot.complete:
                raise CacheError(
                    f"pose cache {self._target} is already complete with "
                    f"{len(snapshot.frames)} frame(s); refusing to resume."
                )
            existing = list(snapshot.frames)
        parent = self._target.parent
        if str(parent) not in ("", "."):
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise CacheError(
                    f"could not create pose cache directory {parent}: {exc}."
                ) from exc
        try:
            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(parent) if str(parent) not in ("", ".") else None,
                prefix=self._target.name + ".tmp-",
                suffix=".jsonl",
                delete=False,
            )
        except OSError as exc:
            raise CacheError(
                f"could not stage pose cache at {self._target}: {exc}."
            ) from exc
        self._handle = handle
        self._tmp_path = handle.name
        try:
            handle.write(_dumps_line(header_to_dict(identity)))
            for observation in existing:
                if not isinstance(observation, FrameObservation):
                    raise CacheError("stored cache holds a non-frame row.")
                handle.write(_dumps_line(_frame_to_dict(observation)))
            handle.flush()
            self._staged = existing
        except (OSError, CacheError) as exc:
            try:
                handle.close()
            except OSError:
                pass
            _remove_quietly(self._tmp_path)
            self._handle = None
            self._tmp_path = None
            if isinstance(exc, CacheError):
                raise
            raise CacheError(
                f"could not stage pose cache at {self._target}: {exc}."
            ) from exc

    @property
    def staged_count(self) -> int:
        """Return the number of staged (written-but-uncommitted) frames."""
        return len(self._staged)

    def append(self, observation: FrameObservation) -> None:
        """Stage one frame observation (still incomplete until commit)."""
        if self._closed:
            raise CacheError("pose cache writer is closed.")
        if self._committed:
            raise CacheError("pose cache writer already committed.")
        if not isinstance(observation, FrameObservation):
            raise CacheError(
                "invalid frame observation: expected FrameObservation, "
                f"got {type(observation).__name__}."
            )
        if self._staged and not (
            observation.time_seconds > self._staged[-1].time_seconds
        ):
            raise CacheError(
                "frame times must be strictly increasing: staged last "
                f"{self._staged[-1].time_seconds!r}, new "
                f"{observation.time_seconds!r}."
            )
        assert self._handle is not None
        try:
            self._handle.write(_dumps_line(_frame_to_dict(observation)))
            self._handle.flush()
        except OSError as exc:
            raise CacheError(
                f"could not stage frame at {observation.time_seconds!r}: {exc}."
            ) from exc
        self._staged.append(observation)

    def commit(self) -> Path:
        """Append the complete footer and atomically publish the cache."""
        if self._closed:
            raise CacheError("pose cache writer is closed.")
        if self._committed:
            raise CacheError("pose cache writer already committed.")
        assert self._handle is not None and self._tmp_path is not None
        try:
            self._handle.write(
                _dumps_line(_footer_to_dict(len(self._staged), complete=True))
            )
            self._handle.flush()
            try:
                os.fsync(self._handle.fileno())
            except OSError:
                pass
            self._handle.close()
            self._handle = None
            os.replace(self._tmp_path, self._target)
        except OSError as exc:
            try:
                if self._handle is not None:
                    self._handle.close()
            except OSError:
                pass
            self._handle = None
            _remove_quietly(self._tmp_path)
            self._tmp_path = None
            self._closed = True
            raise CacheError(
                f"could not commit pose cache to {self._target}: {exc}."
            ) from exc
        self._tmp_path = None
        self._committed = True
        self._closed = True
        return self._target

    def abort(self) -> None:
        """Discard staged rows; the final cache is left untouched."""
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None
        _remove_quietly(self._tmp_path)
        self._tmp_path = None
        self._closed = True

    def __enter__(self) -> PoseCacheWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        # Never auto-commit: an interrupted block must stay incomplete and
        # resumable, never present as complete.
        if exc_type is not None or not self._committed:
            if not self._closed:
                self.abort()
            else:
                # Already committed/closed: nothing staged to discard, and
                # the tmp file (if any) was already published or removed.
                pass
        return False


def open_cache_writer(
    path: Path | str, identity: CacheIdentity, *, resume: bool = False
) -> PoseCacheWriter:
    """Open an incremental :class:`PoseCacheWriter` for ``path``."""
    return PoseCacheWriter(path, identity, resume=resume)
