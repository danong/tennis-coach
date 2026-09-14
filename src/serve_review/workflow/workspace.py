"""Managed workspace paths and atomic JSON manifest I/O."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkspaceError(Exception):
    """Raised when workspace resolution or manifest I/O fails."""


def _path(value: object, label: str) -> Path:
    try:
        raw = os.fspath(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise WorkspaceError(f"{label} is not a supported path: {value!r}.") from exc
    if isinstance(raw, bytes):
        raise WorkspaceError(f"{label} is not a supported path: bytes are not allowed.")
    if not isinstance(raw, str) or not raw:
        raise WorkspaceError(f"{label} must be a non-empty path.")
    try:
        return Path(raw).expanduser()
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise WorkspaceError(f"{label} is not a supported path: {value!r}.") from exc


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """One workspace root with managed child paths derived from it."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _path(self.root, "workspace root"))

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def collections(self) -> Path:
        return self.root / "collections"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def temporary(self) -> Path:
        return self.root / "temporary"

    @property
    def trash(self) -> Path:
        return self.root / "trash"

    @property
    def managed_directories(self) -> tuple[Path, ...]:
        return (
            self.root,
            self.sources,
            self.sessions,
            self.collections,
            self.runs,
            self.temporary,
            self.trash,
        )


def resolve_workspace(
    explicit: object | None = None,
    environ: Mapping[str, object] | None = None,
    home: object | None = None,
) -> WorkspacePaths:
    """Resolve workspace paths without accessing or modifying the filesystem."""

    if explicit is not None:
        root = _path(explicit, "explicit workspace")
    else:
        env: Mapping[str, object] = os.environ if environ is None else environ
        if not isinstance(env, Mapping):
            raise WorkspaceError("environment must be a mapping.")
        if "SERVE_REVIEW_WORKSPACE" in env:
            root = _path(env["SERVE_REVIEW_WORKSPACE"], "SERVE_REVIEW_WORKSPACE")
        elif "XDG_DATA_HOME" in env:
            root = _path(env["XDG_DATA_HOME"], "XDG_DATA_HOME") / "serve-review"
        else:
            root = _path(Path.home() if home is None else home, "home")
            root = root / ".local" / "share" / "serve-review"
    return WorkspacePaths(root)


def ensure_workspace(workspace: WorkspacePaths) -> WorkspacePaths:
    """Create the workspace root and exactly its six managed children."""

    if not isinstance(workspace, WorkspacePaths):
        raise WorkspaceError("ensure_workspace requires WorkspacePaths.")
    for directory in workspace.managed_directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspaceError(
                f"could not create workspace directory {directory}: {exc}."
            ) from exc
    return workspace


def _remove_temporary(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def write_json_atomic(destination: object, data: Any) -> Path:
    """Write deterministic UTF-8 JSON through a unique sibling temporary file."""

    target = _path(destination, "JSON destination")
    try:
        text = json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkspaceError(f"could not encode JSON for {target}: {exc}.") from exc

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkspaceError(
            f"could not create JSON destination directory {target.parent}: {exc}."
        ) from exc

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(target.parent),
            prefix=target.name + ".tmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception as exc:
        _remove_temporary(temporary)
        raise WorkspaceError(f"could not atomically write JSON to {target}: {exc}.") from exc
    return target


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def read_json(source: object) -> dict[str, Any]:
    """Read a strict UTF-8 JSON manifest with an object root."""

    path = _path(source, "JSON source")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_nonstandard_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise WorkspaceError(f"could not read JSON manifest {path}: {exc}.") from exc
    if not isinstance(value, dict):
        raise WorkspaceError(f"JSON manifest {path} must contain an object.")
    return value
