"""Conservative, read-only cleanup inventory for managed workflow artifacts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from serve_review.domain import AttemptDocument
from serve_review.workflow.records import SourceRecord, WorkflowRunRecord
from serve_review.workflow.sources import load_source
from serve_review.workflow.workspace import WorkspacePaths, resolve_workspace

CLASSES = (
    "recomputable-cache",
    "generated-review-media",
    "failed-run-artifact",
    "stale-artifact",
    "manifest-provenance",
    "user-annotation",
    "source-media",
)
_DELETABLE = set(CLASSES[:4])
_SOURCE_ID = re.compile(r"^source-[0-9a-f]{16}$")
_RUN_FILE = re.compile(r"^(run-[0-9a-f]{16})\.json$")
_GENERATED_NAMES = {
    "attempts.json",
    "checkpoints.json",
    "diagnostics.json",
    "index.html",
    "review.json",
    "shadows.json",
    "source.json",
}
_GENERATED_SUFFIXES = {".html", ".jpeg", ".jpg", ".mov", ".mp4", ".png"}


class CleanupInventoryError(ValueError):
    """Raised when a safe inventory cannot be established."""


@dataclass(frozen=True, slots=True)
class CleanupItem:
    """One classified path; this type never offers a deletion operation."""

    path: Path
    classification: str
    future_deletable: bool
    reason: str
    excluded: bool = False

    def __post_init__(self) -> None:
        if self.classification not in CLASSES:
            raise CleanupInventoryError(
                f"unknown cleanup classification: {self.classification}"
            )
        if self.future_deletable != (self.classification in _DELETABLE):
            raise CleanupInventoryError("cleanup deletability invariant violated")
        if (
            not isinstance(self.path, Path)
            or not isinstance(self.reason, str)
            or not self.reason.strip()
        ):
            raise CleanupInventoryError("cleanup item has invalid path or reason")


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Deterministically ordered inventory, candidates, and exclusions."""

    items: tuple[CleanupItem, ...]
    candidates: tuple[CleanupItem, ...]
    protected: tuple[CleanupItem, ...]

    @property
    def totals(self) -> dict[str, int]:
        return {
            classification: sum(
                item.classification == classification for item in self.items
            )
            for classification in CLASSES
        }


def _contained(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _files(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    if not root.is_dir() or root.is_symlink():
        raise CleanupInventoryError(f"managed path is not a directory: {root}")
    result: list[Path] = []
    try:
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            retained: list[str] = []
            for name in sorted(directories):
                path = current_path / name
                if path.is_symlink():
                    result.append(path)
                else:
                    retained.append(name)
            directories[:] = retained
            result.extend(current_path / name for name in sorted(names))
    except OSError as exc:
        raise CleanupInventoryError(f"could not inspect managed path {root}: {exc}") from exc
    return tuple(sorted(result, key=str))


def _load_registry(
    workspace: WorkspacePaths,
) -> tuple[tuple[SourceRecord, ...], tuple[WorkflowRunRecord, ...]]:
    sources: list[SourceRecord] = []
    if workspace.sources.exists():
        if not workspace.sources.is_dir() or workspace.sources.is_symlink():
            raise CleanupInventoryError(
                f"sources path is not a safe directory: {workspace.sources}"
            )
        for entry in sorted(workspace.sources.iterdir(), key=lambda path: path.name):
            if entry.is_symlink():
                continue
            if not entry.is_dir() or _SOURCE_ID.fullmatch(entry.name) is None:
                raise CleanupInventoryError(f"malformed source entry: {entry}")
            try:
                record = load_source(workspace, entry.name)
            except Exception as exc:
                raise CleanupInventoryError(
                    f"malformed source manifest {entry / 'record.json'}: {exc}"
                ) from exc
            if record.source_id != entry.name:
                raise CleanupInventoryError(
                    f"source manifest ID does not match directory {entry}"
                )
            sources.append(record)

    runs: list[WorkflowRunRecord] = []
    if workspace.runs.exists():
        if not workspace.runs.is_dir() or workspace.runs.is_symlink():
            raise CleanupInventoryError(
                f"runs path is not a safe directory: {workspace.runs}"
            )
        for entry in sorted(workspace.runs.iterdir(), key=lambda path: path.name):
            if entry.is_symlink() or not entry.name.endswith(".json"):
                continue
            match = _RUN_FILE.fullmatch(entry.name)
            if match is None or not entry.is_file():
                raise CleanupInventoryError(f"malformed run entry: {entry}")
            try:
                record = WorkflowRunRecord.from_json(entry.read_bytes())
            except Exception as exc:
                raise CleanupInventoryError(
                    f"malformed run manifest {entry}: {exc}"
                ) from exc
            if record.run_id != match.group(1):
                raise CleanupInventoryError(
                    f"run manifest ID does not match filename {entry}"
                )
            runs.append(record)
    return tuple(sources), tuple(runs)


def _selected_sources(
    selectors: Any, sources: tuple[SourceRecord, ...]
) -> set[str]:
    if selectors is None:
        return set()
    if isinstance(selectors, (str, Path)):
        values = (selectors,)
    else:
        try:
            values = tuple(selectors)
        except TypeError as exc:
            raise CleanupInventoryError("source selectors must be paths or IDs") from exc
    selected: set[str] = set()
    for selector in values:
        text = str(selector)
        try:
            path = str(Path(selector).expanduser().resolve(strict=False))
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            raise CleanupInventoryError(
                f"invalid source selector {selector!r}: {exc}"
            ) from exc
        matches = {
            source.source_id
            for source in sources
            if source.source_id == text or path in source.known_paths
        }
        if not matches:
            raise CleanupInventoryError(
                f"source selector did not match a registered source: {selector}"
            )
        selected.update(matches)
    return selected


def _stale_attempt_documents(
    workspace: WorkspacePaths, sources: tuple[SourceRecord, ...]
) -> set[Path]:
    stale: set[Path] = set()
    for source in sources:
        compatibility = workspace.sources / source.source_id / "compatibility"
        for known_path in source.known_paths:
            path = compatibility / Path(known_path).stem / "attempts.json"
            if not _contained(path, compatibility) or not path.is_file():
                continue
            try:
                document = AttemptDocument.from_json(path.read_bytes())
            except Exception as exc:
                raise CleanupInventoryError(
                    f"malformed compatibility artifact {path}: {exc}"
                ) from exc
            if document.source_fingerprint != source.source_fingerprint:
                stale.add(path)
    return stale


def inventory_cleanup(
    workspace: WorkspacePaths | str | Path,
    *,
    source: Any = None,
    stale: bool = False,
    failed_runs: bool = False,
    temporary: bool = False,
) -> CleanupResult:
    """Return a deterministic inventory without changing the filesystem."""

    resolved = (
        workspace
        if isinstance(workspace, WorkspacePaths)
        else resolve_workspace(workspace)
    )
    sources, runs = _load_registry(resolved)
    selected = _selected_sources(source, sources)
    stale_paths = _stale_attempt_documents(resolved, sources)
    source_by_id = {record.source_id: record for record in sources}
    external_sources = {
        Path(path): record.source_id
        for record in sources
        for path in record.known_paths
    }
    run_artifacts = {
        Path(path): (run.source_id, run.overall_status)
        for run in runs
        for attempt in run.attempts
        for path in attempt.artifacts
    }
    failed_run_owners = {
        run.run_id: run.source_id for run in runs if run.overall_status == "failed"
    }
    category_filter = stale or failed_runs or temporary
    managed_roots = (
        resolved.sources,
        resolved.sessions,
        resolved.collections,
        resolved.runs,
        resolved.temporary,
    )
    items: dict[str, CleanupItem] = {}

    for root in managed_roots:
        for path in _files(root):
            classification = "manifest-provenance"
            reason = "unknown managed path is protected"
            source_id: str | None = None
            relative = path.absolute().relative_to(resolved.root.absolute())
            parts = relative.parts

            if path.is_symlink() or not _contained(path, resolved.root):
                reason = "symlink or escaping path is protected; target was not followed"
            elif path in external_sources:
                classification = "source-media"
                source_id = external_sources[path]
                reason = "registered source media is always protected"
            elif (
                path.name == "record.json"
                or (parts[0] in {"sessions", "collections"} and path.suffix == ".json")
                or (parts[0] == "runs" and path.suffix == ".json")
            ):
                reason = "manifest and run provenance are always protected"
            elif "annotation" in path.name.lower() or any(
                "annotation" in part.lower() for part in parts
            ):
                classification = "user-annotation"
                reason = "user annotation is always protected"
            elif path in run_artifacts:
                source_id, run_status = run_artifacts[path]
                classification = (
                    "failed-run-artifact"
                    if run_status == "failed"
                    else "recomputable-cache"
                    if "cache" in parts
                    else "generated-review-media"
                )
                reason = (
                    "artifact is explicitly named by a failed workflow run"
                    if run_status == "failed"
                    else "artifact is explicitly named by workflow provenance"
                )
            elif (
                parts[0] == "runs"
                and len(parts) > 2
                and parts[1] in failed_run_owners
            ):
                source_id = failed_run_owners[parts[1]]
                classification = "failed-run-artifact"
                reason = "artifact belongs to a failed run; provenance remains protected"
            elif path in stale_paths:
                source_id = parts[1] if len(parts) > 1 else None
                classification = "stale-artifact"
                reason = "strict fingerprint mismatch proves artifact stale"
            elif parts[0] == "temporary":
                classification = "generated-review-media"
                reason = "replaceable temporary review artifact"
            elif parts[0] == "sources" and len(parts) > 1:
                source_id = parts[1] if parts[1] in source_by_id else None
                if source_id is not None and "cache" in parts:
                    classification = "recomputable-cache"
                    reason = "cache beneath a registered source is recomputable"
                elif source_id is not None and (
                    path.name in _GENERATED_NAMES
                    or path.suffix.lower() in _GENERATED_SUFFIXES
                ):
                    classification = "generated-review-media"
                    reason = "known application-generated source artifact"
            elif parts[0] in {"sessions", "collections"} and path.suffix == ".html":
                classification = "generated-review-media"
                reason = "application-generated organization review page"

            class_deletable = classification in _DELETABLE
            source_matches = not selected or source_id in selected
            category_matches = (
                not category_filter
                or (stale and classification == "stale-artifact")
                or (failed_runs and classification == "failed-run-artifact")
                or (temporary and parts[0] == "temporary")
            )
            excluded = not class_deletable or not source_matches or not category_matches
            if class_deletable and excluded:
                reason += "; excluded by selector"
            item = CleanupItem(
                path,
                classification,
                class_deletable,
                reason,
                excluded,
            )
            items[str(path)] = item

    for path, source_id in external_sources.items():
        if str(path) in items or (selected and source_id not in selected):
            continue
        items[str(path)] = CleanupItem(
            path,
            "source-media",
            False,
            "registered source media is protected and external",
            True,
        )

    ordered = tuple(items[key] for key in sorted(items))
    candidates = tuple(
        item
        for item in ordered
        if item.future_deletable and not item.excluded
    )
    protected = tuple(
        item
        for item in ordered
        if not item.future_deletable or item.excluded
    )
    return CleanupResult(ordered, candidates, protected)
