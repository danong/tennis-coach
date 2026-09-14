"""Pure target discovery and read-only process planning."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from serve_review.domain import MediaRange
from serve_review.workflow.records import (
    PLAN_SCHEMA_VERSION,
    DiscoveredSource,
    ProcessPlan,
    SourcePlan,
    SourceRecord,
    StepPlan,
    WorkflowRecordError,
    source_id_for_fingerprint,
)

SUPPORTED_SUFFIXES = (".mov", ".mp4")
_STEP_NAMES = ("probe", "detection", "checkpoint_analysis", "landing_page")
_STEP_STATES = {"required", "reusable", "unknown"}


class PlanningError(ValueError):
    """Raised for invalid targets, modes, or collaborator results."""


def _path(value: str | os.PathLike[str]) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise PlanningError(f"invalid target path {value!r}: {exc}") from exc


def discover_targets(
    targets: str
    | os.PathLike[str]
    | Sequence[str | os.PathLike[str]],
    *,
    recursive: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return supported files and separately reported unsupported entries."""

    if isinstance(targets, (str, os.PathLike)):
        values: Sequence[str | os.PathLike[str]] = (targets,)
    elif isinstance(targets, Sequence) and not isinstance(targets, (bytes, bytearray)):
        values = targets
    else:
        raise PlanningError("targets must be a path or sequence of paths.")
    if not values:
        raise PlanningError("at least one explicit target is required.")

    files: set[str] = set()
    unsupported: set[str] = set()
    for raw in values:
        target = _path(raw)
        try:
            exists = target.exists()
            is_file = target.is_file()
            is_directory = target.is_dir()
        except OSError as exc:
            raise PlanningError(f"could not inspect target {target}: {exc}") from exc
        if not exists:
            raise PlanningError(f"target does not exist: {target}")
        if is_file:
            if target.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise PlanningError(f"unsupported target type: {target}")
            files.add(str(target))
            continue
        if not is_directory:
            raise PlanningError(f"target is not a file or directory: {target}")

        try:
            if recursive:
                candidates = sorted(target.rglob("*"), key=lambda path: str(path))
            else:
                candidates = sorted(target.iterdir(), key=lambda path: str(path))
        except OSError as exc:
            raise PlanningError(f"could not inspect directory {target}: {exc}") from exc
        for candidate in candidates:
            try:
                if candidate.is_dir():
                    if not recursive:
                        unsupported.add(str(candidate.resolve(strict=False)))
                    continue
                canonical = str(candidate.resolve(strict=False))
                if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.add(canonical)
                else:
                    unsupported.add(canonical)
            except (OSError, RuntimeError) as exc:
                raise PlanningError(
                    f"could not inspect directory entry {candidate}: {exc}"
                ) from exc
    return tuple(sorted(files)), tuple(sorted(unsupported))


def _fingerprint(value: Any, path: str) -> str:
    try:
        source_id_for_fingerprint(value)
    except WorkflowRecordError as exc:
        raise PlanningError(f"fingerprint for {path} is invalid: {exc}") from exc
    return "sha256:" + value.split(":", 1)[1].lower()


def _states_for(
    record: SourceRecord,
    inspector: Callable[[SourceRecord], Mapping[str, str] | None] | None,
) -> tuple[str, str, str, str]:
    if inspector is None:
        return ("unknown",) * 4
    try:
        result = inspector(record)
    except Exception as exc:
        raise PlanningError(
            f"could not inspect registered source {record.source_id}: {exc}"
        ) from exc
    if result is None:
        return ("unknown",) * 4
    if not isinstance(result, Mapping):
        raise PlanningError("state inspector must return a mapping or None.")
    unknown_keys = set(result) - set(_STEP_NAMES)
    if unknown_keys:
        raise PlanningError(
            f"state inspector returned unknown steps: {sorted(unknown_keys)!r}."
        )
    states = tuple(result.get(name, "unknown") for name in _STEP_NAMES)
    if any(state not in _STEP_STATES for state in states):
        raise PlanningError("state inspector returned an invalid state.")
    return states  # type: ignore[return-value]


def build_process_plan(
    targets: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    recursive: bool = False,
    single_attempt: bool = False,
    explicit_range: MediaRange | None = None,
    fingerprint_fn: Callable[[str], str],
    source_lookup: Callable[[str], SourceRecord | None],
    state_inspector: Callable[[SourceRecord], Mapping[str, str] | None] | None = None,
) -> ProcessPlan:
    """Discover, deduplicate, and conservatively classify work without writes."""

    if not isinstance(single_attempt, bool):
        raise PlanningError("single_attempt must be boolean.")
    if single_attempt and explicit_range is not None:
        raise PlanningError(
            "--single-attempt and explicit range are mutually exclusive."
        )
    if explicit_range is not None and not isinstance(explicit_range, MediaRange):
        raise PlanningError("explicit_range must be MediaRange.")

    files, unsupported = discover_targets(targets, recursive=recursive)
    grouped: dict[str, list[str]] = {}
    for path in files:
        try:
            raw_fingerprint = fingerprint_fn(path)
        except Exception as exc:
            raise PlanningError(f"could not fingerprint {path}: {exc}") from exc
        fingerprint = _fingerprint(raw_fingerprint, path)
        grouped.setdefault(fingerprint, []).append(path)

    source_plans: list[SourcePlan] = []
    groups = sorted(grouped.items(), key=lambda item: min(item[1]))
    for fingerprint, paths in groups:
        aliases = tuple(sorted(paths))
        discovered = DiscoveredSource(fingerprint, aliases[0], aliases)
        try:
            registered = source_lookup(fingerprint)
        except Exception as exc:
            raise PlanningError(
                f"could not look up source {fingerprint}: {exc}"
            ) from exc
        if registered is not None and not isinstance(registered, SourceRecord):
            raise PlanningError("source lookup must return SourceRecord or None.")
        if registered is not None and registered.source_fingerprint != fingerprint:
            raise PlanningError(
                "source lookup returned a fingerprint mismatch."
            )
        states = (
            ("required",) * 4
            if registered is None
            else _states_for(registered, state_inspector)
        )
        source_plans.append(
            SourcePlan(
                discovered,
                registered,
                *(StepPlan(state) for state in states),
            )
        )

    mode = (
        "explicit-range"
        if explicit_range is not None
        else "single-attempt" if single_attempt else "normal"
    )
    if mode != "normal" and len(source_plans) != 1:
        raise PlanningError(
            "single-attempt and explicit-range require exactly one unique source."
        )
    return ProcessPlan(
        PLAN_SCHEMA_VERSION,
        mode,
        explicit_range,
        tuple(source_plans),
        unsupported,
    )
