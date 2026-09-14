"""Public workflow command parsers and orchestration."""

from __future__ import annotations

import argparse
import math
import secrets
import sys
import webbrowser
from pathlib import Path
from typing import Any

from serve_review.domain import MediaRange
from serve_review.media.probe import fingerprint_for_path
from serve_review.workflow.landing import render_batch_index, render_landing_page
from serve_review.workflow.planning import PlanningError, build_process_plan
from serve_review.workflow.processing import process_registered_source
from serve_review.workflow.records import (
    ArtifactRecord,
    AttemptRecord,
    LandingPage,
    SourceRecord,
    source_id_for_fingerprint,
)
from serve_review.workflow.sources import (
    load_source,
    register_source,
    source_record_path,
)
from serve_review.workflow.workspace import (
    WorkspacePaths,
    ensure_workspace,
    resolve_workspace,
)


def _range(value: str) -> MediaRange:
    try:
        parts = value.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError
        start, end = (float(part) for part in parts)
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError
        return MediaRange(start, end)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a finite half-open range START:END with 0 <= START < END"
        ) from exc


def _padding(value: str) -> float:
    try:
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a finite nonnegative number"
        ) from exc


def add_workflow_parsers(subparsers: Any) -> None:
    """Add workflow commands without changing established command parsers."""

    parser = subparsers.add_parser(
        "process", help="discover and process source videos"
    )
    parser.add_argument("targets", nargs="+", type=Path, metavar="TARGET")
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--session", metavar="NAME")
    parser.add_argument("--single-attempt", action="store_true")
    parser.add_argument(
        "--range", dest="explicit_range", type=_range, metavar="START:END"
    )
    parser.add_argument("--padding", type=_padding, default=1.0)
    parser.add_argument("--open", dest="open_page", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(handler=process_command)


def _id(prefix: str) -> str:
    return prefix + secrets.token_hex(8)


def _kind(path: str) -> str:
    lowered = path.lower()
    if lowered.endswith(".json"):
        return "json"
    if lowered.endswith(".html"):
        return "review"
    if lowered.endswith((".mov", ".mp4")):
        return "clip"
    return "other"


def _lookup(workspace: WorkspacePaths, fingerprint: str) -> SourceRecord | None:
    source_id = source_id_for_fingerprint(fingerprint)
    path = source_record_path(workspace, source_id)
    if not path.exists():
        return None
    return load_source(workspace, source_id)


def _page_for(record: SourceRecord, result: Any) -> LandingPage:
    attempts = []
    for outcome in result.attempts:
        artifacts = tuple(
            ArtifactRecord(Path(path).name, path, _kind(path), "available")
            for path in outcome.artifacts
        )
        attempts.append(
            AttemptRecord(
                outcome.attempt_id,
                outcome.attempt_id,
                "available" if outcome.status == "complete" else "failed",
                outcome.error,
                artifacts,
            )
        )
    return LandingPage(record, (), tuple(attempts))


def _source_page_path(
    workspace: WorkspacePaths,
    record: SourceRecord,
    *,
    batch: bool,
    session_id: str | None,
) -> Path:
    if session_id is not None:
        return workspace.sessions / f"{session_id}-{record.source_id}.html"
    if batch:
        return workspace.temporary / f"{record.source_id}.html"
    return workspace.sources / record.source_id / "index.html"


def process_command(
    args: argparse.Namespace,
    *,
    services: dict[str, Any] | None = None,
) -> int:
    """Execute one process command with injectable boundary collaborators."""

    services = {} if services is None else services
    try:
        workspace = resolve_workspace(args.workspace)
        print(f"Workspace: {workspace.root}")
        fingerprint_fn = services.get("fingerprint", fingerprint_for_path)
        source_lookup = services.get(
            "source_lookup", lambda value: _lookup(workspace, value)
        )
        plan = build_process_plan(
            args.targets,
            recursive=args.recursive,
            single_attempt=args.single_attempt,
            explicit_range=args.explicit_range,
            fingerprint_fn=fingerprint_fn,
            source_lookup=source_lookup,
            state_inspector=services.get("state_inspector"),
        )
    except (PlanningError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(plan.human())
    if args.dry_run:
        return 0

    try:
        services.get("ensure_workspace", ensure_workspace)(workspace)
    except Exception as exc:
        print(f"ERROR: could not create workspace: {exc}", file=sys.stderr)
        return 1

    register = services.get("register_source", register_source)
    registered: list[SourceRecord] = []
    operational_failures = 0
    for source_plan in plan.sources:
        record: SourceRecord | None = None
        source_failed = False
        for path in source_plan.source.known_paths:
            try:
                kwargs: dict[str, Any] = {}
                if "probe_fn" in services:
                    kwargs["probe_fn"] = services["probe_fn"]
                record = register(Path(path), workspace, **kwargs)
            except Exception as exc:
                source_failed = True
                print(f"ERROR: registration failed for {path}: {exc}", file=sys.stderr)
        if record is None:
            operational_failures += 1
            continue
        if source_failed:
            operational_failures += 1
        registered.append(record)
        print(f"Registered: {record.source_id}")

    session = None
    if args.session is not None and registered:
        from serve_review.workflow.sessions import (
            add_source,
            create_session,
            resolve_session,
        )

        resolve_session_fn = services.get("resolve_session", resolve_session)
        create_session_fn = services.get("create_session", create_session)
        add_source_fn = services.get("add_session_source", add_source)
        try:
            try:
                session = resolve_session_fn(workspace, args.session)
            except Exception as exc:
                if "not found" not in str(exc):
                    raise
                session = create_session_fn(
                    workspace,
                    args.session,
                    id_factory=services.get(
                        "session_id_factory", lambda: _id("session-")
                    ),
                )
            for record in registered:
                if record.source_id not in session.source_ids:
                    session = add_source_fn(
                        workspace, session.session_id, record.source_id
                    )
            print(f"Session: {session.session_id}")
        except Exception as exc:
            print(f"ERROR: session validation failed: {exc}", file=sys.stderr)
            return 2

    processed: list[tuple[SourceRecord, Any]] = []
    process_fn = services.get("process_source", process_registered_source)
    for record in registered:
        print(f"Processing: {record.source_id}")
        try:
            result = process_fn(
                record,
                workspace,
                plan.mode,
                explicit_range=plan.explicit_range,
                padding_seconds=args.padding,
                run_id_factory=services.get(
                    "run_id_factory", lambda: _id("run-")
                ),
                producer_method=services.get(
                    "producer_method", "serve-review-process-v1"
                ),
                producer_config=services.get(
                    "producer_config", f"padding={args.padding}"
                ),
                **(
                    {"run_cut_fn": services["run_cut_fn"]}
                    if "run_cut_fn" in services
                    else {}
                ),
                **(
                    {"run_analyze_fn": services["run_analyze_fn"]}
                    if "run_analyze_fn" in services
                    else {}
                ),
            )
        except Exception as exc:
            operational_failures += 1
            print(f"ERROR: processing failed for {record.source_id}: {exc}", file=sys.stderr)
            continue
        processed.append((record, result))
        print(f"Result: {record.source_id} {result.overall_status}")
        for outcome in result.attempts:
            for path in outcome.artifacts:
                print(f"Artifact: {path}")

    batch = len(plan.sources) != 1
    session_id = session.session_id if session is not None else None
    pages: list[tuple[str, Path]] = []
    rendered_statuses: list[str] = []
    render = services.get("render", render_landing_page)
    for record, result in processed:
        try:
            destination = _source_page_path(
                workspace, record, batch=batch, session_id=session_id
            )
            rendered = Path(render(_page_for(record, result), destination))
        except Exception as exc:
            operational_failures += 1
            print(f"ERROR: landing failed for {record.source_id}: {exc}", file=sys.stderr)
            continue
        pages.append((record.source_id, rendered))
        rendered_statuses.append(result.overall_status)
        print(f"Landing: {rendered}")

    primary: Path | None = None
    needs_index = batch or len(pages) > 1 or (not plan.sources and not args.session)
    if needs_index:
        destination = (
            workspace.sessions / f"{session_id}-index.html"
            if session_id is not None
            else workspace.temporary / "index.html"
        )
        try:
            primary = Path(
                services.get("render_batch", render_batch_index)(pages, destination)
            )
        except Exception as exc:
            operational_failures += 1
            print(f"ERROR: batch landing failed: {exc}", file=sys.stderr)
    elif pages:
        primary = pages[0][1]

    if primary is not None:
        print(f"Primary landing: {primary}")
        print(f"Recovery: serve-review status --workspace {workspace.root}")
        if args.open_page:
            try:
                services.get("opener", lambda path: webbrowser.open(path.as_uri()))(
                    primary.resolve()
                )
            except Exception as exc:
                operational_failures += 1
                print(f"ERROR: could not open landing page: {exc}", file=sys.stderr)

    has_success = any(
        status in {"complete", "empty", "partial"} for status in rendered_statuses
    )
    fully_complete = (
        len(rendered_statuses) == len(plan.sources)
        and all(status in {"complete", "empty"} for status in rendered_statuses)
        and operational_failures == 0
        and primary is not None
    )
    if fully_complete or (not plan.sources and primary is not None and operational_failures == 0):
        return 0
    if has_success:
        return 3
    return 1
