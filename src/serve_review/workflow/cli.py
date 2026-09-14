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
from serve_review.workflow.status import project_status, render_human


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

    status = subparsers.add_parser("status", help="show read-only workflow status")
    status.add_argument("source", nargs="?", metavar="SOURCE")
    status.add_argument("--workspace", type=Path, default=None)
    selectors = status.add_mutually_exclusive_group()
    selectors.add_argument("--session", metavar="SELECTOR")
    selectors.add_argument("--collection", metavar="SELECTOR")
    status.add_argument("--json", action="store_true", dest="json_output")
    status.set_defaults(handler=status_command)

    # Organization commands deliberately remain thin wrappers around the
    # already-tested manifest stores.  They do not know anything about media
    # processing or artifact ownership.
    session = subparsers.add_parser("session", help="organize recording sessions")
    session_parsers = session.add_subparsers(dest="session_command", required=True)
    _add_organization_parsers(session_parsers, "session")

    collection = subparsers.add_parser("collection", help="organize explicit collections")
    collection_parsers = collection.add_subparsers(dest="collection_command", required=True)
    _add_organization_parsers(collection_parsers, "collection")


def status_command(args: argparse.Namespace) -> int:
    """Render status without invoking any workflow producer or writer."""
    try:
        workspace = resolve_workspace(args.workspace)
        kind = None
        value = args.source
        if args.source is not None and (args.session is not None or args.collection is not None):
            raise ValueError("SOURCE, --session, and --collection are mutually exclusive")
        if args.session is not None: kind, value = "session", args.session
        elif args.collection is not None: kind, value = "collection", args.collection
        if args.source is not None:
            if args.source.startswith("source-"): kind = "source_id"
            else: kind = "source_path"
        document = project_status(workspace, kind, value)
        if args.json_output: print(document.to_json(), end="")
        else: print(render_human(document))
        return 2 if document.errors else 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _add_organization_parsers(parsers: Any, kind: str) -> None:
    """Register the public session/collection grammar."""
    names = ("list", "show", "create", "add", "remove", "tag")
    if kind == "collection":
        names += ("delete",)
    for command in names:
        p = parsers.add_parser(command)
        p.add_argument("--workspace", type=Path, default=None)
        if command == "list":
            p.set_defaults(handler=organization_command, organization=kind, organization_action=command)
        elif command in {"show", "create", "delete"}:
            p.add_argument("selector_or_name")
            p.set_defaults(handler=organization_command, organization=kind, organization_action=command)
        elif command == "tag":
            p.add_argument("selector")
            p.add_argument("--tag", action="append", required=True)
            p.set_defaults(handler=organization_command, organization=kind, organization_action=command)
        elif command == "add":
            p.add_argument("selector")
            p.add_argument("videos", nargs=("*" if kind == "collection" else "+"), type=Path, metavar="VIDEO")
            if kind == "collection":
                p.add_argument("--attempt", action="append", default=[])
            p.set_defaults(handler=organization_command, organization=kind, organization_action=command)
        elif command == "remove":
            p.add_argument("selector")
            p.add_argument("--source", action="append", default=[])
            if kind == "collection":
                p.add_argument("--attempt", action="append", default=[])
            p.set_defaults(handler=organization_command, organization=kind, organization_action=command)


def _id(prefix: str) -> str:
    return prefix + secrets.token_hex(8)


def _organization_error(exc: Exception) -> int:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 2


def _org_path(workspace: WorkspacePaths, kind: str, identifier: str) -> Path:
    if kind == "session":
        from serve_review.workflow.sessions import session_record_path
        return session_record_path(workspace, identifier)
    from serve_review.workflow.collections import collection_record_path
    return collection_record_path(workspace, identifier)


def _print_org(record: Any, kind: str, *, detail: bool = False) -> None:
    ident = record.session_id if kind == "session" else record.collection_id
    print(f"{ident} {record.display_name}")
    if detail:
        print(f"id: {ident}")
        print(f"name: {record.display_name}")
        print("tags: " + (" ".join(record.tags) if record.tags else "-"))
        print("sources: " + (" ".join(record.source_ids) if record.source_ids else "-"))
        if kind == "collection":
            print("attempts: " + (" ".join(record.attempt_ids) if record.attempt_ids else "-"))
    else:
        members = len(record.source_ids) + (len(record.attempt_ids) if kind == "collection" else 0)
        print(f"members: {members}")


def _register_videos(args: argparse.Namespace, workspace: WorkspacePaths, services: dict[str, Any]) -> list[str]:
    register = services.get("register_source", register_source)
    ids: list[str] = []
    for video in args.videos:
        try:
            kwargs = {"probe_fn": services["probe_fn"]} if "probe_fn" in services else {}
            record = register(Path(video), workspace, **kwargs)
            ids.append(record.source_id)
            print(f"Registered: {record.source_id}")
        except Exception as exc:
            raise ValueError(f"registration failed for {video}: {exc}") from exc
    return ids


def organization_command(args: argparse.Namespace, *, services: dict[str, Any] | None = None) -> int:
    """Run one session or explicit-collection command."""
    services = {} if services is None else services
    try:
        workspace = resolve_workspace(args.workspace)
        kind = args.organization
        if kind == "session":
            from serve_review.workflow import sessions as store
            records_fn = services.get("list_sessions", store.list_sessions)
            resolve_fn = services.get("resolve_session", store.resolve_session)
        else:
            from serve_review.workflow import collections as store
            records_fn = services.get("list_collections", store.list_collections)
            resolve_fn = services.get("resolve_collection", store.resolve_collection)

        action = args.organization_action
        if action == "list":
            for record in records_fn(workspace):
                _print_org(record, kind)
            return 0
        if action == "show":
            record = resolve_fn(workspace, args.selector_or_name)
            _print_org(record, kind, detail=True)
            return 0
        if action == "create":
            factory = services.get(f"{kind}_id_factory", lambda: _id(kind + "-"))
            creator = services.get("create_session", store.create_session) if kind == "session" else services.get("create_collection", store.create_collection)
            record = creator(workspace, args.selector_or_name, id_factory=factory)
            print(f"Created: {record.session_id if kind == 'session' else record.collection_id}")
            print(f"Manifest: {_org_path(workspace, kind, record.session_id if kind == 'session' else record.collection_id)}")
            return 0
        if action == "delete":
            record = resolve_fn(workspace, args.selector_or_name)
            ident = record.collection_id
            services.get("delete_collection", store.delete_collection)(workspace, ident)
            print(f"Deleted: {ident}")
            print(f"Manifest: {_org_path(workspace, kind, ident)}")
            return 0
        record = resolve_fn(workspace, args.selector)
        ident = record.session_id if kind == "session" else record.collection_id
        if action == "tag":
            tags = tuple(sorted(set(record.tags).union(args.tag)))
            tag_fn = (
                services.get("set_session_tags", store.set_session_tags)
                if kind == "session"
                else services.get("set_collection_tags", store.set_collection_tags)
            )
            updated = tag_fn(workspace, ident, tags)
        elif action == "add":
            source_ids = _register_videos(args, workspace, services)
            if kind == "session":
                updated = record
                for source_id in source_ids:
                    updated = services.get("add_session_source", store.add_source)(workspace, ident, source_id)
            else:
                if not source_ids and not args.attempt:
                    raise ValueError("collection add requires at least one VIDEO or --attempt reference")
                updated = record
                for source_id in source_ids:
                    updated = services.get("add_collection_source", store.add_source)(workspace, ident, source_id)
                for attempt_id in args.attempt:
                    updated = services.get("add_collection_attempt", store.add_attempt)(workspace, ident, attempt_id)
        elif action == "remove":
            if not args.source and (kind != "collection" or not args.attempt):
                raise ValueError("remove requires at least one --source or --attempt reference")
            updated = record
            for source_id in args.source:
                updated = (services.get("remove_session_source", store.remove_source)(workspace, ident, source_id)
                           if kind == "session" else services.get("remove_collection_source", store.remove_source)(workspace, ident, source_id))
            if kind == "collection":
                for attempt_id in args.attempt:
                    updated = services.get("remove_collection_attempt", store.remove_attempt)(workspace, ident, attempt_id)
        else:
            raise ValueError(f"unsupported {kind} command {action}")
        print(f"Updated: {ident}")
        print(f"Manifest: {_org_path(workspace, kind, ident)}")
        return 0
    except Exception as exc:
        return _organization_error(exc)


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
