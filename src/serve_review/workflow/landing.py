"""Deterministic, side-effect-limited static landing pages."""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

from .records import ArtifactRecord, LandingPage


class LandingRenderError(Exception):
    """Raised when a landing page cannot be rendered or published."""


def _text(value: object) -> str:
    return html.escape(str(value), quote=True)


def _url(path: str, destination: Path) -> str:
    relative = os.path.relpath(path, start=str(destination.parent))
    return quote(relative, safe="/")


def _artifact_sort_key(artifact: ArtifactRecord) -> tuple[object, ...]:
    return (artifact.label, artifact.kind, artifact.path, artifact.status,
            artifact.stage or "", artifact.time_seconds if artifact.time_seconds is not None else -1,
            artifact.attempt_label or "")


def _artifact_html(artifact: ArtifactRecord, destination: Path) -> str:
    link = _url(artifact.path, destination)
    label = _text(artifact.label)
    status = _text(artifact.status)
    details = [f'<span class="status">{status}</span>']
    if artifact.remediation is not None:
        details.append(f'<span class="remediation">{_text(artifact.remediation)}</span>')
    detail = " — ".join(details)
    anchor = f'<a href="{link}">{label}</a>'
    if artifact.kind in {"clip", "compilation"} and artifact.status == "available":
        return (f"<li>{anchor} ({detail})<br>"
                f'<video controls preload="metadata"><source src="{link}">'
                f"Your browser may not support this codec. <a href=\"{link}\">"
                f"Direct media link</a>.</video></li>")
    if (artifact.kind == "checkpoint" and artifact.status == "available"
            and artifact.path.lower().endswith((".jpg", ".jpeg"))):
        caption = (f"{_text(artifact.attempt_label)} — stage {_text(artifact.stage)} — "
                   f"checkpoint {_text(str(artifact.time_seconds))} seconds")
        return (f"<li>{anchor} ({detail})<br>"
                f'<img src="{link}" alt="{caption}"><p>{caption}</p></li>')
    return f"<li>{anchor} ({detail})</li>"


def render_batch_index(entries: object, destination: object) -> Path:
    """Publish an explicit, deterministic index for multiple source pages."""

    temporary: Path | None = None
    try:
        target = Path(destination).expanduser()
        values = sorted(
            ((str(label), str(path)) for label, path in entries),
            key=lambda value: (value[0], value[1]),
        )
        links = [
            f'<li><a href="{quote(os.path.relpath(path, target.parent), safe="/")}">'
            f"{html.escape(label, quote=True)}</a></li>"
            for label, path in values
        ]
        body = "\n".join(
            [
                "<!doctype html>",
                '<html lang="en"><head><meta charset="utf-8">'
                "<title>Serve Review</title></head><body>",
                "<h1>Serve Review</h1>",
                "<ul>",
                *links,
                "</ul>",
                "</body></html>",
                "",
            ]
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(target.parent),
            prefix=target.name + ".tmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        return target
    except Exception as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise LandingRenderError(
            f"could not publish batch landing page: {exc}"
        ) from exc


def render_multi_source_landing(entries: object, destination: object) -> Path:
    """Publish a deterministic index for explicitly supplied source pages."""
    return render_batch_index(entries, destination)


def render_landing_page(page: LandingPage, destination: object) -> Path:
    """Render explicit records without inspecting any source or artifact path."""

    if not isinstance(page, LandingPage):
        raise LandingRenderError("page must be a LandingPage.")
    try:
        target = Path(destination).expanduser()
    except (TypeError, ValueError, OSError) as exc:
        raise LandingRenderError(f"invalid landing destination: {destination!r}") from exc
    if not target.name:
        raise LandingRenderError("landing destination must name a file.")

    artifacts = sorted(page.source_artifacts, key=_artifact_sort_key)
    attempts = sorted(page.attempts, key=lambda item: (item.attempt_id, item.display_label))
    source_paths = "".join(f"<li>{_text(path)}</li>" for path in page.source.known_paths)
    attempt_sections = []
    for attempt in attempts:
        heading = f"<h3>{_text(attempt.display_label)} ({_text(attempt.attempt_id)})</h3>"
        state = f'<p>Status: {_text(attempt.status)}'
        if attempt.remediation is not None:
            state += f" — {_text(attempt.remediation)}"
        state += "</p>"
        links = "".join(_artifact_html(item, target) for item in sorted(attempt.artifacts, key=_artifact_sort_key))
        attempt_sections.append(f"<section>{heading}{state}<ul>{links}</ul></section>")

    artifact_items = "".join(_artifact_html(item, target) for item in artifacts)
    content = "\n".join([
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8"><title>Serve Review</title></head><body>',
        f"<h1>Serve Review: {_text(page.source.source_id)}</h1>",
        f"<p>Source paths:</p><ul>{source_paths}</ul>",
        "<p>Some browsers cannot play every source codec. Use the direct media links; browser-compatible proxies are not generated.</p>",
        f"<h2>Artifacts</h2><ul>{artifact_items}</ul>",
        f"<h2>Attempts</h2>{''.join(attempt_sections)}",
        "</body></html>",
        "",
    ])

    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=str(target.parent), prefix=target.name + ".tmp-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception as exc:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise LandingRenderError(f"could not atomically write landing page {target}: {exc}") from exc
    return target
