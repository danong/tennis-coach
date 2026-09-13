"""Private deterministic phase-debug report (M4 Unit 3, presentation only).

Reads already-produced diagnostic inputs (one PhaseFeatureGrid,
PhaseEvidence, PhaseSolverResult, PhaseSolverConfig, optional manual
stage-time mapping), invokes the accepted builder/reconciliation
(:func:`build_phase_debug_artifact`), and atomically writes a
deterministic phase-debug JSON artifact plus a self-contained
deterministic HTML inspection page.

Private diagnostic presentation only: no media decoding, pose
inference, video/frame rendering, dense cache extraction,
M3/checkpoints mutation, heuristic/solver/evidence configuration
changes, or held-out access. All parsing validates through the
existing codecs; file I/O is confined to explicit input/output paths
with collision/overwrite behavior consistent with the existing CLI
(review/analyze/export).
"""

from __future__ import annotations

import html
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from serve_review.checkpoints.evidence import PhaseEvidence
from serve_review.checkpoints.phase_debug import PhaseDebugArtifact
from serve_review.checkpoints.phase_debug_builder import build_phase_debug_artifact
from serve_review.checkpoints.phase_features import PhaseFeatureGrid
from serve_review.checkpoints.phase_solver import PhaseSolverConfig, PhaseSolverResult
from serve_review.domain import STAGE_ORDER

__all__ = [
    "PHASE_DEBUG_REPORT_VERSION",
    "PhaseDebugReportError",
    "PhaseDebugReportInputError",
    "PhaseDebugReportCollisionError",
    "PhaseDebugReportResult",
    "parse_manual_times_text",
    "build_phase_debug_html",
    "resolve_html_path",
    "run_phase_debug_report",
]

#: Report renderer identity (presentation only; artifacts record their own method).
PHASE_DEBUG_REPORT_VERSION = "phase-debug-report-v1"

_CANONICAL_STAGES: tuple[str, ...] = STAGE_ORDER


class PhaseDebugReportError(Exception):
    """Actionable phase-debug report failure at one stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.stage}: {self.message}"


class PhaseDebugReportInputError(PhaseDebugReportError):
    """Invalid/missing diagnostic input (CLI code 2)."""


class PhaseDebugReportCollisionError(PhaseDebugReportError):
    """Output collision without --overwrite (CLI code 1)."""


@dataclass(frozen=True, slots=True)
class PhaseDebugReportResult:
    """Outcome of :func:`run_phase_debug_report`."""

    attempt_id: str
    output_json: Path
    output_html: Path
    total_objective: float | None


def parse_manual_times_text(text: str) -> dict[str, float | None]:
    """Parse optional manual stage-time mapping text.

    The payload must be a JSON object mapping a subset of the eight
    canonical stage keys to finite numbers >= 0 or null. Missing keys
    read as null. Unknown keys, non-object payloads, or invalid values
    raise :class:`PhaseDebugReportInputError`. Pure; no I/O.
    """
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PhaseDebugReportInputError(
            "manual", f"invalid manual JSON: {exc}."
        ) from exc
    if not isinstance(decoded, dict):
        raise PhaseDebugReportInputError(
            "manual",
            f"manual mapping must be a JSON object, got {type(decoded).__name__}.",
        )
    unknown = sorted(set(decoded.keys()) - set(_CANONICAL_STAGES))
    if unknown:
        raise PhaseDebugReportInputError(
            "manual", f"unknown manual stage keys {unknown!r}."
        )
    cleaned: dict[str, float | None] = {}
    for stage in _CANONICAL_STAGES:
        if stage not in decoded:
            cleaned[stage] = None
            continue
        value = decoded[stage]
        if value is None:
            cleaned[stage] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PhaseDebugReportInputError(
                "manual",
                f"manual time for stage {stage!r} must be a finite number "
                f">= 0 or null, got {value!r}.",
            )
        number = float(value)
        import math as _math

        if not _math.isfinite(number) or number < 0.0:
            raise PhaseDebugReportInputError(
                "manual",
                f"manual time for stage {stage!r} must be finite and >= 0 "
                f"or null, got {value!r}.",
            )
        cleaned[stage] = number
    return cleaned


def resolve_html_path(output_json: Path, html: Path | None) -> Path:
    """Resolve the HTML destination for an explicit JSON destination."""
    if html is not None:
        return Path(html)
    name = output_json.name
    if name.endswith(".json"):
        return output_json.with_name(name[: -len(".json")] + ".html")
    return output_json.with_name(name + ".html")


def _fmt_num(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value))
    return str(value)


def _esc(value: Any) -> str:
    return html.escape(_fmt_num(value), quote=True)


def _esc_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _support_cell(snapshot: Any) -> str:
    if snapshot is None:
        return "null"
    parts = [
        f"query={_fmt_num(snapshot.queried_time_seconds)}",
        f"support={_fmt_num(snapshot.support_time_seconds)}",
        f"offset={_fmt_num(snapshot.support_offset_seconds)}",
        f"observed={'true' if snapshot.observed else 'false'}",
        f"quality={_fmt_num(snapshot.observation_quality)}",
        f"span={_fmt_num(snapshot.interpolation_span_seconds)}",
        f"uncertainty={_fmt_num(snapshot.temporal_uncertainty_seconds)}",
        f"deriv_conf={_fmt_num(snapshot.derivative_confidence)}",
    ]
    values = snapshot.feature_values
    if values:
        items = sorted(values.items())
        rendered = ", ".join(
            f"{key}={_fmt_num(item)}" for key, item in items
        )
    else:
        rendered = "(no available values)"
    parts.append(f"values: {rendered}")
    return html.escape("; ".join(parts), quote=True)


def _candidates_html(stage_key: str, candidates: Any) -> str:
    if not candidates:
        return "<p>No candidates for this stage.</p>"
    rows: list[str] = []
    for entry in candidates:
        interval = entry.interval
        if interval is None:
            interval_text = "null"
        else:
            interval_text = (
                f"[{_fmt_num(interval.start_seconds)}, "
                f"{_fmt_num(interval.end_seconds)})"
            )
        rows.append(
            "<tr>"
            f"<td>{_esc(entry.rank)}</td>"
            f"<td>{_esc(entry.keyframe_seconds)}</td>"
            f"<td>{html.escape(interval_text, quote=True)}</td>"
            f"<td>{_esc(entry.unary_score)}</td>"
            f"<td>{_esc(entry.observation_quality)}</td>"
            f"<td>{_esc(entry.derivative_quality)}</td>"
            f"<td>{html.escape(str(entry.provenance), quote=True) if entry.provenance is not None else 'null'}</td>"
            f"<td>{_esc(entry.temporal_uncertainty_seconds)}</td>"
            f"<td>{html.escape(','.join(entry.evidence_ids) if entry.evidence_ids else '(none)', quote=True)}</td>"
            f"<td>{html.escape(','.join(entry.limitation_ids) if entry.limitation_ids else '(none)', quote=True)}</td>"
            "</tr>"
        )
    return (
        "<table border=\"1\">"
        "<tr><th>rank</th><th>keyframe s</th><th>interval</th>"
        "<th>unary score</th><th>obs quality</th><th>deriv quality</th>"
        "<th>provenance</th><th>uncertainty s</th>"
        "<th>evidence ids</th><th>limitation ids</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _trace_html(stage_key: str, trace: Any) -> str:
    windows = trace.windows
    if windows:
        window_text = "; ".join(
            f"{window.kind}: [{_fmt_num(window.interval.start_seconds)}, "
            f"{_fmt_num(window.interval.end_seconds)})"
            for window in windows
        )
    else:
        window_text = "no windows (manual and selected times absent)"
    samples = trace.samples
    if samples:
        sample_rows = []
        for sample in samples:
            values = sample.values
            if values:
                rendered = ", ".join(
                    f"{key}={_fmt_num(item)}"
                    for key, item in sorted(values.items())
                )
            else:
                rendered = "(no available values)"
            sample_rows.append(
                "<tr>"
                f"<td>{_esc(sample.time_seconds)}</td>"
                f"<td>{html.escape(rendered, quote=True)}</td>"
                "</tr>"
            )
        samples_table = (
            "<table border=\"1\">"
            "<tr><th>sample time s</th><th>channel values</th></tr>"
            + "".join(sample_rows)
            + "</table>"
        )
    else:
        samples_table = "<p>No trace samples in manual/selected windows.</p>"
    return (
        f"<p>windows: {html.escape(window_text, quote=True)}; "
        f"source samples: {_esc(trace.source_sample_count)}; "
        f"kept: {len(samples)}; "
        f"truncated: {'true' if trace.truncated else 'false'}; "
        f"sampling: {html.escape(str(trace.sampling_method), quote=True)}</p>"
        + samples_table
    )


def build_phase_debug_html(artifact: PhaseDebugArtifact) -> str:
    """Build the deterministic self-contained inspection page.

    Pure; no I/O, no scripts, no external assets. All required
    diagnostic fields are rendered as inspectable text tables.
    """
    if not isinstance(artifact, PhaseDebugArtifact):
        raise PhaseDebugReportInputError(
            "report",
            f"'artifact' must be a PhaseDebugArtifact, got {type(artifact).__name__}.",
        )
    stages = artifact.stages
    traces = artifact.traces
    assert isinstance(stages, Mapping)
    assert isinstance(traces, Mapping)
    attempt_range = artifact.attempt_range
    header_rows = (
        f"<tr><th>attempt</th><td>{html.escape(artifact.attempt_id, quote=True)}</td></tr>"
        f"<tr><th>attempt range s</th><td>[{_esc(attempt_range.start_seconds)}, {_esc(attempt_range.end_seconds)})</td></tr>"
        f"<tr><th>method</th><td>{html.escape(artifact.method_version, quote=True)}</td></tr>"
        f"<tr><th>config</th><td>{html.escape(artifact.config_id, quote=True)}</td></tr>"
        f"<tr><th>solver config</th><td>{html.escape(str(artifact.solver_config_id), quote=True) if artifact.solver_config_id is not None else 'null'}</td></tr>"
        f"<tr><th>solver method</th><td>{html.escape(str(artifact.solver_method_version), quote=True) if artifact.solver_method_version is not None else 'null'}</td></tr>"
        f"<tr><th>reconciliation</th><td>{html.escape(str(artifact.reconciliation_method_version), quote=True) if artifact.reconciliation_method_version is not None else 'null'}</td></tr>"
        f"<tr><th>total objective</th><td>{_esc(artifact.total_objective)}</td></tr>"
        f"<tr><th>artifact anomalies</th><td>{html.escape(','.join(artifact.anomalies) if artifact.anomalies else '(none)', quote=True)}</td></tr>"
    )
    stage_sections: list[str] = []
    for stage_key in _CANONICAL_STAGES:
        record = stages[stage_key]
        trace = traces[stage_key]
        anomalies = ",".join(record.anomalies) if record.anomalies else "(none)"
        section = (
            f"<h2 id=\"stage-{html.escape(stage_key, quote=True)}\">stage: {html.escape(stage_key, quote=True)}</h2>"
            "<table border=\"1\">"
            f"<tr><th>manual time s</th><td>{_esc(record.manual_time_seconds)}</td></tr>"
            f"<tr><th>selected time s</th><td>{_esc(record.selected_time_seconds)}</td></tr>"
            f"<tr><th>manual support</th><td>{_support_cell(record.manual_support)}</td></tr>"
            f"<tr><th>selected support</th><td>{_support_cell(record.selected_support)}</td></tr>"
            f"<tr><th>manual score</th><td>{_esc(record.manual_score)}</td></tr>"
            f"<tr><th>manual rank</th><td>{_esc(record.manual_rank)}</td></tr>"
            f"<tr><th>selected score</th><td>{_esc(record.selected_score)}</td></tr>"
            f"<tr><th>selected rank</th><td>{_esc(record.selected_rank)}</td></tr>"
            f"<tr><th>selection explanation</th><td>{html.escape(str(record.selection_explanation), quote=True)}</td></tr>"
            f"<tr><th>predecessor stage</th><td>{html.escape(str(record.predecessor_stage), quote=True) if record.predecessor_stage is not None else 'null'}</td></tr>"
            f"<tr><th>unary contribution</th><td>{_esc(record.unary_contribution)}</td></tr>"
            f"<tr><th>transition contribution</th><td>{_esc(record.transition_contribution)}</td></tr>"
            f"<tr><th>skip contribution</th><td>{_esc(record.skip_contribution)}</td></tr>"
            f"<tr><th>objective contribution</th><td>{_esc(record.objective_contribution)}</td></tr>"
            f"<tr><th>selected contact offset s</th><td>{_esc(record.selected_contact_offset_seconds)}</td></tr>"
            f"<tr><th>manual contact offset s</th><td>{_esc(record.manual_contact_offset_seconds)}</td></tr>"
            f"<tr><th>unavailable reason</th><td>{html.escape(str(record.unavailable_reason), quote=True) if record.unavailable_reason is not None else 'null'}</td></tr>"
            f"<tr><th>anomalies</th><td>{html.escape(anomalies, quote=True)}</td></tr>"
            "</table>"
            f"<h3>candidates ({len(record.candidates)})</h3>"
            + _candidates_html(stage_key, record.candidates)
            + "<h3>bounded trace</h3>"
            + _trace_html(stage_key, trace)
        )
        stage_sections.append(section)
    body_sections = "\n".join(stage_sections)
    return (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<title>Phase debug inspection</title>"
        "<style>body{font-family:sans-serif;max-width:1000px;margin:auto;}"
        "table{border-collapse:collapse;margin:8px 0;}th,td{padding:4px 8px;text-align:left;}"
        "</style></head><body>\n"
        "<h1>Phase debug inspection</h1>\n"
        "<p>Private deterministic diagnostic presentation only. No video "
        "frames are rendered here; times are canonical source seconds.</p>\n"
        "<h2>Limitations (honest labels)</h2>\n"
        "<ul>"
        "<li>release is a toss-height proxy (sustained left-wrist crossing), "
        "not an observed ball release.</li>"
        "<li>cocking is an arm-cocking proxy (elbow flexion, wrist "
        "configuration, trajectory reversal), not a direct observation of "
        "racket orientation or shoulder axial rotation.</li>"
        "<li>contact is an audio-transient anchor with explicit timing "
        "uncertainty; dense body pose may support the surrounding arm "
        "configuration but contact is never claimed as exact visual "
        "ball-racket observation.</li>"
        "<li>All camera-relative quantities stay camera-relative; no "
        "handedness, viewpoint, or anatomical forward/posterior inference.</li>"
        "</ul>\n"
        "<h2>Artifact header</h2>\n"
        f"<table border=\"1\">{header_rows}</table>\n"
        f"{body_sections}\n"
        "</body></html>\n"
    )


def _read_text(path: Path, label: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise PhaseDebugReportInputError(
            label, f"could not read {label} file {path}: {exc}."
        ) from exc


def _parse_grid(text: str) -> PhaseFeatureGrid:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PhaseDebugReportInputError(
            "grid", f"invalid grid JSON: {exc}."
        ) from exc
    if not isinstance(decoded, dict):
        raise PhaseDebugReportInputError(
            "grid",
            f"grid JSON object is required, got {type(decoded).__name__}.",
        )
    try:
        return PhaseFeatureGrid.from_dict(decoded)
    except Exception as exc:
        raise PhaseDebugReportInputError(
            "grid", f"invalid grid: {exc}."
        ) from exc


def _parse_evidence(text: str) -> PhaseEvidence:
    try:
        return PhaseEvidence.from_json(text)
    except Exception as exc:
        raise PhaseDebugReportInputError(
            "evidence", f"invalid evidence: {exc}."
        ) from exc


def _parse_solver_result(text: str) -> PhaseSolverResult:
    try:
        return PhaseSolverResult.from_json(text)
    except Exception as exc:
        raise PhaseDebugReportInputError(
            "solver-result", f"invalid solver result: {exc}."
        ) from exc


def _parse_solver_config(text: str) -> PhaseSolverConfig:
    try:
        return PhaseSolverConfig.from_json(text)
    except Exception as exc:
        raise PhaseDebugReportInputError(
            "solver-config", f"invalid solver config: {exc}."
        ) from exc


def _write_text_atomic(target: Path, text: str) -> None:
    parent = target.parent
    if str(parent) not in ("", "."):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PhaseDebugReportError(
                "report", f"could not create output directory {parent}: {exc}."
            ) from exc
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(parent) if str(parent) not in ("", ".") else None,
            prefix=target.name + ".tmp-",
            delete=False,
        ) as handle:
            tmp_path = handle.name
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, target)
    except PhaseDebugReportError:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    except OSError as exc:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise PhaseDebugReportError(
            "report", f"could not write output file {target}: {exc}."
        ) from exc


def run_phase_debug_report(
    grid_path: Path | str,
    evidence_path: Path | str,
    solver_result_path: Path | str,
    solver_config_path: Path | str,
    output_json: Path | str,
    output_html: Path | str | None = None,
    *,
    manual_path: Path | str | None = None,
    manual_times: Mapping[str, float | None] | None = None,
    overwrite: bool = False,
) -> PhaseDebugReportResult:
    """Build and atomically write the phase-debug JSON and HTML report.

    Inputs are read from explicit JSON files and validated through the
    existing codecs; the accepted builder/reconciliation is invoked;
    both outputs are written atomically with no partial files on
    failure. ``manual_path`` and ``manual_times`` are mutually
    exclusive (at most one may be supplied).
    """
    grid_file = Path(grid_path).expanduser()
    evidence_file = Path(evidence_path).expanduser()
    result_file = Path(solver_result_path).expanduser()
    config_file = Path(solver_config_path).expanduser()
    json_out = Path(output_json).expanduser()
    html_out = resolve_html_path(json_out, Path(output_html).expanduser() if output_html is not None else None)
    manual_file = Path(manual_path).expanduser() if manual_path is not None else None

    if not isinstance(overwrite, bool):
        raise PhaseDebugReportInputError(
            "report", f"invalid overwrite: {overwrite!r}; expected True or False."
        )
    if manual_file is not None and manual_times is not None:
        raise PhaseDebugReportInputError(
            "manual",
            "supply at most one of 'manual_path' and 'manual_times'.",
        )
    if html_out == json_out:
        raise PhaseDebugReportInputError(
            "report", "HTML and JSON outputs must be distinct files."
        )
    for label, candidate in (
        ("grid", grid_file),
        ("evidence", evidence_file),
        ("solver-result", result_file),
        ("solver-config", config_file),
    ):
        if not candidate.is_file():
            raise PhaseDebugReportInputError(
                label, f"{label} file does not exist: {candidate}."
            )
    if manual_file is not None and not manual_file.is_file():
        raise PhaseDebugReportInputError(
            "manual", f"manual file does not exist: {manual_file}."
        )

    grid = _parse_grid(_read_text(grid_file, "grid"))
    evidence = _parse_evidence(_read_text(evidence_file, "evidence"))
    solver_result = _parse_solver_result(_read_text(result_file, "solver-result"))
    solver_config = _parse_solver_config(_read_text(config_file, "solver-config"))

    if manual_file is not None:
        manual = parse_manual_times_text(_read_text(manual_file, "manual"))
    elif manual_times is not None:
        if not isinstance(manual_times, Mapping):
            raise PhaseDebugReportInputError(
                "manual",
                f"'manual_times' must be a mapping, got {type(manual_times).__name__}.",
            )
        manual = parse_manual_times_text(json.dumps(dict(manual_times)))
    else:
        manual = {stage: None for stage in _CANONICAL_STAGES}

    try:
        artifact = build_phase_debug_artifact(
            grid,
            evidence,
            solver_result,
            manual,
            solver_result=solver_result,
            solver_config=solver_config,
        )
    except Exception as exc:
        raise PhaseDebugReportInputError(
            "report", f"invalid phase-debug linkage: {exc}."
        ) from exc

    json_text = artifact.to_json()
    try:
        html_text = build_phase_debug_html(artifact)
    except PhaseDebugReportInputError:
        raise
    except Exception as exc:
        raise PhaseDebugReportError(
            "report", f"could not render inspection page: {exc}."
        ) from exc

    if json_out.exists() and not overwrite:
        raise PhaseDebugReportCollisionError(
            "report",
            f"output collision: {json_out} already exists. "
            "Pass --overwrite to replace outputs.",
        )
    if html_out.exists() and not overwrite:
        raise PhaseDebugReportCollisionError(
            "report",
            f"output collision: {html_out} already exists. "
            "Pass --overwrite to replace outputs.",
        )

    json_existed = json_out.exists()
    html_existed = html_out.exists()
    try:
        _write_text_atomic(json_out, json_text)
    except PhaseDebugReportError:
        raise
    try:
        _write_text_atomic(html_out, html_text)
    except PhaseDebugReportError as exc:
        if not json_existed and json_out.is_file():
            try:
                json_out.unlink()
            except OSError:
                pass
        raise
    return PhaseDebugReportResult(
        attempt_id=artifact.attempt_id,
        output_json=json_out,
        output_html=html_out,
        total_objective=artifact.total_objective,
    )
