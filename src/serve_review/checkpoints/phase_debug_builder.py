"""Deterministic phase-debug extraction (M4 Units 1+2, representation only).

Pure, deterministic, in-memory construction of one
:class:`~serve_review.checkpoints.phase_debug.PhaseDebugArtifact` from one
M4.2 :class:`PhaseFeatureGrid`, one M4.3 :class:`PhaseEvidence`, one
selected M4.1 :class:`AttemptPhase` (or the existing M4.4
:class:`PhaseSolverResult` wrapper), and optional frozen manual stage
query times.

Contract (Unit 1 preserved):

- All eight canonical Kovacs stages are populated; no stage is invented
  or dropped. Candidate summaries keep stored evidence order with
  deterministic 1-based ranks (1 is best).
- Manual semantics are exact and honest: an exact candidate-keyframe
  match at the manual query time yields that candidate's unary
  score/rank as ``manual_score``/``manual_rank``; a nonmatching or null
  manual time yields null/null. Scores are never interpolated or
  invented. Selected score/rank follow the same exact-match rule.
- Manual/selected support snapshots are read from the grid at the
  queried times (nearest grid sample, exact signed offset, grid quality
  channels, allowlist-only feature values). Absent grid support yields
  a null snapshot, never fabricated values.
- Trace windows are bounded (at most one manual plus one selected
  window per stage) and trace samples are bounded to
  ``MAX_TRACE_SAMPLES`` strictly increasing canonical source times with
  an explicit truncation flag and pre-truncation source count.
- The selected unavailable reason and per-stage anomalies are derived
  deterministically from the inputs (selected limitations/candidate
  presence), never invented.
- Range/method linkage is strict: grid, evidence, and selected attempt
  ranges must coincide exactly and grid/evidence method identities must
  match their modules; mismatches raise :class:`PhaseDebugError`.

Contract (Unit 2 reconciliation, opt-in only):

- Every stage record additionally exposes the deterministic
  ``selection_explanation``, ``predecessor_stage`` (nearest preceding
  selected stage, null for first selected and all skips), and
  contact-relative ``selected_contact_offset_seconds`` /
  ``manual_contact_offset_seconds`` (signed ``time - contact`` offsets,
  null exactly when the corresponding time or the contact anchor is
  absent). These are derived from the already-selected/manual times
  without touching solver selection.
- When ``solver_result`` and ``solver_config`` are both supplied, each
  record additionally carries the replayed ``unary_contribution`` /
  ``transition_contribution`` / ``skip_contribution`` /
  ``objective_contribution`` shares and the artifact carries
  ``total_objective`` plus durable ``solver_config_id`` /
  ``solver_method_version`` / ``reconciliation_method_version``. The
  summed shares reconcile to ``PhaseSolverResult.total_score`` within
  ``RECONCILIATION_OBJECTIVE_TOLERANCE``; any config/result/range/
  selected-keyframe mismatch (including a selected time without an
  exact candidate match or an infeasible chosen transition) raises
  :class:`PhaseDebugError` instead of inventing reconciliation.
- When both solver inputs are absent the artifact stays unreconciled
  (Unit 1): contributions are null, ``total_objective`` and solver
  linkage are null, while explanation/predecessor/offsets remain
  populated from the selected/manual times. Supplying exactly one of
  the two solver inputs raises.
- Contact uses the raw candidate unary score like every other stage;
  the advisory body-compatibility confidence adjustment never enters
  the objective. Skipped contact contributes ``-contact_skip_penalty``
  (other skips ``-skip_penalty``); the first selected stage contributes
  transition ``0.0`` and each later selected stage ``transition_bonus``
  over its (possibly non-adjacent) feasible predecessor span.

This module performs no inference, no file I/O, no CLI/pipeline/
rendering, no dense extraction, and no solver/evidence heuristic,
configuration, or production behavior changes. Solver selection is never
re-run or altered here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from serve_review.checkpoints.evidence import (
    EVIDENCE_METHOD_VERSION,
    PhaseEvidence,
    StageCandidate,
)
from serve_review.checkpoints.phase_debug import (
    MAX_TRACE_SAMPLES,
    PHASE_DEBUG_DEFAULT_CONFIG_ID,
    PHASE_DEBUG_METHOD_VERSION,
    RECONCILIATION_METHOD_VERSION,
    TRACE_SAMPLING_METHOD_V1,
    CandidateSummary,
    PhaseDebugArtifact,
    PhaseDebugError,
    StageDebug,
    StageTrace,
    SupportSnapshot,
    TraceSample,
    TraceWindow,
)
from serve_review.checkpoints.phase_features import (
    PHASE_FEATURES_METHOD_VERSION,
    PhaseFeatureGrid,
    PhaseFeatureSample,
)
from serve_review.checkpoints.phase_solver import (
    PhaseSolverConfig,
    PhaseSolverResult,
)
from serve_review.domain import (
    STAGE_ORDER,
    AttemptPhase,
    MediaRange,
    StagePhase,
)

__all__ = [
    "PHASE_DEBUG_BUILDER_METHOD_VERSION",
    "TRACE_WINDOW_HALF_SECONDS",
    "build_phase_debug_artifact",
]

#: Method identity of this diagnostic extraction (informational only;
#: artifacts record :data:`PHASE_DEBUG_METHOD_VERSION`).
PHASE_DEBUG_BUILDER_METHOD_VERSION = "phase-debug-builder-v1"

#: Half-width in canonical source seconds of each manual/selected trace
#: window. Matches the default evidence candidate half-window so trace
#: context stays comparable to candidate intervals.
TRACE_WINDOW_HALF_SECONDS = 0.06

_CANONICAL_STAGES: tuple[str, ...] = STAGE_ORDER


def _check_optional_query_time(
    stage: str, value: Any
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseDebugError(
            f"phase_debug_builder: manual time for stage {stage!r} must be "
            f"a finite number >= 0 or null, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseDebugError(
            f"phase_debug_builder: manual time for stage {stage!r} must be "
            f"finite and >= 0 or null, got {value!r}."
        )
    return number


def _normalize_manual_times(manual_times: Any) -> dict[str, float | None]:
    name = "phase_debug_builder"
    if manual_times is None:
        return {stage: None for stage in _CANONICAL_STAGES}
    if not isinstance(manual_times, Mapping):
        raise PhaseDebugError(
            f"{name}: 'manual_times' must be a mapping of stage key to "
            f"query time or null, got {type(manual_times).__name__}."
        )
    unknown = sorted(set(manual_times.keys()) - set(_CANONICAL_STAGES))
    if unknown:
        raise PhaseDebugError(
            f"{name}: unknown manual stage keys {unknown!r}; expected a "
            f"subset of {list(_CANONICAL_STAGES)}."
        )
    return {
        stage: _check_optional_query_time(stage, manual_times.get(stage))
        for stage in _CANONICAL_STAGES
    }


def _resolve_selected(selected: Any) -> AttemptPhase:
    name = "phase_debug_builder"
    if isinstance(selected, PhaseSolverResult):
        return selected.attempt_phase
    if isinstance(selected, AttemptPhase):
        return selected
    raise PhaseDebugError(
        f"{name}: 'selected' must be an AttemptPhase or a PhaseSolverResult "
        f"wrapper, got {type(selected).__name__}."
    )


def _check_linkage(
    grid: PhaseFeatureGrid,
    evidence: PhaseEvidence,
    selected: AttemptPhase,
) -> MediaRange:
    name = "phase_debug_builder"
    if not isinstance(grid, PhaseFeatureGrid):
        raise PhaseDebugError(
            f"{name}: 'grid' must be a PhaseFeatureGrid, "
            f"got {type(grid).__name__}."
        )
    if not isinstance(evidence, PhaseEvidence):
        raise PhaseDebugError(
            f"{name}: 'evidence' must be a PhaseEvidence, "
            f"got {type(evidence).__name__}."
        )
    attempt = grid.attempt_range
    if (
        evidence.attempt_range.start_seconds != attempt.start_seconds
        or evidence.attempt_range.end_seconds != attempt.end_seconds
    ):
        raise PhaseDebugError(
            f"{name}: evidence attempt range "
            f"{evidence.attempt_range.to_dict()!r} does not match grid "
            f"attempt range {attempt.to_dict()!r}."
        )
    selected_range = selected.attempt_range
    if (
        selected_range.start_seconds != attempt.start_seconds
        or selected_range.end_seconds != attempt.end_seconds
    ):
        raise PhaseDebugError(
            f"{name}: selected attempt range "
            f"{selected_range.to_dict()!r} does not match grid attempt "
            f"range {attempt.to_dict()!r}."
        )
    if grid.method_version != PHASE_FEATURES_METHOD_VERSION:
        raise PhaseDebugError(
            f"{name}: grid method_version {grid.method_version!r} is not "
            f"compatible with {PHASE_FEATURES_METHOD_VERSION!r}."
        )
    if evidence.method_version != EVIDENCE_METHOD_VERSION:
        raise PhaseDebugError(
            f"{name}: evidence method_version {evidence.method_version!r} "
            f"is not compatible with {EVIDENCE_METHOD_VERSION!r}."
        )
    return attempt


def _check_time_in_attempt(
    label: str, stage: str, moment: float, attempt: MediaRange
) -> None:
    if not attempt.contains(moment) and not moment == attempt.start_seconds:
        raise PhaseDebugError(
            f"phase_debug_builder: {label} for stage {stage!r} ({moment!r}) "
            f"must lie in attempt range {attempt.to_dict()!r}."
        )


def _finite_or_missing(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _channel_values(sample: PhaseFeatureSample) -> dict[str, float | bool | None]:
    """Map one grid sample onto the fixed v1 diagnostic channel allowlist.

    Only channels with an honest grid source are emitted; channels with
    no source (face threshold, arc length) or missing inputs are simply
    absent. Camera-relative quantities stay camera-relative; no
    handedness, viewpoint, or anatomical forward/posterior inference.
    """
    values: dict[str, float | bool | None] = {}

    def put(key: str, value: Any) -> None:
        number = _finite_or_missing(value)
        if number is not None:
            values[key] = number

    put("left_wrist_x_camera", sample.wrist_left_x)
    put("left_wrist_y_camera", sample.wrist_left_y)
    upward_left = _finite_or_missing(sample.wrist_left_vy)
    if upward_left is not None:
        values["left_wrist_upward_velocity_camera"] = -upward_left
    speed_left = _wrist_speed(sample.wrist_left_vx, sample.wrist_left_vy)
    if speed_left is not None:
        values["left_wrist_speed_camera"] = speed_left
    # ``face_height_threshold_camera`` has no face source in the grid:
    # honestly omitted (missing, never zero-filled).
    flexion = _combined_knee_flexion(
        sample.knee_angle_left, sample.knee_angle_right
    )
    if flexion is not None:
        values["combined_knee_flexion"] = flexion
    shoulder_tilt = _axis_tilt_deg(
        sample.shoulder_axis_camera_dx, sample.shoulder_axis_camera_dy
    )
    if shoulder_tilt is not None:
        values["shoulder_line_tilt_camera_deg"] = shoulder_tilt
    hip_tilt = _axis_tilt_deg(
        sample.hip_axis_camera_dx, sample.hip_axis_camera_dy
    )
    if hip_tilt is not None:
        values["hip_line_tilt_camera_deg"] = hip_tilt
    if shoulder_tilt is not None and hip_tilt is not None:
        values["shoulder_hip_angle_difference_camera_deg"] = (
            shoulder_tilt - hip_tilt
        )
    elbow_flexion = _flexion01(sample.elbow_angle_right)
    if elbow_flexion is not None:
        values["right_elbow_flexion"] = elbow_flexion
    put("right_wrist_x_camera", sample.wrist_right_x)
    put("right_wrist_y_camera", sample.wrist_right_y)
    elevation = _finite_or_missing(sample.wrist_right_y)
    if elevation is not None:
        # Camera y grows downward, so higher on screen means larger.
        values["right_wrist_elevation_camera"] = -elevation
    upward_right = _finite_or_missing(sample.wrist_right_vy)
    if upward_right is not None:
        values["right_wrist_upward_velocity_camera"] = -upward_right
    speed_right = _wrist_speed(sample.wrist_right_vx, sample.wrist_right_vy)
    if speed_right is not None:
        values["right_wrist_speed_camera"] = speed_right
    # ``right_wrist_arc_length`` needs path integration over many rows:
    # honestly omitted per-sample (missing, never zero-filled).
    put("torso_rotation_proxy_camera", sample.torso_rotation_camera_deg)
    put("audio_transient_energy", sample.audio_energy)
    if isinstance(sample.audio_candidate, bool):
        values["audio_candidate"] = sample.audio_candidate
    return values


def _wrist_speed(vx: Any, vy: Any) -> float | None:
    x = _finite_or_missing(vx)
    y = _finite_or_missing(vy)
    if x is None or y is None:
        return None
    return math.hypot(x, y)


def _flexion01(angle_deg: Any) -> float | None:
    angle = _finite_or_missing(angle_deg)
    if angle is None:
        return None
    return max(0.0, min(1.0, (180.0 - angle) / 180.0))


def _combined_knee_flexion(left: Any, right: Any) -> float | None:
    parts = [
        _flexion01(value) for value in (left, right)
    ]
    available = [value for value in parts if value is not None]
    if not available:
        return None
    return sum(available) / len(available)


def _axis_tilt_deg(dx: Any, dy: Any) -> float | None:
    x = _finite_or_missing(dx)
    y = _finite_or_missing(dy)
    if x is None or y is None:
        return None
    if x == 0.0 and y == 0.0:
        return None
    return math.degrees(math.atan2(y, x))


def _nearest_sample(
    grid: PhaseFeatureGrid, queried: float
) -> PhaseFeatureSample | None:
    best: PhaseFeatureSample | None = None
    best_key: tuple[float, float] | None = None
    for sample in grid.samples:
        key = (abs(float(sample.time_seconds) - queried), float(sample.time_seconds))
        if best_key is None or key < best_key:
            best_key = key
            best = sample
    return best


def _support_at(
    grid: PhaseFeatureGrid, queried: float | None
) -> SupportSnapshot | None:
    """Build the support snapshot at one queried canonical source time.

    The nearest grid sample (ties broken toward the earliest time)
    supplies support; the signed offset is exact ``support - query`` and
    the uncertainty width is widened to cover ``abs(offset)`` and half
    the supporting span so canonical query times never claim precision
    finer than their direct support. Null query times or an empty grid
    yield null (never fabricated support).
    """
    if queried is None:
        return None
    sample = _nearest_sample(grid, queried)
    if sample is None:
        return None
    support_time = float(sample.time_seconds)
    offset = support_time - queried
    span = float(sample.interpolation_span_seconds)
    uncertainty = max(
        float(sample.temporal_uncertainty_seconds),
        abs(offset),
        span / 2.0,
    )
    return SupportSnapshot(
        queried_time_seconds=queried,
        support_time_seconds=support_time,
        support_offset_seconds=offset,
        observed=bool(sample.observed),
        observation_quality=float(sample.observation_quality),
        interpolation_span_seconds=span,
        temporal_uncertainty_seconds=uncertainty,
        derivative_confidence=float(sample.derivative_confidence),
        feature_values=_channel_values(sample),
    )


def _ranked_candidates(
    evidence: PhaseEvidence, stage: str
) -> list[tuple[StageCandidate, int]]:
    """Return ``(candidate, rank)`` pairs in stored evidence order.

    Evidence stores candidates ordered deterministically by
    ``(-score, keyframe_seconds)``; the 1-based stored position is the
    rank (1 is best).
    """
    return [
        (candidate, position + 1)
        for position, candidate in enumerate(evidence.for_stage(stage))
    ]


def _score_rank_at(
    ranked: list[tuple[StageCandidate, int]], queried: float | None
) -> tuple[float | None, int | None]:
    """Exact-match score/rank lookup (never interpolated).

    Only a query time exactly equal to a candidate keyframe inherits
    that candidate's unary score/rank; nonmatching or null query times
    yield null/null.
    """
    if queried is None:
        return None, None
    for candidate, rank in ranked:
        if float(queried) == float(candidate.keyframe_seconds):
            return float(candidate.score), int(rank)
    return None, None


def _candidate_summaries(
    stage: str, ranked: list[tuple[StageCandidate, int]]
) -> tuple[CandidateSummary, ...]:
    return tuple(
        CandidateSummary(
            stage=stage,
            interval=candidate.interval,
            keyframe_seconds=float(candidate.keyframe_seconds),
            unary_score=float(candidate.score),
            rank=rank,
            observation_quality=float(candidate.observation_quality),
            derivative_quality=float(candidate.derivative_quality),
            provenance=candidate.provenance,
            temporal_uncertainty_seconds=float(
                candidate.temporal_uncertainty_seconds
            ),
            evidence_ids=tuple(candidate.evidence),
            limitation_ids=tuple(candidate.limitations),
        )
        for candidate, rank in ranked
    )


def _unavailable_reason(
    selected_stage: StagePhase, has_candidates: bool
) -> str | None:
    """Derive the unavailable reason from the selected stage record.

    Returns null whenever selected information exists (a selected time,
    candidates, or scores -- decided by the caller); otherwise echoes
    the selected stage's own limitations or the honest
    ``no_candidate_available`` fallback. ``has_candidates`` is accepted
    for symmetry with anomaly derivation but the reason itself comes
    from the selected limitations so skipped-for-consistency stages
    keep their solver-provided limitation text.
    """
    _ = has_candidates
    if selected_stage.limitations:
        return "; ".join(selected_stage.limitations)
    return "no_candidate_available"


def _stage_anomalies(
    selected_stage: StagePhase, has_candidates: bool, stage: str
) -> tuple[str, ...]:
    """Derive stable per-stage anomaly identifiers from the inputs."""
    if selected_stage.availability in ("available", "partial"):
        return ()
    if has_candidates:
        return (f"skipped_{stage}_for_consistency",)
    return (f"no_candidate_{stage}",)


def _stage_trace(
    grid: PhaseFeatureGrid,
    attempt: MediaRange,
    stage: str,
    manual_time: float | None,
    selected_time: float | None,
) -> StageTrace:
    """Build the bounded manual/selected trace window record for one stage."""
    windows: list[TraceWindow] = []
    for kind, moment in (("manual", manual_time), ("selected", selected_time)):
        if moment is None:
            continue
        low = max(attempt.start_seconds, float(moment) - TRACE_WINDOW_HALF_SECONDS)
        high = min(attempt.end_seconds, float(moment) + TRACE_WINDOW_HALF_SECONDS)
        if not high > low:
            # Attempt-edge collapse guard: keep a minimal valid width.
            step = 1.0 / float(grid.grid_rate_hz) if float(grid.grid_rate_hz) > 0 else 1e-3
            if float(moment) >= attempt.end_seconds:
                high = attempt.end_seconds
                low = max(attempt.start_seconds, high - step)
            else:
                low = attempt.start_seconds
                high = min(attempt.end_seconds, low + step)
            if not high > low:  # pragma: no cover - degenerate attempt
                raise PhaseDebugError(
                    "phase_debug_builder: cannot place a trace window for "
                    f"stage {stage!r} inside attempt range "
                    f"{attempt.to_dict()!r}."
                )
        windows.append(
            TraceWindow(
                kind=kind,
                interval=MediaRange(start_seconds=low, end_seconds=high),
            )
        )
    covered = [
        sample
        for sample in grid.samples
        if any(
            window.interval.contains(float(sample.time_seconds))
            or float(sample.time_seconds) == window.interval.start_seconds
            for window in windows
        )
    ]
    source_count = len(covered)
    truncated = source_count > MAX_TRACE_SAMPLES
    if truncated:
        # Deterministic even decimation preserving strictly increasing
        # canonical source-time order.
        indices = [
            round(index * (source_count - 1) / (MAX_TRACE_SAMPLES - 1))
            for index in range(MAX_TRACE_SAMPLES)
        ]
        kept = [covered[position] for position in indices]
    else:
        kept = covered
    samples = tuple(
        TraceSample(
            time_seconds=float(sample.time_seconds),
            values=_channel_values(sample),
        )
        for sample in kept
    )
    return StageTrace(
        stage=stage,
        windows=tuple(windows),
        samples=samples,
        source_sample_count=source_count,
        sampling_method=TRACE_SAMPLING_METHOD_V1,
        truncated=truncated,
    )


def _selection_explanation_unreconciled(
    selected_time: float | None,
    selected_rank: int | None,
    has_candidates: bool,
) -> str:
    """Deterministic selected/skipped explanation without solver config.

    Selected times with an exact candidate match name their rank;
    selected times without a match stay honest about the miss (the
    Unit 2 reconciler rejects this state); skips reuse the solver
    limitation vocabulary. Never invented beyond these inputs.
    """
    if selected_time is not None:
        if selected_rank is not None:
            return f"selected_candidate_rank_{int(selected_rank)}"
        return "selected_without_exact_candidate_match"
    if has_candidates:
        return "skipped_for_global_consistency"
    return "no_candidate_available"


def build_phase_debug_artifact(
    grid: PhaseFeatureGrid,
    evidence: PhaseEvidence,
    selected: AttemptPhase | PhaseSolverResult,
    manual_times: Mapping[str, float | None] | None = None,
    *,
    config_id: str = PHASE_DEBUG_DEFAULT_CONFIG_ID,
    solver_result: PhaseSolverResult | None = None,
    solver_config: PhaseSolverConfig | None = None,
) -> PhaseDebugArtifact:
    """Build the deterministic phase-debug artifact for one attempt.

    ``grid`` is the owning M4.2 feature grid, ``evidence`` the M4.3
    candidate set, ``selected`` the M4.1 selected phase (or the M4.4
    solver result wrapping it), and ``manual_times`` the optional frozen
    manual stage query times (missing keys read as null). Pure and
    deterministic: equal inputs yield equal artifacts; no I/O, no
    inference, no heuristic changes. Solver selection is never re-run.

    When ``solver_result`` and ``solver_config`` are both supplied the
    artifact is reconciled (Unit 2): per-stage unary/transition/skip
    shares plus ``total_objective`` and solver linkage are populated and
    must match ``solver_result.total_score`` within the documented
    tolerance. ``selected`` must then equal
    ``solver_result.attempt_phase`` exactly. Supplying exactly one of
    the two solver inputs raises. When both are absent the artifact is
    unreconciled (Unit 1) with null shares/totals but fully populated
    explanation/predecessor/contact-offset fields.
    """
    selected_phase = _resolve_selected(selected)
    attempt = _check_linkage(grid, evidence, selected_phase)
    manual = _normalize_manual_times(manual_times)
    if not isinstance(config_id, str) or not config_id.strip():
        raise PhaseDebugError(
            "phase_debug_builder: 'config_id' must be a non-blank string, "
            f"got {config_id!r}."
        )
    if (solver_result is None) != (solver_config is None):
        raise PhaseDebugError(
            "phase_debug_builder: 'solver_result' and 'solver_config' "
            "must be supplied together or not at all."
        )
    reconciled: dict[str, Any] | None = None
    reconciled_total: float | None = None
    if solver_result is not None and solver_config is not None:
        if not isinstance(solver_result, PhaseSolverResult):
            raise PhaseDebugError(
                "phase_debug_builder: 'solver_result' must be a "
                f"PhaseSolverResult, got {type(solver_result).__name__}."
            )
        if not isinstance(solver_config, PhaseSolverConfig):
            raise PhaseDebugError(
                "phase_debug_builder: 'solver_config' must be a "
                f"PhaseSolverConfig, got {type(solver_config).__name__}."
            )
        if selected_phase != solver_result.attempt_phase:
            raise PhaseDebugError(
                "phase_debug_builder: 'selected' must equal "
                "solver_result.attempt_phase exactly; reconciliation "
                "never mixes runs."
            )
        # Local import keeps module import order stable and avoids a
        # solver-behavior duplicate: all DP semantics live in the
        # reconciliation replay, not here.
        from serve_review.checkpoints.phase_debug_reconciliation import (
            explain_solver_path,
        )

        replay, replay_total = explain_solver_path(
            evidence, solver_result, solver_config
        )
        reconciled = {key: replay[key] for key in _CANONICAL_STAGES}
        reconciled_total = float(replay_total)

    # First pass: Unit 1 values (support/traces/candidates/scores) plus
    # the selected/manual time maps needed for predecessor/offsets.
    selected_times: dict[str, float | None] = {}
    selected_scores: dict[str, float | None] = {}
    selected_ranks: dict[str, int | None] = {}
    candidate_lists: dict[str, tuple[CandidateSummary, ...]] = {}
    for stage in _CANONICAL_STAGES:
        selected_stage = selected_phase.stages[stage]
        moment = selected_stage.keyframe_seconds
        selected_times[stage] = None if moment is None else float(moment)
    contact_selected = selected_times["contact"]
    # Predecessor chain over the existing selection (nearest preceding
    # selected stage; null for first selected and all skips).
    predecessor_of: dict[str, str | None] = {}
    last_selected: str | None = None
    for stage in _CANONICAL_STAGES:
        if selected_times[stage] is not None:
            predecessor_of[stage] = last_selected
            last_selected = stage
        else:
            predecessor_of[stage] = None

    stages: dict[str, StageDebug] = {}
    traces: dict[str, StageTrace] = {}
    for stage in _CANONICAL_STAGES:
        manual_time = manual[stage]
        selected_stage = selected_phase.stages[stage]
        selected_time = selected_times[stage]
        if selected_time is not None:
            _check_time_in_attempt(
                "selected time", stage, selected_time, attempt
            )
        if manual_time is not None:
            _check_time_in_attempt("manual time", stage, manual_time, attempt)
        ranked = _ranked_candidates(evidence, stage)
        candidates = _candidate_summaries(stage, ranked)
        candidate_lists[stage] = candidates
        manual_score, manual_rank = _score_rank_at(ranked, manual_time)
        selected_score, selected_rank = _score_rank_at(ranked, selected_time)
        selected_scores[stage] = selected_score
        selected_ranks[stage] = selected_rank
        has_selected = (
            selected_time is not None
            or any(entry.interval is not None for entry in candidates)
            or selected_score is not None
            or selected_rank is not None
        )
        if has_selected:
            unavailable_reason = None
        else:
            unavailable_reason = _unavailable_reason(
                selected_stage, bool(ranked)
            )
        if contact_selected is None:
            selected_offset: float | None = None
            manual_offset: float | None = None
        else:
            selected_offset = (
                None
                if selected_time is None
                else float(selected_time) - float(contact_selected)
            )
            manual_offset = (
                None
                if manual_time is None
                else float(manual_time) - float(contact_selected)
            )
        if reconciled is None:
            unary_value: float | None = None
            transition_value: float | None = None
            skip_value: float | None = None
            objective_value: float | None = None
            explanation = _selection_explanation_unreconciled(
                selected_time, selected_rank, bool(ranked)
            )
            predecessor: str | None = predecessor_of[stage]
        else:
            entry = reconciled[stage]
            unary_value = entry.unary_contribution
            transition_value = entry.transition_contribution
            skip_value = entry.skip_contribution
            objective_value = entry.objective_contribution
            explanation = entry.selection_explanation
            predecessor = entry.predecessor_stage
        stages[stage] = StageDebug(
            stage=stage,
            manual_time_seconds=manual_time,
            selected_time_seconds=selected_time,
            manual_support=_support_at(grid, manual_time),
            selected_support=_support_at(grid, selected_time),
            candidates=candidates,
            manual_score=manual_score,
            manual_rank=manual_rank,
            selected_score=selected_score,
            selected_rank=selected_rank,
            objective_contribution=objective_value,
            unavailable_reason=unavailable_reason,
            anomalies=_stage_anomalies(
                selected_stage, bool(ranked), stage
            ),
            unary_contribution=unary_value,
            transition_contribution=transition_value,
            skip_contribution=skip_value,
            predecessor_stage=predecessor,
            selection_explanation=explanation,
            selected_contact_offset_seconds=selected_offset,
            manual_contact_offset_seconds=manual_offset,
        )
        traces[stage] = _stage_trace(grid, attempt, stage, manual_time, selected_time)

    if reconciled is None:
        return PhaseDebugArtifact(
            attempt_id=selected_phase.attempt_id,
            attempt_range=MediaRange(
                start_seconds=attempt.start_seconds,
                end_seconds=attempt.end_seconds,
            ),
            method_version=PHASE_DEBUG_METHOD_VERSION,
            config_id=config_id,
            stages=stages,  # type: ignore[arg-type]
            traces=traces,  # type: ignore[arg-type]
            anomalies=tuple(selected_phase.anomalies),
            total_objective=None,
            solver_config_id=None,
            solver_method_version=None,
            reconciliation_method_version=None,
        )
    assert solver_result is not None and solver_config is not None
    assert reconciled_total is not None
    return PhaseDebugArtifact(
        attempt_id=selected_phase.attempt_id,
        attempt_range=MediaRange(
            start_seconds=attempt.start_seconds,
            end_seconds=attempt.end_seconds,
        ),
        method_version=PHASE_DEBUG_METHOD_VERSION,
        config_id=config_id,
        stages=stages,  # type: ignore[arg-type]
        traces=traces,  # type: ignore[arg-type]
        anomalies=tuple(selected_phase.anomalies),
        total_objective=float(reconciled_total),
        solver_config_id=solver_config.config_id,
        solver_method_version=solver_result.method_version,
        reconciliation_method_version=RECONCILIATION_METHOD_VERSION,
    )
