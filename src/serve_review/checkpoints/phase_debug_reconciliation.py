"""Pure deterministic solver reconciliation replay (M4 Unit 2).

Transparent, read-only explanation of one existing
:class:`PhaseSolverResult` DP path from its own
:class:`PhaseEvidence` candidates and :class:`PhaseSolverConfig`.
No re-solving, no candidate generation, no threshold or production
behavior changes: the DP selection is taken as given and only the
chosen path is re-stated as per-stage unary/transition/skip shares.

Shares (documented honest semantics, mirrored in
:mod:`serve_review.checkpoints.phase_debug`)::

- selected stage: ``unary = <exact-match candidate score>``,
  ``transition = transition_bonus`` when a feasible selected
  predecessor exists else ``0.0`` for the first selected stage,
  ``skip = null``; ``objective = unary + transition``;
- skipped stage: ``unary = null``, ``transition = null``,
  ``skip = -skip_cost(stage)`` (``-contact_skip_penalty`` for contact);
  ``objective = skip``.

``total = sum(objective)`` must match
``PhaseSolverResult.total_score`` within
``RECONCILIATION_OBJECTIVE_TOLERANCE``. Contact uses the same raw
candidate score as every other stage: the advisory body-compatibility
confidence adjustment affects only the emitted ``AttemptPhase``
confidence/provenance, never the DP objective, so reconciliation
honestly replays the raw score. Transition feasibility replays the
solver predicate exactly (chronological keyframes, non-overlapping
intervals, gap within ``[min_gap, max_gap * span]`` with the solver
epsilon); an infeasible consecutive selected pair or a selected time
without an exact candidate match is rejected rather than invented.

Mismatches rejected with :class:`PhaseDebugError`: wrong input types,
result/config identity mismatch, range/method linkage mismatch,
``selected_keyframes`` disagreeing with the wrapped ``AttemptPhase``,
selected times without an exact evidence keyframe, infeasible chosen
transitions, and total mismatch. No inference, file I/O, CLI/pipeline,
or dense extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from serve_review.checkpoints.evidence import (
    EVIDENCE_METHOD_VERSION,
    PhaseEvidence,
    StageCandidate,
)
from serve_review.checkpoints.phase_debug import (
    RECONCILIATION_METHOD_VERSION,
    RECONCILIATION_OBJECTIVE_TOLERANCE,
    PhaseDebugError,
)
from serve_review.checkpoints.phase_solver import (
    PHASE_SOLVER_METHOD_VERSION,
    PhaseSolverConfig,
    PhaseSolverResult,
)
from serve_review.domain import STAGE_ORDER

__all__ = [
    "RECONCILIATION_METHOD_VERSION",
    "RECONCILIATION_OBJECTIVE_TOLERANCE",
    "StageReconciliation",
    "explain_solver_path",
]

_CANONICAL_STAGES: tuple[str, ...] = STAGE_ORDER

#: Solver gap epsilon replayed exactly from phase_solver (deterministic).
_GAP_EPSILON_SECONDS = 1e-9


@dataclass(frozen=True, slots=True)
class StageReconciliation:
    """Per-stage replay share for the existing chosen path."""

    stage: str
    selected: bool
    unary_contribution: float | None
    transition_contribution: float | None
    skip_contribution: float | None
    objective_contribution: float
    predecessor_stage: str | None
    selection_explanation: str
    selected_rank: int | None


def _transition_feasible_replay(
    earlier: StageCandidate,
    later: StageCandidate,
    earlier_index: int,
    later_index: int,
    config: PhaseSolverConfig,
) -> bool:
    """Replay the solver transition predicate transparently (no re-solve)."""
    if float(later.keyframe_seconds) < float(earlier.keyframe_seconds):
        return False
    if float(later.interval.start_seconds) < float(
        earlier.interval.end_seconds
    ):
        return False
    gap = float(later.keyframe_seconds) - float(earlier.keyframe_seconds)
    span = later_index - earlier_index
    if gap < float(config.min_transition_gap_seconds) - _GAP_EPSILON_SECONDS:
        return False
    if gap > float(config.max_gap_for_span(span)) + _GAP_EPSILON_SECONDS:
        return False
    return True


def _exact_match(
    candidates: list[StageCandidate], keyframe: float
) -> tuple[StageCandidate, int] | None:
    for position, candidate in enumerate(candidates):
        if float(candidate.keyframe_seconds) == float(keyframe):
            return candidate, position + 1
    return None


def explain_solver_path(
    evidence: PhaseEvidence,
    solver_result: PhaseSolverResult,
    solver_config: PhaseSolverConfig,
) -> tuple[dict[str, StageReconciliation], float]:
    """Explain the existing chosen DP path; return (per_stage, total).

    Raises :class:`PhaseDebugError` on any mismatch instead of inventing
    reconciliation. Pure and deterministic.
    """
    name = "phase_debug_reconciliation"
    if not isinstance(evidence, PhaseEvidence):
        raise PhaseDebugError(
            f"{name}: 'evidence' must be a PhaseEvidence, "
            f"got {type(evidence).__name__}."
        )
    if not isinstance(solver_result, PhaseSolverResult):
        raise PhaseDebugError(
            f"{name}: 'solver_result' must be a PhaseSolverResult, "
            f"got {type(solver_result).__name__}."
        )
    if not isinstance(solver_config, PhaseSolverConfig):
        raise PhaseDebugError(
            f"{name}: 'solver_config' must be a PhaseSolverConfig, "
            f"got {type(solver_config).__name__}."
        )
    if solver_result.config_id != solver_config.config_id:
        raise PhaseDebugError(
            f"{name}: solver result config_id {solver_result.config_id!r} "
            f"does not match solver config {solver_config.config_id!r}; "
            f"reconciliation never mixes configurations."
        )
    if solver_result.method_version != PHASE_SOLVER_METHOD_VERSION:
        raise PhaseDebugError(
            f"{name}: solver result method_version "
            f"{solver_result.method_version!r} is not "
            f"{PHASE_SOLVER_METHOD_VERSION!r}."
        )
    if evidence.method_version != EVIDENCE_METHOD_VERSION:
        raise PhaseDebugError(
            f"{name}: evidence method_version {evidence.method_version!r} "
            f"is not compatible with {EVIDENCE_METHOD_VERSION!r}."
        )
    attempt_range = solver_result.attempt_phase.attempt_range
    if (
        evidence.attempt_range.start_seconds != attempt_range.start_seconds
        or evidence.attempt_range.end_seconds != attempt_range.end_seconds
    ):
        raise PhaseDebugError(
            f"{name}: evidence attempt range "
            f"{evidence.attempt_range.to_dict()!r} does not match solver "
            f"attempt range {attempt_range.to_dict()!r}."
        )
    selected_map: Mapping[str, float | None] = solver_result.selected_keyframes
    if not isinstance(selected_map, Mapping) or set(selected_map.keys()) != set(
        _CANONICAL_STAGES
    ):
        raise PhaseDebugError(
            f"{name}: solver result must carry exactly the eight stage keys."
        )
    # selected_keyframes must agree exactly with the wrapped AttemptPhase.
    for stage in _CANONICAL_STAGES:
        keyed = selected_map[stage]
        phased = solver_result.attempt_phase.stages[stage].keyframe_seconds
        if keyed is None and phased is None:
            continue
        if keyed is None or phased is None:
            raise PhaseDebugError(
                f"{name}: stage {stage!r} selected keyframe "
                f"({keyed!r}) disagrees with wrapped attempt phase "
                f"({phased!r})."
            )
        if float(keyed) != float(phased):
            raise PhaseDebugError(
                f"{name}: stage {stage!r} selected keyframe "
                f"({keyed!r}) disagrees with wrapped attempt phase "
                f"({phased!r})."
            )

    per_stage_candidates: list[list[StageCandidate]] = [
        list(evidence.for_stage(stage)) for stage in _CANONICAL_STAGES
    ]
    matched: dict[str, tuple[StageCandidate, int] | None] = {}
    for index, stage in enumerate(_CANONICAL_STAGES):
        keyed = selected_map[stage]
        if keyed is None:
            matched[stage] = None
            continue
        hit = _exact_match(per_stage_candidates[index], float(keyed))
        if hit is None:
            raise PhaseDebugError(
                f"{name}: stage {stage!r} selected keyframe {float(keyed)!r} "
                f"has no exact candidate match; reconciliation never "
                f"interpolates or invents unary scores."
            )
        matched[stage] = hit

    # Predecessor chain over the existing chosen path.
    selected_indices = [
        index
        for index, stage in enumerate(_CANONICAL_STAGES)
        if matched[stage] is not None
    ]
    predecessor_of: dict[str, str | None] = {stage: None for stage in _CANONICAL_STAGES}
    for pos, index in enumerate(selected_indices):
        if pos == 0:
            predecessor_of[_CANONICAL_STAGES[index]] = None
        else:
            predecessor_of[_CANONICAL_STAGES[index]] = _CANONICAL_STAGES[
                selected_indices[pos - 1]
            ]

    # Feasibility of every consecutive chosen pair (possibly non-adjacent
    # when intermediate stages are skipped).
    for pos in range(1, len(selected_indices)):
        earlier_index = selected_indices[pos - 1]
        later_index = selected_indices[pos]
        earlier_stage = _CANONICAL_STAGES[earlier_index]
        later_stage = _CANONICAL_STAGES[later_index]
        earlier_hit = matched[earlier_stage]
        later_hit = matched[later_stage]
        assert earlier_hit is not None and later_hit is not None
        if not _transition_feasible_replay(
            earlier_hit[0], later_hit[0], earlier_index, later_index, solver_config
        ):
            raise PhaseDebugError(
                f"{name}: chosen transition {earlier_stage!r} -> "
                f"{later_stage!r} is infeasible under the solver bounds; "
                f"reconciliation rejects rather than inventing a bonus."
            )

    out: dict[str, StageReconciliation] = {}
    for index, stage in enumerate(_CANONICAL_STAGES):
        hit = matched[stage]
        if hit is None:
            skip = -float(solver_config.skip_cost(stage))
            if per_stage_candidates[index]:
                explanation = "skipped_for_global_consistency"
            else:
                explanation = "no_candidate_available"
            out[stage] = StageReconciliation(
                stage=stage,
                selected=False,
                unary_contribution=None,
                transition_contribution=None,
                skip_contribution=skip,
                objective_contribution=skip,
                predecessor_stage=None,
                selection_explanation=explanation,
                selected_rank=None,
            )
        else:
            candidate, rank = hit
            unary = float(candidate.score)
            predecessor = predecessor_of[stage]
            transition = (
                0.0 if predecessor is None else float(solver_config.transition_bonus)
            )
            objective = unary + transition
            out[stage] = StageReconciliation(
                stage=stage,
                selected=True,
                unary_contribution=unary,
                transition_contribution=transition,
                skip_contribution=None,
                objective_contribution=objective,
                predecessor_stage=predecessor,
                selection_explanation=f"selected_candidate_rank_{rank}",
                selected_rank=int(rank),
            )

    total = sum(entry.objective_contribution for entry in out.values())
    if (
        abs(float(total) - float(solver_result.total_score))
        > RECONCILIATION_OBJECTIVE_TOLERANCE
    ):
        raise PhaseDebugError(
            f"{name}: reconciled total ({total!r}) does not match solver "
            f"total_score ({solver_result.total_score!r}) within "
            f"{RECONCILIATION_OBJECTIVE_TOLERANCE}."
        )
    return out, float(total)
