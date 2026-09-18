"""Constrained six-anchor dynamic-programming selection (M4.10b).

Pure, deterministic, in-memory selection of one globally coherent,
optionally incomplete six-anchor sequence (``start``, ``release``,
``loading``, ``cocking``, ``contact``, ``finish``) from M4.10a dense
composite candidates
(:class:`~serve_review.checkpoints.composite_anchors.CompositeAnchorCandidate`).

This module performs no filtering, no waveform construction, no
candidate scoring, no media decoding, no cache I/O, no audio handling,
and no CLI. It consumes only dense per-stage candidate lists plus
:class:`SixAnchorSolverConfig`, which owns skip penalties, transition gap
bounds, the optional contact-to-finish cap, and transition bonus.

Adaptation notes (point candidates carry no intervals):

- M4.4 candidates carry ``(interval, keyframe)`` pairs; composite
  anchor candidates carry a single exact waveform PTS
  (``time_seconds``). The interval-overlap check therefore reduces to
  strict PTS order: a later pick must satisfy
  ``later.time_seconds > earlier.time_seconds``. The gap bounds
  (``[min_transition_gap_seconds,
  max_transition_gap_seconds * span]``), the per-span scaling, the
  ``transition_bonus`` for feasible consecutive picks, the explicit
  per-stage skip penalties (larger for contact), and the deterministic
  earliest-on-tie rule are preserved verbatim.
- Strict (rather than non-strict) order is required so every selected
  stage can own a valid half-open linkage interval downstream: two
  point estimates at the identical PTS cannot own non-overlapping
  half-open intervals. Everything else matches M4.4 exactly.
- Only availability-qualified candidates are eligible: the caller
  passes per-stage lists already restricted to ``coverage > 0``
  (at least one cue family with qualified waveform support).
  Zero-coverage frames carry no discriminative evidence and are never
  selectable; a stage with no eligible candidate is honestly skipped.
  ``acceleration``/``deceleration`` are never searched here; the
  caller derives them as exact midpoints after anchor selection.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from serve_review.checkpoints.composite_anchors import CompositeAnchorCandidate

__all__ = [
    "SIX_ANCHOR_STAGES",
    "SIX_ANCHOR_METHOD_VERSION",
    "SixAnchorSolverConfig",
    "SixAnchorSolverError",
    "SixAnchorSolution",
    "transition_feasible",
    "solve_six_anchors",
]

#: Canonical six-anchor order (subset of the Kovacs keys; no reordering).
SIX_ANCHOR_STAGES: tuple[str, ...] = (
    "start",
    "release",
    "loading",
    "cocking",
    "contact",
    "finish",
)

#: Method identity for six-anchor DP selection records.
SIX_ANCHOR_METHOD_VERSION = "six-anchor-dp-v1"
SIX_ANCHOR_DEFAULT_CONFIG_ID = "phase-solver-default-v1"


@dataclass(frozen=True, slots=True)
class SixAnchorSolverConfig:
    """Only the chronology policy consumed by the six-anchor DP."""

    config_id: str = SIX_ANCHOR_DEFAULT_CONFIG_ID
    skip_penalty: float = 0.30
    contact_skip_penalty: float = 0.80
    min_transition_gap_seconds: float = 0.0
    max_transition_gap_seconds: float = 1.50
    max_contact_to_finish_seconds: float | None = None
    transition_bonus: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.skip_penalty,
            self.contact_skip_penalty,
            self.min_transition_gap_seconds,
            self.max_transition_gap_seconds,
            self.transition_bonus,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0 for value in values):
            raise SixAnchorSolverError("six_anchor_solver_config: values must be non-negative.")
        if self.max_transition_gap_seconds <= 0 or self.min_transition_gap_seconds > self.max_transition_gap_seconds:
            raise SixAnchorSolverError("six_anchor_solver_config: invalid transition gap bounds.")
        if self.max_contact_to_finish_seconds is not None and self.max_contact_to_finish_seconds <= 0:
            raise SixAnchorSolverError("six_anchor_solver_config: contact-to-finish cap must be > 0.")
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise SixAnchorSolverError("six_anchor_solver_config: config_id must be non-blank.")

    def skip_cost(self, stage: str) -> float:
        return float(self.contact_skip_penalty if stage == "contact" else self.skip_penalty)

    def max_gap_for_span(self, span: int) -> float:
        return float(self.max_transition_gap_seconds) * span

#: Numerical tolerance for gap-bound comparisons (same as M4.4).
_GAP_EPSILON_SECONDS = 1e-9


class SixAnchorSolverError(ValueError):
    """Raised when six-anchor solver input or configuration is invalid."""


def _check_candidates(
    name: str,
    candidates: Mapping[str, Sequence[CompositeAnchorCandidate]],
) -> dict[str, list[CompositeAnchorCandidate]]:
    if not isinstance(candidates, Mapping):
        raise SixAnchorSolverError(
            f"{name}: 'candidates' must be a mapping of stage key to "
            f"candidate list, got {type(candidates).__name__}."
        )
    if set(candidates.keys()) != set(SIX_ANCHOR_STAGES):
        raise SixAnchorSolverError(
            f"{name}: 'candidates' must contain exactly the six anchor "
            f"stages {list(SIX_ANCHOR_STAGES)}, got "
            f"{sorted(candidates.keys())!r}."
        )
    cleaned: dict[str, list[CompositeAnchorCandidate]] = {}
    for stage in SIX_ANCHOR_STAGES:
        raw = candidates[stage]
        if not isinstance(raw, (list, tuple)):
            raise SixAnchorSolverError(
                f"{name}: candidates for stage {stage!r} must be a list or "
                f"tuple, got {type(raw).__name__}."
            )
        items = list(raw)
        for entry in items:
            if not isinstance(entry, CompositeAnchorCandidate):
                raise SixAnchorSolverError(
                    f"{name}: every candidate for stage {stage!r} must be a "
                    "CompositeAnchorCandidate, got "
                    f"{type(entry).__name__}."
                )
            if entry.stage != stage:
                raise SixAnchorSolverError(
                    f"{name}: candidate stage {entry.stage!r} does not match "
                    f"list stage {stage!r}; evidence from another stage is "
                    "never mixed silently."
                )
        for earlier, later in zip(items, items[1:]):
            if not later.time_seconds > earlier.time_seconds:
                raise SixAnchorSolverError(
                    f"{name}: candidates for stage {stage!r} must be in "
                    "strictly increasing PTS order."
                )
        cleaned[stage] = items
    return cleaned


def transition_feasible(
    earlier: CompositeAnchorCandidate,
    later: CompositeAnchorCandidate,
    earlier_index: int,
    later_index: int,
    config: SixAnchorSolverConfig,
) -> bool:
    """Return True when two selected candidates may be consecutive picks.

    Preserves the M4.4 transition semantics on point candidates:
    strictly chronological PTS, with the keyframe gap inside
    ``[min_transition_gap_seconds,
    max_transition_gap_seconds * (later_index - earlier_index)]``. When
    configured, the direct ``contact -> finish`` pair additionally obeys
    ``max_contact_to_finish_seconds``; this does not constrain any other
    stage pair or the legacy eight-stage solver.
    """
    if not later.time_seconds > earlier.time_seconds:
        return False
    gap = later.time_seconds - earlier.time_seconds
    span = later_index - earlier_index
    if gap < float(config.min_transition_gap_seconds) - _GAP_EPSILON_SECONDS:
        return False
    if gap > float(config.max_gap_for_span(span)) + _GAP_EPSILON_SECONDS:
        return False
    finish_cap = config.max_contact_to_finish_seconds
    if (
        earlier.stage == "contact"
        and later.stage == "finish"
        and finish_cap is not None
        and gap > float(finish_cap) + _GAP_EPSILON_SECONDS
    ):
        return False
    return True


@dataclass(frozen=True, slots=True)
class SixAnchorSolution:
    """Immutable record of one six-anchor DP selection.

    ``choice`` maps each anchor stage to the selected position within
    its eligible candidate list, or null for an honest skip.
    ``total_score`` is the deterministic DP objective value
    (``sum unary + sum bonus - skip penalties``).
    """

    choice: Mapping[str, int | None] = None  # type: ignore[assignment]
    total_score: float = 0.0
    method_version: str = SIX_ANCHOR_METHOD_VERSION
    config_id: str = ""

    def __post_init__(self) -> None:
        name = "six_anchor_solution"
        raw = self.choice
        if not isinstance(raw, Mapping):
            raise SixAnchorSolverError(
                f"{name}: 'choice' must be a mapping of stage key to "
                f"position or null, got {type(raw).__name__}."
            )
        if set(raw.keys()) != set(SIX_ANCHOR_STAGES):
            raise SixAnchorSolverError(
                f"{name}: 'choice' must contain exactly the six anchor "
                f"stages {list(SIX_ANCHOR_STAGES)}."
            )
        cleaned: dict[str, int | None] = {}
        for stage in SIX_ANCHOR_STAGES:
            value = raw[stage]
            if value is None:
                cleaned[stage] = None
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise SixAnchorSolverError(
                    f"{name}: choice for {stage!r} must be an integer >= 0 "
                    f"or null, got {value!r}."
                )
            cleaned[stage] = value
        object.__setattr__(self, "choice", MappingProxyType(cleaned))
        total = self.total_score
        if (
            isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(float(total))
        ):
            raise SixAnchorSolverError(
                f"{name}: 'total_score' must be a finite number, "
                f"got {total!r}."
            )
        object.__setattr__(self, "total_score", float(total))
        if not isinstance(self.method_version, str) or (
            self.method_version != SIX_ANCHOR_METHOD_VERSION
        ):
            raise SixAnchorSolverError(
                f"{name}: 'method_version' must equal "
                f"{SIX_ANCHOR_METHOD_VERSION!r}."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise SixAnchorSolverError(
                f"{name}: 'config_id' must be a non-blank string."
            )

    def selected_index(self, stage: str) -> int | None:
        """Return the selected candidate position for ``stage`` (or null)."""
        if stage not in SIX_ANCHOR_STAGES:
            raise SixAnchorSolverError(
                f"six_anchor_solution: unknown stage {stage!r}."
            )
        value = self.choice[stage]  # type: ignore[index]
        assert value is None or isinstance(value, int)
        return value


def solve_six_anchors(
    candidates: Mapping[str, Sequence[CompositeAnchorCandidate]],
    config: SixAnchorSolverConfig | None = None,
) -> SixAnchorSolution:
    """Select the globally coherent six-anchor sequence.

    Maximizes ``sum S_i(T_i) + sum Q(T_i, T_{i+1})`` with the M4.4
    recurrence: every stage is either assigned one of its eligible
    candidates (unary ``candidate.score``) or an explicit skip worth
    ``-skip_penalty`` (``-contact_skip_penalty`` for contact); a
    feasible consecutive pair earns ``transition_bonus``; ties resolve
    deterministically to the earliest candidate with the earliest
    predecessor (strict ``>`` comparison over fixed orders).

    Args:
        candidates: Mapping of anchor stage to its eligible
            (``coverage > 0``) candidate list in PTS order; an empty
            list honestly means no selectable evidence for that stage.
        config: Solver configuration owning penalties/gap bounds/bonus
            (defaults to :class:`SixAnchorSolverConfig` defaults).

    Returns:
        An immutable :class:`SixAnchorSolution`.
    """
    name = "solve_six_anchors"
    cfg = config if config is not None else SixAnchorSolverConfig()
    if not isinstance(cfg, SixAnchorSolverConfig):
        raise SixAnchorSolverError(
            f"{name}: 'config' must be a SixAnchorSolverConfig, "
            f"got {type(config).__name__}."
        )
    per_stage_map = _check_candidates(name, candidates)
    per_stage = [per_stage_map[stage] for stage in SIX_ANCHOR_STAGES]
    count = len(SIX_ANCHOR_STAGES)
    skip_costs = [cfg.skip_cost(stage) for stage in SIX_ANCHOR_STAGES]

    first: list[list[float]] = []
    back_sel: list[list[tuple[int, int] | None]] = []
    second: list[float] = []
    back_skip_cand: list[int | None] = []

    for i in range(count):
        options = per_stage[i]
        row: list[float] = []
        brow: list[tuple[int, int] | None] = []
        lone_base = -sum(skip_costs[:i])
        for c, candidate in enumerate(options):
            _ = c
            best = float(candidate.score) + lone_base
            back: tuple[int, int] | None = None
            for j in range(i):
                base = -sum(skip_costs[j + 1 : i])
                for d, earlier in enumerate(per_stage[j]):
                    if not first[j]:
                        continue
                    if not transition_feasible(
                        earlier, candidate, j, i, cfg
                    ):
                        continue
                    total = (
                        first[j][d]
                        + base
                        + float(candidate.score)
                        + float(cfg.transition_bonus)
                    )
                    if total > best:
                        best = total
                        back = (j, d)
            row.append(best)
            brow.append(back)
        first.append(row)
        back_sel.append(brow)
        if i == 0:
            second.append(-skip_costs[0])
            back_skip_cand.append(None)
        else:
            cont = second[i - 1] - skip_costs[i]
            best_pred = cont
            best_cand: int | None = None
            for d in range(len(per_stage[i - 1])):
                total = first[i - 1][d] - skip_costs[i]
                if total > best_pred:
                    best_pred = total
                    best_cand = d
            second.append(best_pred)
            back_skip_cand.append(best_cand)

    last = count - 1
    trailing = -sum(skip_costs)
    best_total = trailing
    end: tuple[int, int] | None = None
    for j in range(count):
        base = -sum(skip_costs[j + 1 :])
        for d in range(len(per_stage[j])):
            total = first[j][d] + base
            if total > best_total:
                best_total = total
                end = (j, d)

    choice: dict[str, int | None] = dict.fromkeys(SIX_ANCHOR_STAGES)
    if end is not None:
        end_stage, end_cand = end
        choice[SIX_ANCHOR_STAGES[end_stage]] = end_cand
        cursor: tuple[int, int] | None = back_sel[end_stage][end_cand]
        while cursor is not None:
            prev_stage, prev_cand = cursor
            choice[SIX_ANCHOR_STAGES[prev_stage]] = prev_cand
            cursor = back_sel[prev_stage][prev_cand]

    return SixAnchorSolution(
        choice=choice,
        total_score=float(best_total),
        method_version=SIX_ANCHOR_METHOD_VERSION,
        config_id=cfg.config_id,
    )
