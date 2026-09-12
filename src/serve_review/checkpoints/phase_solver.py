"""Constrained joint phase solver (M4.4).

Pure, deterministic, in-memory selection of one globally coherent,
optionally incomplete eight-stage Kovacs sequence from M4.3
:class:`PhaseEvidence` candidates using dynamic programming.

This module never modifies M3 macro attempts/exports, never generates
pose/audio features, never performs file I/O, and never emits
``PhaseDocument``. It reads one M3 :class:`Attempt` (un-padded linkage
only), one M4.2 :class:`PhaseFeatureGrid`, and one M4.3
:class:`PhaseEvidence`, validates that all three describe the same
un-padded attempt range with compatible method versions, and returns one
M4.1 :class:`AttemptPhase` with exactly the eight canonical stage keys.

Objective
---------

Maximize ``sum S_i(T_i) + sum Q(T_i, T_{i+1})`` where ``S_i`` is the
candidate unary score and ``Q`` is the adjacent transition
compatibility between consecutive *selected* stages:

- every stage is either assigned one of its candidates or an explicit
  skip state worth ``-skip_penalty`` (``-contact_skip_penalty`` for
  contact, which is a strong uncertain anchor);
- a transition between consecutive selected stages ``(i, j)`` (possibly
  non-adjacent when intermediate stages are skipped) is feasible only
  when keyframes are chronological, intervals do not overlap, and the
  keyframe gap lies in ``[min_transition_gap_seconds,
  max_transition_gap_seconds * (j - i)]``; feasible transitions earn a
  small ``transition_bonus``;
- ties resolve deterministically to the earliest candidate in stored
  evidence order with the earliest predecessor in canonical stage
  order (strict ``>`` comparison while scanning fixed orders).

A skip is therefore preferable to forcing chronologically implausible
evidence, and transitions never require every stage to be present.

Contact handling
----------------

The DP-selected audio contact candidate is never deleted by body
evidence. An advisory body-compatibility check inspects only existing
grid rows inside the contact's explicit temporal uncertainty
(``[keyframe - uncertainty, keyframe + uncertainty]``):

- maximum available wrist elevation above a shoulder/torso proxy
  (camera-relative image coordinates only; camera ``y`` grows downward,
  so elevation is ``reference_y - wrist_y`` where the reference is the
  topmost available shoulder/torso proxy row, with no anatomical
  forward/posterior language and no handedness assumption);
- compatible wrist motion before the anchor (acceleration/cocking
  support) and motion/transition support after it
  (deceleration support).

Compatible support upgrades provenance to ``body_pose_audio``;
otherwise provenance stays ``audio_transient`` with lowered confidence,
an honest limitation, and a stable anomaly identifier.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from serve_review.checkpoints.evidence import (
    EVIDENCE_METHOD_VERSION,
    PhaseEvidence,
    StageCandidate,
)
from serve_review.checkpoints.phase_features import (
    PHASE_FEATURES_METHOD_VERSION,
    PhaseFeatureGrid,
)
from serve_review.domain import (
    STAGE_ORDER,
    Attempt,
    AttemptPhase,
    MediaRange,
    StagePhase,
)

__all__ = [
    "PHASE_SOLVER_SCHEMA_VERSION",
    "PHASE_SOLVER_METHOD_VERSION",
    "PHASE_SOLVER_DEFAULT_CONFIG_ID",
    "PhaseSolverError",
    "PhaseSolverConfig",
    "PhaseSolverResult",
    "solve_attempt_phase",
    "solve_with_diagnostics",
]

#: Version of the solver configuration/result schemas.
PHASE_SOLVER_SCHEMA_VERSION = 1
#: Method identity recorded on every emitted attempt phase.
PHASE_SOLVER_METHOD_VERSION = "phase-solver-v1"
#: Default configuration identity.
PHASE_SOLVER_DEFAULT_CONFIG_ID = "phase-solver-default-v1"

_CANONICAL_STAGES: tuple[str, ...] = STAGE_ORDER
_CONTACT_STAGE = "contact"
_CONTACT_INDEX = _CANONICAL_STAGES.index(_CONTACT_STAGE)

#: Numerical tolerance for gap-bound comparisons (deterministic).
_GAP_EPSILON_SECONDS = 1e-9


class PhaseSolverError(ValueError):
    """Raised when solver input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseSolverError(
            f"phase_solver_config: {name!r} must be a number >= 0, "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PhaseSolverError(
            f"phase_solver_config: {name!r} must be finite and >= 0, "
            f"got {value!r}."
        )
    return number


def _check_unit(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseSolverError(
            f"phase_solver_config: {name!r} must be a number in [0, 1], "
            f"got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise PhaseSolverError(
            f"phase_solver_config: {name!r} must lie in [0, 1], got {value!r}."
        )
    return number


@dataclass(frozen=True, slots=True)
class PhaseSolverConfig:
    """Versioned, immutable configuration for the joint phase solver.

    Transition bounds are deliberately broad: a consecutive selected
    pair spanning ``span`` canonical stages is feasible for keyframe gaps
    in ``[min_transition_gap_seconds,
    max_transition_gap_seconds * span]``. Skip penalties make honest
    ``unavailable`` stages preferable to implausible evidence; the
    contact skip penalty is larger because audio contact is a strong
    uncertain anchor.
    """

    config_id: str = PHASE_SOLVER_DEFAULT_CONFIG_ID
    skip_penalty: float = 0.30
    contact_skip_penalty: float = 0.80
    min_transition_gap_seconds: float = 0.0
    max_transition_gap_seconds: float = 1.50
    transition_bonus: float = 0.05
    available_observation_floor: float = 0.50
    body_support_quality_floor: float = 0.30
    body_elevation_margin: float = 0.0
    body_min_wrist_speed: float = 0.05
    contact_body_confidence_bonus: float = 0.05
    contact_body_confidence_penalty: float = 0.15
    schema_version: int = PHASE_SOLVER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_solver_config"
        if not _is_int(self.schema_version):
            raise PhaseSolverError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_SOLVER_SCHEMA_VERSION:
            raise PhaseSolverError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_SOLVER_SCHEMA_VERSION}."
            )
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise PhaseSolverError(
                f"{name}: 'config_id' must be a non-blank string, "
                f"got {self.config_id!r}."
            )
        object.__setattr__(
            self, "skip_penalty", _check_nonnegative("'skip_penalty'", self.skip_penalty)
        )
        object.__setattr__(
            self,
            "contact_skip_penalty",
            _check_nonnegative("'contact_skip_penalty'", self.contact_skip_penalty),
        )
        object.__setattr__(
            self,
            "min_transition_gap_seconds",
            _check_nonnegative(
                "'min_transition_gap_seconds'", self.min_transition_gap_seconds
            ),
        )
        object.__setattr__(
            self,
            "max_transition_gap_seconds",
            _check_nonnegative(
                "'max_transition_gap_seconds'", self.max_transition_gap_seconds
            ),
        )
        if not self.max_transition_gap_seconds > 0.0:
            raise PhaseSolverError(
                f"{name}: 'max_transition_gap_seconds' must be > 0, "
                f"got {self.max_transition_gap_seconds!r}."
            )
        if self.min_transition_gap_seconds > self.max_transition_gap_seconds:
            raise PhaseSolverError(
                f"{name}: 'min_transition_gap_seconds' "
                f"({self.min_transition_gap_seconds!r}) must not exceed "
                f"'max_transition_gap_seconds' "
                f"({self.max_transition_gap_seconds!r})."
            )
        object.__setattr__(
            self,
            "transition_bonus",
            _check_nonnegative("'transition_bonus'", self.transition_bonus),
        )
        object.__setattr__(
            self,
            "available_observation_floor",
            _check_unit(
                "'available_observation_floor'", self.available_observation_floor
            ),
        )
        object.__setattr__(
            self,
            "body_support_quality_floor",
            _check_unit(
                "'body_support_quality_floor'", self.body_support_quality_floor
            ),
        )
        object.__setattr__(
            self,
            "body_elevation_margin",
            _check_nonnegative(
                "'body_elevation_margin'", self.body_elevation_margin
            ),
        )
        object.__setattr__(
            self,
            "body_min_wrist_speed",
            _check_nonnegative(
                "'body_min_wrist_speed'", self.body_min_wrist_speed
            ),
        )
        object.__setattr__(
            self,
            "contact_body_confidence_bonus",
            _check_nonnegative(
                "'contact_body_confidence_bonus'",
                self.contact_body_confidence_bonus,
            ),
        )
        object.__setattr__(
            self,
            "contact_body_confidence_penalty",
            _check_nonnegative(
                "'contact_body_confidence_penalty'",
                self.contact_body_confidence_penalty,
            ),
        )

    def skip_cost(self, stage: str) -> float:
        """Return the explicit skip penalty for ``stage``."""
        if stage not in _CANONICAL_STAGES:
            raise PhaseSolverError(
                f"phase_solver_config: unknown stage key {stage!r}."
            )
        if stage == _CONTACT_STAGE:
            return float(self.contact_skip_penalty)
        return float(self.skip_penalty)

    def max_gap_for_span(self, span: int) -> float:
        """Return the broad maximum gap for a pair spanning ``span`` stages."""
        if not _is_int(span) or span < 1:
            raise PhaseSolverError(
                f"phase_solver_config: 'span' must be an integer >= 1, "
                f"got {span!r}."
            )
        return float(self.max_transition_gap_seconds) * span

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_observation_floor": self.available_observation_floor,
            "body_elevation_margin": self.body_elevation_margin,
            "body_min_wrist_speed": self.body_min_wrist_speed,
            "body_support_quality_floor": self.body_support_quality_floor,
            "config_id": self.config_id,
            "contact_body_confidence_bonus": self.contact_body_confidence_bonus,
            "contact_body_confidence_penalty": self.contact_body_confidence_penalty,
            "contact_skip_penalty": self.contact_skip_penalty,
            "max_transition_gap_seconds": self.max_transition_gap_seconds,
            "min_transition_gap_seconds": self.min_transition_gap_seconds,
            "schema_version": self.schema_version,
            "skip_penalty": self.skip_penalty,
            "transition_bonus": self.transition_bonus,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseSolverConfig:
        name = "phase_solver_config"
        if not isinstance(values, dict):
            raise PhaseSolverError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "available_observation_floor",
            "body_elevation_margin",
            "body_min_wrist_speed",
            "body_support_quality_floor",
            "config_id",
            "contact_body_confidence_bonus",
            "contact_body_confidence_penalty",
            "contact_skip_penalty",
            "max_transition_gap_seconds",
            "min_transition_gap_seconds",
            "schema_version",
            "skip_penalty",
            "transition_bonus",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseSolverError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseSolverError(f"{name}: unknown keys {unknown!r}.")
        try:
            return cls(**{key: values[key] for key in known})  # type: ignore[arg-type]
        except PhaseSolverError:
            raise
        except Exception as exc:
            raise PhaseSolverError(f"{name}: invalid configuration: {exc}.") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PhaseSolverConfig:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PhaseSolverError(
                    "phase_solver_config: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise PhaseSolverError(
                "phase_solver_config: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PhaseSolverError(
                f"phase_solver_config: invalid JSON: {exc}."
            ) from exc
        if not isinstance(decoded, dict):
            raise PhaseSolverError(
                "phase_solver_config: JSON object is required, "
                f"got {type(decoded).__name__}."
            )
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class PhaseSolverResult:
    """Immutable versioned record of one solver run.

    Bundles the emitted :class:`AttemptPhase` with the deterministic
    dynamic-programming objective value (``total_score``) and the
    per-stage selection summary (selected candidate keyframe or null for
    skips). Codecs are deterministic; this type performs no I/O.
    """

    attempt_phase: AttemptPhase = None  # type: ignore[assignment]
    total_score: float = 0.0
    selected_keyframes: Mapping[str, float | None] = None  # type: ignore[assignment]
    method_version: str = PHASE_SOLVER_METHOD_VERSION
    config_id: str = PHASE_SOLVER_DEFAULT_CONFIG_ID
    schema_version: int = PHASE_SOLVER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        name = "phase_solver_result"
        if not _is_int(self.schema_version):
            raise PhaseSolverError(
                f"{name}: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != PHASE_SOLVER_SCHEMA_VERSION:
            raise PhaseSolverError(
                f"{name}: unsupported schema_version {self.schema_version!r}; "
                f"expected {PHASE_SOLVER_SCHEMA_VERSION}."
            )
        if not isinstance(self.attempt_phase, AttemptPhase):
            raise PhaseSolverError(
                f"{name}: 'attempt_phase' must be an AttemptPhase, "
                f"got {type(self.attempt_phase).__name__}."
            )
        if (
            isinstance(self.total_score, bool)
            or not isinstance(self.total_score, (int, float))
            or not math.isfinite(float(self.total_score))
        ):
            raise PhaseSolverError(
                f"{name}: 'total_score' must be a finite number, "
                f"got {self.total_score!r}."
            )
        object.__setattr__(self, "total_score", float(self.total_score))
        if not isinstance(self.method_version, str) or not self.method_version.strip():
            raise PhaseSolverError(f"{name}: 'method_version' must be non-blank.")
        if not isinstance(self.config_id, str) or not self.config_id.strip():
            raise PhaseSolverError(f"{name}: 'config_id' must be non-blank.")
        raw = self.selected_keyframes
        if not isinstance(raw, Mapping):
            raise PhaseSolverError(
                f"{name}: 'selected_keyframes' must be a mapping of stage key "
                f"to keyframe or null, got {type(raw).__name__}."
            )
        if set(raw.keys()) != set(_CANONICAL_STAGES):
            raise PhaseSolverError(
                f"{name}: 'selected_keyframes' must contain exactly the eight "
                f"stage keys {list(_CANONICAL_STAGES)}, "
                f"got {sorted(raw.keys())!r}."
            )
        cleaned: dict[str, float | None] = {}
        for key in _CANONICAL_STAGES:
            value = raw[key]
            if value is None:
                cleaned[key] = None
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PhaseSolverError(
                    f"{name}: selected keyframe for {key!r} must be finite or "
                    f"null, got {value!r}."
                )
            cleaned[key] = float(value)
        object.__setattr__(
            self, "selected_keyframes", MappingProxyType(cleaned)
        )

    def to_dict(self) -> dict[str, Any]:
        selected = self.selected_keyframes
        assert isinstance(selected, Mapping)
        return {
            "attempt_phase": self.attempt_phase.to_dict(),
            "config_id": self.config_id,
            "method_version": self.method_version,
            "schema_version": self.schema_version,
            "selected_keyframes": {
                key: selected[key] for key in _CANONICAL_STAGES
            },
            "total_score": self.total_score,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PhaseSolverResult:
        name = "phase_solver_result"
        if not isinstance(values, dict):
            raise PhaseSolverError(
                f"{name}: mapping is required, got {type(values).__name__}."
            )
        known = {
            "attempt_phase",
            "config_id",
            "method_version",
            "schema_version",
            "selected_keyframes",
            "total_score",
        }
        missing = sorted(known - set(values))
        if missing:
            raise PhaseSolverError(f"{name}: missing required keys {missing!r}.")
        unknown = sorted(set(values) - known)
        if unknown:
            raise PhaseSolverError(f"{name}: unknown keys {unknown!r}.")
        raw_phase = values["attempt_phase"]
        if not isinstance(raw_phase, dict):
            raise PhaseSolverError(
                f"{name}: 'attempt_phase' must decode from a mapping, "
                f"got {type(raw_phase).__name__}."
            )
        try:
            attempt_phase = AttemptPhase.from_dict(raw_phase)
        except Exception as exc:
            raise PhaseSolverError(
                f"{name}: invalid 'attempt_phase': {exc}."
            ) from exc
        raw_selected = values["selected_keyframes"]
        if not isinstance(raw_selected, dict):
            raise PhaseSolverError(
                f"{name}: 'selected_keyframes' must decode from a mapping."
            )
        return cls(
            attempt_phase=attempt_phase,
            total_score=values["total_score"],
            selected_keyframes=dict(raw_selected),
            method_version=values["method_version"],
            config_id=values["config_id"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> PhaseSolverResult:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PhaseSolverError(
                    "phase_solver_result: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise PhaseSolverError(
                "phase_solver_result: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise PhaseSolverError(
                f"phase_solver_result: invalid JSON: {exc}."
            ) from exc
        if not isinstance(decoded, dict):
            raise PhaseSolverError(
                "phase_solver_result: JSON object is required, "
                f"got {type(decoded).__name__}."
            )
        return cls.from_dict(decoded)


# --- Input validation --------------------------------------------------------


def _validate_inputs(
    attempt: Attempt,
    grid: PhaseFeatureGrid,
    evidence: PhaseEvidence,
    config: PhaseSolverConfig,
) -> MediaRange:
    """Validate linkage/range/method consistency; return unpadded range."""
    name = "solve_attempt_phase"
    if not isinstance(config, PhaseSolverConfig):
        raise PhaseSolverError(
            f"{name}: 'config' must be a PhaseSolverConfig, "
            f"got {type(config).__name__}."
        )
    if not isinstance(attempt, Attempt):
        raise PhaseSolverError(
            f"{name}: 'attempt' must be an Attempt linkage, "
            f"got {type(attempt).__name__}."
        )
    if not isinstance(grid, PhaseFeatureGrid):
        raise PhaseSolverError(
            f"{name}: 'grid' must be a PhaseFeatureGrid, "
            f"got {type(grid).__name__}."
        )
    if not isinstance(evidence, PhaseEvidence):
        raise PhaseSolverError(
            f"{name}: 'evidence' must be a PhaseEvidence, "
            f"got {type(evidence).__name__}."
        )
    unpadded = attempt.detected_range
    if (
        grid.attempt_range.start_seconds != unpadded.start_seconds
        or grid.attempt_range.end_seconds != unpadded.end_seconds
    ):
        raise PhaseSolverError(
            f"{name}: grid attempt range "
            f"{grid.attempt_range.to_dict()!r} does not match unpadded "
            f"attempt range {unpadded.to_dict()!r}; evidence from another "
            f"attempt is never mixed silently."
        )
    if (
        evidence.attempt_range.start_seconds != unpadded.start_seconds
        or evidence.attempt_range.end_seconds != unpadded.end_seconds
    ):
        raise PhaseSolverError(
            f"{name}: evidence attempt range "
            f"{evidence.attempt_range.to_dict()!r} does not match unpadded "
            f"attempt range {unpadded.to_dict()!r}; evidence from another "
            f"attempt is never mixed silently."
        )
    if grid.method_version != PHASE_FEATURES_METHOD_VERSION:
        raise PhaseSolverError(
            f"{name}: grid method_version {grid.method_version!r} is not "
            f"compatible with {PHASE_FEATURES_METHOD_VERSION!r}."
        )
    if evidence.method_version != EVIDENCE_METHOD_VERSION:
        raise PhaseSolverError(
            f"{name}: evidence method_version {evidence.method_version!r} is "
            f"not compatible with {EVIDENCE_METHOD_VERSION!r}."
        )
    for candidate in evidence.candidates:
        if candidate.stage not in _CANONICAL_STAGES:
            raise PhaseSolverError(
                f"{name}: unknown candidate stage {candidate.stage!r}."
            )
        if (
            candidate.interval.start_seconds < unpadded.start_seconds
            or candidate.interval.end_seconds > unpadded.end_seconds
        ):
            raise PhaseSolverError(
                f"{name}: candidate interval "
                f"{candidate.interval.to_dict()!r} for stage "
                f"{candidate.stage!r} lies outside the unpadded attempt "
                f"{unpadded.to_dict()!r}."
            )
        if not (
            unpadded.contains(candidate.keyframe_seconds)
            or candidate.keyframe_seconds == unpadded.start_seconds
        ):
            raise PhaseSolverError(
                f"{name}: candidate keyframe {candidate.keyframe_seconds!r} "
                f"for stage {candidate.stage!r} lies outside the unpadded "
                f"attempt {unpadded.to_dict()!r}."
            )
    return unpadded


# --- Transition compatibility -------------------------------------------------


def _transition_feasible(
    earlier: StageCandidate,
    later: StageCandidate,
    earlier_index: int,
    later_index: int,
    config: PhaseSolverConfig,
) -> bool:
    """Return True when two selected candidates may be consecutive picks."""
    if later.keyframe_seconds < earlier.keyframe_seconds:
        return False
    if later.interval.start_seconds < earlier.interval.end_seconds:
        return False
    gap = later.keyframe_seconds - earlier.keyframe_seconds
    span = later_index - earlier_index
    if gap < config.min_transition_gap_seconds - _GAP_EPSILON_SECONDS:
        return False
    if gap > config.max_gap_for_span(span) + _GAP_EPSILON_SECONDS:
        return False
    return True


# --- Dynamic program ----------------------------------------------------------


def _run_dynamic_program(
    per_stage: list[list[StageCandidate]],
    config: PhaseSolverConfig,
) -> tuple[list[int | None], float]:
    """Run the candidate-or-skip DP; return (choice, total_score).

    ``choice[i]`` is the selected candidate position within
    ``per_stage[i]`` or null for an honest skip. Deterministic:
    predecessors and candidates are scanned in canonical/stored order
    with strict ``>`` comparison, so exact ties keep the earliest
    candidate and earliest predecessor.
    """
    count = len(_CANONICAL_STAGES)
    skip_costs = [config.skip_cost(stage) for stage in _CANONICAL_STAGES]

    # F[i][c]: best prefix total for stages 0..i ending with candidate c.
    # back_sel[i][c]: (prev_stage, prev_candidate) or null (lone pick).
    first: list[list[float]] = []
    back_sel: list[list[tuple[int, int] | None]] = []
    # G[i]: best prefix total for stages 0..i with stage i skipped.
    # back_skip[i]: True when the best predecessor selected stage i - 1
    # (with which candidate recorded in back_skip_cand[i]), else the
    # skip chain continues.
    second: list[float] = []
    back_skip_cand: list[int | None] = []

    for i in range(count):
        options = per_stage[i]
        row: list[float] = []
        brow: list[tuple[int, int] | None] = []
        # Cost of skipping every stage before i.
        lone_base = -sum(skip_costs[:i])
        for c, candidate in enumerate(options):
            _ = c
            best = candidate.score + lone_base
            back: tuple[int, int] | None = None
            for j in range(i):
                base = -sum(skip_costs[j + 1 : i])
                for d, earlier in enumerate(per_stage[j]):
                    if not first[j]:
                        continue
                    if not _transition_feasible(
                        earlier, candidate, j, i, config
                    ):
                        continue
                    total = first[j][d] + base + candidate.score + config.transition_bonus
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
    best_total = trailing  # skip every stage
    # (end_stage, end_candidate): null end means the path ends in skips.
    end: tuple[int, int] | None = None
    for j in range(count):
        base = -sum(skip_costs[j + 1 :])
        for d in range(len(per_stage[j])):
            total = first[j][d] + base
            if total > best_total:
                best_total = total
                end = (j, d)

    choice: list[int | None] = [None] * count
    if end is not None:
        end_stage, end_cand = end
        choice[end_stage] = end_cand
        cursor: tuple[int, int] | None = back_sel[end_stage][end_cand]
        # Stages after the last selection stay skipped; stages between
        # consecutive selections stay skipped by construction.
        while cursor is not None:
            prev_stage, prev_cand = cursor
            choice[prev_stage] = prev_cand
            cursor = back_sel[prev_stage][prev_cand]
    return choice, best_total


# --- Contact body compatibility -----------------------------------------------


def _wrist_speed_pair(sample: Any, side: str) -> float | None:
    vx = getattr(sample, f"wrist_{side}_vx")
    vy = getattr(sample, f"wrist_{side}_vy")
    if vx is None or vy is None:
        return None
    value = math.hypot(float(vx), float(vy))
    return value if math.isfinite(value) else None


def _contact_body_support(
    grid: PhaseFeatureGrid,
    candidate: StageCandidate,
    config: PhaseSolverConfig,
) -> tuple[bool, str, tuple[str, ...], tuple[str, ...]]:
    """Assess advisory body support for the selected audio contact.

    Returns ``(compatible, reason, evidence_tokens, limitation_tokens)``
    using only existing grid rows within the candidate's explicit
    temporal uncertainty. ``reason`` is one of ``"supported"``,
    ``"missing"``, ``"occluded"``, or ``"incompatible"``. Camera-relative
    image coordinates only: elevation is ``reference_y - wrist_y``
    (camera ``y`` grows downward); no handedness or anatomical
    forward/posterior claims.
    """
    keyframe = float(candidate.keyframe_seconds)
    uncertainty = float(candidate.temporal_uncertainty_seconds)
    floor = float(config.body_support_quality_floor)
    window = [
        sample
        for sample in grid.samples
        if abs(float(sample.time_seconds) - keyframe)
        <= uncertainty + _GAP_EPSILON_SECONDS
    ]
    if not window:
        return (
            False,
            "missing",
            (),
            ("body_support_unavailable", "camera_relative_only"),
        )
    best_elevation: float | None = None
    best_before: float | None = None
    best_after: float | None = None
    slowing_after: float | None = None
    supporting_rows = 0
    for sample in window:
        refs: list[float] = []
        if (
            sample.shoulder_left_y is not None
            and sample.shoulder_right_y is not None
        ):
            refs.append(
                (float(sample.shoulder_left_y) + float(sample.shoulder_right_y))
                / 2.0
            )
        elif sample.shoulder_left_y is not None:
            refs.append(float(sample.shoulder_left_y))
        elif sample.shoulder_right_y is not None:
            refs.append(float(sample.shoulder_right_y))
        if sample.torso_mid_camera_y is not None:
            refs.append(float(sample.torso_mid_camera_y))
        wrists: list[float] = []
        if sample.wrist_left_y is not None:
            wrists.append(float(sample.wrist_left_y))
        if sample.wrist_right_y is not None:
            wrists.append(float(sample.wrist_right_y))
        usable = (
            sample.observed
            and float(sample.observation_quality) >= floor
            and bool(refs)
            and bool(wrists)
        )
        if usable:
            supporting_rows += 1
        # The wrist must clear the topmost available body proxy (smallest
        # camera y): clearing only a low torso point while sitting below
        # the shoulders is not elevated support.
        ref = min(refs) if refs else None
        if ref is not None:
            for wrist in wrists:
                elevation = ref - wrist
                if math.isfinite(elevation) and (
                    best_elevation is None or elevation > best_elevation
                ):
                    best_elevation = elevation
        before = float(sample.time_seconds) <= keyframe + _GAP_EPSILON_SECONDS
        # The anchor row itself (t == keyframe) belongs to both halves: its
        # velocity describes motion arriving at contact and departing from
        # it, which matters when the uncertainty window spans a single
        # grid row. Rows strictly inside one half support only that half.
        after = float(sample.time_seconds) >= keyframe - _GAP_EPSILON_SECONDS
        for side in ("left", "right"):
            speed = _wrist_speed_pair(sample, side)
            if speed is None:
                continue
            if before and (
                best_before is None or speed > best_before
            ):
                best_before = speed
            if after and (
                best_after is None or speed > best_after
            ):
                best_after = speed
            if after:
                ax = getattr(sample, f"wrist_{side}_ax")
                ay = getattr(sample, f"wrist_{side}_ay")
                vx = getattr(sample, f"wrist_{side}_vx")
                vy = getattr(sample, f"wrist_{side}_vy")
                if (
                    ax is not None
                    and ay is not None
                    and vx is not None
                    and vy is not None
                    and speed > 1e-9
                ):
                    slowing = -(
                        float(vx) * float(ax) + float(vy) * float(ay)
                    ) / speed
                    if math.isfinite(slowing) and (
                        slowing_after is None or slowing > slowing_after
                    ):
                        slowing_after = slowing
    if supporting_rows == 0 or best_elevation is None:
        # Missing or occluded body support: the audio anchor stands, but
        # support is honestly unavailable.
        occluded = any(
            (not sample.observed)
            or sample.wrist_left_y is None
            and sample.wrist_right_y is None
            for sample in window
        )
        reason = "occluded" if occluded else "missing"
        return (
            False,
            reason,
            (),
            (
                "body_support_occluded" if reason == "occluded"
                else "body_support_unavailable",
                "camera_relative_only",
            ),
        )
    min_speed = float(config.body_min_wrist_speed)
    before_ok = best_before is not None and best_before >= min_speed
    after_ok = (best_after is not None and best_after >= min_speed * 0.5) or (
        slowing_after is not None and slowing_after > 0.0
    )
    elevated = best_elevation > float(config.body_elevation_margin)
    if elevated and before_ok and after_ok:
        return (
            True,
            "supported",
            (
                "wrist_elevation_above_shoulder_camera",
                "pre_contact_motion_support_camera",
                "post_contact_motion_support_camera",
            ),
            ("camera_relative_only",),
        )
    return (
        False,
        "incompatible",
        (),
        ("body_support_incompatible", "camera_relative_only"),
    )


# --- Stage emission ------------------------------------------------------------


def _structural_status(availabilities: Sequence[str]) -> str:
    available = sum(1 for value in availabilities if value == "available")
    unavailable = sum(1 for value in availabilities if value == "unavailable")
    total = len(availabilities)
    if available == total:
        return "complete"
    if unavailable == total:
        return "unavailable"
    if available >= 1:
        return "partial"
    return "incomplete"


def _emit_selected(
    stage: str,
    candidate: StageCandidate,
    config: PhaseSolverConfig,
) -> StagePhase:
    name = "solve_attempt_phase"
    if stage == _CONTACT_STAGE:
        raise PhaseSolverError(
            f"{name}: internal error: contact emission uses the "
            f"audio-anchored path."
        )
    if "interpolated_support" in candidate.limitations or (
        float(candidate.observation_quality)
        < float(config.available_observation_floor)
    ):
        availability = "partial"
    else:
        availability = "available"
    return StagePhase(
        availability=availability,
        provenance="body_pose",
        confidence=max(0.0, min(1.0, float(candidate.score))),
        interval=candidate.interval,
        keyframe_seconds=float(candidate.keyframe_seconds),
        temporal_uncertainty_seconds=float(
            candidate.temporal_uncertainty_seconds
        ),
        evidence=tuple(candidate.evidence),
        limitations=tuple(candidate.limitations),
    )


def _merge_unique(first: Sequence[str], second: Sequence[str]) -> tuple[str, ...]:
    """Merge token sequences preserving order without duplicates."""
    merged: list[str] = list(first)
    for token in second:
        if token not in merged:
            merged.append(token)
    return tuple(merged)


def _emit_contact(
    candidate: StageCandidate,
    grid: PhaseFeatureGrid,
    config: PhaseSolverConfig,
) -> tuple[StagePhase, tuple[str, ...]]:
    """Emit the audio-anchored contact; never delete it for body reasons."""
    compatible, reason, body_evidence, body_limitations = _contact_body_support(
        grid, candidate, config
    )
    base = max(0.0, min(1.0, float(candidate.score)))
    if compatible:
        provenance = "body_pose_audio"
        confidence = min(1.0, base + float(config.contact_body_confidence_bonus))
        evidence = _merge_unique(candidate.evidence, body_evidence)
        limitations = _merge_unique(candidate.limitations, body_limitations)
        anomalies: tuple[str, ...] = ()
    else:
        provenance = "audio_transient"
        confidence = max(
            0.0, base - float(config.contact_body_confidence_penalty)
        )
        evidence = tuple(candidate.evidence)
        limitations = _merge_unique(candidate.limitations, body_limitations)
        if reason == "occluded":
            anomalies = ("contact_body_support_occluded",)
        elif reason == "missing":
            anomalies = ("contact_body_support_missing",)
        else:
            anomalies = ("contact_body_support_incompatible",)
    stage = StagePhase(
        availability="available",
        provenance=provenance,
        confidence=confidence,
        interval=candidate.interval,
        keyframe_seconds=float(candidate.keyframe_seconds),
        temporal_uncertainty_seconds=float(
            candidate.temporal_uncertainty_seconds
        ),
        evidence=evidence,
        limitations=limitations,
    )
    return stage, anomalies


def _emit_skipped(
    stage: str, had_candidates: bool
) -> tuple[StagePhase, tuple[str, ...]]:
    if stage == _CONTACT_STAGE:
        provenance = "audio_transient"
    else:
        provenance = "body_pose"
    if had_candidates:
        limitations = ("skipped_for_global_consistency",)
        anomalies = (f"skipped_{stage}_for_consistency",)
    else:
        limitations = ("no_candidate_available",)
        anomalies = (f"no_candidate_{stage}",)
    return (
        StagePhase(
            availability="unavailable",
            provenance=provenance,
            confidence=0.0,
            interval=None,
            keyframe_seconds=None,
            temporal_uncertainty_seconds=None,
            evidence=(),
            limitations=limitations,
        ),
        anomalies,
    )


# --- Public entry points -------------------------------------------------------


def solve_with_diagnostics(
    attempt: Attempt,
    grid: PhaseFeatureGrid,
    evidence: PhaseEvidence,
    config: PhaseSolverConfig | None = None,
) -> PhaseSolverResult:
    """Solve one attempt and return the versioned solver result record."""
    cfg = config if config is not None else PhaseSolverConfig()
    unpadded = _validate_inputs(attempt, grid, evidence, cfg)
    per_stage = [list(evidence.for_stage(stage)) for stage in _CANONICAL_STAGES]
    choice, total = _run_dynamic_program(per_stage, cfg)

    stages: dict[str, StagePhase] = {}
    anomalies: list[str] = []
    for index, stage in enumerate(_CANONICAL_STAGES):
        pick = choice[index]
        if pick is None:
            emitted, stage_anomalies = _emit_skipped(
                stage, had_candidates=bool(per_stage[index])
            )
            stages[stage] = emitted
            anomalies.extend(stage_anomalies)
        elif stage == _CONTACT_STAGE:
            emitted, stage_anomalies = _emit_contact(
                per_stage[index][pick], grid, cfg
            )
            stages[stage] = emitted
            anomalies.extend(stage_anomalies)
        else:
            stages[stage] = _emit_selected(
                stage, per_stage[index][pick], cfg
            )
    status = _structural_status(
        [stages[key].availability for key in _CANONICAL_STAGES]
    )
    attempt_phase = AttemptPhase(
        attempt_id=attempt.attempt_id,
        attempt_range=MediaRange(
            start_seconds=unpadded.start_seconds,
            end_seconds=unpadded.end_seconds,
        ),
        method_version=PHASE_SOLVER_METHOD_VERSION,
        config_id=cfg.config_id,
        stages=stages,
        structural_status=status,
        anomalies=tuple(anomalies),
    )
    selected = {
        stage: (
            float(per_stage[index][choice[index]].keyframe_seconds)
            if choice[index] is not None
            else None
        )
        for index, stage in enumerate(_CANONICAL_STAGES)
    }
    return PhaseSolverResult(
        attempt_phase=attempt_phase,
        total_score=float(total),
        selected_keyframes=selected,
        method_version=PHASE_SOLVER_METHOD_VERSION,
        config_id=cfg.config_id,
    )


def solve_attempt_phase(
    attempt: Attempt,
    grid: PhaseFeatureGrid,
    evidence: PhaseEvidence,
    config: PhaseSolverConfig | None = None,
) -> AttemptPhase:
    """Select the globally coherent phase sequence for one attempt.

    Pure and deterministic: the input :class:`Attempt` is only read (its
    un-padded ``detected_range`` is the temporal authority) and is never
    modified, relabeled, suppressed, or written.
    """
    return solve_with_diagnostics(attempt, grid, evidence, config).attempt_phase
