"""Candidate serve-range state machine (M3.2).

Converts an ordered :class:`FeatureFrame` sequence (M3.1) into unpadded
serve candidate ranges using explicit hysteresis and duration rules.
Pure, deterministic, std-lib only.

Signal model (no fabricated evidence):

- A frame votes *enter-active* when it carries a person, its smoothed
  ``motion_evidence`` is not ``None`` and ``>= active_threshold``, and
  its ``visible_fraction >= min_visible_fraction``.
- While a segment is open, a frame votes *stay-active* under the lower
  ``inactive_threshold`` (``motion_evidence >= inactive_threshold`` with
  the same person/visibility gates). The band
  ``inactive <= motion < active`` therefore keeps a segment alive but
  never starts one: explicit hysteresis.
- Any frame with ``has_person is False``, ``motion_evidence is None``,
  or visibility below the floor votes inactive. Missing/``None``
  evidence never fabricates a candidate.

State machine (``CandidateDetector``):

- ``reset()`` clears all buffered state.
- ``feed(frame)`` consumes one :class:`FeatureFrame` in strictly
  increasing ``time_seconds`` order (violations raise
  :class:`RangeError`). Opening, extending, or closing a raw segment
  follows the enter/stay votes above.
- ``finish()`` closes any open segment, merges raw segments separated
  by ``<= merge_gap_seconds``, enforces duration bounds, applies the
  overhead gate, and returns a tuple of :class:`CandidateRange`
  sorted by start. It then resets so the detector is reusable; calling
  ``finish()`` twice without new feeds yields ``()`` the second time.

Duration and merge rules:

- Merge gaps first: consecutive raw segments with
  ``next.start - prev.end <= merge_gap_seconds`` become one segment
  spanning the gap (gap is included, not padded beyond evidence).
- Minimum duration: merged segments with
  ``end - start < min_duration_seconds`` are discarded (short blips,
  aborted tosses that never sustain motion).
- Maximum duration: merged segments with ``end - start`` above
  ``max_duration_seconds`` are split into ``ceil(duration / max)``
  equal-length pieces covering ``[start, end)`` exactly. Split pieces
  shorter than the minimum are dropped (only reachable with
  pathological ``min ~= max`` configurations).
- A range's ``start`` is the first active frame time and ``end`` is the
  last active (or merged) frame time, half-open ``[start, end)`` with
  ``start < end``. Single-frame blips have zero duration and are
  therefore always dropped by the minimum.

Truncation: a sequence that starts or ends mid-motion is handled
honestly — a leading active frame opens a segment immediately and a
trailing open segment is flushed by ``finish()``. No preceding or
trailing rest is required or invented.

Overhead gate (isolated-toss / unrelated-motion rejection, only where
feature evidence permits): when ``require_overhead`` is true, a merged
segment is discarded if at least one of its frames carries
non-``None`` ``overhead_evidence`` yet none carries ``1.0``. Segments
whose overhead is entirely ``None`` (missing joints) are kept because
the evidence does not permit rejection. Abbreviated serves still pass:
any single ``1.0`` frame in the segment satisfies the gate regardless
of duration.

Determinism: iteration order is input order; no randomness, wall clock,
global state, or I/O. All thresholds and bounds live in the versioned
:class:`RangeConfig` with a JSON codec.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from serve_review.detection.features import FeatureFrame

__all__ = [
    "RANGE_SCHEMA_VERSION",
    "RangeError",
    "RangeConfig",
    "CandidateRange",
    "CandidateDetector",
    "find_candidates",
]

#: Version of the range configuration and candidate schemas.
RANGE_SCHEMA_VERSION = 1


class RangeError(ValueError):
    """Raised when range input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RangeError(f"range_config: {name!r} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise RangeError(
            f"range_config: {name!r} must be finite, got {value!r}."
        )
    return number


def _unit_threshold(name: str, value: Any, *, low_inclusive: bool) -> float:
    number = _finite_number(name, value)
    low_ok = number >= 0.0 if low_inclusive else number > 0.0
    if not low_ok or number > 1.0:
        bound = "[0, 1]" if low_inclusive else "(0, 1]"
        raise RangeError(
            f"range_config: {name!r} must lie in {bound}, got {value!r}."
        )
    return number


@dataclass(frozen=True, slots=True)
class RangeConfig:
    """Versioned thresholds and bounds for candidate extraction.

    - ``active_threshold``: ``motion_evidence`` level that opens a
      segment (in ``(0, 1]``).
    - ``inactive_threshold``: ``motion_evidence`` level that keeps an
      open segment alive (in ``[0, 1]``, ``<= active_threshold``).
    - ``min_duration_seconds`` / ``max_duration_seconds``: accepted
      unpadded duration bounds (``0 < min <= max``).
    - ``merge_gap_seconds``: maximum ``next.start - prev.end`` merged
      into one candidate (``>= 0``).
    - ``min_visible_fraction``: minimum ``visible_fraction`` for an
      active vote (in ``[0, 1]``).
    - ``require_overhead``: apply the overhead gate described in the
      module docstring.
    """

    active_threshold: float = 0.4
    inactive_threshold: float = 0.2
    min_duration_seconds: float = 0.6
    max_duration_seconds: float = 8.0
    merge_gap_seconds: float = 0.5
    min_visible_fraction: float = 0.3
    require_overhead: bool = True
    schema_version: int = RANGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise RangeError(
                "range_config: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != RANGE_SCHEMA_VERSION:
            raise RangeError(
                f"range_config: unsupported schema_version {self.schema_version!r}; "
                f"expected {RANGE_SCHEMA_VERSION}."
            )
        active = _unit_threshold(
            "'active_threshold'", self.active_threshold, low_inclusive=False
        )
        inactive = _unit_threshold(
            "'inactive_threshold'", self.inactive_threshold, low_inclusive=True
        )
        if inactive > active:
            raise RangeError(
                "range_config: 'inactive_threshold' "
                f"({inactive!r}) must be <= 'active_threshold' ({active!r}) "
                "for hysteresis."
            )
        min_dur = _finite_number("'min_duration_seconds'", self.min_duration_seconds)
        if min_dur <= 0.0:
            raise RangeError(
                "range_config: 'min_duration_seconds' must be > 0, "
                f"got {self.min_duration_seconds!r}."
            )
        max_dur = _finite_number("'max_duration_seconds'", self.max_duration_seconds)
        if max_dur < min_dur:
            raise RangeError(
                "range_config: 'max_duration_seconds' "
                f"({self.max_duration_seconds!r}) must be >= "
                f"'min_duration_seconds' ({self.min_duration_seconds!r})."
            )
        gap = _finite_number("'merge_gap_seconds'", self.merge_gap_seconds)
        if gap < 0.0:
            raise RangeError(
                "range_config: 'merge_gap_seconds' must be >= 0, "
                f"got {self.merge_gap_seconds!r}."
            )
        visible = _finite_number(
            "'min_visible_fraction'", self.min_visible_fraction
        )
        if visible < 0.0 or visible > 1.0:
            raise RangeError(
                "range_config: 'min_visible_fraction' must lie in [0, 1], "
                f"got {self.min_visible_fraction!r}."
            )
        if not isinstance(self.require_overhead, bool):
            raise RangeError(
                "range_config: 'require_overhead' must be a bool, "
                f"got {self.require_overhead!r}."
            )
        object.__setattr__(self, "active_threshold", active)
        object.__setattr__(self, "inactive_threshold", inactive)
        object.__setattr__(self, "min_duration_seconds", min_dur)
        object.__setattr__(self, "max_duration_seconds", max_dur)
        object.__setattr__(self, "merge_gap_seconds", gap)
        object.__setattr__(self, "min_visible_fraction", visible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_threshold": self.active_threshold,
            "inactive_threshold": self.inactive_threshold,
            "merge_gap_seconds": self.merge_gap_seconds,
            "min_duration_seconds": self.min_duration_seconds,
            "min_visible_fraction": self.min_visible_fraction,
            "max_duration_seconds": self.max_duration_seconds,
            "require_overhead": self.require_overhead,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> RangeConfig:
        if not isinstance(values, dict):
            raise RangeError(
                "range_config: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "active_threshold",
            "inactive_threshold",
            "merge_gap_seconds",
            "min_duration_seconds",
            "min_visible_fraction",
            "max_duration_seconds",
            "require_overhead",
            "schema_version",
        }
        missing = sorted(known - set(values))
        if missing:
            raise RangeError(
                f"range_config: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise RangeError(f"range_config: unknown keys {unknown!r}.")
        return cls(
            active_threshold=values["active_threshold"],
            inactive_threshold=values["inactive_threshold"],
            min_duration_seconds=values["min_duration_seconds"],
            max_duration_seconds=values["max_duration_seconds"],
            merge_gap_seconds=values["merge_gap_seconds"],
            min_visible_fraction=values["min_visible_fraction"],
            require_overhead=values["require_overhead"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> RangeConfig:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RangeError("range_config: invalid UTF-8 JSON payload.") from exc
        if not isinstance(data, str):
            raise RangeError(
                "range_config: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RangeError(f"range_config: invalid JSON: {exc}.") from exc
        return cls.from_dict(decoded)


def _check_time(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RangeError(
            f"candidate_range: {name!r} must be a number, got {value!r}."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise RangeError(
            f"candidate_range: {name!r} must be finite and >= 0, "
            f"got {value!r}."
        )
    return number


@dataclass(frozen=True, slots=True)
class CandidateRange:
    """One unpadded serve candidate, half-open ``[start, end)``."""

    start_seconds: float = 0.0
    end_seconds: float = 0.0
    schema_version: int = RANGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise RangeError(
                "candidate_range: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != RANGE_SCHEMA_VERSION:
            raise RangeError(
                f"candidate_range: unsupported schema_version {self.schema_version!r}; "
                f"expected {RANGE_SCHEMA_VERSION}."
            )
        start = _check_time("'start_seconds'", self.start_seconds)
        end = _check_time("'end_seconds'", self.end_seconds)
        if not end > start:
            raise RangeError(
                "candidate_range: 'end_seconds' "
                f"({self.end_seconds!r}) must be greater than 'start_seconds' "
                f"({self.start_seconds!r}) for half-open [start, end)."
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)

    @property
    def duration_seconds(self) -> float:
        """Return ``end_seconds - start_seconds``."""
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_seconds": self.end_seconds,
            "schema_version": self.schema_version,
            "start_seconds": self.start_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> CandidateRange:
        if not isinstance(values, dict):
            raise RangeError(
                "candidate_range: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {"end_seconds", "schema_version", "start_seconds"}
        missing = sorted(known - set(values))
        if missing:
            raise RangeError(
                f"candidate_range: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise RangeError(f"candidate_range: unknown keys {unknown!r}.")
        return cls(
            start_seconds=values["start_seconds"],
            end_seconds=values["end_seconds"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> CandidateRange:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RangeError(
                    "candidate_range: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise RangeError(
                "candidate_range: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RangeError(f"candidate_range: invalid JSON: {exc}.") from exc
        return cls.from_dict(decoded)


class _RawSegment:
    """Mutable raw active run with overhead bookkeeping."""

    __slots__ = ("start", "end", "seen_overhead", "seen_evidence")

    def __init__(self, start: float) -> None:
        self.start = start
        self.end = start
        self.seen_overhead = False
        self.seen_evidence = False


class CandidateDetector:
    """Incremental hysteresis state machine over :class:`FeatureFrame`.

    Construct with an optional :class:`RangeConfig` (defaults to
    ``RangeConfig()``). Use ``feed`` per frame in strictly increasing
    time order, then ``finish`` to flush and collect candidates.
    """

    def __init__(self, config: RangeConfig | None = None) -> None:
        if config is None:
            config = RangeConfig()
        if not isinstance(config, RangeConfig):
            raise RangeError(
                "detector: 'config' must be a RangeConfig, "
                f"got {type(config).__name__}."
            )
        self._config = config
        self._raw: list[_RawSegment] = []
        self._open: _RawSegment | None = None
        self._last_time: float | None = None

    @property
    def config(self) -> RangeConfig:
        """Return the detector configuration."""
        return self._config

    def reset(self) -> None:
        """Discard all buffered state."""
        self._raw = []
        self._open = None
        self._last_time = None

    def _vote_enter(self, frame: FeatureFrame) -> bool:
        if not frame.has_person:
            return False
        if frame.motion_evidence is None:
            return False
        if frame.visible_fraction < self._config.min_visible_fraction:
            return False
        return frame.motion_evidence >= self._config.active_threshold

    def _vote_stay(self, frame: FeatureFrame) -> bool:
        if not frame.has_person:
            return False
        if frame.motion_evidence is None:
            return False
        if frame.visible_fraction < self._config.min_visible_fraction:
            return False
        return frame.motion_evidence >= self._config.inactive_threshold

    @staticmethod
    def _note_overhead(segment: _RawSegment, frame: FeatureFrame) -> None:
        evidence = frame.overhead_evidence
        if evidence is None:
            return
        segment.seen_evidence = True
        if evidence == 1.0:
            segment.seen_overhead = True

    def feed(self, frame: FeatureFrame) -> None:
        """Consume one feature frame."""
        if not isinstance(frame, FeatureFrame):
            raise RangeError(
                "detector: every frame must be a FeatureFrame, "
                f"got {type(frame).__name__}."
            )
        moment = frame.time_seconds
        if self._last_time is not None and not moment > self._last_time:
            raise RangeError(
                "detector: frame times must be strictly increasing, "
                f"got {self._last_time!r} followed by {moment!r}."
            )
        self._last_time = moment
        if self._open is None:
            if self._vote_enter(frame):
                segment = _RawSegment(moment)
                self._note_overhead(segment, frame)
                self._open = segment
        else:
            if self._vote_stay(frame):
                self._open.end = moment
                self._note_overhead(segment=self._open, frame=frame)
            else:
                # Close the raw run; short gaps are re-joined by the
                # merge pass in finish().
                self._raw.append(self._open)
                self._open = None

    def feed_all(self, frames: Sequence[FeatureFrame]) -> None:
        """Consume an ordered sequence of feature frames."""
        for frame in frames:
            self.feed(frame)

    def finish(self) -> tuple[CandidateRange, ...]:
        """Flush any open segment and return merged, gated candidates."""
        if self._open is not None:
            self._raw.append(self._open)
            self._open = None
        raw = self._raw
        self._raw = []
        self._last_time = None
        if not raw:
            return ()
        # Merge gaps: raw runs are already in time order.
        merged: list[_RawSegment] = []
        for segment in raw:
            if segment.end < segment.start:
                continue  # Unreachable; guards against time disorder.
            if merged and segment.start - merged[-1].end <= self._config.merge_gap_seconds:
                tail = merged[-1]
                if segment.end > tail.end:
                    tail.end = segment.end
                tail.seen_overhead = tail.seen_overhead or segment.seen_overhead
                tail.seen_evidence = tail.seen_evidence or segment.seen_evidence
            else:
                carry = _RawSegment(segment.start)
                carry.end = segment.end
                carry.seen_overhead = segment.seen_overhead
                carry.seen_evidence = segment.seen_evidence
                merged.append(carry)
        results: list[CandidateRange] = []
        for segment in merged:
            duration = segment.end - segment.start
            if duration < self._config.min_duration_seconds:
                continue
            if (
                self._config.require_overhead
                and segment.seen_evidence
                and not segment.seen_overhead
            ):
                continue
            if duration > self._config.max_duration_seconds:
                pieces = math.ceil(
                    duration / self._config.max_duration_seconds
                )
                if pieces < 1:
                    pieces = 1
                step = duration / pieces
                for index in range(pieces):
                    piece_start = segment.start + index * step
                    piece_end = (
                        segment.end
                        if index == pieces - 1
                        else segment.start + (index + 1) * step
                    )
                    if piece_end - piece_start < self._config.min_duration_seconds:
                        continue
                    results.append(
                        CandidateRange(
                            start_seconds=piece_start, end_seconds=piece_end
                        )
                    )
            else:
                results.append(
                    CandidateRange(
                        start_seconds=segment.start, end_seconds=segment.end
                    )
                )
        results.sort(key=lambda item: (item.start_seconds, item.end_seconds))
        return tuple(results)


def find_candidates(
    frames: Sequence[FeatureFrame],
    config: RangeConfig | None = None,
) -> tuple[CandidateRange, ...]:
    """Convert an ordered feature sequence into unpadded candidates.

    ``frames`` must carry strictly increasing ``time_seconds``; an empty
    sequence yields ``()``. ``config`` defaults to ``RangeConfig()``.
    """
    cfg = config if config is not None else RangeConfig()
    if not isinstance(cfg, RangeConfig):
        raise RangeError(
            "find_candidates: 'config' must be a RangeConfig, "
            f"got {type(cfg).__name__}."
        )
    items = list(frames)
    for entry in items:
        if not isinstance(entry, FeatureFrame):
            raise RangeError(
                "find_candidates: every frame must be a FeatureFrame, "
                f"got {type(entry).__name__}."
            )
    for earlier, later in zip(items, items[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise RangeError(
                "find_candidates: frame times must be strictly increasing, "
                f"got {earlier.time_seconds!r} followed by "
                f"{later.time_seconds!r}."
            )
    if not items:
        return ()
    detector = CandidateDetector(cfg)
    detector.feed_all(items)
    return detector.finish()
