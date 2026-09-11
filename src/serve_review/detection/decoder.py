"""Macro state decoder: frame-aligned audio-visual features to serve ranges (M3.x).

Turns ordered pose :class:`FeatureFrame` values (M3.1, v2 proximal
geometry) plus band-limited audio :class:`AudioEnergy` samples
(``A_t`` series keyed by source ``time_seconds``) into classified serve
ranges. Pure, deterministic, std-lib only.

Input pairing (``F_t`` contract, expressed without touching the
features/audio modules):

- Both sequences must carry strictly increasing ``time_seconds``;
  violations raise :class:`DecoderError`.
- :func:`pair_frames` left-joins audio onto pose frames by exact source
  time: a pose frame whose ``time_seconds`` has no audio sample pairs
  with ``audio_energy=None`` (unknown, never fabricated). Extra audio
  times not on a pose frame are retained for the transient-window scan
  but never invent a pose state.
- Missing pose evidence (``has_person is False``, ``visible_fraction``
  below the floor, or ``None`` geometry/motion fields) propagates as
  unknown: it can bridge (hysteresis) or collapse a hypothesis but never
  fabricates a transition. Missing audio likewise never validates.

Macro states over ``F_t``:

- ``idle``: at rest, waiting for upper-arm activity.
- ``preparation``: entered from ``idle`` on proximal upper-arm activity
  (torso displacement or arm elevation through high-confidence proximal
  geometry). Wrist speed is never consulted.
- ``acceleration``: entered from ``preparation`` when the arm
  accelerates into the overhead quadrant (elbow-speed surge plus
  overhead position with high-confidence elbow geometry present).
- ``follow_through``: entered from ``acceleration`` only when a
  validating audio transient lies within ``+-audio_window_seconds``
  (default 0.4 s, versioned) of the latest overhead-acceleration
  trigger time (the anchor follows the drive phase), where
  validation is scale-invariant: the peak ``A_t`` energy inside the
  window must reach ``audio_transient_ratio`` (default 5.0) times the
  session-median baseline and clear the small ``audio_transient_floor``
  against near-silence. Without validation the hypothesis stays in
  ``acceleration`` until stillness routes it to ``shadow``.

Exits:

- ``follow_through -> idle`` when stillness returns and the wrist drops
  below the shoulder (``rest_evidence`` high and ``overhead == 0.0``),
  emitting a valid-serve :class:`CandidateRange` ``[start, end)`` whose
  boundaries are exactly the preparation-entry and stillness frame
  times (unpadded, planner-compatible, no padding/interpolation).
- ``preparation -> idle`` on stillness without acceleration emits an
  ``aborted`` shadow record (toss without acceleration).
- ``acceleration`` reaching stillness without validation, exceeding the
  dropout budget, or trailing off at end of input emits a ``shadow``
  record (silent overhead / audio outside window). ``follow_through``
  that never observes its stillness exit (truncated or long dropout)
  likewise closes as ``shadow`` so an incomplete hypothesis never
  exports as a serve.
- Dropout resilience: while in an active state, up to
  ``dropout_hysteresis_frames`` (3-5, versioned) consecutive dropout
  frames (``has_person is False``, ``visible_fraction < 0.4``, or fully
  unknown geometry) are bridged without collapsing to ``idle``; the
  next good frame resumes the same state. Longer gaps collapse to
  ``idle`` with an ``aborted``/``shadow`` record as above.

Outputs (:class:`DecodeResult`):

- ``ranges``: valid-serve unpadded :class:`CandidateRange` values sorted
  by ``(start, end)``, directly plannable by
  :mod:`serve_review.detection.plan` (export path).
- ``shadows``: separate :class:`ShadowRecord` values (``aborted`` or
  ``shadow``) that suppress export while remaining inspectable.
- Empty input yields empty outputs. No randomness, wall clock, global
  state, or I/O; JSON codecs sort keys.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

from serve_review.detection.features import FeatureFrame
from serve_review.detection.ranges import CandidateRange
from serve_review.media.audio import AudioEnergy

__all__ = [
    "DECODER_SCHEMA_VERSION",
    "VISIBILITY_FLOOR",
    "REASON_ABORTED",
    "REASON_SHADOW",
    "DecoderError",
    "DecoderConfig",
    "ShadowRecord",
    "PairedFrame",
    "DecodeResult",
    "pair_frames",
    "decode_sequence",
]

#: Version of the decoder configuration and record schemas.
#:
#: Version 2 replaces the absolute ``audio_transient_threshold`` (0.3)
#: with a scale-invariant relative ratio (``audio_transient_ratio``,
#: default 5.0, peak vs session-median baseline) plus a small absolute
#: ``audio_transient_floor`` against near-silence. Version-1 payloads
#: are rejected (missing keys / version mismatch) rather than silently
#: reinterpreted.
DECODER_SCHEMA_VERSION = 2

#: Visibility floor below which a pose frame counts as dropout. Mirrors
#: ``FeatureConfig.visibility_floor`` (default 0.4) without importing it.
VISIBILITY_FLOOR = 0.4

#: Shadow-record reason for preparation that never accelerates.
REASON_ABORTED = "aborted"

#: Shadow-record reason for acceleration without a validating transient
#: (silent overhead, audio outside window, truncated follow-through).
REASON_SHADOW = "shadow"

_REASONS = (REASON_ABORTED, REASON_SHADOW)


class DecoderError(ValueError):
    """Raised when decoder input or configuration is invalid."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecoderError(f"decoder_config: {name!r} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise DecoderError(
            f"decoder_config: {name!r} must be finite and > 0, got {value!r}."
        )
    return number


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecoderError(f"decoder_config: {name!r} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise DecoderError(
            f"decoder_config: {name!r} must be finite and >= 0, got {value!r}."
        )
    return number


def _unit_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecoderError(f"decoder_config: {name!r} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise DecoderError(
            f"decoder_config: {name!r} must lie in [0, 1], got {value!r}."
        )
    return number


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Versioned thresholds for the macro state decoder.

    - ``torso_displacement_threshold``: trailing torso displacement
      (normalized units) that votes for preparation entry.
    - ``acceleration_elbow_speed_threshold``: scale-normalized elbow
      speed (units/second) that, together with the overhead quadrant
      and present elbow geometry, votes for acceleration entry. Wrist
      speed is never consulted.
    - ``rest_evidence_threshold``: ``rest_evidence`` level that, with
      ``overhead == 0.0``, votes for the stillness exit.
    - ``audio_transient_ratio``: scale-invariant peak factor -- the
      maximum ``A_t`` energy inside the audio window must reach this
      multiple (default 5.0) of the session-median baseline to count
      as a validating impact transient.
    - ``audio_transient_floor``: small absolute energy floor (default
      0.01) the window peak must also clear, so near-silence sessions
      never validate on a ratio against a ~0 baseline.
    - ``audio_window_seconds``: half-width of the ``+-`` source-time
      window around the latest overhead-acceleration trigger time
      scanned for the transient.
    - ``dropout_hysteresis_frames``: consecutive dropout frames bridged
      inside an active state (3-5 inclusive).
    """

    torso_displacement_threshold: float = 0.2
    acceleration_elbow_speed_threshold: float = 1.5
    rest_evidence_threshold: float = 0.6
    audio_transient_ratio: float = 5.0
    audio_transient_floor: float = 0.01
    audio_window_seconds: float = 0.4
    dropout_hysteresis_frames: int = 4
    schema_version: int = DECODER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise DecoderError(
                "decoder_config: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != DECODER_SCHEMA_VERSION:
            raise DecoderError(
                f"decoder_config: unsupported schema_version {self.schema_version!r}; "
                f"expected {DECODER_SCHEMA_VERSION}."
            )
        object.__setattr__(
            self,
            "torso_displacement_threshold",
            _finite_positive(
                "'torso_displacement_threshold'",
                self.torso_displacement_threshold,
            ),
        )
        object.__setattr__(
            self,
            "acceleration_elbow_speed_threshold",
            _finite_positive(
                "'acceleration_elbow_speed_threshold'",
                self.acceleration_elbow_speed_threshold,
            ),
        )
        object.__setattr__(
            self,
            "rest_evidence_threshold",
            _unit_threshold(
                "'rest_evidence_threshold'", self.rest_evidence_threshold
            ),
        )
        object.__setattr__(
            self,
            "audio_transient_ratio",
            _finite_positive(
                "'audio_transient_ratio'", self.audio_transient_ratio
            ),
        )
        object.__setattr__(
            self,
            "audio_transient_floor",
            _finite_nonnegative(
                "'audio_transient_floor'", self.audio_transient_floor
            ),
        )
        object.__setattr__(
            self,
            "audio_window_seconds",
            _finite_positive("'audio_window_seconds'", self.audio_window_seconds),
        )
        if (
            not _is_int(self.dropout_hysteresis_frames)
            or self.dropout_hysteresis_frames < 3
            or self.dropout_hysteresis_frames > 5
        ):
            raise DecoderError(
                "decoder_config: 'dropout_hysteresis_frames' must be an "
                f"integer in [3, 5], got {self.dropout_hysteresis_frames!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceleration_elbow_speed_threshold": self.acceleration_elbow_speed_threshold,
            "audio_transient_floor": self.audio_transient_floor,
            "audio_transient_ratio": self.audio_transient_ratio,
            "audio_window_seconds": self.audio_window_seconds,
            "dropout_hysteresis_frames": self.dropout_hysteresis_frames,
            "rest_evidence_threshold": self.rest_evidence_threshold,
            "schema_version": self.schema_version,
            "torso_displacement_threshold": self.torso_displacement_threshold,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DecoderConfig:
        if not isinstance(values, dict):
            raise DecoderError(
                "decoder_config: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {
            "acceleration_elbow_speed_threshold",
            "audio_transient_floor",
            "audio_transient_ratio",
            "audio_window_seconds",
            "dropout_hysteresis_frames",
            "rest_evidence_threshold",
            "schema_version",
            "torso_displacement_threshold",
        }
        missing = sorted(known - set(values))
        if missing:
            raise DecoderError(
                f"decoder_config: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise DecoderError(f"decoder_config: unknown keys {unknown!r}.")
        return cls(
            torso_displacement_threshold=values["torso_displacement_threshold"],
            acceleration_elbow_speed_threshold=values[
                "acceleration_elbow_speed_threshold"
            ],
            rest_evidence_threshold=values["rest_evidence_threshold"],
            audio_transient_ratio=values["audio_transient_ratio"],
            audio_transient_floor=values["audio_transient_floor"],
            audio_window_seconds=values["audio_window_seconds"],
            dropout_hysteresis_frames=values["dropout_hysteresis_frames"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> DecoderConfig:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DecoderError(
                    "decoder_config: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise DecoderError(
                "decoder_config: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise DecoderError(f"decoder_config: invalid JSON: {exc}.") from exc
        return cls.from_dict(decoded)


def _check_time(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecoderError(f"{name}: must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise DecoderError(f"{name}: must be finite and >= 0, got {value!r}.")
    return number


@dataclass(frozen=True, slots=True)
class ShadowRecord:
    """One inspectable non-exported hypothesis, half-open ``[start, end)``.

    ``reason`` is ``"aborted"`` (preparation never accelerated: toss
    without acceleration or dropout collapse) or ``"shadow"``
    (overhead acceleration without a validating transient inside the
    audio window, or truncated follow-through). Shadow records never
    enter the export planner.
    """

    start_seconds: float = 0.0
    end_seconds: float = 0.0
    reason: str = REASON_SHADOW
    schema_version: int = DECODER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise DecoderError(
                "shadow_record: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != DECODER_SCHEMA_VERSION:
            raise DecoderError(
                f"shadow_record: unsupported schema_version {self.schema_version!r}; "
                f"expected {DECODER_SCHEMA_VERSION}."
            )
        start = _check_time("'start_seconds'", self.start_seconds)
        end = _check_time("'end_seconds'", self.end_seconds)
        if not end > start:
            raise DecoderError(
                "shadow_record: 'end_seconds' "
                f"({self.end_seconds!r}) must be greater than 'start_seconds' "
                f"({self.start_seconds!r}) for half-open [start, end)."
            )
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        if self.reason not in _REASONS:
            raise DecoderError(
                f"shadow_record: 'reason' must be one of {list(_REASONS)!r}, "
                f"got {self.reason!r}."
            )

    @property
    def duration_seconds(self) -> float:
        """Return ``end_seconds - start_seconds``."""
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_seconds": self.end_seconds,
            "reason": self.reason,
            "schema_version": self.schema_version,
            "start_seconds": self.start_seconds,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ShadowRecord:
        if not isinstance(values, dict):
            raise DecoderError(
                "shadow_record: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {"end_seconds", "reason", "schema_version", "start_seconds"}
        missing = sorted(known - set(values))
        if missing:
            raise DecoderError(
                f"shadow_record: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise DecoderError(f"shadow_record: unknown keys {unknown!r}.")
        return cls(
            start_seconds=values["start_seconds"],
            end_seconds=values["end_seconds"],
            reason=values["reason"],
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> ShadowRecord:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DecoderError(
                    "shadow_record: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise DecoderError(
                "shadow_record: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise DecoderError(f"shadow_record: invalid JSON: {exc}.") from exc
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class PairedFrame:
    """One frame-aligned ``F_t`` pair: pose feature plus audio (or None)."""

    time_seconds: float
    feature: FeatureFrame
    audio_energy: float | None

    def __post_init__(self) -> None:
        moment = _check_time("'time_seconds'", self.time_seconds)
        object.__setattr__(self, "time_seconds", moment)
        if not isinstance(self.feature, FeatureFrame):
            raise DecoderError(
                "paired_frame: 'feature' must be a FeatureFrame, "
                f"got {type(self.feature).__name__}."
            )
        if self.audio_energy is not None:
            value = self.audio_energy
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise DecoderError(
                    "paired_frame: 'audio_energy' must be finite and >= 0 "
                    f"or None, got {value!r}."
                )
            object.__setattr__(self, "audio_energy", float(value))


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Decoder output: exportable ranges plus inspectable shadows."""

    ranges: tuple[CandidateRange, ...] = ()
    shadows: tuple[ShadowRecord, ...] = ()
    schema_version: int = DECODER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _is_int(self.schema_version):
            raise DecoderError(
                "decode_result: 'schema_version' must be an integer, "
                f"got {self.schema_version!r}."
            )
        if self.schema_version != DECODER_SCHEMA_VERSION:
            raise DecoderError(
                f"decode_result: unsupported schema_version {self.schema_version!r}; "
                f"expected {DECODER_SCHEMA_VERSION}."
            )
        raw_ranges = self.ranges
        if isinstance(raw_ranges, CandidateRange) or not isinstance(
            raw_ranges, (list, tuple)
        ):
            raise DecoderError(
                "decode_result: 'ranges' must be a list or tuple of "
                f"CandidateRange, got {type(raw_ranges).__name__}."
            )
        normalized_ranges = tuple(raw_ranges)
        for entry in normalized_ranges:
            if not isinstance(entry, CandidateRange):
                raise DecoderError(
                    "decode_result: every range must be a CandidateRange, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "ranges", normalized_ranges)
        raw_shadows = self.shadows
        if isinstance(raw_shadows, ShadowRecord) or not isinstance(
            raw_shadows, (list, tuple)
        ):
            raise DecoderError(
                "decode_result: 'shadows' must be a list or tuple of "
                f"ShadowRecord, got {type(raw_shadows).__name__}."
            )
        normalized_shadows = tuple(raw_shadows)
        for entry in normalized_shadows:
            if not isinstance(entry, ShadowRecord):
                raise DecoderError(
                    "decode_result: every shadow must be a ShadowRecord, "
                    f"got {type(entry).__name__}."
                )
        object.__setattr__(self, "shadows", normalized_shadows)
        for first, second in zip(normalized_ranges, normalized_ranges[1:]):
            if (second.start_seconds, second.end_seconds) < (
                first.start_seconds,
                first.end_seconds,
            ):
                raise DecoderError(
                    "decode_result: 'ranges' must be sorted by (start, end)."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranges": [item.to_dict() for item in self.ranges],
            "schema_version": self.schema_version,
            "shadows": [item.to_dict() for item in self.shadows],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DecodeResult:
        if not isinstance(values, dict):
            raise DecoderError(
                "decode_result: mapping is required, "
                f"got {type(values).__name__}."
            )
        known = {"ranges", "schema_version", "shadows"}
        missing = sorted(known - set(values))
        if missing:
            raise DecoderError(
                f"decode_result: missing required keys {missing!r}."
            )
        unknown = sorted(set(values) - known)
        if unknown:
            raise DecoderError(f"decode_result: unknown keys {unknown!r}.")
        if not _is_int(values["schema_version"]):
            raise DecoderError(
                "decode_result: 'schema_version' must be an integer, "
                f"got {values['schema_version']!r}."
            )
        if values["schema_version"] != DECODER_SCHEMA_VERSION:
            raise DecoderError(
                "decode_result: unsupported schema_version "
                f"{values['schema_version']!r}; expected {DECODER_SCHEMA_VERSION}."
            )
        raw_ranges = values["ranges"]
        raw_shadows = values["shadows"]
        if not isinstance(raw_ranges, (list, tuple)):
            raise DecoderError(
                "decode_result: 'ranges' must be a list of range objects, "
                f"got {type(raw_ranges).__name__}."
            )
        if not isinstance(raw_shadows, (list, tuple)):
            raise DecoderError(
                "decode_result: 'shadows' must be a list of shadow objects, "
                f"got {type(raw_shadows).__name__}."
            )
        try:
            parsed_ranges = tuple(
                CandidateRange.from_dict(entry) for entry in raw_ranges
            )
        except ValueError as exc:
            raise DecoderError(
                f"decode_result: invalid range: {exc}."
            ) from exc
        try:
            parsed_shadows = tuple(
                ShadowRecord.from_dict(entry) for entry in raw_shadows
            )
        except ValueError as exc:
            raise DecoderError(
                f"decode_result: invalid shadow: {exc}."
            ) from exc
        return cls(
            ranges=parsed_ranges,
            shadows=parsed_shadows,
            schema_version=values["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> DecodeResult:
        if isinstance(data, (bytes, bytearray)):
            try:
                data = bytes(data).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DecoderError(
                    "decode_result: invalid UTF-8 JSON payload."
                ) from exc
        if not isinstance(data, str):
            raise DecoderError(
                "decode_result: JSON payload must be str or bytes, "
                f"got {type(data).__name__}."
            )
        try:
            decoded = json.loads(data)
        except json.JSONDecodeError as exc:
            raise DecoderError(f"decode_result: invalid JSON: {exc}.") from exc
        return cls.from_dict(decoded)


def pair_frames(
    features: Sequence[FeatureFrame],
    audio: Sequence[AudioEnergy],
) -> tuple[PairedFrame, ...]:
    """Pair pose frames with audio by exact source time (``F_t``).

    Both inputs must carry strictly increasing ``time_seconds``. Each
    pose frame pairs with the audio sample at the identical source time
    when present, else with ``audio_energy=None`` (unknown, never
    fabricated). Audio samples on times without a pose frame do not
    create pairs but remain available to the transient-window scan in
    :func:`decode_sequence`.
    """
    if isinstance(features, (FeatureFrame, AudioEnergy)) or not isinstance(
        features, (list, tuple)
    ):
        raise DecoderError(
            "decoder: 'features' must be a list or tuple of FeatureFrame, "
            f"got {type(features).__name__}."
        )
    if isinstance(audio, (FeatureFrame, AudioEnergy)) or not isinstance(
        audio, (list, tuple)
    ):
        raise DecoderError(
            "decoder: 'audio' must be a list or tuple of AudioEnergy, "
            f"got {type(audio).__name__}."
        )
    frame_list = list(features)
    audio_list = list(audio)
    for entry in frame_list:
        if not isinstance(entry, FeatureFrame):
            raise DecoderError(
                "decoder: every feature must be a FeatureFrame, "
                f"got {type(entry).__name__}."
            )
    for entry in audio_list:
        if not isinstance(entry, AudioEnergy):
            raise DecoderError(
                "decoder: every audio sample must be an AudioEnergy, "
                f"got {type(entry).__name__}."
            )
    for earlier, later in zip(frame_list, frame_list[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise DecoderError(
                "decoder: feature times must be strictly increasing, "
                f"got {earlier.time_seconds!r} followed by "
                f"{later.time_seconds!r}."
            )
    for earlier, later in zip(audio_list, audio_list[1:]):
        if not later.time_seconds > earlier.time_seconds:
            raise DecoderError(
                "decoder: audio times must be strictly increasing, "
                f"got {earlier.time_seconds!r} followed by "
                f"{later.time_seconds!r}."
            )
    by_time = {sample.time_seconds: sample.energy for sample in audio_list}
    return tuple(
        PairedFrame(
            time_seconds=frame.time_seconds,
            feature=frame,
            audio_energy=by_time.get(frame.time_seconds),
        )
        for frame in frame_list
    )


def _is_dropout(frame: FeatureFrame) -> bool:
    """Return True when a frame carries no usable pose evidence.

    Dropout is ``has_person is False``, ``visible_fraction`` below the
    0.4 floor, or a has-person frame whose geometry/motion channels are
    all ``None`` (fully unknown). Dropout frames bridge (hysteresis) or
    collapse a hypothesis but never vote for a transition.
    """
    if not frame.has_person:
        return True
    if frame.visible_fraction < VISIBILITY_FLOOR:
        return True
    if (
        frame.torso_displacement is None
        and frame.overhead_evidence is None
        and frame.rest_evidence is None
        and frame.motion_evidence is None
        and frame.elbow_speed is None
        and frame.elbow_flexion_left is None
        and frame.elbow_flexion_right is None
    ):
        return True
    return False


def _is_preparation_trigger(frame: FeatureFrame, config: DecoderConfig) -> bool:
    """Return True when proximal upper-arm activity opens preparation.

    Uses torso displacement or arm elevation (overhead position with
    high-confidence elbow geometry present: at least one elbow-flexion
    channel non-``None``, which the feature layer only computes above
    the visibility floor). Wrist speed is never consulted; ``None``
    evidence never triggers.
    """
    if _is_dropout(frame):
        return False
    torso = frame.torso_displacement
    if torso is not None and torso >= config.torso_displacement_threshold:
        return True
    if frame.overhead_evidence == 1.0 and (
        frame.elbow_flexion_left is not None
        or frame.elbow_flexion_right is not None
    ):
        return True
    return False


def _is_acceleration_trigger(frame: FeatureFrame, config: DecoderConfig) -> bool:
    """Return True when the arm accelerates into the overhead quadrant.

    Requires overhead position (``overhead == 1.0``), an elbow-speed
    surge at/above threshold, and present elbow geometry (at least one
    flexion channel non-``None``). Unknown (``None``) evidence never
    triggers.
    """
    if _is_dropout(frame):
        return False
    if frame.overhead_evidence != 1.0:
        return False
    speed = frame.elbow_speed
    if speed is None or not speed >= config.acceleration_elbow_speed_threshold:
        return False
    if (
        frame.elbow_flexion_left is None
        and frame.elbow_flexion_right is None
    ):
        return False
    return True


def _is_stillness_exit(frame: FeatureFrame, config: DecoderConfig) -> bool:
    """Return True when stillness returns and the wrist drops.

    Requires ``rest_evidence`` at/above threshold and
    ``overhead == 0.0`` (wrist below shoulder). ``None`` evidence never
    triggers.
    """
    if _is_dropout(frame):
        return False
    rest = frame.rest_evidence
    if rest is None or not rest >= config.rest_evidence_threshold:
        return False
    return frame.overhead_evidence == 0.0


def _session_baseline(audio: Sequence[AudioEnergy]) -> float:
    """Return the session-median ``A_t`` baseline (robust to transients).

    The median over every audio sample's energy; ``0.0`` for empty
    input. A median (not a mean) keeps one narrow impact spike from
    lifting its own baseline, so a real-scale 0.053 peak over a 0.009
    bed validates at ratio ~5.9 while skirt-level peaks near the median
    do not.
    """
    energies = sorted(sample.energy for sample in audio)
    if not energies:
        return 0.0
    midpoint = len(energies) // 2
    if len(energies) % 2 == 1:
        return energies[midpoint]
    return (energies[midpoint - 1] + energies[midpoint]) / 2.0


def _has_validating_transient(
    accel_time: float,
    audio: Sequence[AudioEnergy],
    config: DecoderConfig,
) -> bool:
    """Return True when a scale-invariant transient validates acceleration.

    Scans every audio sample by source time: the peak energy with
    ``|t - accel| <= window`` (1e-9 tolerance) validates when it clears
    both the small absolute ``audio_transient_floor`` (near-silence
    guard) and ``audio_transient_ratio`` times the session-median
    baseline. Empty or missing audio yields False (unknown, never
    fabricated).
    """
    peak: float | None = None
    for sample in audio:
        if abs(sample.time_seconds - accel_time) <= config.audio_window_seconds + 1e-9:
            if peak is None or sample.energy > peak:
                peak = sample.energy
    if peak is None:
        return False
    if peak < config.audio_transient_floor:
        return False
    baseline = _session_baseline(audio)
    return peak >= baseline * config.audio_transient_ratio


def decode_sequence(
    features: Sequence[FeatureFrame],
    audio: Sequence[AudioEnergy],
    config: DecoderConfig | None = None,
) -> DecodeResult:
    """Decode frame-aligned features plus audio into ranges and shadows.

    ``features`` and ``audio`` must each carry strictly increasing
    ``time_seconds``; an empty feature sequence yields empty outputs
    regardless of audio. ``config`` defaults to ``DecoderConfig()``.
    Valid serves become :class:`CandidateRange` values (unpadded,
    half-open, boundary times equal to transition frame times);
    non-validated hypotheses become :class:`ShadowRecord` values with
    reason ``"aborted"`` or ``"shadow"``.
    """
    cfg = config if config is not None else DecoderConfig()
    if not isinstance(cfg, DecoderConfig):
        raise DecoderError(
            "decoder: 'config' must be a DecoderConfig, "
            f"got {type(cfg).__name__}."
        )
    # Validates types/order for both series (pairing contract).
    pair_frames(features, audio)
    frame_list = list(features)
    audio_list = list(audio)
    if not frame_list:
        return DecodeResult(ranges=(), shadows=())

    ranges: list[CandidateRange] = []
    shadows: list[ShadowRecord] = []
    state = "idle"
    prep_start: float | None = None
    accel_time: float | None = None
    dropout = 0

    def _close_shadow(start: float, end: float, reason: str) -> None:
        if end > start:
            shadows.append(
                ShadowRecord(
                    start_seconds=start, end_seconds=end, reason=reason
                )
            )

    for frame in frame_list:
        moment = frame.time_seconds
        drop = _is_dropout(frame)
        if state == "idle":
            if drop:
                continue
            if _is_preparation_trigger(frame, cfg):
                state = "preparation"
                prep_start = moment
                accel_time = None
                dropout = 0
        elif state == "preparation":
            assert prep_start is not None
            if drop:
                dropout += 1
                if dropout > cfg.dropout_hysteresis_frames:
                    _close_shadow(prep_start, moment, REASON_ABORTED)
                    state = "idle"
                    prep_start = None
                    accel_time = None
                    dropout = 0
                continue
            dropout = 0
            if _is_acceleration_trigger(frame, cfg):
                state = "acceleration"
                accel_time = moment
                if _has_validating_transient(moment, audio_list, cfg):
                    state = "follow_through"
            elif _is_stillness_exit(frame, cfg):
                _close_shadow(prep_start, moment, REASON_ABORTED)
                state = "idle"
                prep_start = None
                accel_time = None
        elif state == "acceleration":
            assert prep_start is not None and accel_time is not None
            if drop:
                dropout += 1
                if dropout > cfg.dropout_hysteresis_frames:
                    _close_shadow(prep_start, moment, REASON_SHADOW)
                    state = "idle"
                    prep_start = None
                    accel_time = None
                    dropout = 0
                continue
            dropout = 0
            if _is_acceleration_trigger(frame, cfg):
                accel_time = moment
            if _has_validating_transient(accel_time, audio_list, cfg):
                state = "follow_through"
                continue
            if _is_stillness_exit(frame, cfg):
                _close_shadow(prep_start, moment, REASON_SHADOW)
                state = "idle"
                prep_start = None
                accel_time = None
        elif state == "follow_through":
            assert prep_start is not None
            if drop:
                dropout += 1
                if dropout > cfg.dropout_hysteresis_frames:
                    _close_shadow(prep_start, moment, REASON_SHADOW)
                    state = "idle"
                    prep_start = None
                    accel_time = None
                    dropout = 0
                continue
            dropout = 0
            if _is_stillness_exit(frame, cfg):
                if moment > prep_start:
                    ranges.append(
                        CandidateRange(
                            start_seconds=prep_start, end_seconds=moment
                        )
                    )
                state = "idle"
                prep_start = None
                accel_time = None
        else:  # pragma: no cover - unreachable state guard
            raise DecoderError(f"decoder: unknown state {state!r}.")

    if state == "preparation" and prep_start is not None:
        _close_shadow(prep_start, frame_list[-1].time_seconds, REASON_ABORTED)
    elif state == "acceleration" and prep_start is not None:
        _close_shadow(prep_start, frame_list[-1].time_seconds, REASON_SHADOW)
    elif state == "follow_through" and prep_start is not None:
        _close_shadow(prep_start, frame_list[-1].time_seconds, REASON_SHADOW)

    ranges.sort(key=lambda item: (item.start_seconds, item.end_seconds))
    shadows.sort(key=lambda item: (item.start_seconds, item.end_seconds))
    return DecodeResult(ranges=tuple(ranges), shadows=tuple(shadows))


#: Backwards-compatible alias for :func:`decode_sequence`.
decode_serves = decode_sequence
