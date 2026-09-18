"""Persisted schema and fixed contract for ServeFingerprintV1.

Extraction helpers are added below this schema layer in the same module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from serve_review.checkpoints.kinematic_waveforms import (
    CHANNEL_INDEX as WAVEFORM_CHANNEL_INDEX,
    KinematicWaveformTrack,
)
from serve_review.domain import AttemptPhase

SCHEMA = "serve-fingerprint-v1"
SCHEMA_VERSION = 1
METHOD_VERSION = "serve-fingerprint-v1"
DEFAULT_CONFIG_ID = "serve-fingerprint-default-v1"
ARTIFACT_FILENAME = "serve-fingerprint-v1.json"
COMPARISON_DOMAIN = "right-handed-compatible-rear-view-v1"
LAYOUT_ID = "serve-five-segment-layout-v1"
RESAMPLING_METHOD = "linear-source-time-complete-support-v1"
SAMPLES_PER_SEGMENT = 16
SHAPE = (5, 16, 12)
ANCHOR_NAMES = ("start", "release", "loading", "cocking", "contact", "finish")

METRIC_UNITS: dict[str, str] = {
    "start_to_release_duration": "seconds",
    "release_to_loading_duration": "seconds",
    "loading_to_cocking_duration": "seconds",
    "cocking_to_contact_duration": "seconds",
    "contact_to_finish_duration": "seconds",
    "knee_flexion_left_at_loading": "degrees",
    "knee_flexion_right_at_loading": "degrees",
    "knee_flexion_left_max_release_to_cocking": "degrees",
    "knee_flexion_right_max_release_to_cocking": "degrees",
    "knee_extension_rate_left_peak_loading_to_contact": "degrees/second",
    "knee_extension_rate_right_peak_loading_to_contact": "degrees/second",
    "shoulder_hip_separation_max_release_to_contact": "degrees",
    "shoulder_hip_separation_max_time_relative_to_contact": "seconds",
    "shoulder_tilt_at_loading": "degrees",
    "shoulder_tilt_at_contact": "degrees",
    "hip_tilt_at_loading": "degrees",
    "hip_tilt_at_contact": "degrees",
    "torso_verticality_at_loading": "body_lengths",
    "hip_ankle_vertical_extent_at_loading": "body_lengths",
    "toss_arm_elevation_at_release": "body_lengths",
    "toss_arm_elevation_max_release_to_cocking": "body_lengths",
    "toss_arm_extension_at_release": "body_lengths",
    "hitting_elbow_flexion_at_cocking": "degrees",
    "hitting_elbow_flexion_at_contact": "degrees",
    "hitting_elbow_extension_rate_peak_cocking_to_contact": "degrees/second",
    "hitting_elbow_extension_peak_time_relative_to_contact": "seconds",
    "right_wrist_speed_peak_cocking_to_contact": "body_lengths/second",
    "right_wrist_speed_peak_time_relative_to_contact": "seconds",
    "right_wrist_shoulder_distance_at_cocking": "body_lengths",
    "right_wrist_shoulder_distance_at_contact": "body_lengths",
}
METRIC_NAMES = tuple(METRIC_UNITS)

CHANNEL_UNITS: dict[str, str] = {
    "knee_flexion_left": "degrees",
    "knee_flexion_right": "degrees",
    "knee_flexion_velocity_left": "degrees/second",
    "knee_flexion_velocity_right": "degrees/second",
    "elbow_flexion_right": "degrees",
    "elbow_flexion_velocity_right": "degrees/second",
    "shoulder_hip_separation_transverse_deg": "degrees",
    "shoulder_tilt_deg": "degrees",
    "hip_tilt_deg": "degrees",
    "left_arm_elevation": "body_lengths",
    "left_arm_extension": "body_lengths",
    "right_wrist_speed": "body_lengths/second",
}
CHANNEL_NAMES = tuple(CHANNEL_UNITS)
SEGMENT_LAYOUT = (
    ("start_to_release", "start", "release"),
    ("release_to_loading", "release", "loading"),
    ("loading_to_cocking", "loading", "cocking"),
    ("cocking_to_contact", "cocking", "contact"),
    ("contact_to_finish", "contact", "finish"),
)
SEGMENT_NAMES = tuple(row[0] for row in SEGMENT_LAYOUT)


class ServeFingerprintError(ValueError):
    """Invalid ServeFingerprintV1 value or JSON representation."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ServeFingerprintError(f"{name} must be a finite number.")
    return float(value)


def _keys(data: Any, required: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping) or set(data) != required:
        raise ServeFingerprintError(f"{name} must contain exactly {sorted(required)}.")
    return data


def _deterministic_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"


@dataclass(frozen=True)
class ServeFingerprintConfig:
    """V1 identity pin; it cannot alter the persisted contract."""

    config_id: str = DEFAULT_CONFIG_ID

    def __post_init__(self) -> None:
        if self.config_id != DEFAULT_CONFIG_ID:
            raise ServeFingerprintError(f"config_id is fixed to {DEFAULT_CONFIG_ID!r} for V1.")


@dataclass(frozen=True)
class AnchorValue:
    available: bool
    time_seconds: float | None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise ServeFingerprintError("anchor available must be boolean.")
        if self.available:
            object.__setattr__(self, "time_seconds", _finite(self.time_seconds, "anchor time_seconds"))
        elif self.time_seconds is not None:
            raise ServeFingerprintError("unavailable anchor time_seconds must be null.")

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "time_seconds": self.time_seconds}


@dataclass(frozen=True)
class MetricValue:
    available: bool
    value: float | None
    unit: str

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise ServeFingerprintError("metric available must be boolean.")
        if self.available:
            object.__setattr__(self, "value", _finite(self.value, "metric value"))
        elif self.value is not None:
            raise ServeFingerprintError("unavailable metric value must be null.")

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "value": self.value, "unit": self.unit}


@dataclass(frozen=True)
class SequenceSegment:
    id: str
    available: bool
    duration_metric: str
    values: tuple[tuple[float | None, ...], ...]
    availability: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        if self.id not in SEGMENT_NAMES or type(self.available) is not bool:
            raise ServeFingerprintError("invalid sequence segment identity/availability.")
        expected_duration = f"{self.id}_duration"
        if self.duration_metric != expected_duration:
            raise ServeFingerprintError(f"{self.id} duration_metric must be {expected_duration!r}.")
        if len(self.values) != SAMPLES_PER_SEGMENT or len(self.availability) != SAMPLES_PER_SEGMENT:
            raise ServeFingerprintError("each sequence segment must have exactly 16 rows.")
        for row, mask in zip(self.values, self.availability):
            if len(row) != len(CHANNEL_NAMES) or len(mask) != len(CHANNEL_NAMES):
                raise ServeFingerprintError("each sequence row/mask must have exactly 12 columns.")
            for value, supported in zip(row, mask):
                if type(supported) is not bool:
                    raise ServeFingerprintError("sequence availability cells must be boolean.")
                if supported:
                    _finite(value, "available sequence cell")
                elif value is not None:
                    raise ServeFingerprintError("unavailable sequence cells must be null.")
                if not self.available and (supported or value is not None):
                    raise ServeFingerprintError("structurally unavailable segments must be all null/false.")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "available": self.available,
                "duration_metric": self.duration_metric,
                "values": [list(row) for row in self.values],
                "availability": [list(row) for row in self.availability]}


@dataclass(frozen=True)
class ServeFingerprintV1:
    source_fingerprint: str
    attempt_start_seconds: float
    attempt_end_seconds: float
    provenance: Mapping[str, str]
    anchors: Mapping[str, AnchorValue]
    metrics: Mapping[str, MetricValue]
    segments: tuple[SequenceSegment, ...]
    config_id: str = DEFAULT_CONFIG_ID

    def __post_init__(self) -> None:
        if not isinstance(self.source_fingerprint, str) or not self.source_fingerprint.strip():
            raise ServeFingerprintError("source_fingerprint must be non-blank.")
        start = _finite(self.attempt_start_seconds, "attempt_start_seconds")
        end = _finite(self.attempt_end_seconds, "attempt_end_seconds")
        if end <= start:
            raise ServeFingerprintError("attempt range must have positive duration.")
        object.__setattr__(self, "attempt_start_seconds", start)
        object.__setattr__(self, "attempt_end_seconds", end)
        ServeFingerprintConfig(self.config_id)
        provenance_keys = {"coordinate_convention", "waveform_method_version", "waveform_config_id",
                           "checkpoint_method_version", "checkpoint_config_id", "body_model_name", "body_model_version"}
        _keys(self.provenance, provenance_keys, "provenance")
        if any(not isinstance(v, str) or not v.strip() for v in self.provenance.values()):
            raise ServeFingerprintError("provenance values must be non-blank strings.")
        _keys(self.anchors, set(ANCHOR_NAMES), "anchors")
        if any(not isinstance(v, AnchorValue) for v in self.anchors.values()):
            raise ServeFingerprintError("anchors must contain AnchorValue entries.")
        _keys(self.metrics, set(METRIC_NAMES), "metrics")
        for metric, value in self.metrics.items():
            if not isinstance(value, MetricValue) or value.unit != METRIC_UNITS[metric]:
                raise ServeFingerprintError(f"metric {metric!r} has invalid type or unit.")
        if len(self.segments) != len(SEGMENT_LAYOUT):
            raise ServeFingerprintError("exactly five sequence segments are required.")
        if tuple(s.id for s in self.segments) != SEGMENT_NAMES:
            raise ServeFingerprintError("sequence segment order does not match V1 layout.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
            "extractor": {"config_id": self.config_id, "method_version": METHOD_VERSION},
            "identity": {"source_fingerprint": self.source_fingerprint,
                         "attempt_start_seconds": self.attempt_start_seconds,
                         "attempt_end_seconds": self.attempt_end_seconds},
            "comparison_domain": COMPARISON_DOMAIN,
            "provenance": dict(self.provenance),
            "phase_layout": {"layout_id": LAYOUT_ID, "samples_per_segment": SAMPLES_PER_SEGMENT,
                             "resampling_method": RESAMPLING_METHOD,
                             "segments": [{"id": s, "start_anchor": a, "end_anchor": b}
                                          for s, a, b in SEGMENT_LAYOUT]},
            "anchors": {name: self.anchors[name].to_dict() for name in ANCHOR_NAMES},
            "metrics": {name: self.metrics[name].to_dict() for name in METRIC_NAMES},
            "normalized_sequence": {"channel_order": list(CHANNEL_NAMES),
                                    "channel_units": [CHANNEL_UNITS[n] for n in CHANNEL_NAMES],
                                    "shape": list(SHAPE),
                                    "segments": [s.to_dict() for s in self.segments]},
        }

    def to_json(self) -> str:
        return _deterministic_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServeFingerprintV1":
        root = _keys(payload, {"schema", "schema_version", "extractor", "identity", "comparison_domain",
                               "provenance", "phase_layout", "anchors", "metrics", "normalized_sequence"}, "fingerprint")
        if root["schema"] != SCHEMA or root["schema_version"] != SCHEMA_VERSION:
            raise ServeFingerprintError("unsupported fingerprint schema/version.")
        extractor = _keys(root["extractor"], {"config_id", "method_version"}, "extractor")
        if extractor["method_version"] != METHOD_VERSION:
            raise ServeFingerprintError("unsupported extractor method_version.")
        identity = _keys(root["identity"], {"source_fingerprint", "attempt_start_seconds", "attempt_end_seconds"}, "identity")
        if root["comparison_domain"] != COMPARISON_DOMAIN:
            raise ServeFingerprintError("unsupported comparison_domain.")
        layout = _keys(root["phase_layout"], {"layout_id", "samples_per_segment", "resampling_method", "segments"}, "phase_layout")
        expected_layout = [{"id": s, "start_anchor": a, "end_anchor": b} for s, a, b in SEGMENT_LAYOUT]
        if layout != {"layout_id": LAYOUT_ID, "samples_per_segment": SAMPLES_PER_SEGMENT,
                      "resampling_method": RESAMPLING_METHOD, "segments": expected_layout}:
            raise ServeFingerprintError("phase_layout does not match fixed V1 layout.")
        seq = _keys(root["normalized_sequence"], {"channel_order", "channel_units", "shape", "segments"}, "normalized_sequence")
        if seq["channel_order"] != list(CHANNEL_NAMES) or seq["channel_units"] != [CHANNEL_UNITS[n] for n in CHANNEL_NAMES] or seq["shape"] != list(SHAPE):
            raise ServeFingerprintError("normalized_sequence inventory/shape does not match V1.")
        anchors_raw = _keys(root["anchors"], set(ANCHOR_NAMES), "anchors")
        anchors: dict[str, AnchorValue] = {}
        for name, item in anchors_raw.items():
            item = _keys(item, {"available", "time_seconds"}, f"anchor {name}")
            anchors[name] = AnchorValue(item["available"], item["time_seconds"])
        metrics_raw = _keys(root["metrics"], set(METRIC_NAMES), "metrics")
        metrics: dict[str, MetricValue] = {}
        for name, item in metrics_raw.items():
            item = _keys(item, {"available", "value", "unit"}, f"metric {name}")
            metrics[name] = MetricValue(item["available"], item["value"], item["unit"])
        segs = []
        if not isinstance(seq["segments"], list) or len(seq["segments"]) != 5:
            raise ServeFingerprintError("normalized_sequence requires exactly five segments.")
        for item in seq["segments"]:
            item = _keys(item, {"id", "available", "duration_metric", "values", "availability"}, "sequence segment")
            segs.append(SequenceSegment(item["id"], item["available"], item["duration_metric"],
                                        tuple(tuple(r) for r in item["values"]),
                                        tuple(tuple(r) for r in item["availability"])))
        return cls(identity["source_fingerprint"], identity["attempt_start_seconds"], identity["attempt_end_seconds"],
                   root["provenance"], anchors, metrics, tuple(segs), extractor["config_id"])

    @classmethod
    def from_json(cls, data: str | bytes | bytearray) -> "ServeFingerprintV1":
        if isinstance(data, (bytes, bytearray)):
            data = bytes(data).decode("utf-8")
        if not isinstance(data, str):
            raise ServeFingerprintError("JSON payload must be str or bytes.")
        try:
            decoded = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ServeFingerprintError(f"invalid fingerprint JSON: {exc}.") from exc
        if not isinstance(decoded, dict):
            raise ServeFingerprintError("fingerprint JSON must be an object.")
        return cls.from_dict(decoded)


def selected_anchor_times(attempt_phase: AttemptPhase) -> dict[str, float | None]:
    """Return V1's six directly selected anchors from an attempt phase.

    ``acceleration`` and ``deceleration`` are intentionally not fingerprint
    anchors; the V1 sequence layout is defined by the six named phases.
    """
    if not isinstance(attempt_phase, AttemptPhase):
        raise ServeFingerprintError("attempt_phase must be an AttemptPhase.")
    return {
        name: attempt_phase.stages[name].keyframe_seconds
        if attempt_phase.stages[name].availability != "unavailable"
        else None
        for name in ANCHOR_NAMES
    }


def _sample_value(sample: Any, channel: str) -> float | None:
    value = sample.channel_values[WAVEFORM_CHANNEL_INDEX[channel]]
    available = sample.channel_available[WAVEFORM_CHANNEL_INDEX[channel]]
    return float(value) if available and value is not None else None


def _metric(value: float | None, name: str) -> MetricValue:
    return MetricValue(value is not None, value, METRIC_UNITS[name])


def extract_scalar_metrics(
    waveform: KinematicWaveformTrack,
    anchors: Mapping[str, float | None],
) -> dict[str, MetricValue]:
    """Extract the fixed V1 scalar inventory from existing waveform values.

    Anchor samples require exact canonical source-time equality. Interval
    reductions use closed intervals and require at least three rows with
    complete availability for the selected channel. Ties resolve to the
    earliest waveform row because tracks are ordered by source time.
    """
    if not isinstance(waveform, KinematicWaveformTrack):
        raise ServeFingerprintError("waveform must be a KinematicWaveformTrack.")
    if set(anchors) != set(ANCHOR_NAMES):
        raise ServeFingerprintError(f"anchors must contain exactly {list(ANCHOR_NAMES)}.")
    times: dict[str, float | None] = {}
    for name, value in anchors.items():
        times[name] = None if value is None else _finite(value, f"anchor {name}")

    by_time = {sample.time_seconds: sample for sample in waveform.samples}

    def exact(anchor: str, channel: str) -> float | None:
        time = times[anchor]
        sample = by_time.get(time) if time is not None else None
        return _sample_value(sample, channel) if sample is not None else None

    def interval(start: str, end: str, channel: str) -> tuple[float, float] | None:
        left, right = times[start], times[end]
        if left is None or right is None or right < left:
            return None
        rows = [s for s in waveform.samples if left <= s.time_seconds <= right]
        values = [_sample_value(s, channel) for s in rows]
        if len(rows) < 3 or any(value is None for value in values):
            return None
        winner = max(range(len(rows)), key=lambda i: (values[i], -rows[i].time_seconds))
        return float(values[winner]), rows[winner].time_seconds

    metrics: dict[str, MetricValue] = {}

    def put(name: str, value: float | None) -> None:
        metrics[name] = _metric(value, name)

    for start, end, name in (
        ("start", "release", "start_to_release_duration"),
        ("release", "loading", "release_to_loading_duration"),
        ("loading", "cocking", "loading_to_cocking_duration"),
        ("cocking", "contact", "cocking_to_contact_duration"),
        ("contact", "finish", "contact_to_finish_duration"),
    ):
        a, b = times[start], times[end]
        put(name, b - a if a is not None and b is not None and b >= a else None)

    for name, channel, anchor in (
        ("knee_flexion_left_at_loading", "knee_flexion_left", "loading"),
        ("knee_flexion_right_at_loading", "knee_flexion_right", "loading"),
        ("shoulder_tilt_at_loading", "shoulder_tilt_deg", "loading"),
        ("shoulder_tilt_at_contact", "shoulder_tilt_deg", "contact"),
        ("hip_tilt_at_loading", "hip_tilt_deg", "loading"),
        ("hip_tilt_at_contact", "hip_tilt_deg", "contact"),
        ("torso_verticality_at_loading", "torso_verticality", "loading"),
        ("hip_ankle_vertical_extent_at_loading", "hip_ankle_vertical_extent", "loading"),
        ("toss_arm_elevation_at_release", "left_arm_elevation", "release"),
        ("toss_arm_extension_at_release", "left_arm_extension", "release"),
        ("hitting_elbow_flexion_at_cocking", "elbow_flexion_right", "cocking"),
        ("hitting_elbow_flexion_at_contact", "elbow_flexion_right", "contact"),
        ("right_wrist_shoulder_distance_at_cocking", "right_wrist_rel_shoulder_distance", "cocking"),
        ("right_wrist_shoulder_distance_at_contact", "right_wrist_rel_shoulder_distance", "contact"),
    ):
        put(name, exact(anchor, channel))

    for side in ("left", "right"):
        name = f"knee_flexion_{side}_max_release_to_cocking"
        result = interval("release", "cocking", f"knee_flexion_{side}")
        put(name, result[0] if result else None)
        name = f"knee_extension_rate_{side}_peak_loading_to_contact"
        result = interval("loading", "contact", f"knee_flexion_velocity_{side}")
        put(name, -min((float(_sample_value(s, f"knee_flexion_velocity_{side}"))
                        for s in waveform.samples
                        if times["loading"] is not None and times["contact"] is not None
                        and times["loading"] <= s.time_seconds <= times["contact"]), default=math.inf)
            if result is not None else None)

    sep = interval("release", "contact", "shoulder_hip_separation_transverse_deg")
    put("shoulder_hip_separation_max_release_to_contact", sep[0] if sep else None)
    contact = times["contact"]
    put("shoulder_hip_separation_max_time_relative_to_contact",
        sep[1] - contact if sep is not None and contact is not None else None)
    elevation = interval("release", "cocking", "left_arm_elevation")
    put("toss_arm_elevation_max_release_to_cocking", elevation[0] if elevation else None)
    elbow = interval("cocking", "contact", "elbow_flexion_velocity_right")
    # Extension is the positive magnitude of the most negative flexion rate.
    elbow_rows = [s for s in waveform.samples if times["cocking"] is not None
                  and times["contact"] is not None
                  and times["cocking"] <= s.time_seconds <= times["contact"]]
    elbow_values = [_sample_value(s, "elbow_flexion_velocity_right") for s in elbow_rows]
    elbow_ext = (-min(elbow_values), elbow_rows[min(range(len(elbow_values)),
                 key=lambda i: elbow_values[i])].time_seconds) if elbow is not None else None
    put("hitting_elbow_extension_rate_peak_cocking_to_contact", elbow_ext[0] if elbow_ext else None)
    put("hitting_elbow_extension_peak_time_relative_to_contact",
        elbow_ext[1] - contact if elbow_ext is not None and contact is not None else None)
    wrist = interval("cocking", "contact", "right_wrist_speed")
    put("right_wrist_speed_peak_cocking_to_contact", wrist[0] if wrist else None)
    put("right_wrist_speed_peak_time_relative_to_contact",
        wrist[1] - contact if wrist is not None and contact is not None else None)

    # Keep output ordering tied to the persisted schema inventory.
    return {name: metrics[name] for name in METRIC_NAMES}


def _sequence_segments(
    waveform: KinematicWaveformTrack,
    anchors: Mapping[str, float | None],
) -> tuple[SequenceSegment, ...]:
    """Resample the fixed phase layout with complete native support."""
    rows = waveform.samples
    times = tuple(sample.time_seconds for sample in rows)
    result: list[SequenceSegment] = []
    for segment_id, start_name, end_name in SEGMENT_LAYOUT:
        start, end = anchors[start_name], anchors[end_name]
        native_indices = (
            [i for i, time in enumerate(times) if start <= time <= end]
            if start is not None and end is not None and end >= start
            else []
        )
        structurally_available = start is not None and end is not None and len(native_indices) >= 3
        values: list[tuple[float | None, ...]] = []
        masks: list[tuple[bool, ...]] = []
        if not structurally_available:
            values = [(None,) * len(CHANNEL_NAMES) for _ in range(SAMPLES_PER_SEGMENT)]
            masks = [(False,) * len(CHANNEL_NAMES) for _ in range(SAMPLES_PER_SEGMENT)]
        else:
            assert start is not None and end is not None
            target_times = [start + i * (end - start) / (SAMPLES_PER_SEGMENT - 1)
                            for i in range(SAMPLES_PER_SEGMENT)]
            target_times[0], target_times[-1] = start, end
            for channel in CHANNEL_NAMES:
                native_values = [_sample_value(rows[index], channel) for index in native_indices]
                # V1 support is whole-series: no target may bridge an unavailable row.
                if any(value is None for value in native_values):
                    channel_values: list[float | None] = [None] * SAMPLES_PER_SEGMENT
                    channel_masks = [False] * SAMPLES_PER_SEGMENT
                else:
                    channel_values = []
                    channel_masks = []
                    for target in target_times:
                        exact_index = next((index for index in native_indices if times[index] == target), None)
                        if exact_index is not None:
                            value = _sample_value(rows[exact_index], channel)
                            assert value is not None
                            channel_values.append(value)
                            channel_masks.append(True)
                            continue
                        right_pos = next((p for p, index in enumerate(native_indices) if times[index] > target), None)
                        if right_pos is None or right_pos == 0:
                            channel_values.append(None)
                            channel_masks.append(False)
                            continue
                        left_index, right_index = native_indices[right_pos - 1], native_indices[right_pos]
                        left_value = _sample_value(rows[left_index], channel)
                        right_value = _sample_value(rows[right_index], channel)
                        if left_value is None or right_value is None:
                            channel_values.append(None)
                            channel_masks.append(False)
                            continue
                        fraction = ((target - times[left_index]) /
                                    (times[right_index] - times[left_index]))
                        channel_values.append(left_value + fraction * (right_value - left_value))
                        channel_masks.append(True)
                # Build rows below in channel order.
                values.append(tuple(channel_values))
                masks.append(tuple(channel_masks))
            # The loop above produces channels by row; transpose into [sample][channel].
            values = [tuple(values[channel][sample] for channel in range(len(CHANNEL_NAMES)))
                      for sample in range(SAMPLES_PER_SEGMENT)]
            masks = [tuple(masks[channel][sample] for channel in range(len(CHANNEL_NAMES)))
                     for sample in range(SAMPLES_PER_SEGMENT)]
        result.append(SequenceSegment(
            segment_id, structurally_available, f"{segment_id}_duration",
            tuple(values), tuple(masks),
        ))
    return tuple(result)


def build_serve_fingerprint_v1(
    track: KinematicWaveformTrack,
    attempt_phase: AttemptPhase,
    *,
    source_fingerprint: str,
    body_model_name: str,
    body_model_version: str,
    config: ServeFingerprintConfig | None = None,
) -> ServeFingerprintV1:
    """Build a deterministic V1 fingerprint from existing waveforms and anchors."""
    if not isinstance(track, KinematicWaveformTrack):
        raise ServeFingerprintError("track must be a KinematicWaveformTrack.")
    if not isinstance(attempt_phase, AttemptPhase):
        raise ServeFingerprintError("attempt_phase must be an AttemptPhase.")
    config = config or ServeFingerprintConfig()
    if not isinstance(config, ServeFingerprintConfig):
        raise ServeFingerprintError("config must be a ServeFingerprintConfig.")
    times = selected_anchor_times(attempt_phase)
    grid = {sample.time_seconds for sample in track.samples}
    for name, time in times.items():
        if time is not None and time not in grid:
            raise ServeFingerprintError(
                f"available direct anchor {name!r} at {time!r} is not on the waveform source-time grid."
            )
    anchors = {name: AnchorValue(time is not None, time) for name, time in times.items()}
    metrics = extract_scalar_metrics(track, times)
    segments = _sequence_segments(track, times)
    provenance = {
        "coordinate_convention": track.coordinate_convention,
        "waveform_method_version": track.method_version,
        "waveform_config_id": track.config.config_id,
        "checkpoint_method_version": attempt_phase.method_version,
        "checkpoint_config_id": attempt_phase.config_id,
        "body_model_name": body_model_name,
        "body_model_version": body_model_version,
    }
    return ServeFingerprintV1(
        source_fingerprint=source_fingerprint,
        attempt_start_seconds=attempt_phase.attempt_range.start_seconds,
        attempt_end_seconds=attempt_phase.attempt_range.end_seconds,
        provenance=provenance,
        anchors=anchors,
        metrics=metrics,
        segments=segments,
        config_id=config.config_id,
    )
