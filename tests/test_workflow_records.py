from __future__ import annotations

import hashlib
import json

import pytest

from serve_review.domain import SourceMetadata
from serve_review.workflow.records import (
    SourceRecord,
    WorkflowRecordError,
    attempt_id_for,
    source_id_for_fingerprint,
)


def metadata(fingerprint: str, *, duration: float = 1.0) -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=duration,
        width=1920,
        height=1080,
        frame_rate_num=30000,
        frame_rate_den=1001,
        video_codec="h264",
    )


def record_payload(tmp_path, fingerprint: str | None = None) -> dict:
    fingerprint = fingerprint or "sha256:" + "a" * 64
    record = SourceRecord(
        schema_version=1,
        source_id=source_id_for_fingerprint(fingerprint),
        source_fingerprint=fingerprint,
        metadata=metadata(fingerprint),
        known_paths=(str(tmp_path / "video.mov"),),
    )
    return record.to_dict()


def test_source_id_uses_complete_fingerprint_not_path_or_name() -> None:
    fingerprint = "sha256:" + "A" * 64
    expected = hashlib.sha256(fingerprint.lower().encode("ascii")).hexdigest()[:16]
    assert source_id_for_fingerprint(fingerprint) == "source-" + expected
    assert source_id_for_fingerprint(fingerprint.lower()) == source_id_for_fingerprint(
        fingerprint
    )
    assert source_id_for_fingerprint("sha256:" + "a" * 63 + "b") != (
        source_id_for_fingerprint(fingerprint)
    )


@pytest.mark.parametrize(
    "fingerprint",
    ["", "a" * 64, "sha1:" + "a" * 64, "sha256:abc", "sha256:" + "g" * 64],
)
def test_source_id_rejects_malformed_fingerprints(fingerprint: str) -> None:
    with pytest.raises(WorkflowRecordError, match="64 hex"):
        source_id_for_fingerprint(fingerprint)


def test_source_record_codec_is_deterministic_and_normalizes_hex(tmp_path) -> None:
    fingerprint = "sha256:" + "A" * 64
    record = SourceRecord(
        1,
        source_id_for_fingerprint(fingerprint),
        fingerprint,
        metadata(fingerprint),
        (str(tmp_path / "video.mov"),),
    )
    assert record.source_fingerprint == fingerprint.lower()
    assert record.metadata.fingerprint == fingerprint.lower()
    assert record.to_json() == record.to_json()
    assert record.to_json().endswith("\n")
    assert SourceRecord.from_json(record.to_json()) == record
    assert SourceRecord.from_json(record.to_json().encode()) == record


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "unsupported schema_version"),
        ({"schema_version": True}, "must be an integer"),
        ({"source_id": "source-0000000000000000"}, "does not match"),
        ({"source_fingerprint": "bad"}, "64 hex"),
        ({"known_paths": []}, "nonempty tuple"),
        ({"known_paths": ["relative.mov"]}, "absolute strings"),
    ],
)
def test_source_record_rejects_invalid_fields(tmp_path, change, message) -> None:
    payload = record_payload(tmp_path)
    payload.update(change)
    with pytest.raises(WorkflowRecordError, match=message):
        SourceRecord.from_dict(payload)


def test_source_record_rejects_noncanonical_known_paths(tmp_path) -> None:
    payload = record_payload(tmp_path)
    first = str(tmp_path / "a.mov")
    second = str(tmp_path / "b.mov")
    for paths in ([second, first], [first, first], [first + "/../a.mov"]):
        payload["known_paths"] = paths
        with pytest.raises(WorkflowRecordError, match="known_paths|normalized"):
            SourceRecord.from_dict(payload)


def test_source_record_rejects_missing_unknown_and_malformed_nested_keys(tmp_path) -> None:
    payload = record_payload(tmp_path)
    missing = dict(payload)
    missing.pop("source_id")
    unknown = dict(payload, surprise=True)
    nested = json.loads(json.dumps(payload))
    nested["metadata"]["surprise"] = True
    for invalid in (missing, unknown, nested):
        with pytest.raises(WorkflowRecordError):
            SourceRecord.from_dict(invalid)


@pytest.mark.parametrize(
    "payload",
    [
        "[]",
        "null",
        "{bad",
        '{"schema_version": 1, "schema_version": 1}',
        "{\"duration\": NaN}",
        b"\xff",
    ],
)
def test_source_record_rejects_invalid_strict_json(payload) -> None:
    with pytest.raises(WorkflowRecordError):
        SourceRecord.from_json(payload)


def test_attempt_id_is_deterministic_and_uses_every_exact_input() -> None:
    source = "source-0123456789abcdef"
    baseline = attempt_id_for(source, "detector:v1", "config:abc", 0.0, 1.0)
    assert baseline == attempt_id_for(source, "detector:v1", "config:abc", 0, 1)
    assert baseline.startswith("attempt-")
    assert len(baseline.removeprefix("attempt-")) >= 16
    variants = [
        attempt_id_for("source-1123456789abcdef", "detector:v1", "config:abc", 0, 1),
        attempt_id_for(source, "detector:v2", "config:abc", 0, 1),
        attempt_id_for(source, "detector:v1", "config:def", 0, 1),
        attempt_id_for(source, "detector:v1", "config:abc", 0.0000001, 1),
        attempt_id_for(source, "detector:v1", "config:abc", 0, 1.0000001),
    ]
    assert baseline not in variants
    assert len(set(variants)) == len(variants)


@pytest.mark.parametrize(
    "arguments",
    [
        ("source", "method", "config", 0, 1),
        ("source-0123456789abcdef", "", "config", 0, 1),
        ("source-0123456789abcdef", "method", " ", 0, 1),
        ("source-0123456789abcdef", "method", "config", True, 1),
        ("source-0123456789abcdef", "method", "config", -1, 1),
        ("source-0123456789abcdef", "method", "config", 1, 1),
        ("source-0123456789abcdef", "method", "config", 2, 1),
        ("source-0123456789abcdef", "method", "config", float("nan"), 1),
        ("source-0123456789abcdef", "method", "config", 0, float("inf")),
    ],
)
def test_attempt_id_rejects_invalid_inputs(arguments) -> None:
    with pytest.raises(WorkflowRecordError):
        attempt_id_for(*arguments)
