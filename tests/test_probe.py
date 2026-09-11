"""Tests for the ffprobe adapter (M1.2).

All ffprobe interaction is faked via monkeypatched subprocess output; no
real or private video is required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from serve_review.domain import SourceMetadata
from serve_review.media import probe as probe_module
from serve_review.media.probe import (
    ProbeError,
    build_ffprobe_args,
    extract_rotation,
    normalize_rotation_value,
    parse_ffprobe_payload,
    probe_source,
    run_ffprobe,
    write_source_json,
)


def _make_video(tmp_path: Path, name: str = "clip.mov", data: bytes = b"fake-video-bytes") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _payload(**overrides) -> dict:
    stream = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "hevc",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "120/1",
        "r_frame_rate": "120/1",
        "time_base": "1/90000",
        "duration": "12.5",
    }
    stream.update(overrides.pop("stream", {}))
    payload = {
        "streams": [stream],
        "format": {"duration": "12.5"},
    }
    payload.update(overrides)
    return payload


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=["ffprobe"], returncode=returncode, stdout=stdout, stderr=stderr)


def _patch_run(monkeypatch, payload: dict | None = None, *, stdout=None, stderr="", returncode=0):
    if stdout is None:
        stdout = json.dumps(payload) if payload is not None else ""
    captured: dict = {}

    def _fake(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _completed(stdout=stdout, stderr=stderr, returncode=returncode)

    monkeypatch.setattr(probe_module.subprocess, "run", _fake)
    return captured


def test_hevc_metadata_preserves_rationals(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    payload = _payload(
        stream={
            "codec_name": "hevc",
            "width": 1920,
            "height": 1080,
            "avg_frame_rate": "24000/1001",
            "r_frame_rate": "240/1",
            "time_base": "1/90000",
            "duration": "10.5",
        }
    )
    _patch_run(monkeypatch, payload)
    meta = probe_source(video)
    assert isinstance(meta, SourceMetadata)
    assert meta.video_codec == "hevc"
    assert meta.width == 1920
    assert meta.height == 1080
    assert (meta.frame_rate_num, meta.frame_rate_den) == (24000, 1001)
    assert (meta.time_base_num, meta.time_base_den) == (1, 90000)
    assert meta.duration_seconds == pytest.approx(10.5)
    assert meta.rotation_degrees == 0
    assert meta.fingerprint.startswith("sha256:")


def test_variable_rate_fallback_to_r_frame_rate(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    payload = _payload(stream={"avg_frame_rate": "0/0", "r_frame_rate": "30/1"})
    _patch_run(monkeypatch, payload)
    meta = probe_source(video)
    assert (meta.frame_rate_num, meta.frame_rate_den) == (30, 1)


def test_stream_duration_preferred_over_format(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    payload = _payload(stream={"duration": "7.25"})
    payload["format"] = {"duration": "99.0"}
    _patch_run(monkeypatch, payload)
    meta = probe_source(video)
    assert meta.duration_seconds == pytest.approx(7.25)


def test_format_duration_fallback(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    stream = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "hevc",
        "width": 1280,
        "height": 720,
        "avg_frame_rate": "60/1",
        "r_frame_rate": "60/1",
        "time_base": "1/60000",
    }
    payload = {"streams": [stream], "format": {"duration": "3.5"}}
    _patch_run(monkeypatch, payload)
    meta = probe_source(video)
    assert meta.duration_seconds == pytest.approx(3.5)


def test_uses_argument_array_without_shell(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    captured = _patch_run(monkeypatch, _payload())
    probe_source(video)
    args = captured["args"]
    assert isinstance(args, list)
    assert args[0] == "ffprobe"
    assert str(video) in args
    assert "-of" in args and "json" in args
    assert captured["kwargs"].get("capture_output") is True
    assert "shell" not in captured["kwargs"]


def test_build_ffprobe_args_is_array(tmp_path: Path) -> None:
    args = build_ffprobe_args(tmp_path / "a b.mov")
    assert isinstance(args, list)
    assert all(isinstance(part, str) for part in args)
    assert str(tmp_path / "a b.mov") in args


def test_missing_video_file_raises_actionable(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="does not exist"):
        probe_source(tmp_path / "missing.mov")


def test_missing_tool_raises_actionable(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    monkeypatch.setattr(probe_module.shutil, "which", lambda _exe: None)
    with pytest.raises(ProbeError, match="not found.*install FFmpeg"):
        run_ffprobe(video)


def test_missing_tool_via_file_not_found(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)

    def _boom(args, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(probe_module.shutil, "which", lambda _exe: "/usr/bin/ffprobe")
    monkeypatch.setattr(probe_module.subprocess, "run", _boom)
    with pytest.raises(ProbeError, match="not found"):
        run_ffprobe(video)


def test_process_failure_raises_actionable(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    _patch_run(monkeypatch, stdout="", stderr="Invalid data found", returncode=1)
    with pytest.raises(ProbeError, match="ffprobe failed.*exit 1.*Invalid data"):
        probe_source(video)


def test_malformed_json_raises_actionable(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    _patch_run(monkeypatch, stdout="not json{{{", returncode=0)
    with pytest.raises(ProbeError, match="malformed JSON"):
        probe_source(video)


def test_non_object_json_raises(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    _patch_run(monkeypatch, stdout="[1,2]", returncode=0)
    with pytest.raises(ProbeError, match="JSON object"):
        probe_source(video)


def test_missing_video_stream_raises(tmp_path: Path) -> None:
    payload = {"streams": [{"codec_type": "audio", "codec_name": "aac"}], "format": {}}
    with pytest.raises(ProbeError, match="no video stream"):
        parse_ffprobe_payload(payload, "sha256:abc")


def test_empty_streams_raises(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="streams.*missing or empty"):
        parse_ffprobe_payload({"streams": [], "format": {}}, "sha256:abc")


def test_malformed_rational_raises(tmp_path: Path) -> None:
    payload = _payload(stream={"avg_frame_rate": "bogus", "r_frame_rate": "30/1"})
    with pytest.raises(ProbeError, match="malformed.*avg_frame_rate"):
        parse_ffprobe_payload(payload, "sha256:abc")


def test_missing_duration_raises(tmp_path: Path) -> None:
    stream = {
        "codec_type": "video",
        "codec_name": "hevc",
        "width": 64,
        "height": 48,
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
        "time_base": "1/90000",
    }
    with pytest.raises(ProbeError, match="duration is missing"):
        parse_ffprobe_payload({"streams": [stream], "format": {}}, "sha256:abc")


def test_missing_dimensions_raise(tmp_path: Path) -> None:
    payload = _payload(stream={"width": 0})
    with pytest.raises(ProbeError, match="width"):
        parse_ffprobe_payload(payload, "sha256:abc")


# --- Rotation normalization ---


def test_normalize_negative_modulo_360() -> None:
    assert normalize_rotation_value(-90, "t") == 270
    assert normalize_rotation_value("-90", "t") == 270
    assert normalize_rotation_value(-180, "t") == 180
    assert normalize_rotation_value(360, "t") == 0
    assert normalize_rotation_value("360", "t") == 0
    assert normalize_rotation_value(90.0, "t") == 90
    assert normalize_rotation_value("270.0", "t") == 270


def test_normalize_rejects_malformed() -> None:
    for bad in ("ninety", "", "90.5", 45.5, None, True, [90]):
        with pytest.raises(ProbeError, match="malformed rotation|unsupported rotation"):
            normalize_rotation_value(bad, "t")


def test_normalize_rejects_unsupported_degrees() -> None:
    for bad in (45, "45", 30, 135, 100):
        with pytest.raises(ProbeError, match="unsupported rotation"):
            normalize_rotation_value(bad, "t")


def test_rotation_default_zero() -> None:
    assert extract_rotation(_payload()["streams"][0]) == 0


def test_rotation_direct_stream_field() -> None:
    stream = _payload(stream={"rotation": 90})["streams"][0]
    assert extract_rotation(stream) == 90


def test_rotation_direct_field_string_negative() -> None:
    stream = _payload(stream={"rotation": "-90"})["streams"][0]
    assert extract_rotation(stream) == 270


def test_rotation_tags_rotate() -> None:
    stream = _payload(stream={"tags": {"rotate": "270"}})["streams"][0]
    assert extract_rotation(stream) == 270


def test_rotation_side_data_positive() -> None:
    stream = _payload(
        stream={"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 90}]}
    )["streams"][0]
    assert extract_rotation(stream) == 90


def test_rotation_side_data_negative_normalizes() -> None:
    stream = _payload(
        stream={"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}
    )["streams"][0]
    assert extract_rotation(stream) == 270


def test_rotation_side_data_negative_string_normalizes() -> None:
    stream = _payload(
        stream={"side_data_list": [{"side_data_type": "Display Matrix", "rotation": "-90"}]}
    )["streams"][0]
    assert extract_rotation(stream) == 270


def test_rotation_side_data_malformed_rejected() -> None:
    stream = _payload(
        stream={"side_data_list": [{"side_data_type": "Display Matrix", "rotation": "tilted"}]}
    )["streams"][0]
    with pytest.raises(ProbeError, match="malformed rotation"):
        extract_rotation(stream)


def test_rotation_side_data_unsupported_rejected() -> None:
    stream = _payload(
        stream={"side_data_list": [{"side_data_type": "Display Matrix", "rotation": 45}]}
    )["streams"][0]
    with pytest.raises(ProbeError, match="unsupported rotation"):
        extract_rotation(stream)


def test_rotation_direct_malformed_rejected() -> None:
    stream = _payload(stream={"rotation": "sideways"})["streams"][0]
    with pytest.raises(ProbeError, match="malformed rotation"):
        extract_rotation(stream)


def test_rotation_direct_unsupported_rejected() -> None:
    stream = _payload(stream={"rotation": 45})["streams"][0]
    with pytest.raises(ProbeError, match="unsupported rotation"):
        extract_rotation(stream)


def test_rotation_precedence_direct_over_tags_over_side_data() -> None:
    # Direct field wins over both tags and side data.
    stream = _payload(
        stream={
            "rotation": 180,
            "tags": {"rotate": "90"},
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": 270}],
        }
    )["streams"][0]
    assert extract_rotation(stream) == 180


def test_rotation_precedence_tags_over_side_data() -> None:
    # Tags win over side_data_list when no direct field is present.
    stream = _payload(
        stream={
            "tags": {"rotate": "90"},
            "side_data_list": [{"side_data_type": "Display Matrix", "rotation": 270}],
        }
    )["streams"][0]
    assert extract_rotation(stream) == 90


def test_rotation_conflicting_side_data_first_entry_wins() -> None:
    stream = _payload(
        stream={
            "side_data_list": [
                {"side_data_type": "Display Matrix", "rotation": 90},
                {"side_data_type": "Display Matrix", "rotation": 180},
            ]
        }
    )["streams"][0]
    assert extract_rotation(stream) == 90


def test_rotation_ignores_non_display_matrix_entries() -> None:
    stream = _payload(
        stream={
            "side_data_list": [
                {"side_data_type": "Some Other Data", "rotation": 180},
                {"side_data_type": "Display Matrix", "rotation": 270},
            ]
        }
    )["streams"][0]
    assert extract_rotation(stream) == 270


def test_rotation_display_matrix_convention_is_counter_clockwise() -> None:
    """Document the FFmpeg display-matrix convention used by the sampler.

    FFmpeg's ``-display_rotation`` sets a pure counter-clockwise rotation
    and ffprobe reports it in ``side_data_list`` Display Matrix entries
    (negative equivalents such as ``-90``/``-180`` normalize via modulo
    360 to ``270``/``180``). The sampler applies the probed value
    counter-clockwise so upright frames match the autorotate/export
    orientation; see ``tests/test_frames.py`` for the pixel-level
    ground-truth comparison at 0/90/180/270.
    """
    # Negative Display Matrix values normalize to their CCW equivalents.
    for raw, expected in ((-90, 270), ("-90", 270), (-180, 180), (90, 90)):
        stream = _payload(
            stream={"side_data_list": [
                {"side_data_type": "Display Matrix", "rotation": raw}
            ]}
        )["streams"][0]
        assert extract_rotation(stream) == expected


def test_probe_end_to_end_side_data_negative(tmp_path: Path, monkeypatch) -> None:
    video = _make_video(tmp_path)
    payload = _payload(
        stream={"side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}]}
    )
    _patch_run(monkeypatch, payload)
    meta = probe_source(video)
    assert meta.rotation_degrees == 270


# --- Atomic JSON output ---


def test_write_source_json_round_trip(tmp_path: Path) -> None:
    meta = SourceMetadata(
        fingerprint="sha256:abc",
        duration_seconds=5.0,
        width=64,
        height=48,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
        time_base_num=1,
        time_base_den=15360,
    )
    dest = tmp_path / "nested" / "source.json"
    written = write_source_json(meta, dest)
    assert written == dest
    assert dest.is_file()
    assert SourceMetadata.from_json(dest.read_text(encoding="utf-8")) == meta
    leftovers = list((tmp_path / "nested").glob("source.json.tmp-*"))
    assert leftovers == []


def test_write_source_json_no_partial_on_failure(tmp_path: Path, monkeypatch) -> None:
    meta = SourceMetadata(
        fingerprint="sha256:abc",
        duration_seconds=5.0,
        width=64,
        height=48,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
    )
    dest = tmp_path / "source.json"

    def _boom(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(probe_module.os, "replace", _boom)
    with pytest.raises(ProbeError, match="could not write"):
        write_source_json(meta, dest)
    assert not dest.exists()
