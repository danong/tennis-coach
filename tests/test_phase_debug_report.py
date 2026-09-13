"""Focused tests for the private phase-debug report command (M4 Unit 3).

Deterministic, offline, synthetic only; no private footage, no media
decoding, no pose inference, no checkpoints mutation, no held-out
access. Covers parser contract, deterministic JSON/HTML, output
collision/overwrite, malformed/linkage failures, optional manual
mapping, and input immutability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from serve_review.checkpoints.evidence import (
    EVIDENCE_DEFAULT_CONFIG_ID,
    EVIDENCE_METHOD_VERSION,
    PhaseEvidence,
    StageCandidate,
)
from serve_review.checkpoints.phase_features import (
    PHASE_FEATURES_METHOD_VERSION,
    PhaseFeatureGrid,
    PhaseFeatureSample,
)
from serve_review.checkpoints.phase_solver import (
    PhaseSolverConfig,
    solve_with_diagnostics,
)
from serve_review.cli import build_parser, phase_debug_cmd
from serve_review.domain import STAGE_ORDER, Attempt, MediaRange

ATTEMPT_START = 10.0
ATTEMPT_END = 12.0
GRID_RATE_HZ = 30.0


def _attempt_range() -> MediaRange:
    return MediaRange(start_seconds=ATTEMPT_START, end_seconds=ATTEMPT_END)


def _attempt() -> Attempt:
    detected = _attempt_range()
    return Attempt(
        attempt_id="serve-001",
        detected_range=detected,
        effective_range=MediaRange(
            start_seconds=ATTEMPT_START, end_seconds=ATTEMPT_END
        ),
        confidence=0.9,
        evidence={"motion": 1.0},
    )


def _keyframe(index: int) -> float:
    return 10.20 + 0.15 * index


def _grid_sample(time: float) -> PhaseFeatureSample:
    return PhaseFeatureSample(
        time_seconds=time,
        observed=True,
        observation_quality=0.9,
        interpolation_span_seconds=0.0,
        source_time_offset_seconds=0.0,
        temporal_uncertainty_seconds=0.002,
        derivative_confidence=0.8,
        derivative_quality=0.7,
        wrist_left_x=0.10,
        wrist_left_y=-0.20,
        wrist_left_vx=0.50,
        wrist_left_vy=-1.00,
        wrist_right_x=0.20,
        wrist_right_y=-0.30,
        wrist_right_vx=0.40,
        wrist_right_vy=-0.80,
        elbow_angle_right=90.0,
        knee_angle_left=150.0,
        knee_angle_right=140.0,
        shoulder_axis_camera_dx=1.0,
        shoulder_axis_camera_dy=0.1,
        hip_axis_camera_dx=1.0,
        hip_axis_camera_dy=-0.05,
        torso_rotation_camera_deg=5.0,
        audio_energy=0.30,
        audio_candidate=False,
    )


def _grid() -> PhaseFeatureGrid:
    step = 1.0 / GRID_RATE_HZ
    count = int(round((ATTEMPT_END - ATTEMPT_START) / step))
    samples = tuple(
        _grid_sample(ATTEMPT_START + index * step) for index in range(count)
    )
    return PhaseFeatureGrid(
        attempt_range=_attempt_range(),
        method_version=PHASE_FEATURES_METHOD_VERSION,
        config_id="phase-features-default-v2",
        source_frame_rate_hz=240.0,
        pose_observation_rate_hz=GRID_RATE_HZ,
        grid_rate_hz=GRID_RATE_HZ,
        position_window_samples=3,
        derivative_window_samples=3,
        direct_observation_tolerance_seconds=0.015,
        samples=samples,
    )


def _candidate(stage: str, keyframe: float, *, score: float = 0.70) -> StageCandidate:
    return StageCandidate(
        stage=stage,
        keyframe_seconds=keyframe,
        interval=MediaRange(
            start_seconds=keyframe - 0.02, end_seconds=keyframe + 0.02
        ),
        score=score,
        observation_quality=0.8,
        derivative_quality=0.7,
        evidence=("cue_a",),
        limitations=("camera_relative_only",),
        provenance="audio_transient" if stage == "contact" else "body_pose",
        temporal_uncertainty_seconds=0.02,
    )


def _evidence() -> PhaseEvidence:
    members = [
        _candidate(stage, _keyframe(index), score=0.70 - 0.02 * index)
        for index, stage in enumerate(STAGE_ORDER)
    ]
    return PhaseEvidence(
        attempt_range=_attempt_range(),
        method_version=EVIDENCE_METHOD_VERSION,
        config_id=EVIDENCE_DEFAULT_CONFIG_ID,
        candidates=tuple(members),
    )


def _write_inputs(tmp_path: Path, *, manual: dict | None = None):
    grid = _grid()
    evidence = _evidence()
    config = PhaseSolverConfig()
    result = solve_with_diagnostics(_attempt(), grid, evidence, config)
    grid_path = tmp_path / "grid.json"
    grid_path.write_text(
        json.dumps(grid.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(evidence.to_json(), encoding="utf-8")
    result_path = tmp_path / "result.json"
    result_path.write_text(result.to_json(), encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.to_json(), encoding="utf-8")
    manual_path = None
    if manual is not None:
        manual_path = tmp_path / "manual.json"
        manual_path.write_text(
            json.dumps(manual, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return grid_path, evidence_path, result_path, config_path, manual_path, result


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parser_contract() -> None:
    args = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", "grid.json",
            "--evidence", "evidence.json",
            "--solver-result", "result.json",
            "--solver-config", "config.json",
            "--output", "out.json",
        ]
    )
    assert args.grid == Path("grid.json")
    assert args.evidence == Path("evidence.json")
    assert args.solver_result == Path("result.json")
    assert args.solver_config == Path("config.json")
    assert args.manual is None
    assert args.output == Path("out.json")
    assert args.html is None
    assert args.overwrite is False

    args2 = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", "g.json",
            "--evidence", "e.json",
            "--solver-result", "r.json",
            "--solver-config", "c.json",
            "--manual", "m.json",
            "--output", "o.json",
            "--html", "o.html",
            "--overwrite",
        ]
    )
    assert args2.manual == Path("m.json")
    assert args2.html == Path("o.html")
    assert args2.overwrite is True


def test_help_documents_phase_debug(capsys) -> None:
    top_help = build_parser().format_help()
    assert "phase-debug" in top_help
    parser = build_parser()
    import pytest as _pytest

    with _pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["phase-debug", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--grid" in out
    assert "--evidence" in out
    assert "--solver-result" in out
    assert "--solver-config" in out
    assert "--output" in out


def test_deterministic_json_html(tmp_path: Path, capsys) -> None:
    grid_p, ev_p, res_p, cfg_p, _, _ = _write_inputs(tmp_path)
    out_json = tmp_path / "phase-debug.json"
    out_html = tmp_path / "phase-debug.html"
    args = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(grid_p),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--output", str(out_json),
            "--html", str(out_html),
        ]
    )
    assert phase_debug_cmd(args) == 0
    capsys.readouterr()
    first_json = out_json.read_bytes()
    first_html = out_html.read_bytes()
    payload = json.loads(first_json.decode("utf-8"))
    assert payload["attempt_id"] == "serve-001"
    assert set(payload["stages"].keys()) == set(STAGE_ORDER)
    assert payload["total_objective"] is not None
    assert payload["solver_config_id"] == PhaseSolverConfig().config_id

    text = first_html.decode("utf-8")
    assert "<script" not in text.lower()
    assert "http://" not in text and "https://" not in text
    for token in (
        "manual time", "selected time", "manual support", "selected support",
        "manual score", "manual rank", "selected score", "selected rank",
        "candidates", "selection explanation", "unary contribution",
        "transition contribution", "skip contribution",
        "objective contribution", "contact offset",
        "unavailable reason", "anomalies", "bounded trace",
        "release", "cocking", "contact",
    ):
        assert token in text.lower(), token
    # Honest limitation labels.
    assert "not an observed ball release" in text
    assert "never claimed as exact visual" in text
    assert "camera-relative" in text

    # Determinism: rerun with overwrite yields byte-identical outputs.
    args2 = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(grid_p),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--output", str(out_json),
            "--html", str(out_html),
            "--overwrite",
        ]
    )
    assert phase_debug_cmd(args2) == 0
    capsys.readouterr()
    assert out_json.read_bytes() == first_json
    assert out_html.read_bytes() == first_html


def test_default_html_sibling(tmp_path: Path, capsys) -> None:
    grid_p, ev_p, res_p, cfg_p, _, _ = _write_inputs(tmp_path)
    out_json = tmp_path / "report.json"
    args = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(grid_p),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--output", str(out_json),
        ]
    )
    assert phase_debug_cmd(args) == 0
    capsys.readouterr()
    assert out_json.is_file()
    assert (tmp_path / "report.html").is_file()


def test_collision_and_overwrite(tmp_path: Path, capsys) -> None:
    grid_p, ev_p, res_p, cfg_p, _, _ = _write_inputs(tmp_path)
    out_json = tmp_path / "phase-debug.json"
    out_html = tmp_path / "phase-debug.html"
    base = [
        "phase-debug",
        "--grid", str(grid_p),
        "--evidence", str(ev_p),
        "--solver-result", str(res_p),
        "--solver-config", str(cfg_p),
        "--output", str(out_json),
        "--html", str(out_html),
    ]
    assert phase_debug_cmd(build_parser().parse_args(base)) == 0
    capsys.readouterr()
    before_json = out_json.read_bytes()
    before_html = out_html.read_bytes()
    # Collision without overwrite.
    assert phase_debug_cmd(build_parser().parse_args(base)) == 1
    err = capsys.readouterr().err
    assert "collision" in err.lower()
    assert out_json.read_bytes() == before_json
    assert out_html.read_bytes() == before_html
    # Overwrite succeeds.
    assert phase_debug_cmd(build_parser().parse_args([*base, "--overwrite"])) == 0
    capsys.readouterr()
    assert out_json.read_bytes() == before_json
    assert out_html.read_bytes() == before_html


def test_malformed_input_leaves_no_partial_outputs(tmp_path: Path, capsys) -> None:
    grid_p, ev_p, res_p, cfg_p, _, _ = _write_inputs(tmp_path)
    bad_grid = tmp_path / "bad-grid.json"
    bad_grid.write_text("{not json", encoding="utf-8")
    out_json = tmp_path / "out.json"
    out_html = tmp_path / "out.html"
    args = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(bad_grid),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--output", str(out_json),
            "--html", str(out_html),
        ]
    )
    assert phase_debug_cmd(args) == 2
    assert "ERROR" in capsys.readouterr().err
    assert not out_json.exists()
    assert not out_html.exists()
    # Missing input file also returns 2 with no outputs.
    args2 = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(tmp_path / "missing.json"),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--output", str(out_json),
            "--html", str(out_html),
        ]
    )
    assert phase_debug_cmd(args2) == 2
    assert not out_json.exists()
    assert not out_html.exists()
    _ = grid_p  # inputs preserved


def test_linkage_failure_leaves_no_partial_outputs(tmp_path: Path, capsys) -> None:
    grid_p, ev_p, res_p, cfg_p, _, _ = _write_inputs(tmp_path)
    # Evidence on a shifted attempt range breaks grid/evidence linkage.
    shifted_members = []
    for index, stage in enumerate(STAGE_ORDER):
        key = _keyframe(index) + 100.0
        shifted_members.append(
            _candidate(stage, key).to_dict()
        )
    bad_evidence = {
        "attempt_range": MediaRange(110.0, 112.0).to_dict(),
        "candidates": shifted_members,
        "config_id": EVIDENCE_DEFAULT_CONFIG_ID,
        "method_version": EVIDENCE_METHOD_VERSION,
        "schema_version": 1,
    }
    # Resolve evidence schema version from a valid payload instead of assuming.
    valid = json.loads(ev_p.read_text(encoding="utf-8"))
    bad_evidence["schema_version"] = valid["schema_version"]
    bad_path = tmp_path / "bad-evidence.json"
    bad_path.write_text(json.dumps(bad_evidence), encoding="utf-8")
    out_json = tmp_path / "out.json"
    out_html = tmp_path / "out.html"
    args = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(grid_p),
            "--evidence", str(bad_path),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--output", str(out_json),
            "--html", str(out_html),
        ]
    )
    assert phase_debug_cmd(args) == 2
    assert "ERROR" in capsys.readouterr().err
    assert not out_json.exists()
    assert not out_html.exists()


def test_optional_manual_mapping(tmp_path: Path, capsys) -> None:
    # Without manual: manual times/scores are null.
    grid_p, ev_p, res_p, cfg_p, _, _ = _write_inputs(tmp_path)
    out_json = tmp_path / "no-manual.json"
    args = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(grid_p),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--output", str(out_json),
        ]
    )
    assert phase_debug_cmd(args) == 0
    capsys.readouterr()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["stages"]["start"]["manual_time_seconds"] is None
    assert payload["stages"]["start"]["manual_score"] is None

    # With exact manual mapping: manual score/rank inherit candidate values.
    manual = {stage: _keyframe(i) for i, stage in enumerate(STAGE_ORDER)}
    _, _, _, _, manual_path, _ = _write_inputs(tmp_path / "m2" if False else tmp_path, manual=None)
    _ = manual_path
    manual_file = tmp_path / "manual.json"
    manual_file.write_text(json.dumps(manual), encoding="utf-8")
    out_json2 = tmp_path / "with-manual.json"
    args2 = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(grid_p),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--manual", str(manual_file),
            "--output", str(out_json2),
        ]
    )
    assert phase_debug_cmd(args2) == 0
    capsys.readouterr()
    payload2 = json.loads(out_json2.read_text(encoding="utf-8"))
    assert payload2["stages"]["start"]["manual_time_seconds"] == manual["start"]
    assert payload2["stages"]["start"]["manual_score"] is not None
    assert payload2["stages"]["start"]["manual_rank"] == 1


def test_inputs_not_modified(tmp_path: Path, capsys) -> None:
    manual = {stage: _keyframe(i) for i, stage in enumerate(STAGE_ORDER)}
    grid_p, ev_p, res_p, cfg_p, manual_p, _ = _write_inputs(tmp_path, manual=manual)
    before = {p: _hash(p) for p in (grid_p, ev_p, res_p, cfg_p, manual_p)}
    out_json = tmp_path / "out.json"
    args = build_parser().parse_args(
        [
            "phase-debug",
            "--grid", str(grid_p),
            "--evidence", str(ev_p),
            "--solver-result", str(res_p),
            "--solver-config", str(cfg_p),
            "--manual", str(manual_p),
            "--output", str(out_json),
        ]
    )
    assert phase_debug_cmd(args) == 0
    capsys.readouterr()
    for path, digest in before.items():
        assert _hash(path) == digest
