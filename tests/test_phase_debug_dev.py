"""Focused tests for the private dev-only phase-debug bridge.

Deterministic, offline, synthetic only; no private footage, no media
decoding, no pose inference, no checkpoints mutation, no held-out
execution. Pose caches are real JSONL files in temporary directories;
probe/audio stages are injected fakes (CLI tests monkeypatch the real
adapters). Covers parser contract, dev-manifest enforcement, manual
mapping, exact attempt linkage, output collision/overwrite,
deterministic provenance writes, and non-mutation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from serve_review.checkpoint_evaluation import PhaseAnnotation, PhaseAnnotationManifest
from serve_review.checkpoints.evidence import EVIDENCE_DEFAULT_CONFIG_ID
from serve_review.checkpoints.phase_features import PhaseFeatureGrid
from serve_review.checkpoints.evidence import PhaseEvidence
from serve_review.checkpoints.phase_solver import (
    PhaseSolverConfig,
    PhaseSolverResult,
)
from serve_review.cli import build_parser, phase_debug_dev_cmd
from serve_review.domain import (
    STAGE_ORDER,
    Attempt,
    AttemptDocument,
    MediaRange,
    SourceMetadata,
)
from serve_review.media.audio import AudioEnergy
from serve_review.phase_debug_dev import (
    DEBUG_HTML_FILENAME,
    DEBUG_JSON_FILENAME,
    EVIDENCE_FILENAME,
    GRID_FILENAME,
    SOLVER_CONFIG_FILENAME,
    SOLVER_RESULT_FILENAME,
    PhaseDebugDevCollisionError,
    PhaseDebugDevInputError,
    manual_keyframes_for_attempt,
    run_phase_debug_dev,
)
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose.mediapipe import MODEL_NAME, MODEL_VERSION
from serve_review.pose.schema import CacheIdentity, FrameObservation

FINGERPRINT = "sha256:phase-debug-dev-fixture"
SESSION = "sess-dev"
MEDIA = "sess-dev/cap01.mov"


def make_metadata(
    duration: float = 10.0, fingerprint: str = FINGERPRINT
) -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=duration,
        width=320,
        height=240,
        frame_rate_num=120,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
    )


def make_attempt(index: int, start: float, end: float) -> Attempt:
    detected = MediaRange(start_seconds=start, end_seconds=end)
    return Attempt(
        attempt_id=f"serve-{index:03d}",
        detected_range=detected,
        effective_range=MediaRange(start_seconds=start, end_seconds=end),
        confidence=0.9,
        evidence={"motion": 1.0},
    )


def make_document(
    metadata: SourceMetadata, spans: list[tuple[float, float]]
) -> AttemptDocument:
    attempts = tuple(
        make_attempt(position, start, end)
        for position, (start, end) in enumerate(spans, start=1)
    )
    export = tuple(MediaRange(start_seconds=s, end_seconds=e) for s, e in spans)
    return AttemptDocument(
        source_fingerprint=metadata.fingerprint,
        source_duration_seconds=metadata.duration_seconds,
        padding_seconds=0.0,
        method_version="candidate-ranges-v1+plan-v1",
        attempts=attempts,
        export_ranges=export,
    )


def make_obs(moment: float) -> FrameObservation:
    return FrameObservation(
        time_seconds=moment,
        timestamp_ms=round(moment * 1000),
        persons=(),
    )


def expected_identity(metadata: SourceMetadata) -> CacheIdentity:
    return CacheIdentity(
        source_fingerprint=metadata.fingerprint,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )


def make_source(tmp_path: Path, name: str = "session.mov") -> Path:
    video = tmp_path / name
    video.write_bytes(b"fake-source-bytes")
    return video


def write_attempts(path: Path, document: AttemptDocument) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.to_json(), encoding="utf-8")
    return path


def write_cache(
    path: Path,
    identity: CacheIdentity,
    frames: list[FrameObservation],
    *,
    complete: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if complete:
        return cache_module.write_complete_cache(path, identity, frames)
    return cache_module.write_partial_cache(path, identity, frames)


def fake_probe(metadata: SourceMetadata):
    def _probe(video: Path) -> SourceMetadata:
        return metadata

    return _probe


def fake_audio(energies: dict[float, float] | None = None, floor: float = 0.01):
    calls: list[dict] = []
    table = dict(energies or {})

    def _audio(video, schedule, **kwargs):
        calls.append({"video": Path(video), "count": len(tuple(schedule))})
        out = []
        for moment in schedule:
            energy = floor
            for spike_time, spike_energy in table.items():
                if abs(float(moment) - float(spike_time)) <= 0.003:
                    energy = spike_energy
                    break
            out.append(AudioEnergy(time_seconds=float(moment), energy=energy))
        return out

    _audio.calls = calls  # type: ignore[attr-defined]
    return _audio


def make_manifest(
    attempt: Attempt,
    *,
    split: str = "dev",
    unavailable: set[str] | None = None,
    ambiguous: set[str] | None = None,
    keyframes: dict[str, float] | None = None,
) -> PhaseAnnotationManifest:
    unavailable = set(unavailable or set())
    ambiguous = set(ambiguous or set())
    rows: list[PhaseAnnotation] = []
    start = attempt.detected_range.start_seconds
    for position, stage in enumerate(STAGE_ORDER):
        base = start + 0.10 + position * 0.05
        if stage in ambiguous:
            rows.append(
                PhaseAnnotation(
                    session_id=SESSION,
                    media=MEDIA,
                    attempt_id=attempt.attempt_id,
                    attempt_start_seconds=start,
                    attempt_end_seconds=attempt.detected_range.end_seconds,
                    stage=stage,
                    status="ambiguous",
                    interval_start_seconds=None,
                    interval_end_seconds=None,
                    manual_keyframe_seconds=None,
                    confidence=None,
                    attempt_label="serve",
                )
            )
        elif stage in unavailable:
            rows.append(
                PhaseAnnotation(
                    session_id=SESSION,
                    media=MEDIA,
                    attempt_id=attempt.attempt_id,
                    attempt_start_seconds=start,
                    attempt_end_seconds=attempt.detected_range.end_seconds,
                    stage=stage,
                    status="unavailable",
                    interval_start_seconds=None,
                    interval_end_seconds=None,
                    manual_keyframe_seconds=None,
                    confidence=None,
                    attempt_label="serve",
                )
            )
        else:
            key = (
                keyframes[stage]
                if keyframes is not None and stage in keyframes
                else base
            )
            rows.append(
                PhaseAnnotation(
                    session_id=SESSION,
                    media=MEDIA,
                    attempt_id=attempt.attempt_id,
                    attempt_start_seconds=start,
                    attempt_end_seconds=attempt.detected_range.end_seconds,
                    stage=stage,
                    status="available",
                    interval_start_seconds=key,
                    interval_end_seconds=key + 0.02,
                    manual_keyframe_seconds=key,
                    confidence=None,
                    attempt_label="serve",
                )
            )
    return PhaseAnnotationManifest(dataset_split=split, annotations=tuple(rows))  # type: ignore[arg-type]


def prepare(
    tmp_path: Path,
    *,
    spans: list[tuple[float, float]] = [(1.0, 2.0)],
    duration: float = 10.0,
    obs_step: float = 0.05,
    energies: dict[float, float] | None = None,
    unavailable: set[str] | None = None,
) -> dict:
    metadata = make_metadata(duration=duration)
    video = make_source(tmp_path)
    attempts_path = tmp_path / "attempts.json"
    cache_path = tmp_path / "pose-v1.jsonl"
    annotations_path = tmp_path / "annotations.json"
    document = make_document(metadata, spans)
    write_attempts(attempts_path, document)
    moments: list[float] = []
    moment = 0.0
    while moment < duration:
        moments.append(round(moment, 9))
        moment += obs_step
    frames = [make_obs(m) for m in moments]
    write_cache(cache_path, expected_identity(metadata), frames)
    attempt = document.attempts[0]
    manifest = make_manifest(attempt, unavailable=unavailable)
    annotations_path.write_text(manifest.to_json(), encoding="utf-8")
    return {
        "metadata": metadata,
        "video": video,
        "attempts": attempts_path,
        "cache": cache_path,
        "annotations": annotations_path,
        "document": document,
        "attempt": attempt,
        "manifest": manifest,
        "audio": fake_audio(energies),
        "frames": frames,
        "output_dir": tmp_path / "phase-debug-dev" / attempt.attempt_id,
    }


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Parser -----------------------------------------------------------------


def test_parser_contract() -> None:
    args = build_parser().parse_args(
        [
            "phase-debug-dev",
            "video.mov",
            "--attempts", "attempts.json",
            "--cache", "cache.jsonl",
            "--annotations", "annotations.json",
            "--attempt-id", "serve-001",
            "--output-dir", "out",
        ]
    )
    assert args.video == Path("video.mov")
    assert args.attempts == Path("attempts.json")
    assert args.cache == Path("cache.jsonl")
    assert args.annotations == Path("annotations.json")
    assert args.attempt_id == "serve-001"
    assert args.output_dir == Path("out")
    assert args.overwrite is False
    # Aliases resolve to the same destinations.
    alt = build_parser().parse_args(
        [
            "phase-debug-dev",
            "video.mov",
            "--attempts", "a.json",
            "--cache", "c.jsonl",
            "--manifest", "m.json",
            "--attempt-id", "serve-002",
            "--output-dir", "o",
            "--overwrite",
        ]
    )
    assert alt.annotations == Path("m.json")
    assert alt.overwrite is True


def test_help_documents_phase_debug_dev(capsys) -> None:
    top_help = build_parser().format_help()
    assert "phase-debug-dev" in top_help
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["phase-debug-dev", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--attempts" in out
    assert "--cache" in out
    assert "--annotations" in out or "--manifest" in out
    assert "--attempt-id" in out
    assert "--output-dir" in out


# --- Dev-manifest enforcement -------------------------------------------------


def test_heldout_manifest_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    heldout = make_manifest(ctx["attempt"], split="heldout")
    ctx["annotations"].write_text(heldout.to_json(), encoding="utf-8")
    with pytest.raises(PhaseDebugDevInputError, match="heldout|dev"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id=ctx["attempt"].attempt_id,
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert not ctx["output_dir"].exists() or not any(ctx["output_dir"].iterdir())


def test_ambiguous_row_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    manifest = make_manifest(ctx["attempt"], ambiguous={"contact"})
    ctx["annotations"].write_text(manifest.to_json(), encoding="utf-8")
    with pytest.raises(PhaseDebugDevInputError, match="ambiguous"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id=ctx["attempt"].attempt_id,
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )


def test_unknown_attempt_id_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    with pytest.raises(PhaseDebugDevInputError, match="unknown attempt"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id="serve-999",
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )


# --- Manual mapping -----------------------------------------------------------


def test_manual_mapping_available_and_unavailable(tmp_path: Path) -> None:
    ctx = prepare(tmp_path, unavailable={"finish"})
    result = run_phase_debug_dev(
        ctx["video"],
        attempts_path=ctx["attempts"],
        cache_path=ctx["cache"],
        annotations_path=ctx["annotations"],
        attempt_id=ctx["attempt"].attempt_id,
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
    )
    payload = json.loads(result.output_json.read_text(encoding="utf-8"))
    assert payload["attempt_id"] == ctx["attempt"].attempt_id
    assert set(payload["stages"].keys()) == set(STAGE_ORDER)
    # Available stages echo manifest keyframes; unavailable reads as null.
    for stage in STAGE_ORDER:
        row = next(
            entry for entry in ctx["manifest"].annotations if entry.stage == stage
        )
        if row.status == "unavailable":
            assert payload["stages"][stage]["manual_time_seconds"] is None
        else:
            assert (
                payload["stages"][stage]["manual_time_seconds"]
                == row.manual_keyframe_seconds
            )
    # Pure helper agrees with the report input.
    mapping = manual_keyframes_for_attempt(
        ctx["manifest"], ctx["attempt"].attempt_id, ctx["attempt"]
    )
    for stage in STAGE_ORDER:
        assert mapping[stage] == payload["stages"][stage]["manual_time_seconds"]


def test_provenance_inputs_parse_with_existing_codecs(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    result = run_phase_debug_dev(
        ctx["video"],
        attempts_path=ctx["attempts"],
        cache_path=ctx["cache"],
        annotations_path=ctx["annotations"],
        attempt_id=ctx["attempt"].attempt_id,
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
    )
    grid = PhaseFeatureGrid.from_dict(
        json.loads(result.grid_path.read_text(encoding="utf-8"))
    )
    evidence = PhaseEvidence.from_json(result.evidence_path.read_text(encoding="utf-8"))
    solver_result = PhaseSolverResult.from_json(
        result.solver_result_path.read_text(encoding="utf-8")
    )
    solver_config = PhaseSolverConfig.from_json(
        result.solver_config_path.read_text(encoding="utf-8")
    )
    assert grid.attempt_range == ctx["attempt"].detected_range
    assert evidence.attempt_range == ctx["attempt"].detected_range
    assert solver_result.attempt_phase.attempt_id == ctx["attempt"].attempt_id
    # Defaults are preserved (no tuning).
    assert grid.config_id == "phase-features-default-v2"
    assert evidence.config_id == EVIDENCE_DEFAULT_CONFIG_ID
    assert solver_config.config_id == PhaseSolverConfig().config_id
    assert result.total_objective == solver_result.total_score


# --- Exact linkage ------------------------------------------------------------


def test_source_fingerprint_mismatch_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    other = make_metadata(fingerprint="sha256:other")
    with pytest.raises(PhaseDebugDevInputError, match="fingerprint"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id=ctx["attempt"].attempt_id,
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(other),
            audio_fn=ctx["audio"],
        )
    assert not result_exists(ctx["output_dir"])


def test_source_duration_mismatch_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    other = make_metadata(duration=99.0)
    with pytest.raises(PhaseDebugDevInputError, match="duration"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id=ctx["attempt"].attempt_id,
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(other),
            audio_fn=ctx["audio"],
        )


def test_manifest_range_mismatch_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    shifted_attempt = make_attempt(1, 3.0, 4.0)
    manifest = make_manifest(shifted_attempt)
    # Keep the same attempt id but a different range linkage.
    ctx["annotations"].write_text(manifest.to_json(), encoding="utf-8")
    with pytest.raises(PhaseDebugDevInputError, match="range"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id=ctx["attempt"].attempt_id,
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )


def test_stale_cache_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    stale_identity = CacheIdentity(
        source_fingerprint="sha256:other",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )
    write_cache(ctx["cache"], stale_identity, ctx["frames"])
    with pytest.raises(PhaseDebugDevInputError, match="stale"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id=ctx["attempt"].attempt_id,
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )


def test_incomplete_cache_rejected(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    write_cache(
        ctx["cache"], expected_identity(ctx["metadata"]), ctx["frames"], complete=False
    )
    with pytest.raises(PhaseDebugDevInputError, match="incomplete"):
        run_phase_debug_dev(
            ctx["video"],
            attempts_path=ctx["attempts"],
            cache_path=ctx["cache"],
            annotations_path=ctx["annotations"],
            attempt_id=ctx["attempt"].attempt_id,
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )


def result_exists(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    return any(output_dir.iterdir())


# --- Collision / determinism / non-mutation ------------------------------------


def test_collision_and_deterministic_overwrite(tmp_path: Path, capsys) -> None:
    ctx = prepare(tmp_path)
    kwargs = dict(
        attempts_path=ctx["attempts"],
        cache_path=ctx["cache"],
        annotations_path=ctx["annotations"],
        attempt_id=ctx["attempt"].attempt_id,
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
    )
    first = run_phase_debug_dev(ctx["video"], **kwargs)  # type: ignore[arg-type]
    capsys.readouterr()
    before = {
        path: _hash(path)
        for path in (
            first.grid_path,
            first.evidence_path,
            first.solver_result_path,
            first.solver_config_path,
            first.output_json,
            first.output_html,
        )
    }
    # Collision without overwrite.
    with pytest.raises(PhaseDebugDevCollisionError, match="collision"):
        run_phase_debug_dev(ctx["video"], **kwargs)  # type: ignore[arg-type]
    for path, digest in before.items():
        assert _hash(path) == digest
    # Overwrite succeeds and is byte-identical.
    second = run_phase_debug_dev(ctx["video"], overwrite=True, **kwargs)  # type: ignore[arg-type]
    assert second.output_json == first.output_json
    for path, digest in before.items():
        assert _hash(path) == digest


def test_inputs_not_modified_and_no_m3_mutation(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    before = {
        path: _hash(path)
        for path in (
            ctx["video"],
            ctx["attempts"],
            ctx["cache"],
            ctx["annotations"],
        )
    }
    session_files_before = set()
    result = run_phase_debug_dev(
        ctx["video"],
        attempts_path=ctx["attempts"],
        cache_path=ctx["cache"],
        annotations_path=ctx["annotations"],
        attempt_id=ctx["attempt"].attempt_id,
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
    )
    for path, digest in before.items():
        assert _hash(path) == digest
    # No M3/checkpoint side effects beside the explicit output dir.
    assert result.output_dir == ctx["output_dir"]
    assert (ctx["output_dir"] / GRID_FILENAME).is_file()
    assert (ctx["output_dir"] / EVIDENCE_FILENAME).is_file()
    assert (ctx["output_dir"] / SOLVER_RESULT_FILENAME).is_file()
    assert (ctx["output_dir"] / SOLVER_CONFIG_FILENAME).is_file()
    assert (ctx["output_dir"] / DEBUG_JSON_FILENAME).is_file()
    assert (ctx["output_dir"] / DEBUG_HTML_FILENAME).is_file()
    assert not (tmp_path / "checkpoints.json").exists()
    assert _hash(ctx["attempts"]) == before[ctx["attempts"]]
    assert _hash(ctx["cache"]) == before[ctx["cache"]]
    _ = session_files_before


def test_cli_success_collision_and_input_codes(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    ctx = prepare(tmp_path)
    monkeypatch.setattr(
        "serve_review.media.probe.probe_source", lambda *a, **k: ctx["metadata"]
    )

    def _fake_iter(video, times, **kwargs):
        return [AudioEnergy(time_seconds=float(t), energy=0.01) for t in times]

    monkeypatch.setattr(
        "serve_review.media.audio.iter_audio_energy", _fake_iter
    )
    base = [
        "phase-debug-dev",
        str(ctx["video"]),
        "--attempts", str(ctx["attempts"]),
        "--cache", str(ctx["cache"]),
        "--annotations", str(ctx["annotations"]),
        "--attempt-id", ctx["attempt"].attempt_id,
        "--output-dir", str(ctx["output_dir"]),
    ]
    assert phase_debug_dev_cmd(build_parser().parse_args(base)) == 0
    capsys.readouterr()
    before_json = (ctx["output_dir"] / DEBUG_JSON_FILENAME).read_bytes()
    before_html = (ctx["output_dir"] / DEBUG_HTML_FILENAME).read_bytes()
    # Collision without overwrite returns 1 and preserves outputs.
    assert phase_debug_dev_cmd(build_parser().parse_args(base)) == 1
    err = capsys.readouterr().err
    assert "collision" in err.lower()
    assert (ctx["output_dir"] / DEBUG_JSON_FILENAME).read_bytes() == before_json
    assert (ctx["output_dir"] / DEBUG_HTML_FILENAME).read_bytes() == before_html
    # Overwrite returns 0 and is deterministic.
    assert (
        phase_debug_dev_cmd(build_parser().parse_args([*base, "--overwrite"])) == 0
    )
    capsys.readouterr()
    assert (ctx["output_dir"] / DEBUG_JSON_FILENAME).read_bytes() == before_json
    assert (ctx["output_dir"] / DEBUG_HTML_FILENAME).read_bytes() == before_html
    # Missing annotations file returns 2 with no new outputs.
    missing_out = tmp_path / "missing-out"
    args = build_parser().parse_args(
        [
            "phase-debug-dev",
            str(ctx["video"]),
            "--attempts", str(ctx["attempts"]),
            "--cache", str(ctx["cache"]),
            "--annotations", str(tmp_path / "nope.json"),
            "--attempt-id", ctx["attempt"].attempt_id,
            "--output-dir", str(missing_out),
        ]
    )
    assert phase_debug_dev_cmd(args) == 2
    assert not missing_out.exists() or not any(missing_out.iterdir())
    # Held-out manifest returns 2 and writes nothing new.
    heldout = make_manifest(ctx["attempt"], split="heldout")
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text(heldout.to_json(), encoding="utf-8")
    heldout_out = tmp_path / "heldout-out"
    args2 = build_parser().parse_args(
        [
            "phase-debug-dev",
            str(ctx["video"]),
            "--attempts", str(ctx["attempts"]),
            "--cache", str(ctx["cache"]),
            "--annotations", str(heldout_path),
            "--attempt-id", ctx["attempt"].attempt_id,
            "--output-dir", str(heldout_out),
        ]
    )
    assert phase_debug_dev_cmd(args2) == 2
    assert not heldout_out.exists() or not any(heldout_out.iterdir())
