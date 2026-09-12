"""Focused M4.5 tests for the analyze orchestration pipeline.

Fully fake/synthetic only: no private footage, no network, no model
inference. Pose caches are real JSONL files in temporary directories;
probe/audio/inference stages are injected fakes except where the real
M4.2/M4.3/M4.4 modules are exercised on synthetic rows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from serve_review.analysis_pipeline import (
    AnalyzeCancelled,
    AnalyzeError,
    default_attempts_path_for,
    derive_contact_candidate_times,
    run_analyze,
)
from serve_review.checkpoints.phase_solver import (
    PHASE_SOLVER_DEFAULT_CONFIG_ID,
    PHASE_SOLVER_METHOD_VERSION,
)
from serve_review.checkpoints.phase_features import PhaseFeaturesConfig
from serve_review.detection.decoder import DecoderConfig
from serve_review.domain import (
    STAGE_ORDER,
    Attempt,
    AttemptDocument,
    AttemptPhase,
    MediaRange,
    PhaseDocument,
    SourceMetadata,
    StagePhase,
)
from serve_review.media import audio as audio_module
from serve_review.media.audio import AudioEnergy
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose.mediapipe import MODEL_NAME, MODEL_VERSION
from serve_review.pose.schema import CacheIdentity, FrameObservation

FINGERPRINT = "sha256:analyze-fixture"


# --- Builders ---------------------------------------------------------------


def make_metadata(
    duration: float = 10.0, fingerprint: str = FINGERPRINT
) -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=duration,
        width=320,
        height=240,
        frame_rate_num=30,
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


def session_paths(video: Path, output_dir: Path) -> dict[str, Path]:
    session = output_dir / video.stem
    return {
        "session": session,
        "attempts": session / "attempts.json",
        "cache": extract_module.default_cache_path_for(video, output_dir),
        "checkpoints": session / "checkpoints.json",
    }


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
    """Build an audio fake emitting ``energies`` per scheduled time.

    Times absent from ``energies`` emit ``floor``. Returns the fake plus
    a record of calls (to assert single-demux behavior).
    """
    calls: list[dict] = []
    table = dict(energies or {})

    def _audio(video, schedule, **kwargs):
        calls.append({"video": Path(video), "count": len(tuple(schedule))})
        out = []
        for moment in schedule:
            # The dense schedule is a floating-point grid that rarely lands
            # exactly on a spike time; match by proximity instead of exact
            # equality so synthetic transients actually apply.
            energy = floor
            for spike_time, spike_energy in table.items():
                if abs(float(moment) - float(spike_time)) <= 0.003:
                    energy = spike_energy
                    break
            out.append(AudioEnergy(time_seconds=float(moment), energy=energy))
        return out

    _audio.calls = calls  # type: ignore[attr-defined]
    return _audio


def complete_phase_for(attempt: Attempt) -> AttemptPhase:
    """Build a fully available eight-stage phase inside the attempt."""
    start = attempt.detected_range.start_seconds
    end = attempt.detected_range.end_seconds
    width = (end - start) / 8.0
    stages: dict[str, StagePhase] = {}
    for position, key in enumerate(STAGE_ORDER):
        low = start + position * width
        high = low + width * 0.9
        keyframe = (low + high) / 2.0
        provenance = "audio_transient" if key == "contact" else "body_pose"
        stages[key] = StagePhase(
            availability="available",
            provenance=provenance,
            confidence=0.8,
            interval=MediaRange(start_seconds=low, end_seconds=high),
            keyframe_seconds=keyframe,
            temporal_uncertainty_seconds=0.02 if key == "contact" else None,
            evidence=("test_cue",),
            limitations=("test_only",),
        )
    return AttemptPhase(
        attempt_id=attempt.attempt_id,
        attempt_range=MediaRange(start_seconds=start, end_seconds=end),
        method_version=PHASE_SOLVER_METHOD_VERSION,
        config_id=PHASE_SOLVER_DEFAULT_CONFIG_ID,
        stages=stages,
        structural_status="complete",
        anomalies=(),
    )


def fake_solver_full():
    seen: dict = {}

    def _solver(attempt, grid, evidence, config):
        seen[attempt.attempt_id] = (grid, evidence, config)
        return complete_phase_for(attempt)

    _solver.seen = seen  # type: ignore[attr-defined]
    return _solver


def prepare(
    tmp_path: Path,
    *,
    spans: list[tuple[float, float]] = [(1.0, 2.0), (5.0, 6.0)],
    duration: float = 10.0,
    obs_step: float = 0.5,
    energies: dict[float, float] | None = None,
) -> dict:
    """Create video + attempts.json + complete cache + fakes."""
    metadata = make_metadata(duration=duration)
    video = make_source(tmp_path)
    output_dir = tmp_path / "output"
    paths = session_paths(video, output_dir)
    document = make_document(metadata, spans)
    write_attempts(paths["attempts"], document)
    moments: list[float] = []
    moment = 0.0
    while moment < duration:
        moments.append(round(moment, 9))
        moment += obs_step
    frames = [make_obs(moment) for moment in moments]
    write_cache(paths["cache"], expected_identity(metadata), frames)
    audio = fake_audio(energies)
    return {
        "metadata": metadata,
        "video": video,
        "output_dir": output_dir,
        "paths": paths,
        "document": document,
        "audio": audio,
        "frames": frames,
    }


# --- Full orchestration -------------------------------------------------------


def test_full_multi_attempt_orchestration_real_modules(tmp_path: Path) -> None:
    spike = {1.5: 0.30, 5.5: 0.25}
    ctx = prepare(tmp_path, energies=spike)
    before = hashlib.sha256(ctx["paths"]["attempts"].read_bytes()).hexdigest()
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
    )
    assert result.cache_hit is True
    assert result.frame_count == len(ctx["frames"])
    assert result.audio_sample_count > 0
    assert result.candidate_count >= 1
    assert len(ctx["audio"].calls) == 1  # demuxed exactly once
    assert result.checkpoints_path.is_file()
    reparsed = PhaseDocument.from_dict(
        json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    )
    assert reparsed.source_fingerprint == FINGERPRINT
    assert reparsed.source_duration_seconds == 10.0
    assert [phase.attempt_id for phase in reparsed] == ["serve-001", "serve-002"]
    for phase in reparsed:
        assert phase.method_version == PHASE_SOLVER_METHOD_VERSION
        assert set(phase.stages.keys()) == set(STAGE_ORDER)
    assert (
        hashlib.sha256(ctx["paths"]["attempts"].read_bytes()).hexdigest() == before
    )


def test_full_orchestration_with_available_phases(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    solver = fake_solver_full()
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=solver,
    )
    assert result.attempt_failures == ()
    assert len(result.phase_document) == 2
    for phase in result.phase_document:
        assert phase.structural_status == "complete"
        assert phase.config_id == PHASE_SOLVER_DEFAULT_CONFIG_ID


def test_explicit_attempts_path(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    explicit = tmp_path / "custom-attempts.json"
    explicit.write_text(ctx["document"].to_json(), encoding="utf-8")
    ctx["paths"]["attempts"].unlink()
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        attempts_path=explicit,
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=fake_solver_full(),
    )
    assert result.attempts_path == explicit
    assert result.checkpoints_path.is_file()


def test_discovered_attempts_path_is_deterministic(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    assert default_attempts_path_for(
        ctx["video"], ctx["output_dir"]
    ) == ctx["paths"]["attempts"]
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=fake_solver_full(),
    )
    assert result.attempts_path == ctx["paths"]["attempts"]


def test_missing_discovered_attempts_fails_actionably(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    ctx["paths"]["attempts"].unlink()
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "attempts"
    assert "cut" in excinfo.value.message
    assert not ctx["paths"]["checkpoints"].exists()


def test_explicit_attempts_missing_file_fails(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            attempts_path=tmp_path / "nope.json",
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "attempts"


def test_source_fingerprint_mismatch_fails(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    other = make_metadata(fingerprint="sha256:other-video")
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(other),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "attempts"
    assert "fingerprint" in excinfo.value.message
    assert not ctx["paths"]["checkpoints"].exists()


def test_source_duration_mismatch_fails(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    other = make_metadata(duration=12.0)
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(other),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "attempts"
    assert "duration" in excinfo.value.message


def test_malformed_attempts_variants(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    for bad in (
        "{not json",
        json.dumps([1, 2, 3]),
        json.dumps({"schema_version": 1}),
        json.dumps(
            {
                "attempts": [],
                "export_ranges": [],
                "method_version": "x",
                "padding_seconds": 0.0,
                "schema_version": 999,
                "source_duration_seconds": 10.0,
                "source_fingerprint": FINGERPRINT,
            }
        ),
    ):
        ctx["paths"]["attempts"].write_text(bad, encoding="utf-8")
        with pytest.raises(AnalyzeError) as excinfo:
            run_analyze(
                ctx["video"],
                output_dir=ctx["output_dir"],
                probe_fn=fake_probe(ctx["metadata"]),
                audio_fn=ctx["audio"],
            )
        assert excinfo.value.stage == "attempts"
    assert not ctx["paths"]["checkpoints"].exists()


# --- Cache reuse contract -----------------------------------------------------


def test_missing_cache_fails_actionably(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    ctx["paths"]["cache"].unlink()
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "pose"
    assert "extract-poses" in excinfo.value.message or "cut" in excinfo.value.message


def test_incomplete_cache_is_never_reused(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    write_cache(
        ctx["paths"]["cache"],
        expected_identity(ctx["metadata"]),
        ctx["frames"][:10],
        complete=False,
    )
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "pose"
    assert "incomplete" in excinfo.value.message


def test_stale_cache_fingerprint_fails(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    stale = CacheIdentity(
        source_fingerprint="sha256:other-video",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )
    write_cache(ctx["paths"]["cache"], stale, ctx["frames"])
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "pose"
    assert "stale" in excinfo.value.message


def test_stale_cache_model_fails(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    stale = CacheIdentity(
        source_fingerprint=FINGERPRINT,
        model_name=MODEL_NAME,
        model_version="heavy-deadbeefdeadbeef",
        sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )
    write_cache(ctx["paths"]["cache"], stale, ctx["frames"])
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "pose"
    assert "stale" in excinfo.value.message


def test_stale_cache_sampling_rate_fails(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    stale = CacheIdentity(
        source_fingerprint=FINGERPRINT,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=60.0,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )
    write_cache(ctx["paths"]["cache"], stale, ctx["frames"])
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "pose"


def test_corrupt_cache_fails_actionably(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    ctx["paths"]["cache"].write_text("{\"type\": \"header\"\n torn", encoding="utf-8")
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
        )
    assert excinfo.value.stage == "pose"
    assert "corrupt" in excinfo.value.message


# --- Audio alignment and relative transient policy ----------------------------


def test_audio_alignment_and_candidate_reuse_decoder_policy(tmp_path: Path) -> None:
    ctx = prepare(tmp_path, obs_step=1.0, energies={1.0: 0.30})
    seen: dict = {}

    def _grid(observations, attempt_range, **kwargs):
        seen[attempt_range.start_seconds] = dict(kwargs)
        from serve_review.checkpoints.phase_features import (
            build_phase_feature_grid as real,
        )

        return real(observations, attempt_range, **kwargs)

    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        grid_fn=_grid,
    )
    assert result.candidate_count >= 1
    first_kwargs = seen[1.0]
    aligned = first_kwargs["audio_energies"]
    candidates = first_kwargs["audio_candidate_times"]
    # Aligned raw audio echoes cached pose times (F_t contract).
    assert tuple(a.time_seconds for a in aligned) == tuple(
        f.time_seconds for f in ctx["frames"]
    )
    # The owning pose frame inherits the narrow spike via max-pooling.
    by_time = {a.time_seconds: a.energy for a in aligned}
    assert by_time[1.0] == pytest.approx(0.30)
    assert 1.0 in candidates
    # Candidates derive from the decoder ratio/floor policy, not an
    # absolute threshold: the quiet bed stays excluded.
    assert all(c in (1.0,) for c in candidates if c < 2.0)


def test_derive_contact_candidates_is_scale_invariant() -> None:
    base = [AudioEnergy(time_seconds=t * 0.1, energy=0.01) for t in range(20)]
    spiked = list(base)
    spiked[10] = AudioEnergy(time_seconds=1.0, energy=0.20)
    config = DecoderConfig()
    small = derive_contact_candidate_times(tuple(spiked), config)
    assert small == (1.0,)
    scaled = tuple(
        AudioEnergy(time_seconds=a.time_seconds, energy=a.energy * 10.0)
        for a in spiked
    )
    assert derive_contact_candidate_times(scaled, config) == small


def test_derive_contact_candidates_honest_quiet_and_floor() -> None:
    quiet = tuple(
        AudioEnergy(time_seconds=t * 0.1, energy=0.005) for t in range(10)
    )
    # Uniform bed below the near-silence floor yields no candidates even
    # though a ratio against the tiny median would otherwise fire.
    assert derive_contact_candidate_times(quiet, DecoderConfig()) == ()
    assert derive_contact_candidate_times((), DecoderConfig()) == ()


def test_decoder_config_ratio_controls_candidates(tmp_path: Path) -> None:
    ctx = prepare(tmp_path, energies={1.0: 0.30, 5.0: 0.25})
    strict = DecoderConfig(
        torso_displacement_threshold=0.2,
        acceleration_elbow_speed_threshold=1.5,
        rest_evidence_threshold=0.6,
        audio_transient_ratio=100.0,
        audio_transient_floor=0.01,
        audio_window_seconds=0.4,
        dropout_hysteresis_frames=4,
    )
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        decoder_config=strict,
        solver_fn=fake_solver_full(),
    )
    assert result.candidate_count == 0
    assert len(result.phase_document) == 2


# --- Partial / no-person / empty ----------------------------------------------


def test_no_person_frames_yield_honest_unavailable(tmp_path: Path) -> None:
    ctx = prepare(tmp_path, obs_step=2.0)
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
    )
    assert result.checkpoints_path.is_file()
    for phase in result.phase_document:
        assert phase.structural_status in (
            "partial",
            "incomplete",
            "unavailable",
        )


def test_empty_document_writes_honest_empty(tmp_path: Path) -> None:
    ctx = prepare(tmp_path, spans=[])
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
    )
    assert result.empty is True
    assert len(result.phase_document) == 0
    reparsed = PhaseDocument.from_dict(
        json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    )
    assert len(reparsed) == 0
    assert reparsed.source_fingerprint == FINGERPRINT


def test_unpadded_ranges_drive_phase_windows(tmp_path: Path) -> None:
    metadata = make_metadata()
    video = make_source(tmp_path)
    output_dir = tmp_path / "output"
    paths = session_paths(video, output_dir)
    detected = MediaRange(start_seconds=2.0, end_seconds=4.0)
    attempt = Attempt(
        attempt_id="serve-001",
        detected_range=detected,
        effective_range=MediaRange(start_seconds=1.0, end_seconds=5.0),
        confidence=0.7,
        evidence={},
    )
    document = AttemptDocument(
        source_fingerprint=metadata.fingerprint,
        source_duration_seconds=metadata.duration_seconds,
        padding_seconds=1.0,
        method_version="candidate-ranges-v1+plan-v1",
        attempts=(attempt,),
        export_ranges=(MediaRange(start_seconds=1.0, end_seconds=5.0),),
    )
    write_attempts(paths["attempts"], document)
    frames = [make_obs(round(t * 0.5, 9)) for t in range(20)]
    write_cache(paths["cache"], expected_identity(metadata), frames)
    seen_ranges: list[MediaRange] = []

    def _grid(observations, attempt_range, **kwargs):
        seen_ranges.append(attempt_range)
        from serve_review.checkpoints.phase_features import (
            build_phase_feature_grid as real,
        )

        return real(observations, attempt_range, **kwargs)

    run_analyze(
        video,
        output_dir=output_dir,
        probe_fn=fake_probe(metadata),
        audio_fn=fake_audio(),
        grid_fn=_grid,
    )
    assert seen_ranges == [detected]


def test_phase_error_isolation_per_attempt(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)

    def _solver(attempt, grid, evidence, config):
        if attempt.attempt_id == "serve-002":
            raise RuntimeError("synthetic solver boom")
        return complete_phase_for(attempt)

    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=_solver,
    )
    assert result.checkpoints_path.is_file()
    assert len(result.attempt_failures) == 1
    failure = result.attempt_failures[0]
    assert failure.attempt_id == "serve-002"
    assert failure.stage == "phase"
    by_id = {phase.attempt_id: phase for phase in result.phase_document}
    assert by_id["serve-001"].structural_status == "complete"
    failed = by_id["serve-002"]
    assert failed.structural_status == "unavailable"
    assert "phase_error" in failed.anomalies
    assert failed.method_version == PHASE_SOLVER_METHOD_VERSION


def test_evidence_failure_isolated_like_solver(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)

    def _evidence(grid, config):
        raise RuntimeError("synthetic evidence boom")

    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        evidence_fn=_evidence,
    )
    assert len(result.phase_document) == 2
    assert len(result.attempt_failures) == 2
    for phase in result.phase_document:
        assert phase.structural_status == "unavailable"


# --- Atomicity / collision / cancellation -------------------------------------


def test_collision_without_overwrite_preserves_existing(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    ctx["paths"]["checkpoints"].parent.mkdir(parents=True, exist_ok=True)
    ctx["paths"]["checkpoints"].write_bytes(b"{\"stale\": true}\n")
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
            solver_fn=fake_solver_full(),
        )
    assert excinfo.value.stage == "checkpoints"
    assert "collision" in excinfo.value.message
    assert ctx["paths"]["checkpoints"].read_bytes() == b"{\"stale\": true}\n"


def test_overwrite_replaces_existing(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    ctx["paths"]["checkpoints"].parent.mkdir(parents=True, exist_ok=True)
    ctx["paths"]["checkpoints"].write_bytes(b"{\"stale\": true}\n")
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        overwrite=True,
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=fake_solver_full(),
    )
    assert result.checkpoints_path.read_bytes() != b"{\"stale\": true}\n"
    PhaseDocument.from_dict(
        json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    )


def test_audio_failure_writes_no_partial(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)

    def _boom(video, schedule, **kwargs):
        raise audio_module.AudioError("synthetic demux failure")

    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=_boom,
        )
    assert excinfo.value.stage == "audio"
    assert not ctx["paths"]["checkpoints"].exists()
    leftovers = list(ctx["paths"]["session"].rglob("checkpoints.json.tmp-*"))
    assert leftovers == []


def test_cancellation_before_probe_writes_nothing(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    with pytest.raises(AnalyzeCancelled):
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
            is_cancelled=lambda: True,
        )
    assert not ctx["paths"]["checkpoints"].exists()


def test_cancellation_mid_phase_cleans_temp_siblings(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    stray = ctx["paths"]["session"] / "checkpoints.json.tmp-stray"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"partial")
    calls = {"count": 0}

    def _cancel():
        calls["count"] += 1
        return calls["count"] > 3

    with pytest.raises(AnalyzeCancelled):
        run_analyze(
            ctx["video"],
            output_dir=ctx["output_dir"],
            probe_fn=fake_probe(ctx["metadata"]),
            audio_fn=ctx["audio"],
            solver_fn=fake_solver_full(),
            is_cancelled=_cancel,
        )
    assert not ctx["paths"]["checkpoints"].exists()
    assert not stray.exists()


def test_solver_returning_wrong_type_is_isolated(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)

    def _bad_solver(attempt, grid, evidence, config):
        return {"not": "a phase"}

    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=_bad_solver,
    )
    assert len(result.attempt_failures) == 2
    for phase in result.phase_document:
        assert phase.structural_status == "unavailable"


# --- Determinism and immutability ----------------------------------------------


def test_deterministic_rerun_bytes_identical(tmp_path: Path) -> None:
    ctx = prepare(tmp_path, energies={1.5: 0.30})
    first = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=fake_audio({1.5: 0.30}),
    )
    payload_one = ctx["paths"]["checkpoints"].read_bytes()
    second = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        overwrite=True,
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=fake_audio({1.5: 0.30}),
    )
    payload_two = ctx["paths"]["checkpoints"].read_bytes()
    assert payload_one == payload_two
    assert first.phase_document.to_dict() == second.phase_document.to_dict()


def test_attempts_clips_compilation_bytes_unchanged(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    clips_dir = ctx["paths"]["session"] / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip = clips_dir / "serve-001.mov"
    clip.write_bytes(b"fake-clip-bytes")
    compilation = ctx["paths"]["session"] / "serves.mov"
    compilation.write_bytes(b"fake-compilation-bytes")
    digests_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ctx["paths"]["attempts"], clip, compilation)
    }
    video_before = hashlib.sha256(ctx["video"].read_bytes()).hexdigest()
    run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=fake_solver_full(),
    )
    for path, digest in digests_before.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert hashlib.sha256(ctx["video"].read_bytes()).hexdigest() == video_before


def test_result_carries_timing_and_method_identities(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    result = run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=fake_solver_full(),
    )
    assert result.stage_timings_seconds is not None
    for stage in ("probe", "attempts", "pose", "audio", "phase", "write"):
        assert stage in result.stage_timings_seconds
    for phase in result.phase_document:
        assert phase.method_version == PHASE_SOLVER_METHOD_VERSION
        assert phase.config_id == PHASE_SOLVER_DEFAULT_CONFIG_ID


def test_missing_video_fails_in_validate(tmp_path: Path) -> None:
    with pytest.raises(AnalyzeError) as excinfo:
        run_analyze(
            tmp_path / "missing.mov",
            output_dir=tmp_path / "output",
            probe_fn=fake_probe(make_metadata()),
        )
    assert excinfo.value.stage == "validate"


def test_progress_callback_receives_stage_messages(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    messages: list[str] = []
    run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        solver_fn=fake_solver_full(),
        progress_callback=messages.append,
    )
    joined = "\n".join(messages)
    assert "probing" in joined
    assert "checkpoints" in joined


def test_grid_config_identity_flows_to_features(tmp_path: Path) -> None:
    ctx = prepare(tmp_path)
    seen: dict = {}
    config = PhaseFeaturesConfig()

    def _grid(observations, attempt_range, **kwargs):
        seen["config"] = kwargs["config"]
        from serve_review.checkpoints.phase_features import (
            build_phase_feature_grid as real,
        )

        return real(observations, attempt_range, **kwargs)

    run_analyze(
        ctx["video"],
        output_dir=ctx["output_dir"],
        probe_fn=fake_probe(ctx["metadata"]),
        audio_fn=ctx["audio"],
        features_config=config,
        grid_fn=_grid,
    )
    assert seen["config"] is config
