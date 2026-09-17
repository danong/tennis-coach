"""Focused tests for the single-command 3D path with audio cue (analyze-serve).

Deterministic, offline, and independent of private footage, model
weights, and network access: source videos are tiny fake byte files in
temporary directories, probing/timestamps/inference/sampling/encoding
and audio are faked through the injectable ``run_analyze_serve`` hooks,
and the real M4.8 filter, M4.9 waveform, M4.10a candidate, and M4.10b DP
modules run for real over synthetic constant-geometry world poses.
Legacy sparse functions (phase features/evidence, sparse pose cache,
the old phase solver entry points) are monkeypatched to fail loudly so
any regression off the 3D path errors instead of passing silently.
Audio is faked through the ``audio_energies_fn`` hook (present versus
absent) so no FFmpeg/ffprobe subprocess runs here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_review.analyze_serve import (
    ANALYZE_SERVE_ATTEMPT_ID,
    AnalyzeServeError,
    run_analyze_serve,
)
from serve_review.checkpoints import six_anchor_solver as six_module
from serve_review.checkpoints.composite_anchors import (
    COMPOSITE_ANCHOR_STAGES,
    COMPOSITE_CUE_NAMES,
    CompositeAnchorCandidate,
)
from serve_review.checkpoints.phase_solver import PhaseSolverConfig
from serve_review.domain import (
    STAGE_ORDER,
    AttemptPhase,
    MediaRange,
    PhaseDocument,
    PhaseError,
    SourceMetadata,
    StagePhase,
)
from serve_review.media.frames import SampledFrame
from serve_review.pose.schema import JOINT_INDEX, JOINT_NAMES, FrameObservation
from serve_review.pose.world import WorldFrameObservation, WorldLandmark
from serve_review.tracking.racketvision import RacketVisionFrameObservation

DURATION = 2.0
N_FRAMES = 30
TIMES = tuple(i / 30.0 for i in range(N_FRAMES))

NAMED_GEOMETRY: dict[str, tuple[float, float, float]] = {
    "left_shoulder": (-0.2, 0.5, 0.0),
    "right_shoulder": (0.2, 0.5, 0.0),
    "left_elbow": (-0.3, 0.3, 0.0),
    "right_elbow": (0.3, 0.3, 0.05),
    "left_wrist": (-0.25, 0.1, 0.05),
    "right_wrist": (0.25, 0.62, 0.1),
    "left_hip": (-0.1, 0.0, 0.0),
    "right_hip": (0.1, 0.0, 0.0),
    "left_knee": (-0.11, -0.25, 0.02),
    "right_knee": (0.11, -0.25, -0.02),
    "left_ankle": (-0.12, -0.5, 0.0),
    "right_ankle": (0.12, -0.5, 0.0),
}


def _metadata() -> SourceMetadata:
    return SourceMetadata(
        fingerprint="sha256:test-serve",
        duration_seconds=DURATION,
        width=320,
        height=240,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
    )


def _world_observation(moment: float, *, missing: tuple[str, ...] = ()) -> WorldFrameObservation:
    joints: list[WorldLandmark | None] = []
    for index in range(33):
        name = JOINT_NAMES[index]
        if name in missing:
            joints.append(None)
        elif name in NAMED_GEOMETRY:
            x, y, z = NAMED_GEOMETRY[name]
            joints.append(WorldLandmark(x, y, z))
        else:
            joints.append(
                WorldLandmark(0.01 * index - 0.16, 0.05 + 0.01 * index, 0.005 * index)
            )
    companion = FrameObservation(
        time_seconds=moment,
        timestamp_ms=int(round(moment * 1000)),
        persons=(),
    )
    return WorldFrameObservation(
        time_seconds=moment,
        timestamp_ms=int(round(moment * 1000)),
        world_landmarks=tuple(joints),
        frame_2d=companion,
    )


class _FakeRacketVisionTracker:
    def __init__(self, config=None) -> None:
        self.config = config

    def track_frames(self, frames):
        return tuple(
            RacketVisionFrameObservation(frame.time_seconds, None, None)
            for frame in frames
        )


class _FakeBackend:
    """Synthetic world-pose backend with a call counter for cache tests."""

    def __init__(self, *, missing: tuple[str, ...] = ()) -> None:
        self.model_name = "test-world-backend"
        self.model_version = "test-v1"
        self.missing = missing
        self.calls = 0

    def infer_world(self, image: object, moment: float) -> WorldFrameObservation:
        self.calls += 1
        return _world_observation(float(moment), missing=self.missing)

    def close(self) -> None:
        return None


def _native_frames(remaining: tuple[float, ...]):
    for moment in remaining:
        yield SimpleNamespace(
            time_seconds=float(moment),
            image=np.zeros((16, 16, 3), dtype=np.uint8),
        )


@pytest.fixture(autouse=True)
def fake_racketvision(monkeypatch: pytest.MonkeyPatch) -> None:
    import serve_review.analyze_serve as analyze_module

    monkeypatch.setattr(
        analyze_module,
        "fingerprint_racketvision_config",
        lambda config: "sha256:test-racketvision",
    )
    monkeypatch.setattr(
        analyze_module,
        "iter_racketvision_model_frames",
        lambda video, times, **kwargs: _native_frames(tuple(times)),
    )
    monkeypatch.setattr(
        analyze_module,
        "RacketVisionTracker",
        _FakeRacketVisionTracker,
    )


@pytest.fixture
def no_legacy_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any legacy sparse/M3 function is invoked."""
    import serve_review.checkpoints.evidence as evidence_module
    import serve_review.checkpoints.phase_features as features_module
    import serve_review.checkpoints.phase_solver as solver_module
    import serve_review.pose.cache as cache_module
    import serve_review.pose.extract as extract_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "legacy sparse path must never run on analyze-serve"
        )

    monkeypatch.setattr(features_module, "build_phase_feature_grid", _boom)
    monkeypatch.setattr(evidence_module, "build_phase_evidence", _boom)
    monkeypatch.setattr(solver_module, "solve_attempt_phase", _boom)
    monkeypatch.setattr(solver_module, "solve_with_diagnostics", _boom)
    monkeypatch.setattr(cache_module, "load_cache", _boom)
    monkeypatch.setattr(extract_module, "extract_poses", _boom)


def _present_audio_energies(video_path: Path, frame_times: tuple[float, ...]):
    """Synthetic present-audio hook: quiet bed with one impact peak.

    The spike sits at index 5, the DP-selected contact for the constant
    synthetic geometry, so the selected contact carries a strictly
    positive audio cue (body_pose_audio); a spike elsewhere would lose
    the DP tie-break to motion peaks and correctly stay body_pose.
    """
    times = list(frame_times)
    peak = 5 if len(times) > 6 else len(times) // 2
    out: list[float] = []
    for index in range(len(times)):
        out.append(0.9 if index == peak else 0.01)
    return out


def _absent_audio_energies(video_path: Path, frame_times: tuple[float, ...]):
    """Synthetic absent-audio hook: no audio stream."""
    return None


def _failing_audio_energies(video_path: Path, frame_times: tuple[float, ...]):
    """Synthetic demux-failure hook: must fall back to unavailable."""
    from serve_review.media.audio import AudioError

    raise AudioError("synthetic demux failure")


def _run(
    tmp_path: Path,
    *,
    missing: tuple[str, ...] = (),
    start: float | None = None,
    end: float | None = None,
    force: bool = False,
    backend_holder: list | None = None,
    sampled_holder: list | None = None,
    encoded_holder: list | None = None,
    audio: str = "absent",
):
    from serve_review.analyze_serve import run_analyze_serve as _run_fn

    video = tmp_path / "serve.mov"
    video.write_bytes(b"fake-video-bytes")
    backend = _FakeBackend(missing=missing)
    if backend_holder is not None:
        backend_holder.append(backend)

    def _sample(video_path: Path, moment: float, metadata: SourceMetadata):
        if sampled_holder is not None:
            sampled_holder.append(float(moment))
        return SampledFrame(
            time_seconds=float(moment),
            timestamp_ms=int(round(float(moment) * 1000)),
            width=16,
            height=16,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
        )

    def _encode(image: np.ndarray, dest: Path) -> None:
        if encoded_holder is not None:
            encoded_holder.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-jpeg-bytes")

    if audio == "present":
        audio_hook = _present_audio_energies
    elif audio == "failing":
        audio_hook = _failing_audio_energies
    else:
        audio_hook = _absent_audio_energies

    result = _run_fn(
        video,
        start_seconds=start,
        end_seconds=end,
        force=force,
        probe_fn=lambda path: _metadata(),
        native_times_fn=lambda _v, s, e: tuple(
            t for t in TIMES if t >= s - 1e-9 and t < e
        ),
        native_frame_factory=lambda remaining, _meta: _native_frames(
            tuple(remaining)
        ),
        backend_factory=lambda: backend,
        sample_frame_fn=_sample,
        encode_jpeg_fn=_encode,
        audio_energies_fn=audio_hook,
    )
    return result, backend


def _available_keyframes(phase: AttemptPhase) -> dict[str, float]:
    return {
        stage: float(phase.stages[stage].keyframe_seconds)  # type: ignore[arg-type]
        for stage in STAGE_ORDER
        if phase.stages[stage].availability == "available"
    }


# --- CLI contract ------------------------------------------------------------


def test_cli_defaults() -> None:
    from serve_review.cli import build_parser

    args = build_parser().parse_args(["analyze-serve", "session.mov"])
    assert args.video == Path("session.mov")
    assert args.start_seconds is None
    assert args.end_seconds is None
    assert args.output_dir is None
    assert args.cache is None
    assert args.model == Path("models/pose_landmarker_heavy.task")
    assert args.dry_run is False
    assert args.force is False
    assert args.ffmpeg == "ffmpeg"
    assert args.ffprobe == "ffprobe"


def test_cli_rejects_missing_video(tmp_path: Path, capsys) -> None:
    from serve_review.cli import analyze_serve, build_parser

    args = build_parser().parse_args(["analyze-serve", str(tmp_path / "missing.mov")])
    assert analyze_serve(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_cli_maps_validate_to_exit_2(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.analyze_serve import AnalyzeServeError
    from serve_review.cli import analyze_serve, build_parser
    import serve_review.analyze_serve as serve_module

    source = tmp_path / "serve.mov"
    source.write_bytes(b"fake")

    def _boom(*args: object, **kwargs: object) -> None:
        raise AnalyzeServeError("validate", "bad range")

    monkeypatch.setattr(serve_module, "run_analyze_serve", _boom)
    args = build_parser().parse_args(["analyze-serve", str(source)])
    assert analyze_serve(args) == 2


def test_cli_reports_outputs_on_success(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import analyze_serve, build_parser
    import serve_review.analyze_serve as serve_module

    source = tmp_path / "serve.mov"
    source.write_bytes(b"fake")
    checkpoints = tmp_path / "output" / "serve" / "checkpoints.json"
    diagnostics = tmp_path / "output" / "serve" / "serve-3d-diagnostics.json"
    index = tmp_path / "output" / "serve" / "review-serve-3d" / "index.html"
    fake = SimpleNamespace(
        checkpoints_path=checkpoints,
        diagnostics_path=diagnostics,
        index_html=index,
        attempt_phase=SimpleNamespace(
            attempt_id="serve-001", structural_status="partial"
        ),
        attempt_range=SimpleNamespace(start_seconds=0.0, end_seconds=1.0),
        image_count=4,
    )
    monkeypatch.setattr(
        serve_module, "run_analyze_serve", lambda *a, **k: fake
    )
    args = build_parser().parse_args(
        ["analyze-serve", str(source), "--output-dir", str(tmp_path / "output")]
    )
    assert analyze_serve(args) == 0
    out = capsys.readouterr().out
    assert str(checkpoints) in out
    assert str(diagnostics) in out
    assert str(index) in out


# --- range validation --------------------------------------------------------


@pytest.mark.parametrize(
    "start,end",
    [(-0.5, None), (0.0, 0.0), (0.8, 0.4), (0.0, DURATION + 1.0), (DURATION, None)],
)
def test_range_validation_rejects_bad_ranges(
    tmp_path: Path, start: float | None, end: float | None
) -> None:
    from serve_review.analyze_serve import AnalyzeServeError

    video = tmp_path / "serve.mov"
    video.write_bytes(b"fake-video-bytes")
    with pytest.raises(AnalyzeServeError) as excinfo:
        run_analyze_serve(
            video,
            start_seconds=start,
            end_seconds=end,
            probe_fn=lambda path: _metadata(),
        )
    assert excinfo.value.stage == "validate"


def test_racketvision_diagnostics_are_added_without_changing_checkpoints(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    result, _ = _run(tmp_path)

    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert "visual_evidence" in diagnostics
    visual = diagnostics["visual_evidence"]
    assert visual["release"]["selected_time_seconds"] == diagnostics["selected"]["release"]["time_seconds"]
    assert visual["contact"]["selected_time_seconds"] == diagnostics["selected"]["contact"]["time_seconds"]
    assert visual["release"]["ball_to_left_wrist"]["available"] is False
    assert visual["contact"]["ball_to_racket_hoop"]["available"] is False


def test_default_range_covers_entire_source(tmp_path: Path, no_legacy_sparse: None) -> None:
    result, _ = _run(tmp_path)
    assert result.requested_range == MediaRange(0.0, DURATION)
    assert result.attempt_range == MediaRange(0.0, DURATION)
    assert result.frame_count == N_FRAMES


def test_explicit_range_flows_to_outputs(tmp_path: Path, no_legacy_sparse: None) -> None:
    result, _ = _run(tmp_path, start=0.0, end=1.0)
    assert result.requested_range == MediaRange(0.0, 1.0)
    assert result.frame_count == N_FRAMES
    payload = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert payload["requested_range"] == {"end_seconds": 1.0, "schema_version": 1, "start_seconds": 0.0}


# --- 3D orchestration with audio cue, cache reuse, artifacts -----------------


def test_absent_audio_fallback_uses_body_pose(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    holders: list = []
    result, backend = _run(tmp_path, backend_holder=holders, audio="absent")
    assert result.cache_hit is False
    assert result.racketvision_cache_hit is False
    assert result.racketvision_cache_path.is_file()
    assert backend.calls == N_FRAMES

    # Cache reuse: drop outputs but keep the cache; the second run
    # performs no inference.
    result.checkpoints_path.unlink()
    result.diagnostics_path.unlink()
    import shutil as _shutil

    _shutil.rmtree(result.review_dir)
    result2, backend2 = _run(tmp_path, audio="absent")
    assert result2.cache_hit is True
    assert result2.racketvision_cache_hit is True
    assert backend2.calls == 0
    assert result2.frame_count == N_FRAMES

    # Checkpoints: normal compatible document for synthetic serve-001.
    payload = json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    document = PhaseDocument.from_dict(payload)
    assert len(document) == 1
    phase = document[0]
    assert phase.attempt_id == ANALYZE_SERVE_ATTEMPT_ID
    assert set(phase.stages.keys()) == set(STAGE_ORDER)
    assert document.source_fingerprint == "sha256:test-serve"

    # Absent-audio fallback: body_pose contact with no deprecated tokens.
    contact = phase.stages["contact"]
    assert contact.availability == "available"
    assert contact.provenance == "body_pose"
    assert "audio_transient" not in contact.evidence
    assert "contact_not_directly_observed" not in contact.limitations
    assert "audio_transient_not_used" not in contact.limitations

    # Six-anchor chronology plus exact derived midpoints.
    keyframes = _available_keyframes(phase)
    assert len(keyframes) == len(STAGE_ORDER)
    ordered = [keyframes[stage] for stage in STAGE_ORDER]
    for earlier, later in zip(ordered, ordered[1:]):
        assert later > earlier
    assert keyframes["acceleration"] == (
        keyframes["cocking"] + keyframes["contact"]
    ) / 2.0
    assert keyframes["deceleration"] == (
        keyframes["contact"] + keyframes["finish"]
    ) / 2.0

    # Diagnostic JSON: exact inputs, path/skips, identity, PTS.
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["schema_version"] == 1
    assert diagnostics["attempt_id"] == ANALYZE_SERVE_ATTEMPT_ID
    assert diagnostics["audio"] == "unavailable"
    assert diagnostics["coordinate_2d"] == "not_used"
    assert diagnostics["pts_seconds"] == list(TIMES)
    assert diagnostics["model"] == {"name": "test-world-backend", "version": "test-v1"}
    for stage in COMPOSITE_ANCHOR_STAGES:
        entry = diagnostics["candidates"][stage]
        assert entry["total"] == N_FRAMES
        assert entry["eligible"] == N_FRAMES
        selected = diagnostics["selected"][stage]
        assert selected["status"] == "selected"
        assert selected["coverage"] is not None and selected["coverage"] > 0.0
        assert set(selected["cue_values"].keys()) == set(
            COMPOSITE_CUE_NAMES[stage]
        )
    assert diagnostics["selected"]["contact"]["cue_values"]["audio_transient"] is None
    assert diagnostics["derived"]["acceleration"]["status"] == "selected"
    assert diagnostics["derived"]["deceleration"]["status"] == "selected"

    # Review: JPEG per available stage plus self-contained index.
    assert result.image_count == len(STAGE_ORDER)
    assert result.index_html.is_file()
    html = result.index_html.read_text(encoding="utf-8")
    assert "Contact is a body-pose estimate and is never claimed" not in html
    assert "contact_not_directly_observed" not in html
    assert "audio_transient_not_used" not in html
    review = json.loads(result.review_json.read_text(encoding="utf-8"))
    assert len(review["entries"]) == len(STAGE_ORDER)
    for entry in review["entries"]:
        assert entry["status"] == "rendered"
        assert entry["image"] is not None
        assert (result.review_dir / entry["image"]).is_file()
        # PTS rendering linkage: requested keyframe versus sampled frame.
        assert entry["requested_source_time"] == keyframes[entry["stage"]]
        assert entry["actual_source_time"] == entry["requested_source_time"]


def test_present_audio_uses_body_pose_audio(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    result, _ = _run(tmp_path, audio="present")
    payload = json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    phase = PhaseDocument.from_dict(payload)[0]
    contact = phase.stages["contact"]
    assert contact.availability == "available"
    assert contact.provenance == "body_pose_audio"
    assert "audio_transient" in contact.evidence
    assert "contact_not_directly_observed" not in contact.limitations
    assert "audio_transient_not_used" not in contact.limitations
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["audio"] == "present"
    selected = diagnostics["selected"]["contact"]
    assert selected["status"] == "selected"
    assert selected["cue_values"]["audio_transient"] is not None
    keyframes = _available_keyframes(phase)
    ordered = [keyframes[stage] for stage in STAGE_ORDER]
    for earlier, later in zip(ordered, ordered[1:]):
        assert later > earlier


def test_audio_failure_falls_back_to_unavailable(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    result, _ = _run(tmp_path, audio="failing")
    payload = json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    phase = PhaseDocument.from_dict(payload)[0]
    contact = phase.stages["contact"]
    assert contact.availability == "available"
    assert contact.provenance == "body_pose"
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["audio"] == "unavailable"
    assert diagnostics["selected"]["contact"]["cue_values"]["audio_transient"] is None


def test_old_audio_free_wording_is_absent(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    for mode in ("absent", "present"):
        target = tmp_path / mode
        target.mkdir()
        result, _ = _run(target, audio=mode)
        checkpoints = result.checkpoints_path.read_text(encoding="utf-8")
        diagnostics = result.diagnostics_path.read_text(encoding="utf-8")
        html = result.index_html.read_text(encoding="utf-8")
        for blob in (checkpoints, diagnostics, html):
            assert "audio_transient_not_used" not in blob
            assert "contact_not_directly_observed" not in blob
        assert '"audio": "not_used"' not in diagnostics
        assert diagnostics and '"audio": "present"' in diagnostics or '"audio": "unavailable"' in diagnostics
        assert "Contact is a body-pose estimate and is never claimed" not in html
        assert "no audio transient\nwas used" not in html.lower()


def test_missing_wrist_support_skips_honestly(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    result, _ = _run(tmp_path, missing=("right_wrist",))
    payload = json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    phase = PhaseDocument.from_dict(payload)[0]
    # Finish loses every cue family (all three cues need right-wrist
    # waveform support) so it is honestly skipped, and deceleration
    # follows its missing bounding anchor. Contact stays selectable on
    # torso-rise support alone, with honestly low coverage.
    for stage in ("finish", "deceleration"):
        assert phase.stages[stage].availability == "unavailable"
        assert phase.stages[stage].keyframe_seconds is None
    for stage in ("start", "release", "loading", "cocking", "contact", "acceleration"):
        assert phase.stages[stage].availability == "available"
    assert phase.structural_status == "partial"

    review = json.loads(result.review_json.read_text(encoding="utf-8"))
    rendered = [e for e in review["entries"] if e["status"] == "rendered"]
    missing_entries = [e for e in review["entries"] if e["status"] != "rendered"]
    assert {e["stage"] for e in rendered} == {
        "start",
        "release",
        "loading",
        "cocking",
        "contact",
        "acceleration",
    }
    assert {e["stage"] for e in missing_entries} == {"finish", "deceleration"}
    for entry in missing_entries:
        assert entry["image"] is None
        assert entry["actual_source_time"] is None
    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["candidates"]["finish"]["eligible"] == 0
    assert diagnostics["selected"]["finish"]["status"] == "skipped"
    assert diagnostics["derived"]["deceleration"]["status"] == "skipped"
    contact_record = diagnostics["selected"]["contact"]
    assert contact_record["status"] == "selected"
    assert contact_record["coverage"] == pytest.approx(0.10)


def test_outputs_are_deterministic(tmp_path: Path, no_legacy_sparse: None) -> None:
    first, _ = _run(tmp_path)
    before_checkpoints = first.checkpoints_path.read_bytes()
    before_diagnostics = first.diagnostics_path.read_bytes()
    second, _ = _run(tmp_path, force=True)
    assert second.checkpoints_path.read_bytes() == before_checkpoints
    assert second.diagnostics_path.read_bytes() == before_diagnostics


def test_outputs_collide_without_overwrite(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    from serve_review.analyze_serve import AnalyzeServeError

    _run(tmp_path)
    video = tmp_path / "serve.mov"
    with pytest.raises(AnalyzeServeError) as excinfo:
        run_analyze_serve(
            video,
            probe_fn=lambda path: _metadata(),
            native_times_fn=lambda _v, s, e: tuple(
                t for t in TIMES if t >= s - 1e-9 and t < e
            ),
            native_frame_factory=lambda remaining, _meta: _native_frames(
                tuple(remaining)
            ),
            backend_factory=_FakeBackend,
            sample_frame_fn=lambda _v, m, _meta: SampledFrame(
                time_seconds=float(m),
                timestamp_ms=int(round(float(m) * 1000)),
                width=16,
                height=16,
                image=np.zeros((16, 16, 3), dtype=np.uint8),
            ),
            encode_jpeg_fn=lambda image, dest: dest.write_bytes(b"x"),
            audio_energies_fn=_absent_audio_energies,
        )
    assert excinfo.value.stage in ("write", "review")


# --- six-anchor DP semantics -------------------------------------------------


def _direct_candidate(
    stage: str, moment: float, *, score: float = 0.5, coverage: float = 1.0
) -> CompositeAnchorCandidate:
    cues = {cue: 0.5 for cue in COMPOSITE_CUE_NAMES[stage]}
    return CompositeAnchorCandidate(
        stage=stage,
        time_seconds=moment,
        timestamp_ms=int(round(moment * 1000)),
        score=score,
        coverage=coverage,
        cue_values=cues,
        provenance="kinematic_waveform",
        temporal_uncertainty_seconds=0.002,
        method_version="composite-anchors-v4",
        config_id="composite-anchors-default-v5",
        schema_version=4,
    )


def test_contact_to_finish_cap_only_limits_that_stage_pair() -> None:
    cfg = PhaseSolverConfig(max_contact_to_finish_seconds=0.8)
    contact = _direct_candidate("contact", 4.0)
    finish_at_cap = _direct_candidate("finish", 4.8)
    finish_after_cap = _direct_candidate("finish", 4.800001)
    assert six_module.transition_feasible(contact, finish_at_cap, 4, 5, cfg)
    assert not six_module.transition_feasible(contact, finish_after_cap, 4, 5, cfg)
    # The same temporal gap remains legal for another adjacent six-anchor pair.
    assert six_module.transition_feasible(
        _direct_candidate("cocking", 4.0),
        _direct_candidate("contact", 4.800001),
        3,
        4,
        cfg,
    )


def test_dp_skips_empty_stage_and_stays_chronological() -> None:
    grid = [i * 0.1 for i in range(5)]
    candidates = {
        stage: (
            [] if stage == "contact" else [_direct_candidate(stage, t) for t in grid]
        )
        for stage in six_module.SIX_ANCHOR_STAGES
    }
    first = six_module.solve_six_anchors(candidates, PhaseSolverConfig())
    second = six_module.solve_six_anchors(candidates, PhaseSolverConfig())
    assert first.choice["contact"] is None
    assert first.total_score == second.total_score
    assert dict(first.choice) == dict(second.choice)
    picked = [
        (stage, candidates[stage][first.choice[stage]].time_seconds)  # type: ignore[index]
        for stage in six_module.SIX_ANCHOR_STAGES
        if first.choice[stage] is not None
    ]
    times = [t for _, t in picked]
    for earlier, later in zip(times, times[1:]):
        assert later > earlier


# --- domain exception --------------------------------------------------------


def _stage(
    start: float,
    end: float,
    *,
    provenance: str = "body_pose",
    keyframe: float | None = 15.5,
    uncertainty: float | None = 0.05,
    limitations: tuple[str, ...] = (),
) -> StagePhase:
    return StagePhase(
        availability="available",
        provenance=provenance,
        confidence=0.7,
        interval=MediaRange(start, end),
        keyframe_seconds=keyframe,
        temporal_uncertainty_seconds=uncertainty,
        evidence=("wrist_elevation",),
        limitations=limitations,
    )


def _attempt_with_contact(contact: StagePhase) -> AttemptPhase:
    stages = {
        "start": _stage(10.0, 11.0, keyframe=10.5),
        "release": _stage(11.0, 12.0, keyframe=11.5),
        "loading": _stage(12.0, 13.0, keyframe=12.5),
        "cocking": _stage(13.0, 14.0, keyframe=13.5),
        "acceleration": _stage(14.0, 15.0, keyframe=14.5),
        "contact": contact,
        "deceleration": _stage(16.0, 17.0, keyframe=16.5),
        "finish": _stage(17.0, 18.0, keyframe=17.5),
    }
    return AttemptPhase(
        attempt_id="serve-001",
        attempt_range=MediaRange(10.0, 18.0),
        method_version="serve-waveform-v1",
        config_id="phase-solver-default-v1",
        stages=stages,
        structural_status="complete",
        anomalies=(),
    )


def test_body_pose_contact_is_allowed_without_legacy_tokens() -> None:
    contact = _stage(15.0, 16.0, limitations=())
    assert _attempt_with_contact(contact).stage("contact").provenance == "body_pose"


def test_body_pose_audio_contact_is_allowed() -> None:
    contact = _stage(15.0, 16.0, provenance="body_pose_audio", limitations=())
    assert (
        _attempt_with_contact(contact).stage("contact").provenance
        == "body_pose_audio"
    )


def test_contact_requires_anchor_and_uncertainty() -> None:
    for provenance in ("body_pose", "body_pose_audio"):
        with pytest.raises(PhaseError):
            _attempt_with_contact(
                _stage(15.0, 16.0, provenance=provenance, keyframe=None)
            )
        with pytest.raises(PhaseError):
            _attempt_with_contact(
                _stage(15.0, 16.0, provenance=provenance, uncertainty=None)
            )


# --- anchor2 manual comparison (fixed minimal JSON + CLI flag) --------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHOR2_RELPATH = Path("refs/annotations/dev/single-serve-02.anchor2.json")

ANCHOR2_EXPECTED = {
    "start": 2.75,
    "release": 3.13,
    "loading": 3.51,
    "cocking": 3.85,
    "contact": 3.98,
    "finish": 4.58,
    "acceleration": 3.915,
    "deceleration": 4.28,
}

# Synthetic in-range manual PTS for tmp-local anchor2 runs (duration 2.0s).
SYNTH_ANCHOR2 = {
    "start": 0.20,
    "release": 0.40,
    "loading": 0.60,
    "cocking": 0.80,
    "contact": 1.00,
    "finish": 1.50,
    "acceleration": 0.90,
    "deceleration": 1.25,
}


def test_anchor2_file_content_and_midpoints() -> None:
    path = REPO_ROOT / ANCHOR2_RELPATH
    if not path.is_file():
        pytest.skip("the deliberately ignored local anchor2 labels are unavailable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == ANCHOR2_EXPECTED
    assert set(payload.keys()) == set(STAGE_ORDER)
    for forbidden in (
        "schema_version",
        "schema",
        "version",
        "fingerprint",
        "source_fingerprint",
        "hash",
        "source_hash",
        "range",
        "interval",
        "start_seconds",
        "end_seconds",
    ):
        assert forbidden not in payload
    assert payload["acceleration"] == pytest.approx(
        (payload["cocking"] + payload["contact"]) / 2.0
    )
    assert payload["deceleration"] == pytest.approx(
        (payload["contact"] + payload["finish"]) / 2.0
    )


def test_cli_anchor2_flag_parsing() -> None:
    from serve_review.cli import build_parser

    default = build_parser().parse_args(["analyze-serve", "session.mov"])
    assert default.anchor2comparison is False
    long = build_parser().parse_args(
        ["analyze-serve", "session.mov", "--anchor2comparison"]
    )
    assert long.anchor2comparison is True
    aliased = build_parser().parse_args(
        ["analyze-serve", "session.mov", "--anchor2-comparison"]
    )
    assert aliased.anchor2comparison is True


def test_anchor2_rejects_other_basename(tmp_path: Path) -> None:
    from serve_review.analyze_serve import AnalyzeServeError, run_analyze_serve

    video = tmp_path / "other.mov"
    video.write_bytes(b"fake-video-bytes")
    with pytest.raises(AnalyzeServeError) as excinfo:
        run_analyze_serve(
            video,
            probe_fn=lambda path: _metadata(),
            anchor2comparison=True,
        )
    assert excinfo.value.stage == "validate"
    assert "single-serve-02.mov" in str(excinfo.value)


def _write_tmp_anchor2(tmp_path: Path, payload: dict) -> None:
    dest = tmp_path / ANCHOR2_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _run_anchor2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    *,
    missing: tuple[str, ...] = (),
    no_legacy_sparse: None = None,
):
    from serve_review.analyze_serve import run_analyze_serve as _run_fn

    _write_tmp_anchor2(tmp_path, payload)
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "single-serve-02.mov"
    video.write_bytes(b"fake-video-bytes")
    backend = _FakeBackend(missing=missing)
    sampled: list[float] = []
    encoded: list[Path] = []

    def _sample(video_path: Path, moment: float, metadata: SourceMetadata):
        sampled.append(float(moment))
        return SampledFrame(
            time_seconds=float(moment),
            timestamp_ms=int(round(float(moment) * 1000)),
            width=16,
            height=16,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
        )

    def _encode(image: np.ndarray, dest: Path) -> None:
        encoded.append(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-jpeg-bytes")

    result = _run_fn(
        video,
        probe_fn=lambda path: _metadata(),
        native_times_fn=lambda _v, s, e: tuple(t for t in TIMES if t >= s - 1e-9 and t < e),
        native_frame_factory=lambda remaining, _meta: _native_frames(tuple(remaining)),
        backend_factory=lambda: backend,
        sample_frame_fn=_sample,
        encode_jpeg_fn=_encode,
        audio_energies_fn=_absent_audio_energies,
        anchor2comparison=True,
    )
    return result, sampled, encoded


def test_anchor2_manual_images_pts_and_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_legacy_sparse: None
) -> None:
    result, sampled, _ = _run_anchor2(
        tmp_path, monkeypatch, dict(SYNTH_ANCHOR2), no_legacy_sparse=no_legacy_sparse
    )
    keyframes = _available_keyframes(result.attempt_phase)
    review = json.loads(result.review_json.read_text(encoding="utf-8"))
    assert review.get("anchor2comparison") is True
    assert review.get("anchor2_source") == str(ANCHOR2_RELPATH)
    assert len(review["entries"]) == len(STAGE_ORDER)
    for entry in review["entries"]:
        stage = entry["stage"]
        manual_requested = SYNTH_ANCHOR2[stage]
        assert entry["manual_requested_source_time"] == pytest.approx(manual_requested)
        # Native sampler actual PTS is retained verbatim (fake sampler echoes).
        assert entry["manual_actual_source_time"] == pytest.approx(manual_requested)
        selected = keyframes.get(stage)
        assert selected is not None
        assert entry["requested_source_time"] == pytest.approx(selected)
        assert entry["delta_selected_minus_manual_ms"] == pytest.approx(
            (selected - manual_requested) * 1000.0
        )
        manual_rel = entry["manual_image"]
        assert manual_rel == f"manual-{stage}.jpg"
        assert (result.review_dir / manual_rel).is_file()
        assert (result.review_dir / f"{stage}.jpg").is_file()
        assert float(manual_requested) in sampled
    html = result.index_html.read_text(encoding="utf-8")
    assert "manual requested s" in html
    assert "delta selected-minus-manual ms" in html
    assert "manual-start.jpg" in html


def test_anchor2_absent_sides_are_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_legacy_sparse: None
) -> None:
    # Pipeline side absent: missing right wrist drops finish/deceleration.
    result, _, _ = _run_anchor2(
        tmp_path, monkeypatch, dict(SYNTH_ANCHOR2),
        missing=("right_wrist",), no_legacy_sparse=no_legacy_sparse,
    )
    review = json.loads(result.review_json.read_text(encoding="utf-8"))
    by_stage = {entry["stage"]: entry for entry in review["entries"]}
    for stage in ("finish", "deceleration"):
        entry = by_stage[stage]
        assert entry["requested_source_time"] is None
        assert entry["actual_source_time"] is None
        assert entry["manual_requested_source_time"] == pytest.approx(SYNTH_ANCHOR2[stage])
        assert entry["manual_actual_source_time"] == pytest.approx(SYNTH_ANCHOR2[stage])
        assert entry["delta_selected_minus_manual_ms"] is None
        assert entry["manual_image"] == f"manual-{stage}.jpg"
        assert (result.review_dir / f"manual-{stage}.jpg").is_file()
    # Manual side absent: drop finish from the tmp manual JSON.
    tmp_path2 = tmp_path / "second"
    tmp_path2.mkdir()
    payload = {k: v for k, v in SYNTH_ANCHOR2.items() if k != "finish"}
    _write_tmp_anchor2(tmp_path2, payload)
    monkeypatch.chdir(tmp_path2)
    from serve_review.analyze_serve import run_analyze_serve as _run_fn

    video = tmp_path2 / "single-serve-02.mov"
    video.write_bytes(b"fake-video-bytes")
    backend = _FakeBackend()

    def _sample(video_path: Path, moment: float, metadata: SourceMetadata):
        return SampledFrame(
            time_seconds=float(moment),
            timestamp_ms=int(round(float(moment) * 1000)),
            width=16,
            height=16,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
        )

    def _encode(image: np.ndarray, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-jpeg-bytes")

    result2 = _run_fn(
        video,
        output_dir=tmp_path2 / "output",
        probe_fn=lambda path: _metadata(),
        native_times_fn=lambda _v, s, e: tuple(t for t in TIMES if t >= s - 1e-9 and t < e),
        native_frame_factory=lambda remaining, _meta: _native_frames(tuple(remaining)),
        backend_factory=lambda: backend,
        sample_frame_fn=_sample,
        encode_jpeg_fn=_encode,
        audio_energies_fn=_absent_audio_energies,
        anchor2comparison=True,
    )
    review2 = json.loads(result2.review_json.read_text(encoding="utf-8"))
    by_stage2 = {entry["stage"]: entry for entry in review2["entries"]}
    assert by_stage2["finish"]["manual_requested_source_time"] is None
    assert by_stage2["finish"]["manual_actual_source_time"] is None
    assert by_stage2["finish"]["manual_image"] is None
    assert by_stage2["finish"]["delta_selected_minus_manual_ms"] is None
    assert not (result2.review_dir / "manual-finish.jpg").exists()


def test_absent_flag_leaves_default_artifacts(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    result, _ = _run(tmp_path)
    review = json.loads(result.review_json.read_text(encoding="utf-8"))
    assert "anchor2comparison" not in review
    assert "anchor2_source" not in review
    for entry in review["entries"]:
        assert "manual_requested_source_time" not in entry
        assert "manual_actual_source_time" not in entry
        assert "manual_image" not in entry
        assert "delta_selected_minus_manual_ms" not in entry
    assert list(result.review_dir.glob("manual-*.jpg")) == []
    html = result.index_html.read_text(encoding="utf-8")
    assert "manual" not in html.lower()


# --- Shared transient policy supplies explicit waveform flags ---


def _run_with_audio_hook(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audio_hook,
    *,
    no_legacy_sparse: None = None,
    capture: dict | None = None,
):
    from serve_review.analyze_serve import run_analyze_serve as _run_fn
    from serve_review.checkpoints import kinematic_waveforms as kw_module

    if capture is not None:
        real_build = kw_module.build_kinematic_waveform_track

        def _spy(track, config=None, audio_energies=None, audio=None, **kwargs):
            capture["audio_energies"] = audio_energies
            capture["audio"] = audio
            return real_build(
                track, config, audio_energies=audio_energies, audio=audio
            )

        monkeypatch.setattr(kw_module, "build_kinematic_waveform_track", _spy)

    video = tmp_path / "serve.mov"
    video.write_bytes(b"fake-video-bytes")
    backend = _FakeBackend()

    def _sample(video_path: Path, moment: float, metadata: SourceMetadata):
        return SampledFrame(
            time_seconds=float(moment),
            timestamp_ms=int(round(float(moment) * 1000)),
            width=16,
            height=16,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
        )

    def _encode(image: np.ndarray, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-jpeg-bytes")

    return _run_fn(
        video,
        probe_fn=lambda path: _metadata(),
        native_times_fn=lambda _v, s, e: tuple(
            t for t in TIMES if t >= s - 1e-9 and t < e
        ),
        native_frame_factory=lambda remaining, _meta: _native_frames(
            tuple(remaining)
        ),
        backend_factory=lambda: backend,
        sample_frame_fn=_sample,
        encode_jpeg_fn=_encode,
        audio_energies_fn=audio_hook,
    )


def test_analyze_serve_supplies_explicit_shared_policy_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_legacy_sparse: None
) -> None:
    from serve_review.detection.decoder import DecoderConfig
    from serve_review.media import audio as audio_module

    # Rippled bed (every 0.03 row is a strict local energy maximum) plus
    # one shared-policy spike: local peak-picking would flag ~14 rows,
    # the shared session-relative policy qualifies exactly one.
    energies = [0.03 if i % 2 else 0.02 for i in range(N_FRAMES)]
    energies[5] = 0.90

    def _rippled(video_path: Path, frame_times: tuple[float, ...]):
        assert tuple(frame_times) == TIMES
        return list(energies)

    capture: dict = {}
    result = _run_with_audio_hook(
        tmp_path, monkeypatch, _rippled,
        no_legacy_sparse=no_legacy_sparse, capture=capture,
    )
    supplied = capture["audio_energies"]
    assert isinstance(supplied, list) and len(supplied) == N_FRAMES
    # Every row carries an explicit 1/0 flag: the builder can never fall
    # back to derived local maxima.
    for entry in supplied:
        assert isinstance(entry, (list, tuple)) and len(entry) == 2
        assert entry[1] in (0.0, 1.0)
    flags = [entry[1] for entry in supplied]
    assert flags[5] == 1.0
    assert sum(flags) == 1.0
    # Ripple rows that strict local-maximum peak-picking would flag stay
    # honestly 0.0 under the shared policy.
    assert energies[13] > energies[12] and energies[13] > energies[14]
    assert flags[13] == 0.0
    assert [entry[0] for entry in supplied] == pytest.approx(energies)
    # Flags equal the shared helper at DecoderConfig defaults.
    defaults = DecoderConfig()
    policy_audio = tuple(
        audio_module.AudioEnergy(time_seconds=t, energy=e)
        for t, e in zip(TIMES, energies)
    )
    qualified = audio_module.qualify_audio_transients(
        policy_audio, defaults.audio_transient_ratio, defaults.audio_transient_floor
    )
    assert qualified == (TIMES[5],)
    assert [TIMES[i] for i, flag in enumerate(flags) if flag == 1.0] == list(
        qualified
    )
    # The qualified spike still supports contact through the audio cue.
    phase = PhaseDocument.from_dict(
        json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    )[0]
    assert phase.stages["contact"].availability == "available"
    assert phase.stages["contact"].provenance == "body_pose_audio"


def test_analyze_serve_silence_yields_explicit_zeros_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_legacy_sparse: None
) -> None:
    from serve_review.media import audio as audio_module

    quiet = [0.005] * N_FRAMES

    def _quiet(video_path: Path, frame_times: tuple[float, ...]):
        return list(quiet)

    capture: dict = {}
    result = _run_with_audio_hook(
        tmp_path, monkeypatch, _quiet,
        no_legacy_sparse=no_legacy_sparse, capture=capture,
    )
    supplied = capture["audio_energies"]
    assert isinstance(supplied, list) and len(supplied) == N_FRAMES
    # No qualified transient: explicit 0.0 everywhere, never derived
    # peaks, never missing, never a crash.
    assert all(
        isinstance(entry, (list, tuple))
        and len(entry) == 2
        and entry[1] == 0.0
        for entry in supplied
    )
    assert [entry[0] for entry in supplied] == pytest.approx(quiet)
    assert (
        audio_module.qualify_audio_transients(
            tuple(
                audio_module.AudioEnergy(time_seconds=t, energy=e)
                for t, e in zip(TIMES, quiet)
            ),
            5.0,
            0.01,
        )
        == ()
    )
    phase = PhaseDocument.from_dict(
        json.loads(result.checkpoints_path.read_text(encoding="utf-8"))
    )[0]
    assert phase.stages["contact"].availability == "available"
    assert result.image_count == len(STAGE_ORDER)


# --- M4 feature-correctness: strictly-positive audio provenance -----------------


def _quiet_audio_energies(video_path: Path, frame_times: tuple[float, ...]):
    """Synthetic present-but-quiet hook: constant bed, no qualified transient."""
    return [0.01] * len(frame_times)


def test_audio_cue_zero_gives_body_pose_while_one_gives_body_pose_audio(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    from serve_review.analyze_serve import run_analyze_serve as _run_fn

    # Quiet bed: shared qualification yields all-zero explicit flags, so the
    # selected contact carries audio_transient == 0.0 -> body_pose.
    video = tmp_path / "serve.mov"
    video.write_bytes(b"fake-video-bytes")
    backend = _FakeBackend()
    result_quiet = _run_fn(
        video,
        output_dir=tmp_path / "output-quiet",
        probe_fn=lambda path: _metadata(),
        native_times_fn=lambda _v, s, e: tuple(t for t in TIMES if t >= s - 1e-9 and t < e),
        native_frame_factory=lambda remaining, _meta: _native_frames(tuple(remaining)),
        backend_factory=lambda: backend,
        sample_frame_fn=lambda vp, m, md: SampledFrame(
            time_seconds=float(m),
            timestamp_ms=int(round(float(m) * 1000)),
            width=16,
            height=16,
            image=__import__("numpy").zeros((16, 16, 3), dtype=__import__("numpy").uint8),
        ),
        encode_jpeg_fn=lambda image, dest: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"fake-jpeg-bytes")),
        audio_energies_fn=_quiet_audio_energies,
    )
    quiet_phase = PhaseDocument.from_dict(
        __import__("json").loads(result_quiet.checkpoints_path.read_text(encoding="utf-8"))
    )[0]
    quiet_contact = quiet_phase.stages["contact"]
    assert quiet_contact.provenance == "body_pose"

    # Impact peak at the DP-selected index: selected contact carries
    # audio_transient == 1.0 -> body_pose_audio.
    result_loud, _ = _run(tmp_path, audio="present")
    import json as _json
    loud_doc = PhaseDocument.from_dict(
        _json.loads(result_loud.checkpoints_path.read_text(encoding="utf-8"))
    )
    loud_contact = loud_doc[0].stages["contact"]
    assert loud_contact.provenance == "body_pose_audio"


def test_local_default_without_nested_stem(tmp_path: Path, no_legacy_sparse: None) -> None:
    import inspect as _inspect

    assert "overwrite" not in _inspect.signature(run_analyze_serve).parameters
    assert "dry_run" in _inspect.signature(run_analyze_serve).parameters
    assert "force" in _inspect.signature(run_analyze_serve).parameters
    result, _ = _run(tmp_path)
    assert result is not None
    expected = tmp_path / "metadata" / "serve" / "manual-analysis"
    assert result.session_dir == expected
    assert result.checkpoints_path == expected / "checkpoints.json"
    assert result.cache_path == expected / "cache" / "kinematic-track-v1.jsonl"
    assert result.cache_path.parent.parent == expected


def test_explicit_output_is_exact(tmp_path: Path, no_legacy_sparse: None) -> None:
    from serve_review.analyze_serve import run_analyze_serve as _run_fn

    video = tmp_path / "serve.mov"
    video.write_bytes(b"fake-video-bytes")
    backend = _FakeBackend()
    exact = tmp_path / "custom" / "serve-001"
    result = _run_fn(
        video,
        output_dir=exact,
        probe_fn=lambda path: _metadata(),
        native_times_fn=lambda _v, s, e: tuple(t for t in TIMES if t >= s - 1e-9 and t < e),
        native_frame_factory=lambda remaining, _meta: _native_frames(tuple(remaining)),
        backend_factory=lambda: backend,
        sample_frame_fn=lambda vp, m, md: SampledFrame(
            time_seconds=float(m),
            timestamp_ms=int(round(float(m) * 1000)),
            width=16,
            height=16,
            image=__import__("numpy").zeros((16, 16, 3), dtype=__import__("numpy").uint8),
        ),
        encode_jpeg_fn=lambda image, dest: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"fake-jpeg-bytes")),
        audio_energies_fn=_absent_audio_energies,
    )
    assert result is not None
    assert result.session_dir == exact
    assert result.checkpoints_path.parent == exact
    assert result.cache_path == exact / "cache" / "kinematic-track-v1.jsonl"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    from serve_review.analyze_serve import run_analyze_serve as _run_fn

    video = tmp_path / "serve.mov"
    video.write_bytes(b"fake-video-bytes")
    messages: list[str] = []

    def _boom(*a, **k):
        raise AssertionError("producer work must not run during dry-run")

    out = _run_fn(
        video,
        start_seconds=0.0,
        end_seconds=1.0,
        probe_fn=_boom,
        native_times_fn=_boom,
        backend_factory=_boom,
        sample_frame_fn=_boom,
        encode_jpeg_fn=_boom,
        audio_energies_fn=_boom,
        progress_callback=messages.append,
        dry_run=True,
    )
    assert out is None
    assert not (tmp_path / "metadata").exists()
    joined = "\n".join(messages)
    assert str(tmp_path / "metadata" / "serve" / "manual-analysis") in joined


def test_force_boundary_for_existing_output(tmp_path: Path, no_legacy_sparse: None) -> None:
    from serve_review.analyze_serve import AnalyzeServeError
    from serve_review.analyze_serve import run_analyze_serve as _run_fn

    _run(tmp_path)
    video = tmp_path / "serve.mov"
    with pytest.raises(AnalyzeServeError, match="collision"):
        _run_fn(
            video,
            probe_fn=lambda path: _metadata(),
            native_times_fn=lambda _v, s, e: tuple(t for t in TIMES if t >= s - 1e-9 and t < e),
            native_frame_factory=lambda remaining, _meta: _native_frames(tuple(remaining)),
            backend_factory=_FakeBackend,
            sample_frame_fn=lambda vp, m, md: SampledFrame(
                time_seconds=float(m),
                timestamp_ms=int(round(float(m) * 1000)),
                width=16,
                height=16,
                image=__import__("numpy").zeros((16, 16, 3), dtype=__import__("numpy").uint8),
            ),
            encode_jpeg_fn=lambda image, dest: dest.write_bytes(b"x"),
            audio_energies_fn=_absent_audio_energies,
        )
    result2, backend2 = _run(tmp_path, force=True)
    assert result2 is not None
    assert result2.checkpoints_path.is_file()
