"""Focused tests for the single-command audio-free 3D path (analyze-serve).

Deterministic, offline, and independent of private footage, model
weights, and network access: source videos are tiny fake byte files in
temporary directories, probing/timestamps/inference/sampling/encoding
are faked through the injectable ``run_analyze_serve`` hooks, and the
real M4.8 filter, M4.9 waveform, M4.10a candidate, and M4.10b DP
modules run for real over synthetic constant-geometry world poses.
Legacy sparse functions (phase features/evidence, sparse pose cache,
audio, the old phase solver entry points) are monkeypatched to fail
loudly so any regression off the 3D-only path errors instead of
passing silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_review.analyze_serve import (
    ANALYZE_SERVE_ATTEMPT_ID,
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


@pytest.fixture
def no_legacy_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any legacy sparse/M3 function is invoked."""
    import serve_review.checkpoints.evidence as evidence_module
    import serve_review.checkpoints.phase_features as features_module
    import serve_review.checkpoints.phase_solver as solver_module
    import serve_review.media.audio as audio_module
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
    monkeypatch.setattr(audio_module, "iter_audio_energy", _boom)


def _run(
    tmp_path: Path,
    *,
    missing: tuple[str, ...] = (),
    start: float | None = None,
    end: float | None = None,
    overwrite: bool = False,
    backend_holder: list | None = None,
    sampled_holder: list | None = None,
    encoded_holder: list | None = None,
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

    result = _run_fn(
        video,
        start_seconds=start,
        end_seconds=end,
        output_dir=tmp_path / "output",
        overwrite=overwrite,
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
    assert args.output_dir == Path("output")
    assert args.cache is None
    assert args.model == Path("models/pose_landmarker_heavy.task")
    assert args.overwrite is False
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
            output_dir=tmp_path / "output",
            probe_fn=lambda path: _metadata(),
        )
    assert excinfo.value.stage == "validate"


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


# --- 3D-only orchestration, cache reuse, artifacts ---------------------------


def test_full_run_uses_cache_and_writes_artifacts(
    tmp_path: Path, no_legacy_sparse: None
) -> None:
    holders: list = []
    result, backend = _run(tmp_path, backend_holder=holders)
    assert result.cache_hit is False
    assert backend.calls == N_FRAMES

    # Cache reuse: drop outputs but keep the cache; the second run
    # performs no inference.
    result.checkpoints_path.unlink()
    result.diagnostics_path.unlink()
    import shutil as _shutil

    _shutil.rmtree(result.review_dir)
    result2, backend2 = _run(tmp_path)
    assert result2.cache_hit is True
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

    # Audio-free contact honesty: body_pose + stable limitations, never audio.
    contact = phase.stages["contact"]
    assert contact.availability == "available"
    assert contact.provenance == "body_pose"
    assert "contact_not_directly_observed" in contact.limitations
    assert "audio_transient_not_used" in contact.limitations
    for stage in STAGE_ORDER:
        assert phase.stages[stage].provenance != "body_pose_audio"

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
    assert diagnostics["audio"] == "not_used"
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
    assert diagnostics["derived"]["acceleration"]["status"] == "selected"
    assert diagnostics["derived"]["deceleration"]["status"] == "selected"

    # Review: JPEG per available stage plus self-contained index.
    assert result.image_count == len(STAGE_ORDER)
    assert result.index_html.is_file()
    html = result.index_html.read_text(encoding="utf-8")
    assert "never claimed as exact visual observation" in html
    review = json.loads(result.review_json.read_text(encoding="utf-8"))
    assert len(review["entries"]) == len(STAGE_ORDER)
    for entry in review["entries"]:
        assert entry["status"] == "rendered"
        assert entry["image"] is not None
        assert (result.review_dir / entry["image"]).is_file()
        # PTS rendering linkage: requested keyframe versus sampled frame.
        assert entry["requested_source_time"] == keyframes[entry["stage"]]
        assert entry["actual_source_time"] == entry["requested_source_time"]


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
    assert contact_record["coverage"] == pytest.approx(0.15)


def test_outputs_are_deterministic(tmp_path: Path, no_legacy_sparse: None) -> None:
    first, _ = _run(tmp_path)
    before_checkpoints = first.checkpoints_path.read_bytes()
    before_diagnostics = first.diagnostics_path.read_bytes()
    second, _ = _run(tmp_path, overwrite=True)
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
            output_dir=tmp_path / "output",
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
        method_version="composite-anchors-v1",
        config_id="composite-anchors-default-v1",
        schema_version=1,
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


def test_audio_free_body_pose_contact_is_allowed() -> None:
    contact = _stage(
        15.0,
        16.0,
        limitations=("contact_not_directly_observed", "audio_transient_not_used"),
    )
    assert _attempt_with_contact(contact).stage("contact").provenance == "body_pose"


def test_body_pose_contact_without_limitations_stays_rejected() -> None:
    with pytest.raises(PhaseError):
        _attempt_with_contact(_stage(15.0, 16.0, limitations=()))


def test_body_pose_contact_requires_anchor_and_uncertainty() -> None:
    limited = ("contact_not_directly_observed", "audio_transient_not_used")
    with pytest.raises(PhaseError):
        _attempt_with_contact(_stage(15.0, 16.0, keyframe=None, limitations=limited))
    with pytest.raises(PhaseError):
        _attempt_with_contact(
            _stage(15.0, 16.0, uncertainty=None, limitations=limited)
        )
