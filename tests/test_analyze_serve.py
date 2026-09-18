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
from serve_review.checkpoints.six_anchor_solver import SixAnchorSolverConfig
from serve_review.domain import (
    STAGE_ORDER,
    AttemptPhase,
    MediaRange,
    PhaseDocument,
    PhaseError,
    SourceMetadata,
    StagePhase,
)
from serve_review.media.color import ColorMetadata
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
    monkeypatch.setattr(
        analyze_module.color_module,
        "probe_color_metadata",
        lambda video, *, ffprobe: ColorMetadata("bt709", "bt709", "bt709", "tv"),
    )


@pytest.fixture
def no_legacy_sparse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any legacy sparse/M3 function is invoked."""
    import serve_review.pose.cache as cache_module
    import serve_review.pose.extract as extract_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "legacy sparse path must never run on analyze-serve"
        )

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

def test_cli_analyze_reports_outputs_on_success(tmp_path: Path, monkeypatch) -> None:
    from click.testing import CliRunner
    from serve_review.cli.main import cli
    import serve_review.analyze_serve as serve_module

    source = tmp_path / "serve.mov"
    source.write_bytes(b"fake")
    fake = SimpleNamespace(
        checkpoints_path=tmp_path / "checkpoints.json",
        diagnostics_path=tmp_path / "diagnostics.json",
        index_html=tmp_path / "review" / "index.html",
        attempt_phase=SimpleNamespace(attempt_id="serve-001", structural_status="partial"),
        attempt_range=SimpleNamespace(start_seconds=0.0, end_seconds=1.0),
        image_count=4,
    )
    monkeypatch.setattr(serve_module, "run_analyze_serve", lambda *a, **k: fake)
    result = CliRunner().invoke(cli, ["analyze-serve", str(source)])
    assert result.exit_code == 0
    assert str(fake.checkpoints_path) in result.output

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


def test_cli_anchor2_comparison_is_development_only() -> None:
    from click.testing import CliRunner
    from serve_review.cli.main import cli

    runner = CliRunner()
    assert runner.invoke(cli, ["analyze-serve", "session.mov", "--anchor2comparison"]).exit_code != 0
    assert runner.invoke(cli, ["dev", "anchor2-comparison", "session.mov", "--help"]).exit_code == 0


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
