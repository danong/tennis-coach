"""Focused M3.5 tests for the end-to-end cut pipeline.

Fully faked orchestration plus generated-media integration. Deterministic,
offline, and independent of private footage. Generated fixtures live in
temporary directories; inference stages are injected fakes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from serve_review.detection.features import FeatureConfig, FeatureFrame
from serve_review.detection.plan import PlanConfig
from serve_review.detection.ranges import CandidateRange, RangeConfig
from serve_review.domain import AttemptDocument, SourceMetadata
from serve_review.media import export as export_module
from serve_review.pipeline import CutCancelled, CutError, run_cut


def make_metadata(
    duration: float = 10.0, fingerprint: str = "sha256:cut-fixture"
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


def make_feature(
    t: float,
    *,
    motion: float | None = 0.8,
    overhead: float | None = 1.0,
    visible: float = 0.9,
    has_person: bool = True,
) -> FeatureFrame:
    if not has_person:
        return FeatureFrame(time_seconds=t, has_person=False, visible_fraction=0.0)
    rest = None if motion is None else 1.0 - motion
    return FeatureFrame(
        time_seconds=t,
        has_person=True,
        visible_fraction=visible,
        torso_scale=0.3,
        player_scale=0.3,
        wrist_speed=0.1,
        elbow_speed=0.1,
        body_motion=0.5,
        overhead_evidence=overhead,
        rest_evidence=rest,
        motion_evidence=motion,
    )


def make_source(tmp_path: Path, name: str = "session.mov") -> Path:
    video = tmp_path / name
    video.write_bytes(b"fake-source-bytes")
    return video


def fake_stages(
    metadata: SourceMetadata,
    *,
    candidates: tuple[CandidateRange, ...] = (CandidateRange(1.0, 2.0),),
    cache_hit: bool = False,
    export_mode_files: bool = True,
    seen: dict | None = None,
):
    """Build injected stage fakes recording calls."""
    if seen is None:
        seen = {}
    seen.setdefault("export_calls", [])

    def _probe(video: Path) -> SourceMetadata:
        seen["probe_video"] = Path(video)
        return metadata

    def _extract(video, cache_path, **kwargs):
        seen["extract_kwargs"] = dict(kwargs)
        seen["extract_cache"] = Path(cache_path)
        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{\"type\": \"header\"}\n")
        return SimpleNamespace(cache_hit=cache_hit)

    def _load(cache_path: Path):
        seen["load_cache"] = Path(cache_path)
        return SimpleNamespace(frames=())

    def _features(observations, config):
        assert isinstance(config, FeatureConfig)
        seen["feature_config"] = config
        return (make_feature(0.0), make_feature(0.5))

    def _candidates(frames, config):
        assert isinstance(config, RangeConfig)
        seen["range_config"] = config
        return tuple(candidates)

    def _export(video, plan, out_dir, **kwargs):
        seen["export_calls"].append(
            {"plan": plan, "out_dir": Path(out_dir), "kwargs": dict(kwargs)}
        )
        mode = kwargs.get("mode", "compilation")
        session = Path(out_dir)
        comp = None
        clips: list[Path] = []
        if export_mode_files:
            if mode in ("compilation", "both"):
                comp = session / export_module.COMPILATION_FILENAME
                comp.parent.mkdir(parents=True, exist_ok=True)
                comp.write_bytes(b"fake-compilation")
            if mode in ("clips", "both"):
                clips_dir = session / export_module.CLIPS_SUBDIR
                clips_dir.mkdir(parents=True, exist_ok=True)
                for number in range(1, len(plan.ranges) + 1):
                    clip = clips_dir / export_module.clip_filename(number)
                    clip.write_bytes(b"fake-clip")
                    clips.append(clip)
        return {"compilation": comp, "clips": clips}

    return _probe, _extract, _load, _features, _candidates, _export, seen


def run_faked(
    tmp_path: Path,
    *,
    duration: float = 10.0,
    candidates=(CandidateRange(1.0, 2.0),),
    padding: float = 1.0,
    mode: str = "compilation",
    cache_hit: bool = False,
    **overrides,
):
    video = make_source(tmp_path)
    metadata = make_metadata(duration=duration)
    probe, extract, load, feats, cands, export, seen = fake_stages(
        metadata, candidates=tuple(candidates), cache_hit=cache_hit
    )
    params = dict(
        output_dir=tmp_path / "output",
        padding_seconds=padding,
        mode=mode,
        probe_fn=probe,
        extract_fn=extract,
        load_cache_fn=load,
        features_fn=feats,
        candidates_fn=cands,
        export_fn=export,
    )
    params.update(overrides)
    result = run_cut(video, **params)
    return video, metadata, result, seen


# --- faked orchestration ---


def test_run_cut_compilation_layout_and_deterministic_json(tmp_path: Path) -> None:
    video, metadata, result, seen = run_faked(tmp_path)
    session = tmp_path / "output" / video.stem
    assert result.session_dir == session
    assert (session / "source.json").is_file()
    assert (session / "attempts.json").is_file()
    assert (session / "run.json").is_file()
    assert result.compilation is not None and result.compilation.is_file()
    assert result.clips == ()
    assert result.empty is False
    assert result.cache_hit is False
    stored = AttemptDocument.from_json(
        (session / "attempts.json").read_text(encoding="utf-8")
    )
    assert stored == result.attempts_document
    assert stored.padding_seconds == 1.0
    # Effective range is padded [1,2) -> [0,3).
    assert stored.attempts[0].effective_range.start_seconds == 0.0
    assert stored.attempts[0].effective_range.end_seconds == 3.0
    assert stored.attempts[0].detected_range.start_seconds == 1.0
    # Frozen defaults are used.
    assert seen["feature_config"] == FeatureConfig()
    assert seen["range_config"] == RangeConfig()
    # Deterministic rerun in a fresh output dir with identical inputs.
    video_b = tmp_path / "session-b.mov"
    video_b.write_bytes(b"fake-source-bytes")
    probe, extract, load, feats, cands, export, _ = fake_stages(
        metadata, candidates=(CandidateRange(1.0, 2.0),)
    )
    run_cut(
        video_b,
        output_dir=tmp_path / "output-b",
        padding_seconds=1.0,
        mode="compilation",
        probe_fn=probe,
        extract_fn=extract,
        load_cache_fn=load,
        features_fn=feats,
        candidates_fn=cands,
        export_fn=export,
    )
    second_doc = (tmp_path / "output-b" / video_b.stem / "attempts.json").read_text(
        encoding="utf-8"
    )
    assert AttemptDocument.from_json(second_doc) == stored


@pytest.mark.parametrize("mode", ["compilation", "clips", "both"])
def test_run_cut_all_output_modes(tmp_path: Path, mode: str) -> None:
    _, _, result, seen = run_faked(tmp_path, mode=mode)
    assert seen["export_calls"][0]["kwargs"]["mode"] == mode
    if mode == "compilation":
        assert result.compilation is not None
        assert result.clips == ()
    elif mode == "clips":
        assert result.compilation is None
        assert len(result.clips) == 1
    else:
        assert result.compilation is not None
        assert len(result.clips) == 1


def test_run_cut_padding_zero_and_clamped(tmp_path: Path) -> None:
    _, _, result, _ = run_faked(tmp_path, padding=0.0)
    attempt = result.attempts_document.attempts[0]
    assert attempt.effective_range.start_seconds == 1.0
    assert attempt.effective_range.end_seconds == 2.0

    # Candidate near the source end clamps to duration.
    _, _, result2, _ = run_faked(
        tmp_path,
        duration=10.0,
        candidates=(CandidateRange(9.0, 9.9),),
        padding=5.0,
        mode="clips",
    )
    attempt2 = result2.attempts_document.attempts[0]
    assert attempt2.effective_range.start_seconds == 4.0
    assert attempt2.effective_range.end_seconds == 10.0

    # CLI-supplied padding flows into the export plan.
    _, _, _, seen = run_faked(tmp_path, padding=2.0)
    plan = seen["export_calls"][0]["plan"]
    assert plan.ranges[0].start_seconds == 0.0
    assert plan.ranges[0].end_seconds == 4.0


def test_run_cut_empty_yields_honest_empty_and_no_media(tmp_path: Path) -> None:
    video, _, result, seen = run_faked(tmp_path, candidates=())
    session = tmp_path / "output" / video.stem
    assert result.empty is True
    assert result.compilation is None
    assert result.clips == ()
    assert seen["export_calls"] == []
    assert not (session / export_module.COMPILATION_FILENAME).exists()
    assert not (session / export_module.CLIPS_SUBDIR).exists()
    stored = AttemptDocument.from_json(
        (session / "attempts.json").read_text(encoding="utf-8")
    )
    assert len(stored) == 0
    assert stored.export_ranges == ()
    run_payload = json.loads((session / "run.json").read_text(encoding="utf-8"))
    assert run_payload["status"] == "empty"
    assert run_payload["attempt_count"] == 0


def test_run_cut_stage_failure_cleans_partial_media(tmp_path: Path) -> None:
    video = make_source(tmp_path)
    metadata = make_metadata()
    probe, extract, load, feats, cands, _, seen = fake_stages(
        metadata, candidates=(CandidateRange(1.0, 2.0),)
    )

    def _failing_export(video_p, plan, out_dir, **kwargs):
        session = Path(out_dir)
        comp = session / export_module.COMPILATION_FILENAME
        comp.parent.mkdir(parents=True, exist_ok=True)
        comp.write_bytes(b"partial-bytes")
        clips_dir = session / export_module.CLIPS_SUBDIR
        clips_dir.mkdir(parents=True, exist_ok=True)
        partial_clip = clips_dir / export_module.clip_filename(1)
        partial_clip.write_bytes(b"partial-clip")
        raise export_module.ExportError("fake ffmpeg boom")

    with pytest.raises(CutError) as excinfo:
        run_cut(
            video,
            output_dir=tmp_path / "output",
            padding_seconds=1.0,
            mode="both",
            probe_fn=probe,
            extract_fn=extract,
            load_cache_fn=load,
            features_fn=feats,
            candidates_fn=cands,
            export_fn=_failing_export,
        )
    assert excinfo.value.stage == "export"
    session = tmp_path / "output" / video.stem
    assert not (session / export_module.COMPILATION_FILENAME).exists()
    assert not (session / export_module.CLIPS_SUBDIR / "serve-001.mov").exists()
    leftovers = list(session.glob("*.tmp-*")) + list(
        (session / export_module.CLIPS_SUBDIR).glob("*.tmp-*")
    ) if (session / export_module.CLIPS_SUBDIR).exists() else list(session.glob("*.tmp-*"))
    assert leftovers == []


def test_run_cut_probe_failure_stage_and_no_source_json(tmp_path: Path) -> None:
    video = make_source(tmp_path)

    def _boom(video_p: Path) -> SourceMetadata:
        raise RuntimeError("fake ffprobe missing")

    with pytest.raises(CutError) as excinfo:
        run_cut(
            video,
            output_dir=tmp_path / "output",
            probe_fn=_boom,
        )
    assert excinfo.value.stage == "probe"
    assert not (tmp_path / "output" / video.stem / "source.json").exists()


def test_run_cut_pose_and_detect_failures_are_stage_specific(tmp_path: Path) -> None:
    video = make_source(tmp_path)
    metadata = make_metadata()
    probe, extract, load, feats, cands, export, _ = fake_stages(metadata)

    def _pose_boom(*args, **kwargs):
        raise RuntimeError("fake inference failure")

    with pytest.raises(CutError) as excinfo:
        run_cut(
            video,
            output_dir=tmp_path / "o1",
            probe_fn=probe,
            extract_fn=_pose_boom,
            load_cache_fn=load,
            features_fn=feats,
            candidates_fn=cands,
            export_fn=export,
        )
    assert excinfo.value.stage == "pose"

    def _detect_boom(frames, config):
        raise RuntimeError("fake range failure")

    with pytest.raises(CutError) as excinfo2:
        run_cut(
            video,
            output_dir=tmp_path / "o2",
            probe_fn=probe,
            extract_fn=extract,
            load_cache_fn=load,
            features_fn=feats,
            candidates_fn=_detect_boom,
            export_fn=export,
        )
    assert excinfo2.value.stage == "detect"


def test_run_cut_export_collision_is_stage_error(tmp_path: Path) -> None:
    video = make_source(tmp_path)
    metadata = make_metadata()
    probe, extract, load, feats, cands, _, _ = fake_stages(metadata)

    def _collision(video_p, plan, out_dir, **kwargs):
        raise export_module.ExportCollisionError("fake collision")

    with pytest.raises(CutError) as excinfo:
        run_cut(
            video,
            output_dir=tmp_path / "output",
            probe_fn=probe,
            extract_fn=extract,
            load_cache_fn=load,
            features_fn=feats,
            candidates_fn=cands,
            export_fn=_collision,
        )
    assert excinfo.value.stage == "export"
    assert "collision" in str(excinfo.value).lower()


def test_run_cut_cache_hit_propagates_and_reuses(tmp_path: Path) -> None:
    video, _, result, seen = run_faked(tmp_path, cache_hit=True)
    assert result.cache_hit is True
    # Cache reuse: extraction never requests a cache replacement.
    assert seen["extract_kwargs"].get("overwrite", False) is not True
    assert seen["extract_cache"] == result.cache_path
    assert result.cache_path.is_file()


def test_run_cut_invalid_inputs(tmp_path: Path) -> None:
    video = make_source(tmp_path)
    with pytest.raises(CutError):
        run_cut(video, output_dir=tmp_path / "o", padding_seconds=-1.0)
    with pytest.raises(CutError):
        run_cut(video, output_dir=tmp_path / "o", mode="everything")
    with pytest.raises(CutError):
        run_cut(tmp_path / "missing.mov", output_dir=tmp_path / "o")


def test_run_cut_cancellation_cleans_media(tmp_path: Path) -> None:
    video = make_source(tmp_path)
    metadata = make_metadata()
    probe, extract, load, feats, cands, export, _ = fake_stages(metadata)
    calls = {"count": 0}

    def _cancel():
        calls["count"] += 1
        return calls["count"] >= 2

    with pytest.raises(CutError):
        run_cut(
            video,
            output_dir=tmp_path / "output",
            probe_fn=probe,
            extract_fn=extract,
            load_cache_fn=load,
            features_fn=feats,
            candidates_fn=cands,
            export_fn=export,
            is_cancelled=_cancel,
        )


def test_run_cut_progress_reports_stages(tmp_path: Path) -> None:
    messages: list[str] = []
    run_faked(tmp_path, progress_callback=messages.append)
    joined = "\n".join(messages)
    for stage in ("probing", "poses", "features", "candidates", "planning"):
        assert stage in joined


def test_run_cut_default_configs_are_frozen(tmp_path: Path) -> None:
    _, _, _, seen = run_faked(tmp_path)
    assert seen["feature_config"] == FeatureConfig()
    assert seen["range_config"] == RangeConfig()
    assert PlanConfig(padding_seconds=1.0).padding_seconds == 1.0


# --- generated-media integration (real probe + real export) ---


def _needs_tools() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")


def _integration_fakes(candidate: CandidateRange | None):
    from types import SimpleNamespace as _NS

    def _extract(video, cache_path, **kwargs):
        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{\"type\": \"header\"}\n")
        return _NS(cache_hit=False)

    def _load(cache_path):
        return _NS(frames=())

    def _features(observations, config):
        return (make_feature(0.0), make_feature(0.5))

    def _candidates(frames, config):
        return () if candidate is None else (candidate,)

    return _extract, _load, _features, _candidates


def _generate(tmp_path: Path, name: str = "session.mov", duration: float = 2.0) -> Path:
    from media_factory import generate_fixture, landscape_spec

    return generate_fixture(tmp_path / name, landscape_spec(duration_seconds=duration))


def test_cut_integration_compilation_media(tmp_path: Path) -> None:
    _needs_tools()
    from serve_review.media.probe import probe_source

    video = _generate(tmp_path)
    before = video.read_bytes()
    extract, load, feats, cands = _integration_fakes(CandidateRange(0.2, 0.7))
    result = run_cut(
        video,
        output_dir=tmp_path / "output",
        padding_seconds=0.5,
        mode="compilation",
        extract_fn=extract,
        load_cache_fn=load,
        features_fn=feats,
        candidates_fn=cands,
    )
    assert result.empty is False
    assert result.compilation is not None and result.compilation.is_file()
    assert result.clips == ()
    probed = probe_source(result.compilation)
    # Padded [0.2,0.7) with 0.5 -> [0,1.2); duration within one source sample.
    assert probed.duration_seconds == pytest.approx(1.2, abs=0.08)
    attempts = AttemptDocument.from_json(result.attempts_path.read_text(encoding="utf-8"))
    assert attempts.attempts[0].detected_range.start_seconds == pytest.approx(0.2)
    assert attempts.attempts[0].effective_range.end_seconds == pytest.approx(1.2)
    assert (result.session_dir / "source.json").is_file()
    assert (result.session_dir / "run.json").is_file()
    assert video.read_bytes() == before


def test_cut_integration_both_modes_media(tmp_path: Path) -> None:
    _needs_tools()
    video = _generate(tmp_path)
    extract, load, feats, cands = _integration_fakes(CandidateRange(0.2, 0.7))
    result = run_cut(
        video,
        output_dir=tmp_path / "output",
        padding_seconds=0.0,
        mode="both",
        extract_fn=extract,
        load_cache_fn=load,
        features_fn=feats,
        candidates_fn=cands,
    )
    assert result.compilation is not None and result.compilation.is_file()
    assert len(result.clips) == 1 and result.clips[0].is_file()


def test_cut_integration_clips_mode_media(tmp_path: Path) -> None:
    _needs_tools()
    video = _generate(tmp_path)
    extract, load, feats, cands = _integration_fakes(CandidateRange(0.2, 0.7))
    result = run_cut(
        video,
        output_dir=tmp_path / "output",
        padding_seconds=0.0,
        mode="clips",
        extract_fn=extract,
        load_cache_fn=load,
        features_fn=feats,
        candidates_fn=cands,
    )
    assert result.compilation is None
    assert len(result.clips) == 1 and result.clips[0].is_file()
    assert not (result.session_dir / export_module.COMPILATION_FILENAME).exists()


def test_cut_integration_empty_writes_no_media(tmp_path: Path) -> None:
    _needs_tools()
    video = _generate(tmp_path)
    before = video.read_bytes()
    extract, load, feats, cands = _integration_fakes(None)
    result = run_cut(
        video,
        output_dir=tmp_path / "output",
        padding_seconds=1.0,
        mode="both",
        extract_fn=extract,
        load_cache_fn=load,
        features_fn=feats,
        candidates_fn=cands,
    )
    assert result.empty is True
    assert not (result.session_dir / export_module.COMPILATION_FILENAME).exists()
    assert not (result.session_dir / export_module.CLIPS_SUBDIR).exists()
    stored = AttemptDocument.from_json(result.attempts_path.read_text(encoding="utf-8"))
    assert len(stored) == 0
    assert video.read_bytes() == before


def test_cut_integration_deterministic_attempts(tmp_path: Path) -> None:
    _needs_tools()
    video = _generate(tmp_path)
    extract, load, feats, cands = _integration_fakes(CandidateRange(0.2, 0.7))
    first = run_cut(
        video,
        output_dir=tmp_path / "o1",
        padding_seconds=0.25,
        mode="clips",
        extract_fn=extract,
        load_cache_fn=load,
        features_fn=feats,
        candidates_fn=cands,
    )
    extract2, load2, feats2, cands2 = _integration_fakes(CandidateRange(0.2, 0.7))
    second = run_cut(
        video,
        output_dir=tmp_path / "o2",
        padding_seconds=0.25,
        mode="clips",
        extract_fn=extract2,
        load_cache_fn=load2,
        features_fn=feats2,
        candidates_fn=cands2,
    )
    assert first.attempts_path.read_bytes() == second.attempts_path.read_bytes()
