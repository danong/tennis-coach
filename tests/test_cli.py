from pathlib import Path
from types import SimpleNamespace

import pytest

from serve_review.cli import build_parser, cut, export_cmd, probe


def test_cut_defaults() -> None:
    args = build_parser().parse_args(["cut", "session.mov"])

    assert args.video == Path("session.mov")
    assert args.padding == 1.0
    assert args.output == "compilation"
    assert args.output_dir == Path("output")
    assert args.overwrite is False
    assert args.ffmpeg == "ffmpeg"
    assert args.ffprobe == "ffprobe"


def test_cut_rejects_missing_input(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(["cut", str(tmp_path / "missing.mov")])

    assert cut(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_cut_rejects_negative_padding(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.mov"
    source.touch()
    args = build_parser().parse_args(["cut", str(source), "--padding", "-0.1"])

    assert cut(args) == 2
    assert "zero or greater" in capsys.readouterr().err


def test_cut_success_reports_attempts_and_outputs(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.pipeline import CutResult
    from serve_review.domain import AttemptDocument
    from serve_review import pipeline as pipeline_module

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")
    metadata = SimpleNamespace(fingerprint="sha256:x", duration_seconds=10.0)
    document = AttemptDocument(
        source_fingerprint="sha256:x",
        source_duration_seconds=10.0,
        padding_seconds=1.0,
        method_version="candidate-ranges-v1+plan-v1",
        attempts=(),
        export_ranges=(),
    )
    session = tmp_path / "output" / source.stem
    session.mkdir(parents=True)
    attempts_path = session / "attempts.json"
    attempts_path.write_text(document.to_json(), encoding="utf-8")
    comp = session / "serves.mov"
    comp.write_bytes(b"fake")
    clip = session / "clips" / "serve-001.mov"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"fake")
    result = CutResult(
        video=source,
        session_dir=session,
        source_metadata=metadata,  # type: ignore[arg-type]
        attempts_document=document,
        attempts_path=attempts_path,
        source_path=session / "source.json",
        run_path=session / "run.json",
        cache_path=session / "cache" / "pose-v1.jsonl",
        cache_hit=False,
        mode="both",
        compilation=comp,
        clips=(clip,),
        empty=False,
    )
    seen: dict = {}

    def _fake_run_cut(video, **kwargs):
        seen.update(kwargs)
        seen["video"] = Path(video)
        return result

    monkeypatch.setattr(pipeline_module, "run_cut", _fake_run_cut)
    args = build_parser().parse_args(
        ["cut", str(source), "--padding", "1.5", "--output", "both",
         "--output-dir", str(tmp_path / "output")]
    )
    assert cut(args) == 0
    out = capsys.readouterr().out
    assert str(attempts_path) in out
    assert str(comp) in out
    assert str(clip) in out
    assert seen["padding_seconds"] == 1.5
    assert seen["mode"] == "both"
    assert seen["overwrite"] is False


def test_cut_empty_result_reports_no_media(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.pipeline import CutResult
    from serve_review.domain import AttemptDocument
    from serve_review import pipeline as pipeline_module

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")
    document = AttemptDocument(
        source_fingerprint="sha256:x",
        source_duration_seconds=10.0,
        padding_seconds=1.0,
        method_version="candidate-ranges-v1+plan-v1",
        attempts=(),
        export_ranges=(),
    )
    session = tmp_path / "output" / source.stem
    session.mkdir(parents=True)
    attempts_path = session / "attempts.json"
    attempts_path.write_text(document.to_json(), encoding="utf-8")
    result = CutResult(
        video=source,
        session_dir=session,
        source_metadata=SimpleNamespace(fingerprint="sha256:x"),  # type: ignore[arg-type]
        attempts_document=document,
        attempts_path=attempts_path,
        source_path=session / "source.json",
        run_path=session / "run.json",
        cache_path=session / "cache" / "pose-v1.jsonl",
        cache_hit=True,
        mode="compilation",
        compilation=None,
        clips=(),
        empty=True,
    )
    monkeypatch.setattr(pipeline_module, "run_cut", lambda video, **k: result)
    args = build_parser().parse_args(["cut", str(source), "--output-dir", str(tmp_path / "output")])
    assert cut(args) == 0
    out = capsys.readouterr().out
    assert str(attempts_path) in out
    assert "no serves detected" in out.lower()
    assert not (session / "serves.mov").exists()


def test_cut_reports_stage_failure(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.pipeline import CutError
    from serve_review import pipeline as pipeline_module

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")

    def _boom(video, **kwargs):
        raise CutError("pose", "pose extraction failed: fake.")

    monkeypatch.setattr(pipeline_module, "run_cut", _boom)
    args = build_parser().parse_args(["cut", str(source)])
    assert cut(args) == 1
    err = capsys.readouterr().err
    assert "pose" in err
    assert "ERROR" in err


def test_cut_forwards_overwrite_and_tools(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review import pipeline as pipeline_module

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")
    seen: dict = {}

    def _fake(video, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(pipeline_module, "run_cut", _fake)
    args = build_parser().parse_args(
        ["cut", str(source), "--overwrite", "--ffmpeg", "/bin/ffmpeg", "--ffprobe", "/bin/ffprobe"]
    )
    with __import__("pytest").raises(SystemExit):
        cut(args)
    assert seen["overwrite"] is True
    assert seen["ffmpeg"] == "/bin/ffmpeg"
    assert seen["ffprobe"] == "/bin/ffprobe"


def test_cut_integration_generated_media(tmp_path: Path, capsys, monkeypatch) -> None:
    import shutil

    from media_factory import generate_av_fixture, landscape_spec
    from serve_review.detection.features import FeatureFrame
    from serve_review.pose import cache as cache_module
    from serve_review.pose import extract as extract_module
    from serve_review.detection import features as features_module

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        __import__("pytest").skip("FFmpeg and ffprobe are required")
    # In-band transient near 0.55 s; the faked pose track accelerates
    # overhead at 0.5 s so the scale-invariant validator fires.
    audio_expr = "0.2*sin(2*PI*220*t)+if(lt(abs(t-0.55),0.004),sin(2*PI*2000*t),0)"
    video = generate_av_fixture(
        tmp_path / "session.mov",
        landscape_spec(duration_seconds=2.0),
        audio_expr=audio_expr,
    )
    before = video.read_bytes()

    def _feat(moment, *, torso=0.0, overhead=0.0, elbow_speed=0.1, rest=0.9):
        return FeatureFrame(
            time_seconds=moment,
            has_person=True,
            visible_fraction=1.0,
            torso_scale=1.0,
            player_scale=1.0,
            wrist_speed=0.1,
            elbow_speed=elbow_speed,
            body_motion=0.2,
            overhead_evidence=overhead,
            rest_evidence=float(rest),
            motion_evidence=1.0 - float(rest),
            elbow_flexion_left=150.0,
            elbow_flexion_right=150.0,
            shoulder_tilt=0.0,
            torso_displacement=torso,
        )

    def _fake_extract(video_p, cache_path, **kwargs):
        from types import SimpleNamespace as _NS

        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{}" + b"\n")
        return _NS(cache_hit=False, cache_path=target)

    def _fake_load(cache_path):
        from types import SimpleNamespace as _NS

        return _NS(frames=())

    def _fake_features(observations, config):
        frames = [_feat(0.0), _feat(0.1)]
        for i in range(3):
            frames.append(
                _feat(0.2 + i * 0.1, torso=0.5, overhead=0.0,
                      elbow_speed=0.4, rest=0.15)
            )
        frames.append(_feat(0.5, torso=0.5, overhead=1.0, elbow_speed=5.0, rest=0.1))
        frames.append(_feat(0.6, torso=0.3, overhead=1.0, elbow_speed=2.0, rest=0.2))
        frames.append(_feat(1.0))
        frames.append(_feat(1.1))
        return tuple(frames)

    monkeypatch.setattr(extract_module, "extract_poses", _fake_extract)
    monkeypatch.setattr(cache_module, "load_cache", _fake_load)
    monkeypatch.setattr(features_module, "extract_features", _fake_features)
    args = build_parser().parse_args(
        ["cut", str(video), "--padding", "0", "--output", "compilation",
         "--output-dir", str(tmp_path / "output")]
    )
    assert cut(args) == 0
    session = tmp_path / "output" / video.stem
    assert (session / "attempts.json").is_file()
    assert (session / "source.json").is_file()
    assert (session / "run.json").is_file()
    assert (session / "shadows.json").is_file()
    assert (session / "serves.mov").is_file()
    assert video.read_bytes() == before
    out = capsys.readouterr().out
    assert "serves.mov" in out


def test_cut_integration_empty_detection_no_media(tmp_path: Path, capsys, monkeypatch) -> None:
    import shutil

    from media_factory import generate_fixture, landscape_spec
    from serve_review.pose import cache as cache_module
    from serve_review.pose import extract as extract_module
    from serve_review.detection import features as features_module

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        __import__("pytest").skip("FFmpeg and ffprobe are required")
    video = generate_fixture(tmp_path / "session.mov", landscape_spec(duration_seconds=2.0))

    def _fake_extract(video_p, cache_path, **kwargs):
        from types import SimpleNamespace as _NS

        target = Path(cache_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"{}" + b"\n")
        return _NS(cache_hit=True, cache_path=target)

    def _fake_load(cache_path):
        from types import SimpleNamespace as _NS

        return _NS(frames=())

    def _fake_features(observations, config):
        return ()

    monkeypatch.setattr(extract_module, "extract_poses", _fake_extract)
    monkeypatch.setattr(cache_module, "load_cache", _fake_load)
    monkeypatch.setattr(features_module, "extract_features", _fake_features)
    args = build_parser().parse_args(
        ["cut", str(video), "--output", "both", "--output-dir", str(tmp_path / "output")]
    )
    assert cut(args) == 0
    session = tmp_path / "output" / video.stem
    assert "no serves detected" in capsys.readouterr().out.lower()
    assert not (session / "serves.mov").exists()
    assert not (session / "clips").exists()


def test_probe_help_documents_command(capsys) -> None:
    top_help = build_parser().format_help()
    assert "probe" in top_help
    assert "source.json" in top_help

    parser = build_parser()
    with __import__("pytest").raises(SystemExit) as excinfo:
        parser.parse_args(["probe", "--help"])
    assert excinfo.value.code == 0
    probe_help = capsys.readouterr().out
    assert "ffprobe" in probe_help.lower()
    assert "--output" in probe_help


def test_probe_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(["probe", "session.mov"])

    assert args.video == Path("session.mov")
    assert args.output is None
    assert args.ffprobe == "ffprobe"


def test_probe_rejects_missing_input(tmp_path: Path, capsys) -> None:
    args = build_parser().parse_args(["probe", str(tmp_path / "missing.mov")])

    assert probe(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_probe_success_writes_source_json(tmp_path, monkeypatch, capsys) -> None:
    import json

    from serve_review.domain import SourceMetadata
    from serve_review.media import probe as probe_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    dest = tmp_path / "out" / "source.json"
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "120/1",
                "r_frame_rate": "120/1",
                "time_base": "1/90000",
                "duration": "4.0",
                "side_data_list": [
                    {"side_data_type": "Display Matrix", "rotation": -90}
                ],
            }
        ],
        "format": {"duration": "4.0"},
    }

    import subprocess

    def _fake_run(argv, **kwargs):
        assert isinstance(argv, list)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(probe_module.subprocess, "run", _fake_run)
    args = build_parser().parse_args(["probe", str(source), "--output", str(dest)])

    assert probe(args) == 0
    assert dest.is_file()
    stored = SourceMetadata.from_json(dest.read_text(encoding="utf-8"))
    assert stored.rotation_degrees == 270
    assert (stored.frame_rate_num, stored.frame_rate_den) == (120, 1)
    assert str(dest) in capsys.readouterr().out


def test_probe_reports_probe_failure(tmp_path, monkeypatch, capsys) -> None:
    import subprocess

    from serve_review.media import probe as probe_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    dest = tmp_path / "source.json"

    def _fake_fail(argv, **kwargs):
        return subprocess.CompletedProcess(
            args=argv, returncode=1, stdout="", stderr="Invalid data"
        )

    monkeypatch.setattr(probe_module.subprocess, "run", _fake_fail)
    args = build_parser().parse_args(["probe", str(source), "--output", str(dest)])

    assert probe(args) == 1
    assert "ERROR" in capsys.readouterr().err
    assert not dest.exists()


def test_export_help_documents_command(capsys) -> None:
    top_help = build_parser().format_help()
    assert "export" in top_help

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["export", "--help"])
    assert excinfo.value.code == 0
    export_help = capsys.readouterr().out
    assert "--ranges" in export_help
    assert "compilation" in export_help


def test_export_defaults() -> None:
    args = build_parser().parse_args(
        ["export", "session.mov", "--ranges", "ranges.json"]
    )

    assert args.video == Path("session.mov")
    assert args.ranges == Path("ranges.json")
    assert args.output == "compilation"
    assert args.output_dir == Path("output")
    assert args.overwrite is False
    assert args.ffmpeg == "ffmpeg"
    assert args.ffprobe == "ffprobe"


def test_export_requires_ranges_option() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["export", "session.mov"])


def test_export_rejects_missing_video(tmp_path: Path, capsys) -> None:
    ranges = tmp_path / "ranges.json"
    ranges.write_text("[]", encoding="utf-8")
    args = build_parser().parse_args(
        ["export", str(tmp_path / "missing.mov"), "--ranges", str(ranges)]
    )

    assert export_cmd(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_export_rejects_missing_ranges_file(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source.mov"
    source.touch()
    args = build_parser().parse_args(
        ["export", str(source), "--ranges", str(tmp_path / "missing.json")]
    )

    assert export_cmd(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_export_rejects_malformed_ranges_json(tmp_path: Path, capsys) -> None:
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=1.0)
    )
    bad = tmp_path / "ranges.json"
    bad.write_text("{not json", encoding="utf-8")
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(bad), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "invalid ranges JSON" in capsys.readouterr().err


def test_export_rejects_empty_ranges(tmp_path: Path, capsys) -> None:
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=1.0)
    )
    ranges = tmp_path / "ranges.json"
    ranges.write_text("[]", encoding="utf-8")
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "at least one range" in capsys.readouterr().err


def test_export_rejects_overlapping_ranges(tmp_path: Path, capsys) -> None:
    import json
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=2.0)
    )
    ranges = tmp_path / "ranges.json"
    ranges.write_text(
        json.dumps(
            [
                {"start_seconds": 0.0, "end_seconds": 0.6},
                {"start_seconds": 0.4, "end_seconds": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "invalid ranges" in capsys.readouterr().err


def test_export_rejects_out_of_bounds_ranges(tmp_path: Path, capsys) -> None:
    import json
    import shutil

    from media_factory import ffmpeg_available, generate_fixture, landscape_spec

    if not ffmpeg_available() or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=1.0)
    )
    ranges = tmp_path / "ranges.json"
    ranges.write_text(
        json.dumps([{"start_seconds": 0.5, "end_seconds": 5.0}]),
        encoding="utf-8",
    )
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 2
    assert "invalid ranges" in capsys.readouterr().err


def _make_export_fixture(tmp_path: Path):
    from media_factory import generate_fixture, landscape_spec
    from serve_review.media.probe import probe_source

    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=2.0)
    )
    meta = probe_source(video)
    return video, meta


def _write_ranges(tmp_path: Path, entries) -> Path:
    import json

    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_export_compilation_mode_reports_output(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    from serve_review.media.probe import probe_source

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path,
        [
            {"start_seconds": 0.2, "end_seconds": 0.7},
            {"start_seconds": 1.2, "end_seconds": 1.8},
        ],
    )
    out_base = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output",
            "compilation",
            "--output-dir",
            str(out_base),
        ]
    )
    before = video.read_bytes()

    assert export_cmd(args) == 0
    comp = out_base / video.stem / "serves.mov"
    assert comp.is_file()
    assert str(comp) in capsys.readouterr().out
    probed = probe_source(comp)
    assert probed.duration_seconds == _pytest.approx(
        1.1, abs=1.0 / meta.frames_per_second + 0.02
    )
    assert video.read_bytes() == before


def test_export_clips_and_both_modes(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    from serve_review.media.probe import probe_source

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path, [{"start_seconds": 0.2, "end_seconds": 0.7}]
    )

    clips_base = tmp_path / "out-clips"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output",
            "clips",
            "--output-dir",
            str(clips_base),
        ]
    )
    assert export_cmd(args) == 0
    clip = clips_base / video.stem / "clips" / "serve-001.mov"
    assert clip.is_file()
    assert str(clip) in capsys.readouterr().out
    assert not (clips_base / video.stem / "serves.mov").exists()

    both_base = tmp_path / "out-both"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output",
            "both",
            "--output-dir",
            str(both_base),
        ]
    )
    assert export_cmd(args) == 0
    assert (both_base / video.stem / "serves.mov").is_file()
    assert (both_base / video.stem / "clips" / "serve-001.mov").is_file()
    out = capsys.readouterr().out
    assert "serves.mov" in out and "serve-001.mov" in out
    probed = probe_source(both_base / video.stem / "serves.mov")
    assert probed.duration_seconds == _pytest.approx(
        0.5, abs=1.0 / meta.frames_per_second + 0.02
    )


def test_export_accepts_full_plan_document(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    from serve_review.domain import ExportPlan, MediaRange

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    plan = ExportPlan.for_source(meta, [MediaRange(0.2, 0.7)])
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.to_json(), encoding="utf-8")
    out_base = tmp_path / "out"
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(plan_path),
            "--output-dir",
            str(out_base),
        ]
    )

    assert export_cmd(args) == 0
    assert (out_base / video.stem / "serves.mov").is_file()


def test_export_rejects_plan_fingerprint_mismatch(tmp_path: Path, capsys) -> None:
    import json
    import shutil

    from media_factory import generate_fixture, landscape_spec
    from serve_review.domain import ExportPlan, MediaRange
    from serve_review.media.probe import probe_source

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video = generate_fixture(
        tmp_path / "source.mov", landscape_spec(duration_seconds=2.0)
    )
    meta = probe_source(video)
    plan = ExportPlan.for_source(meta, [MediaRange(0.2, 0.7)])
    payload = plan.to_dict()
    payload["source_fingerprint"] = "sha256:" + "f" * 64
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(plan_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert export_cmd(args) == 2
    assert "fingerprint" in capsys.readouterr().err


def test_export_collision_reports_actionable_error(tmp_path: Path, capsys) -> None:
    import shutil

    import pytest as _pytest

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path, [{"start_seconds": 0.2, "end_seconds": 0.7}]
    )
    out_base = tmp_path / "out"
    first = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(out_base)]
    )
    assert export_cmd(first) == 0
    capsys.readouterr()

    second = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(out_base)]
    )
    assert export_cmd(second) == 1
    assert "collision" in capsys.readouterr().err.lower()

    retry = build_parser().parse_args(
        [
            "export",
            str(video),
            "--ranges",
            str(ranges),
            "--output-dir",
            str(out_base),
            "--overwrite",
        ]
    )
    assert export_cmd(retry) == 0


def test_export_reports_ffmpeg_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    import shutil
    import subprocess

    from serve_review.media import export as export_module

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    video, meta = _make_export_fixture(tmp_path)
    ranges = _write_ranges(
        tmp_path, [{"start_seconds": 0.2, "end_seconds": 0.7}]
    )
    real_run = subprocess.run

    def _fail_ffmpeg_only(args, **kwargs):
        if str(args[0]).endswith("ffprobe"):
            return real_run(args, **kwargs)
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="boom"
        )

    monkeypatch.setattr(export_module.subprocess, "run", _fail_ffmpeg_only)
    args = build_parser().parse_args(
        ["export", str(video), "--ranges", str(ranges), "--output-dir", str(tmp_path / "out")]
    )

    assert export_cmd(args) == 1
    assert "ERROR" in capsys.readouterr().err
    assert not (tmp_path / "out" / video.stem / "serves.mov").exists()


# --- extract-poses diagnostic command (M2.5) ---

def test_extract_poses_defaults() -> None:
    from pathlib import Path as _Path

    args = build_parser().parse_args(["extract-poses", "session.mov"])

    assert args.video == _Path("session.mov")
    assert args.model == _Path("models/pose_landmarker_heavy.task")
    assert args.sample_rate == 30.0
    assert args.cache is None
    assert args.output_dir == _Path("output")
    assert args.overwrite is False
    assert args.ffmpeg == "ffmpeg"
    assert args.ffprobe == "ffprobe"


def test_extract_poses_help_documents_command(capsys) -> None:
    top_help = build_parser().format_help()
    assert "extract-poses" in top_help

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["extract-poses", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--sample-rate" in help_text
    assert "--model" in help_text
    assert "--cache" in help_text


def test_extract_poses_rejects_missing_video(tmp_path: Path, capsys) -> None:
    from serve_review.cli import extract_poses_cmd

    args = build_parser().parse_args(["extract-poses", str(tmp_path / "missing.mov")])

    assert extract_poses_cmd(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_extract_poses_rejects_bad_sample_rate(tmp_path: Path, capsys) -> None:
    from serve_review.cli import extract_poses_cmd

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake")
    for bad in ("0", "-3", "500"):
        args = build_parser().parse_args(
            ["extract-poses", str(source), "--sample-rate", bad]
        )
        assert extract_poses_cmd(args) == 2
        assert "sample-rate" in capsys.readouterr().err


def test_extract_poses_success_reports_cache_path(tmp_path: Path, capsys, monkeypatch) -> None:
    from pathlib import Path as _Path

    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    cache = tmp_path / "pose-v1.jsonl"
    result = extract_module.ExtractionResult(
        video=source,
        cache_path=cache,
        model_name="fake-heavy",
        model_version="fake-v1",
        sampling_rate_hz=30.0,
        sampling_start_seconds=0.0,
        frame_count=6,
        cached_frames=0,
        inferred_frames=6,
        cache_hit=False,
        complete=True,
    )
    seen: dict = {}

    def _fake_extract(video, cache_path, **kwargs):
        seen["video"] = _Path(video)
        seen["cache"] = _Path(cache_path)
        seen["rate"] = kwargs.get("sample_rate_hz")
        seen["model"] = _Path(kwargs.get("model_path"))
        assert kwargs.get("overwrite") is False
        return result

    monkeypatch.setattr(extract_module, "extract_poses", _fake_extract)
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--cache", str(cache)]
    )

    assert extract_poses_cmd(args) == 0
    out = capsys.readouterr().out
    assert str(cache) in out
    assert "6 frames" in out
    assert seen["video"] == source
    assert seen["cache"] == cache
    assert seen["rate"] == 30.0


def test_extract_poses_reports_cache_hit(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    cache = tmp_path / "pose-v1.jsonl"
    result = extract_module.ExtractionResult(
        video=source,
        cache_path=cache,
        model_name="fake-heavy",
        model_version="fake-v1",
        sampling_rate_hz=30.0,
        sampling_start_seconds=0.0,
        frame_count=4,
        cached_frames=4,
        inferred_frames=0,
        cache_hit=True,
        complete=True,
    )
    monkeypatch.setattr(
        extract_module, "extract_poses", lambda *a, **k: result
    )
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--cache", str(cache)]
    )

    assert extract_poses_cmd(args) == 0
    assert "cache hit" in capsys.readouterr().out


def test_extract_poses_uses_default_cache_layout(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "session.mov"
    source.write_bytes(b"fake-video")
    captured: dict = {}

    def _fake_extract(video, cache_path, **kwargs):
        from pathlib import Path as _Path

        captured["cache"] = _Path(cache_path)
        result = extract_module.ExtractionResult(
            video=_Path(video),
            cache_path=_Path(cache_path),
            model_name="m",
            model_version="v",
            sampling_rate_hz=30.0,
            sampling_start_seconds=0.0,
            frame_count=1,
            cached_frames=0,
            inferred_frames=1,
            cache_hit=False,
            complete=True,
        )
        return result

    monkeypatch.setattr(extract_module, "extract_poses", _fake_extract)
    out_base = tmp_path / "out"
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--output-dir", str(out_base)]
    )

    assert extract_poses_cmd(args) == 0
    assert captured["cache"] == out_base / "session" / "cache" / "pose-v1.jsonl"


def test_extract_poses_reports_failures_actionably(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")

    def _boom(video, cache_path, **kwargs):
        raise extract_module.ExtractionError("fake probe failure")

    monkeypatch.setattr(extract_module, "extract_poses", _boom)
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--cache", str(tmp_path / "p.jsonl")]
    )

    assert extract_poses_cmd(args) == 1
    assert "ERROR" in capsys.readouterr().err


# --- extract-poses --overlay diagnostic flag ---

def test_extract_poses_overlay_defaults_off() -> None:
    args = build_parser().parse_args(["extract-poses", "session.mov"])

    assert args.overlay is None


def test_extract_poses_help_documents_overlay(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["extract-poses", "--help"])
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--overlay" in help_text


def test_extract_poses_forwards_overlay_path(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    overlay = tmp_path / "overlay.mp4"
    cache = tmp_path / "pose-v1.jsonl"
    seen: dict = {}

    def _fake_extract(video, cache_path, **kwargs):
        from pathlib import Path as _Path

        seen["overlay_path"] = kwargs.get("overlay_path")
        seen["overlay_progress"] = kwargs.get("overlay_progress_callback")
        overlay_dest = _Path(kwargs.get("overlay_path"))
        return extract_module.ExtractionResult(
            video=_Path(video),
            cache_path=_Path(cache_path),
            model_name="m",
            model_version="v",
            sampling_rate_hz=30.0,
            sampling_start_seconds=0.0,
            frame_count=2,
            cached_frames=0,
            inferred_frames=2,
            cache_hit=False,
            complete=True,
            overlay_path=overlay_dest,
        )

    monkeypatch.setattr(extract_module, "extract_poses", _fake_extract)
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--cache", str(cache),
         "--overlay", str(overlay)]
    )

    assert extract_poses_cmd(args) == 0
    assert seen["overlay_path"] == overlay.expanduser()
    assert callable(seen["overlay_progress"])
    out = capsys.readouterr().out
    assert "overlay:" in out
    assert str(overlay) in out


def test_extract_poses_without_overlay_passes_none(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")
    cache = tmp_path / "pose-v1.jsonl"
    seen: dict = {}

    def _fake_extract(video, cache_path, **kwargs):
        from pathlib import Path as _Path

        seen["overlay_path"] = kwargs.get("overlay_path")
        seen["overlay_progress"] = kwargs.get("overlay_progress_callback")
        return extract_module.ExtractionResult(
            video=_Path(video),
            cache_path=_Path(cache_path),
            model_name="m",
            model_version="v",
            sampling_rate_hz=30.0,
            sampling_start_seconds=0.0,
            frame_count=1,
            cached_frames=0,
            inferred_frames=1,
            cache_hit=False,
            complete=True,
        )

    monkeypatch.setattr(extract_module, "extract_poses", _fake_extract)
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--cache", str(cache)]
    )

    assert extract_poses_cmd(args) == 0
    assert seen["overlay_path"] is None
    assert seen["overlay_progress"] is None
    assert "overlay:" not in capsys.readouterr().out


def test_extract_poses_reports_overlay_failure(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")

    def _boom(video, cache_path, **kwargs):
        raise extract_module.ExtractionError(
            "could not write pose overlay to overlay.mp4: ffmpeg failed."
        )

    monkeypatch.setattr(extract_module, "extract_poses", _boom)
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--cache", str(tmp_path / "p.jsonl"),
         "--overlay", str(tmp_path / "overlay.mp4")]
    )

    assert extract_poses_cmd(args) == 1
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "overlay" in err


def test_extract_poses_reports_overlay_cancellation(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review.cli import extract_poses_cmd
    from serve_review.pose import extract as extract_module

    source = tmp_path / "clip.mov"
    source.write_bytes(b"fake-video")

    def _cancelled(video, cache_path, **kwargs):
        raise extract_module.ExtractionCancelled("pose overlay was cancelled.")

    monkeypatch.setattr(extract_module, "extract_poses", _cancelled)
    args = build_parser().parse_args(
        ["extract-poses", str(source), "--cache", str(tmp_path / "p.jsonl"),
         "--overlay", str(tmp_path / "overlay.mp4")]
    )

    assert extract_poses_cmd(args) == 1
    assert "cancelled" in capsys.readouterr().err


# --- analyze command ------------------------------------------------------------


def test_analyze_defaults() -> None:
    args = build_parser().parse_args(["analyze", "session.mov"])

    assert args.video == Path("session.mov")
    assert args.attempts is None
    assert args.output_dir == Path("output")
    assert args.overwrite is False
    assert args.ffmpeg == "ffmpeg"
    assert args.ffprobe == "ffprobe"


def test_analyze_help_documents_attempts_option(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["analyze", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--attempts" in out
    assert "checkpoints.json" in out


def test_analyze_rejects_missing_input(tmp_path: Path, capsys) -> None:
    from serve_review.cli import analyze

    args = build_parser().parse_args(["analyze", str(tmp_path / "missing.mov")])

    assert analyze(args) == 2
    assert "does not exist" in capsys.readouterr().err


def test_analyze_rejects_missing_attempts_file(tmp_path: Path, capsys) -> None:
    from serve_review.cli import analyze

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")
    args = build_parser().parse_args(
        ["analyze", str(source), "--attempts", str(tmp_path / "nope.json")]
    )

    assert analyze(args) == 2
    assert "attempts file does not exist" in capsys.readouterr().err


def test_analyze_success_reports_checkpoints(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review import analysis_pipeline as analysis_module
    from serve_review.cli import analyze

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")
    (tmp_path / "attempts.json").write_text("{}", encoding="utf-8")
    checkpoints = tmp_path / "output" / source.stem / "checkpoints.json"
    result = SimpleNamespace(
        checkpoints_path=checkpoints,
        phase_document=["serve-001", "serve-002"],
        attempt_failures=(),
        empty=False,
    )
    seen: dict = {}

    def _fake_run_analyze(video, **kwargs):
        seen.update(kwargs)
        seen["video"] = Path(video)
        return result

    monkeypatch.setattr(analysis_module, "run_analyze", _fake_run_analyze)
    args = build_parser().parse_args(
        [
            "analyze",
            str(source),
            "--attempts",
            str(tmp_path / "attempts.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--overwrite",
        ]
    )
    assert analyze(args) == 0
    out = capsys.readouterr().out
    assert str(checkpoints) in out
    assert "2 attempt(s)" in out
    assert seen["attempts_path"] == Path(tmp_path / "attempts.json")
    assert seen["overwrite"] is True


def test_analyze_empty_result_reports_empty_checkpoints(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from serve_review import analysis_pipeline as analysis_module
    from serve_review.cli import analyze

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")
    checkpoints = tmp_path / "output" / source.stem / "checkpoints.json"
    result = SimpleNamespace(
        checkpoints_path=checkpoints,
        phase_document=[],
        attempt_failures=(),
        empty=True,
    )

    def _fake_run_analyze(video, **kwargs):
        return result

    monkeypatch.setattr(analysis_module, "run_analyze", _fake_run_analyze)
    args = build_parser().parse_args(["analyze", str(source)])

    assert analyze(args) == 0
    out = capsys.readouterr().out
    assert str(checkpoints) in out
    assert "empty checkpoints" in out


def test_analyze_reports_isolated_failures(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review import analysis_pipeline as analysis_module
    from serve_review.cli import analyze

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")
    (tmp_path / "attempts.json").write_text("{}", encoding="utf-8")
    checkpoints = tmp_path / "output" / source.stem / "checkpoints.json"
    result = SimpleNamespace(
        checkpoints_path=checkpoints,
        phase_document=["serve-001", "serve-002"],
        attempt_failures=(SimpleNamespace(attempt_id="serve-002"),),
        empty=False,
    )

    def _fake_run_analyze(video, **kwargs):
        return result

    monkeypatch.setattr(analysis_module, "run_analyze", _fake_run_analyze)
    args = build_parser().parse_args(["analyze", str(source)])

    assert analyze(args) == 0
    assert "1 isolated phase failure" in capsys.readouterr().out


def test_analyze_stage_error_reports_stage(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review import analysis_pipeline as analysis_module
    from serve_review.cli import analyze

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")

    def _boom(video, **kwargs):
        raise analysis_module.AnalyzeError("pose", "pose cache is stale: x.")

    monkeypatch.setattr(analysis_module, "run_analyze", _boom)
    args = build_parser().parse_args(["analyze", str(source)])

    assert analyze(args) == 1
    err = capsys.readouterr().err
    assert "analyze failed at pose" in err
    assert "stale" in err


def test_analyze_cancellation_reports_stage(tmp_path: Path, capsys, monkeypatch) -> None:
    from serve_review import analysis_pipeline as analysis_module
    from serve_review.cli import analyze

    source = tmp_path / "source.mov"
    source.write_bytes(b"fake-source")

    def _cancelled(video, **kwargs):
        raise analysis_module.AnalyzeCancelled("audio", "cancelled at audio.")

    monkeypatch.setattr(analysis_module, "run_analyze", _cancelled)
    args = build_parser().parse_args(["analyze", str(source)])

    assert analyze(args) == 1
    err = capsys.readouterr().err
    assert "cancelled at audio" in err
