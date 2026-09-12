"""Focused tests for the local phase-review renderer.

Fully faked/synthetic only: no private footage, no network, no model
inference. Pose caches are real JSONL files in temporary directories
except where injected fakes isolate a stage; frame sampling and JPEG
encoding are injected fakes except for one generated-media integration
that exercises the real sampler/FFmpeg path when available.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from serve_review.domain import (
    STAGE_ORDER,
    AttemptPhase,
    MediaRange,
    PhaseDocument,
    SourceMetadata,
    StagePhase,
)
from serve_review.phase_review import (
    INDEX_HTML_FILENAME,
    MANIFEST_TXT_FILENAME,
    POSE_SUPPORT_TOLERANCE_SECONDS,
    REVIEW_JSON_FILENAME,
    ReviewCancelled,
    ReviewError,
    build_caption_lines,
    default_checkpoints_path_for,
    default_review_dir_for,
    find_nearest_pose,
    render_caption,
    run_review,
)
from serve_review.pose import cache as cache_module
from serve_review.pose import extract as extract_module
from serve_review.pose import overlay as overlay_module
from serve_review.pose.mediapipe import MODEL_NAME, MODEL_VERSION
from serve_review.pose.schema import (
    NUM_KEYPOINTS,
    BodyKeypoint,
    CacheIdentity,
    FrameObservation,
    PersonBox,
    PersonObservation,
)

FINGERPRINT = "sha256:phase-review-fixture"


def make_metadata(duration: float = 10.0, fingerprint: str = FINGERPRINT) -> SourceMetadata:
    return SourceMetadata(
        fingerprint=fingerprint,
        duration_seconds=duration,
        width=160,
        height=120,
        frame_rate_num=30,
        frame_rate_den=1,
        video_codec="h264",
        rotation_degrees=0,
    )


def expected_identity(metadata: SourceMetadata) -> CacheIdentity:
    return CacheIdentity(
        source_fingerprint=metadata.fingerprint,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )


def make_person_two_joints() -> PersonObservation:
    joints: list[BodyKeypoint | None] = [None] * NUM_KEYPOINTS
    joints[11] = BodyKeypoint(x=0.25, y=0.25, visibility=0.9)
    joints[13] = BodyKeypoint(x=0.35, y=0.45, visibility=0.9)
    joints[0] = BodyKeypoint(x=0.5, y=0.1, visibility=0.9)
    return PersonObservation(
        box=PersonBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9),
        keypoints=tuple(joints),
        score=0.9,
    )


def make_obs(moment: float, with_person: bool = True) -> FrameObservation:
    persons = (make_person_two_joints(),) if with_person else ()
    return FrameObservation(
        time_seconds=moment,
        timestamp_ms=round(moment * 1000),
        persons=persons,
    )


def make_stage(
    key: str,
    start: float,
    end: float,
    keyframe: float | None,
    *,
    availability: str = "available",
    provenance: str | None = None,
    confidence: float = 0.8,
) -> StagePhase:
    if availability == "unavailable":
        return StagePhase(
            availability="unavailable",
            provenance="body_pose" if key != "contact" else "audio_transient",
            confidence=0.0,
            interval=None,
            keyframe_seconds=None,
            temporal_uncertainty_seconds=None,
            evidence=(),
            limitations=("test_unavailable",),
        )
    prov = provenance
    if prov is None:
        prov = "audio_transient" if key == "contact" else "body_pose"
        if key == "contact" and availability in ("available", "partial"):
            prov = "body_pose_audio"
    unc = 0.02 if (key == "contact" and "audio" in prov) else None
    return StagePhase(
        availability=availability,
        provenance=prov,  # type: ignore[arg-type]
        confidence=confidence,
        interval=MediaRange(start_seconds=start, end_seconds=end),
        keyframe_seconds=keyframe,
        temporal_uncertainty_seconds=unc,
        evidence=("test_evidence",),
        limitations=("body_pose_estimate",),
    )


def make_attempt_phase(attempt_id: str, base: float, *, unavailable: tuple[str, ...] = ()) -> AttemptPhase:
    # Eight chronological 0.4s intervals inside [base, base+3.2].
    stages: dict[str, StagePhase] = {}
    for idx, key in enumerate(STAGE_ORDER):
        s = round(base + idx * 0.4, 6)
        e = round(s + 0.4, 6)
        k = round(s + 0.2, 6)
        if key in unavailable:
            stages[key] = make_stage(key, s, e, None, availability="unavailable")
        else:
            stages[key] = make_stage(key, s, e, k)
    counts = {"available": sum(1 for v in stages.values() if v.availability == "available")}
    if len(unavailable) == 0:
        status = "complete"
    elif len(unavailable) == len(STAGE_ORDER):
        status = "unavailable"
    else:
        status = "partial"
    return AttemptPhase(
        attempt_id=attempt_id,
        attempt_range=MediaRange(start_seconds=base, end_seconds=round(base + 3.2, 6)),
        method_version="test-solver-v1",
        config_id="test-config",
        stages=stages,
        structural_status=status,  # type: ignore[arg-type]
        anomalies=("test_anomaly",) if status != "complete" else (),
    )


def make_document(phases: list[AttemptPhase], metadata: SourceMetadata) -> PhaseDocument:
    return PhaseDocument(
        source_fingerprint=metadata.fingerprint,
        source_duration_seconds=metadata.duration_seconds,
        attempts=tuple(phases),
    )


def write_checkpoints(path: Path, document: PhaseDocument) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.to_json(), encoding="utf-8")
    return path


def write_cache(path: Path, identity: CacheIdentity, moments: list[float]) -> Path:
    frames = [make_obs(m) for m in moments]
    return cache_module.write_complete_cache(path, identity, frames)


def fake_sampler_factory(images: dict[float, np.ndarray]):
    def _sample(video_path: Path, requested: float, metadata: SourceMetadata):
        # Deterministic gray image; actual time equals requested.
        # Large enough that the caption banner covers only the top.
        img = np.full((240, 320, 3), 128, dtype=np.uint8)
        return SimpleNamespace(time_seconds=float(requested), image=img)
    return _sample


def capturing_encoder(store: dict):
    def _encode(image: np.ndarray, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        store[str(dest)] = np.ascontiguousarray(image.copy())
        # Write deterministic fake JPEG bytes so files exist.
        dest.write_bytes(b"FAKEJPEG:" + np.ascontiguousarray(image).tobytes()[:32])
    return _encode


def run_ok(tmp_path: Path, metadata: SourceMetadata, document: PhaseDocument, moments: list[float], **kw):
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake-video-bytes")
    out_base = tmp_path / "output"
    ckpt = out_base / video.stem / "checkpoints.json"
    write_checkpoints(ckpt, document)
    cache_path = out_base / video.stem / "cache" / "pose-v1.jsonl"
    write_cache(cache_path, expected_identity(metadata), moments)
    store: dict = {}
    result = run_review(
        video,
        output_dir=out_base,
        probe_fn=lambda p: metadata,
        sample_frame_fn=fake_sampler_factory({}),
        encode_jpeg_fn=capturing_encoder(store),
        **kw,
    )
    return video, out_base, result, store


# --- caption / support unit tests -------------------------------------------


def test_caption_lines_label_all_required_fields() -> None:
    lines = build_caption_lines(
        "serve-001", "contact",
        requested_time=1.234, actual_time=1.240,
        availability="available", provenance="body_pose_audio",
        confidence=0.87, support_time=1.233, support_delta=-0.001,
        anomalies=("wobble",),
    )
    blob = "\n".join(lines)
    assert "serve-001" in blob
    assert "contact" in blob
    assert "1.234" in blob
    assert "1.240" in blob
    assert "available" in blob
    assert "body_pose_audio" in blob
    assert "0.87" in blob
    assert "wobble" in blob
    # Never claim exact visual observation.
    assert "visual" in blob.lower() or "estimate" in blob.lower()
    assert "observ" not in blob.lower().replace("observation", "X").replace("visual observation", "X")


def test_contact_caption_carries_estimate_disclaimer() -> None:
    lines = build_caption_lines(
        "serve-002", "contact",
        requested_time=2.0, actual_time=2.0,
        availability="partial", provenance="audio_transient",
        confidence=0.5, support_time=2.0, support_delta=0.0,
        anomalies=(),
    )
    assert any("not visual observation" in line for line in lines)


def test_render_caption_burns_banner_deterministically() -> None:
    image = np.full((48, 64, 3), 128, dtype=np.uint8)
    first = render_caption(image, ["SERVE-001 CONTACT T=1.000S"])
    second = render_caption(image, ["SERVE-001 CONTACT T=1.000S"])
    assert np.array_equal(first, second)
    # Banner at top differs from source.
    assert not np.array_equal(first, image)
    assert (first[0, 0] == np.array([0, 0, 0], dtype=np.uint8)).all()
    # Glyph ink present (white pixels in banner).
    assert (first[: first.shape[0]] == 255).any()
    # Input never mutated.
    assert (image == 128).all()


def test_find_nearest_pose_exact_and_signed_delta() -> None:
    obs = [make_obs(1.0), make_obs(1.2), make_obs(1.5)]
    best, delta = find_nearest_pose(obs, 1.2)
    assert best.time_seconds == pytest.approx(1.2)
    assert delta == pytest.approx(0.0)
    best2, delta2 = find_nearest_pose(obs, 1.27)
    assert best2.time_seconds == pytest.approx(1.2)
    assert delta2 == pytest.approx(-0.07)


def test_default_paths() -> None:
    assert default_checkpoints_path_for("v.mov", "output") == Path("output/v/checkpoints.json")
    assert default_review_dir_for("v.mov", "output") == Path("output/v/review-phases")


# --- happy path --------------------------------------------------------------


def test_rendered_images_carry_skeleton_and_caption(tmp_path: Path) -> None:
    metadata = make_metadata()
    phase = make_attempt_phase("serve-001", 1.0)
    document = make_document([phase], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8)]
    video, out_base, result, store = run_ok(tmp_path, metadata, document, keyframes)
    assert result.image_count == 8
    assert result.entry_count == 8
    assert (result.review_dir / REVIEW_JSON_FILENAME).is_file()
    assert (result.review_dir / INDEX_HTML_FILENAME).is_file()
    assert (result.review_dir / MANIFEST_TXT_FILENAME).is_file()
    # One image per phase under attempt-stable directories with safe names.
    for key in STAGE_ORDER:
        assert (result.review_dir / "serve-001" / f"{key}.jpg").is_file()
    # Captured arrays contain skeleton line color and caption banner.
    # (Encoder captures staging paths; match by filename after rename.)
    sample = next(v for k, v in store.items() if k.endswith("serve-001/start.jpg") or k.endswith("start.jpg"))
    assert (sample[0, 0] == np.array([0, 0, 0], dtype=np.uint8)).all()  # caption banner
    assert ((sample == np.array(overlay_module.LINE_COLOR, dtype=np.uint8)).all(axis=2)).any()
    # Manifest ordering deterministic: stage order.
    payload = json.loads((result.review_dir / REVIEW_JSON_FILENAME).read_text(encoding="utf-8"))
    assert [e["stage"] for e in payload["entries"]] == list(STAGE_ORDER)
    assert payload["phase_review_version"] == "phase-review-v1"
    entry = payload["entries"][0]
    assert entry["requested_source_time"] == pytest.approx(1.2)
    assert entry["actual_source_time"] == pytest.approx(1.2)
    assert entry["cache_time"] == pytest.approx(1.2)
    assert entry["cache_delta_seconds"] == pytest.approx(0.0)
    assert entry["image"] == "serve-001/start.jpg"
    assert entry["status"] == "rendered"
    # Source/checkpoints untouched.
    assert video.read_bytes() == b"fake-video-bytes"


def test_deterministic_output_across_runs(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8)]
    (tmp_path / "a").mkdir(parents=True, exist_ok=True)
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    _, _, first, store_a = run_ok(tmp_path / "a", metadata, document, keyframes)
    _, _, second, store_b = run_ok(tmp_path / "b", metadata, document, keyframes)
    text_a = (first.review_dir / REVIEW_JSON_FILENAME).read_bytes()
    text_b = (second.review_dir / REVIEW_JSON_FILENAME).read_bytes()
    # Same logical entries modulo absolute paths: compare entry lists.
    payload_a = json.loads(text_a.decode())
    payload_b = json.loads(text_b.decode())
    assert payload_a["entries"] == payload_b["entries"]
    for key in store_a:
        rel_a = Path(key).name
        match = [v for k, v in store_b.items() if Path(k).name == rel_a]
        assert match and np.array_equal(store_a[key], match[0])


def test_unavailable_phases_produce_no_image_but_appear_in_index(tmp_path: Path) -> None:
    metadata = make_metadata()
    phase = make_attempt_phase("serve-001", 1.0, unavailable=("contact", "finish"))
    document = make_document([phase], metadata)
    moments = [1.2, 1.6, 2.0, 2.4, 2.8, 3.6]
    video, out_base, result, store = run_ok(tmp_path, metadata, document, moments)
    assert result.image_count == 6
    assert not (result.review_dir / "serve-001" / "contact.jpg").exists()
    assert not (result.review_dir / "serve-001" / "finish.jpg").exists()
    payload = json.loads((result.review_dir / REVIEW_JSON_FILENAME).read_text(encoding="utf-8"))
    by_stage = {e["stage"]: e for e in payload["entries"]}
    assert by_stage["contact"]["image"] is None
    assert by_stage["contact"]["status"] == "unavailable"
    assert by_stage["finish"]["image"] is None
    html = (result.review_dir / INDEX_HTML_FILENAME).read_text(encoding="utf-8")
    assert "contact" in html and "no image" in html


def test_missing_keyframe_phase_has_no_image(tmp_path: Path) -> None:
    metadata = make_metadata()
    stages: dict[str, StagePhase] = {}
    for idx, key in enumerate(STAGE_ORDER):
        s = 1.0 + idx * 0.4
        if key == "release":
            stages[key] = StagePhase(
                availability="partial",
                provenance="body_pose",
                confidence=0.4,
                interval=MediaRange(start_seconds=s, end_seconds=s + 0.4),
                keyframe_seconds=None,
                temporal_uncertainty_seconds=None,
                evidence=(),
                limitations=("no_keyframe",),
            )
        else:
            stages[key] = make_stage(key, s, s + 0.4, s + 0.2)
    phase = AttemptPhase(
        attempt_id="serve-001",
        attempt_range=MediaRange(start_seconds=1.0, end_seconds=4.2),
        method_version="test-solver-v1",
        config_id="test-config",
        stages=stages,
        structural_status="partial",
        anomalies=(),
    )
    document = make_document([phase], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8) if i != 1]
    _, _, result, _ = run_ok(tmp_path, metadata, document, keyframes)
    payload = json.loads((result.review_dir / REVIEW_JSON_FILENAME).read_text(encoding="utf-8"))
    by_stage = {e["stage"]: e for e in payload["entries"]}
    assert by_stage["release"]["image"] is None
    assert by_stage["release"]["reason"] == "missing_keyframe"


# --- failure safety ----------------------------------------------------------


def test_missing_checkpoints_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata()
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=tmp_path / "output", probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "checkpoints"
    assert not (tmp_path / "output" / video.stem / "review-phases").exists()


def test_corrupt_checkpoints_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata()
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    ckpt = out_base / video.stem / "checkpoints.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "checkpoints"
    assert not (out_base / video.stem / "review-phases").exists()


def test_source_fingerprint_mismatch_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata()
    other = make_metadata(fingerprint="sha256:other")
    phase = make_attempt_phase("serve-001", 1.0)
    document = make_document([phase], other)
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "checkpoints"


def test_source_duration_mismatch_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata(duration=10.0)
    other = make_metadata(duration=11.0)
    phase = make_attempt_phase("serve-001", 1.0)
    document = make_document([phase], other)
    # Fix fingerprint to match so duration is the failure.
    document = PhaseDocument(
        source_fingerprint=metadata.fingerprint,
        source_duration_seconds=other.duration_seconds,
        attempts=document.attempts,
    )
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "checkpoints"


def test_missing_cache_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "pose"
    assert not (out_base / video.stem / "review-phases").exists()


def test_stale_cache_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    stale_identity = CacheIdentity(
        source_fingerprint="sha256:stale",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )
    cache_path = out_base / video.stem / "cache" / "pose-v1.jsonl"
    write_cache(cache_path, stale_identity, [1.2])
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "pose"


def test_corrupt_cache_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    cache_path = out_base / video.stem / "cache" / "pose-v1.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{corrupt\n", encoding="utf-8")
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "pose"
    assert not (out_base / video.stem / "review-phases").exists()


def test_incomplete_cache_fails_safely(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    cache_path = out_base / video.stem / "cache" / "pose-v1.jsonl"
    cache_module.write_partial_cache(cache_path, expected_identity(metadata), [make_obs(1.2)])
    with pytest.raises(ReviewError) as excinfo:
        run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert excinfo.value.stage == "pose"


def test_pose_support_beyond_tolerance_has_no_image(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    # Cache only at t=0.1: every keyframe (>=1.2) is beyond tolerance.
    _, _, result, _ = run_ok(tmp_path, metadata, document, [0.1])
    assert result.image_count == 0
    payload = json.loads((result.review_dir / REVIEW_JSON_FILENAME).read_text(encoding="utf-8"))
    assert all(e["image"] is None for e in payload["entries"])
    assert all(e["reason"] == "pose_support_beyond_tolerance" for e in payload["entries"])
    assert all(e["cache_delta_seconds"] is not None for e in payload["entries"])


def test_nearest_support_delta_is_labelled(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8)]
    moments = [k + 0.02 for k in keyframes]  # within tolerance, nonzero delta
    _, _, result, _ = run_ok(tmp_path, metadata, document, moments)
    payload = json.loads((result.review_dir / REVIEW_JSON_FILENAME).read_text(encoding="utf-8"))
    for entry in payload["entries"]:
        assert entry["cache_delta_seconds"] == pytest.approx(0.02, abs=1e-9)
        assert entry["status"] == "rendered"


# --- atomicity / collision / cancel ------------------------------------------


def test_collision_without_overwrite_leaves_no_partial_dir(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8)]
    video, out_base, result, _ = run_ok(tmp_path, metadata, document, keyframes)
    before = (result.review_dir / REVIEW_JSON_FILENAME).read_bytes()
    with pytest.raises(ReviewError) as excinfo:
        run_ok(tmp_path, metadata, document, keyframes)
    assert excinfo.value.stage == "review"
    assert "collision" in excinfo.value.message.lower()
    assert (result.review_dir / REVIEW_JSON_FILENAME).read_bytes() == before
    leftovers = [p for p in result.review_dir.parent.iterdir() if ".tmp-" in p.name]
    assert leftovers == []


def test_overwrite_replaces_review_dir(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8)]
    video, out_base, _, _ = run_ok(tmp_path, metadata, document, keyframes)
    store: dict = {}
    result = run_review(
        video,
        output_dir=out_base,
        probe_fn=lambda p: metadata,
        sample_frame_fn=fake_sampler_factory({}),
        encode_jpeg_fn=capturing_encoder(store),
        overwrite=True,
    )
    assert result.image_count == 8
    assert (result.review_dir / REVIEW_JSON_FILENAME).is_file()


def test_cancel_cleans_up_staging(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    write_cache(out_base / video.stem / "cache" / "pose-v1.jsonl", expected_identity(metadata), [1.2 + i * 0.4 for i in range(8)])
    with pytest.raises(ReviewCancelled):
        run_review(
            video,
            output_dir=out_base,
            probe_fn=lambda p: metadata,
            sample_frame_fn=fake_sampler_factory({}),
            encode_jpeg_fn=capturing_encoder({}),
            is_cancelled=lambda: True,
        )
    assert not (out_base / video.stem / "review-phases").exists()
    leftovers = [p for p in (out_base / video.stem).iterdir() if ".tmp-" in p.name]
    assert leftovers == []


def test_cancel_mid_run_removes_staging_and_final(tmp_path: Path) -> None:
    metadata = make_metadata()
    phases = [make_attempt_phase("serve-001", 1.0), make_attempt_phase("serve-002", 5.0)]
    document = make_document(phases, metadata)
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    moments = [1.2 + i * 0.4 for i in range(8)] + [5.2 + i * 0.4 for i in range(8)]
    write_cache(out_base / video.stem / "cache" / "pose-v1.jsonl", expected_identity(metadata), moments)
    calls = {"n": 0}

    def _sampling(video_path: Path, requested: float, meta: SourceMetadata):
        calls["n"] += 1
        if calls["n"] > 3:
            raise ReviewCancelled("review", "cancelled at frame 3")
        return SimpleNamespace(time_seconds=float(requested), image=np.full((48, 64, 3), 128, dtype=np.uint8))

    with pytest.raises(ReviewCancelled):
        run_review(
            video,
            output_dir=out_base,
            probe_fn=lambda p: metadata,
            sample_frame_fn=_sampling,
            encode_jpeg_fn=capturing_encoder({}),
        )
    assert not (out_base / video.stem / "review-phases").exists()
    leftovers = [p for p in (out_base / video.stem).iterdir() if ".tmp-" in p.name]
    assert leftovers == []


def test_does_not_overwrite_checkpoints_or_clips(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8)]
    video, out_base, result, _ = run_ok(tmp_path, metadata, document, keyframes)
    ckpt_before = (out_base / video.stem / "checkpoints.json").read_bytes()
    clips = out_base / video.stem / "clips" / "serve-001.mov"
    clips.parent.mkdir(parents=True, exist_ok=True)
    clips.write_bytes(b"clip-bytes")
    store: dict = {}
    run_review(
        video,
        output_dir=out_base,
        probe_fn=lambda p: metadata,
        sample_frame_fn=fake_sampler_factory({}),
        encode_jpeg_fn=capturing_encoder(store),
        overwrite=True,
    )
    assert (out_base / video.stem / "checkpoints.json").read_bytes() == ckpt_before
    assert clips.read_bytes() == b"clip-bytes"


def test_explicit_cache_and_output_paths(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([make_attempt_phase("serve-001", 1.0)], metadata)
    keyframes = [1.2 + i * 0.4 for i in range(8)]
    video = tmp_path / "session.mov"
    video.write_bytes(b"fake")
    ckpt = tmp_path / "custom" / "ck.json"
    write_checkpoints(ckpt, document)
    cache = tmp_path / "custom" / "pose.jsonl"
    write_cache(cache, expected_identity(metadata), keyframes)
    review = tmp_path / "custom" / "review-out"
    result = run_review(
        video,
        output_dir=tmp_path / "output",
        checkpoints_path=ckpt,
        cache_path=cache,
        output=review,
        probe_fn=lambda p: metadata,
        sample_frame_fn=fake_sampler_factory({}),
        encode_jpeg_fn=capturing_encoder({}),
    )
    assert result.review_dir == review
    assert (review / REVIEW_JSON_FILENAME).is_file()


def test_empty_document_writes_empty_index(tmp_path: Path) -> None:
    metadata = make_metadata()
    document = make_document([], metadata)
    _, _, result, _ = run_ok(tmp_path, metadata, document, [])
    assert result.empty is True
    assert result.image_count == 0
    payload = json.loads((result.review_dir / REVIEW_JSON_FILENAME).read_text(encoding="utf-8"))
    assert payload["entries"] == []


def test_generated_media_integration_when_ffmpeg_available(tmp_path: Path) -> None:
    import shutil as _shutil

    if _shutil.which("ffmpeg") is None or _shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    from media_factory import generate_fixture, landscape_spec
    from serve_review.media.probe import probe_source

    video = generate_fixture(tmp_path / "session.mov", landscape_spec(duration_seconds=2.0))
    metadata = probe_source(video)
    # Single available keyframe at 1.0s; cache support exactly at 1.0s.
    stages: dict[str, StagePhase] = {}
    for idx, key in enumerate(STAGE_ORDER):
        s = 0.2 + idx * 0.15
        e = s + 0.1
        k = s + 0.05
        if key in ("start", "contact"):
            stages[key] = make_stage(key, s, e, 1.0 if key == "contact" else k)
            if key == "contact":
                stages[key] = StagePhase(
                    availability="available",
                    provenance="audio_transient",
                    confidence=0.7,
                    interval=MediaRange(start_seconds=0.9, end_seconds=1.1),
                    keyframe_seconds=1.0,
                    temporal_uncertainty_seconds=0.02,
                    evidence=("audio",),
                    limitations=("audio_anchored",),
                )
        else:
            stages[key] = make_stage(key, s, e, None, availability="unavailable")
    phase = AttemptPhase(
        attempt_id="serve-001",
        attempt_range=MediaRange(start_seconds=0.2, end_seconds=1.8),
        method_version="test-solver-v1",
        config_id="test-config",
        stages=stages,
        structural_status="partial",
        anomalies=(),
    )
    document = PhaseDocument(
        source_fingerprint=metadata.fingerprint,
        source_duration_seconds=metadata.duration_seconds,
        attempts=(phase,),
    )
    out_base = tmp_path / "output"
    write_checkpoints(out_base / video.stem / "checkpoints.json", document)
    identity = CacheIdentity(
        source_fingerprint=metadata.fingerprint,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        sampling_rate_hz=extract_module.DEFAULT_SAMPLE_RATE_HZ,
        sampling_start_seconds=extract_module.DEFAULT_SAMPLING_START_SECONDS,
    )
    write_cache(out_base / video.stem / "cache" / "pose-v1.jsonl", identity, [0.35, 1.0])
    result = run_review(video, output_dir=out_base, probe_fn=lambda p: metadata)
    assert (result.review_dir / "serve-001" / "contact.jpg").is_file()
    assert (result.review_dir / "serve-001" / "contact.jpg").stat().st_size > 0
    assert not (result.review_dir / "serve-001" / "loading.jpg").exists()
