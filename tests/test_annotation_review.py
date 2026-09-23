"""The review UI keeps human labels separate from regenerable predictions."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest

from serve_review.annotation_review import ReviewError, ReviewServer, ReviewSession


def session(tmp_path: Path) -> ReviewSession:
    video = tmp_path / "serve.MOV"
    video.write_bytes(b"sample video bytes")
    metadata = tmp_path / "metadata" / "serve"
    metadata.mkdir(parents=True)
    (metadata / "source.json").write_text(json.dumps({
        "fingerprint": "sha256:test", "duration_seconds": 10,
        "frame_rate_num": 30, "frame_rate_den": 1,
    }))
    (metadata / "attempts.json").write_text(json.dumps({"attempts": [{
        "attempt_id": "serve-001",
        "detected_range": {"start_seconds": 2, "end_seconds": 5},
        "confidence": 0.4,
    }]}))
    checkpoint_dir = metadata / "attempts" / "serve-001"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoints.json").write_text(json.dumps({"attempts": [{
        "attempt_id": "crop-local-001",
        "stages": {"contact": {"keyframe_seconds": 4.25}},
    }]}))
    thumbnail = checkpoint_dir / "review-serve-3d"
    thumbnail.mkdir()
    (thumbnail / "contact.jpg").write_bytes(b"image bytes")
    return ReviewSession(tmp_path)


def annotation() -> dict:
    return {
        "source_fingerprint": "sha256:test", "view": "rear-quarter",
        "fully_reviewed": True,
        "labels": [{
            "id": "label-1", "source_id": "attempt:serve-001",
            "start_seconds": 1.8, "end_seconds": 5.1, "label": "serve",
            "checkpoints": {
                "contact": {"status": "corrected", "time_seconds": 4.25},
                "release": {"status": "uncertain", "time_seconds": None},
            },
        }],
    }


def test_annotations_persist_without_changing_generated_artifacts(tmp_path: Path) -> None:
    review = session(tmp_path)
    attempts_path = tmp_path / "metadata" / "serve" / "attempts.json"
    original = attempts_path.read_bytes()
    review.save_video("serve.MOV", annotation())
    assert attempts_path.read_bytes() == original
    assert review.state()["videos"][0]["annotation"] == annotation()
    assert review.state()["videos"][0]["stages"]["serve-001"]["contact"] == {
        "keyframe_seconds": 4.25, "thumbnail": True,
    }
    assert review.annotations_path.is_file()


def test_unprocessed_videos_are_listed_as_pending(tmp_path: Path) -> None:
    review = session(tmp_path)
    (tmp_path / "later.MOV").write_bytes(b"later")
    review = ReviewSession(tmp_path)
    state = review.state()
    assert [item["name"] for item in state["videos"]] == ["serve.MOV"]
    assert "shadows" not in state["videos"][0]
    assert state["pending"] == ["later.MOV"]


def test_rejects_changed_source_and_invalid_ranges(tmp_path: Path) -> None:
    review = session(tmp_path)
    wrong = annotation()
    wrong["source_fingerprint"] = "sha256:other"
    with pytest.raises(ReviewError, match="fingerprint"):
        review.save_video("serve.MOV", wrong)
    wrong = annotation()
    wrong["labels"][0]["end_seconds"] = 11
    with pytest.raises(ReviewError, match="range"):
        review.save_video("serve.MOV", wrong)
    assert not review.annotations_path.exists()


def test_video_byte_ranges_and_state_are_served_locally(tmp_path: Path) -> None:
    review = session(tmp_path)
    server = ReviewServer(("127.0.0.1", 0), review)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/state") as response:
            state = json.load(response)
            assert state["videos"][0]["name"] == "serve.MOV"
        request = Request(base + "/media/serve.MOV", headers={"Range": "bytes=2-7"})
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == b"mple v"
        with urlopen(base + "/thumb?video=serve.MOV&attempt=serve-001&stage=contact") as response:
            assert response.read() == b"image bytes"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
