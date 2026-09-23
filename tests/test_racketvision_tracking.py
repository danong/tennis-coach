from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from serve_review.scene import Point2D
from serve_review.tracking.racketvision import (
    RacketVisionConfig,
    RacketVisionError,
    RacketVisionFrameObservation,
    _map_ball,
    _map_racket,
    resolve_racketvision_device,
)


def test_config_rejects_bad_threshold_and_reports_missing_files(tmp_path: Path) -> None:
    with pytest.raises(RacketVisionError, match="ball_threshold"):
        RacketVisionConfig(ball_threshold=1.1)

    config = RacketVisionConfig(
        source_root=tmp_path / "source",
        ball_checkpoint=tmp_path / "ball.pth",
        racket_detector_checkpoint=tmp_path / "detector.pth",
        racket_keypoints_checkpoint=tmp_path / "keypoints.pth",
    )
    with pytest.raises(RacketVisionError, match="missing RacketVision"):
        config.require_files()


@pytest.mark.parametrize(("available", "expected"), [(True, "cuda"), (False, "cpu")])
def test_auto_device_uses_cuda_only_when_pytorch_reports_it_available(
    monkeypatch, available: bool, expected: str
) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available)),
    )
    assert resolve_racketvision_device("auto") == expected


def test_explicit_device_override_does_not_probe_cuda(monkeypatch) -> None:
    def unexpected_import(name: str, *args, **kwargs):
        if name == "torch":
            raise AssertionError("explicit device should not probe PyTorch")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", unexpected_import)
    assert resolve_racketvision_device("cpu") == "cpu"


def test_explicit_cuda_reports_an_actionable_error_when_unavailable(monkeypatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(RacketVisionError, match="CUDA-enabled PyTorch"):
        resolve_racketvision_device("cuda")


def test_ball_mapping_reverses_letterbox_and_preserves_confidence() -> None:
    point = _map_ball(
        {"Visibility": 1, "X": 256, "Y": 144, "Confidence": 0.15},
        width=1080,
        height=1920,
        scale=0.15,
        left=175,
        top=0,
    )
    assert point is not None
    assert point.x == pytest.approx(0.5)
    assert point.y == pytest.approx(0.5)
    assert point.confidence == pytest.approx(0.15)

    assert (
        _map_ball(
            {"Visibility": 0, "X": 0, "Y": 0, "Confidence": 0.0},
            width=1080,
            height=1920,
            scale=0.15,
            left=175,
            top=0,
        )
        is None
    )


def test_racket_mapping_normalizes_geometry_and_keeps_missing_keypoints() -> None:
    racket = _map_racket(
        {
            "bbox": [[10, 20, 90, 180]],
            "bbox_score": 0.8,
            "keypoints": [[50, 20], [50, 180], [50, 150], [-5, 80], [80, 80]],
            "keypoint_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        },
        width=100,
        height=200,
    )
    assert racket is not None
    assert racket.bbox == pytest.approx((0.1, 0.1, 0.9, 0.9))
    assert racket.bbox_confidence == pytest.approx(0.8)
    assert racket.keypoints["Top"] == Point2D(0.5, 0.1, 0.9)
    assert racket.keypoints["Left"] is None
    assert racket.keypoints["Right"] == Point2D(0.8, 0.4, 0.5)


def test_racketvision_observation_uses_only_canonical_seconds() -> None:
    observation = RacketVisionFrameObservation(
        time_seconds=1.25,
        ball_2d=Point2D(0.5, 0.5, 0.2),
        racket_2d=None,
    )
    assert observation.time_seconds == 1.25
    assert not hasattr(observation, "timestamp_ms")
