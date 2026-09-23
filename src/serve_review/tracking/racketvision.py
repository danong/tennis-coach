"""In-process adapter for RacketVision BallTrack and RacketPose.

Heavy optional dependencies are imported only when a tracker is opened. The
adapter consumes the project's upright, timestamped RGB frames and returns the
existing normalized scene geometry types. RacketVision never owns time.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from serve_review.media.frames import SampledFrame
from serve_review.scene import Point2D, Racket2D

__all__ = [
    "RACKET_KEYPOINT_NAMES",
    "RacketVisionConfig",
    "RacketVisionError",
    "RacketVisionFrameObservation",
    "RacketVisionTracker",
    "fingerprint_racketvision_config",
]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SOURCE_ROOT = _REPOSITORY_ROOT / "vendor" / "racketvision"
_DEFAULT_MODEL_ROOT = _REPOSITORY_ROOT / "models" / "racketvision"
RACKET_KEYPOINT_NAMES = ("Top", "Bottom", "Handle", "Left", "Right")


class RacketVisionError(RuntimeError):
    """RacketVision cannot be configured or run safely."""


def resolve_racketvision_device(device: str = "auto") -> str:
    """Resolve ``auto`` to CUDA when PyTorch reports it is available.

    PyTorch is optional until RacketVision is actually used, so importing it
    here is deliberately lazy. A broken or CPU-only PyTorch installation
    safely selects CPU in automatic mode.
    """
    normalized = device.strip().lower() if isinstance(device, str) else ""
    if normalized != "auto":
        if not normalized:
            raise RacketVisionError("device must be a non-blank string.")
        if normalized == "cuda" or normalized.startswith("cuda:"):
            if not racketvision_cuda_available():
                raise RacketVisionError(
                    "CUDA was requested but PyTorch cannot access a CUDA device; "
                    "install a CUDA-enabled PyTorch build and verify the NVIDIA "
                    "driver, or select --device cpu."
                )
        return device.strip()
    return "cuda" if racketvision_cuda_available() else "cpu"


def racketvision_cuda_available() -> bool:
    """Return whether the installed PyTorch can access CUDA."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        # Preserve the CPU path when PyTorch cannot inspect CUDA on this host.
        return False


@dataclass(frozen=True, slots=True)
class RacketVisionConfig:
    """Paths and thresholds for the two RacketVision pipelines."""

    source_root: Path = _DEFAULT_SOURCE_ROOT
    ball_checkpoint: Path = _DEFAULT_MODEL_ROOT / "balltrack.pth"
    racket_detector_checkpoint: Path = _DEFAULT_MODEL_ROOT / "racket-detector.pth"
    racket_keypoints_checkpoint: Path = _DEFAULT_MODEL_ROOT / "racket-keypoints.pth"
    ball_threshold: float = 0.10
    racket_bbox_threshold: float = 0.30
    ball_batch_size: int = 20
    device: str = "auto"

    def __post_init__(self) -> None:
        for field in (
            "source_root",
            "ball_checkpoint",
            "racket_detector_checkpoint",
            "racket_keypoints_checkpoint",
        ):
            value = Path(getattr(self, field)).expanduser()
            object.__setattr__(self, field, value)
        for field in ("ball_threshold", "racket_bbox_threshold"):
            raw = getattr(self, field)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise RacketVisionError(f"{field} must be a number in [0, 1].")
            value = float(raw)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RacketVisionError(f"{field} must be a finite number in [0, 1].")
            object.__setattr__(self, field, value)
        if (
            isinstance(self.ball_batch_size, bool)
            or not isinstance(self.ball_batch_size, int)
            or self.ball_batch_size < 1
        ):
            raise RacketVisionError("ball_batch_size must be an integer >= 1.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise RacketVisionError("device must be a non-blank string.")
        object.__setattr__(self, "device", self.device.strip())

    @property
    def ball_source(self) -> Path:
        return self.source_root / "source" / "BallTrack"

    @property
    def racket_source(self) -> Path:
        return self.source_root / "source" / "RacketPose"

    @property
    def ball_config(self) -> Path:
        return self.ball_source / "configs" / "tracknetv3_base.py"

    @property
    def racket_detector_config(self) -> Path:
        return self.racket_source / "configs" / "detection" / "rtmdet_m_racket_infer.py"

    @property
    def racket_keypoints_config(self) -> Path:
        return self.racket_source / "configs" / "pose" / "rtmpose_m_racket_infer.py"

    def resolved_device(self) -> RacketVisionConfig:
        """Return a copy with automatic device selection frozen for caching."""
        resolved = resolve_racketvision_device(self.device)
        return self if resolved == self.device else replace(self, device=resolved)

    def require_files(self) -> None:
        required = (
            self.ball_config,
            self.racket_detector_config,
            self.racket_keypoints_config,
            self.ball_checkpoint,
            self.racket_detector_checkpoint,
            self.racket_keypoints_checkpoint,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RacketVisionError(
                "missing RacketVision prerequisite(s): " + ", ".join(missing)
            )


@lru_cache(maxsize=32)
def _fingerprint_file(path_text: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_racketvision_config(config: RacketVisionConfig) -> str:
    """Identify all files and settings that can affect raw observations."""
    if not isinstance(config, RacketVisionConfig):
        raise TypeError("config must be a RacketVisionConfig value.")
    config.require_files()
    files = {
        name: getattr(config, name)
        for name in (
            "ball_config",
            "racket_detector_config",
            "racket_keypoints_config",
            "ball_checkpoint",
            "racket_detector_checkpoint",
            "racket_keypoints_checkpoint",
        )
    }
    payload = {
        "ball_batch_size": config.ball_batch_size,
        "ball_threshold": config.ball_threshold,
        "device": config.device,
        "files": {
            name: _fingerprint_file(
                str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns
            )
            for name, path in files.items()
        },
        "racket_bbox_threshold": config.racket_bbox_threshold,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True, slots=True)
class RacketVisionFrameObservation:
    """Raw ball/racket observations for one canonical source PTS."""

    time_seconds: float
    ball_2d: Point2D | None
    racket_2d: Racket2D | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.time_seconds, bool)
            or not isinstance(self.time_seconds, (int, float))
            or not math.isfinite(float(self.time_seconds))
            or float(self.time_seconds) < 0.0
        ):
            raise RacketVisionError("time_seconds must be finite and >= 0.")
        object.__setattr__(self, "time_seconds", float(self.time_seconds))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RacketVisionError(f"could not load RacketVision module at {path}.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RacketVisionError(
            f"could not import RacketVision module at {path}: {exc}"
        ) from exc
    return module


def _letterbox(
    image: np.ndarray, width: int, height: int, cv2: Any
) -> tuple[np.ndarray, float, int, int]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, scale, left, top


def _normalized_point(
    x: object,
    y: object,
    confidence: object,
    width: int,
    height: int,
) -> Point2D | None:
    try:
        px, py, score = float(x), float(y), float(confidence)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (px, py, score)):
        return None
    if not 0.0 <= score <= 1.0 or not 0.0 <= px < width or not 0.0 <= py < height:
        return None
    return Point2D(px / width, py / height, score)


def _map_ball(
    result: dict[str, Any],
    *,
    width: int,
    height: int,
    scale: float,
    left: int,
    top: int,
) -> Point2D | None:
    if not result.get("Visibility"):
        return None
    try:
        x = (float(result["X"]) - left) / scale
        y = (float(result["Y"]) - top) / scale
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return _normalized_point(x, y, result.get("Confidence"), width, height)


def _map_racket(result: dict[str, Any], *, width: int, height: int) -> Racket2D | None:
    try:
        raw_box = result["bbox"][0]
        score = float(result["bbox_score"])
        x1, y1, x2, y2 = (float(value) for value in raw_box)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if not (math.isfinite(score) and 0.0 <= score <= 1.0):
        return None
    if not (x1 < x2 and y1 < y2):
        return None
    raw_points = result.get("keypoints", ())
    raw_scores = result.get("keypoint_scores", ())
    keypoints: dict[str, Point2D | None] = {}
    for index, name in enumerate(RACKET_KEYPOINT_NAMES):
        try:
            x, y = raw_points[index]
            confidence = raw_scores[index]
        except (IndexError, TypeError, ValueError):
            keypoints[name] = None
        else:
            keypoints[name] = _normalized_point(x, y, confidence, width, height)
    return Racket2D(
        bbox=(x1 / width, y1 / height, x2 / width, y2 / height),
        bbox_confidence=score,
        keypoints=keypoints,
    )


class RacketVisionTracker:
    """Load both RacketVision pipelines and infer timestamped RGB frames."""

    def __init__(self, config: RacketVisionConfig | None = None) -> None:
        selected_config = config or RacketVisionConfig()
        if not isinstance(selected_config, RacketVisionConfig):
            raise TypeError("config must be RacketVisionConfig or None.")
        self.config = selected_config.resolved_device()
        self.config.require_files()
        try:
            import cv2
            import torch
        except ImportError as exc:
            raise RacketVisionError(
                "RacketVision dependencies are unavailable; run `mise run setup`."
            ) from exc
        self._cv2 = cv2
        ball_source = str(self.config.ball_source)
        if ball_source not in sys.path:
            sys.path.insert(0, ball_source)
        ball_module = _load_module(
            "serve_review_racketvision_ball", self.config.ball_source / "inference.py"
        )
        original_torch_load = torch.load
        try:
            racket_module = _load_module(
                "serve_review_racketvision_racket",
                self.config.racket_source / "tools" / "inference.py",
            )
        finally:
            torch.load = original_torch_load
        try:
            self._ball = ball_module.BallInferencer(
                str(self.config.ball_config),
                str(self.config.ball_checkpoint),
                device=self.config.device,
                thre=self.config.ball_threshold,
                batchsize=self.config.ball_batch_size,
            )
            with racket_module.DefaultScope.overwrite_default_scope("mmdet"):
                self._racket_detector = racket_module.init_detector(
                    str(self.config.racket_detector_config),
                    str(self.config.racket_detector_checkpoint),
                    device=self.config.device,
                )
            self._racket_pose = racket_module.init_pose_model(
                str(self.config.racket_keypoints_config),
                str(self.config.racket_keypoints_checkpoint),
                device=self.config.device,
            )
        except Exception as exc:
            raise RacketVisionError(
                f"could not initialize RacketVision models: {exc}"
            ) from exc
        self._racket_module = racket_module

    def track_frames(
        self, frames: Iterable[SampledFrame]
    ) -> tuple[RacketVisionFrameObservation, ...]:
        """Infer raw observations while preserving caller-supplied PTS."""
        times: list[float] = []
        racket_results: list[dict[str, Any] | None] = []
        transforms: list[tuple[int, int, float, int, int]] = []
        with tempfile.TemporaryDirectory(prefix="racketvision-frames-") as temporary:
            directory = Path(temporary)
            paths: list[str] = []
            model_samples: list[np.ndarray] = []
            previous_time: float | None = None
            for index, frame in enumerate(frames):
                if not isinstance(frame, SampledFrame):
                    raise RacketVisionError("frames must contain SampledFrame values.")
                moment = float(frame.time_seconds)
                if previous_time is not None and moment <= previous_time:
                    raise RacketVisionError("frame PTS must be strictly increasing.")
                image = np.asarray(frame.image)
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise RacketVisionError("frame images must be uint8 RGB arrays.")
                height, width = image.shape[:2]
                bgr = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2BGR)
                model_frame, scale, left, top = _letterbox(bgr, 512, 288, self._cv2)
                path = directory / f"{index:06d}.jpg"
                if not self._cv2.imwrite(str(path), model_frame):
                    raise RacketVisionError(f"could not stage model frame {index}.")
                paths.append(str(path))
                model_samples.append(model_frame)
                try:
                    rackets = self._racket_module.detect_and_estimate_pose(
                        self._racket_detector,
                        self._racket_pose,
                        bgr,
                        cat_id=2,
                        bbox_thr=self.config.racket_bbox_threshold,
                    )
                except Exception as exc:
                    raise RacketVisionError(
                        f"racket inference failed at t={moment}: {exc}"
                    ) from exc
                racket_results.append(rackets[0] if rackets else None)
                transforms.append((width, height, scale, left, top))
                times.append(moment)
                previous_time = moment
            if not paths:
                return ()
            sample_indices = np.linspace(
                0, len(model_samples) - 1, min(64, len(model_samples)), dtype=int
            )
            median = np.median(
                np.stack([model_samples[index] for index in sample_indices]), axis=0
            ).astype(np.uint8)
            median_path = directory / "median.npz"
            np.savez(median_path, median=median)
            try:
                ball_results = self._ball(paths, str(median_path))
            except Exception as exc:
                raise RacketVisionError(f"ball inference failed: {exc}") from exc
        if len(ball_results) != len(times):
            raise RacketVisionError(
                f"BallTrack returned {len(ball_results)} rows for {len(times)} frames."
            )
        observations = []
        for moment, ball, racket, transform in zip(
            times, ball_results, racket_results, transforms
        ):
            width, height, scale, left, top = transform
            observations.append(
                RacketVisionFrameObservation(
                    time_seconds=moment,
                    ball_2d=_map_ball(
                        ball,
                        width=width,
                        height=height,
                        scale=scale,
                        left=left,
                        top=top,
                    ),
                    racket_2d=(
                        _map_racket(racket, width=width, height=height)
                        if racket is not None
                        else None
                    ),
                )
            )
        return tuple(observations)
