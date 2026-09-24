#!/usr/bin/env python3
"""Compare iPhone source frames with a rendered slow-motion export.

The AAE SlowMotion adjustment supplies an initial piecewise-linear time map.
For selected source times, this script searches nearby export frames for a
similar motion foreground and reports the residual from that map. It is a
diagnostic, not a ground-truth decoder for Apple's editing format.

Example:
    uv run --no-sync python tools/compare_slowmo_frames.py \
      --original corpus/originals/2026-09-08/IMG_1080.MOV \
      --export corpus/2026-09-08/IMG_1080.MOV \
      --aae corpus/originals/2026-09-08/IMG_1080.AAE
"""

from __future__ import annotations

import argparse
import json
import plistlib
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FPS = 30
FRAME_WIDTH = 96
FRAME_HEIGHT = 96


@dataclass(frozen=True)
class SlowRegion:
    start: float
    duration: float
    rate: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def _seconds(value: dict[str, object]) -> float:
    return float(value["value"]) / float(value["timescale"])


def read_slow_regions(path: Path) -> list[SlowRegion]:
    outer = plistlib.loads(path.read_bytes())
    payload = outer.get("adjustmentData")
    if not isinstance(payload, bytes):
        raise ValueError(f"{path} has no adjustmentData payload")

    # iOS 26.6 AAE payloads in this corpus use raw DEFLATE compressed JSON.
    try:
        document = json.loads(zlib.decompress(payload, -zlib.MAX_WBITS))
    except (zlib.error, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode adjustmentData in {path}: {exc}") from exc

    adjustments = document.get("adjustments", [])
    slow = [item for item in adjustments if item.get("identifier") == "SlowMotion"]
    if not slow:
        raise ValueError(f"No SlowMotion adjustment in {path}")

    settings = slow[-1]["settings"]
    rate = float(settings["rate"])
    if not 0 < rate <= 1:
        raise ValueError(f"Unexpected playback rate {rate} in {path}")
    regions = []
    for item in settings.get("regions", []):
        time_range = item["timeRange"]
        region = SlowRegion(
            _seconds(time_range["start"]),
            _seconds(time_range["duration"]),
            rate,
        )
        if region.duration > 0:
            regions.append(region)
    if not regions:
        raise ValueError(f"No non-empty SlowMotion regions in {path}")
    return sorted(regions, key=lambda item: item.start)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def decode_frames(path: Path) -> np.ndarray:
    vf = (
        f"fps={FPS},"
        f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease:flags=area,"
        f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "format=gray"
    )
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        check=True,
        capture_output=True,
    )
    frame_size = FRAME_WIDTH * FRAME_HEIGHT
    if len(result.stdout) % frame_size:
        raise ValueError(f"ffmpeg returned an incomplete frame for {path}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, FRAME_HEIGHT, FRAME_WIDTH)


def foreground_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Remove the static court/background so the player's changing pose carries
    # more weight than the nearly identical fixed camera view.
    background = np.median(frames, axis=0).astype(np.int16)
    residual = np.abs(frames.astype(np.int16) - background).astype(np.float32)
    residual[residual < 12.0] = 0.0
    flat = residual.reshape(len(frames), -1)
    norms = np.linalg.norm(flat, axis=1)
    normalized = flat / np.maximum(norms[:, None], 1e-6)
    return normalized, flat.mean(axis=1)


def map_source_time(time: float, regions: list[SlowRegion]) -> float:
    """Map original media PTS to rendered playback time, assuming hard edges."""
    mapped = time
    for region in regions:
        if time <= region.start:
            break
        mapped += (min(time, region.end) - region.start) * ((1.0 / region.rate) - 1.0)
        if time <= region.end:
            break
    return mapped


def predicted_export_duration(original_duration: float, regions: list[SlowRegion]) -> float:
    return original_duration + sum(region.duration * (1.0 / region.rate - 1.0) for region in regions)


def build_anchor_times(duration: float, regions: list[SlowRegion]) -> list[float]:
    anchors = set(np.arange(0.5, max(0.5, duration - 0.25), 1.0).tolist())
    for region in regions:
        # Dense samples around each edge reveal whether the actual time warp
        # eases into/out of slow motion instead of changing speed instantly.
        for center in (region.start, region.end):
            anchors.update(np.arange(center - 2.0, center + 2.001, 0.2).tolist())
        anchors.update(np.arange(region.start + 0.5, region.end, 1.0).tolist())
    return sorted(t for t in anchors if 0.0 <= t < duration)


def region_name(time: float, regions: list[SlowRegion]) -> str:
    for index, region in enumerate(regions, start=1):
        if region.start <= time <= region.end:
            return f"slow-{index}"
        if time < region.start:
            return f"before-{index}"
    return "after"


def compare(
    original: np.ndarray,
    export: np.ndarray,
    regions: list[SlowRegion],
    search_seconds: float,
) -> list[tuple[float, float, float, float, float, float, str]]:
    original_features, original_energy = foreground_features(original)
    export_features, export_energy = foreground_features(export)
    radius = int(round(search_seconds * FPS))
    rows = []
    for source_time in build_anchor_times(len(original) / FPS, regions):
        source_index = min(round(source_time * FPS), len(original) - 1)
        if original_energy[source_index] < 0.5:
            continue
        predicted_time = map_source_time(source_time, regions)
        predicted_index = round(predicted_time * FPS)
        low = max(0, predicted_index - radius)
        high = min(len(export), predicted_index + radius + 1)
        if low >= high:
            continue
        scores = export_features[low:high] @ original_features[source_index]
        # Prefer the AAE-predicted instant when scene content is ambiguous.
        distances = np.abs(np.arange(low, high) - predicted_index) / max(radius, 1)
        scores = scores - 0.015 * distances
        chosen = low + int(np.argmax(scores))
        rows.append((
            source_time,
            predicted_time,
            chosen / FPS,
            chosen / FPS - predicted_time,
            float(scores[chosen - low]),
            float(original_energy[source_index]),
            region_name(source_time, regions),
        ))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--aae", type=Path, required=True)
    parser.add_argument("--search-seconds", type=float, default=2.0,
                        help="Search radius around each AAE-predicted export time (default: 2 s)")
    args = parser.parse_args()

    regions = read_slow_regions(args.aae)
    original_duration = probe_duration(args.original)
    export_duration = probe_duration(args.export)
    expected_duration = predicted_export_duration(original_duration, regions)
    print(f"Original duration: {original_duration:.3f}s")
    print(f"AAE-predicted duration (instantaneous boundaries): {expected_duration:.3f}s")
    print(f"Export duration: {export_duration:.3f}s")
    print(f"Export minus prediction: {export_duration - expected_duration:+.3f}s")
    print("Slow-motion regions:")
    for region in regions:
        print(f"  {region.start:.3f}..{region.end:.3f}s at rate {region.rate:g}")
    print(f"Decoding frames at {FPS} fps and {FRAME_WIDTH}x{FRAME_HEIGHT} for content matching...", flush=True)
    original_frames = decode_frames(args.original)
    export_frames = decode_frames(args.export)
    rows = compare(original_frames, export_frames, regions, args.search_seconds)

    print("\nsource_s,predicted_export_s,matched_export_s,residual_s,similarity,motion_energy,region")
    for row in rows:
        print(f"{row[0]:.3f},{row[1]:.3f},{row[2]:.3f},{row[3]:+.3f},{row[4]:.3f},{row[5]:.3f},{row[6]}")

    if rows:
        residuals = np.array([row[3] for row in rows])
        scores = np.array([row[4] for row in rows if row[5] >= 1.5])
        print(f"\nMatched anchors: {len(rows)}; median residual={np.median(residuals):+.3f}s; "
              f"high-motion median similarity={np.median(scores):.3f}" if len(scores) else
              f"\nMatched anchors: {len(rows)}; median residual={np.median(residuals):+.3f}s; no high-motion anchors")
        for label in dict.fromkeys(row[6] for row in rows):
            group = [row for row in rows if row[6] == label]
            group_residuals = np.array([row[3] for row in group])
            print(f"  {label}: n={len(group)}, median residual={np.median(group_residuals):+.3f}s, "
                  f"range=[{group_residuals.min():+.3f},{group_residuals.max():+.3f}]s")
        print("Low-motion anchors may match the fixed background instead of the player; inspect motion_energy and similarity.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
