#!/usr/bin/env python3
"""Align original and rendered slow-motion pose sequences using AAE timing.

This is an offline research tool. It reads existing 30 Hz pose caches, constrains
sequence alignment to the AAE piecewise-linear time map, fits simple rate ramps,
and writes correspondences plus a residual plot. It does not alter production
timestamps or detector output.

Example:
    uv run --no-sync python tools/align_slowmo_sequences.py \
      --original-dir corpus/originals/2026-09-08 \
      --export-dir corpus/2026-09-08 \
      --output-dir corpus/originals/2026-09-08/timing-alignment
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import plistlib
import re
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SAMPLE_EVERY_FRAMES = 1  # Use all frames from the pipeline's 30 Hz pose caches.
JOINTS = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
SLOPE_PENALTY = 0.00008


@dataclass(frozen=True)
class SlowRegion:
    start: float
    duration: float
    rate: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class PoseSequence:
    times: np.ndarray
    xy: np.ndarray
    visibility: np.ndarray


@dataclass
class RecordingAlignment:
    original: Path
    export: Path
    region: SlowRegion
    rows: list[dict[str, float | str]]


def _seconds(value: dict[str, object]) -> float:
    return float(value["value"]) / float(value["timescale"])


def read_slow_region(path: Path) -> SlowRegion:
    outer = plistlib.loads(path.read_bytes())
    payload = outer.get("adjustmentData")
    if not isinstance(payload, bytes):
        raise ValueError(f"No adjustmentData in {path}")
    document = json.loads(zlib.decompress(payload, -zlib.MAX_WBITS))
    adjustments = [a for a in document.get("adjustments", []) if a.get("identifier") == "SlowMotion"]
    if not adjustments:
        raise ValueError(f"No SlowMotion adjustment in {path}")
    settings = adjustments[-1]["settings"]
    regions = settings.get("regions", [])
    if len(regions) != 1:
        raise ValueError(f"Expected one SlowMotion region in {path}, found {len(regions)}")
    time_range = regions[0]["timeRange"]
    region = SlowRegion(_seconds(time_range["start"]), _seconds(time_range["duration"]), float(settings["rate"]))
    if not 0 < region.rate <= 1 or region.duration <= 0:
        raise ValueError(f"Invalid SlowMotion region in {path}: {region}")
    return region


def read_pose(path: Path) -> PoseSequence:
    times: list[float] = []
    xy_rows: list[np.ndarray] = []
    visibility_rows: list[np.ndarray] = []
    for line in path.open():
        record = json.loads(line)
        if record.get("type") != "frame":
            continue
        observation = record["observation"]
        time = float(observation["time_seconds"])
        people = observation.get("persons", [])
        times.append(time)
        coords = np.zeros((len(JOINTS), 2), dtype=np.float32)
        visibility = np.zeros(len(JOINTS), dtype=np.float32)
        if people:
            person = max(
                people,
                key=lambda p: p.get("score", 0.0)
                * max(0.0, p["box"]["x_max"] - p["box"]["x_min"])
                * max(0.0, p["box"]["y_max"] - p["box"]["y_min"]),
            )
            keypoints = person.get("keypoints", [])
            for output_index, joint_index in enumerate(JOINTS):
                point = keypoints[int(joint_index)] if int(joint_index) < len(keypoints) else None
                if point is None:
                    continue
                confidence = float(point.get("visibility", 0.0))
                if confidence >= 0.35:
                    coords[output_index] = (float(point["x"]), float(point["y"]))
                    visibility[output_index] = confidence
        xy_rows.append(coords)
        visibility_rows.append(visibility)
    return PoseSequence(np.asarray(times), np.asarray(xy_rows), np.asarray(visibility_rows))


def instantaneous_map(times: np.ndarray, region: SlowRegion) -> np.ndarray:
    elapsed = np.clip(times - region.start, 0.0, region.duration)
    return times + elapsed * (1.0 / region.rate - 1.0)


def pairwise_pose_cost(source_xy: np.ndarray, source_vis: np.ndarray,
                       export_xy: np.ndarray, export_vis: np.ndarray) -> np.ndarray:
    weights = np.minimum(export_vis, source_vis[None, :])
    distances = np.linalg.norm(export_xy - source_xy[None, :, :], axis=2)
    denominator = weights.sum(axis=1)
    costs = (distances * weights).sum(axis=1) / np.maximum(denominator, 1e-6)
    costs[denominator < 3.0] = 0.025
    return costs


def align_sequences(original: PoseSequence, export: PoseSequence, region: SlowRegion,
                    search_seconds: float) -> list[dict[str, float | str]]:
    sample_indices = np.arange(0, len(original.times), SAMPLE_EVERY_FRAMES, dtype=int)
    source_times = original.times[sample_indices]
    predicted_times = instantaneous_map(source_times, region)
    band_frames = int(round(search_seconds * 30.0))
    export_xy = export.xy.astype(np.float32, copy=False)
    export_vis = export.visibility.astype(np.float32, copy=False)

    candidates: list[np.ndarray] = []
    local_costs: list[np.ndarray] = []
    for sample_index, predicted_time in zip(sample_indices, predicted_times, strict=True):
        center = int(round(predicted_time * 30.0))
        low = max(0, center - band_frames)
        high = min(len(export.times), center + band_frames + 1)
        choices = np.arange(low, high, dtype=int)
        candidates.append(choices)
        local_costs.append(pairwise_pose_cost(
            original.xy[sample_index], original.visibility[sample_index],
            export_xy[choices], export_vis[choices],
        ))

    # Monotone dynamic programming: local pose similarity picks a frame while
    # the transition cost discourages implausible jumps from the AAE map.
    dp = local_costs[0].copy()
    backpointers: list[np.ndarray] = [np.full(len(candidates[0]), -1, dtype=np.int32)]
    for i in range(1, len(candidates)):
        previous, current = candidates[i - 1], candidates[i]
        expected_step = (predicted_times[i] - predicted_times[i - 1]) * 30.0
        max_step = max(6, int(math.ceil((source_times[i] - source_times[i - 1])
                                       * 30.0 / region.rate * 1.8)))
        next_dp = np.full(len(current), np.inf, dtype=np.float64)
        back = np.full(len(current), -1, dtype=np.int32)
        for j, frame_index in enumerate(current):
            lo = np.searchsorted(previous, frame_index - max_step, side="left")
            hi = np.searchsorted(previous, frame_index, side="left")
            if lo >= hi:
                continue
            prior_indices = previous[lo:hi]
            advances = frame_index - prior_indices
            transition_cost = SLOPE_PENALTY * np.square(advances - expected_step)
            costs = dp[lo:hi] + transition_cost
            best = int(np.argmin(costs))
            next_dp[j] = local_costs[i][j] + costs[best]
            back[j] = lo + best
        if not np.isfinite(next_dp).any():
            raise ValueError(f"No monotone alignment path near source time {source_times[i]:.3f}s")
        dp = next_dp
        backpointers.append(back)

    state = int(np.argmin(dp))
    matched = np.empty(len(candidates), dtype=int)
    for i in range(len(candidates) - 1, -1, -1):
        matched[i] = candidates[i][state]
        if i:
            state = int(backpointers[i][state])
            if state < 0:
                raise ValueError(f"Broken monotone alignment path at row {i}")

    rows = []
    for source_time, predicted_time, export_index, sample_index, local_cost in zip(
        source_times, predicted_times, matched, sample_indices, local_costs, strict=True
    ):
        source_vis = original.visibility[sample_index]
        exported_vis = export.visibility[export_index]
        weights = np.minimum(source_vis, exported_vis)
        visible = int(np.count_nonzero(weights >= 0.35))
        pose_cost = float(pairwise_pose_cost(
            original.xy[sample_index], source_vis,
            export.xy[export_index:export_index + 1], exported_vis[None, :],
        )[0])
        matched_time = float(export.times[export_index])
        rows.append({
            "source_seconds": float(source_time),
            "predicted_export_seconds": float(predicted_time),
            "matched_export_seconds": matched_time,
            "residual_seconds": matched_time - float(predicted_time),
            "pose_cost": pose_cost,
            "visible_joints": visible,
        })
    return rows


def finite_transition_correction(times: np.ndarray, region: SlowRegion,
                                 d_in: float, d_out: float, shape: str) -> np.ndarray:
    """Return smooth-transition map minus the instantaneous AAE map."""
    length = region.duration
    d_in = min(max(d_in, 0.0), length)
    d_out = min(max(d_out, 0.0), length - d_in)
    rate_excess = 1.0 / region.rate - 1.0
    x = np.clip(times - region.start, 0.0, length)

    def integral_ramp(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, 0.0, 1.0)
        if shape == "linear":
            return 0.5 * np.square(z)
        return np.power(z, 3) - 0.5 * np.power(z, 4)

    if d_in <= 1e-9 and d_out <= 1e-9:
        return np.zeros_like(times)
    slow_in_end = d_in
    slow_out_start = length - d_out
    integrated_excess_units = np.zeros_like(times)
    entering = (x > 0) & (x < slow_in_end) & (d_in > 1e-9)
    integrated_excess_units[entering] = d_in * integral_ramp(x[entering] / d_in)
    plateau = (x >= slow_in_end) & (x <= slow_out_start)
    integrated_excess_units[plateau] = 0.5 * d_in + x[plateau] - d_in
    leaving = (x > slow_out_start) & (x < length) & (d_out > 1e-9)
    if d_out > 1e-9:
        z = (x[leaving] - slow_out_start) / d_out
        integrated_excess_units[leaving] = (
            length - 0.5 * d_in - d_out + d_out * (z - integral_ramp(z))
        )
    after = x >= length
    integrated_excess_units[after] = length - 0.5 * (d_in + d_out)
    smooth_extra = rate_excess * integrated_excess_units
    instantaneous_extra = rate_excess * x
    return smooth_extra - instantaneous_extra


def fit_transition_model(alignments: list[RecordingAlignment]) -> dict[str, object]:
    fit_points = []
    point_counts = {}
    for alignment in alignments:
        region = alignment.region
        source = np.asarray([float(row["source_seconds"]) for row in alignment.rows])
        residual = np.asarray([float(row["residual_seconds"]) for row in alignment.rows])
        pose_cost = np.asarray([float(row["pose_cost"]) for row in alignment.rows])
        x = source - region.start
        # Give each edge equal influence and trim clear pose-detector mismatches.
        reliable = pose_cost <= 0.02
        entry_before = reliable & (x >= -1.0) & (x < 0.0)
        entry_after = reliable & (x >= 0.0) & (x <= 1.5)
        exit_before = reliable & (x >= region.duration - 1.5) & (x <= region.duration)
        exit_after = reliable & (x > region.duration) & (x <= region.duration + 1.0)
        entry_window = (x >= -1.0) & (x <= 1.5)
        exit_window = (x >= region.duration - 1.5) & (x <= region.duration + 1.0)
        entry_keep = entry_window & reliable
        exit_keep = exit_window & reliable
        keep = entry_keep | exit_keep
        fit_points.append((region, source[keep], residual[keep]))
        point_counts[alignment.original.stem] = {
            "entry_before": int(entry_before.sum()), "entry_after": int(entry_after.sum()),
            "exit_before": int(exit_before.sum()), "exit_after": int(exit_after.sum()),
        }

    reports = {}
    for shape in ("linear", "smoothstep"):
        def score(d_in: float, d_out: float) -> tuple[float, list[float]]:
            errors = []
            for region, source, residual in fit_points:
                predicted = finite_transition_correction(source, region, d_in, d_out, shape)
                errors.append(float(np.median(np.abs(residual - predicted))))
            return float(np.mean(errors)), errors

        coarse = np.arange(0.0, 0.901, 0.01)
        best = (float("inf"), 0.0, 0.0)
        for d_in in coarse:
            for d_out in coarse:
                candidate_score, _ = score(float(d_in), float(d_out))
                if candidate_score < best[0]:
                    best = (candidate_score, float(d_in), float(d_out))
        refine_in = np.arange(max(0.0, best[1] - 0.05), best[1] + 0.051, 0.005)
        refine_out = np.arange(max(0.0, best[2] - 0.05), best[2] + 0.051, 0.005)
        for d_in in refine_in:
            for d_out in refine_out:
                candidate_score, _ = score(float(d_in), float(d_out))
                if candidate_score < best[0]:
                    best = (candidate_score, float(d_in), float(d_out))
        per_clip = {}
        for alignment, (region, source, residual) in zip(alignments, fit_points, strict=True):
            x = source - region.start
            entry = (x >= -1.0) & (x <= 1.5)
            exit_edge = (x >= region.duration - 1.5) & (x <= region.duration + 1.0)
            entry_best = None
            counts = point_counts[alignment.original.stem]
            if counts["entry_before"] >= 3 and counts["entry_after"] >= 3:
                candidates_in = np.arange(0.0, 0.901, 0.005)
                errors_in = [np.median(np.abs(residual[entry] - finite_transition_correction(
                    source[entry], region, float(d_in), best[2], shape))) for d_in in candidates_in]
                entry_best = float(candidates_in[int(np.argmin(errors_in))])
            exit_best = None
            if counts["exit_before"] >= 3 and counts["exit_after"] >= 3:
                candidates_out = np.arange(0.0, 0.901, 0.005)
                errors_out = [np.median(np.abs(residual[exit_edge] - finite_transition_correction(
                    source[exit_edge], region, best[1], float(d_out), shape))) for d_out in candidates_out]
                exit_best = float(candidates_out[int(np.argmin(errors_out))])
            per_clip[alignment.original.stem] = {"entry_seconds": entry_best, "exit_seconds": exit_best}
        reports[shape] = {
            "entry_seconds": best[1],
            "exit_seconds": best[2],
            "mean_recording_median_absolute_error_seconds": best[0],
            "per_recording_best_transitions_seconds": per_clip,
            "usable_edge_correspondences": point_counts,
        }
    return reports


def pair_exports(original_dir: Path, export_dir: Path) -> list[tuple[Path, Path, Path]]:
    exports = [p for p in export_dir.iterdir() if p.is_file() and p.suffix.lower() in {".mov", ".mp4", ".m4v"}]
    pairings = []
    for aae in sorted(original_dir.glob("*.AAE")):
        original = original_dir / f"{aae.stem}.MOV"
        if not original.exists():
            original = original_dir / f"{aae.stem}.mov"
        if not original.exists():
            continue
        key = re.sub(r"\(\d+\)$", "", aae.stem).casefold()
        candidates = [p for p in exports if re.sub(r"\(\d+\)$", "", p.stem).casefold() == key]
        if not candidates:
            continue
        # Duplicate Google Photos names (e.g. IMG_1078 and IMG_1078(1)) may
        # include a short standalone clip. Pick the duration closest to AAE's
        # instantaneous prediction for the full source recording.
        region = read_slow_region(aae)
        original_duration = probe_duration(original)
        expected = original_duration + region.duration * (1.0 / region.rate - 1.0)
        export = min(candidates, key=lambda p: abs(probe_duration(p) - expected))
        pairings.append((original, export, aae))
    return pairings


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def save_residual_plot(alignments: list[RecordingAlignment], fits: dict[str, object], output: Path) -> None:
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for index, alignment in enumerate(alignments):
        region = alignment.region
        source = np.asarray([float(row["source_seconds"]) for row in alignment.rows])
        residual = np.asarray([float(row["residual_seconds"]) for row in alignment.rows])
        costs = np.asarray([float(row["pose_cost"]) for row in alignment.rows])
        color = colors[index % len(colors)]
        start_x = source - region.start
        end_x = source - region.end
        start_mask = (start_x >= -1.0) & (start_x <= 1.5) & (costs <= 0.02)
        end_mask = (end_x >= -1.5) & (end_x <= 1.0) & (costs <= 0.02)
        axes[0].plot(start_x[start_mask], residual[start_mask], ".-", ms=2.5, lw=0.8,
                     alpha=0.7, color=color, label=alignment.original.stem)
        axes[1].plot(end_x[end_mask], residual[end_mask], ".-", ms=2.5, lw=0.8,
                     alpha=0.7, color=color, label=alignment.original.stem)

    chosen_shape = min(fits, key=lambda name: fits[name]["mean_recording_median_absolute_error_seconds"])
    d_in = float(fits[chosen_shape]["entry_seconds"])
    d_out = float(fits[chosen_shape]["exit_seconds"])
    # Overlay an illustrative rate-0.25 model; every September 8 AAE has this rate.
    example = alignments[0].region
    for ax, offsets in ((axes[0], np.linspace(-1, 1.5, 400)), (axes[1], np.linspace(-1.5, 1, 400))):
        absolute = example.start + offsets if ax is axes[0] else example.end + offsets
        correction = finite_transition_correction(absolute, example, d_in, d_out, chosen_shape)
        ax.plot(offsets, correction, color="black", lw=2.0,
                label=f"shared {chosen_shape}, entry={d_in:.3f}s, exit={d_out:.3f}s")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axhline(0, color="gray", lw=0.7)
        ax.grid(alpha=0.2)
    axes[0].set_title("Slow-motion region entry")
    axes[0].set_xlabel("Original time relative to AAE start (s)")
    axes[1].set_title("Slow-motion region exit")
    axes[1].set_xlabel("Original time relative to AAE end (s)")
    axes[0].set_ylabel("Matched export time − instantaneous AAE prediction (s)")
    axes[1].legend(fontsize=7, loc="best")
    axes[0].legend(fontsize=7, loc="best")
    figure.suptitle("September 8 original-to-export alignment around AAE boundaries")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-seconds", type=float, default=2.0)
    args = parser.parse_args()

    alignments: list[RecordingAlignment] = []
    for original_path, export_path, aae_path in pair_exports(args.original_dir, args.export_dir):
        stem = original_path.stem
        original_cache = args.original_dir / "metadata" / stem / "cache" / "pose-v1.jsonl"
        export_cache = args.export_dir / "metadata" / export_path.stem / "cache" / "pose-v1.jsonl"
        if not original_cache.exists() or not export_cache.exists():
            print(f"skip {stem}: missing pose cache", file=sys.stderr)
            continue
        region = read_slow_region(aae_path)
        original_pose = read_pose(original_cache)
        export_pose = read_pose(export_cache)
        rows = align_sequences(original_pose, export_pose, region, args.search_seconds)
        alignment = RecordingAlignment(original_path, export_path, region, rows)
        alignments.append(alignment)
        residual = np.asarray([float(row["residual_seconds"]) for row in rows])
        reliable = np.asarray([float(row["pose_cost"]) <= 0.02 for row in rows])
        times = np.asarray([float(row["source_seconds"]) for row in rows])
        masks = [times < region.start, (times >= region.start) & (times <= region.end), times > region.end]
        reliable_medians = [float(np.median(residual[reliable & mask]))
                            if np.any(reliable & mask) else float("nan") for mask in masks]
        print(f"{stem}: {len(rows)} correspondences, {reliable.sum()} low-cost; "
              f"duration {probe_duration(original_path):.3f}s -> {probe_duration(export_path):.3f}s; "
              f"low-cost median residual before/inside/after="
              f"{reliable_medians[0]:+.3f}/{reliable_medians[1]:+.3f}/{reliable_medians[2]:+.3f}s")

    if len(alignments) < 2:
        raise ValueError(f"Need at least two paired recordings with pose caches; found {len(alignments)}")

    fits = fit_transition_model(alignments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "correspondences.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["recording", "source_seconds", "predicted_export_seconds",
                                                  "matched_export_seconds", "residual_seconds", "pose_cost",
                                                  "visible_joints"])
        writer.writeheader()
        for alignment in alignments:
            for row in alignment.rows:
                writer.writerow({"recording": alignment.original.stem, **row})
    plot_path = args.output_dir / "residuals.png"
    save_residual_plot(alignments, fits, plot_path)
    result = {
        "records": len(alignments),
        "alignment_method": "30 Hz MediaPipe pose correspondences with monotone dynamic programming and AAE prediction band",
        "fits": fits,
        "files": {"correspondences": str(csv_path), "plot": str(plot_path)},
    }
    json_path = args.output_dir / "fit.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print("\nShared transition fits (equal weight per recording; edge correspondences only):")
    for shape, fit in fits.items():
        local = list(fit["per_recording_best_transitions_seconds"].values())
        entry_values = [x["entry_seconds"] for x in local if x["entry_seconds"] is not None]
        exit_values = [x["exit_seconds"] for x in local if x["exit_seconds"] is not None]
        print(f"  {shape}: entry={fit['entry_seconds']:.3f}s, exit={fit['exit_seconds']:.3f}s, "
              f"mean recording median abs error={fit['mean_recording_median_absolute_error_seconds']:.4f}s; "
              f"per-recording entry range={min(entry_values):.3f}..{max(entry_values):.3f}s, "
              f"exit range={min(exit_values):.3f}..{max(exit_values):.3f}s")
    print(f"\nWrote {csv_path}, {plot_path}, and {json_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, zlib.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
