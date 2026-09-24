#!/usr/bin/env python3
"""Compare serve detection on Google exports and time-warped originals.

Uses the shared September 8 transition fit to map hand-reviewed export ranges
back to original source time, then applies the existing recording-scoped range
evaluator to predictions generated from each version. Writes a JSON report and
a CSV of transferred labels. This is a retrospective same-session comparison,
not a held-out performance estimate.

Example:
    UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python tools/evaluate_warped_detection.py \
      --original-dir corpus/originals/2026-09-08 \
      --export-dir corpus/2026-09-08 \
      --alignment-dir corpus/originals/2026-09-08/timing-alignment \
      --annotations corpus/2026-09-08/annotations/review-annotations-v1.json \
      --output-dir corpus/originals/2026-09-08/timing-alignment/evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_slowmo_sequences import (  # noqa: E402
    SlowRegion,
    finite_transition_correction,
    instantaneous_map,
    pair_exports,
    read_slow_region,
)
from serve_review.domain import MediaRange  # noqa: E402
from serve_review.evaluation import (  # noqa: E402
    IOU_THRESHOLD,
    LABEL_AMBIGUOUS,
    LABEL_SERVE,
    StratumInput,
    evaluate_report,
    iou,
    match_contact_events,
)


def forward_map(value: float, region: SlowRegion, entry: float, exit_: float, shape: str) -> float:
    time = np.asarray([value], dtype=float)
    return float(instantaneous_map(time, region)[0] + finite_transition_correction(
        time, region, entry, exit_, shape
    )[0])


def invert_map(value: float, region: SlowRegion, entry: float, exit_: float, shape: str,
               duration: float) -> float:
    """Invert monotone original→export time mapping by bisection."""
    low, high = 0.0, duration
    mapped_low = forward_map(low, region, entry, exit_, shape)
    mapped_high = forward_map(high, region, entry, exit_, shape)
    if value < mapped_low - 0.04 or value > mapped_high + 0.04:
        raise ValueError(f"export time {value:.3f}s falls outside mapped source duration "
                         f"[{mapped_low:.3f}, {mapped_high:.3f}]s")
    value = min(max(value, mapped_low), mapped_high)
    for _ in range(50):
        mid = (low + high) / 2
        if forward_map(mid, region, entry, exit_, shape) < value:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def map_range(item: dict, region: SlowRegion, entry: float, exit_: float, shape: str,
              duration: float) -> MediaRange:
    start = invert_map(float(item["start_seconds"]), region, entry, exit_, shape, duration)
    end = invert_map(float(item["end_seconds"]), region, entry, exit_, shape, duration)
    return MediaRange(start, end)


def load_detection_events(metadata_dir: Path) -> list[dict]:
    """Load coarse candidate ranges and their generated contact checkpoints."""
    document = json.loads((metadata_dir / "attempts.json").read_text())
    events = []
    for item in document["attempts"]:
        attempt_id = item["attempt_id"]
        detected = item["detected_range"]
        checkpoint_path = metadata_dir / "attempts" / attempt_id / "checkpoints.json"
        contact = None
        if checkpoint_path.is_file():
            checkpoint_doc = json.loads(checkpoint_path.read_text())
            rows = checkpoint_doc.get("attempts", [])
            if len(rows) == 1:
                proposal = rows[0].get("stages", {}).get("contact", {})
                value = proposal.get("keyframe_seconds")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    contact = float(value)
        events.append({
            "id": attempt_id,
            "contact_seconds": contact,
            "range": MediaRange(float(detected["start_seconds"]), float(detected["end_seconds"])),
        })
    return events


def _event_metrics(truth: list[dict], predictions: list[dict], ambiguous: tuple[MediaRange, ...],
                   tolerance: float) -> tuple[dict, tuple, set[int]]:
    matches = match_contact_events(
        [item["contact_seconds"] for item in truth],
        [item["contact_seconds"] for item in predictions],
        tolerance_seconds=tolerance,
    )
    matched_predictions = {item.pred_index for item in matches}
    excused = set()
    for index, prediction in enumerate(predictions):
        if index in matched_predictions:
            continue
        contact = prediction["contact_seconds"]
        if contact is not None and any(
            region.start_seconds <= contact < region.end_seconds for region in ambiguous
        ):
            excused.add(index)
        elif any(iou(prediction["range"], region) >= IOU_THRESHOLD for region in ambiguous):
            excused.add(index)
    tp = len(matches)
    fp = len(predictions) - tp - len(excused)
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(truth) if truth else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else
          (0.0 if truth or predictions else None))
    errors = [match.contact_error_seconds * 1000 for match in matches]
    result = {
        "contact_tolerance_ms": tolerance * 1000,
        "truth_events": len(truth),
        "predicted_events": len(predictions),
        "matched_events": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "excused_ambiguous": len(excused),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_abs_contact_error_ms": sum(abs(value) for value in errors) / len(errors) if errors else None,
        "mean_signed_contact_error_ms": sum(errors) / len(errors) if errors else None,
        "matches": [
            {"truth_index": match.truth_index, "prediction_index": match.pred_index,
             "contact_error_ms": match.contact_error_seconds * 1000}
            for match in matches
        ],
        "unavailable_truth_contacts": sum(item["contact_seconds"] is None for item in truth),
        "unavailable_predicted_contacts": sum(item["contact_seconds"] is None for item in predictions),
    }
    return result, matches, excused


def _match_candidate_windows(truth: list[dict], predictions: list[dict]) -> tuple[tuple[int, int], ...]:
    """Match a coarse candidate when its window contains the labeled contact.

    This measures whether a candidate was proposed around each event without
    requiring candidate boundaries to resemble the human analysis interval.
    Contact checkpoint accuracy is scored separately by ``_event_metrics``.
    """
    pairs = []
    for truth_index, event in enumerate(truth):
        contact = event["contact_seconds"]
        if contact is None:
            continue
        for pred_index, candidate in enumerate(predictions):
            window = candidate["range"]
            if window.start_seconds <= contact < window.end_seconds:
                midpoint_error = abs((window.start_seconds + window.end_seconds) / 2 - contact)
                pairs.append((midpoint_error, truth_index, pred_index))
    pairs.sort()
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches = []
    for _, truth_index, pred_index in pairs:
        if truth_index in used_truth or pred_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(pred_index)
        matches.append((truth_index, pred_index))
    return tuple(sorted(matches))


def _candidate_window_metrics(truth: list[dict], predictions: list[dict],
                              ambiguous: tuple[MediaRange, ...]) -> tuple[dict, tuple[tuple[int, int], ...]]:
    matches = _match_candidate_windows(truth, predictions)
    matched_predictions = {pred_index for _, pred_index in matches}
    excused = {
        index for index, prediction in enumerate(predictions)
        if index not in matched_predictions and any(
            iou(prediction["range"], region) >= IOU_THRESHOLD for region in ambiguous
        )
    }
    tp = len(matches)
    fp = len(predictions) - tp - len(excused)
    fn = len(truth) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(truth) if truth else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else
          (0.0 if truth or predictions else None))
    return ({
        "matching_rule": "one-to-one; a coarse candidate window must contain the labeled contact",
        "truth_events": len(truth),
        "predicted_candidates": len(predictions),
        "matched_events": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "excused_ambiguous": len(excused),
        "unavailable_truth_contacts": sum(item["contact_seconds"] is None for item in truth),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, matches)


def evaluate_event_sets(strata: dict[str, dict], tolerances: tuple[float, ...]) -> dict:
    """Report coarse event coverage, contact localization, and boundaries."""
    per_recording = {}
    overall_by_tolerance = {}
    candidate_window_recordings = {}
    candidate_window_tp = candidate_window_fp = candidate_window_fn = 0
    candidate_window_excused = 0
    boundary_pairs = {}
    for name in sorted(strata):
        truth = strata[name]["truth_events"]
        predictions = strata[name]["prediction_events"]
        metrics, matches = _candidate_window_metrics(truth, predictions, strata[name]["ambiguous"])
        candidate_window_recordings[name] = metrics
        boundary_pairs[name] = matches
        candidate_window_tp += metrics["matched_events"]
        candidate_window_fp += metrics["false_positives"]
        candidate_window_fn += metrics["false_negatives"]
        candidate_window_excused += metrics["excused_ambiguous"]
    for tolerance in tolerances:
        tp = fp = fn = excused = truth_count = prediction_count = 0
        error_values: list[float] = []
        per_recording[str(tolerance)] = {}
        for name in sorted(strata):
            truth = strata[name]["truth_events"]
            predictions = strata[name]["prediction_events"]
            metrics, matches, _ = _event_metrics(truth, predictions, strata[name]["ambiguous"], tolerance)
            per_recording[str(tolerance)][name] = metrics
            tp += metrics["matched_events"]
            fp += metrics["false_positives"]
            fn += metrics["false_negatives"]
            excused += metrics["excused_ambiguous"]
            truth_count += metrics["truth_events"]
            prediction_count += metrics["predicted_events"]
            error_values.extend(match.contact_error_seconds * 1000 for match in matches)
        scored_predictions = prediction_count - excused
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / truth_count if truth_count else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else
              (0.0 if truth_count or scored_predictions else None))
        overall_by_tolerance[str(tolerance)] = {
            "contact_tolerance_ms": tolerance * 1000,
            "truth_events": truth_count,
            "predicted_events": prediction_count,
            "matched_events": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "excused_ambiguous": excused,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_abs_contact_error_ms": sum(abs(value) for value in error_values) / len(error_values) if error_values else None,
            "mean_signed_contact_error_ms": sum(error_values) / len(error_values) if error_values else None,
        }

    boundary_rows = []
    for name in sorted(strata):
        truth = strata[name]["truth_events"]
        predictions = strata[name]["prediction_events"]
        for truth_index, pred_index in boundary_pairs.get(name, ()):
            expected = truth[truth_index]["range"]
            candidate = predictions[pred_index]["range"]
            start_error = candidate.start_seconds - expected.start_seconds
            end_error = candidate.end_seconds - expected.end_seconds
            predicted_contact = predictions[pred_index]["contact_seconds"]
            truth_contact = truth[truth_index]["contact_seconds"]
            boundary_rows.append({
                "recording": name,
                "truth_id": truth[truth_index]["id"],
                "prediction_id": predictions[pred_index]["id"],
                "contact_error_ms": ((predicted_contact - truth_contact) * 1000
                                     if predicted_contact is not None and truth_contact is not None else None),
                "start_error_ms": start_error * 1000,
                "end_error_ms": end_error * 1000,
                "candidate_duration_ms": candidate.duration_seconds * 1000,
                "truth_duration_ms": expected.duration_seconds * 1000,
                "candidate_overhang_before_ms": max(0.0, -start_error) * 1000,
                "candidate_overhang_after_ms": max(0.0, end_error) * 1000,
                "candidate_undercoverage_before_ms": max(0.0, start_error) * 1000,
                "candidate_undercoverage_after_ms": max(0.0, -end_error) * 1000,
            })
    if boundary_rows:
        def mean_abs(key: str) -> float:
            return sum(abs(row[key]) for row in boundary_rows) / len(boundary_rows)
        def mean_signed(key: str) -> float:
            return sum(row[key] for row in boundary_rows) / len(boundary_rows)
        boundary_summary = {
            "matched_events_by_candidate_window": True,
            "matched_events": len(boundary_rows),
            "mean_abs_start_error_ms": mean_abs("start_error_ms"),
            "mean_signed_start_error_ms": mean_signed("start_error_ms"),
            "mean_abs_end_error_ms": mean_abs("end_error_ms"),
            "mean_signed_end_error_ms": mean_signed("end_error_ms"),
            "mean_candidate_duration_ms": sum(row["candidate_duration_ms"] for row in boundary_rows) / len(boundary_rows),
            "mean_truth_duration_ms": sum(row["truth_duration_ms"] for row in boundary_rows) / len(boundary_rows),
            "mean_candidate_overhang_before_ms": sum(row["candidate_overhang_before_ms"] for row in boundary_rows) / len(boundary_rows),
            "mean_candidate_overhang_after_ms": sum(row["candidate_overhang_after_ms"] for row in boundary_rows) / len(boundary_rows),
        }
    else:
        boundary_summary = {"matched_events_by_candidate_window": True, "matched_events": 0}
    return {
        "event_detection": {
            "matching_rule": "one-to-one candidate-window coverage of labeled contact; independent of candidate boundaries",
            "truth_events": candidate_window_tp + candidate_window_fn,
            "predicted_candidates": candidate_window_tp + candidate_window_fp + candidate_window_excused,
            "matched_events": candidate_window_tp,
            "false_positives": candidate_window_fp,
            "false_negatives": candidate_window_fn,
            "excused_ambiguous": candidate_window_excused,
            "precision": (candidate_window_tp / (candidate_window_tp + candidate_window_fp)
                          if candidate_window_tp + candidate_window_fp else None),
            "recall": (candidate_window_tp / (candidate_window_tp + candidate_window_fn)
                       if candidate_window_tp + candidate_window_fn else None),
            "f1": (2 * candidate_window_tp / (2 * candidate_window_tp + candidate_window_fp + candidate_window_fn)
                   if 2 * candidate_window_tp + candidate_window_fp + candidate_window_fn else None),
            "per_recording": candidate_window_recordings,
        },
        "contact_localization": {
            "matching_rule": "one-to-one minimum absolute contact-time error, then stable index tie-break",
        "tolerances": [tolerance * 1000 for tolerance in tolerances],
        "overall_by_tolerance": overall_by_tolerance,
        "per_recording_by_tolerance": per_recording,
        },
        "boundary_localization_after_candidate_event_match": {
            "matching_rule": "coarse candidate window contains labeled contact",
            **boundary_summary,
            "matches": boundary_rows,
        },
    }


def false_positive_classes(predictions: tuple[MediaRange, ...], truth: tuple[MediaRange, ...],
                           ambiguous: tuple[MediaRange, ...], negatives: list[tuple[str, MediaRange]]) -> dict[str, int]:
    metrics = evaluate_report({"recording": StratumInput(truth, predictions, ambiguous)}).strata[0].metrics
    matched = {match.pred_index for match in metrics.matches}
    result = {"shadow_swing": 0, "toss_abort": 0, "other": 0, "unlabeled": 0,
              "ambiguous_excused": 0}
    for index, prediction in enumerate(predictions):
        if index in matched:
            continue
        if any(iou(prediction, region) >= IOU_THRESHOLD for region in ambiguous):
            result["ambiguous_excused"] += 1
            continue
        categories = [(iou(prediction, region), label) for label, region in negatives
                      if iou(prediction, region) >= IOU_THRESHOLD]
        if categories:
            result[max(categories)[1]] += 1
        else:
            result["unlabeled"] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--alignment-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    alignment = json.loads((args.alignment_dir / "fit.json").read_text())
    fit = alignment["fits"]["smoothstep"]
    entry = float(fit["entry_seconds"])
    exit_ = float(fit["exit_seconds"])
    shape = "smoothstep"
    annotation_doc = json.loads(args.annotations.read_text())
    annotations = annotation_doc["videos"]
    export_strata: dict[str, StratumInput] = {}
    original_strata: dict[str, StratumInput] = {}
    export_event_strata: dict[str, dict] = {}
    original_event_strata: dict[str, dict] = {}
    transferred: list[dict] = []
    detail: dict[str, dict] = {}

    for original, export, aae in pair_exports(args.original_dir, args.export_dir):
        name = export.name
        if name not in annotations:
            # Photos sometimes strips the duplicate suffix in the annotation UI.
            fallback = next((key for key in annotations if re.sub(r"\(\d+\)(?=\.)", "", key).casefold()
                             == name.casefold()), None)
            if fallback is None:
                continue
            name = fallback
        labels = annotations[name].get("labels", [])
        region = read_slow_region(aae)
        exp_duration = float(json.loads((args.export_dir / "metadata" / export.stem / "source.json").read_text())
                             .get("duration_seconds", 0.0))
        orig_meta = json.loads((args.original_dir / "metadata" / original.stem / "source.json").read_text())
        orig_duration = float(orig_meta["duration_seconds"])
        serve_labels = [item for item in labels if item["label"] == LABEL_SERVE]
        export_truth_events = []
        original_truth_events = []
        for label in serve_labels:
            bounds = MediaRange(float(label["start_seconds"]), float(label["end_seconds"]))
            contact = label.get("checkpoints", {}).get("contact", {})
            contact_time = contact.get("time_seconds") if isinstance(contact, dict) else None
            if isinstance(contact_time, bool) or not isinstance(contact_time, (int, float)):
                contact_time = None
            export_truth_events.append({"id": label["id"], "range": bounds,
                                        "contact_seconds": float(contact_time) if contact_time is not None else None})
            original_truth_events.append({
                "id": label["id"],
                "range": map_range(label, region, entry, exit_, shape, orig_duration),
                "contact_seconds": invert_map(float(contact_time), region, entry, exit_, shape, orig_duration)
                                 if contact_time is not None else None,
            })
        export_truth = tuple(item["range"] for item in export_truth_events)
        export_ambiguous = tuple(MediaRange(float(item["start_seconds"]), float(item["end_seconds"]))
                                 for item in labels if item["label"] == LABEL_AMBIGUOUS)
        export_negatives = [(item["label"], MediaRange(float(item["start_seconds"]), float(item["end_seconds"])))
                            for item in labels if item["label"] in {"shadow_swing", "toss_abort", "other"}]
        orig_truth = tuple(item["range"] for item in original_truth_events)
        orig_ambiguous = tuple(map_range(item, region, entry, exit_, shape, orig_duration)
                               for item in labels if item["label"] == LABEL_AMBIGUOUS)
        orig_negatives = [(label, map_range(item, region, entry, exit_, shape, orig_duration))
                          for item in labels if (label := item["label"]) in {"shadow_swing", "toss_abort", "other"}]
        export_prediction_events = load_detection_events(args.export_dir / "metadata" / export.stem)
        original_prediction_events = load_detection_events(args.original_dir / "metadata" / original.stem)
        export_predictions = tuple(item["range"] for item in export_prediction_events)
        original_predictions = tuple(item["range"] for item in original_prediction_events)
        export_strata[original.stem] = StratumInput(export_truth, export_predictions, export_ambiguous)
        original_strata[original.stem] = StratumInput(orig_truth, original_predictions, orig_ambiguous)
        export_event_strata[original.stem] = {
            "truth_events": export_truth_events,
            "prediction_events": export_prediction_events,
            "ambiguous": export_ambiguous,
        }
        original_event_strata[original.stem] = {
            "truth_events": original_truth_events,
            "prediction_events": original_prediction_events,
            "ambiguous": orig_ambiguous,
        }
        for source_label, mapped in zip(serve_labels, orig_truth, strict=True):
            transferred.append({"recording": original.stem, "label_id": source_label["id"],
                                "export_start_seconds": source_label["start_seconds"],
                                "export_end_seconds": source_label["end_seconds"],
                                "original_start_seconds": mapped.start_seconds,
                                "original_end_seconds": mapped.end_seconds})
        detail[original.stem] = {
            "export_file": export.name,
            "rate": region.rate,
            "original_duration_seconds": orig_duration,
            "export_duration_seconds": exp_duration,
            "labeled_serves": len(export_truth),
            "export_predictions": len(export_predictions),
            "original_predictions": len(original_predictions),
            "export_false_positive_classes": false_positive_classes(export_predictions, export_truth,
                                                                      export_ambiguous, export_negatives),
            "original_false_positive_classes": false_positive_classes(original_predictions, orig_truth,
                                                                        orig_ambiguous, orig_negatives),
        }

    if not export_strata:
        raise ValueError("No annotated paired recordings found")
    export_report = evaluate_report(export_strata)
    original_report = evaluate_report(original_strata)
    tolerances = (0.25, 0.5, 1.0)
    export_events_report = evaluate_event_sets(export_event_strata, tolerances)
    original_events_report = evaluate_event_sets(original_event_strata, tolerances)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "transferred_labels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(transferred[0]))
        writer.writeheader()
        writer.writerows(transferred)
    output = {
        "method": "smoothstep inverse AAE map, shared September 8 fit; labels mapped export→original",
        "transition_fit_seconds": {"entry": entry, "exit": exit_, "shape": shape},
        "event_match_tolerances_ms": [250, 500, 1000],
        "boundary_match_tolerance_ms": 500,
        "limitation": "Same-session retrospective comparison; pose alignment and model fit use these recordings.",
        "recordings": detail,
        "event_detection": {
            "google_export": export_events_report,
            "original_with_transferred_labels": original_events_report,
        },
        "interval_iou_diagnostic": {
            "iou_threshold": IOU_THRESHOLD,
            "google_export": export_report.to_dict(),
            "original_with_transferred_labels": original_report.to_dict(),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    for label, report in (("Google export", export_events_report),
                          ("Original with transferred labels", original_events_report)):
        print(label)
        for tolerance_ms, metrics in report["contact_localization"]["overall_by_tolerance"].items():
            print(f"  contact ±{int(float(tolerance_ms) * 1000)}ms: "
                  f"TP={metrics['matched_events']}, FP={metrics['false_positives']}, "
                  f"FN={metrics['false_negatives']}, F1={metrics['f1']}")
        coverage = report["event_detection"]
        print(f"  coarse candidate coverage: TP={coverage['matched_events']}, "
              f"FP={coverage['false_positives']}, FN={coverage['false_negatives']}, "
              f"F1={coverage['f1']}")
        boundary = report["boundary_localization_after_candidate_event_match"]
        print(f"  matched boundary events={boundary['matched_events']}, "
              f"start MAE={boundary.get('mean_abs_start_error_ms')}, "
              f"end MAE={boundary.get('mean_abs_end_error_ms')}")
    print(f"Wrote {args.output_dir / 'report.json'} and transferred_labels.csv")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
