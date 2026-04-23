"""
Weekly evaluation metrics for final agenda quality.

All the metrics that matter: task precision/recall/F1, time accuracy
at various tolerances, overlap counts, and improvement over baselines.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluation.matching import hungarian_match, _label


def _count_overlaps(
    events: list[dict[str, Any]],
    scope: str = "same_device",
) -> int:
    """Count overlapping event pairs."""
    if scope == "same_device":
        by_device: dict[str, list[dict]] = defaultdict(list)
        for ev in events:
            dev = ev.get("device_id", "__default__")
            by_device[dev].append(ev)
        total = 0
        for dev_events in by_device.values():
            sorted_evs = sorted(dev_events, key=lambda e: e.get("start_bin", 0))
            for i in range(len(sorted_evs)):
                for j in range(i + 1, len(sorted_evs)):
                    si = sorted_evs[i].get("start_bin", 0)
                    di = sorted_evs[i].get("duration_bins", sorted_evs[i].get("duration", 0))
                    sj = sorted_evs[j].get("start_bin", 0)
                    if sj < si + di:
                        total += 1
                    else:
                        break
        return total
    else:
        # Global
        sorted_evs = sorted(events, key=lambda e: e.get("start_bin", 0))
        total = 0
        for i in range(len(sorted_evs)):
            for j in range(i + 1, len(sorted_evs)):
                si = sorted_evs[i].get("start_bin", 0)
                di = sorted_evs[i].get("duration_bins", sorted_evs[i].get("duration", 0))
                sj = sorted_evs[j].get("start_bin", 0)
                if sj < si + di:
                    total += 1
                else:
                    break
        return total


def evaluate_weekly_predictions(
    true_tasks: list[dict[str, Any]],
    pred_tasks: list[dict[str, Any]],
    bin_minutes: int = 5,
    time_tolerance_minutes: int = 5,
    secondary_tolerance_minutes: int = 10,
    duration_tolerance_minutes: int = 2,
) -> dict[str, float]:
    """
    Compute full weekly evaluation metrics.

    Returns dict with:
      task_precision, task_recall, task_f1,
      time_exact_accuracy, time_close_accuracy_5m, time_close_accuracy_10m,
      start_mae_minutes, duration_mae_minutes,
      overlap_same_device_count, overlap_global_count,
    """
    tol_bins = time_tolerance_minutes // bin_minutes
    tol2_bins = secondary_tolerance_minutes // bin_minutes

    # Task-level P/R/F1
    true_by_task: dict[str, int] = defaultdict(int)
    pred_by_task: dict[str, int] = defaultdict(int)
    for t in true_tasks:
        true_by_task[_label(t)] += 1
    for p in pred_tasks:
        pred_by_task[_label(p)] += 1

    all_tasks = set(true_by_task) | set(pred_by_task)
    tp = sum(min(true_by_task.get(t, 0), pred_by_task.get(t, 0)) for t in all_tasks)
    total_true = sum(true_by_task.values())
    total_pred = sum(pred_by_task.values())

    precision = tp / max(total_pred, 1) * 100.0
    recall = tp / max(total_true, 1) * 100.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Time accuracy via Hungarian matching
    pairs = hungarian_match(true_tasks, pred_tasks, bin_minutes)

    time_exact = 0
    time_5m = 0
    time_10m = 0
    start_error_sum = 0.0
    dur_error_sum = 0.0
    matched = len(pairs)

    for true_item, pred_item in pairs:
        diff_bins = abs(int(true_item["start_bin"]) - int(pred_item["start_bin"]))
        diff_min = diff_bins * bin_minutes
        start_error_sum += diff_min

        if diff_bins == 0:
            time_exact += 1
        if diff_bins <= tol_bins:
            time_5m += 1
        if diff_bins <= tol2_bins:
            time_10m += 1

        dur_diff = abs(float(true_item.get("duration", 0)) - float(pred_item.get("duration", 0)))
        dur_error_sum += dur_diff

    # Overlaps
    overlap_same = _count_overlaps(pred_tasks, "same_device")
    overlap_global = _count_overlaps(pred_tasks, "global")

    return {
        "task_precision": precision,
        "task_recall": recall,
        "task_f1": f1,
        "time_exact_accuracy": (time_exact / max(matched, 1)) * 100.0,
        "time_close_accuracy_5m": (time_5m / max(matched, 1)) * 100.0,
        "time_close_accuracy_10m": (time_10m / max(matched, 1)) * 100.0,
        "start_mae_minutes": start_error_sum / max(matched, 1),
        "duration_mae_minutes": dur_error_sum / max(matched, 1),
        "overlap_same_device_count": overlap_same,
        "overlap_global_count": overlap_global,
        "matched_pairs": matched,
        "total_true": total_true,
        "total_pred": total_pred,
    }
