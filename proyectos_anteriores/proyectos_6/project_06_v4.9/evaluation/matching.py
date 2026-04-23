"""
Evaluation matching — Hungarian matching for weekly schedule comparison.

Ported from v4.9 with parameterized config (no hardcoded imports).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _label(item: dict[str, Any]) -> str:
    if "task_name" in item:
        return str(item["task_name"])
    if "type" in item:
        return str(item["type"])
    raise KeyError("Each item must include 'task_name' or 'type'")


def _normalized(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_name": _label(item),
        "start_bin": int(item["start_bin"]),
        "duration": float(item.get("duration", item.get("duration_minutes", item.get("duration_bins", 0)))),
    }


def task_cost(
    real: dict[str, Any],
    pred: dict[str, Any],
    bin_minutes: int = 5,
    mismatch_penalty: float = 10_000.0,
) -> float:
    if _label(real) != _label(pred):
        return mismatch_penalty
    return abs(int(real["start_bin"]) - int(pred["start_bin"])) * bin_minutes + abs(
        float(real.get("duration", 0)) - float(pred.get("duration", 0))
    )


def _group_by_task(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        grouped[item["task_name"]].append(item)
    for task_name in grouped:
        grouped[task_name] = sorted(grouped[task_name], key=lambda x: x["start_bin"])
    return grouped


def hungarian_match(
    true_tasks: list[dict],
    pred_tasks: list[dict],
    bin_minutes: int = 5,
    mismatch_penalty: float = 10_000.0,
    dummy_penalty: float = 25_000.0,
) -> list[tuple[dict, dict]]:
    """Match true and predicted tasks using Hungarian algorithm per task type."""
    norm_true = [_normalized(t) for t in true_tasks]
    norm_pred = [_normalized(p) for p in pred_tasks]
    true_by_task = _group_by_task(norm_true)
    pred_by_task = _group_by_task(norm_pred)

    pairs: list[tuple[dict, dict]] = []
    for task_name in sorted(set(true_by_task) | set(pred_by_task)):
        tg = true_by_task.get(task_name, [])
        pg = pred_by_task.get(task_name, [])
        n, m = len(tg), len(pg)
        size = max(n, m)
        if size == 0:
            continue
        cost_matrix = np.full((size, size), dummy_penalty, dtype=np.float32)
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = task_cost(tg[i], pg[j], bin_minutes, mismatch_penalty)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if r < n and c < m and cost_matrix[r, c] < dummy_penalty:
                pairs.append((tg[r], pg[c]))
    return pairs
