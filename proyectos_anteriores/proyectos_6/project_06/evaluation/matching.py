from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from proyectos_anteriores.proyectos_6.project_06.config import MATCH_DUMMY_PENALTY, MATCH_TASK_MISMATCH_PENALTY, BIN_MINUTES


def _label(item: dict[str, Any]) -> str:
    if 'task_name' in item:
        return str(item['task_name'])
    if 'type' in item:
        return str(item['type'])
    raise KeyError("Cada item debe incluir 'task_name' o 'type'")


def _normalized(item: dict[str, Any]) -> dict[str, Any]:
    return {
        'task_name': _label(item),
        'start_bin': int(item['start_bin']),
        'duration': float(item['duration']),
    }


def task_cost(real: dict[str, Any], pred: dict[str, Any]) -> float:
    real_name = _label(real)
    pred_name = _label(pred)
    if real_name != pred_name:
        return MATCH_TASK_MISMATCH_PENALTY
    return abs(int(real['start_bin']) - int(pred['start_bin'])) * BIN_MINUTES + abs(float(real['duration']) - float(pred['duration']))


def _hungarian_same_task(true_tasks: list[dict[str, Any]], pred_tasks: list[dict[str, Any]]):
    n, m = len(true_tasks), len(pred_tasks)
    size = max(n, m)
    cost_matrix = np.full((size, size), MATCH_DUMMY_PENALTY, dtype=np.float32)
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = task_cost(true_tasks[i], pred_tasks[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m and cost_matrix[r, c] < MATCH_DUMMY_PENALTY:
            pairs.append((true_tasks[r], pred_tasks[c]))
    return pairs


def hungarian_match(true_tasks, pred_tasks):
    normalized_true = [_normalized(item) for item in true_tasks]
    normalized_pred = [_normalized(item) for item in pred_tasks]

    true_by_task = defaultdict(list)
    pred_by_task = defaultdict(list)
    for item in normalized_true:
        true_by_task[item['task_name']].append(item)
    for item in normalized_pred:
        pred_by_task[item['task_name']].append(item)

    pairs = []
    for task_name in sorted(set(true_by_task) | set(pred_by_task)):
        tg = sorted(true_by_task.get(task_name, []), key=lambda x: x['start_bin'])
        pg = sorted(pred_by_task.get(task_name, []), key=lambda x: x['start_bin'])
        pairs.extend(_hungarian_same_task(tg, pg))
    return pairs
