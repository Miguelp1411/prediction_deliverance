from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import MATCH_DUMMY_PENALTY, MATCH_TASK_MISMATCH_PENALTY

BIN_MINUTES = 5


def task_cost(real, pred):
    if real['task_name'] != pred['task_name']:
        return MATCH_TASK_MISMATCH_PENALTY
    return abs(real['start_bin'] - pred['start_bin']) * BIN_MINUTES + abs(real['duration'] - pred['duration'])


def _hungarian_same_task(true_tasks, pred_tasks):
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
    true_by_task = defaultdict(list)
    pred_by_task = defaultdict(list)
    for item in true_tasks:
        true_by_task[item['task_name']].append(item)
    for item in pred_tasks:
        pred_by_task[item['task_name']].append(item)
    pairs = []
    for task_name in sorted(set(true_by_task) | set(pred_by_task)):
        tg = sorted(true_by_task.get(task_name, []), key=lambda x: x['start_bin'])
        pg = sorted(pred_by_task.get(task_name, []), key=lambda x: x['start_bin'])
        pairs.extend(_hungarian_same_task(tg, pg))
    return pairs
