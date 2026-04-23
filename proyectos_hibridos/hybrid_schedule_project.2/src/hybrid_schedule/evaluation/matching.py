from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment



def hungarian_match(pred_events: list[dict], true_events: list[dict], task_mismatch_penalty: float = 10_000.0, dummy_penalty: float = 25_000.0):
    n = max(len(pred_events), len(true_events))
    if n == 0:
        return []
    cost = np.full((n, n), dummy_penalty, dtype=np.float64)
    for i, pred in enumerate(pred_events):
        for j, truth in enumerate(true_events):
            if pred['task_type'] != truth['task_type']:
                cost[i, j] = task_mismatch_penalty
                continue
            start_cost = abs(int(pred['start_bin']) - int(truth['start_bin']))
            dur_cost = abs(int(pred['duration_bins']) - int(truth['duration_bins']))
            cost[i, j] = start_cost + 0.25 * dur_cost
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if i < len(pred_events) and j < len(true_events) and cost[i, j] < task_mismatch_penalty:
            pairs.append((pred_events[i], true_events[j], float(cost[i, j])))
    return pairs
