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



def pair_template_slots_to_events(template_slots: list, target_events: list, bins_per_day: int, dummy_penalty: float = 10_000.0):
    if not template_slots or not target_events:
        return []
    n = max(len(template_slots), len(target_events))
    cost = np.full((n, n), dummy_penalty, dtype=np.float64)
    for i, slot in enumerate(template_slots):
        slot_start = int(slot.start_bin if hasattr(slot, 'start_bin') else slot['start_bin'])
        slot_dur = int(slot.duration_bins if hasattr(slot, 'duration_bins') else slot['duration_bins'])
        for j, evt in enumerate(target_events):
            start_gap = abs(slot_start - int(evt.start_bin))
            day_gap = abs(slot_start // bins_per_day - int(evt.start_bin) // bins_per_day)
            dur_gap = abs(slot_dur - int(evt.duration_bins))
            cost[i, j] = float(4.0 * day_gap + start_gap + 0.25 * dur_gap)
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        if i < len(template_slots) and j < len(target_events) and cost[i, j] < dummy_penalty:
            pairs.append((template_slots[i], target_events[j], float(cost[i, j])))
    return pairs
