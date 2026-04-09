from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import MATCH_DUMMY_PENALTY, MATCH_TASK_MISMATCH_PENALTY, BIN_MINUTES


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


def _hungarian_same_task_indices(true_tasks: list[dict[str, Any]], pred_tasks: list[dict[str, Any]]) -> list[tuple[int, int]]:
    n, m = len(true_tasks), len(pred_tasks)
    size = max(n, m)
    if size == 0:
        return []
    cost_matrix = np.full((size, size), MATCH_DUMMY_PENALTY, dtype=np.float32)
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = task_cost(true_tasks[i], pred_tasks[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    pairs: list[tuple[int, int]] = []
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r < n and c < m and cost_matrix[r, c] < MATCH_DUMMY_PENALTY:
            pairs.append((int(r), int(c)))
    return pairs


def _ordered_same_task_indices(true_tasks: list[dict[str, Any]], pred_tasks: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(idx, idx) for idx in range(min(len(true_tasks), len(pred_tasks)))]


def _pairs_from_indices(true_tasks: list[dict[str, Any]], pred_tasks: list[dict[str, Any]], indices: list[tuple[int, int]]):
    return [(true_tasks[r], pred_tasks[c]) for r, c in indices]


def _hungarian_same_task(true_tasks: list[dict[str, Any]], pred_tasks: list[dict[str, Any]]):
    return _pairs_from_indices(true_tasks, pred_tasks, _hungarian_same_task_indices(true_tasks, pred_tasks))


def _group_by_task(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for item in items:
        grouped[item['task_name']].append(item)
    for task_name in grouped:
        grouped[task_name] = sorted(grouped[task_name], key=lambda x: x['start_bin'])
    return grouped


def ordered_match(true_tasks, pred_tasks):
    normalized_true = [_normalized(item) for item in true_tasks]
    normalized_pred = [_normalized(item) for item in pred_tasks]
    true_by_task = _group_by_task(normalized_true)
    pred_by_task = _group_by_task(normalized_pred)

    pairs = []
    for task_name in sorted(set(true_by_task) | set(pred_by_task)):
        tg = true_by_task.get(task_name, [])
        pg = pred_by_task.get(task_name, [])
        pairs.extend(_pairs_from_indices(tg, pg, _ordered_same_task_indices(tg, pg)))
    return pairs


def hungarian_match(true_tasks, pred_tasks):
    normalized_true = [_normalized(item) for item in true_tasks]
    normalized_pred = [_normalized(item) for item in pred_tasks]

    true_by_task = _group_by_task(normalized_true)
    pred_by_task = _group_by_task(normalized_pred)

    pairs = []
    for task_name in sorted(set(true_by_task) | set(pred_by_task)):
        tg = true_by_task.get(task_name, [])
        pg = pred_by_task.get(task_name, [])
        pairs.extend(_pairs_from_indices(tg, pg, _hungarian_same_task_indices(tg, pg)))
    return pairs


def matching_diagnostics(true_tasks, pred_tasks, time_tolerance_minutes: int = 5) -> dict[str, float]:
    normalized_true = [_normalized(item) for item in true_tasks]
    normalized_pred = [_normalized(item) for item in pred_tasks]
    true_by_task = _group_by_task(normalized_true)
    pred_by_task = _group_by_task(normalized_pred)

    reassigned_pairs = 0.0
    assignment_shift_sum = 0.0
    ordered_exact_count = 0.0
    ordered_close_5m_count = 0.0
    hungarian_exact_count = 0.0
    hungarian_close_5m_count = 0.0
    ordered_cost_sum = 0.0
    hungarian_cost_sum = 0.0
    compared_pairs = 0.0

    for task_name in sorted(set(true_by_task) | set(pred_by_task)):
        tg = true_by_task.get(task_name, [])
        pg = pred_by_task.get(task_name, [])
        ordered_indices = _ordered_same_task_indices(tg, pg)
        hungarian_indices = _hungarian_same_task_indices(tg, pg)

        ordered_map = {pred_idx: true_idx for true_idx, pred_idx in ordered_indices}
        hungarian_map = {pred_idx: true_idx for true_idx, pred_idx in hungarian_indices}
        for pred_idx in sorted(set(ordered_map) & set(hungarian_map)):
            compared_pairs += 1.0
            if ordered_map[pred_idx] != hungarian_map[pred_idx]:
                reassigned_pairs += 1.0
                assignment_shift_sum += abs(int(ordered_map[pred_idx]) - int(hungarian_map[pred_idx]))

        for true_idx, pred_idx in ordered_indices:
            true_item, pred_item = tg[true_idx], pg[pred_idx]
            diff_minutes = abs(int(true_item['start_bin']) - int(pred_item['start_bin'])) * BIN_MINUTES
            ordered_cost_sum += task_cost(true_item, pred_item)
            if diff_minutes == 0:
                ordered_exact_count += 1.0
            if diff_minutes <= time_tolerance_minutes:
                ordered_close_5m_count += 1.0

        for true_idx, pred_idx in hungarian_indices:
            true_item, pred_item = tg[true_idx], pg[pred_idx]
            diff_minutes = abs(int(true_item['start_bin']) - int(pred_item['start_bin'])) * BIN_MINUTES
            hungarian_cost_sum += task_cost(true_item, pred_item)
            if diff_minutes == 0:
                hungarian_exact_count += 1.0
            if diff_minutes <= time_tolerance_minutes:
                hungarian_close_5m_count += 1.0

    reassignment_rate = reassigned_pairs / compared_pairs if compared_pairs > 0 else 0.0
    avg_assignment_shift = assignment_shift_sum / reassigned_pairs if reassigned_pairs > 0 else 0.0
    return {
        'matching_reassigned_pairs': reassigned_pairs,
        'matching_assignment_shift_sum': assignment_shift_sum,
        'matching_compared_pairs': compared_pairs,
        'matching_reassignment_rate': reassignment_rate,
        'matching_avg_assignment_shift': avg_assignment_shift,
        'matching_ordered_exact_count': ordered_exact_count,
        'matching_hungarian_exact_count': hungarian_exact_count,
        'matching_ordered_close_5m_count': ordered_close_5m_count,
        'matching_hungarian_close_5m_count': hungarian_close_5m_count,
        'matching_ordered_cost_sum': ordered_cost_sum,
        'matching_hungarian_cost_sum': hungarian_cost_sum,
        'matching_cost_gain_sum': ordered_cost_sum - hungarian_cost_sum,
    }
