from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluation.matching import hungarian_match

BIN_MINUTES = 5


def _event_label(event: Any) -> str:
    if hasattr(event, 'task_name'):
        return str(event.task_name)
    if hasattr(event, 'type'):
        return str(event.type)
    raise AttributeError("El evento no tiene ni 'task_name' ni 'type'")


def _pred_label(item: dict[str, Any]) -> str:
    if 'task_name' in item:
        return str(item['task_name'])
    if 'type' in item:
        return str(item['type'])
    raise KeyError("La predicción no contiene ni 'task_name' ni 'type'")


def _normalize_true(true_tasks):
    if hasattr(true_tasks, 'events_by_task'):
        out = []
        for events in true_tasks.events_by_task.values():
            for e in events:
                out.append({
                    'task_name': _event_label(e),
                    'device_uid': getattr(e, 'device_uid', None),
                    'start_bin': int(e.start_bin),
                    'duration': float(e.duration_minutes),
                })
        return out
    return [{
        'task_name': str(item.get('task_name', item.get('type'))),
        'device_uid': item.get('device_uid'),
        'start_bin': int(item['start_bin']),
        'duration': float(item['duration']),
    } for item in true_tasks]


def _normalize_pred(pred_tasks):
    return [{
        'task_name': _pred_label(item),
        'device_uid': item.get('device_uid'),
        'start_bin': int(item['start_bin']),
        'duration': float(item['duration']),
    } for item in pred_tasks]


def _count_overlaps(pred_tasks: list[dict[str, Any]]) -> int:
    intervals_by_device: dict[str | None, list[tuple[int, int]]] = defaultdict(list)
    for item in pred_tasks:
        start = int(item['start_bin'])
        duration_bins = max(1, int(round(float(item['duration']) / BIN_MINUTES)))
        intervals_by_device[item.get('device_uid')].append((start, start + duration_bins))

    overlaps = 0
    for intervals in intervals_by_device.values():
        intervals.sort()
        prev_end = None
        for start, end in intervals:
            if prev_end is not None and start < prev_end:
                overlaps += 1
                prev_end = max(prev_end, end)
            else:
                prev_end = end
    return overlaps


def _result_with_aliases(total_tasks, correct_tasks, time_exact, time_5, time_10, duration_2, start_mae_minutes, overlap_count, per_task):
    result = {
        'total_tasks': float(total_tasks),
        'correct_tasks': float(correct_tasks),
        'task_accuracy': float(correct_tasks / total_tasks) if total_tasks > 0 else 0.0,
        'time_exact_accuracy': float(time_exact / total_tasks) if total_tasks > 0 else 0.0,
        'time_close_accuracy': float(time_5 / total_tasks) if total_tasks > 0 else 0.0,
        'time_close_accuracy_5m': float(time_5 / total_tasks) if total_tasks > 0 else 0.0,
        'time_close_accuracy_10m': float(time_10 / total_tasks) if total_tasks > 0 else 0.0,
        'duration_close_accuracy': float(duration_2 / total_tasks) if total_tasks > 0 else 0.0,
        'start_mae_minutes': float(start_mae_minutes),
        'overlap_count': float(overlap_count),
        'per_task': per_task,
    }
    result.update({
        'tasks_per_week': result['total_tasks'],
        'task_correct': result['correct_tasks'],
        'task_acc': result['task_accuracy'] * 100.0,
        'start_exact_acc': result['time_exact_accuracy'] * 100.0,
        'start_tol_acc': result['time_close_accuracy'] * 100.0,
        'start_tol_acc_5m': result['time_close_accuracy_5m'] * 100.0,
        'start_tol_acc_10m': result['time_close_accuracy_10m'] * 100.0,
        'duration_tol_acc_2m': result['duration_close_accuracy'] * 100.0,
    })
    return result


def evaluate_weekly_predictions(true_tasks, pred_tasks, time_tolerance_minutes: int = 5, secondary_time_tolerance_minutes: int = 10, duration_tolerance_minutes: int = 2):
    true_tasks = sorted(_normalize_true(true_tasks), key=lambda x: (x['start_bin'], x['task_name']))
    pred_tasks = sorted(_normalize_pred(pred_tasks), key=lambda x: (x['start_bin'], x['task_name']))

    total_tasks = len(true_tasks)
    if total_tasks == 0:
        return _result_with_aliases(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    task_correct = time_exact = time_close_5m = time_close_10m = duration_close = 0
    start_abs_error_sum = 0.0
    per_task_counts = defaultdict(lambda: {'total': 0.0, 'task_correct': 0.0, 'time_exact': 0.0})

    pairs = hungarian_match(true_tasks, pred_tasks)
    for true_item, pred_item in pairs:
        device_uid = true_item.get('device_uid')
        task_name = true_item['task_name'] if device_uid in (None, '') else f"{true_item['task_name']}@{device_uid}"
        per_task_counts[task_name]['total'] += 1.0
        if true_item['task_name'] == pred_item['task_name']:
            task_correct += 1
            per_task_counts[task_name]['task_correct'] += 1.0
        diff_minutes = abs(int(true_item['start_bin']) - int(pred_item['start_bin'])) * BIN_MINUTES
        start_abs_error_sum += diff_minutes
        if diff_minutes == 0:
            time_exact += 1
            per_task_counts[task_name]['time_exact'] += 1.0
        if diff_minutes <= time_tolerance_minutes:
            time_close_5m += 1
        if diff_minutes <= secondary_time_tolerance_minutes:
            time_close_10m += 1
        if abs(float(true_item['duration']) - float(pred_item['duration'])) <= duration_tolerance_minutes:
            duration_close += 1

    matched_pairs = max(len(pairs), 1)
    start_mae_minutes = start_abs_error_sum / matched_pairs
    per_task = {
        task: {
            'total': float(values['total']),
            'task_accuracy': float(values['task_correct'] / values['total']) if values['total'] > 0 else 0.0,
            'time_exact_accuracy': float(values['time_exact'] / values['total']) if values['total'] > 0 else 0.0,
        }
        for task, values in per_task_counts.items()
    }

    return _result_with_aliases(total_tasks, task_correct, time_exact, time_close_5m, time_close_10m, duration_close, start_mae_minutes, _count_overlaps(pred_tasks), per_task)
