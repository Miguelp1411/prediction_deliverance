from __future__ import annotations

from typing import Any

from config import bins_per_day
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
                    'start_bin': int(e.start_bin),
                    'duration': float(e.duration_minutes),
                })
        return out

    out = []
    for item in true_tasks:
        out.append({
            'task_name': str(item.get('task_name', item.get('type'))),
            'start_bin': int(item['start_bin']),
            'duration': float(item['duration']),
        })
    return out


def _normalize_pred(pred_tasks):
    out = []
    for item in pred_tasks:
        out.append({
            'task_name': _pred_label(item),
            'start_bin': int(item['start_bin']),
            'duration': float(item['duration']),
        })
    return out


def _result_with_aliases(
    total_tasks: float,
    correct_tasks: float,
    time_exact: float,
    time_5: float,
    time_10: float,
    duration_2: float,
    start_mae_minutes: float,
    day_exact: float,
    day_close_1d: float,
    start_mae_when_day_correct_minutes: float,
) -> dict[str, float]:
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
        'day_exact_accuracy': float(day_exact / total_tasks) if total_tasks > 0 else 0.0,
        'day_close_accuracy_1d': float(day_close_1d / total_tasks) if total_tasks > 0 else 0.0,
        'start_mae_when_day_correct_minutes': float(start_mae_when_day_correct_minutes),
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
        'day_exact_acc': result['day_exact_accuracy'] * 100.0,
        'day_tol_acc_1d': result['day_close_accuracy_1d'] * 100.0,
    })
    return result


def evaluate_weekly_predictions(true_tasks, pred_tasks, time_tolerance_minutes: int = 5, secondary_time_tolerance_minutes: int = 10, duration_tolerance_minutes: int = 2) -> dict[str, float]:
    true_tasks = sorted(_normalize_true(true_tasks), key=lambda x: (x['start_bin'], x['task_name']))
    pred_tasks = sorted(_normalize_pred(pred_tasks), key=lambda x: (x['start_bin'], x['task_name']))

    total_tasks = len(true_tasks)
    if total_tasks == 0:
        return _result_with_aliases(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    task_correct = 0
    time_exact = 0
    time_close_5m = 0
    time_close_10m = 0
    duration_close = 0
    day_exact = 0
    day_close_1d = 0
    start_abs_error_sum = 0.0
    start_abs_error_same_day_sum = 0.0
    same_day_pairs = 0
    bins_day = bins_per_day()

    pairs = hungarian_match(true_tasks, pred_tasks)
    for true_item, pred_item in pairs:
        if true_item['task_name'] == pred_item['task_name']:
            task_correct += 1

        true_start = int(true_item['start_bin'])
        pred_start = int(pred_item['start_bin'])
        diff_minutes = abs(true_start - pred_start) * BIN_MINUTES
        start_abs_error_sum += diff_minutes

        true_day = true_start // bins_day
        pred_day = pred_start // bins_day
        day_diff = abs(true_day - pred_day)
        if day_diff == 0:
            day_exact += 1
            same_day_pairs += 1
            start_abs_error_same_day_sum += diff_minutes
        if day_diff <= 1:
            day_close_1d += 1

        if diff_minutes == 0:
            time_exact += 1
        if diff_minutes <= time_tolerance_minutes:
            time_close_5m += 1
        if diff_minutes <= secondary_time_tolerance_minutes:
            time_close_10m += 1
        if abs(float(true_item['duration']) - float(pred_item['duration'])) <= duration_tolerance_minutes:
            duration_close += 1

    matched_pairs = max(len(pairs), 1)
    start_mae_minutes = start_abs_error_sum / matched_pairs
    start_mae_same_day = start_abs_error_same_day_sum / same_day_pairs if same_day_pairs > 0 else 0.0

    return _result_with_aliases(
        total_tasks,
        task_correct,
        time_exact,
        time_close_5m,
        time_close_10m,
        duration_close,
        start_mae_minutes,
        day_exact,
        day_close_1d,
        start_mae_same_day,
    )
