import torch
import numpy as np
from proyectos_anteriores.project.evaluation.matching import hungarian_match

BIN_MINUTES = 5


def evaluate_weekly_predictions(
    true_tasks,
    pred_tasks,
    time_tolerance_minutes=10,
    duration_tolerance_minutes=2
):
    # Convert WeekRecord to list of dicts if needed
    if hasattr(true_tasks, "events_by_task"):
        true_list = []
        for events in true_tasks.events_by_task.values():
            for e in events:
                true_list.append({
                    "task_name": e.task_name,
                    "start_bin": e.start_bin,
                    "duration": e.duration_minutes
                })
        true_tasks = true_list

    total_tasks = len(true_tasks)

    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "task_accuracy": 0.0,
            "time_exact_accuracy": 0.0,
            "time_close_accuracy": 0.0,
            "duration_close_accuracy": 0.0,
        }

    task_correct = 0
    time_exact = 0
    time_close = 0
    duration_close = 0

    true_tasks = sorted(true_tasks, key=lambda x: x["start_bin"])
    pred_tasks = sorted(pred_tasks, key=lambda x: x["start_bin"])

    pairs = hungarian_match(true_tasks, pred_tasks)

    for t, p in pairs:

        # tarea
        if t["task_name"] == p["task_name"]:
            task_correct += 1

        # tiempo
        true_bin = t["start_bin"]
        pred_bin = p["start_bin"]

        diff_minutes = abs(true_bin - pred_bin) * BIN_MINUTES

        if diff_minutes == 0:
            time_exact += 1

        if diff_minutes <= time_tolerance_minutes:
            time_close += 1

        # duración
        dur_diff = abs(t["duration"] - p["duration"])

        if dur_diff <= duration_tolerance_minutes:
            duration_close += 1

    stats = {}
    stats["total_tasks"] = total_tasks
    stats["task_accuracy"] = task_correct / total_tasks
    stats["time_exact_accuracy"] = time_exact / total_tasks
    stats["time_close_accuracy"] = time_close / total_tasks
    stats["duration_close_accuracy"] = duration_close / total_tasks

    return stats