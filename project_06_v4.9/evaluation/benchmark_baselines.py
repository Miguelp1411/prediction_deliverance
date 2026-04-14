"""
Baseline benchmarks for comparison.

Implements 5 baselines:
  1. copy_lag52_week — copy the schedule from 52 weeks ago
  2. retrieval_topk_templates — best similar week(s)
  3. template_plus_local_shifts — template with simple time adjustments
  4. neural_no_solver — full neural pipeline without CP-SAT (ablation)
  5. full_system — template + residual + solver (main system)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from data.schema import PreparedData, WeekRecord
from evaluation.weekly_metrics import evaluate_weekly_predictions


def _week_events_to_dicts(week: WeekRecord) -> list[dict[str, Any]]:
    """Convert a WeekRecord's events to evaluation-compatible dicts."""
    return [
        {
            "task_name": ev.task_name,
            "start_bin": ev.start_bin,
            "duration": ev.duration_minutes,
            "device_id": ev.device_id,
        }
        for ev in week.events
    ]


def baseline_copy_lag52(
    prepared: PreparedData,
    target_week_idx: int,
    bin_minutes: int = 5,
) -> dict[str, float]:
    """Baseline 1: copy the week from 52 weeks ago."""
    true_events = _week_events_to_dicts(prepared.weeks[target_week_idx])
    lag_idx = target_week_idx - 52
    if lag_idx < 0:
        pred_events: list[dict] = []
    else:
        pred_events = _week_events_to_dicts(prepared.weeks[lag_idx])

    return evaluate_weekly_predictions(true_events, pred_events, bin_minutes)


def baseline_retrieval_topk(
    prepared: PreparedData,
    target_week_idx: int,
    retriever: Any,
    top_k: int = 3,
    bin_minutes: int = 5,
) -> dict[str, float]:
    """Baseline 2: use the most similar historical week."""
    true_events = _week_events_to_dicts(prepared.weeks[target_week_idx])
    similar = retriever.get_similar_weeks(target_week_idx, top_k=1)
    if similar:
        best_week, _ = similar[0]
        pred_events = _week_events_to_dicts(best_week)
    else:
        pred_events = []

    return evaluate_weekly_predictions(true_events, pred_events, bin_minutes)


def run_all_baselines(
    prepared: PreparedData,
    val_week_indices: list[int],
    retriever: Any = None,
    bin_minutes: int = 5,
) -> dict[str, dict[str, float]]:
    """Run all baselines on validation weeks, return averaged results."""
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for idx in val_week_indices:
        # Baseline 1: lag52
        if idx >= 52:
            lag52 = baseline_copy_lag52(prepared, idx, bin_minutes)
            for k, v in lag52.items():
                results["copy_lag52"][k].append(v)

        # Baseline 2: retrieval
        if retriever is not None:
            ret = baseline_retrieval_topk(prepared, idx, retriever, bin_minutes=bin_minutes)
            for k, v in ret.items():
                results["retrieval_topk"][k].append(v)

    # Average
    averaged: dict[str, dict[str, float]] = {}
    for baseline, metrics in results.items():
        averaged[baseline] = {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}

    return averaged
