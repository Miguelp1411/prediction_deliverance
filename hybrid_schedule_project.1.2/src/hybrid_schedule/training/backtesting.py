from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from hybrid_schedule.data.features import GlobalContext, SeriesBundle
from hybrid_schedule.evaluation.metrics import evaluate_week
from hybrid_schedule.inference.predictor import HybridWeekPredictor
from hybrid_schedule.retrieval.template_retriever import build_template_week



def _prefix_series(series: SeriesBundle, target_week_idx: int) -> SeriesBundle:
    return SeriesBundle(
        database_id=series.database_id,
        robot_id=series.robot_id,
        task_names=series.task_names,
        task_to_idx=series.task_to_idx,
        week_starts=list(series.week_starts[:target_week_idx]),
        counts=series.counts[:target_week_idx].copy(),
        day_hist=series.day_hist[:target_week_idx].copy(),
        mean_start=series.mean_start[:target_week_idx].copy(),
        mean_duration=series.mean_duration[:target_week_idx].copy(),
        events=list(series.events[:target_week_idx]),
    )



def _prefix_context(context: GlobalContext, database_id: str, robot_id: str, target_week_idx: int) -> GlobalContext:
    series_map = dict(context.series)
    series_map[(database_id, robot_id)] = _prefix_series(context.series[(database_id, robot_id)], target_week_idx)
    return GlobalContext(
        task_names=context.task_names,
        task_to_idx=context.task_to_idx,
        database_to_idx=context.database_to_idx,
        robot_to_idx=context.robot_to_idx,
        series=series_map,
    )



def _count_metrics(pred_counts: dict[int, int], true_counts: dict[int, int]) -> dict[str, float]:
    errors = [abs(float(pred_counts.get(k, 0)) - float(v)) for k, v in true_counts.items()]
    exact = [float(pred_counts.get(k, 0) == v) for k, v in true_counts.items()]
    close1 = [float(abs(float(pred_counts.get(k, 0)) - float(v)) <= 1) for k, v in true_counts.items()]
    close2 = [float(abs(float(pred_counts.get(k, 0)) - float(v)) <= 2) for k, v in true_counts.items()]
    mae = sum(errors) / max(len(errors), 1)
    return {
        'count_exact_acc': 100.0 * sum(exact) / max(len(exact), 1),
        'close_acc_1': 100.0 * sum(close1) / max(len(close1), 1),
        'close_acc_2': 100.0 * sum(close2) / max(len(close2), 1),
        'count_mae': mae,
        'selector_score': 0.60 * (100.0 * sum(close2) / max(len(close2), 1)) + 0.40 * (100.0 * sum(exact) / max(len(exact), 1)) - 10.0 * mae,
    }



def run_occurrence_selector_backtest(context: GlobalContext, selector_indices: list[tuple[str, str, int]], config: dict, occurrence_model, device) -> dict[str, float]:
    rows = []
    for database_id, robot_id, target_week_idx in selector_indices:
        series = context.series[(database_id, robot_id)]
        if target_week_idx <= 0 or target_week_idx >= len(series.week_starts):
            continue
        prefix_ctx = _prefix_context(context, database_id, robot_id, target_week_idx)
        prefix_series = prefix_ctx.series[(database_id, robot_id)]
        fast_config = copy.deepcopy(config)
        fast_config.setdefault('scheduler', {})['use_exact_milp'] = False
        fast_config['scheduler']['topk_schedules'] = 1
        predictor = HybridWeekPredictor(prefix_ctx, fast_config, occurrence_model, temporal_model=None, device=device)
        template = build_template_week(prefix_series, None, topk=int(config['calendar']['topk_templates']))
        pred_counts, _ = predictor._predict_counts(prefix_series, template)  # noqa: SLF001
        true_counts = {task_idx: int(series.counts[target_week_idx, task_idx]) for task_idx in range(len(context.task_names))}
        rows.append(_count_metrics(pred_counts, true_counts))
    if not rows:
        return {'selector_score': -1e9}
    df = pd.DataFrame(rows)
    return df.mean(numeric_only=True).to_dict()



def run_schedule_selector_backtest(context: GlobalContext, selector_indices: list[tuple[str, str, int]], config: dict, occurrence_model, temporal_model, device) -> dict[str, float]:
    rows = []
    for database_id, robot_id, target_week_idx in selector_indices:
        series = context.series[(database_id, robot_id)]
        if target_week_idx <= 0 or target_week_idx >= len(series.week_starts):
            continue
        prefix_ctx = _prefix_context(context, database_id, robot_id, target_week_idx)
        fast_config = copy.deepcopy(config)
        fast_config.setdefault('scheduler', {})['use_exact_milp'] = False
        fast_config['scheduler']['topk_schedules'] = 1
        predictor = HybridWeekPredictor(prefix_ctx, fast_config, occurrence_model, temporal_model, device)
        pred_events, _ = predictor.predict_series(prefix_ctx.series[(database_id, robot_id)])
        true_events = [
            {
                'robot_id': evt.robot_id,
                'task_type': evt.task_type,
                'start_bin': int(evt.start_bin),
                'duration_bins': int(evt.duration_bins),
            }
            for evt in series.events[target_week_idx]
        ]
        pred_events_simple = [
            {
                'robot_id': e['robot_id'],
                'task_type': e['task_type'],
                'start_bin': int(e['start_bin']),
                'duration_bins': int(e['duration_bins']),
            }
            for e in pred_events
        ]
        metrics = evaluate_week(pred_events_simple, true_events, bin_minutes=int(config['calendar']['bin_minutes']))
        metrics['selector_score'] = metrics['task_f1'] + 0.35 * metrics['time_close_accuracy_10m'] - 0.02 * metrics['start_mae_minutes']
        rows.append(metrics)
    if not rows:
        return {'selector_score': -1e9}
    df = pd.DataFrame(rows)
    return df.mean(numeric_only=True).to_dict()



def run_holdout_backtest(context: GlobalContext, val_indices: list[tuple[str, str, int]], config: dict, occurrence_model, temporal_model, device, output_dir: str | Path) -> tuple[list[dict], dict]:
    rows = []
    for database_id, robot_id, target_week_idx in val_indices:
        series = context.series[(database_id, robot_id)]
        if target_week_idx <= 0 or target_week_idx >= len(series.week_starts):
            continue
        prefix_ctx = _prefix_context(context, database_id, robot_id, target_week_idx)
        predictor = HybridWeekPredictor(prefix_ctx, config, occurrence_model, temporal_model, device)
        pred_events, explanation = predictor.predict_series(prefix_ctx.series[(database_id, robot_id)])
        true_events = [
            {
                'robot_id': evt.robot_id,
                'task_type': evt.task_type,
                'start_bin': int(evt.start_bin),
                'duration_bins': int(evt.duration_bins),
            }
            for evt in series.events[target_week_idx]
        ]
        pred_events_simple = [
            {
                'robot_id': e['robot_id'],
                'task_type': e['task_type'],
                'start_bin': int(e['start_bin']),
                'duration_bins': int(e['duration_bins']),
            }
            for e in pred_events
        ]
        metrics = evaluate_week(pred_events_simple, true_events, bin_minutes=int(config['calendar']['bin_minutes']))
        metrics['database_id'] = database_id
        metrics['robot_id'] = robot_id
        metrics['target_week'] = str(series.week_starts[target_week_idx])
        metrics['num_alternative_schedules'] = len(explanation.get('alternative_schedules', []))
        rows.append(metrics)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / 'backtest_weekly_metrics.csv', index=False)
        summary = df.select_dtypes(include=['number']).mean().to_dict()
    else:
        summary = {}
    return rows, summary



def run_leave_one_database_out_backtest(context: GlobalContext, config: dict, occurrence_model, temporal_model, device, output_dir: str | Path) -> dict[str, dict[str, float]]:
    by_db: dict[str, dict[str, float]] = {}
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_rows = []
    weeks = int(config['splits'].get('backtest_weeks', 12))
    for database_id in sorted(context.database_to_idx.keys()):
        db_indices: list[tuple[str, str, int]] = []
        for (db_id, robot_id), series in context.series.items():
            if db_id != database_id:
                continue
            start = max(int(config['calendar']['min_history_weeks']), len(series.week_starts) - weeks)
            for idx in range(start, len(series.week_starts)):
                db_indices.append((db_id, robot_id, idx))
        _, summary = run_holdout_backtest(context, db_indices, config, occurrence_model, temporal_model, device, out_dir / f'leave_one_db_{database_id}')
        by_db[database_id] = summary
        row = {'database_id': database_id}
        row.update(summary)
        db_rows.append(row)
    if db_rows:
        pd.DataFrame(db_rows).to_csv(out_dir / 'leave_one_database_out_metrics.csv', index=False)
    return by_db
