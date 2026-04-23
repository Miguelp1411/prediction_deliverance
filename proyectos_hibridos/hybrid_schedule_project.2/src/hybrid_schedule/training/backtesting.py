from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from hybrid_schedule.data.features import GlobalContext, SeriesBundle
from hybrid_schedule.evaluation.metrics import evaluate_week
from hybrid_schedule.inference.predictor import HybridWeekPredictor



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
        mean_start_p25=series.mean_start_p25[:target_week_idx].copy(),
        mean_start_p75=series.mean_start_p75[:target_week_idx].copy(),
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



def run_holdout_backtest(context: GlobalContext, val_indices: list[tuple[str, str, int]], config: dict, occurrence_model, temporal_model, device, output_dir: str | Path) -> tuple[list[dict], dict]:
    rows = []
    for database_id, robot_id, target_week_idx in val_indices:
        series = context.series[(database_id, robot_id)]
        if target_week_idx <= 0 or target_week_idx >= len(series.week_starts):
            continue
        prefix_ctx = _prefix_context(context, database_id, robot_id, target_week_idx)
        predictor = HybridWeekPredictor(prefix_ctx, config, occurrence_model, temporal_model, device)
        pred_events, _ = predictor.predict_series(prefix_ctx.series[(database_id, robot_id)])
        true_events = []
        for evt in series.events[target_week_idx]:
            true_events.append({
                'robot_id': evt.robot_id,
                'task_type': evt.task_type,
                'start_bin': int(evt.start_bin),
                'duration_bins': int(evt.duration_bins),
            })
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
