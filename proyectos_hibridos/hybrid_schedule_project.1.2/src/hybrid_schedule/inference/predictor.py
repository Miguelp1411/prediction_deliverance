from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hybrid_schedule.data.features import (
    GlobalContext,
    SeriesBundle,
    build_future_history_tensor,
    deterministic_task_duration,
    normalized_task_load,
    per_task_recent_stats,
    recent_day_distribution,
    recent_time_distribution,
    seasonal_count_baseline,
)
from hybrid_schedule.retrieval.template_retriever import build_template_week, propose_extra_slots
from hybrid_schedule.scheduler.solver import solve_week_schedule


class HybridWeekPredictor:
    def __init__(self, context: GlobalContext, config: dict, occurrence_model, temporal_model, device: torch.device):
        self.context = context
        self.config = config
        self.occurrence_model = occurrence_model
        self.temporal_model = temporal_model
        self.device = device
        self.window_weeks = int(config['calendar']['window_weeks'])
        self.topk_templates = int(config['calendar']['topk_templates'])
        self.bin_minutes = int(config['calendar']['bin_minutes'])
        self.bins_per_day = (24 * 60) // self.bin_minutes
        self.max_delta = int(config['models']['occurrence']['max_delta'])
        self.topk_day = int(config['models']['temporal']['topk_day'])
        self.topk_time = int(config['models']['temporal']['topk_time'])

    def _future_week_start(self, series: SeriesBundle) -> pd.Timestamp:
        return series.week_starts[-1] + pd.Timedelta(days=7)

    def _occ_numeric_features(self, series: SeriesBundle, template, task_idx: int) -> list[float]:
        target_week_idx = len(series.week_starts)
        recent = per_task_recent_stats(series, target_week_idx, task_idx, window=8)
        template_count = float(template.counts[task_idx])
        seasonal_count = float(seasonal_count_baseline(series, target_week_idx, task_idx))
        support_values = [score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx]
        support_mean = float(np.mean(support_values)) if support_values else 0.0
        primary_score = float(template.week_scores[0] if template.week_scores else 0.0)
        norm_template = normalized_task_load(series, target_week_idx, task_idx, template_count)
        norm_recent = normalized_task_load(series, target_week_idx, task_idx, recent['recent_mean'])
        return [
            template_count,
            recent['recent_mean'],
            recent['recent_last'],
            recent['recent_median'],
            seasonal_count,
            recent['recent_std'],
            support_mean,
            primary_score,
            norm_template,
            norm_recent,
        ]

    def _count_interval_from_probs(self, template_count: int, probs: np.ndarray, alpha: float) -> tuple[int, int]:
        delta_values = np.arange(-self.max_delta, self.max_delta + 1)
        cdf = np.cumsum(probs)
        lo_idx = int(np.searchsorted(cdf, alpha / 2.0, side='left'))
        hi_idx = int(np.searchsorted(cdf, 1.0 - alpha / 2.0, side='left'))
        lo = max(0, int(round(template_count + delta_values[min(max(lo_idx, 0), len(delta_values) - 1)])))
        hi = max(lo, int(round(template_count + delta_values[min(max(hi_idx, 0), len(delta_values) - 1)])))
        return lo, hi

    def _predict_counts(self, series: SeriesBundle, template) -> tuple[dict[int, int], dict[int, dict[str, float]]]:
        history = build_future_history_tensor(series, self.window_weeks)
        db_idx = self.context.database_to_idx[series.database_id]
        robot_idx = self.context.robot_to_idx[f'{series.database_id}::{series.robot_id}']
        task_ids = list(range(len(self.context.task_names)))
        numeric_rows = [self._occ_numeric_features(series, template, task_idx) for task_idx in task_ids]
        template_counts = [int(template.counts[task_idx]) for task_idx in task_ids]
        batch = {
            'history': torch.tensor(np.repeat(history[None, ...], len(task_ids), axis=0), dtype=torch.float32, device=self.device),
            'task_id': torch.tensor(task_ids, dtype=torch.long, device=self.device),
            'database_id': torch.tensor([db_idx] * len(task_ids), dtype=torch.long, device=self.device),
            'robot_id': torch.tensor([robot_idx] * len(task_ids), dtype=torch.long, device=self.device),
            'numeric_features': torch.tensor(numeric_rows, dtype=torch.float32, device=self.device),
            'template_count': torch.tensor(template_counts, dtype=torch.long, device=self.device),
        }
        predictions: dict[int, int] = {}
        debug: dict[int, dict[str, float]] = {}
        weights_cfg = self.config['models']['occurrence'].get('ensemble_weights', {})
        w_template = float(weights_cfg.get('template', 0.20))
        w_recent = float(weights_cfg.get('recent', 0.20))
        w_seasonal = float(weights_cfg.get('seasonal', 0.20))
        w_neural = float(weights_cfg.get('neural', 0.40))
        alpha = float(self.config.get('uncertainty', {}).get('count_alpha', 0.10))

        self.occurrence_model.eval()
        with torch.no_grad():
            outputs = self.occurrence_model(**batch)
            delta_probs = outputs['delta_probs'].cpu().numpy()
            change_probs = outputs['change_logits'].softmax(dim=-1).cpu().numpy()
            expected_neural_counts = outputs['pred_count'].cpu().numpy()
            for idx, task_idx in enumerate(task_ids):
                recent = per_task_recent_stats(series, len(series.week_starts), task_idx, window=8)
                seasonal = float(seasonal_count_baseline(series, len(series.week_starts), task_idx))
                ensemble_count = (
                    w_template * template_counts[idx]
                    + w_recent * recent['recent_mean']
                    + w_seasonal * seasonal
                    + w_neural * float(expected_neural_counts[idx])
                )
                pred_count = max(0, int(round(ensemble_count)))
                low_count, high_count = self._count_interval_from_probs(template_counts[idx], delta_probs[idx], alpha=alpha)
                predictions[task_idx] = pred_count
                debug[task_idx] = {
                    'template_count': int(template_counts[idx]),
                    'recent_mean': float(recent['recent_mean']),
                    'seasonal_count': float(seasonal),
                    'neural_expected_count': float(expected_neural_counts[idx]),
                    'pred_count': int(pred_count),
                    'count_low': int(low_count),
                    'count_high': int(max(high_count, pred_count)),
                    'changed_prob': float(change_probs[idx, 1]),
                }
        return predictions, debug

    def _build_planned_slots(self, series: SeriesBundle, template, count_predictions: dict[int, int]) -> list[dict[str, Any]]:
        template_by_task: dict[int, list] = defaultdict(list)
        for evt in sorted(template.events, key=lambda e: (e.task_idx, e.start_bin)):
            template_by_task[evt.task_idx].append(evt)

        planned = []
        for task_idx, pred_count in count_predictions.items():
            base_slots = list(template_by_task.get(task_idx, []))
            base_slots.sort(key=lambda e: (template.support_by_slot.get((task_idx, e.start_bin), 0.0), -e.start_bin), reverse=True)
            chosen_slots = base_slots[:pred_count]
            if pred_count > len(chosen_slots):
                extras = propose_extra_slots(series, template, task_idx, None, required=pred_count - len(chosen_slots))
                for start_bin, duration_bins, score in extras:
                    chosen_slots.append({
                        'task_idx': task_idx,
                        'task_type': self.context.task_names[task_idx],
                        'start_bin': int(start_bin),
                        'duration_bins': int(duration_bins),
                        'support': float(score),
                    })
            chosen_slots = sorted(chosen_slots, key=lambda x: x.start_bin if hasattr(x, 'start_bin') else x['start_bin'])
            for slot_rank, slot in enumerate(chosen_slots):
                if hasattr(slot, 'start_bin'):
                    start_bin = int(slot.start_bin)
                    duration_bins = int(slot.duration_bins)
                    task_type = slot.task_type
                    support = float(template.support_by_slot.get((task_idx, start_bin), 0.0))
                else:
                    start_bin = int(slot['start_bin'])
                    duration_bins = int(slot['duration_bins'])
                    task_type = slot['task_type']
                    support = float(slot.get('support', 0.0))
                planned.append({
                    'robot_id': series.robot_id,
                    'task_idx': task_idx,
                    'task_type': task_type,
                    'slot_rank': slot_rank,
                    'anchor_start_bin': start_bin,
                    'anchor_duration_bins': duration_bins,
                    'support': support,
                    'template_task_count': int(template.counts[task_idx]),
                    'pred_task_count': int(pred_count),
                })
        planned.sort(key=lambda x: (x['anchor_start_bin'], x['task_type']))
        return planned

    def _recent_time_prior(self, series: SeriesBundle, task_idx: int) -> np.ndarray:
        return recent_time_distribution(
            series,
            len(series.week_starts),
            task_idx,
            bins_per_day=self.bins_per_day,
            max_weeks=16,
        )

    def _temporal_feature_row(self, series: SeriesBundle, template, slot: dict[str, Any]) -> list[float]:
        task_idx = int(slot['task_idx'])
        target_week_idx = len(series.week_starts)
        recent = per_task_recent_stats(series, target_week_idx, task_idx, window=8)
        recent_day = recent_day_distribution(series, target_week_idx, task_idx, window=8)
        return [
            float(slot['anchor_start_bin']) / (7 * self.bins_per_day),
            float(slot['anchor_start_bin'] // self.bins_per_day) / 6.0,
            float(slot['anchor_start_bin'] % self.bins_per_day) / max(1, self.bins_per_day - 1),
            float(slot['anchor_duration_bins']) / 12.0,
            float(slot['slot_rank']) / max(1, slot['pred_task_count'] - 1 if slot['pred_task_count'] > 1 else 1),
            float(slot['template_task_count']),
            float(recent['recent_mean']),
            float(seasonal_count_baseline(series, target_week_idx, task_idx)),
            float(np.argmax(recent_day)) / 6.0,
            float(slot['support']),
        ]

    def _temporal_candidates(self, series: SeriesBundle, template, planned_slots: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not planned_slots:
            return [], []
        if self.temporal_model is None:
            fallback = []
            event_debug = []
            for slot in planned_slots:
                duration_bins = deterministic_task_duration(series, len(series.week_starts), int(slot['task_idx']), fallback=int(slot['anchor_duration_bins']))
                fallback.append({
                    'robot_id': series.robot_id,
                    'task_idx': slot['task_idx'],
                    'task_type': slot['task_type'],
                    'anchor_start_bin': int(slot['anchor_start_bin']),
                    'candidates': [{'start_bin': int(slot['anchor_start_bin']), 'duration_bins': int(duration_bins), 'score': float(slot['support'])}],
                })
                event_debug.append({'task_type': slot['task_type'], 'anchor_start_bin': int(slot['anchor_start_bin']), 'confidence': 0.0})
            return fallback, event_debug

        history = build_future_history_tensor(series, self.window_weeks)
        db_idx = self.context.database_to_idx[series.database_id]
        robot_idx = self.context.robot_to_idx[f'{series.database_id}::{series.robot_id}']
        history_batch = np.repeat(history[None, ...], len(planned_slots), axis=0)
        numeric_rows = [self._temporal_feature_row(series, template, slot) for slot in planned_slots]
        task_ids = [int(slot['task_idx']) for slot in planned_slots]
        batch = {
            'history': torch.tensor(history_batch, dtype=torch.float32, device=self.device),
            'task_id': torch.tensor(task_ids, dtype=torch.long, device=self.device),
            'database_id': torch.tensor([db_idx] * len(planned_slots), dtype=torch.long, device=self.device),
            'robot_id': torch.tensor([robot_idx] * len(planned_slots), dtype=torch.long, device=self.device),
            'numeric_features': torch.tensor(numeric_rows, dtype=torch.float32, device=self.device),
        }
        self.temporal_model.eval()
        scheduled_events: list[dict[str, Any]] = []
        event_debug: list[dict[str, Any]] = []
        day_blend = float(self.config['models']['temporal'].get('day_prior_blend', 0.25))
        time_blend = float(self.config['models']['temporal'].get('time_prior_blend', 0.20))
        with torch.no_grad():
            outputs = self.temporal_model(**batch)
            day_probs_all = outputs['day_logits'].softmax(dim=-1).cpu().numpy()
            time_probs_all = outputs['time_logits'].softmax(dim=-1).cpu().numpy()
            for idx, slot in enumerate(planned_slots):
                task_idx = int(slot['task_idx'])
                day_prior = recent_day_distribution(series, len(series.week_starts), task_idx, window=8)
                time_prior = self._recent_time_prior(series, task_idx)
                day_probs = (1.0 - day_blend) * day_probs_all[idx] + day_blend * day_prior
                time_probs = (1.0 - time_blend) * time_probs_all[idx] + time_blend * time_prior
                day_probs = day_probs / max(day_probs.sum(), 1e-8)
                time_probs = time_probs / max(time_probs.sum(), 1e-8)
                duration_bins = deterministic_task_duration(series, len(series.week_starts), task_idx, fallback=int(slot['anchor_duration_bins']))
                day_top = np.argsort(day_probs)[::-1][: self.topk_day]
                time_top = np.argsort(time_probs)[::-1][: self.topk_time]
                candidates = []
                anchor = int(slot['anchor_start_bin'])
                for day_idx in day_top:
                    for time_idx in time_top:
                        start_bin = int(day_idx * self.bins_per_day + time_idx)
                        distance = abs(start_bin - anchor)
                        score = float(np.log(day_probs[day_idx] + 1e-9) + np.log(time_probs[time_idx] + 1e-9))
                        score += 0.35 * float(slot['support'])
                        score += float(self.config['scheduler']['template_bonus']) * (1.0 if start_bin == anchor else 0.0)
                        score -= float(self.config['scheduler']['movement_penalty']) * (distance / max(1, self.bins_per_day))
                        candidates.append({'start_bin': start_bin, 'duration_bins': int(duration_bins), 'score': score})
                candidates.append({'start_bin': anchor, 'duration_bins': int(duration_bins), 'score': float(slot['support'])})
                dedup: dict[tuple[int, int], float] = {}
                for cand in candidates:
                    key = (cand['start_bin'], cand['duration_bins'])
                    dedup[key] = max(dedup.get(key, -1e18), cand['score'])
                final_candidates = [{'start_bin': k[0], 'duration_bins': k[1], 'score': v} for k, v in dedup.items()]
                final_candidates.sort(key=lambda x: x['score'], reverse=True)
                confidence = float(final_candidates[0]['score'] - final_candidates[1]['score']) if len(final_candidates) > 1 else float(final_candidates[0]['score'])
                scheduled_events.append({
                    'robot_id': series.robot_id,
                    'task_idx': slot['task_idx'],
                    'task_type': slot['task_type'],
                    'anchor_start_bin': anchor,
                    'candidates': final_candidates[: max(6, self.topk_day * self.topk_time)],
                })
                event_debug.append({
                    'task_type': slot['task_type'],
                    'anchor_start_bin': anchor,
                    'confidence': confidence,
                    'top_day_prob': float(day_probs[day_top[0]]),
                    'top_time_prob': float(time_probs[time_top[0]]),
                })
        return scheduled_events, event_debug

    def _build_payload(self, series: SeriesBundle, solved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        week_start = self._future_week_start(series)
        payload = []
        for evt in solved:
            start_time = week_start + pd.Timedelta(minutes=int(evt['start_bin'] * self.bin_minutes))
            end_time = start_time + pd.Timedelta(minutes=int(evt['duration_bins'] * self.bin_minutes))
            payload.append({
                'database_id': series.database_id,
                'robot_id': series.robot_id,
                'task_type': evt['task_type'],
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'start_bin': int(evt['start_bin']),
                'duration_bins': int(evt['duration_bins']),
            })
        payload.sort(key=lambda x: (x['robot_id'], x['start_time'], x['task_type']))
        return payload

    def _alternative_schedules(self, candidate_events: list[dict[str, Any]], topk: int) -> list[list[dict[str, Any]]]:
        if topk <= 1 or not candidate_events:
            return []
        schedules: list[list[dict[str, Any]]] = []
        base = solve_week_schedule(
            candidate_events,
            use_exact_milp=bool(self.config['scheduler']['use_exact_milp']),
            min_gap_bins=int(self.config['scheduler']['min_gap_bins']),
            max_solver_seconds=int(self.config['scheduler']['max_solver_seconds']),
        )
        schedules.append(base)
        for alt_idx in range(1, topk):
            modified = copy.deepcopy(candidate_events)
            event_to_perturb = min(alt_idx - 1, len(modified) - 1)
            if modified[event_to_perturb]['candidates']:
                modified[event_to_perturb]['candidates'][0]['score'] -= 1.25 + 0.25 * alt_idx
            alt = solve_week_schedule(
                modified,
                use_exact_milp=bool(self.config['scheduler']['use_exact_milp']),
                min_gap_bins=int(self.config['scheduler']['min_gap_bins']),
                max_solver_seconds=int(self.config['scheduler']['max_solver_seconds']),
            )
            schedules.append(alt)
        return schedules[1:]

    def predict_series(self, series: SeriesBundle) -> tuple[list[dict], dict]:
        template = build_template_week(series, None, topk=self.topk_templates)
        count_predictions, occ_debug = self._predict_counts(series, template)
        planned_slots = self._build_planned_slots(series, template, count_predictions)
        candidate_events, temporal_debug = self._temporal_candidates(series, template, planned_slots)
        solved = solve_week_schedule(
            candidate_events,
            use_exact_milp=bool(self.config['scheduler']['use_exact_milp']),
            min_gap_bins=int(self.config['scheduler']['min_gap_bins']),
            max_solver_seconds=int(self.config['scheduler']['max_solver_seconds']),
        )
        payload = self._build_payload(series, solved)
        topk_schedules = int(self.config.get('scheduler', {}).get('topk_schedules', 1))
        alternatives = [self._build_payload(series, alt) for alt in self._alternative_schedules(candidate_events, topk_schedules)]
        explanation = {
            'database_id': series.database_id,
            'robot_id': series.robot_id,
            'template_primary_week': str(series.week_starts[template.primary_week_idx]),
            'template_source_weeks': [str(series.week_starts[idx]) for idx in template.source_weeks],
            'occurrence_debug': {self.context.task_names[k]: v for k, v in occ_debug.items()},
            'temporal_debug': temporal_debug,
            'num_predicted_events': len(payload),
            'alternative_schedules': alternatives,
        }
        return payload, explanation

    def predict_database(self, database_id: str) -> tuple[list[dict], dict]:
        all_events = []
        explanations = {}
        for (db_id, _robot_id), series in self.context.series.items():
            if db_id != database_id:
                continue
            pred, expl = self.predict_series(series)
            all_events.extend(pred)
            explanations[series.robot_id] = expl
        all_events.sort(key=lambda x: (x['robot_id'], x['start_time'], x['task_type']))
        return all_events, explanations

    def save_prediction(self, database_id: str, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        events, explanation = self.predict_database(database_id)
        (output_dir / 'predicted_week.json').write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf-8')
        (output_dir / 'prediction_explanation.json').write_text(json.dumps(explanation, ensure_ascii=False, indent=2), encoding='utf-8')
