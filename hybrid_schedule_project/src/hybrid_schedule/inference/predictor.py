from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hybrid_schedule.data.features import GlobalContext, SeriesBundle, build_future_history_tensor, per_task_recent_stats
from hybrid_schedule.retrieval.template_retriever import build_template_week, propose_extra_slots
from hybrid_schedule.scheduler import solve_week_schedule
from hybrid_schedule.utils import compose_week_bin, macroblock_bins_from_minutes, num_macroblocks_per_day


class HybridWeekPredictor:
    def __init__(self, context: GlobalContext, config: dict, occurrence_model, temporal_model, device: torch.device):
        self.context = context
        self.config = config
        self.occurrence_model = occurrence_model
        self.temporal_model = temporal_model
        self.device = device
        self.window_weeks = int(config['calendar']['window_weeks'])
        self.bin_minutes = int(config['calendar']['bin_minutes'])
        self.bins_per_day = (24 * 60) // self.bin_minutes
        self.topk_templates = int(config['calendar']['topk_templates'])
        self.max_delta = int(config['models']['occurrence']['max_delta'])
        self.topk_day = int(config['models']['temporal'].get('topk_day', 3))
        self.topk_macro = int(config['models']['temporal'].get('topk_macro', 4))
        self.topk_fine = int(config['models']['temporal'].get('topk_fine', 4))
        self.macroblock_minutes = int(config['models']['temporal'].get('macroblock_minutes', 60))
        self.macroblock_bins = macroblock_bins_from_minutes(self.bin_minutes, self.macroblock_minutes)
        self.num_macroblocks = num_macroblocks_per_day(self.bins_per_day, self.macroblock_bins)

    def _future_week_start(self, series: SeriesBundle) -> pd.Timestamp:
        if len(series.week_starts) == 0:
            return pd.Timestamp.now(tz=self.config['calendar']['timezone_default']).floor('D')
        if len(series.week_starts) < 2:
            return series.week_starts[-1] + pd.Timedelta(days=7)
        return series.week_starts[-1] + (series.week_starts[-1] - series.week_starts[-2])

    def _predict_counts(self, series: SeriesBundle, template) -> tuple[dict[int, int], dict[str, Any]]:
        history = build_future_history_tensor(series, self.window_weeks)
        db_idx = self.context.database_to_idx[series.database_id]
        robot_idx = self.context.robot_to_idx[f'{series.database_id}::{series.robot_id}']
        rows = []
        debug = {}
        primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
        for task_idx, task_name in enumerate(self.context.task_names):
            recent = per_task_recent_stats(series, None, task_idx, window=4)
            template_count = int(template.counts[task_idx])
            support_values = [score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx]
            support_mean = float(np.mean(support_values)) if support_values else 0.0
            numeric = np.asarray([
                float(template_count),
                recent['recent_mean'],
                recent['recent_last'],
                recent['recent_median'],
                support_mean,
                primary_score,
            ], dtype=np.float32)
            rows.append({
                'history': history,
                'task_id': task_idx,
                'database_id': db_idx,
                'robot_id': robot_idx,
                'numeric_features': numeric,
                'template_count': template_count,
            })
            debug[task_name] = {'template_count': template_count, 'recent': recent}

        batch = {
            'history': torch.tensor(np.stack([r['history'] for r in rows]), dtype=torch.float32, device=self.device),
            'task_id': torch.tensor([r['task_id'] for r in rows], dtype=torch.long, device=self.device),
            'database_id': torch.tensor([r['database_id'] for r in rows], dtype=torch.long, device=self.device),
            'robot_id': torch.tensor([r['robot_id'] for r in rows], dtype=torch.long, device=self.device),
            'numeric_features': torch.tensor(np.stack([r['numeric_features'] for r in rows]), dtype=torch.float32, device=self.device),
            'template_count': torch.tensor([r['template_count'] for r in rows], dtype=torch.long, device=self.device),
        }
        self.occurrence_model.eval()
        with torch.no_grad():
            outputs = self.occurrence_model(
                history=batch['history'],
                task_id=batch['task_id'],
                database_id=batch['database_id'],
                robot_id=batch['robot_id'],
                numeric_features=batch['numeric_features'],
                template_count=batch['template_count'], # <-- Add this line
            )
            delta_values = torch.arange(-self.max_delta, self.max_delta + 1, device=self.device)
            pred_delta = delta_values[outputs['delta_logits'].argmax(dim=-1)].cpu().numpy()
            pred_count = torch.clamp(batch['template_count'] + delta_values[outputs['delta_logits'].argmax(dim=-1)], min=0).cpu().numpy()
            change_probs = outputs['change_logits'].softmax(dim=-1)[:, 1].cpu().numpy()

        count_predictions = {}
        for idx, task_name in enumerate(self.context.task_names):
            count_predictions[idx] = int(pred_count[idx])
            debug[task_name].update({
                'pred_delta': int(pred_delta[idx]),
                'pred_count': int(pred_count[idx]),
                'change_prob': float(change_probs[idx]),
            })
        return count_predictions, debug

    def _build_planned_slots(self, series: SeriesBundle, template, count_predictions: dict[int, int]) -> list[dict[str, Any]]:
        template_by_task = defaultdict(list)
        for evt in sorted(template.events, key=lambda e: (e.task_idx, e.start_bin)):
            template_by_task[evt.task_idx].append(evt)

        planned = []
        for task_idx, pred_count in count_predictions.items():
            base_slots = template_by_task.get(task_idx, [])
            chosen_slots = []
            if pred_count <= len(base_slots):
                chosen_slots = base_slots[:pred_count]
            else:
                chosen_slots = list(base_slots)
                extras = propose_extra_slots(series, template, task_idx, None, required=pred_count - len(base_slots))
                for start_bin, duration_bins, score in extras:
                    chosen_slots.append({
                        'task_idx': task_idx,
                        'task_type': self.context.task_names[task_idx],
                        'start_bin': int(start_bin),
                        'duration_bins': int(duration_bins),
                        'support': float(score),
                    })
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

    def _temporal_candidates(self, series: SeriesBundle, planned_slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not planned_slots:
            return []
        history = build_future_history_tensor(series, self.window_weeks)
        db_idx = self.context.database_to_idx[series.database_id]
        robot_idx = self.context.robot_to_idx[f'{series.database_id}::{series.robot_id}']
        history_batch = np.repeat(history[None, ...], len(planned_slots), axis=0)
        numeric_rows = []
        task_ids = []
        for slot in planned_slots:
            anchor = int(slot['anchor_start_bin'])
            anchor_duration = int(slot['anchor_duration_bins'])
            numeric_rows.append([
                float(anchor) / (7 * self.bins_per_day),
                float(anchor // self.bins_per_day) / 6.0,
                float(anchor % self.bins_per_day) / max(1, self.bins_per_day - 1),
                float(anchor_duration) / 12.0,
                float(slot['slot_rank']) / max(1, slot['pred_task_count'] - 1 if slot['pred_task_count'] > 1 else 1),
                float(slot['template_task_count']),
                float(slot['pred_task_count']),
                float(slot['support']),
            ])
            task_ids.append(int(slot['task_idx']))

        batch = {
            'history': torch.tensor(history_batch, dtype=torch.float32, device=self.device),
            'task_id': torch.tensor(task_ids, dtype=torch.long, device=self.device),
            'database_id': torch.tensor([db_idx] * len(planned_slots), dtype=torch.long, device=self.device),
            'robot_id': torch.tensor([robot_idx] * len(planned_slots), dtype=torch.long, device=self.device),
            'numeric_features': torch.tensor(numeric_rows, dtype=torch.float32, device=self.device),
        }
        self.temporal_model.eval()
        scheduled_events = []
        with torch.no_grad():
            outputs = self.temporal_model(**batch)
            day_probs_all = outputs['day_logits'].softmax(dim=-1).cpu().numpy()
            macro_probs_all = outputs['macroblock_logits'].softmax(dim=-1).cpu().numpy()
            fine_probs_all = outputs['fine_offset_logits'].softmax(dim=-1).cpu().numpy()
            duration_delta_all = outputs['duration_delta'].cpu().numpy()
            for idx, slot in enumerate(planned_slots):
                anchor = int(slot['anchor_start_bin'])
                anchor_duration = int(slot['anchor_duration_bins'])
                day_probs = day_probs_all[idx]
                macro_probs = macro_probs_all[idx]
                fine_probs = fine_probs_all[idx]
                duration_delta = float(duration_delta_all[idx])
                day_top = np.argsort(day_probs)[::-1][: self.topk_day]
                macro_top = np.argsort(macro_probs)[::-1][: self.topk_macro]
                fine_top = np.argsort(fine_probs)[::-1][: self.topk_fine]
                candidates = []
                for day_idx in day_top:
                    for macro_idx in macro_top:
                        for fine_idx in fine_top:
                            start_bin = compose_week_bin(int(day_idx), int(macro_idx), int(fine_idx), self.bins_per_day, self.macroblock_bins)
                            start_bin = int(np.clip(start_bin, 0, 7 * self.bins_per_day - 1))
                            duration_bins = max(1, int(round(anchor_duration + duration_delta)))
                            distance = abs(start_bin - anchor)
                            score = float(np.log(day_probs[day_idx] + 1e-9) + np.log(macro_probs[macro_idx] + 1e-9) + np.log(fine_probs[fine_idx] + 1e-9))
                            score += float(self.config['scheduler']['template_bonus']) * (1.0 if start_bin == anchor else 0.0)
                            score -= float(self.config['scheduler']['movement_penalty']) * distance
                            score -= float(self.config['scheduler']['duration_penalty']) * abs(duration_bins - anchor_duration)
                            candidates.append({'start_bin': start_bin, 'duration_bins': duration_bins, 'score': score})
                candidates.append({'start_bin': anchor, 'duration_bins': anchor_duration, 'score': 0.0})
                dedup = {}
                for cand in candidates:
                    key = (cand['start_bin'], cand['duration_bins'])
                    dedup[key] = max(dedup.get(key, -1e18), cand['score'])
                final_candidates = [{'start_bin': k[0], 'duration_bins': k[1], 'score': v} for k, v in dedup.items()]
                final_candidates.sort(key=lambda x: x['score'], reverse=True)
                scheduled_events.append({
                    'robot_id': series.robot_id,
                    'task_idx': slot['task_idx'],
                    'task_type': slot['task_type'],
                    'anchor_start_bin': anchor,
                    'candidates': final_candidates[: max(5, self.topk_day * self.topk_macro * self.topk_fine)],
                })
        return scheduled_events

    def predict_series(self, series: SeriesBundle) -> tuple[list[dict], dict]:
        template = build_template_week(series, None, topk=self.topk_templates)
        count_predictions, occ_debug = self._predict_counts(series, template)
        planned_slots = self._build_planned_slots(series, template, count_predictions)
        candidate_events = self._temporal_candidates(series, planned_slots)
        solved = solve_week_schedule(
            candidate_events,
            use_exact_milp=bool(self.config['scheduler']['use_exact_milp']),
            min_gap_bins=int(self.config['scheduler']['min_gap_bins']),
            max_solver_seconds=int(self.config['scheduler']['max_solver_seconds']),
        )
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
        explanation = {
            'database_id': series.database_id,
            'robot_id': series.robot_id,
            'template_primary_week': str(series.week_starts[template.primary_week_idx]) if template.primary_week_idx is not None else "N/A",
            'template_source_weeks': [str(series.week_starts[idx]) for idx in template.source_weeks],
            'occurrence_debug': occ_debug,  # <--- Asignación directa
            'num_predicted_events': len(payload),
            'temporal_decoder': {
                'mode': 'hierarchical_day_macroblock_fine',
                'macroblock_minutes': self.macroblock_minutes,
                'topk_day': self.topk_day,
                'topk_macro': self.topk_macro,
                'topk_fine': self.topk_fine,
            },
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
