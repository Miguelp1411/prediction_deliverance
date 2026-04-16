from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from hybrid_schedule.data.features import (
    GlobalContext,
    SeriesBundle,
    assign_events_to_prototypes,
    build_future_history_tensor,
    build_occurrence_numeric_features,
    build_temporal_numeric_features,
    task_duration_median,
)
from hybrid_schedule.retrieval.template_retriever import build_template_week, propose_extra_slots
from hybrid_schedule.scheduler import solve_week_schedule
from hybrid_schedule.utils import index_to_day_offset, index_to_local_offset


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
        self.max_slot_prototypes = int(config['calendar'].get('max_slot_prototypes', 32))
        self.temporal_candidate_topk_templates = int(config['calendar'].get('temporal_candidate_topk_templates', max(self.topk_templates, 20)))
        self.max_delta = int(config['models']['occurrence']['max_delta'])
        self.topk_day = int(config['models']['temporal'].get('topk_day', 3))
        self.topk_local = int(config['models']['temporal'].get('topk_local', 6))
        self.day_offset_radius = int(config['models']['temporal'].get('day_offset_radius', 3))
        self.local_offset_radius = int(config['models']['temporal'].get('local_offset_radius', 72))
        self.duration_blend_alpha = float(config['models']['temporal'].get('duration_blend_alpha', 0.30))
        self.empirical_slot_neighbor_radius = int(config['scheduler'].get('empirical_slot_neighbor_radius', 1))
        self.empirical_candidate_limit = int(config['scheduler'].get('empirical_candidate_limit', 24))
        self.empirical_support_weight = float(config['scheduler'].get('empirical_support_weight', 1.25))
        self.empirical_model_weight = float(config['scheduler'].get('empirical_model_weight', 0.75))
        self.empirical_out_of_range_penalty = float(config['scheduler'].get('empirical_out_of_range_penalty', -0.75))

    def _future_week_start(self, series: SeriesBundle) -> pd.Timestamp:
        if len(series.week_starts) == 0:
            return pd.Timestamp.now(tz=self.config['calendar']['timezone_default']).floor('D')
        if len(series.week_starts) < 2:
            return series.week_starts[-1] + pd.Timedelta(days=7)
        return series.week_starts[-1] + (series.week_starts[-1] - series.week_starts[-2])

    def _predict_counts(self, series: SeriesBundle, template) -> tuple[dict[int, int], dict[str, Any]]:
        history = build_future_history_tensor(series, self.window_weeks, bin_minutes=self.bin_minutes)
        db_idx = self.context.database_to_idx[series.database_id]
        robot_idx = self.context.robot_to_idx[f'{series.database_id}::{series.robot_id}']
        rows = []
        debug = {}
        primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
        for task_idx, task_name in enumerate(self.context.task_names):
            template_count = int(template.counts[task_idx])
            support_values = [score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx]
            support_mean = float(np.mean(support_values)) if support_values else 0.0
            numeric, aux = build_occurrence_numeric_features(series, None, task_idx, template_count, support_mean, primary_score)
            baseline_count = int(np.clip(round(aux['baseline_count']), 0, 999))
            rows.append({
                'history': history,
                'task_id': task_idx,
                'database_id': db_idx,
                'robot_id': robot_idx,
                'numeric_features': numeric,
                'baseline_count': baseline_count,
                'template_count': template_count,
            })
            debug[task_name] = {'template_count': template_count, 'baseline_count': baseline_count, 'feature_summary': aux}

        batch = {
            'history': torch.tensor(np.stack([r['history'] for r in rows]), dtype=torch.float32, device=self.device),
            'task_id': torch.tensor([r['task_id'] for r in rows], dtype=torch.long, device=self.device),
            'database_id': torch.tensor([r['database_id'] for r in rows], dtype=torch.long, device=self.device),
            'robot_id': torch.tensor([r['robot_id'] for r in rows], dtype=torch.long, device=self.device),
            'numeric_features': torch.tensor(np.stack([r['numeric_features'] for r in rows]), dtype=torch.float32, device=self.device),
            'baseline_count': torch.tensor([r['baseline_count'] for r in rows], dtype=torch.long, device=self.device),
        }
        self.occurrence_model.eval()
        with torch.no_grad():
            outputs = self.occurrence_model(**batch)
            pred_count = outputs['pred_count'].cpu().numpy()
            expected_count = outputs['expected_count'].cpu().numpy()
            change_probs = outputs['change_logits'].softmax(dim=-1)[:, 1].cpu().numpy()

        count_predictions = {}
        for idx, task_name in enumerate(self.context.task_names):
            count_predictions[idx] = int(pred_count[idx])
            debug[task_name].update({
                'pred_count': int(pred_count[idx]),
                'expected_count': float(expected_count[idx]),
                'change_prob': float(change_probs[idx]),
            })
        return count_predictions, debug

    def _build_planned_slots(self, series: SeriesBundle, template, count_predictions: dict[int, int]) -> list[dict[str, Any]]:
        planned = []
        for task_idx, pred_count in count_predictions.items():
            prototypes = sorted(template.slot_prototypes_by_task.get(task_idx, []), key=lambda p: (p.center_bin, -p.support))
            chosen = []
            used_keys: set[tuple[int, int]] = set()
            next_slot_id = max((int(proto.slot_id) for proto in prototypes), default=-1) + 1
            for proto in prototypes[:max(0, pred_count)]:
                chosen.append({
                    'task_idx': task_idx,
                    'task_type': self.context.task_names[task_idx],
                    'slot_id': int(proto.slot_id),
                    'anchor_start_bin': int(proto.center_bin),
                    'anchor_duration_bins': int(proto.duration_bins),
                    'support': float(proto.support),
                    'template_task_count': int(template.counts[task_idx]),
                    'pred_task_count': int(pred_count),
                })
                used_keys.add((int(proto.center_bin), int(proto.duration_bins)))
            if pred_count > len(chosen):
                extras = propose_extra_slots(
                    series,
                    template,
                    task_idx,
                    None,
                    required=pred_count - len(chosen),
                    used_keys=used_keys,
                    start_slot_id=next_slot_id,
                )
                for start_bin, duration_bins, score, slot_id in extras:
                    chosen.append({
                        'task_idx': task_idx,
                        'task_type': self.context.task_names[task_idx],
                        'slot_id': int(slot_id),
                        'anchor_start_bin': int(start_bin),
                        'anchor_duration_bins': int(duration_bins),
                        'support': float(score),
                        'template_task_count': int(template.counts[task_idx]),
                        'pred_task_count': int(pred_count),
                    })
                    used_keys.add((int(start_bin), int(duration_bins)))
            planned.extend(chosen[:pred_count])
        planned.sort(key=lambda x: (x['anchor_start_bin'], x['task_type'], x['slot_id']))
        return planned

    def _build_empirical_candidate_bank(self, series: SeriesBundle, template) -> dict[int, dict[str, dict]]:
        week_weights = np.exp(np.asarray(template.week_scores, dtype=np.float64))
        bank: dict[int, dict[str, dict]] = {}
        for task_idx in range(len(self.context.task_names)):
            prototypes = template.slot_prototypes_by_task.get(task_idx, [])
            slot_bank: dict[int, dict[tuple[int, int], float]] = {}
            task_bank: dict[tuple[int, int], float] = {}
            if not prototypes:
                bank[task_idx] = {'slot': slot_bank, 'task': task_bank}
                continue
            for weight, week_idx in zip(week_weights.tolist(), template.source_weeks):
                task_events = [evt for evt in series.events[week_idx] if evt.task_idx == task_idx]
                if not task_events:
                    continue
                assignments = assign_events_to_prototypes(task_events, prototypes)
                for slot_id, evt, _ in assignments:
                    key = (int(evt.start_bin), int(evt.duration_bins))
                    slot_bank.setdefault(int(slot_id), {})
                    slot_bank[int(slot_id)][key] = slot_bank[int(slot_id)].get(key, 0.0) + float(weight)
                    task_bank[key] = task_bank.get(key, 0.0) + float(weight)
            bank[task_idx] = {'slot': slot_bank, 'task': task_bank}
        return bank

    def _score_empirical_candidate(
        self,
        anchor: int,
        anchor_duration: int,
        start_bin: int,
        duration_bins: int,
        support_score: float,
        day_probs: np.ndarray,
        local_probs: np.ndarray,
    ) -> float:
        diff = int(start_bin) - int(anchor)
        day_offset = int(round(diff / self.bins_per_day))
        local_offset = int(diff - day_offset * self.bins_per_day)
        distance = abs(diff)
        score = self.empirical_support_weight * float(support_score)
        if abs(day_offset) <= self.day_offset_radius and abs(local_offset) <= self.local_offset_radius:
            day_idx = day_offset + self.day_offset_radius
            local_idx = local_offset + self.local_offset_radius
            score += self.empirical_model_weight * float(np.log(day_probs[day_idx] + 1e-9) + np.log(local_probs[local_idx] + 1e-9))
        else:
            score += self.empirical_out_of_range_penalty
        score += float(self.config['scheduler']['template_bonus']) * (1.0 if int(start_bin) == int(anchor) else 0.0)
        score -= float(self.config['scheduler']['movement_penalty']) * distance
        score -= float(self.config['scheduler']['duration_penalty']) * abs(int(duration_bins) - int(anchor_duration))
        return score

    def _temporal_candidates(self, series: SeriesBundle, planned_slots: list[dict[str, Any]], empirical_template=None) -> list[dict[str, Any]]:
        if not planned_slots:
            return []
        history = build_future_history_tensor(series, self.window_weeks, bin_minutes=self.bin_minutes)
        db_idx = self.context.database_to_idx[series.database_id]
        robot_idx = self.context.robot_to_idx[f'{series.database_id}::{series.robot_id}']
        empirical_template = empirical_template or build_template_week(
            series,
            None,
            topk=self.temporal_candidate_topk_templates,
            max_slot_prototypes=self.max_slot_prototypes,
        )
        empirical_bank = self._build_empirical_candidate_bank(series, empirical_template)
        history_batch = np.repeat(history[None, ...], len(planned_slots), axis=0)
        numeric_rows = []
        task_ids = []
        anchor_starts = []
        anchor_durations = []
        for slot in planned_slots:
            anchor = int(slot['anchor_start_bin'])
            anchor_duration = int(slot['anchor_duration_bins'])
            numeric_rows.append(build_temporal_numeric_features(
                series,
                None,
                int(slot['task_idx']),
                int(slot['slot_id']),
                anchor,
                anchor_duration,
                int(slot['pred_task_count']),
                int(slot['template_task_count']),
                float(slot['support']),
                max_slots=self.max_slot_prototypes,
                bin_minutes=self.bin_minutes,
            ))
            task_ids.append(int(slot['task_idx']))
            anchor_starts.append(anchor)
            anchor_durations.append(anchor_duration)

        batch = {
            'history': torch.tensor(history_batch, dtype=torch.float32, device=self.device),
            'task_id': torch.tensor(task_ids, dtype=torch.long, device=self.device),
            'database_id': torch.tensor([db_idx] * len(planned_slots), dtype=torch.long, device=self.device),
            'robot_id': torch.tensor([robot_idx] * len(planned_slots), dtype=torch.long, device=self.device),
            'numeric_features': torch.tensor(np.stack(numeric_rows), dtype=torch.float32, device=self.device),
        }
        self.temporal_model.eval()
        scheduled_events = []
        with torch.no_grad():
            outputs = self.temporal_model(**batch)
            day_probs_all = outputs['day_offset_logits'].softmax(dim=-1).cpu().numpy()
            local_probs_all = outputs['local_offset_logits'].softmax(dim=-1).cpu().numpy()
            duration_delta_all = outputs['duration_delta'].cpu().numpy()
            for idx, slot in enumerate(planned_slots):
                anchor = anchor_starts[idx]
                anchor_duration = anchor_durations[idx]
                day_probs = day_probs_all[idx]
                local_probs = local_probs_all[idx]
                duration_delta = float(duration_delta_all[idx])
                day_top = np.argsort(day_probs)[::-1][: self.topk_day]
                local_top = np.argsort(local_probs)[::-1][: self.topk_local]
                duration_med = task_duration_median(series, None, int(slot['task_idx']))
                candidates = []
                for day_idx in day_top:
                    for local_idx in local_top:
                        day_offset = index_to_day_offset(int(day_idx), self.day_offset_radius)
                        local_offset = index_to_local_offset(int(local_idx), self.local_offset_radius)
                        start_bin = int(np.clip(anchor + day_offset * self.bins_per_day + local_offset, 0, 7 * self.bins_per_day - 1))
                        raw_duration = max(1.0, anchor_duration + duration_delta)
                        duration_bins = max(1, int(round((1.0 - self.duration_blend_alpha) * raw_duration + self.duration_blend_alpha * duration_med)))
                        distance = abs(start_bin - anchor)
                        score = float(np.log(day_probs[day_idx] + 1e-9) + np.log(local_probs[local_idx] + 1e-9))
                        score += float(self.config['scheduler']['template_bonus']) * (1.0 if start_bin == anchor else 0.0)
                        score -= float(self.config['scheduler']['movement_penalty']) * distance
                        score -= float(self.config['scheduler']['duration_penalty']) * abs(duration_bins - anchor_duration)
                        candidates.append({'start_bin': start_bin, 'duration_bins': duration_bins, 'score': score})
                anchor_duration_pred = max(1, int(round((1.0 - self.duration_blend_alpha) * max(1.0, anchor_duration + duration_delta) + self.duration_blend_alpha * duration_med)))
                anchor_score = float(np.log(day_probs[self.day_offset_radius] + 1e-9) + np.log(local_probs[self.local_offset_radius] + 1e-9))
                anchor_score += float(self.config['scheduler']['template_bonus'])
                anchor_score -= float(self.config['scheduler']['duration_penalty']) * abs(anchor_duration_pred - anchor_duration)
                candidates.append({'start_bin': anchor, 'duration_bins': anchor_duration_pred, 'score': anchor_score})

                empirical_slot_bank = empirical_bank.get(int(slot['task_idx']), {'slot': {}, 'task': {}})
                empirical_candidates: dict[tuple[int, int], float] = {}
                for neighbor_slot in range(int(slot['slot_id']) - self.empirical_slot_neighbor_radius, int(slot['slot_id']) + self.empirical_slot_neighbor_radius + 1):
                    for key, support_score in empirical_slot_bank.get('slot', {}).get(int(neighbor_slot), {}).items():
                        empirical_candidates[key] = max(empirical_candidates.get(key, 0.0), float(support_score))
                if not empirical_candidates:
                    empirical_candidates = dict(empirical_slot_bank.get('task', {}))
                for (start_bin, duration_bins), support_score in sorted(
                    empirical_candidates.items(),
                    key=lambda item: (-item[1], item[0][0], item[0][1]),
                )[: self.empirical_candidate_limit]:
                    score = self._score_empirical_candidate(
                        anchor=anchor,
                        anchor_duration=anchor_duration,
                        start_bin=int(start_bin),
                        duration_bins=int(duration_bins),
                        support_score=float(support_score),
                        day_probs=day_probs,
                        local_probs=local_probs,
                    )
                    candidates.append({'start_bin': int(start_bin), 'duration_bins': int(duration_bins), 'score': score})
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
                    'candidates': final_candidates[: max(5, self.topk_day * self.topk_local)],
                })
        return scheduled_events

    def predict_series(self, series: SeriesBundle) -> tuple[list[dict], dict]:
        template = build_template_week(series, None, topk=self.topk_templates, max_slot_prototypes=self.max_slot_prototypes)
        empirical_template = build_template_week(
            series,
            None,
            topk=max(self.topk_templates, self.temporal_candidate_topk_templates),
            max_slot_prototypes=self.max_slot_prototypes,
        )
        count_predictions, occ_debug = self._predict_counts(series, template)
        planned_slots = self._build_planned_slots(series, template, count_predictions)
        candidate_events = self._temporal_candidates(series, planned_slots, empirical_template=empirical_template)
        solved = solve_week_schedule(
            candidate_events,
            use_exact_milp=bool(self.config['scheduler']['use_exact_milp']),
            min_gap_bins=int(self.config['scheduler']['min_gap_bins']),
            max_exact_events=int(self.config['scheduler'].get('max_exact_events', 128)),
            max_exact_variables=int(self.config['scheduler'].get('max_exact_variables', 2500)),
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
            'template_primary_week': str(series.week_starts[template.primary_week_idx]) if template.primary_week_idx is not None else 'N/A',
            'template_source_weeks': [str(series.week_starts[idx]) for idx in template.source_weeks],
            'occurrence_debug': occ_debug,
            'num_predicted_events': len(payload),
            'temporal_decoder': {
                'mode': 'relative_day_and_local_offsets_plus_empirical_bank',
                'day_offset_radius': self.day_offset_radius,
                'local_offset_radius': self.local_offset_radius,
                'topk_day': self.topk_day,
                'topk_local': self.topk_local,
                'duration_blend_alpha': self.duration_blend_alpha,
                'empirical_candidate_topk_templates': max(self.topk_templates, self.temporal_candidate_topk_templates),
                'empirical_slot_neighbor_radius': self.empirical_slot_neighbor_radius,
                'empirical_candidate_limit': self.empirical_candidate_limit,
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
