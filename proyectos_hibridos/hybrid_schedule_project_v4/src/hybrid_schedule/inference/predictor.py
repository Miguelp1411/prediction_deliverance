from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from proyectos_hibridos.hybrid_schedule_project_v4.src.hybrid_schedule.data.features import (
    GlobalContext,
    SeriesBundle,
    build_temporal_candidate_features,
    build_future_history_tensor,
    build_occurrence_numeric_features,
    build_temporal_numeric_features,
    build_temporal_slot_context,
    build_slot_plan_features,
    task_temporal_profile,
)
from proyectos_hibridos.hybrid_schedule_project_v4.src.hybrid_schedule.retrieval.template_retriever import (
    build_empirical_candidate_bank,
    build_template_week,
    gather_empirical_candidates,
    propose_extra_slots,
    build_planned_slots_from_counts,
)
from proyectos_hibridos.hybrid_schedule_project_v4.src.hybrid_schedule.scheduler import solve_week_schedule


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
        self.temporal_candidate_topk_templates = int(config['models']['temporal'].get('candidate_topk_templates', config['calendar'].get('temporal_candidate_topk_templates', max(self.topk_templates, 20))))
        self.max_delta = int(config['models']['occurrence']['max_delta'])
        self.temporal_candidate_neighbor_radius = int(config['models']['temporal'].get('candidate_neighbor_radius', 1))
        self.temporal_max_candidates = max(1, int(config['models']['temporal'].get('max_candidates', 32)))
        self.temporal_solver_candidates = max(1, int(config['models']['temporal'].get('solver_candidates', self.temporal_max_candidates)))

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
        return build_planned_slots_from_counts(series, template, self.context.task_names, count_predictions, target_week_idx=None)

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
        empirical_bank = build_empirical_candidate_bank(series, empirical_template)
        task_dispersion = {
            task_idx: float(task_temporal_profile(series, None, task_idx, bin_minutes=self.bin_minutes)['start_dispersion'])
            for task_idx in range(len(self.context.task_names))
        }
        plan_feature_map = build_slot_plan_features(planned_slots, task_dispersion_by_task=task_dispersion, bin_minutes=self.bin_minutes)
        history_batch = np.repeat(history[None, ...], len(planned_slots), axis=0)
        numeric_rows = []
        task_ids = []
        candidate_feature_rows = []
        candidate_mask_rows = []
        candidate_start_rows = []
        candidate_duration_rows = []
        candidate_support_rows = []
        slot_rows = []
        for slot in planned_slots:
            anchor = int(slot['anchor_start_bin'])
            anchor_duration = int(slot['anchor_duration_bins'])
            slot_context = build_temporal_slot_context(
                series,
                None,
                int(slot['task_idx']),
                int(slot['slot_id']),
                max_slots=self.max_slot_prototypes,
                bin_minutes=self.bin_minutes,
            )
            plan_features = plan_feature_map.get((int(slot['task_idx']), int(slot['slot_id'])), {})
            numeric = build_temporal_numeric_features(
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
                slot_context=slot_context,
                plan_features=plan_features,
            )
            candidates = gather_empirical_candidates(
                empirical_bank,
                task_idx=int(slot['task_idx']),
                slot_id=int(slot['slot_id']),
                neighbor_radius=self.temporal_candidate_neighbor_radius,
                limit=self.temporal_max_candidates,
                fallback_anchor=(anchor, anchor_duration),
            )
            candidate_features: list[np.ndarray] = []
            candidate_starts: list[int] = []
            candidate_durations: list[int] = []
            candidate_supports: list[float] = []
            for cand_start, cand_duration, cand_support in candidates[:self.temporal_max_candidates]:
                candidate_features.append(build_temporal_candidate_features(
                    series,
                    None,
                    int(slot['task_idx']),
                    int(slot['slot_id']),
                    anchor,
                    anchor_duration,
                    int(cand_start),
                    int(cand_duration),
                    float(cand_support),
                    max_slots=self.max_slot_prototypes,
                    bin_minutes=self.bin_minutes,
                    slot_context=slot_context,
                    plan_features=plan_features,
                ))
                candidate_starts.append(int(cand_start))
                candidate_durations.append(max(1, int(cand_duration)))
                candidate_supports.append(float(cand_support))
            if not candidate_features:
                continue
            feature_dim = len(candidate_features[0])
            candidate_features_arr = np.zeros((self.temporal_max_candidates, feature_dim), dtype=np.float32)
            candidate_mask_arr = np.zeros(self.temporal_max_candidates, dtype=np.bool_)
            candidate_start_arr = np.zeros(self.temporal_max_candidates, dtype=np.int64)
            candidate_duration_arr = np.ones(self.temporal_max_candidates, dtype=np.float32)
            candidate_support_arr = np.zeros(self.temporal_max_candidates, dtype=np.float32)

            candidate_count = len(candidate_features)
            candidate_features_arr[:candidate_count] = np.stack(candidate_features, axis=0)
            candidate_mask_arr[:candidate_count] = True
            candidate_start_arr[:candidate_count] = np.asarray(candidate_starts, dtype=np.int64)
            candidate_duration_arr[:candidate_count] = np.asarray(candidate_durations, dtype=np.float32)
            candidate_support_arr[:candidate_count] = np.asarray(candidate_supports, dtype=np.float32)
            candidate_feature_rows.append(candidate_features_arr)
            candidate_mask_rows.append(candidate_mask_arr)
            candidate_start_rows.append(candidate_start_arr)
            candidate_duration_rows.append(candidate_duration_arr)
            numeric_rows.append(numeric)
            task_ids.append(int(slot['task_idx']))
            candidate_support_rows.append(candidate_support_arr)
            slot_rows.append({
                'robot_id': series.robot_id,
                'task_idx': int(slot['task_idx']),
                'task_type': slot['task_type'],
                'slot_id': int(slot['slot_id']),
                'anchor_start_bin': anchor,
                'anchor_duration_bins': anchor_duration,
            })

        if not slot_rows:
            return []

        batch = {
            'history': torch.tensor(history_batch[:len(slot_rows)], dtype=torch.float32, device=self.device),
            'task_id': torch.tensor(task_ids, dtype=torch.long, device=self.device),
            'database_id': torch.tensor([db_idx] * len(slot_rows), dtype=torch.long, device=self.device),
            'robot_id': torch.tensor([robot_idx] * len(slot_rows), dtype=torch.long, device=self.device),
            'numeric_features': torch.tensor(np.stack(numeric_rows), dtype=torch.float32, device=self.device),
            'candidate_features': torch.tensor(np.stack(candidate_feature_rows), dtype=torch.float32, device=self.device),
            'candidate_mask': torch.tensor(np.stack(candidate_mask_rows), dtype=torch.bool, device=self.device),
        }
        self.temporal_model.eval()
        scheduled_events = []
        with torch.no_grad():
            outputs = self.temporal_model(**batch)
            logits = outputs['candidate_logits'].cpu().numpy()
            for idx, slot_row in enumerate(slot_rows):
                valid_idx = np.flatnonzero(candidate_mask_rows[idx])
                if valid_idx.size == 0:
                    continue
                ordered = valid_idx[np.argsort(logits[idx, valid_idx])[::-1]]
                final_candidates = [{
                    'start_bin': int(candidate_start_rows[idx][cand_idx]),
                    'duration_bins': int(round(float(candidate_duration_rows[idx][cand_idx]))),
                    'score': float(logits[idx, cand_idx]),  # compatibilidad hacia atrás
                    'model_score': float(logits[idx, cand_idx]),
                    'empirical_support': float(candidate_support_rows[idx][cand_idx]),
                    'anchor_start_bin': int(slot_row['anchor_start_bin']),
                    'anchor_duration_bins': int(slot_row['anchor_duration_bins']),
                    'slot_id': int(slot_row['slot_id']),
                } for cand_idx in ordered[:max(1, min(self.temporal_solver_candidates, len(ordered)))]]
                scheduled_events.append({
                    'robot_id': slot_row['robot_id'],
                    'task_idx': slot_row['task_idx'],
                    'task_type': slot_row['task_type'],
                    'anchor_start_bin': slot_row['anchor_start_bin'],
                    'candidates': final_candidates,
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
            scheduler_cfg=self.config['scheduler'],
            bin_minutes=self.bin_minutes,
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
                'mode': 'historical_candidate_ranker',
                'candidate_topk_templates': self.temporal_candidate_topk_templates,
                'candidate_neighbor_radius': self.temporal_candidate_neighbor_radius,
                'max_candidates': self.temporal_max_candidates,
                'solver_candidates': self.temporal_solver_candidates,
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
