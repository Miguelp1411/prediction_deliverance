from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hybrid_schedule.data.features import (
    GlobalContext,
    SeriesBundle,
    SlotPrototype,
    assign_events_to_prototypes,
    build_temporal_candidate_features,
    build_future_history_tensor,
    build_history_tensor,
    build_occurrence_numeric_features,
    build_temporal_numeric_features,
    build_temporal_slot_context,
    build_slot_plan_features,
    per_task_recent_stats,
    task_slot_prototypes,
    task_temporal_profile,
)
from hybrid_schedule.data.scaling import transform_feature_matrix
from hybrid_schedule.retrieval.template_retriever import (
    TemplateWeek,
    build_empirical_candidate_bank,
    build_template_week,
    build_target_candidate_context,
    gather_empirical_candidates,
    propose_extra_slots,
    build_planned_slots_from_counts,
)


@dataclass
class DatasetSplit:
    train_indices: list[tuple[str, str, int]]
    val_indices: list[tuple[str, str, int]]


def build_time_split(context: GlobalContext, train_ratio: float, min_history_weeks: int) -> DatasetSplit:
    train_indices: list[tuple[str, str, int]] = []
    val_indices: list[tuple[str, str, int]] = []
    for (database_id, robot_id), series in context.series.items():
        n = len(series.week_starts)
        if n <= min_history_weeks + 2:
            continue
        split = max(min_history_weeks + 1, int(round(n * train_ratio)))
        for idx in range(min_history_weeks, split):
            train_indices.append((database_id, robot_id, idx))
        for idx in range(split, n):
            val_indices.append((database_id, robot_id, idx))
    return DatasetSplit(train_indices=train_indices, val_indices=val_indices)


class OccurrenceDataset(Dataset):
    def __init__(
        self,
        context: GlobalContext,
        indices: list[tuple[str, str, int]],
        window_weeks: int,
        topk_templates: int,
        max_delta: int,
        bin_minutes: int = 5,
        use_lag52: bool = False,
    ):
        self.samples: list[dict[str, Any]] = []
        self.window_weeks = int(window_weeks)
        self.max_delta = int(max_delta)
        self.bin_minutes = int(bin_minutes)
        for database_id, robot_id, target_week_idx in indices:
            series = context.series[(database_id, robot_id)]
            template = build_template_week(series, target_week_idx, topk=topk_templates)
            history = build_history_tensor(series, target_week_idx, window_weeks, bin_minutes=self.bin_minutes)
            db_idx = context.database_to_idx[database_id]
            robot_key = f'{database_id}::{robot_id}'
            robot_idx = context.robot_to_idx[robot_key]
            primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
            for task_idx in range(len(context.task_names)):
                support_values = [score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx]
                support_mean = float(np.mean(support_values)) if support_values else 0.0
                template_count = int(template.counts[task_idx])
                numeric, aux = build_occurrence_numeric_features(series, target_week_idx, task_idx, template_count, support_mean, primary_score, use_lag52=use_lag52)
                baseline_count = int(np.clip(round(aux['baseline_count']), 0, 999))
                target_count = int(series.counts[target_week_idx, task_idx])
                delta = int(np.clip(target_count - baseline_count, -self.max_delta, self.max_delta))
                change = int(delta != 0)
                self.samples.append({
                    'history': history.astype(np.float32),
                    'task_id': task_idx,
                    'database_id': db_idx,
                    'robot_id': robot_idx,
                    'numeric_features': numeric,
                    'baseline_count': baseline_count,
                    'template_count': template_count,
                    'target_count': target_count,
                    'change_target': change,
                    'delta_target': delta + self.max_delta,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'history': torch.tensor(s['history'], dtype=torch.float32),
            'task_id': torch.tensor(s['task_id'], dtype=torch.long),
            'database_id': torch.tensor(s['database_id'], dtype=torch.long),
            'robot_id': torch.tensor(s['robot_id'], dtype=torch.long),
            'numeric_features': torch.tensor(s['numeric_features'], dtype=torch.float32),
            'baseline_count': torch.tensor(s['baseline_count'], dtype=torch.long),
            'template_count': torch.tensor(s['template_count'], dtype=torch.long),
            'target_count': torch.tensor(s['target_count'], dtype=torch.long),
            'change_target': torch.tensor(s['change_target'], dtype=torch.long),
            'delta_target': torch.tensor(s['delta_target'], dtype=torch.long),
        }


class TemporalDataset(Dataset):
    def __init__(
        self,
        context: GlobalContext,
        indices: list[tuple[str, str, int]],
        window_weeks: int,
        topk_templates: int,
        candidate_topk_templates: int,
        candidate_neighbor_radius: int,
        max_candidates: int,
        start_temperature_bins: float,
        duration_temperature_bins: float,
        duration_cost_weight: float,
        max_slot_prototypes: int = 32,
        count_lookup: dict[tuple[str, str, int, int], int] | None = None,
        bin_minutes: int = 5,
        candidate_source_quotas: dict[str, int] | None = None,
        prototype_start_offsets: list[int] | None = None,
        prototype_duration_offsets: list[int] | None = None,
        regime_lookback_weeks: int = 8,
    ):
        self.samples: list[dict[str, Any]] = []
        self.candidate_topk_templates = int(candidate_topk_templates)
        self.candidate_neighbor_radius = int(candidate_neighbor_radius)
        self.max_candidates = max(1, int(max_candidates))
        self.start_temperature_bins = float(max(1e-3, start_temperature_bins))
        self.duration_temperature_bins = float(max(1e-3, duration_temperature_bins))
        self.duration_cost_weight = float(max(0.0, duration_cost_weight))
        self.max_slot_prototypes = int(max_slot_prototypes)
        self.bin_minutes = int(bin_minutes)
        self.candidate_source_quotas = {str(k): int(v) for k, v in (candidate_source_quotas or {}).items() if int(v) > 0}
        self.prototype_start_offsets = list(prototype_start_offsets) if prototype_start_offsets is not None else None
        self.prototype_duration_offsets = list(prototype_duration_offsets) if prototype_duration_offsets is not None else None
        self.regime_lookback_weeks = int(max(1, regime_lookback_weeks))
        for database_id, robot_id, target_week_idx in indices:
            series = context.series[(database_id, robot_id)]
            template = build_template_week(series, target_week_idx, topk=topk_templates, max_slot_prototypes=self.max_slot_prototypes)
            candidate_template = build_template_week(
                series,
                target_week_idx,
                topk=self.candidate_topk_templates,
                max_slot_prototypes=self.max_slot_prototypes,
            )
            candidate_bank = build_empirical_candidate_bank(series, candidate_template)
            history = build_history_tensor(series, target_week_idx, window_weeks, bin_minutes=self.bin_minutes)
            db_idx = context.database_to_idx[database_id]
            robot_key = f'{database_id}::{robot_id}'
            robot_idx = context.robot_to_idx[robot_key]
            actual_events = list(series.events[target_week_idx])
            actual_by_task: dict[int, list] = {}
            for evt in actual_events:
                actual_by_task.setdefault(evt.task_idx, []).append(evt)

            count_predictions: dict[int, int] = {}
            for task_idx in range(len(context.task_names)):
                predicted_count = None
                if count_lookup is not None:
                    predicted_count = count_lookup.get((database_id, robot_id, target_week_idx, task_idx))
                if predicted_count is None:
                    predicted_count = int(template.counts[task_idx])
                predicted_count = max(int(predicted_count), len(actual_by_task.get(task_idx, [])))
                count_predictions[task_idx] = int(predicted_count)
            task_dispersion = {
                task_idx: float(task_temporal_profile(series, target_week_idx, task_idx, bin_minutes=self.bin_minutes)['start_dispersion'])
                for task_idx in range(len(context.task_names))
            }
            planned_slots = build_planned_slots_from_counts(
                series,
                template,
                context.task_names,
                count_predictions,
                target_week_idx=target_week_idx,
            )
            plan_feature_map = build_slot_plan_features(planned_slots, task_dispersion_by_task=task_dispersion, bin_minutes=self.bin_minutes)

            for task_idx, target_task_events in actual_by_task.items():
                target_task_events = sorted(target_task_events, key=lambda e: e.start_bin)
                predicted_count = int(count_predictions.get(task_idx, len(target_task_events)))
                prototypes = list(template.slot_prototypes_by_task.get(task_idx) or task_slot_prototypes(series, target_week_idx, task_idx, max_slots=self.max_slot_prototypes))
                if len(prototypes) < len(target_task_events):
                    extra_slots = propose_extra_slots(
                        series,
                        template,
                        task_idx,
                        target_week_idx,
                        required=len(target_task_events) - len(prototypes),
                        used_keys={(int(proto.center_bin), int(proto.duration_bins)) for proto in prototypes},
                        start_slot_id=max((int(proto.slot_id) for proto in prototypes), default=-1) + 1,
                    )
                    prototypes.extend(
                        SlotPrototype(
                            task_idx=task_idx,
                            slot_id=int(slot_id),
                            center_bin=int(start_bin),
                            duration_bins=max(1, int(duration_bins)),
                            support=float(score),
                        )
                        for start_bin, duration_bins, score, slot_id in extra_slots
                    )
                if not prototypes:
                    continue
                assignments = assign_events_to_prototypes(target_task_events, prototypes)
                for slot_id, target_evt, proto in assignments:
                    anchor_start = int(proto.center_bin)
                    anchor_duration = int(proto.duration_bins)
                    slot_support = float(proto.support)
                    slot_context = build_temporal_slot_context(
                        series,
                        target_week_idx,
                        task_idx,
                        slot_id,
                        max_slots=self.max_slot_prototypes,
                        bin_minutes=self.bin_minutes,
                    )
                    plan_features = plan_feature_map.get((int(task_idx), int(slot_id)), {})
                    numeric = build_temporal_numeric_features(
                        series,
                        target_week_idx,
                        task_idx,
                        slot_id,
                        anchor_start,
                        anchor_duration,
                        predicted_count,
                        int(template.counts[task_idx]),
                        slot_support,
                        max_slots=self.max_slot_prototypes,
                        bin_minutes=self.bin_minutes,
                        slot_context=slot_context,
                        plan_features=plan_features,
                    )
                    target_candidate_context = build_target_candidate_context(
                        series,
                        target_week_idx,
                        planned_slots=planned_slots,
                        task_idx=task_idx,
                        slot_id=slot_id,
                        regime_lookback_weeks=self.regime_lookback_weeks,
                    )
                    empirical_candidates = gather_empirical_candidates(
                        candidate_bank,
                        task_idx=task_idx,
                        slot_id=slot_id,
                        target_context=target_candidate_context,
                        neighbor_radius=self.candidate_neighbor_radius,
                        limit=self.max_candidates,
                        fallback_anchor=(anchor_start, anchor_duration),
                        source_quotas=self.candidate_source_quotas,
                        prototype_start_offsets=self.prototype_start_offsets,
                        prototype_duration_offsets=self.prototype_duration_offsets,
                    )
                    candidate_features = []
                    candidate_starts = []
                    candidate_durations = []
                    candidate_costs = []
                    candidate_target_scores = []
                    for cand in empirical_candidates[:self.max_candidates]:
                        cand_start = int(cand['start_bin'])
                        cand_duration = max(1, int(cand['duration_bins']))
                        cand_support = float(cand.get('support', 0.0))
                        candidate_features.append(build_temporal_candidate_features(
                            series,
                            target_week_idx,
                            task_idx,
                            slot_id,
                            anchor_start,
                            anchor_duration,
                            cand_start,
                            cand_duration,
                            cand_support,
                            max_slots=self.max_slot_prototypes,
                            bin_minutes=self.bin_minutes,
                            slot_context=slot_context,
                            plan_features=plan_features,
                            candidate_meta=cand,
                        ))
                        candidate_starts.append(cand_start)
                        candidate_durations.append(cand_duration)
                        start_gap = abs(cand_start - int(target_evt.start_bin)) / self.start_temperature_bins
                        duration_gap = abs(cand_duration - int(target_evt.duration_bins)) / self.duration_temperature_bins
                        cost = float(start_gap + self.duration_cost_weight * duration_gap)
                        candidate_costs.append(cost)
                        candidate_target_scores.append(-cost)
                    if not candidate_features:
                        continue
                    candidate_count = len(candidate_features)
                    candidate_features_arr = np.zeros((self.max_candidates, len(candidate_features[0])), dtype=np.float32)
                    candidate_starts_arr = np.zeros(self.max_candidates, dtype=np.int64)
                    candidate_durations_arr = np.ones(self.max_candidates, dtype=np.float32)
                    candidate_cost_arr = np.full(self.max_candidates, 16.0, dtype=np.float32)
                    candidate_mask_arr = np.zeros(self.max_candidates, dtype=np.bool_)
                    target_probs_arr = np.zeros(self.max_candidates, dtype=np.float32)
                    candidate_features_arr[:candidate_count] = np.stack(candidate_features, axis=0)
                    candidate_starts_arr[:candidate_count] = np.asarray(candidate_starts, dtype=np.int64)
                    candidate_durations_arr[:candidate_count] = np.asarray(candidate_durations, dtype=np.float32)
                    candidate_cost_arr[:candidate_count] = np.asarray(candidate_costs, dtype=np.float32)
                    candidate_mask_arr[:candidate_count] = True
                    target_logits = np.asarray(candidate_target_scores, dtype=np.float32)
                    target_logits = target_logits - float(target_logits.max())
                    target_weights = np.exp(target_logits)
                    target_probs_arr[:candidate_count] = target_weights / max(float(target_weights.sum()), 1e-8)
                    self.samples.append({
                        'history': history.astype(np.float32),
                        'task_id': task_idx,
                        'database_id': db_idx,
                        'robot_id': robot_idx,
                        'numeric_features': numeric,
                        'candidate_features': candidate_features_arr,
                        'candidate_starts': candidate_starts_arr,
                        'candidate_durations': candidate_durations_arr,
                        'candidate_costs': candidate_cost_arr,
                        'candidate_mask': candidate_mask_arr,
                        'candidate_target_probs': target_probs_arr,
                        'slot_id': slot_id,
                        'baseline_pred_count': predicted_count,
                        'anchor_start': anchor_start,
                        'anchor_duration': anchor_duration,
                        'target_start': int(target_evt.start_bin),
                        'target_duration': int(target_evt.duration_bins),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'history': torch.tensor(s['history'], dtype=torch.float32),
            'task_id': torch.tensor(s['task_id'], dtype=torch.long),
            'database_id': torch.tensor(s['database_id'], dtype=torch.long),
            'robot_id': torch.tensor(s['robot_id'], dtype=torch.long),
            'numeric_features': torch.tensor(s['numeric_features'], dtype=torch.float32),
            'candidate_features': torch.tensor(s['candidate_features'], dtype=torch.float32),
            'candidate_starts': torch.tensor(s['candidate_starts'], dtype=torch.long),
            'candidate_durations': torch.tensor(s['candidate_durations'], dtype=torch.float32),
            'candidate_costs': torch.tensor(s['candidate_costs'], dtype=torch.float32),
            'candidate_mask': torch.tensor(s['candidate_mask'], dtype=torch.bool),
            'candidate_target_probs': torch.tensor(s['candidate_target_probs'], dtype=torch.float32),
            'slot_id': torch.tensor(s['slot_id'], dtype=torch.long),
            'baseline_pred_count': torch.tensor(s['baseline_pred_count'], dtype=torch.long),
            'anchor_start': torch.tensor(s['anchor_start'], dtype=torch.long),
            'anchor_duration': torch.tensor(s['anchor_duration'], dtype=torch.float32),
            'target_start': torch.tensor(s['target_start'], dtype=torch.long),
            'target_duration': torch.tensor(s['target_duration'], dtype=torch.float32),
        }




class DirectTemporalDataset(Dataset):
    def __init__(
        self,
        context: GlobalContext,
        indices: list[tuple[str, str, int]],
        window_weeks: int,
        topk_templates: int,
        max_slot_prototypes: int = 32,
        count_lookup: dict[tuple[str, str, int, int], int] | None = None,
        bin_minutes: int = 5,
    ):
        self.samples: list[dict[str, Any]] = []
        self.max_slot_prototypes = int(max_slot_prototypes)
        self.bin_minutes = int(bin_minutes)
        self.bins_per_day = int(24 * 60 / self.bin_minutes)
        for database_id, robot_id, target_week_idx in indices:
            series = context.series[(database_id, robot_id)]
            template = build_template_week(series, target_week_idx, topk=topk_templates, max_slot_prototypes=self.max_slot_prototypes)
            history = build_history_tensor(series, target_week_idx, window_weeks, bin_minutes=self.bin_minutes)
            db_idx = context.database_to_idx[database_id]
            robot_key = f'{database_id}::{robot_id}'
            robot_idx = context.robot_to_idx[robot_key]
            actual_events = list(series.events[target_week_idx])
            actual_by_task: dict[int, list] = {}
            for evt in actual_events:
                actual_by_task.setdefault(evt.task_idx, []).append(evt)

            count_predictions: dict[int, int] = {}
            for task_idx in range(len(context.task_names)):
                predicted_count = None
                if count_lookup is not None:
                    predicted_count = count_lookup.get((database_id, robot_id, target_week_idx, task_idx))
                if predicted_count is None:
                    predicted_count = int(template.counts[task_idx])
                predicted_count = max(int(predicted_count), len(actual_by_task.get(task_idx, [])))
                count_predictions[task_idx] = int(predicted_count)

            task_dispersion = {
                task_idx: float(task_temporal_profile(series, target_week_idx, task_idx, bin_minutes=self.bin_minutes)['start_dispersion'])
                for task_idx in range(len(context.task_names))
            }
            planned_slots = build_planned_slots_from_counts(
                series,
                template,
                context.task_names,
                count_predictions,
                target_week_idx=target_week_idx,
            )
            plan_feature_map = build_slot_plan_features(planned_slots, task_dispersion_by_task=task_dispersion, bin_minutes=self.bin_minutes)

            for task_idx, target_task_events in actual_by_task.items():
                target_task_events = sorted(target_task_events, key=lambda e: e.start_bin)
                predicted_count = int(count_predictions.get(task_idx, len(target_task_events)))
                prototypes = list(template.slot_prototypes_by_task.get(task_idx) or task_slot_prototypes(series, target_week_idx, task_idx, max_slots=self.max_slot_prototypes))
                if len(prototypes) < len(target_task_events):
                    extra_slots = propose_extra_slots(
                        series,
                        template,
                        task_idx,
                        target_week_idx,
                        required=len(target_task_events) - len(prototypes),
                        used_keys={(int(proto.center_bin), int(proto.duration_bins)) for proto in prototypes},
                        start_slot_id=max((int(proto.slot_id) for proto in prototypes), default=-1) + 1,
                    )
                    prototypes.extend(
                        SlotPrototype(
                            task_idx=task_idx,
                            slot_id=int(slot_id),
                            center_bin=int(start_bin),
                            duration_bins=max(1, int(duration_bins)),
                            support=float(score),
                        )
                        for start_bin, duration_bins, score, slot_id in extra_slots
                    )
                if not prototypes:
                    continue
                assignments = assign_events_to_prototypes(target_task_events, prototypes)
                for slot_id, target_evt, proto in assignments:
                    anchor_start = int(proto.center_bin)
                    anchor_duration = int(proto.duration_bins)
                    slot_support = float(proto.support)
                    slot_context = build_temporal_slot_context(
                        series,
                        target_week_idx,
                        task_idx,
                        slot_id,
                        max_slots=self.max_slot_prototypes,
                        bin_minutes=self.bin_minutes,
                    )
                    plan_features = plan_feature_map.get((int(task_idx), int(slot_id)), {})
                    numeric = build_temporal_numeric_features(
                        series,
                        target_week_idx,
                        task_idx,
                        slot_id,
                        anchor_start,
                        anchor_duration,
                        predicted_count,
                        int(template.counts[task_idx]),
                        slot_support,
                        max_slots=self.max_slot_prototypes,
                        bin_minutes=self.bin_minutes,
                        slot_context=slot_context,
                        plan_features=plan_features,
                    )
                    target_start = int(target_evt.start_bin)
                    target_duration = max(1, int(target_evt.duration_bins))
                    self.samples.append({
                        'history': history.astype(np.float32),
                        'task_id': int(task_idx),
                        'database_id': int(db_idx),
                        'robot_id': int(robot_idx),
                        'slot_id': int(slot_id),
                        'anchor_day': int(anchor_start // self.bins_per_day),
                        'anchor_time': int(anchor_start % self.bins_per_day),
                        'numeric_features': numeric.astype(np.float32),
                        'baseline_pred_count': int(predicted_count),
                        'anchor_start': int(anchor_start),
                        'anchor_duration': int(anchor_duration),
                        'target_start': int(target_start),
                        'target_duration': int(target_duration),
                        'target_day_idx': int(target_start // self.bins_per_day),
                        'target_time_bin_idx': int(target_start % self.bins_per_day),
                        'target_log_duration': float(np.log1p(target_duration)),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'history': torch.tensor(s['history'], dtype=torch.float32),
            'task_id': torch.tensor(s['task_id'], dtype=torch.long),
            'database_id': torch.tensor(s['database_id'], dtype=torch.long),
            'robot_id': torch.tensor(s['robot_id'], dtype=torch.long),
            'slot_id': torch.tensor(s['slot_id'], dtype=torch.long),
            'anchor_day': torch.tensor(s['anchor_day'], dtype=torch.long),
            'anchor_time': torch.tensor(s['anchor_time'], dtype=torch.long),
            'numeric_features': torch.tensor(s['numeric_features'], dtype=torch.float32),
            'baseline_pred_count': torch.tensor(s['baseline_pred_count'], dtype=torch.long),
            'anchor_start': torch.tensor(s['anchor_start'], dtype=torch.long),
            'anchor_duration': torch.tensor(s['anchor_duration'], dtype=torch.float32),
            'target_start': torch.tensor(s['target_start'], dtype=torch.long),
            'target_duration': torch.tensor(s['target_duration'], dtype=torch.float32),
            'target_day_idx': torch.tensor(s['target_day_idx'], dtype=torch.long),
            'target_time_bin_idx': torch.tensor(s['target_time_bin_idx'], dtype=torch.long),
            'target_log_duration': torch.tensor(s['target_log_duration'], dtype=torch.float32),
        }

def build_future_occurrence_features(context: GlobalContext, database_id: str, robot_id: str, template: TemplateWeek, window_weeks: int, bin_minutes: int = 5, use_lag52: bool = False) -> list[dict[str, Any]]:
    series = context.series[(database_id, robot_id)]
    history = build_future_history_tensor(series, window_weeks, bin_minutes=bin_minutes)
    db_idx = context.database_to_idx[database_id]
    robot_idx = context.robot_to_idx[f'{database_id}::{robot_id}']
    primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
    rows: list[dict[str, Any]] = []
    for task_idx in range(len(context.task_names)):
        support_values = [score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx]
        support_mean = float(np.mean(support_values)) if support_values else 0.0
        template_count = int(template.counts[task_idx])
        numeric, aux = build_occurrence_numeric_features(series, None, task_idx, template_count, support_mean, primary_score, use_lag52=use_lag52)
        rows.append({
            'history': history.astype(np.float32),
            'task_id': task_idx,
            'database_id': db_idx,
            'robot_id': robot_idx,
            'numeric_features': numeric,
            'baseline_count': int(np.clip(round(aux['baseline_count']), 0, 999)),
            'template_count': template_count,
        })
    return rows


def predict_occurrence_counts_for_indices(
    context: GlobalContext,
    indices: list[tuple[str, str, int]],
    occurrence_model,
    device: torch.device,
    window_weeks: int,
    topk_templates: int,
    bin_minutes: int = 5,
    batch_size: int = 256,
    feature_scaling: dict[str, Any] | None = None,
    use_lag52: bool = False,
) -> dict[tuple[str, str, int, int], int]:
    rows = []
    for database_id, robot_id, target_week_idx in indices:
        series = context.series[(database_id, robot_id)]
        template = build_template_week(series, target_week_idx, topk=topk_templates)
        history = build_history_tensor(series, target_week_idx, window_weeks, bin_minutes=bin_minutes)
        db_idx = context.database_to_idx[database_id]
        robot_idx = context.robot_to_idx[f'{database_id}::{robot_id}']
        primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
        for task_idx in range(len(context.task_names)):
            support_values = [score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx]
            support_mean = float(np.mean(support_values)) if support_values else 0.0
            template_count = int(template.counts[task_idx])
            numeric, aux = build_occurrence_numeric_features(series, target_week_idx, task_idx, template_count, support_mean, primary_score, use_lag52=use_lag52)
            history_features = history.astype(np.float32)
            if feature_scaling and feature_scaling.get('history') is not None:
                history_features = transform_feature_matrix(history_features, feature_scaling.get('history'))
            if feature_scaling and feature_scaling.get('occurrence_numeric') is not None:
                numeric = transform_feature_matrix(numeric, feature_scaling.get('occurrence_numeric'))
            rows.append({
                'key': (database_id, robot_id, target_week_idx, task_idx),
                'history': history_features,
                'task_id': task_idx,
                'database_id': db_idx,
                'robot_id': robot_idx,
                'numeric_features': numeric,
                'baseline_count': int(np.clip(round(aux['baseline_count']), 0, 999)),
            })
    out: dict[tuple[str, str, int, int], int] = {}
    if not rows:
        return out
    occurrence_model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), max(1, batch_size)):
            chunk = rows[start:start+batch_size]
            batch = {
                'history': torch.tensor(np.stack([r['history'] for r in chunk]), dtype=torch.float32, device=device),
                'task_id': torch.tensor([r['task_id'] for r in chunk], dtype=torch.long, device=device),
                'database_id': torch.tensor([r['database_id'] for r in chunk], dtype=torch.long, device=device),
                'robot_id': torch.tensor([r['robot_id'] for r in chunk], dtype=torch.long, device=device),
                'numeric_features': torch.tensor(np.stack([r['numeric_features'] for r in chunk]), dtype=torch.float32, device=device),
                'baseline_count': torch.tensor([r['baseline_count'] for r in chunk], dtype=torch.long, device=device),
            }
            outputs = occurrence_model(**batch)
            preds = outputs['pred_count'].detach().cpu().numpy().astype(int)
            for row, pred in zip(chunk, preds):
                out[row['key']] = max(0, int(pred))
    return out
