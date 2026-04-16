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
    build_future_history_tensor,
    build_history_tensor,
    build_occurrence_numeric_features,
    build_temporal_numeric_features,
    per_task_recent_stats,
    task_slot_prototypes,
)
from hybrid_schedule.retrieval.template_retriever import TemplateWeek, build_template_week, propose_extra_slots
from hybrid_schedule.utils import day_offset_to_index, local_offset_to_index


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
                numeric, aux = build_occurrence_numeric_features(series, target_week_idx, task_idx, template_count, support_mean, primary_score)
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
        bins_per_day: int,
        day_offset_radius: int,
        local_offset_radius: int,
        max_slot_prototypes: int = 32,
        count_lookup: dict[tuple[str, str, int, int], int] | None = None,
        bin_minutes: int = 5,
    ):
        self.samples: list[dict[str, Any]] = []
        self.bins_per_day = int(bins_per_day)
        self.day_offset_radius = int(day_offset_radius)
        self.local_offset_radius = int(local_offset_radius)
        self.max_slot_prototypes = int(max_slot_prototypes)
        self.bin_minutes = int(bin_minutes)
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

            for task_idx, target_task_events in actual_by_task.items():
                target_task_events = sorted(target_task_events, key=lambda e: e.start_bin)
                predicted_count = None
                if count_lookup is not None:
                    predicted_count = count_lookup.get((database_id, robot_id, target_week_idx, task_idx))
                if predicted_count is None:
                    predicted_count = int(max(len(target_task_events), int(template.counts[task_idx])))
                predicted_count = max(int(predicted_count), len(target_task_events))
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
                    day_offset = int(round((int(target_evt.start_bin) - anchor_start) / self.bins_per_day))
                    local_offset = int((int(target_evt.start_bin) - anchor_start) - day_offset * self.bins_per_day)
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
                    )
                    self.samples.append({
                        'history': history.astype(np.float32),
                        'task_id': task_idx,
                        'database_id': db_idx,
                        'robot_id': robot_idx,
                        'numeric_features': numeric,
                        'slot_id': slot_id,
                        'baseline_pred_count': predicted_count,
                        'anchor_start': anchor_start,
                        'anchor_duration': anchor_duration,
                        'target_start': int(target_evt.start_bin),
                        'target_duration': int(target_evt.duration_bins),
                        'day_offset_target': day_offset_to_index(day_offset, self.day_offset_radius),
                        'local_offset_target': local_offset_to_index(local_offset, self.local_offset_radius),
                        'duration_delta': float(target_evt.duration_bins - anchor_duration),
                        'day_offset_radius': self.day_offset_radius,
                        'local_offset_radius': self.local_offset_radius,
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
            'slot_id': torch.tensor(s['slot_id'], dtype=torch.long),
            'baseline_pred_count': torch.tensor(s['baseline_pred_count'], dtype=torch.long),
            'anchor_start': torch.tensor(s['anchor_start'], dtype=torch.long),
            'anchor_duration': torch.tensor(s['anchor_duration'], dtype=torch.float32),
            'target_start': torch.tensor(s['target_start'], dtype=torch.long),
            'target_duration': torch.tensor(s['target_duration'], dtype=torch.float32),
            'day_offset_target': torch.tensor(s['day_offset_target'], dtype=torch.long),
            'local_offset_target': torch.tensor(s['local_offset_target'], dtype=torch.long),
            'duration_delta': torch.tensor(s['duration_delta'], dtype=torch.float32),
            'day_offset_radius': torch.tensor(s['day_offset_radius'], dtype=torch.long),
            'local_offset_radius': torch.tensor(s['local_offset_radius'], dtype=torch.long),
        }


def build_future_occurrence_features(context: GlobalContext, database_id: str, robot_id: str, template: TemplateWeek, window_weeks: int, bin_minutes: int = 5) -> list[dict[str, Any]]:
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
        numeric, aux = build_occurrence_numeric_features(series, None, task_idx, template_count, support_mean, primary_score)
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
            numeric, aux = build_occurrence_numeric_features(series, target_week_idx, task_idx, template_count, support_mean, primary_score)
            rows.append({
                'key': (database_id, robot_id, target_week_idx, task_idx),
                'history': history.astype(np.float32),
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
