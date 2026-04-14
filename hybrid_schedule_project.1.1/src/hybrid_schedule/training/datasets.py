from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hybrid_schedule.data.features import (
    GlobalContext,
    build_history_tensor,
    normalized_task_load,
    per_task_recent_stats,
    recent_day_distribution,
    seasonal_count_baseline,
)
from hybrid_schedule.evaluation.matching import pair_template_slots_to_events
from hybrid_schedule.retrieval.template_retriever import build_template_week, propose_extra_slots


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



def build_balanced_sample_weights(indices: list[tuple[str, str, int]]) -> list[float]:
    if not indices:
        return []
    db_counts = Counter(db for db, _, _ in indices)
    robot_counts = Counter(f'{db}::{robot}' for db, robot, _ in indices)
    weights = []
    for db, robot, _ in indices:
        db_w = 1.0 / max(db_counts[db], 1)
        robot_w = 1.0 / max(robot_counts[f'{db}::{robot}'], 1)
        weights.append(float(0.5 * db_w + 0.5 * robot_w))
    total = sum(weights)
    if total <= 0:
        return [1.0] * len(indices)
    scale = len(indices) / total
    return [w * scale for w in weights]



def _occ_numeric_features(series, template, target_week_idx: int, task_idx: int) -> np.ndarray:
    recent = per_task_recent_stats(series, target_week_idx, task_idx, window=8)
    template_count = float(template.counts[task_idx])
    seasonal_count = float(seasonal_count_baseline(series, target_week_idx, task_idx))
    support_values = [score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx]
    support_mean = float(np.mean(support_values)) if support_values else 0.0
    primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
    norm_template = normalized_task_load(series, target_week_idx, task_idx, template_count)
    norm_recent = normalized_task_load(series, target_week_idx, task_idx, recent['recent_mean'])
    return np.asarray([
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
    ], dtype=np.float32)



def _temp_numeric_features(series, template, target_week_idx: int, task_idx: int, anchor_start: int, anchor_duration: int, slot_rank: int, slot_count: int, support: float) -> np.ndarray:
    recent = per_task_recent_stats(series, target_week_idx, task_idx, window=8)
    recent_day = recent_day_distribution(series, target_week_idx, task_idx, window=8)
    seasonal_count = float(seasonal_count_baseline(series, target_week_idx, task_idx))
    return np.asarray([
        float(anchor_start) / (7 * 288),
        float(anchor_start // 288) / 6.0,
        float(anchor_start % 288) / 287.0,
        float(anchor_duration) / 12.0,
        float(slot_rank) / max(1, slot_count - 1 if slot_count > 1 else 1),
        float(template.counts[task_idx]),
        recent['recent_mean'],
        seasonal_count,
        float(np.argmax(recent_day)) / 6.0,
        float(support),
    ], dtype=np.float32)


class OccurrenceDataset(Dataset):
    def __init__(
        self,
        context: GlobalContext,
        indices: list[tuple[str, str, int]],
        window_weeks: int,
        topk_templates: int,
        max_delta: int,
    ):
        self.samples: list[dict[str, Any]] = []
        self.window_weeks = int(window_weeks)
        self.max_delta = int(max_delta)
        for database_id, robot_id, target_week_idx in indices:
            series = context.series[(database_id, robot_id)]
            template = build_template_week(series, target_week_idx, topk=topk_templates)
            history = build_history_tensor(series, target_week_idx, window_weeks)
            db_idx = context.database_to_idx[database_id]
            robot_idx = context.robot_to_idx[f'{database_id}::{robot_id}']
            for task_idx in range(len(context.task_names)):
                target_count = int(series.counts[target_week_idx, task_idx])
                template_count = int(template.counts[task_idx])
                delta = int(np.clip(target_count - template_count, -self.max_delta, self.max_delta))
                self.samples.append({
                    'history': history.astype(np.float32),
                    'task_id': task_idx,
                    'database_id': db_idx,
                    'robot_id': robot_idx,
                    'numeric_features': _occ_numeric_features(series, template, target_week_idx, task_idx),
                    'template_count': template_count,
                    'target_count': target_count,
                    'change_target': int(delta != 0),
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
    ):
        self.samples: list[dict[str, Any]] = []
        self.bins_per_day = int(bins_per_day)
        for database_id, robot_id, target_week_idx in indices:
            series = context.series[(database_id, robot_id)]
            template = build_template_week(series, target_week_idx, topk=topk_templates)
            history = build_history_tensor(series, target_week_idx, window_weeks)
            db_idx = context.database_to_idx[database_id]
            robot_idx = context.robot_to_idx[f'{database_id}::{robot_id}']
            actual_events = list(series.events[target_week_idx])
            template_by_task: dict[int, list] = {}
            actual_by_task: dict[int, list] = {}
            for evt in template.events:
                template_by_task.setdefault(evt.task_idx, []).append(evt)
            for evt in actual_events:
                actual_by_task.setdefault(evt.task_idx, []).append(evt)

            for task_idx in range(len(context.task_names)):
                actual_task_events = sorted(actual_by_task.get(task_idx, []), key=lambda e: e.start_bin)
                template_slots = list(sorted(template_by_task.get(task_idx, []), key=lambda e: e.start_bin))
                needed = max(len(actual_task_events), len(template_slots))
                if needed == 0:
                    continue
                if needed > len(template_slots):
                    extras = propose_extra_slots(series, template, task_idx, target_week_idx, required=needed - len(template_slots))
                    for extra_idx, (start_bin, duration_bins, support) in enumerate(extras):
                        template_slots.append({
                            'task_idx': task_idx,
                            'task_type': context.task_names[task_idx],
                            'start_bin': int(start_bin),
                            'duration_bins': int(duration_bins),
                            'support': float(support),
                            'rank': len(template_slots) + extra_idx,
                        })
                if not actual_task_events or not template_slots:
                    continue
                matches = pair_template_slots_to_events(template_slots, actual_task_events, bins_per_day=self.bins_per_day)
                slot_count = len(template_slots)
                for slot_rank, (slot, target_evt, _cost) in enumerate(matches):
                    anchor_start = int(slot.start_bin if hasattr(slot, 'start_bin') else slot['start_bin'])
                    anchor_duration = int(slot.duration_bins if hasattr(slot, 'duration_bins') else slot['duration_bins'])
                    support = float(template.support_by_slot.get((task_idx, anchor_start), 0.0))
                    if not hasattr(slot, 'start_bin'):
                        support = float(slot.get('support', support))
                    self.samples.append({
                        'history': history.astype(np.float32),
                        'task_id': task_idx,
                        'database_id': db_idx,
                        'robot_id': robot_idx,
                        'numeric_features': _temp_numeric_features(series, template, target_week_idx, task_idx, anchor_start, anchor_duration, slot_rank, slot_count, support),
                        'anchor_start': anchor_start,
                        'anchor_duration': anchor_duration,
                        'target_start': int(target_evt.start_bin),
                        'target_duration': int(target_evt.duration_bins),
                        'day_target': int(target_evt.start_bin // self.bins_per_day),
                        'time_target': int(target_evt.start_bin % self.bins_per_day),
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
            'anchor_start': torch.tensor(s['anchor_start'], dtype=torch.long),
            'anchor_duration': torch.tensor(s['anchor_duration'], dtype=torch.long),
            'target_start': torch.tensor(s['target_start'], dtype=torch.long),
            'target_duration': torch.tensor(s['target_duration'], dtype=torch.long),
            'day_target': torch.tensor(s['day_target'], dtype=torch.long),
            'time_target': torch.tensor(s['time_target'], dtype=torch.long),
        }
