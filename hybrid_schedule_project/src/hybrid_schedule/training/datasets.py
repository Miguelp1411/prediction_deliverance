from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hybrid_schedule.data.features import GlobalContext, SeriesBundle, build_future_history_tensor, build_history_tensor, per_task_recent_stats
from hybrid_schedule.retrieval.template_retriever import TemplateWeek, build_template_week, propose_extra_slots
from hybrid_schedule.utils import split_week_bin


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
    ):
        self.samples: list[dict[str, Any]] = []
        self.window_weeks = int(window_weeks)
        self.max_delta = int(max_delta)
        for database_id, robot_id, target_week_idx in indices:
            series = context.series[(database_id, robot_id)]
            template = build_template_week(series, target_week_idx, topk=topk_templates)
            history = build_history_tensor(series, target_week_idx, window_weeks)
            db_idx = context.database_to_idx[database_id]
            robot_key = f'{database_id}::{robot_id}'
            robot_idx = context.robot_to_idx[robot_key]
            primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
            for task_idx in range(len(context.task_names)):
                recent = per_task_recent_stats(series, target_week_idx, task_idx, window=4)
                template_count = int(template.counts[task_idx])
                target_count = int(series.counts[target_week_idx, task_idx])
                delta = int(np.clip(target_count - template_count, -self.max_delta, self.max_delta))
                change = int(delta != 0)
                support_values = [
                    score for (tidx, _), score in template.support_by_slot.items() if tidx == task_idx
                ]
                support_mean = float(np.mean(support_values)) if support_values else 0.0
                numeric = np.asarray([
                    float(template_count),
                    recent['recent_mean'],
                    recent['recent_last'],
                    recent['recent_median'],
                    support_mean,
                    primary_score,
                ], dtype=np.float32)
                self.samples.append({
                    'history': history.astype(np.float32),
                    'task_id': task_idx,
                    'database_id': db_idx,
                    'robot_id': robot_idx,
                    'numeric_features': numeric,
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
        macroblock_bins: int,
    ):
        self.samples: list[dict[str, Any]] = []
        self.bins_per_day = int(bins_per_day)
        self.macroblock_bins = int(macroblock_bins)
        for database_id, robot_id, target_week_idx in indices:
            series = context.series[(database_id, robot_id)]
            template = build_template_week(series, target_week_idx, topk=topk_templates)
            history = build_history_tensor(series, target_week_idx, window_weeks)
            db_idx = context.database_to_idx[database_id]
            robot_key = f'{database_id}::{robot_id}'
            robot_idx = context.robot_to_idx[robot_key]
            actual_events = list(series.events[target_week_idx])
            template_by_task: dict[int, list] = {}
            for evt in template.events:
                template_by_task.setdefault(evt.task_idx, []).append(evt)
            actual_by_task: dict[int, list] = {}
            for evt in actual_events:
                actual_by_task.setdefault(evt.task_idx, []).append(evt)
            for task_idx, target_task_events in actual_by_task.items():
                temp_events = sorted(template_by_task.get(task_idx, []), key=lambda e: e.start_bin)
                target_task_events = sorted(target_task_events, key=lambda e: e.start_bin)
                extras = propose_extra_slots(series, template, task_idx, target_week_idx, required=max(0, len(target_task_events) - len(temp_events)))
                for slot_rank, target_evt in enumerate(target_task_events):
                    if slot_rank < len(temp_events):
                        anchor_start = int(temp_events[slot_rank].start_bin)
                        anchor_duration = int(temp_events[slot_rank].duration_bins)
                        support = float(template.support_by_slot.get((task_idx, anchor_start), 0.0))
                    else:
                        extra_idx = slot_rank - len(temp_events)
                        anchor_start, anchor_duration, support = extras[min(extra_idx, len(extras) - 1)]

                    target_day, target_macro, target_fine = split_week_bin(int(target_evt.start_bin), self.bins_per_day, self.macroblock_bins)
                    numeric = np.asarray([
                        float(anchor_start) / (7 * self.bins_per_day),
                        float(anchor_start // self.bins_per_day) / 6.0,
                        float(anchor_start % self.bins_per_day) / max(1, self.bins_per_day - 1),
                        float(anchor_duration) / 12.0,
                        float(slot_rank) / max(1, len(target_task_events) - 1 if len(target_task_events) > 1 else 1),
                        float(template.counts[task_idx]),
                        float(series.counts[target_week_idx, task_idx]),
                        float(support),
                    ], dtype=np.float32)
                    self.samples.append({
                        'history': history.astype(np.float32),
                        'task_id': task_idx,
                        'database_id': db_idx,
                        'robot_id': robot_idx,
                        'numeric_features': numeric,
                        'anchor_start': anchor_start,
                        'anchor_duration': anchor_duration,
                        'target_start': int(target_evt.start_bin),
                        'target_duration': int(target_evt.duration_bins),
                        'day_target': target_day,
                        'macroblock_target': target_macro,
                        'fine_offset_target': target_fine,
                        'duration_delta': float(target_evt.duration_bins - anchor_duration),
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
            'anchor_duration': torch.tensor(s['anchor_duration'], dtype=torch.float32),
            'target_start': torch.tensor(s['target_start'], dtype=torch.long),
            'target_duration': torch.tensor(s['target_duration'], dtype=torch.float32),
            'day_target': torch.tensor(s['day_target'], dtype=torch.long),
            'macroblock_target': torch.tensor(s['macroblock_target'], dtype=torch.long),
            'fine_offset_target': torch.tensor(s['fine_offset_target'], dtype=torch.long),
            'duration_delta': torch.tensor(s['duration_delta'], dtype=torch.float32),
        }



def build_future_occurrence_features(context: GlobalContext, database_id: str, robot_id: str, template: TemplateWeek, window_weeks: int) -> list[dict[str, Any]]:
    series = context.series[(database_id, robot_id)]
    history = build_future_history_tensor(series, window_weeks)
    db_idx = context.database_to_idx[database_id]
    robot_idx = context.robot_to_idx[f'{database_id}::{robot_id}']
    primary_score = float(template.week_scores[0]) if template.week_scores else 0.0
    rows: list[dict[str, Any]] = []
    for task_idx in range(len(context.task_names)):
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
            'history': history.astype(np.float32),
            'task_id': task_idx,
            'database_id': db_idx,
            'robot_id': robot_idx,
            'numeric_features': numeric,
            'template_count': template_count,
        })
    return rows
