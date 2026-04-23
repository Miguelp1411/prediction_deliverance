from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TRAIN_RATIO, WINDOW_WEEKS
from data.preprocessing import PreparedData, build_temporal_context, week_to_feature_vector


@dataclass
class SplitIndices:
    train_target_week_indices: list[int]
    val_target_week_indices: list[int]


def build_split_indices(prepared: PreparedData, train_ratio: float = TRAIN_RATIO) -> SplitIndices:
    n_samples = max(0, len(prepared.weeks) - WINDOW_WEEKS)
    target_indices = list(range(WINDOW_WEEKS, WINDOW_WEEKS + n_samples))
    split_at = max(int(len(target_indices) * train_ratio), 1)
    split_at = min(split_at, len(target_indices) - 1) if len(target_indices) > 1 else len(target_indices)
    return SplitIndices(target_indices[:split_at], target_indices[split_at:])


def _build_context_sequence(prepared: PreparedData, target_week_idx: int) -> np.ndarray:
    context_weeks = prepared.weeks[max(0, target_week_idx - WINDOW_WEEKS):target_week_idx]
    if not context_weeks:
        return np.zeros((WINDOW_WEEKS, prepared.week_feature_dim), dtype=np.float32)
    seq = np.stack([week_to_feature_vector(week) for week in context_weeks]).astype(np.float32)
    if seq.shape[0] < WINDOW_WEEKS:
        pad = np.zeros((WINDOW_WEEKS - seq.shape[0], prepared.week_feature_dim), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    return seq


class OccurrenceDataset(Dataset):
    def __init__(self, prepared: PreparedData, target_week_indices: list[int]):
        self.sequences = []
        self.targets = []
        self.week_indices = target_week_indices
        for idx in target_week_indices:
            self.sequences.append(_build_context_sequence(prepared, idx))
            self.targets.append(prepared.weeks[idx].counts.astype(np.int64))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return {
            'sequence': torch.tensor(self.sequences[index], dtype=torch.float32),
            'target_counts': torch.tensor(self.targets[index], dtype=torch.long),
            'target_week_index': torch.tensor(self.week_indices[index], dtype=torch.long),
        }


class TemporalDataset(Dataset):
    def __init__(self, prepared: PreparedData, target_week_indices: list[int]):
        self.samples: list[dict[str, np.ndarray | int | float]] = []
        duration_span = max(prepared.duration_max - prepared.duration_min, 1e-6)

        for week_idx in target_week_indices:
            seq = _build_context_sequence(prepared, week_idx)
            target_week = prepared.weeks[week_idx]
            for task_id, events in target_week.events_by_task.items():
                if not events:
                    continue
                task_count = min(len(events), prepared.max_count_cap)
                sorted_events = sorted(events, key=lambda x: (x.start_bin, x.start_time))[:prepared.max_count_cap]
                for occurrence_index, event in enumerate(sorted_events):
                    context = build_temporal_context(
                        prepared.weeks,
                        week_idx,
                        task_id,
                        occurrence_index,
                        prepared.duration_min,
                        prepared.duration_max,
                        max_occurrences_per_task=prepared.max_occurrences_per_task,
                    )
                    target_start_bin = int(event.start_bin)
                    self.samples.append({
                        'sequence': seq,
                        'task_id': int(task_id),
                        'occurrence_index': int(occurrence_index),
                        'predicted_count_norm': float(task_count / prepared.max_count_cap),
                        'history_features': context.history_features,
                        'anchor_start_bin': int(context.anchor_start_bin),
                        'anchor_day': int(context.anchor_day),
                        'target_start_bin': target_start_bin,
                        'target_duration_norm': float((event.duration_minutes - prepared.duration_min) / duration_span),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        return {
            'sequence': torch.tensor(item['sequence'], dtype=torch.float32),
            'task_id': torch.tensor(item['task_id'], dtype=torch.long),
            'occurrence_index': torch.tensor(item['occurrence_index'], dtype=torch.long),
            'predicted_count_norm': torch.tensor(item['predicted_count_norm'], dtype=torch.float32),
            'history_features': torch.tensor(item['history_features'], dtype=torch.float32),
            'anchor_start_bin': torch.tensor(item['anchor_start_bin'], dtype=torch.long),
            'anchor_day': torch.tensor(item['anchor_day'], dtype=torch.long),
            'target_start_bin': torch.tensor(item['target_start_bin'], dtype=torch.long),
            'target_duration_norm': torch.tensor(item['target_duration_norm'], dtype=torch.float32),
        }
