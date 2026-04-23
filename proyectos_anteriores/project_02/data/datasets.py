from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from proyectos_anteriores.project_02.config import TRAIN_RATIO, WINDOW_WEEKS
from proyectos_anteriores.project_02.data.preprocessing import (
    PreparedData,
    build_history_features,
    week_to_feature_vector,
)


@dataclass
class SplitIndices:
    train_target_week_indices: list[int]
    val_target_week_indices: list[int]


def build_split_indices(prepared: PreparedData, train_ratio: float = TRAIN_RATIO) -> SplitIndices:
    n_samples = max(0, len(prepared.weeks) - WINDOW_WEEKS)
    target_indices = list(range(WINDOW_WEEKS, WINDOW_WEEKS + n_samples))
    split_at = max(int(len(target_indices) * train_ratio), 1)
    split_at = min(split_at, len(target_indices) - 1) if len(target_indices) > 1 else len(target_indices)
    return SplitIndices(
        train_target_week_indices=target_indices[:split_at],
        val_target_week_indices=target_indices[split_at:],
    )


def _build_context_sequence(prepared: PreparedData, target_week_idx: int) -> np.ndarray:
    context_weeks = prepared.weeks[target_week_idx - WINDOW_WEEKS : target_week_idx]
    return np.stack([week_to_feature_vector(week) for week in context_weeks]).astype(np.float32)


class OccurrenceDataset(Dataset):
    def __init__(self, prepared: PreparedData, target_week_indices: list[int]):
        self.sequences: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.week_indices = target_week_indices

        for idx in target_week_indices:
            seq = _build_context_sequence(prepared, idx)
            target = prepared.weeks[idx].counts.astype(np.int64)
            self.sequences.append(seq)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return {
            "sequence": torch.tensor(self.sequences[index], dtype=torch.float32),
            "target_counts": torch.tensor(self.targets[index], dtype=torch.long),
            "target_week_index": torch.tensor(self.week_indices[index], dtype=torch.long),
        }


class TemporalDataset(Dataset):
    def __init__(self, prepared: PreparedData, target_week_indices: list[int]):
        self.samples: list[dict[str, np.ndarray | int | float]] = []

        for week_idx in target_week_indices:
            seq = _build_context_sequence(prepared, week_idx)
            target_week = prepared.weeks[week_idx]

            for task_id, events in target_week.events_by_task.items():
                if not events:
                    continue

                task_count = min(len(events), prepared.max_count_cap)

                history_features = build_history_features(
                    weeks=prepared.weeks,
                    target_week_index=week_idx,
                    task_id=task_id,
                    duration_min=prepared.duration_min,
                    duration_max=prepared.duration_max,
                )

                sorted_events = sorted(events, key=lambda x: (x.start_bin, x.start_time))[: prepared.max_count_cap]

                for occurrence_index, event in enumerate(sorted_events):
                    self.samples.append(
                        {
                            "sequence": seq,
                            "task_id": int(task_id),
                            "occurrence_index": int(occurrence_index),
                            "predicted_count_norm": float(task_count / prepared.max_count_cap),
                            "history_features": history_features,
                            "target_start_bin": int(event.start_bin),
                            "target_duration_norm": float(
                                (event.duration_minutes - prepared.duration_min)
                                / max(prepared.duration_max - prepared.duration_min, 1e-6)
                            ),
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        return {
            "sequence": torch.tensor(item["sequence"], dtype=torch.float32),
            "task_id": torch.tensor(item["task_id"], dtype=torch.long),
            "occurrence_index": torch.tensor(item["occurrence_index"], dtype=torch.long),
            "predicted_count_norm": torch.tensor(item["predicted_count_norm"], dtype=torch.float32),
            "history_features": torch.tensor(item["history_features"], dtype=torch.float32),
            "target_start_bin": torch.tensor(item["target_start_bin"], dtype=torch.long),
            "target_duration_norm": torch.tensor(item["target_duration_norm"], dtype=torch.float32),
        }