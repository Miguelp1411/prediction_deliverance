from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TRAIN_RATIO, WINDOW_WEEKS, bins_per_day
from data.preprocessing import PreparedData, build_temporal_context, week_to_feature_vector


def print_progress_inline(prefix: str, current: int, total: int, extra: str = ""):
    pct = (current / total) * 100 if total else 100.0
    msg = f"\r{prefix}: {current}/{total} ({pct:5.1f}%)"
    if extra:
        msg += f" | {extra}"
    print(msg, end="", flush=True)
    if current == total:
        print()


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


def build_occurrence_count_lookup(
    prepared: PreparedData,
    target_week_indices: list[int],
    occurrence_model,
    device: torch.device,
) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    occurrence_model.eval()
    with torch.no_grad():
        for week_idx in target_week_indices:
            seq = _build_context_sequence(prepared, week_idx)
            seq_tensor = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
            logits = occurrence_model(seq_tensor)
            pred_counts = torch.argmax(logits, dim=-1).squeeze(0).detach().cpu().numpy().astype(np.int64)
            lookup[int(week_idx)] = np.clip(pred_counts, 0, prepared.max_count_cap)
    return lookup


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
    def __init__(
        self,
        prepared: PreparedData,
        target_week_indices: list[int],
        count_lookup: dict[int, np.ndarray] | None = None,
        count_blend_alpha: float = 0.0,
        show_progress: bool = False,
        desc: str = "TemporalDataset",
    ):
        self.samples: list[dict[str, np.ndarray | int | float]] = []
        per_day_bins = bins_per_day()
        total_weeks = len(target_week_indices)
        count_blend_alpha = float(np.clip(count_blend_alpha, 0.0, 1.0))

        for i, week_idx in enumerate(target_week_indices):
            if show_progress:
                print_progress_inline(desc, i + 1, total_weeks, extra=f"muestras={len(self.samples)}")

            seq = _build_context_sequence(prepared, week_idx)
            target_week = prepared.weeks[week_idx]
            predicted_counts = None
            if count_lookup is not None and int(week_idx) in count_lookup:
                predicted_counts = np.asarray(count_lookup[int(week_idx)], dtype=np.int64)
            target_counts = target_week.counts.astype(np.int64)
            if predicted_counts is None:
                effective_counts = target_counts
            else:
                blended = np.round(
                    (1.0 - count_blend_alpha) * predicted_counts.astype(np.float32)
                    + count_blend_alpha * target_counts.astype(np.float32)
                ).astype(np.int64)
                effective_counts = np.clip(blended, 0, prepared.max_count_cap)

            for task_id, events in target_week.events_by_task.items():
                if not events:
                    continue
                task_count = int(min(len(events), prepared.max_count_cap))
                predicted_task_count = int(np.clip(effective_counts[task_id], 0, prepared.max_count_cap))
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
                    denom = max(predicted_task_count - 1, 1)
                    occurrence_progress = float(occurrence_index / denom) if predicted_task_count > 1 else 0.0
                    self.samples.append({
                        'sequence': seq,
                        'task_id': int(task_id),
                        'occurrence_index': int(occurrence_index),
                        'occurrence_progress': occurrence_progress,
                        'predicted_count_norm': float(predicted_task_count / prepared.max_count_cap),
                        'target_count_norm': float(task_count / prepared.max_count_cap),
                        'history_features': context.history_features,
                        'anchor_start_bin': int(context.anchor_start_bin),
                        'anchor_day': int(context.anchor_day),
                        'target_start_bin': target_start_bin,
                        'target_day_idx': int(target_start_bin // per_day_bins),
                        'target_time_bin_idx': int(target_start_bin % per_day_bins),
                        'target_duration_norm': float((event.duration_minutes - prepared.duration_min) / max(prepared.duration_max - prepared.duration_min, 1e-6)),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        return {
            'sequence': torch.tensor(item['sequence'], dtype=torch.float32),
            'task_id': torch.tensor(item['task_id'], dtype=torch.long),
            'occurrence_index': torch.tensor(item['occurrence_index'], dtype=torch.long),
            'occurrence_progress': torch.tensor(item['occurrence_progress'], dtype=torch.float32),
            'predicted_count_norm': torch.tensor(item['predicted_count_norm'], dtype=torch.float32),
            'target_count_norm': torch.tensor(item['target_count_norm'], dtype=torch.float32),
            'history_features': torch.tensor(item['history_features'], dtype=torch.float32),
            'anchor_start_bin': torch.tensor(item['anchor_start_bin'], dtype=torch.long),
            'anchor_day': torch.tensor(item['anchor_day'], dtype=torch.long),
            'target_start_bin': torch.tensor(item['target_start_bin'], dtype=torch.long),
            'target_day_idx': torch.tensor(item['target_day_idx'], dtype=torch.long),
            'target_time_bin_idx': torch.tensor(item['target_time_bin_idx'], dtype=torch.long),
            'target_duration_norm': torch.tensor(item['target_duration_norm'], dtype=torch.float32),
        }
