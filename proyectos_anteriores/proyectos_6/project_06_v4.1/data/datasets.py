from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from config import OCC_COUNT_LOOKUP_BATCH_SIZE, TRAIN_RATIO, WINDOW_WEEKS, bins_per_day
from data.preprocessing import (
    PreparedData,
    build_context_sequence_features,
    build_target_occurrence_slot_assignments,
    build_temporal_context,
    global_day_offset_to_index,
    local_start_offset_to_index,
)


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
    return build_context_sequence_features(prepared.weeks, target_week_idx, WINDOW_WEEKS, len(prepared.task_names))


def build_occurrence_count_lookup(
    prepared: PreparedData,
    target_week_indices: list[int],
    occurrence_model,
    device: torch.device,
    batch_size: int = OCC_COUNT_LOOKUP_BATCH_SIZE,
) -> dict[int, np.ndarray]:
    lookup: dict[int, np.ndarray] = {}
    if not target_week_indices:
        return lookup

    sequences = np.stack([_build_context_sequence(prepared, week_idx) for week_idx in target_week_indices]).astype(np.float32)
    occurrence_model.eval()

    with torch.no_grad():
        for start in range(0, len(target_week_indices), max(1, batch_size)):
            end = min(start + max(1, batch_size), len(target_week_indices))
            seq_tensor = torch.from_numpy(sequences[start:end]).to(device, non_blocking=True)
            logits = occurrence_model(seq_tensor)
            pred_counts = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
            pred_counts = np.clip(pred_counts, 0, prepared.max_count_cap)
            for local_idx, week_idx in enumerate(target_week_indices[start:end]):
                lookup[int(week_idx)] = pred_counts[local_idx]
    return lookup


class OccurrenceDataset(Dataset):
    def __init__(self, prepared: PreparedData, target_week_indices: list[int]):
        self.week_indices = torch.tensor(target_week_indices, dtype=torch.long)
        num_tasks = len(prepared.task_names)
        if target_week_indices:
            sequences = np.stack([_build_context_sequence(prepared, idx) for idx in target_week_indices]).astype(np.float32)
            targets = np.stack([prepared.weeks[idx].counts.astype(np.int64) for idx in target_week_indices]).astype(np.int64)
        else:
            sequences = np.zeros((0, WINDOW_WEEKS, prepared.week_feature_dim), dtype=np.float32)
            targets = np.zeros((0, num_tasks), dtype=np.int64)

        self.sequences = torch.from_numpy(sequences)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, index: int):
        return {
            'sequence': self.sequences[index],
            'target_counts': self.targets[index],
            'target_week_index': self.week_indices[index],
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
        sequence_bank: list[np.ndarray] = []
        sequence_index: list[int] = []
        task_id: list[int] = []
        occurrence_slot: list[int] = []
        occurrence_progress: list[float] = []
        predicted_count_norm: list[float] = []
        target_count_norm: list[float] = []
        history_features: list[np.ndarray] = []
        anchor_start_bin: list[int] = []
        anchor_day: list[int] = []
        target_start_bin: list[int] = []
        target_day_offset_idx: list[int] = []
        target_local_offset_idx: list[int] = []
        target_duration_norm: list[float] = []

        per_day_bins = bins_per_day()
        total_weeks = len(target_week_indices)
        count_blend_alpha = float(np.clip(count_blend_alpha, 0.0, 1.0))

        for i, week_idx in enumerate(target_week_indices):
            if show_progress:
                print_progress_inline(desc, i + 1, total_weeks, extra=f"muestras={len(task_id)}")

            seq = _build_context_sequence(prepared, week_idx).astype(np.float32, copy=False)
            sequence_bank.append(seq)
            current_sequence_idx = len(sequence_bank) - 1

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

            for this_task_id, events in target_week.events_by_task.items():
                if not events:
                    continue

                task_count = int(min(len(events), prepared.max_count_cap))
                predicted_task_count = int(np.clip(effective_counts[this_task_id], 0, prepared.max_count_cap))
                assigned_events = build_target_occurrence_slot_assignments(
                    prepared.weeks,
                    week_idx,
                    this_task_id,
                    max_occurrences_per_task=prepared.max_occurrences_per_task,
                )
                for sample_position, (this_occurrence_slot, event) in enumerate(assigned_events):
                    context = build_temporal_context(
                        prepared.weeks,
                        week_idx,
                        this_task_id,
                        this_occurrence_slot,
                        prepared.duration_min,
                        prepared.duration_max,
                        max_occurrences_per_task=prepared.max_occurrences_per_task,
                    )
                    this_target_start_bin = int(event.start_bin)
                    denom = max(predicted_task_count - 1, 1)
                    this_occurrence_progress = float(sample_position / denom) if predicted_task_count > 1 else 0.0
                    this_target_day = int(this_target_start_bin // per_day_bins)
                    this_anchor_day = int(context.anchor_start_bin // per_day_bins)
                    this_target_local_bin = int(this_target_start_bin % per_day_bins)
                    this_anchor_local_bin = int(context.anchor_start_bin % per_day_bins)

                    sequence_index.append(current_sequence_idx)
                    task_id.append(int(this_task_id))
                    occurrence_slot.append(int(this_occurrence_slot))
                    occurrence_progress.append(this_occurrence_progress)
                    predicted_count_norm.append(float(predicted_task_count / prepared.max_count_cap))
                    target_count_norm.append(float(task_count / prepared.max_count_cap))
                    history_features.append(np.asarray(context.history_features, dtype=np.float32))
                    anchor_start_bin.append(int(context.anchor_start_bin))
                    anchor_day.append(int(context.anchor_day))
                    target_start_bin.append(this_target_start_bin)
                    target_day_offset_idx.append(int(global_day_offset_to_index(this_target_day - this_anchor_day)))
                    target_local_offset_idx.append(int(local_start_offset_to_index(this_target_local_bin - this_anchor_local_bin)))
                    target_duration_norm.append(
                        float(
                            (event.duration_minutes - prepared.duration_min)
                            / max(prepared.duration_max - prepared.duration_min, 1e-6)
                        )
                    )

        if sequence_bank:
            self.sequence_bank = torch.from_numpy(np.stack(sequence_bank).astype(np.float32))
        else:
            self.sequence_bank = torch.zeros((0, WINDOW_WEEKS, prepared.week_feature_dim), dtype=torch.float32)

        if sequence_index:
            self.sequence_index = torch.tensor(sequence_index, dtype=torch.long)
            self.task_id = torch.tensor(task_id, dtype=torch.long)
            self.occurrence_slot = torch.tensor(occurrence_slot, dtype=torch.long)
            self.occurrence_progress = torch.tensor(occurrence_progress, dtype=torch.float32)
            self.predicted_count_norm = torch.tensor(predicted_count_norm, dtype=torch.float32)
            self.target_count_norm = torch.tensor(target_count_norm, dtype=torch.float32)
            self.history_features = torch.from_numpy(np.stack(history_features).astype(np.float32))
            self.anchor_start_bin = torch.tensor(anchor_start_bin, dtype=torch.long)
            self.anchor_day = torch.tensor(anchor_day, dtype=torch.long)
            self.target_start_bin = torch.tensor(target_start_bin, dtype=torch.long)
            self.target_day_offset_idx = torch.tensor(target_day_offset_idx, dtype=torch.long)
            self.target_local_offset_idx = torch.tensor(target_local_offset_idx, dtype=torch.long)
            self.target_duration_norm = torch.tensor(target_duration_norm, dtype=torch.float32)
        else:
            self.sequence_index = torch.zeros(0, dtype=torch.long)
            self.task_id = torch.zeros(0, dtype=torch.long)
            self.occurrence_slot = torch.zeros(0, dtype=torch.long)
            self.occurrence_progress = torch.zeros(0, dtype=torch.float32)
            self.predicted_count_norm = torch.zeros(0, dtype=torch.float32)
            self.target_count_norm = torch.zeros(0, dtype=torch.float32)
            self.history_features = torch.zeros((0, prepared.history_feature_dim), dtype=torch.float32)
            self.anchor_start_bin = torch.zeros(0, dtype=torch.long)
            self.anchor_day = torch.zeros(0, dtype=torch.long)
            self.target_start_bin = torch.zeros(0, dtype=torch.long)
            self.target_day_offset_idx = torch.zeros(0, dtype=torch.long)
            self.target_local_offset_idx = torch.zeros(0, dtype=torch.long)
            self.target_duration_norm = torch.zeros(0, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.task_id.shape[0])

    def __getitem__(self, index: int):
        seq_idx = int(self.sequence_index[index])
        return {
            'sequence': self.sequence_bank[seq_idx],
            'task_id': self.task_id[index],
            'occurrence_slot': self.occurrence_slot[index],
            'occurrence_progress': self.occurrence_progress[index],
            'predicted_count_norm': self.predicted_count_norm[index],
            'target_count_norm': self.target_count_norm[index],
            'history_features': self.history_features[index],
            'anchor_start_bin': self.anchor_start_bin[index],
            'anchor_day': self.anchor_day[index],
            'target_start_bin': self.target_start_bin[index],
            'target_day_offset_idx': self.target_day_offset_idx[index],
            'target_local_offset_idx': self.target_local_offset_idx[index],
            'target_duration_norm': self.target_duration_norm[index],
        }
