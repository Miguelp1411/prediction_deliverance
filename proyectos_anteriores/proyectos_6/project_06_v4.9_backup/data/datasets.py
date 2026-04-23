from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from config import OCC_COUNT_LOOKUP_BATCH_SIZE, TRAIN_RATIO, WINDOW_WEEKS, bins_per_day
from data.preprocessing import (
    PreparedData,
    build_context_sequence_features,
    build_prediction_occurrence_slots,
    build_target_occurrence_slot_assignments,
    build_temporal_context,
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


def _occurrence_logits(outputs):
    if isinstance(outputs, dict):
        return outputs['logits']
    return outputs


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
            seq_tensor = torch.from_numpy(sequences[start:end]).to(device)
            outputs = occurrence_model(seq_tensor)
            logits = _occurrence_logits(outputs)
            pred_counts = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
            pred_counts = np.clip(pred_counts, 0, prepared.max_count_cap)
            for local_idx, week_idx in enumerate(target_week_indices[start:end]):
                lookup[int(week_idx)] = pred_counts[local_idx]
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
            'sequence': np.asarray(self.sequences[index], dtype=np.float32),
            'target_counts': np.asarray(self.targets[index], dtype=np.int64),
            'target_week_index': np.int64(self.week_indices[index]),
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
                assigned_events = build_target_occurrence_slot_assignments(
                    prepared.weeks,
                    week_idx,
                    task_id,
                    max_occurrences_per_task=prepared.max_occurrences_per_task,
                )
                for sample_position, (occurrence_slot, event) in enumerate(assigned_events):
                    context = build_temporal_context(
                        prepared.weeks,
                        week_idx,
                        task_id,
                        occurrence_slot,
                        prepared.duration_min,
                        prepared.duration_max,
                        max_occurrences_per_task=prepared.max_occurrences_per_task,
                        predicted_count=predicted_task_count,
                    )
                    target_start_bin = int(event.start_bin)
                    denom = max(predicted_task_count - 1, 1)
                    occurrence_progress = float(sample_position / denom) if predicted_task_count > 1 else 0.0
                    target_day = int(target_start_bin // per_day_bins)
                    target_time_bin = int(target_start_bin % per_day_bins)
                    self.samples.append({
                        'sequence': seq,
                        'task_id': int(task_id),
                        'occurrence_slot': int(occurrence_slot),
                        'occurrence_progress': occurrence_progress,
                        'predicted_count_norm': float(predicted_task_count / prepared.max_count_cap),
                        'target_count_norm': float(task_count / prepared.max_count_cap),
                        'history_features': context.history_features,
                        'anchor_start_bin': int(context.anchor_start_bin),
                        'anchor_day': int(context.anchor_day),
                        'anchor_time_bin': int(context.anchor_time_bin),
                        'target_start_bin': target_start_bin,
                        'target_day_idx': target_day,
                        'target_time_bin_idx': target_time_bin,
                        'target_duration_norm': float((event.duration_minutes - prepared.duration_min) / max(prepared.duration_max - prepared.duration_min, 1e-6)),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        return {
            'sequence': np.asarray(item['sequence'], dtype=np.float32),
            'task_id': np.int64(item['task_id']),
            'occurrence_slot': np.int64(item['occurrence_slot']),
            'occurrence_progress': np.float32(item['occurrence_progress']),
            'predicted_count_norm': np.float32(item['predicted_count_norm']),
            'target_count_norm': np.float32(item['target_count_norm']),
            'history_features': np.asarray(item['history_features'], dtype=np.float32),
            'anchor_start_bin': np.int64(item['anchor_start_bin']),
            'anchor_day': np.int64(item['anchor_day']),
            'anchor_time_bin': np.int64(item['anchor_time_bin']),
            'target_start_bin': np.int64(item['target_start_bin']),
            'target_day_idx': np.int64(item['target_day_idx']),
            'target_time_bin_idx': np.int64(item['target_time_bin_idx']),
            'target_duration_norm': np.float32(item['target_duration_norm']),
        }
