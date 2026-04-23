from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from proyectos_anteriores.proyectos_6.project_06_v4.config import OCC_COUNT_LOOKUP_BATCH_SIZE, TRAIN_RATIO, WINDOW_WEEKS, bins_per_day
from proyectos_anteriores.proyectos_6.project_06_v4.data.preprocessing import (
    PreparedData,
    build_context_sequence_features,
    build_prediction_occurrence_slots,
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
            seq_tensor = torch.from_numpy(sequences[start:end]).to(device)
            logits = occurrence_model(seq_tensor)
            pred_counts = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
            pred_counts = np.clip(pred_counts, 0, prepared.max_count_cap)
            for local_idx, week_idx in enumerate(target_week_indices[start:end]):
                lookup[int(week_idx)] = pred_counts[local_idx]
    return lookup


class OccurrenceDataset(Dataset):
    def __init__(self, prepared: PreparedData, target_week_indices: list[int]):
        self.week_indices = target_week_indices
        seqs = [_build_context_sequence(prepared, idx) for idx in target_week_indices]
        tgts = [prepared.weeks[idx].counts.astype(np.int64) for idx in target_week_indices]
        if seqs:
            self._sequences = torch.from_numpy(np.stack(seqs).astype(np.float32))
            self._targets = torch.from_numpy(np.stack(tgts))
        else:
            self._sequences = torch.zeros(0, dtype=torch.float32)
            self._targets = torch.zeros(0, dtype=torch.long)
        self._week_idx_tensor = torch.tensor(target_week_indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.week_indices)

    def __getitem__(self, index: int):
        return {
            'sequence': self._sequences[index],
            'target_counts': self._targets[index],
            'target_week_index': self._week_idx_tensor[index],
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
        raw_samples: list[dict[str, np.ndarray | int | float]] = []
        per_day_bins = bins_per_day()
        total_weeks = len(target_week_indices)
        count_blend_alpha = float(np.clip(count_blend_alpha, 0.0, 1.0))

        for i, week_idx in enumerate(target_week_indices):
            if show_progress:
                print_progress_inline(desc, i + 1, total_weeks, extra=f"muestras={len(raw_samples)}")

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
                    )
                    target_start_bin = int(event.start_bin)
                    denom = max(predicted_task_count - 1, 1)
                    occurrence_progress = float(sample_position / denom) if predicted_task_count > 1 else 0.0
                    target_day = int(target_start_bin // per_day_bins)
                    anchor_day = int(context.anchor_start_bin // per_day_bins)
                    target_local_bin = int(target_start_bin % per_day_bins)
                    anchor_local_bin = int(context.anchor_start_bin % per_day_bins)
                    raw_samples.append({
                        'sequence': seq,
                        'task_id': int(task_id),
                        'occurrence_slot': int(occurrence_slot),
                        'occurrence_progress': occurrence_progress,
                        'predicted_count_norm': float(predicted_task_count / prepared.max_count_cap),
                        'target_count_norm': float(task_count / prepared.max_count_cap),
                        'history_features': context.history_features,
                        'anchor_start_bin': int(context.anchor_start_bin),
                        'anchor_day': int(context.anchor_day),
                        'target_start_bin': target_start_bin,
                        'target_day_offset_idx': int(global_day_offset_to_index(target_day - anchor_day)),
                        'target_local_offset_idx': int(local_start_offset_to_index(target_local_bin - anchor_local_bin)),
                        'target_duration_norm': float((event.duration_minutes - prepared.duration_min) / max(prepared.duration_max - prepared.duration_min, 1e-6)),
                    })

        # Pre-stack all fields into contiguous tensors for fast __getitem__
        self._len = len(raw_samples)
        if self._len > 0:
            self._sequence = torch.from_numpy(np.stack([s['sequence'] for s in raw_samples]).astype(np.float32))
            self._history_features = torch.from_numpy(np.stack([s['history_features'] for s in raw_samples]).astype(np.float32))
            self._task_id = torch.tensor([s['task_id'] for s in raw_samples], dtype=torch.long)
            self._occurrence_slot = torch.tensor([s['occurrence_slot'] for s in raw_samples], dtype=torch.long)
            self._occurrence_progress = torch.tensor([s['occurrence_progress'] for s in raw_samples], dtype=torch.float32)
            self._predicted_count_norm = torch.tensor([s['predicted_count_norm'] for s in raw_samples], dtype=torch.float32)
            self._target_count_norm = torch.tensor([s['target_count_norm'] for s in raw_samples], dtype=torch.float32)
            self._anchor_start_bin = torch.tensor([s['anchor_start_bin'] for s in raw_samples], dtype=torch.long)
            self._anchor_day = torch.tensor([s['anchor_day'] for s in raw_samples], dtype=torch.long)
            self._target_start_bin = torch.tensor([s['target_start_bin'] for s in raw_samples], dtype=torch.long)
            self._target_day_offset_idx = torch.tensor([s['target_day_offset_idx'] for s in raw_samples], dtype=torch.long)
            self._target_local_offset_idx = torch.tensor([s['target_local_offset_idx'] for s in raw_samples], dtype=torch.long)
            self._target_duration_norm = torch.tensor([s['target_duration_norm'] for s in raw_samples], dtype=torch.float32)
        else:
            self._sequence = torch.zeros(0, dtype=torch.float32)
            self._history_features = torch.zeros(0, dtype=torch.float32)
            self._task_id = torch.zeros(0, dtype=torch.long)
            self._occurrence_slot = torch.zeros(0, dtype=torch.long)
            self._occurrence_progress = torch.zeros(0, dtype=torch.float32)
            self._predicted_count_norm = torch.zeros(0, dtype=torch.float32)
            self._target_count_norm = torch.zeros(0, dtype=torch.float32)
            self._anchor_start_bin = torch.zeros(0, dtype=torch.long)
            self._anchor_day = torch.zeros(0, dtype=torch.long)
            self._target_start_bin = torch.zeros(0, dtype=torch.long)
            self._target_day_offset_idx = torch.zeros(0, dtype=torch.long)
            self._target_local_offset_idx = torch.zeros(0, dtype=torch.long)
            self._target_duration_norm = torch.zeros(0, dtype=torch.float32)
        # Free raw_samples memory
        del raw_samples

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int):
        return {
            'sequence': self._sequence[index],
            'task_id': self._task_id[index],
            'occurrence_slot': self._occurrence_slot[index],
            'occurrence_progress': self._occurrence_progress[index],
            'predicted_count_norm': self._predicted_count_norm[index],
            'target_count_norm': self._target_count_norm[index],
            'history_features': self._history_features[index],
            'anchor_start_bin': self._anchor_start_bin[index],
            'anchor_day': self._anchor_day[index],
            'target_start_bin': self._target_start_bin[index],
            'target_day_offset_idx': self._target_day_offset_idx[index],
            'target_local_offset_idx': self._target_local_offset_idx[index],
            'target_duration_norm': self._target_duration_norm[index],
        }
