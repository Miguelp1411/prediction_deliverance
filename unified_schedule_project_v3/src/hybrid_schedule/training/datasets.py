from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hybrid_schedule.data.features import (
    GlobalContext,
    SlotPrototype,
    assign_events_to_prototypes,
    build_history_tensor,
    build_slot_plan_features,
    build_temporal_numeric_features,
    task_slot_prototypes,
    task_temporal_profile,
)
from hybrid_schedule.data.scaling import transform_feature_matrix


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
        split = max(min_history_weeks + 1, int(round(n * float(train_ratio))))
        for idx in range(min_history_weeks, split):
            train_indices.append((database_id, robot_id, idx))
        for idx in range(split, n):
            val_indices.append((database_id, robot_id, idx))
    return DatasetSplit(train_indices=train_indices, val_indices=val_indices)


def _default_prototype(slot_id: int, bins_per_day: int) -> SlotPrototype:
    # Martes 10:00 como ancla neutra; evita que los huecos vacíos colapsen en 0 absoluto.
    center_bin = 1 * bins_per_day + int(round(10 * 60 / 5))
    return SlotPrototype(task_idx=0, slot_id=int(slot_id), center_bin=int(center_bin), duration_bins=12, support=0.0)


def _expand_prototypes(
    prototypes: list[SlotPrototype],
    task_idx: int,
    max_slots: int,
    bins_per_day: int,
) -> list[SlotPrototype]:
    protos = sorted(prototypes, key=lambda p: int(p.slot_id))
    if not protos:
        protos = [_default_prototype(0, bins_per_day)]
    base_center = int(np.median([p.center_bin for p in protos]))
    base_duration = max(1, int(round(np.median([p.duration_bins for p in protos]))))
    out: list[SlotPrototype] = []
    proto_map = {int(p.slot_id): p for p in protos}
    for slot_id in range(max_slots):
        if slot_id in proto_map:
            p = proto_map[slot_id]
            out.append(SlotPrototype(
                task_idx=int(task_idx),
                slot_id=int(slot_id),
                center_bin=int(p.center_bin),
                duration_bins=max(1, int(p.duration_bins)),
                support=float(p.support),
            ))
            continue
        offset = (slot_id - len(protos) + 1) * max(3, bins_per_day // 8)
        center = int(np.clip(base_center + offset, 0, 7 * bins_per_day - 1))
        out.append(SlotPrototype(
            task_idx=int(task_idx),
            slot_id=int(slot_id),
            center_bin=center,
            duration_bins=base_duration,
            support=0.0,
        ))
    return out


class UnifiedWeekSlotDataset(Dataset):
    def __init__(
        self,
        context: GlobalContext,
        indices: list[tuple[str, str, int]],
        window_weeks: int,
        max_slots: int,
        max_slots_by_task: list[int] | None = None,
        bin_minutes: int = 5,
        history_stats: dict[str, Any] | None = None,
        numeric_stats: dict[str, Any] | None = None,
    ):
        self.context = context
        self.indices = list(indices)
        self.window_weeks = int(window_weeks)
        self.max_slots = int(max_slots)
        self.bin_minutes = int(bin_minutes)
        self.bins_per_day = int(24 * 60 / self.bin_minutes)
        self.num_tasks = len(context.task_names)

        if max_slots_by_task is None:
            self.max_slots_by_task = [self.max_slots] * self.num_tasks
        else:
            self.max_slots_by_task = [int(x) for x in max_slots_by_task]
            if len(self.max_slots_by_task) != self.num_tasks:
                raise ValueError('max_slots_by_task debe tener longitud num_tasks')
            if any(x <= 0 for x in self.max_slots_by_task):
                raise ValueError('Todos los valores de max_slots_by_task deben ser > 0')
            if any(x > self.max_slots for x in self.max_slots_by_task):
                raise ValueError('Ningún valor de max_slots_by_task puede superar max_slots')

        self.query_count = self.num_tasks * self.max_slots
        self.history_stats = history_stats
        self.numeric_stats = numeric_stats
        self.samples: list[dict[str, Any]] = []
        for database_id, robot_id, target_week_idx in self.indices:
            self.samples.append(self._build_sample(database_id, robot_id, target_week_idx))

    def _build_sample(self, database_id: str, robot_id: str, target_week_idx: int) -> dict[str, Any]:
        series = self.context.series[(database_id, robot_id)]
        history = build_history_tensor(series, target_week_idx, self.window_weeks, bin_minutes=self.bin_minutes)
        history = transform_feature_matrix(history, self.history_stats)

        db_idx = self.context.database_to_idx[database_id]
        robot_key = f'{database_id}::{robot_id}'
        robot_idx = self.context.robot_to_idx[robot_key]

        task_dispersion = {
            task_idx: float(task_temporal_profile(series, target_week_idx, task_idx, bin_minutes=self.bin_minutes).get('start_dispersion', 0.0))
            for task_idx in range(self.num_tasks)
        }

        prototypes_by_task: dict[int, list[SlotPrototype]] = {}
        planned_slots: list[dict[str, Any]] = []
        template_count_by_task: dict[int, int] = {}
        for task_idx in range(self.num_tasks):
            task_cap = int(self.max_slots_by_task[task_idx])
            raw_protos = task_slot_prototypes(series, target_week_idx, task_idx, max_slots=task_cap, max_weeks=104)
            protos = _expand_prototypes(raw_protos, task_idx=task_idx, max_slots=self.max_slots, bins_per_day=self.bins_per_day)
            prototypes_by_task[task_idx] = protos
            template_count_by_task[task_idx] = int(sum(1 for p in raw_protos if float(p.support) > 0))
            for proto in protos:
                if int(proto.slot_id) >= task_cap:
                    continue
                planned_slots.append({
                    'task_idx': int(task_idx),
                    'slot_id': int(proto.slot_id),
                    'anchor_start_bin': int(proto.center_bin),
                    'anchor_duration_bins': int(proto.duration_bins),
                    'support': float(proto.support),
                    'pred_task_count': int(task_cap),
                    'template_task_count': int(template_count_by_task[task_idx]),
                })
        plan_features = build_slot_plan_features(planned_slots, task_dispersion_by_task=task_dispersion, bin_minutes=self.bin_minutes)

        events_by_task: dict[int, list[Any]] = {}
        target_events = series.events[target_week_idx] if 0 <= int(target_week_idx) < len(series.events) else []
        for event in target_events:
            events_by_task.setdefault(int(event.task_idx), []).append(event)

        assigned_by_key: dict[tuple[int, int], Any] = {}
        for task_idx in range(self.num_tasks):
            assignments = assign_events_to_prototypes(events_by_task.get(task_idx, []), prototypes_by_task[task_idx])
            for slot_id, event, _proto in assignments:
                if 0 <= int(slot_id) < self.max_slots:
                    assigned_by_key[(task_idx, int(slot_id))] = event

        numeric_rows = []
        task_ids = []
        slot_ids = []
        anchor_days = []
        anchor_times = []
        query_mask = []
        active_targets = []
        target_days = []
        target_times = []
        target_log_durations = []
        target_starts = []
        target_durations = []

        for task_idx in range(self.num_tasks):
            task_cap = int(self.max_slots_by_task[task_idx])
            for proto in prototypes_by_task[task_idx]:
                slot_id = int(proto.slot_id)
                anchor_start = int(proto.center_bin)
                anchor_duration = int(proto.duration_bins)
                anchor_day = anchor_start // self.bins_per_day
                anchor_time = anchor_start % self.bins_per_day
                is_valid_query = slot_id < task_cap
                query_mask.append(bool(is_valid_query))
                numeric = build_temporal_numeric_features(
                    series=series,
                    target_week_idx=target_week_idx,
                    task_idx=task_idx,
                    slot_id=slot_id,
                    anchor_start=anchor_start,
                    anchor_duration=anchor_duration,
                    pred_task_count=max(1, int(task_cap)),
                    template_task_count=int(template_count_by_task[task_idx]),
                    slot_support=float(proto.support),
                    max_slots=task_cap,
                    bin_minutes=self.bin_minutes,
                    plan_features=plan_features.get((task_idx, slot_id), {}),
                )
                numeric = transform_feature_matrix(numeric, self.numeric_stats)
                evt = assigned_by_key.get((task_idx, slot_id))
                active = 1.0 if evt is not None and is_valid_query else 0.0
                if evt is not None:
                    target_start = int(evt.start_bin)
                    target_duration = max(1, int(evt.duration_bins))
                else:
                    target_start = anchor_start
                    target_duration = anchor_duration

                numeric_rows.append(numeric.astype(np.float32))
                task_ids.append(int(task_idx))
                slot_ids.append(int(slot_id))
                anchor_days.append(int(anchor_day))
                anchor_times.append(int(anchor_time))
                active_targets.append(float(active))
                target_days.append(int(target_start // self.bins_per_day))
                target_times.append(int(target_start % self.bins_per_day))
                target_log_durations.append(float(np.log1p(max(1, target_duration))))
                target_starts.append(int(target_start))
                target_durations.append(int(target_duration))

        true_counts = (
            series.counts[target_week_idx].astype(np.float32)
            if 0 <= int(target_week_idx) < len(series.counts)
            else np.zeros(self.num_tasks, dtype=np.float32)
        )
        return {
            'history': history.astype(np.float32),
            'task_ids': np.asarray(task_ids, dtype=np.int64),
            'database_ids': np.full(self.query_count, db_idx, dtype=np.int64),
            'robot_ids': np.full(self.query_count, robot_idx, dtype=np.int64),
            'slot_ids': np.asarray(slot_ids, dtype=np.int64),
            'anchor_days': np.asarray(anchor_days, dtype=np.int64),
            'anchor_times': np.asarray(anchor_times, dtype=np.int64),
            'numeric_features': np.stack(numeric_rows, axis=0).astype(np.float32),
            'query_mask': np.asarray(query_mask, dtype=np.bool_),
            'active_targets': np.asarray(active_targets, dtype=np.float32),
            'target_day_idx': np.asarray(target_days, dtype=np.int64),
            'target_time_bin_idx': np.asarray(target_times, dtype=np.int64),
            'target_log_duration': np.asarray(target_log_durations, dtype=np.float32),
            'target_start': np.asarray(target_starts, dtype=np.int64),
            'target_duration': np.asarray(target_durations, dtype=np.float32),
            'true_counts': true_counts,
            'database_id_str': database_id,
            'robot_id_str': robot_id,
            'week_index': int(target_week_idx),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            'history': torch.tensor(s['history'], dtype=torch.float32),
            'task_ids': torch.tensor(s['task_ids'], dtype=torch.long),
            'database_ids': torch.tensor(s['database_ids'], dtype=torch.long),
            'robot_ids': torch.tensor(s['robot_ids'], dtype=torch.long),
            'slot_ids': torch.tensor(s['slot_ids'], dtype=torch.long),
            'anchor_days': torch.tensor(s['anchor_days'], dtype=torch.long),
            'anchor_times': torch.tensor(s['anchor_times'], dtype=torch.long),
            'numeric_features': torch.tensor(s['numeric_features'], dtype=torch.float32),
            'query_mask': torch.tensor(s['query_mask'], dtype=torch.bool),
            'active_targets': torch.tensor(s['active_targets'], dtype=torch.float32),
            'target_day_idx': torch.tensor(s['target_day_idx'], dtype=torch.long),
            'target_time_bin_idx': torch.tensor(s['target_time_bin_idx'], dtype=torch.long),
            'target_log_duration': torch.tensor(s['target_log_duration'], dtype=torch.float32),
            'target_start': torch.tensor(s['target_start'], dtype=torch.long),
            'target_duration': torch.tensor(s['target_duration'], dtype=torch.float32),
            'true_counts': torch.tensor(s['true_counts'], dtype=torch.float32),
        }
