from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import (
    ANCHOR_LOOKBACK_WEEKS,
    BIN_MINUTES,
    CAP_INFERENCE_SCOPE,
    DEFAULT_MAX_OCCURRENCES_PER_TASK,
    FEATURE_SCHEMA_VERSION,
    GLOBAL_DAY_OFFSET_RADIUS_BINS,
    HISTORY_SCALES,
    LOCAL_START_OFFSET_RADIUS_BINS,
    OCC_SEASONAL_LAGS,
    RECENCY_DECAY_BASE,
    TMP_SEASONAL_LAGS,
    TEMPORAL_NUM_ANCHOR_CANDIDATES,
    WINDOW_WEEKS,
    bins_per_day,
    num_time_bins,
)

RECENT_FREQ_SCALES = (2, 4, 12)
RECENT_OFFSET_SCALES = (1, 2, 4, 8)

_CONTEXT_SEQUENCE_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}
_TASK_TEMPORAL_CACHE: dict[tuple[int, int, int, int], 'TaskTemporalCache'] = {}


@dataclass
class EventRecord:
    task_id: int
    task_name: str
    start_bin: int
    duration_minutes: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp

    @property
    def type(self) -> str:
        return self.task_name


@dataclass
class WeekRecord:
    week_index: int
    week_start: pd.Timestamp
    counts: np.ndarray
    mean_start_sin: np.ndarray
    mean_start_cos: np.ndarray
    mean_duration_norm: np.ndarray
    start_circular_dispersion: np.ndarray
    active_days_norm: np.ndarray
    day_multimodality_norm: np.ndarray
    day_distribution: np.ndarray
    total_tasks_norm: float
    week_of_year_sin: float
    week_of_year_cos: float
    month_sin: float
    month_cos: float
    day_of_year_sin: float
    day_of_year_cos: float
    events_by_task: dict[int, list[EventRecord]]


@dataclass
class TemporalContext:
    history_features: np.ndarray
    anchor_start_bin: int
    anchor_day: int
    anchor_time_bin: int
    anchor_candidates: tuple[int, ...] = ()
    anchor_candidate_weights: tuple[float, ...] = ()


@dataclass
class OccurrencePrototype:
    slot_id: int
    center_bin: int
    support: float


@dataclass
class PreparedData:
    df: pd.DataFrame
    task_names: list[str]
    task_to_id: dict[str, int]
    weeks: list[WeekRecord]
    duration_min: float
    duration_max: float
    task_duration_medians: dict[str, float]
    max_count_cap: int
    week_feature_dim: int
    history_feature_dim: int
    max_occurrences_per_task: int
    max_tasks_per_week: int
    cap_inference_scope: str
    inferred_train_max_occurrences_per_task: int
    inferred_train_max_tasks_per_week: int
    inferred_full_max_occurrences_per_task: int
    inferred_full_max_tasks_per_week: int


@dataclass
class TaskTemporalCache:
    history: list[WeekRecord]
    anchor_history: list[WeekRecord]
    target_week_start: pd.Timestamp
    counts_per_week: np.ndarray
    sin_per_week: np.ndarray
    cos_per_week: np.ndarray
    dur_per_week: np.ndarray
    recent_all_bins: list[int]
    anchor_all_bins: list[int]
    prototypes: list[OccurrencePrototype]
    recent_slot_bins: dict[int, list[int]]
    anchor_slot_bins: dict[int, list[int]]
    lag_slot_to_bin: dict[int, dict[int, int]]


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    day_start = ts.normalize()
    return day_start - pd.Timedelta(days=int(ts.dayofweek))


def _duration_to_norm(duration: float, dur_min: float, dur_max: float) -> float:
    span = max(dur_max - dur_min, 1e-6)
    return float(np.clip((duration - dur_min) / span, 0.0, 1.0))


def _sanitize_cap(value: int | None, fallback: int) -> int:
    if value is None:
        return max(1, int(fallback))
    return max(1, int(value))

def print_progress_inline(prefix: str, current: int, total: int, extra: str = ""):
    pct = (current / total) * 100 if total else 100.0
    msg = f"\r{prefix}: {current}/{total} ({pct:5.1f}%)"
    if extra:
        msg += f" | {extra}"
    print(msg, end="", flush=True)
    if current == total:
        print()

def infer_preprocessing_caps(df_events: pd.DataFrame) -> tuple[int, int]:
    if df_events.empty:
        return 1, 1

    grouped_by_task = df_events.groupby(['week_start', 'task_id']).size()
    grouped_by_week = df_events.groupby('week_start').size()

    max_occurrences_per_task = int(grouped_by_task.max()) if not grouped_by_task.empty else 1
    max_tasks_per_week = int(grouped_by_week.max()) if not grouped_by_week.empty else 1
    return max(1, max_occurrences_per_task), max(1, max_tasks_per_week)


def resolve_preprocessing_caps(
    df_events: pd.DataFrame,
    week_starts: list[pd.Timestamp],
    train_ratio: float,
    max_occurrences_per_task: int | None = None,
    max_tasks_per_week: int | None = None,
    cap_inference_scope: str = CAP_INFERENCE_SCOPE,
) -> dict[str, int | str]:
    split_week_idx = max(int(len(week_starts) * train_ratio), WINDOW_WEEKS + 1)
    train_week_starts = week_starts[:split_week_idx]
    train_weeks_df = df_events[df_events['week_start'].isin(train_week_starts)]

    train_occ, train_tasks = infer_preprocessing_caps(train_weeks_df)
    full_occ, full_tasks = infer_preprocessing_caps(df_events)

    normalized_scope = str(cap_inference_scope or CAP_INFERENCE_SCOPE).strip().lower()
    if normalized_scope not in {'train', 'full_dataset'}:
        raise ValueError(
            "cap_inference_scope debe ser 'train' o 'full_dataset'"
        )

    if normalized_scope == 'train':
        fallback_occ = train_occ
        fallback_tasks = train_tasks
    else:
        fallback_occ = full_occ
        fallback_tasks = full_tasks

    return {
        'cap_inference_scope': normalized_scope,
        'train_max_occurrences_per_task': train_occ,
        'train_max_tasks_per_week': train_tasks,
        'full_max_occurrences_per_task': full_occ,
        'full_max_tasks_per_week': full_tasks,
        'max_occurrences_per_task': _sanitize_cap(max_occurrences_per_task, fallback_occ),
        'max_tasks_per_week': _sanitize_cap(max_tasks_per_week, fallback_tasks),
    }


def denormalize_duration(norm_value: float, dur_min: float, dur_max: float) -> float:
    span = max(dur_max - dur_min, 1e-6)
    return float(np.clip(norm_value, 0.0, 1.0) * span + dur_min)


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    if period <= 0:
        return 0.0, 0.0
    angle = 2.0 * np.pi * float(value) / float(period)
    return float(np.sin(angle)), float(np.cos(angle))


def _normalized_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(np.clip(-(probs * np.log(probs)).sum() / np.log(7.0), 0.0, 1.0))


def _circular_mean_bin(values: list[int] | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return None
    angles = 2.0 * np.pi * arr / num_time_bins()
    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))
    if abs(mean_sin) < 1e-8 and abs(mean_cos) < 1e-8:
        return float(np.mean(arr))
    angle = float(np.arctan2(mean_sin, mean_cos))
    if angle < 0:
        angle += 2.0 * np.pi
    return float(angle * num_time_bins() / (2.0 * np.pi))


def _circular_dispersion(mean_sin: float, mean_cos: float) -> float:
    resultant = float(np.clip(np.hypot(mean_sin, mean_cos), 0.0, 1.0))
    return float(np.clip(1.0 - resultant, 0.0, 1.0))


def base_week_feature_dim(num_tasks: int) -> int:
    per_task_scalar_blocks = 7 * num_tasks
    per_task_day_distribution = 7 * num_tasks
    calendar_features = 7
    return per_task_scalar_blocks + per_task_day_distribution + calendar_features


def occurrence_extra_feature_dim(num_tasks: int) -> int:
    return len(OCC_SEASONAL_LAGS) * num_tasks * 2


def expected_week_feature_dim(num_tasks: int) -> int:
    return base_week_feature_dim(num_tasks) + occurrence_extra_feature_dim(num_tasks)


def expected_history_feature_dim() -> int:
    history_scale_features = len(HISTORY_SCALES) * 5
    global_task_stats = 5
    calendar_context = 7
    recent_frequency_features = len(RECENT_FREQ_SCALES)
    recent_mean_features = 3
    anchor_features = 11
    seasonal_start_anchor_features = len(TMP_SEASONAL_LAGS) * 3 + 3
    recent_offset_features = 1 + len(RECENT_OFFSET_SCALES) + 1 + 2
    recent_same_day_offset_features = 2
    anchor_coverage_features = 2
    decayed_recency_features = 5
    return (
        history_scale_features
        + global_task_stats
        + calendar_context
        + recent_frequency_features
        + recent_mean_features
        + anchor_features
        + seasonal_start_anchor_features
        + recent_offset_features
        + recent_same_day_offset_features
        + anchor_coverage_features
        + decayed_recency_features
    )


def clip_start_bin(start_bin: int | float) -> int:
    return int(np.clip(round(float(start_bin)), 0, num_time_bins() - 1))


def clip_global_day_offset_bins(offset: int | float) -> int:
    return int(np.clip(round(float(offset)), -GLOBAL_DAY_OFFSET_RADIUS_BINS, GLOBAL_DAY_OFFSET_RADIUS_BINS))


def clip_local_start_offset_bins(offset: int | float) -> int:
    return int(np.clip(round(float(offset)), -LOCAL_START_OFFSET_RADIUS_BINS, LOCAL_START_OFFSET_RADIUS_BINS))


def global_day_offset_to_index(offset: int | float) -> int:
    return clip_global_day_offset_bins(offset) + GLOBAL_DAY_OFFSET_RADIUS_BINS


def global_day_index_to_offset(index: int | float) -> int:
    return int(round(float(index))) - GLOBAL_DAY_OFFSET_RADIUS_BINS


def local_start_offset_to_index(offset: int | float) -> int:
    return clip_local_start_offset_bins(offset) + LOCAL_START_OFFSET_RADIUS_BINS


def local_start_index_to_offset(index: int | float) -> int:
    return int(round(float(index))) - LOCAL_START_OFFSET_RADIUS_BINS


def _build_events(df: pd.DataFrame, task_to_id: dict[str, int], dur_min: float, dur_max: float) -> pd.DataFrame:
    out = df.copy()
    out['task_id'] = out['task_name'].map(task_to_id)
    out['week_start'] = out['start_time'].map(_week_start)
    out['day_of_week'] = out['start_time'].dt.dayofweek.astype(int)
    out['hour'] = out['start_time'].dt.hour.astype(int)
    out['minute'] = out['start_time'].dt.minute.astype(int)
    out['minute_bin'] = (out['minute'] // BIN_MINUTES).astype(int)
    out['start_bin'] = (
        out['day_of_week'] * bins_per_day()
        + out['hour'] * (60 // BIN_MINUTES)
        + out['minute_bin']
    ).astype(int)
    out['duration_norm'] = out['duration_minutes'].map(lambda x: _duration_to_norm(x, dur_min, dur_max))
    return out


def _continuous_week_starts(df_events: pd.DataFrame) -> list[pd.Timestamp]:
    first_week = df_events['week_start'].min()
    last_week = df_events['week_start'].max()
    return list(pd.date_range(start=first_week, end=last_week, freq='7D', tz=first_week.tz))


def _compute_week_features(
    grouped: pd.DataFrame,
    week_index: int,
    week_start: pd.Timestamp,
    num_tasks: int,
    max_count_cap: int,
    max_tasks_per_week: int,
) -> WeekRecord:
    counts = np.zeros(num_tasks, dtype=np.float32)
    mean_start_sin = np.zeros(num_tasks, dtype=np.float32)
    mean_start_cos = np.zeros(num_tasks, dtype=np.float32)
    mean_duration_norm = np.zeros(num_tasks, dtype=np.float32)
    start_circular_dispersion = np.zeros(num_tasks, dtype=np.float32)
    active_days_norm = np.zeros(num_tasks, dtype=np.float32)
    day_multimodality_norm = np.zeros(num_tasks, dtype=np.float32)
    day_distribution = np.zeros((num_tasks, 7), dtype=np.float32)
    events_by_task: dict[int, list[EventRecord]] = {task_id: [] for task_id in range(num_tasks)}

    if not grouped.empty:
        grouped = grouped.sort_values(['task_id', 'start_bin', 'start_time']).reset_index(drop=True)

        for task_id, task_df in grouped.groupby('task_id'):
            task_df = task_df.sort_values(['start_bin', 'start_time'])
            n = int(len(task_df))
            counts[task_id] = float(min(n, max_count_cap))
            angles = 2.0 * np.pi * task_df['start_bin'].to_numpy(dtype=np.float32) / num_time_bins()
            mean_start_sin[task_id] = float(np.mean(np.sin(angles)))
            mean_start_cos[task_id] = float(np.mean(np.cos(angles)))
            mean_duration_norm[task_id] = float(task_df['duration_norm'].mean())
            start_circular_dispersion[task_id] = _circular_dispersion(mean_start_sin[task_id], mean_start_cos[task_id])

            day_counts = np.bincount(task_df['day_of_week'].to_numpy(dtype=np.int64), minlength=7).astype(np.float32)
            if day_counts.sum() > 0:
                day_probs = day_counts / day_counts.sum()
                day_distribution[task_id] = day_probs
                active_days_norm[task_id] = float(np.count_nonzero(day_counts) / 7.0)
                day_multimodality_norm[task_id] = _normalized_entropy(day_probs)

            events_by_task[task_id] = [
                EventRecord(
                    task_id=int(row.task_id),
                    task_name=str(row.task_name),
                    start_bin=int(row.start_bin),
                    duration_minutes=float(row.duration_minutes),
                    start_time=row.start_time,
                    end_time=row.end_time,
                )
                for row in task_df.itertuples(index=False)
            ]

    iso = week_start.isocalendar()
    week_of_year_sin, week_of_year_cos = _cyclical_encode(int(iso.week) - 1, 52.0)
    month_sin, month_cos = _cyclical_encode(int(week_start.month) - 1, 12.0)
    day_of_year_sin, day_of_year_cos = _cyclical_encode(int(week_start.dayofyear) - 1, 366.0)
    total_tasks_norm = float(np.clip(np.sum(counts) / max(max_tasks_per_week, 1), 0.0, 1.0))

    return WeekRecord(
        week_index=week_index,
        week_start=week_start,
        counts=counts,
        mean_start_sin=mean_start_sin,
        mean_start_cos=mean_start_cos,
        mean_duration_norm=mean_duration_norm,
        start_circular_dispersion=start_circular_dispersion,
        active_days_norm=active_days_norm,
        day_multimodality_norm=day_multimodality_norm,
        day_distribution=day_distribution,
        total_tasks_norm=total_tasks_norm,
        week_of_year_sin=week_of_year_sin,
        week_of_year_cos=week_of_year_cos,
        month_sin=month_sin,
        month_cos=month_cos,
        day_of_year_sin=day_of_year_sin,
        day_of_year_cos=day_of_year_cos,
        events_by_task=events_by_task,
    )


def week_to_feature_vector(week: WeekRecord) -> np.ndarray:
    return np.concatenate([
        week.counts,
        week.mean_start_sin,
        week.mean_start_cos,
        week.mean_duration_norm,
        week.start_circular_dispersion,
        week.active_days_norm,
        week.day_multimodality_norm,
        week.day_distribution.reshape(-1),
        np.array([
            week.total_tasks_norm,
            week.week_of_year_sin,
            week.week_of_year_cos,
            week.month_sin,
            week.month_cos,
            week.day_of_year_sin,
            week.day_of_year_cos,
        ], dtype=np.float32),
    ]).astype(np.float32)


def _lag_week_record(weeks: list[WeekRecord], target_week_index: int, lag_weeks: int) -> WeekRecord | None:
    source_idx = int(target_week_index) - int(lag_weeks)
    if 0 <= source_idx < len(weeks):
        return weeks[source_idx]
    return None


def seasonal_lag_count_features(weeks: list[WeekRecord], target_week_index: int, num_tasks: int) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for lag in OCC_SEASONAL_LAGS:
        source_week = _lag_week_record(weeks, target_week_index, lag)
        if source_week is None:
            counts = np.zeros(num_tasks, dtype=np.float32)
            availability = np.zeros(num_tasks, dtype=np.float32)
        else:
            counts = source_week.counts.astype(np.float32, copy=False)
            availability = np.ones(num_tasks, dtype=np.float32)
        blocks.extend([counts, availability])
    return np.concatenate(blocks, axis=0).astype(np.float32) if blocks else np.zeros(0, dtype=np.float32)


def build_context_sequence_features(weeks: list[WeekRecord], target_week_index: int, window_weeks: int, num_tasks: int) -> np.ndarray:
    cache_key = (id(weeks), int(target_week_index), int(window_weeks), int(num_tasks))
    cached = _CONTEXT_SEQUENCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    context_weeks = weeks[max(0, target_week_index - window_weeks):target_week_index]
    if not context_weeks:
        seq = np.zeros((window_weeks, base_week_feature_dim(num_tasks)), dtype=np.float32)
    else:
        seq = np.stack([week_to_feature_vector(week) for week in context_weeks]).astype(np.float32)
        if seq.shape[0] < window_weeks:
            pad = np.zeros((window_weeks - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
    lag_features = seasonal_lag_count_features(weeks, target_week_index, num_tasks)
    if lag_features.size == 0:
        out = seq.astype(np.float32)
    else:
        repeated_lags = np.repeat(lag_features[None, :], seq.shape[0], axis=0)
        out = np.concatenate([seq, repeated_lags], axis=1).astype(np.float32)

    _CONTEXT_SEQUENCE_CACHE[cache_key] = out
    return out


def _assignments_to_slot_bins(assignments_by_week: list[list[tuple[int, EventRecord]]]) -> dict[int, list[int]]:
    slot_bins: dict[int, list[int]] = {}
    for assignments in assignments_by_week:
        for slot_id, event in assignments:
            slot_bins.setdefault(int(slot_id), []).append(int(event.start_bin))
    return slot_bins


def _prepare_task_temporal_cache(
    weeks: list[WeekRecord],
    target_week_index: int,
    task_id: int,
    max_occurrences_per_task: int,
) -> TaskTemporalCache:
    cache_key = (id(weeks), int(target_week_index), int(task_id), int(max_occurrences_per_task))
    cached = _TASK_TEMPORAL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    history = _history_slice(weeks, target_week_index)
    anchor_history = _anchor_history_slice(weeks, target_week_index)
    target_week_start = _infer_target_week_start(weeks, target_week_index)

    if history:
        counts_per_week = np.array([week.counts[task_id] for week in history], dtype=np.float32)
        sin_per_week = np.array([week.mean_start_sin[task_id] for week in history], dtype=np.float32)
        cos_per_week = np.array([week.mean_start_cos[task_id] for week in history], dtype=np.float32)
        dur_per_week = np.array([week.mean_duration_norm[task_id] for week in history], dtype=np.float32)
    else:
        counts_per_week = np.zeros(0, dtype=np.float32)
        sin_per_week = np.zeros(0, dtype=np.float32)
        cos_per_week = np.zeros(0, dtype=np.float32)
        dur_per_week = np.zeros(0, dtype=np.float32)

    prototypes = _build_occurrence_prototypes(anchor_history, task_id, max_occurrences_per_task)
    recent_assignments = [
        _assign_events_to_prototype_slots(week.events_by_task.get(task_id, []), prototypes, max_occurrences_per_task)
        for week in history
    ]
    anchor_assignments = [
        _assign_events_to_prototype_slots(week.events_by_task.get(task_id, []), prototypes, max_occurrences_per_task)
        for week in anchor_history
    ]

    lag_slot_to_bin: dict[int, dict[int, int]] = {}
    for lag_weeks in TMP_SEASONAL_LAGS:
        source_week = _lag_week_record(weeks, target_week_index, lag_weeks)
        if source_week is None:
            lag_slot_to_bin[int(lag_weeks)] = {}
            continue
        assignments = _assign_events_to_prototype_slots(source_week.events_by_task.get(task_id, []), prototypes, max_occurrences_per_task)
        lag_slot_to_bin[int(lag_weeks)] = {int(slot_id): int(event.start_bin) for slot_id, event in assignments}

    task_cache = TaskTemporalCache(
        history=history,
        anchor_history=anchor_history,
        target_week_start=target_week_start,
        counts_per_week=counts_per_week,
        sin_per_week=sin_per_week,
        cos_per_week=cos_per_week,
        dur_per_week=dur_per_week,
        recent_all_bins=_task_history_bins(history, task_id),
        anchor_all_bins=_task_history_bins(anchor_history, task_id),
        prototypes=prototypes,
        recent_slot_bins=_assignments_to_slot_bins(recent_assignments),
        anchor_slot_bins=_assignments_to_slot_bins(anchor_assignments),
        lag_slot_to_bin=lag_slot_to_bin,
    )
    _TASK_TEMPORAL_CACHE[cache_key] = task_cache
    return task_cache


def _infer_target_week_start(weeks: list[WeekRecord], target_week_index: int) -> pd.Timestamp:
    if 0 <= target_week_index < len(weeks):
        return weeks[target_week_index].week_start
    if target_week_index == len(weeks):
        return weeks[-1].week_start + pd.Timedelta(days=7)
    return weeks[0].week_start + pd.Timedelta(days=7 * target_week_index)


def _history_slice(weeks: list[WeekRecord], target_week_index: int) -> list[WeekRecord]:
    start = max(0, target_week_index - WINDOW_WEEKS)
    return weeks[start:target_week_index]


def _anchor_history_slice(weeks: list[WeekRecord], target_week_index: int) -> list[WeekRecord]:
    start = max(0, target_week_index - ANCHOR_LOOKBACK_WEEKS)
    return weeks[start:target_week_index]


def _recent_count_frequency(history: list[WeekRecord], task_id: int, k: int, max_occurrences_per_task: int) -> float:
    if not history:
        return 0.0
    window = history[-min(k, len(history)):]
    total = float(sum(week.counts[task_id] for week in window))
    denom = float(len(window) * max(max_occurrences_per_task, 1))
    return float(np.clip(total / max(denom, 1e-6), 0.0, 1.0))


def _days_since_last_occurrence(history: list[WeekRecord], task_id: int, target_week_start: pd.Timestamp) -> float:
    last_start_time: pd.Timestamp | None = None
    for week in history:
        for event in week.events_by_task.get(task_id, []):
            if last_start_time is None or event.start_time > last_start_time:
                last_start_time = event.start_time
    if last_start_time is None:
        return float(max(12 * 7, WINDOW_WEEKS * 7))
    delta_days = (target_week_start - last_start_time).total_seconds() / (24.0 * 3600.0)
    return float(max(delta_days, 0.0))


def _recent_mean_start_bin(history: list[WeekRecord], task_id: int) -> float | None:
    values = [int(event.start_bin) for week in history for event in week.events_by_task.get(task_id, [])]
    return _circular_mean_bin(values)


def _recent_mean_duration_norm(history: list[WeekRecord], task_id: int, dur_min: float, dur_max: float) -> float:
    values = [float(event.duration_minutes) for week in history for event in week.events_by_task.get(task_id, [])]
    if not values:
        return 0.0
    return _duration_to_norm(float(np.mean(values)), dur_min, dur_max)



def _task_history_bins(history: list[WeekRecord], task_id: int) -> list[int]:
    return [int(event.start_bin) for week in history for event in week.events_by_task.get(task_id, [])]


def _median_bin(values: list[int]) -> int:
    if not values:
        return 0
    return clip_start_bin(int(round(float(np.median(values)))))


def _group_bins_by_day(values: list[int]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    for value in values:
        grouped.setdefault(int(value // bins_per_day()), []).append(int(value))
    return grouped


def _dominant_day(values: list[int]) -> tuple[int | None, float]:
    if not values:
        return None, 0.0
    grouped = _group_bins_by_day(values)
    day, bins = max(grouped.items(), key=lambda item: (len(item[1]), item[0]))
    return int(day), float(len(bins) / max(len(values), 1))


def _select_day_cluster(values: list[int], preferred_day: int | None = None) -> tuple[list[int], int | None, float]:
    if not values:
        return [], None, 0.0
    grouped = _group_bins_by_day(values)
    if preferred_day is not None and preferred_day in grouped:
        selected_day = int(preferred_day)
    else:
        selected_day, bins = max(grouped.items(), key=lambda item: (len(item[1]), item[0]))
        selected_day = int(selected_day)
    selected = grouped[selected_day]
    return list(selected), selected_day, float(len(selected) / max(len(values), 1))


def _recency_weights(length: int, base: float = RECENCY_DECAY_BASE) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    exponents = np.arange(length - 1, -1, -1, dtype=np.float32)
    weights = np.power(np.float32(base), exponents, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0:
        return np.full(length, 1.0 / length, dtype=np.float32)
    return (weights / total).astype(np.float32)


def _decayed_history_features(
    counts_per_week: np.ndarray,
    sin_per_week: np.ndarray,
    cos_per_week: np.ndarray,
    dur_per_week: np.ndarray,
    max_occurrences_per_task: int,
) -> list[float]:
    if len(counts_per_week) == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    weights = _recency_weights(len(counts_per_week))
    mask = (counts_per_week > 0).astype(np.float32)
    active_weights = weights * mask
    active_total = float(active_weights.sum())
    if active_total > 0:
        active_weights = active_weights / active_total
        decayed_sin = float(np.dot(active_weights, sin_per_week))
        decayed_cos = float(np.dot(active_weights, cos_per_week))
        decayed_dur = float(np.dot(active_weights, dur_per_week))
    else:
        decayed_sin = 0.0
        decayed_cos = 0.0
        decayed_dur = 0.0

    decayed_count = float(np.dot(weights, counts_per_week) / max(max_occurrences_per_task, 1))
    decayed_presence = float(np.dot(weights, mask))
    return [decayed_count, decayed_presence, decayed_sin, decayed_cos, decayed_dur]


def _circular_bin_distance(a: int | float, b: int | float, period: int | None = None) -> int:
    period = num_time_bins() if period is None else max(int(period), 1)
    diff = abs(int(round(float(a))) - int(round(float(b))))
    return int(min(diff, period - diff))


def _fit_start_bin_prototypes(values: list[int], k: int, max_iters: int = 8) -> list[int]:
    if not values or k <= 0:
        return []
    arr = np.asarray(sorted(int(v) for v in values), dtype=np.int32)
    if arr.size == 0:
        return []
    k = max(1, min(int(k), int(arr.size)))
    init_indices = np.linspace(0, arr.size - 1, num=k, dtype=np.int64)
    centers = arr[init_indices].astype(np.int32)

    for _ in range(max_iters):
        dist = np.zeros((arr.size, len(centers)), dtype=np.float32)
        for j, center in enumerate(centers.tolist()):
            direct = np.abs(arr - int(center))
            dist[:, j] = np.minimum(direct, num_time_bins() - direct)
        assign = dist.argmin(axis=1)

        new_centers: list[int] = []
        for j in range(len(centers)):
            cluster = arr[assign == j]
            if cluster.size == 0:
                new_centers.append(int(centers[j]))
                continue
            candidates = np.unique(cluster)
            best_center = min(
                candidates.tolist(),
                key=lambda candidate: (
                    float(np.minimum(np.abs(cluster - int(candidate)), num_time_bins() - np.abs(cluster - int(candidate))).sum()),
                    int(candidate),
                ),
            )
            new_centers.append(int(best_center))

        deduped = sorted(dict.fromkeys(new_centers))
        if len(deduped) < k:
            for value in arr.tolist():
                if int(value) not in deduped:
                    deduped.append(int(value))
                if len(deduped) >= k:
                    break
            deduped = sorted(dict.fromkeys(deduped))[:k]

        updated = np.asarray(deduped, dtype=np.int32)
        if np.array_equal(centers, updated):
            break
        centers = updated

    return [int(v) for v in sorted(dict.fromkeys(centers.tolist()))]


def _build_occurrence_prototypes(history: list[WeekRecord], task_id: int, max_occurrences_per_task: int) -> list[OccurrencePrototype]:
    all_bins: list[int] = []
    weekly_max = 0
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))[:max_occurrences_per_task]
        weekly_max = max(weekly_max, len(events))
        all_bins.extend(int(event.start_bin) for event in events)

    if not all_bins:
        return []

    k = max(1, min(max_occurrences_per_task, weekly_max or 1, len(set(all_bins))))
    centers = _fit_start_bin_prototypes(all_bins, k)
    if not centers:
        return []

    supports = [0 for _ in centers]
    for start_bin in all_bins:
        nearest_idx = min(range(len(centers)), key=lambda idx: (_circular_bin_distance(start_bin, centers[idx]), idx))
        supports[nearest_idx] += 1

    ordered = sorted(range(len(centers)), key=lambda idx: (centers[idx], -supports[idx], idx))
    return [
        OccurrencePrototype(slot_id=slot_id, center_bin=int(centers[idx]), support=float(supports[idx]))
        for slot_id, idx in enumerate(ordered)
    ]


def _nearest_occurrence_prototype(prototypes: list[OccurrencePrototype], occurrence_slot: int) -> OccurrencePrototype | None:
    if not prototypes:
        return None
    if 0 <= int(occurrence_slot) < len(prototypes):
        return prototypes[int(occurrence_slot)]
    return min(
        prototypes,
        key=lambda proto: (abs(int(proto.slot_id) - int(occurrence_slot)), -float(proto.support), int(proto.center_bin), int(proto.slot_id)),
    )


def _assign_events_to_prototype_slots(
    events: list[EventRecord],
    prototypes: list[OccurrencePrototype],
    max_occurrences_per_task: int,
) -> list[tuple[int, EventRecord]]:
    limited_events = sorted(events, key=lambda e: (e.start_bin, e.start_time))[:max_occurrences_per_task]
    if not limited_events:
        return []
    if not prototypes:
        return [(slot_id, event) for slot_id, event in enumerate(limited_events)]

    n_events = len(limited_events)
    n_slots = len(prototypes)
    cost = np.zeros((n_events, n_slots), dtype=np.float32)
    for i, event in enumerate(limited_events):
        for j, prototype in enumerate(prototypes):
            cost[i, j] = float(_circular_bin_distance(int(event.start_bin), int(prototype.center_bin)))

    row_ind, col_ind = linear_sum_assignment(cost)
    assigned_rows: set[int] = set()
    assignments: list[tuple[int, EventRecord]] = []

    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if row < n_events and col < n_slots:
            assigned_rows.add(int(row))
            assignments.append((int(prototypes[col].slot_id), limited_events[row]))

    overflow_slot = len(prototypes)
    for row_idx, event in enumerate(limited_events):
        if row_idx in assigned_rows:
            continue
        assignments.append((int(overflow_slot), event))
        overflow_slot += 1

    assignments.sort(key=lambda item: (int(item[0]), int(item[1].start_bin), item[1].start_time))
    return assignments[:max_occurrences_per_task]


def build_target_occurrence_slot_assignments(
    weeks: list[WeekRecord],
    target_week_index: int,
    task_id: int,
    max_occurrences_per_task: int | None = None,
) -> list[tuple[int, EventRecord]]:
    max_occurrences_per_task = _sanitize_cap(max_occurrences_per_task, DEFAULT_MAX_OCCURRENCES_PER_TASK)
    task_cache = _prepare_task_temporal_cache(weeks, target_week_index, task_id, max_occurrences_per_task)
    target_week = weeks[target_week_index]
    return _assign_events_to_prototype_slots(target_week.events_by_task.get(task_id, []), task_cache.prototypes, max_occurrences_per_task)


def build_prediction_occurrence_slots(
    weeks: list[WeekRecord],
    target_week_index: int,
    task_id: int,
    predicted_count: int,
    max_occurrences_per_task: int | None = None,
) -> list[int]:
    max_occurrences_per_task = _sanitize_cap(max_occurrences_per_task, DEFAULT_MAX_OCCURRENCES_PER_TASK)
    predicted_count = int(np.clip(predicted_count, 0, max_occurrences_per_task))
    if predicted_count <= 0:
        return []

    task_cache = _prepare_task_temporal_cache(weeks, target_week_index, task_id, max_occurrences_per_task)
    ranked_slots = [
        int(proto.slot_id)
        for proto in sorted(task_cache.prototypes, key=lambda proto: (-float(proto.support), int(proto.center_bin), int(proto.slot_id)))
    ]
    selected = ranked_slots[:predicted_count]
    next_slot = len(task_cache.prototypes)
    while len(selected) < predicted_count and next_slot < max_occurrences_per_task:
        selected.append(int(next_slot))
        next_slot += 1
    return sorted(selected)


def _task_occurrence_history_bins(
    history: list[WeekRecord],
    task_id: int,
    occurrence_slot: int,
    prototypes: list[OccurrencePrototype],
    max_occurrences_per_task: int,
) -> list[int]:
    out: list[int] = []
    for week in history:
        assignments = _assign_events_to_prototype_slots(week.events_by_task.get(task_id, []), prototypes, max_occurrences_per_task)
        for slot_id, event in assignments:
            if int(slot_id) == int(occurrence_slot):
                out.append(int(event.start_bin))
    return out


def _compute_anchor_start_bin(
    task_cache: TaskTemporalCache,
    occurrence_slot: int,
) -> tuple[int, float, float, float, float, list[OccurrencePrototype]]:
    recent_history = task_cache.history
    anchor_history = task_cache.anchor_history
    prototypes = task_cache.prototypes
    same_occ_recent = task_cache.recent_slot_bins.get(int(occurrence_slot), [])
    same_occ_long = task_cache.anchor_slot_bins.get(int(occurrence_slot), [])
    recent_all_bins = task_cache.recent_all_bins
    anchor_all_bins = task_cache.anchor_all_bins

    preferred_day, _ = _dominant_day(same_occ_recent)
    if preferred_day is None:
        preferred_day, _ = _dominant_day(same_occ_long)
    if preferred_day is None:
        preferred_day, _ = _dominant_day(recent_all_bins)
    if preferred_day is None:
        preferred_day, _ = _dominant_day(anchor_all_bins)

    if same_occ_long:
        long_day_bins, selected_day, long_day_ratio = _select_day_cluster(same_occ_long, preferred_day)
        base_anchor = _median_bin(long_day_bins if long_day_bins else same_occ_long)

        recent_bins, _, recent_day_ratio = _select_day_cluster(same_occ_recent, selected_day) if same_occ_recent else ([], None, 0.0)
        if recent_bins:
            recent_anchor = _median_bin(recent_bins)
            anchor = clip_start_bin(round(0.75 * base_anchor + 0.25 * recent_anchor))
            recent_support_count = len(recent_bins)
            proto_quality = 0.5 + 0.5 * recent_day_ratio
        elif same_occ_recent:
            recent_anchor = _median_bin(same_occ_recent)
            anchor = clip_start_bin(round(0.85 * base_anchor + 0.15 * recent_anchor))
            recent_support_count = len(same_occ_recent)
            proto_quality = 0.5
        else:
            anchor = base_anchor
            recent_support_count = 0
            proto_quality = 0.5 + 0.5 * long_day_ratio

        recent_support = float(np.clip(recent_support_count / max(len(recent_history), 1), 0.0, 1.0))
        long_support = float(np.clip(len(long_day_bins if long_day_bins else same_occ_long) / max(len(anchor_history), 1), 0.0, 1.0))
        return anchor, recent_support, float(len(same_occ_recent)), long_support, float(np.clip(proto_quality, 0.0, 1.0)), prototypes

    nearest_proto = _nearest_occurrence_prototype(prototypes, occurrence_slot)
    if nearest_proto is not None:
        proto_bins = task_cache.anchor_slot_bins.get(int(nearest_proto.slot_id), [])
        if proto_bins:
            proto_day_bins, _, proto_day_ratio = _select_day_cluster(proto_bins, preferred_day)
            anchor = _median_bin(proto_day_bins if proto_day_bins else proto_bins)
            long_support = float(np.clip(len(proto_day_bins if proto_day_bins else proto_bins) / max(len(anchor_history), 1), 0.0, 1.0))
            distance_penalty = 1.0 / (1.0 + float(abs(int(nearest_proto.slot_id) - int(occurrence_slot))))
            proto_match = distance_penalty * (0.5 + 0.5 * proto_day_ratio)
            return anchor, 0.0, 0.0, long_support, float(np.clip(proto_match, 0.0, 1.0)), prototypes

    if anchor_all_bins:
        all_day_bins, _, all_day_ratio = _select_day_cluster(anchor_all_bins, preferred_day)
        anchor = _median_bin(all_day_bins if all_day_bins else anchor_all_bins)
        long_support = float(
            np.clip(
                len(all_day_bins if all_day_bins else anchor_all_bins)
                / max(len(anchor_history) * max(len(prototypes), 1), 1),
                0.0,
                1.0,
            )
        )
        return anchor, 0.0, 0.0, long_support, float(np.clip(0.25 + 0.5 * all_day_ratio, 0.0, 1.0)), prototypes

    return 0, 0.0, 0.0, 0.0, 0.0, prototypes

def _align_to_day(start_bin: int, preferred_day: int | None) -> int:
    if preferred_day is None:
        return clip_start_bin(start_bin)
    local_bin = int(start_bin % bins_per_day())
    return clip_start_bin(int(preferred_day) * bins_per_day() + local_bin)


def _register_anchor_candidate(pool: dict[int, dict[str, float | str]], start_bin: int | None, weight: float, source: str, preferred_day: int | None = None) -> None:
    if start_bin is None or weight <= 0.0:
        return
    aligned = _align_to_day(int(start_bin), preferred_day if source.startswith('aligned_') else None)
    entry = pool.get(int(aligned))
    if entry is None:
        pool[int(aligned)] = {'start_bin': int(aligned), 'weight': float(weight), 'source': str(source)}
        return
    entry['weight'] = float(entry['weight']) + float(weight)


def _build_anchor_bundle(
    task_cache: TaskTemporalCache,
    occurrence_slot: int,
    predicted_count: int | None = None,
) -> tuple[int, float, float, float, float, list[OccurrencePrototype], list[int], list[float]]:
    recent_history = task_cache.history
    anchor_history = task_cache.anchor_history
    prototypes = task_cache.prototypes
    same_occ_recent = task_cache.recent_slot_bins.get(int(occurrence_slot), [])
    same_occ_long = task_cache.anchor_slot_bins.get(int(occurrence_slot), [])
    recent_all_bins = task_cache.recent_all_bins
    anchor_all_bins = task_cache.anchor_all_bins

    preferred_day, preferred_ratio = _dominant_day(same_occ_recent)
    if preferred_day is None:
        preferred_day, preferred_ratio = _dominant_day(same_occ_long)
    if preferred_day is None:
        preferred_day, preferred_ratio = _dominant_day(recent_all_bins)
    if preferred_day is None:
        preferred_day, preferred_ratio = _dominant_day(anchor_all_bins)

    pool: dict[int, dict[str, float | str]] = {}

    recent_cluster, _, recent_day_ratio = _select_day_cluster(same_occ_recent, preferred_day) if same_occ_recent else ([], None, 0.0)
    if recent_cluster:
        _register_anchor_candidate(pool, _median_bin(recent_cluster), 4.0 * (0.5 + recent_day_ratio), 'same_occ_recent')

    long_cluster, _, long_day_ratio = _select_day_cluster(same_occ_long, preferred_day) if same_occ_long else ([], None, 0.0)
    if long_cluster:
        _register_anchor_candidate(pool, _median_bin(long_cluster), 3.0 * (0.5 + long_day_ratio), 'same_occ_long')

    lag52 = task_cache.lag_slot_to_bin.get(52, {}).get(int(occurrence_slot))
    lag26 = task_cache.lag_slot_to_bin.get(26, {}).get(int(occurrence_slot))
    _register_anchor_candidate(pool, lag52, 2.8, 'lag52')
    _register_anchor_candidate(pool, lag26, 2.0, 'lag26')

    hist_same_occ = _median_bin(same_occ_long) if same_occ_long else None
    _register_anchor_candidate(pool, hist_same_occ, 1.8, 'same_occ_median')

    if recent_all_bins:
        recent_all_cluster, _, recent_all_ratio = _select_day_cluster(recent_all_bins, preferred_day)
        _register_anchor_candidate(pool, _median_bin(recent_all_cluster if recent_all_cluster else recent_all_bins), 1.4 * (0.5 + recent_all_ratio), 'recent_all')

    if anchor_all_bins:
        anchor_all_cluster, _, anchor_all_ratio = _select_day_cluster(anchor_all_bins, preferred_day)
        _register_anchor_candidate(pool, _median_bin(anchor_all_cluster if anchor_all_cluster else anchor_all_bins), 1.2 * (0.5 + anchor_all_ratio), 'anchor_all')

    nearest_proto = _nearest_occurrence_prototype(prototypes, occurrence_slot)
    if nearest_proto is not None:
        proto_weight = 1.5 / (1.0 + abs(int(nearest_proto.slot_id) - int(occurrence_slot)))
        _register_anchor_candidate(pool, int(nearest_proto.center_bin), proto_weight, 'prototype')

    ordered_protos = sorted(prototypes, key=lambda proto: int(proto.center_bin))
    if predicted_count is not None and predicted_count > 0 and ordered_protos:
        slot_progress = float(np.clip(int(occurrence_slot) / max(int(predicted_count) - 1, 1), 0.0, 1.0)) if int(predicted_count) > 1 else 0.0
        rank_idx = int(round(slot_progress * max(len(ordered_protos) - 1, 0)))
        schedule_proto = ordered_protos[min(rank_idx, len(ordered_protos) - 1)]
        _register_anchor_candidate(pool, int(schedule_proto.center_bin), 1.3, 'count_conditioned_prototype')

    if not pool:
        fallback_anchor, recent_support, same_occ_support, long_support, proto_match, prototypes = _compute_anchor_start_bin(task_cache, occurrence_slot)
        return fallback_anchor, recent_support, same_occ_support, long_support, proto_match, prototypes, [int(fallback_anchor)], [1.0]

    ranked = sorted(
        pool.values(),
        key=lambda item: (-float(item['weight']), int(item['start_bin'])),
    )[: max(1, int(TEMPORAL_NUM_ANCHOR_CANDIDATES))]
    candidate_bins = [int(item['start_bin']) for item in ranked]
    raw_weights = np.asarray([float(item['weight']) for item in ranked], dtype=np.float32)
    weight_sum = float(raw_weights.sum())
    normalized_weights = (raw_weights / weight_sum).tolist() if weight_sum > 0 else [1.0 / len(candidate_bins) for _ in candidate_bins]

    best_anchor = int(candidate_bins[0])
    recent_support = float(np.clip(len(same_occ_recent) / max(len(recent_history), 1), 0.0, 1.0))
    long_support = float(np.clip(len(same_occ_long) / max(len(anchor_history), 1), 0.0, 1.0))
    proto_match = float(np.clip(normalized_weights[0] + 0.25 * preferred_ratio, 0.0, 1.0))
    return best_anchor, recent_support, float(len(same_occ_recent)), long_support, proto_match, prototypes, candidate_bins, normalized_weights


def _same_occurrence_start_bin_at_lag(
    weeks: list[WeekRecord],
    target_week_index: int,
    task_id: int,
    occurrence_slot: int,
    lag_weeks: int,
    prototypes: list[OccurrencePrototype],
    max_occurrences_per_task: int,
) -> int | None:
    source_week = _lag_week_record(weeks, target_week_index, lag_weeks)
    if source_week is None:
        return None
    assignments = _assign_events_to_prototype_slots(source_week.events_by_task.get(task_id, []), prototypes, max_occurrences_per_task)
    for slot_id, event in assignments:
        if int(slot_id) == int(occurrence_slot):
            return int(event.start_bin)
    return None


def _same_occurrence_historical_median_bin(
    history: list[WeekRecord],
    task_id: int,
    occurrence_slot: int,
    prototypes: list[OccurrencePrototype],
    max_occurrences_per_task: int,
) -> int | None:
    same_occ_bins = _task_occurrence_history_bins(history, task_id, occurrence_slot, prototypes, max_occurrences_per_task)
    if same_occ_bins:
        return _median_bin(same_occ_bins)
    all_bins = _task_history_bins(history, task_id)
    if all_bins:
        return _median_bin(all_bins)
    return None


def _append_start_bin_feature(feats: list[float], start_bin: int | None) -> None:
    if start_bin is None:
        feats.extend([0.0, 0.0, 0.0])
        return
    s, c = _cyclical_encode(int(start_bin), num_time_bins())
    feats.extend([s, c, 1.0])


def _collect_recent_offsets(
    history: list[WeekRecord],
    task_id: int,
    occurrence_slot: int,
    anchor_start_bin: int,
    prototypes: list[OccurrencePrototype],
    max_occurrences_per_task: int,
) -> list[int]:
    out: list[int] = []
    for week in history:
        assignments = _assign_events_to_prototype_slots(week.events_by_task.get(task_id, []), prototypes, max_occurrences_per_task)
        for slot_id, event in assignments:
            if int(slot_id) == int(occurrence_slot):
                out.append(int(event.start_bin) - int(anchor_start_bin))
                break
    return out


def _collect_recent_same_day_offsets(
    history: list[WeekRecord],
    task_id: int,
    occurrence_slot: int,
    anchor_start_bin: int,
    prototypes: list[OccurrencePrototype],
    max_occurrences_per_task: int,
) -> list[int]:
    anchor_day = int(anchor_start_bin // bins_per_day())
    out: list[int] = []
    for week in history:
        assignments = _assign_events_to_prototype_slots(week.events_by_task.get(task_id, []), prototypes, max_occurrences_per_task)
        for slot_id, event in assignments:
            if int(slot_id) != int(occurrence_slot):
                continue
            if int(event.start_bin // bins_per_day()) == anchor_day:
                out.append(int(event.start_bin) - int(anchor_start_bin))
            break
    return out


def build_temporal_context(
    weeks: list[WeekRecord],
    target_week_index: int,
    task_id: int,
    occurrence_slot: int,
    duration_min: float | None = None,
    duration_max: float | None = None,
    max_occurrences_per_task: int | None = None,
    predicted_count: int | None = None,
) -> TemporalContext:
    duration_min = 0.0 if duration_min is None else duration_min
    duration_max = 1.0 if duration_max is None else duration_max
    max_occurrences_per_task = _sanitize_cap(max_occurrences_per_task, DEFAULT_MAX_OCCURRENCES_PER_TASK)

    task_cache = _prepare_task_temporal_cache(weeks, target_week_index, task_id, max_occurrences_per_task)
    history = task_cache.history
    anchor_history = task_cache.anchor_history
    target_week_start = task_cache.target_week_start
    counts_per_week = task_cache.counts_per_week
    sin_per_week = task_cache.sin_per_week
    cos_per_week = task_cache.cos_per_week
    dur_per_week = task_cache.dur_per_week

    anchor_start_bin, anchor_support, same_occ_support, anchor_long_support, anchor_proto_match, prototypes, anchor_candidates, anchor_candidate_weights = _build_anchor_bundle(
        task_cache,
        occurrence_slot,
        predicted_count=predicted_count,
    )
    anchor_day = int(anchor_start_bin // bins_per_day())
    anchor_minute_of_day = int(anchor_start_bin % bins_per_day())
    anchor_week_sin, anchor_week_cos = _cyclical_encode(anchor_start_bin, num_time_bins())
    anchor_day_sin, anchor_day_cos = _cyclical_encode(anchor_day, 7.0)
    anchor_time_sin, anchor_time_cos = _cyclical_encode(anchor_minute_of_day, bins_per_day())

    same_occ_recent = task_cache.recent_slot_bins.get(int(occurrence_slot), [])
    recent_offsets = [int(start_bin) - int(anchor_start_bin) for start_bin in same_occ_recent]
    anchor_day_for_offsets = int(anchor_start_bin // bins_per_day())
    recent_same_day_offsets = [
        int(start_bin) - int(anchor_start_bin)
        for start_bin in same_occ_recent
        if int(start_bin // bins_per_day()) == anchor_day_for_offsets
    ]

    feats: list[float] = []
    for scale in HISTORY_SCALES:
        window_counts = counts_per_week[-scale:]
        scale_len = max(len(window_counts), 1)
        feats.append(float(window_counts.sum() / (max_occurrences_per_task * scale_len)))
        feats.append(float((window_counts > 0).mean()) if len(window_counts) > 0 else 0.0)
        window_sin = sin_per_week[-scale:]
        window_cos = cos_per_week[-scale:]
        window_dur = dur_per_week[-scale:]
        if len(window_counts) > 0 and np.any(window_counts > 0):
            mask = window_counts > 0
            feats.append(float(window_sin[mask].mean()))
            feats.append(float(window_cos[mask].mean()))
            feats.append(float(window_dur[mask].mean()))
        else:
            feats.extend([0.0, 0.0, 0.0])

    if len(counts_per_week) > 0:
        nonzero_indices = np.where(counts_per_week > 0)[0]
        weeks_since_last = float(len(history) - 1 - nonzero_indices[-1]) if len(nonzero_indices) > 0 else float(WINDOW_WEEKS)
        feats.extend([
            float(counts_per_week.mean() / max_occurrences_per_task),
            float(counts_per_week.std() / max(max_occurrences_per_task, 1)),
            float((counts_per_week > 0).mean()),
            float(min(weeks_since_last, WINDOW_WEEKS) / max(WINDOW_WEEKS, 1)),
            float(np.clip(np.sum(counts_per_week) / (WINDOW_WEEKS * max_occurrences_per_task), 0.0, 1.0)),
        ])
    else:
        feats.extend([0.0, 0.0, 0.0, 1.0, 0.0])

    iso = target_week_start.isocalendar()
    week_sin, week_cos = _cyclical_encode(int(iso.week) - 1, 52.0)
    month_sin, month_cos = _cyclical_encode(int(target_week_start.month) - 1, 12.0)
    doy_sin, doy_cos = _cyclical_encode(int(target_week_start.dayofyear) - 1, 366.0)
    feats.extend([week_sin, week_cos, month_sin, month_cos, doy_sin, doy_cos])
    days_since_last = _days_since_last_occurrence(history, task_id, target_week_start)
    feats.append(float(np.clip(days_since_last / float(max(12 * 7, WINDOW_WEEKS * 7)), 0.0, 1.0)))

    for scale in RECENT_FREQ_SCALES:
        feats.append(_recent_count_frequency(history, task_id, scale, max_occurrences_per_task))

    mean_start_bin = _recent_mean_start_bin(history, task_id)
    if mean_start_bin is None:
        feats.extend([0.0, 0.0])
    else:
        s, c = _cyclical_encode(mean_start_bin, num_time_bins())
        feats.extend([s, c])
    feats.append(_recent_mean_duration_norm(history, task_id, duration_min, duration_max))

    feats.extend([
        anchor_week_sin,
        anchor_week_cos,
        anchor_day_sin,
        anchor_day_cos,
        anchor_time_sin,
        anchor_time_cos,
        anchor_support,
        float(np.clip(same_occ_support / max(WINDOW_WEEKS, 1), 0.0, 1.0)),
        float(np.clip(anchor_long_support, 0.0, 1.0)),
        float(np.clip(anchor_proto_match, 0.0, 1.0)),
        float(np.clip(occurrence_slot / max(max_occurrences_per_task - 1, 1), 0.0, 1.0)),
    ])

    lag26_start = task_cache.lag_slot_to_bin.get(26, {}).get(int(occurrence_slot))
    lag52_start = task_cache.lag_slot_to_bin.get(52, {}).get(int(occurrence_slot))
    hist_same_occ_bins = task_cache.anchor_slot_bins.get(int(occurrence_slot), [])
    hist_median_start = _median_bin(hist_same_occ_bins) if hist_same_occ_bins else (_median_bin(task_cache.anchor_all_bins) if task_cache.anchor_all_bins else None)
    _append_start_bin_feature(feats, lag26_start)
    _append_start_bin_feature(feats, lag52_start)
    _append_start_bin_feature(feats, hist_median_start)

    max_offset_norm = float(max(bins_per_day(), 1))
    if recent_offsets:
        feats.append(float(np.clip(recent_offsets[-1] / max_offset_norm, -1.0, 1.0)))
        for scale in RECENT_OFFSET_SCALES:
            window = recent_offsets[-scale:]
            feats.append(float(np.clip(np.mean(window) / max_offset_norm, -1.0, 1.0)))
        feats.append(float(np.clip(np.std(recent_offsets) / LOCAL_START_OFFSET_RADIUS_BINS, 0.0, 1.0)))
        abs_offsets = np.abs(np.array(recent_offsets))
        feats.append(float(np.mean(abs_offsets <= 1)))
        feats.append(float(np.mean(abs_offsets <= 2)))
    else:
        feats.extend([0.0] * 8)

    if recent_same_day_offsets:
        feats.append(float(np.clip(recent_same_day_offsets[-1] / max_offset_norm, -1.0, 1.0)))
        feats.append(float(np.clip(np.mean(recent_same_day_offsets) / max_offset_norm, -1.0, 1.0)))
    else:
        feats.extend([0.0, 0.0])

    anchor_recent_day_ratio = float(np.mean([int(x // bins_per_day()) == anchor_day for x in task_cache.recent_all_bins])) if task_cache.recent_all_bins else 0.0
    feats.extend([anchor_recent_day_ratio, float(len(anchor_history) / max(ANCHOR_LOOKBACK_WEEKS, 1))])
    feats.extend(_decayed_history_features(counts_per_week, sin_per_week, cos_per_week, dur_per_week, max_occurrences_per_task))

    return TemporalContext(
        np.array(feats, dtype=np.float32),
        anchor_start_bin,
        anchor_day,
        anchor_minute_of_day,
        tuple(int(x) for x in anchor_candidates),
        tuple(float(x) for x in anchor_candidate_weights),
    )


def build_target_day_offsets(
    weeks: list[WeekRecord],
    target_week_index: int,
    max_occurrences_per_task: int | None = None,
) -> dict[tuple[int, int], int]:
    per_day_bins = bins_per_day()
    per_sample_anchor: dict[tuple[int, int], tuple[int, int]] = {}
    offsets_by_day: dict[int, list[int]] = {day: [] for day in range(7)}

    target_week = weeks[target_week_index]
    for task_id, events in target_week.events_by_task.items():
        if not events:
            continue
        assignments = build_target_occurrence_slot_assignments(
            weeks,
            target_week_index,
            task_id,
            max_occurrences_per_task=max_occurrences_per_task,
        )
        for occurrence_slot, event in assignments:
            context = build_temporal_context(
                weeks,
                target_week_index,
                task_id,
                occurrence_slot,
                max_occurrences_per_task=max_occurrences_per_task,
            )
            per_sample_anchor[(task_id, occurrence_slot)] = (context.anchor_start_bin, context.anchor_day)
            target_day = int(event.start_bin // per_day_bins)
            offsets_by_day[context.anchor_day].append(target_day - context.anchor_day)

    global_offsets_by_day = {
        day: clip_global_day_offset_bins(float(np.median(values))) if values else 0
        for day, values in offsets_by_day.items()
    }
    return {
        key: int(global_offsets_by_day[anchor_day])
        for key, (_, anchor_day) in per_sample_anchor.items()
    }


def prepare_data(
    df: pd.DataFrame,
    train_ratio: float,
    max_occurrences_per_task: int | None = None,
    max_tasks_per_week: int | None = None,
    cap_inference_scope: str = CAP_INFERENCE_SCOPE,
    show_progress: bool = False,
) -> PreparedData:
    task_names = sorted(df['task_name'].astype(str).unique().tolist())
    task_to_id = {name: idx for idx, name in enumerate(task_names)}
    duration_min = float(df['duration_minutes'].min())
    duration_max = float(df['duration_minutes'].max())
    task_duration_medians = {
        str(task_name): float(value)
        for task_name, value in df.groupby('task_name')['duration_minutes'].median().items()
    }
    df_events = _build_events(df, task_to_id, duration_min, duration_max)

    week_starts = _continuous_week_starts(df_events)
    caps = resolve_preprocessing_caps(
        df_events,
        week_starts,
        train_ratio=train_ratio,
        max_occurrences_per_task=max_occurrences_per_task,
        max_tasks_per_week=max_tasks_per_week,
        cap_inference_scope=cap_inference_scope,
    )
    max_occurrences_per_task = int(caps['max_occurrences_per_task'])
    max_tasks_per_week = int(caps['max_tasks_per_week'])
    max_count_cap = max_occurrences_per_task

    weeks: list[WeekRecord] = []
    total_weeks = len(week_starts)
    grouped_weeks = {
        week_start: group.copy()
        for week_start, group in df_events.groupby('week_start', sort=False)
    }

    for week_index, week_start in enumerate(week_starts):
        if show_progress:
            print_progress_inline(
                "Preprocesando semanas",
                week_index + 1,
                total_weeks,
            )

        week_df = grouped_weeks.get(week_start)
        if week_df is None:
            week_df = df_events.iloc[0:0].copy()
        weeks.append(
            _compute_week_features(
                week_df,
                week_index,
                week_start,
                len(task_names),
                max_count_cap,
                max_tasks_per_week,
            )
        )

    sample_feature = build_context_sequence_features(weeks, min(WINDOW_WEEKS, len(weeks)), WINDOW_WEEKS, len(task_names))[-1] if weeks else np.zeros(expected_week_feature_dim(len(task_names)), dtype=np.float32)
    sample_history = (
        build_temporal_context(
            weeks,
            min(WINDOW_WEEKS, len(weeks)),
            0,
            0,
            duration_min,
            duration_max,
            max_occurrences_per_task=max_occurrences_per_task,
        ).history_features
        if task_names
        else np.zeros(expected_history_feature_dim(), dtype=np.float32)
    )

    return PreparedData(
        df=df_events,
        task_names=task_names,
        task_to_id=task_to_id,
        weeks=weeks,
        duration_min=duration_min,
        duration_max=duration_max,
        task_duration_medians=task_duration_medians,
        max_count_cap=max_count_cap,
        week_feature_dim=int(sample_feature.shape[0]),
        history_feature_dim=int(sample_history.shape[0]),
        max_occurrences_per_task=max_occurrences_per_task,
        max_tasks_per_week=max_tasks_per_week,
        cap_inference_scope=str(caps['cap_inference_scope']),
        inferred_train_max_occurrences_per_task=int(caps['train_max_occurrences_per_task']),
        inferred_train_max_tasks_per_week=int(caps['train_max_tasks_per_week']),
        inferred_full_max_occurrences_per_task=int(caps['full_max_occurrences_per_task']),
        inferred_full_max_tasks_per_week=int(caps['full_max_tasks_per_week']),
    )


def serialize_metadata(prepared: PreparedData) -> dict[str, Any]:
    return {
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'task_names': prepared.task_names,
        'duration_min': prepared.duration_min,
        'duration_max': prepared.duration_max,
        'max_count_cap': prepared.max_count_cap,
        'max_occurrences_per_task': prepared.max_occurrences_per_task,
        'max_tasks_per_week': prepared.max_tasks_per_week,
        'cap_inference_scope': prepared.cap_inference_scope,
        'inferred_train_max_occurrences_per_task': prepared.inferred_train_max_occurrences_per_task,
        'inferred_train_max_tasks_per_week': prepared.inferred_train_max_tasks_per_week,
        'inferred_full_max_occurrences_per_task': prepared.inferred_full_max_occurrences_per_task,
        'inferred_full_max_tasks_per_week': prepared.inferred_full_max_tasks_per_week,
        'week_feature_dim': prepared.week_feature_dim,
        'history_feature_dim': prepared.history_feature_dim,
        'window_weeks': WINDOW_WEEKS,
        'anchor_lookback_weeks': ANCHOR_LOOKBACK_WEEKS,
        'bin_minutes': BIN_MINUTES,
        'num_time_bins': num_time_bins(),
        'bins_per_day': bins_per_day(),
        'global_day_offset_radius_bins': GLOBAL_DAY_OFFSET_RADIUS_BINS,
        'local_start_offset_radius_bins': LOCAL_START_OFFSET_RADIUS_BINS,
        'num_day_classes': 7,
        'num_time_of_day_classes': bins_per_day(),
        'num_day_offset_classes': GLOBAL_DAY_OFFSET_RADIUS_BINS * 2 + 1,
        'num_local_start_offset_classes': LOCAL_START_OFFSET_RADIUS_BINS * 2 + 1,
        'temporal_target_schema': 'absolute_day_time_v1',
        'temporal_num_anchor_candidates': TEMPORAL_NUM_ANCHOR_CANDIDATES,
    }
