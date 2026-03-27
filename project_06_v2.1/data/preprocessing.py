from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ANCHOR_LOOKBACK_WEEKS,
    BIN_MINUTES,
    CAP_INFERENCE_SCOPE,
    DEFAULT_MAX_OCCURRENCES_PER_TASK,
    FEATURE_SCHEMA_VERSION,
    GLOBAL_DAY_OFFSET_RADIUS_BINS,
    HISTORY_SCALES,
    LOCAL_START_OFFSET_RADIUS_BINS,
    RECENCY_DECAY_BASE,
    WINDOW_WEEKS,
    bins_per_day,
    num_time_bins,
)
from data.io import DEFAULT_DEVICE_UID

RECENT_FREQ_SCALES = (2, 4, 12)
RECENT_OFFSET_SCALES = (1, 2, 4, 8)


@dataclass
class EventRecord:
    task_id: int
    task_name: str
    device_uid: str
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


@dataclass
class PreparedData:
    df: pd.DataFrame
    task_names: list[str]
    task_to_id: dict[str, int]
    task_base_names: list[str]
    task_device_uids: list[str]
    has_explicit_device_uids: bool
    weeks: list[WeekRecord]
    duration_min: float
    duration_max: float
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


def build_series_name(device_uid: str, task_name: str) -> str:
    return f'{device_uid}::{task_name}'


def split_series_name(series_name: str) -> tuple[str, str]:
    device_uid, task_name = str(series_name).split('::', 1)
    return device_uid, task_name


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


def expected_week_feature_dim(num_tasks: int) -> int:
    per_task_scalar_blocks = 7 * num_tasks
    per_task_day_distribution = 7 * num_tasks
    calendar_features = 7
    return per_task_scalar_blocks + per_task_day_distribution + calendar_features


def expected_history_feature_dim() -> int:
    history_scale_features = len(HISTORY_SCALES) * 5
    global_task_stats = 5
    calendar_context = 7
    recent_frequency_features = len(RECENT_FREQ_SCALES)
    recent_mean_features = 3
    anchor_features = 11
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
    out['series_name'] = [
        build_series_name(str(device_uid), str(task_name))
        for device_uid, task_name in zip(out['device_uid'].tolist(), out['task_name'].tolist())
    ]
    out['task_id'] = out['series_name'].map(task_to_id)
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
                    device_uid=str(row.device_uid),
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


def _task_occurrence_history_bins(history: list[WeekRecord], task_id: int, occurrence_index: int) -> list[int]:
    out: list[int] = []
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))
        if occurrence_index < len(events):
            out.append(int(events[occurrence_index].start_bin))
    return out


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


def _build_occurrence_prototypes(history: list[WeekRecord], task_id: int, max_occurrences_per_task: int) -> dict[int, list[int]]:
    prototypes: dict[int, list[int]] = {}
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))
        for occ_idx, event in enumerate(events[:max_occurrences_per_task]):
            prototypes.setdefault(occ_idx, []).append(int(event.start_bin))
    return prototypes


def _nearest_occurrence_prototype(prototypes: dict[int, list[int]], occurrence_index: int) -> tuple[int | None, list[int]]:
    if occurrence_index in prototypes and prototypes[occurrence_index]:
        return occurrence_index, prototypes[occurrence_index]
    if not prototypes:
        return None, []
    nearest_idx = min(prototypes.keys(), key=lambda idx: (abs(idx - occurrence_index), idx))
    return nearest_idx, prototypes.get(nearest_idx, [])


def _compute_anchor_start_bin(
    recent_history: list[WeekRecord],
    anchor_history: list[WeekRecord],
    task_id: int,
    occurrence_index: int,
    max_occurrences_per_task: int,
) -> tuple[int, float, float, float, float]:
    same_occ_recent = _task_occurrence_history_bins(recent_history, task_id, occurrence_index)
    same_occ_long = _task_occurrence_history_bins(anchor_history, task_id, occurrence_index)
    recent_all_bins = _task_history_bins(recent_history, task_id)
    anchor_all_bins = _task_history_bins(anchor_history, task_id)

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
        return anchor, recent_support, float(len(same_occ_recent)), long_support, float(np.clip(proto_quality, 0.0, 1.0))

    prototypes = _build_occurrence_prototypes(anchor_history, task_id, max_occurrences_per_task)
    proto_idx, proto_bins = _nearest_occurrence_prototype(prototypes, occurrence_index)
    if proto_bins:
        proto_day_bins, _, proto_day_ratio = _select_day_cluster(proto_bins, preferred_day)
        anchor = _median_bin(proto_day_bins if proto_day_bins else proto_bins)
        long_support = float(np.clip(len(proto_day_bins if proto_day_bins else proto_bins) / max(len(anchor_history), 1), 0.0, 1.0))
        distance_penalty = 1.0 / (1.0 + float(abs((proto_idx or 0) - occurrence_index)))
        proto_match = distance_penalty * (0.5 + 0.5 * proto_day_ratio)
        return anchor, 0.0, 0.0, long_support, float(np.clip(proto_match, 0.0, 1.0))

    if anchor_all_bins:
        all_day_bins, _, all_day_ratio = _select_day_cluster(anchor_all_bins, preferred_day)
        anchor = _median_bin(all_day_bins if all_day_bins else anchor_all_bins)
        long_support = float(
            np.clip(
                len(all_day_bins if all_day_bins else anchor_all_bins)
                / max(len(anchor_history) * max_occurrences_per_task, 1),
                0.0,
                1.0,
            )
        )
        return anchor, 0.0, 0.0, long_support, float(np.clip(0.25 + 0.5 * all_day_ratio, 0.0, 1.0))

    return 0, 0.0, 0.0, 0.0, 0.0


def _collect_recent_offsets(history: list[WeekRecord], task_id: int, occurrence_index: int, anchor_start_bin: int) -> list[int]:
    out: list[int] = []
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))
        if occurrence_index < len(events):
            out.append(int(events[occurrence_index].start_bin) - int(anchor_start_bin))
    return out


def _collect_recent_same_day_offsets(history: list[WeekRecord], task_id: int, occurrence_index: int, anchor_start_bin: int) -> list[int]:
    anchor_day = int(anchor_start_bin // bins_per_day())
    out: list[int] = []
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))
        if occurrence_index < len(events):
            event = events[occurrence_index]
            if int(event.start_bin // bins_per_day()) == anchor_day:
                out.append(int(event.start_bin) - int(anchor_start_bin))
    return out


def build_temporal_context(
    weeks: list[WeekRecord],
    target_week_index: int,
    task_id: int,
    occurrence_index: int,
    duration_min: float | None = None,
    duration_max: float | None = None,
    max_occurrences_per_task: int | None = None,
) -> TemporalContext:
    history = _history_slice(weeks, target_week_index)
    anchor_history = _anchor_history_slice(weeks, target_week_index)
    target_week_start = _infer_target_week_start(weeks, target_week_index)
    duration_min = 0.0 if duration_min is None else duration_min
    duration_max = 1.0 if duration_max is None else duration_max
    max_occurrences_per_task = _sanitize_cap(max_occurrences_per_task, DEFAULT_MAX_OCCURRENCES_PER_TASK)

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

    anchor_start_bin, anchor_support, same_occ_support, anchor_long_support, anchor_proto_match = _compute_anchor_start_bin(
        history,
        anchor_history,
        task_id,
        occurrence_index,
        max_occurrences_per_task,
    )
    anchor_day = int(anchor_start_bin // bins_per_day())
    anchor_minute_of_day = int(anchor_start_bin % bins_per_day())
    anchor_week_sin, anchor_week_cos = _cyclical_encode(anchor_start_bin, num_time_bins())
    anchor_day_sin, anchor_day_cos = _cyclical_encode(anchor_day, 7.0)
    anchor_time_sin, anchor_time_cos = _cyclical_encode(anchor_minute_of_day, bins_per_day())

    recent_offsets = _collect_recent_offsets(history, task_id, occurrence_index, anchor_start_bin)
    recent_same_day_offsets = _collect_recent_same_day_offsets(history, task_id, occurrence_index, anchor_start_bin)

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
        float(np.clip(occurrence_index / max(max_occurrences_per_task - 1, 1), 0.0, 1.0)),
    ])

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

    anchor_recent_day_ratio = float(np.mean([int(x // bins_per_day()) == anchor_day for x in _task_history_bins(history, task_id)])) if history else 0.0
    feats.extend([anchor_recent_day_ratio, float(len(anchor_history) / max(ANCHOR_LOOKBACK_WEEKS, 1))])
    feats.extend(_decayed_history_features(counts_per_week, sin_per_week, cos_per_week, dur_per_week, max_occurrences_per_task))

    return TemporalContext(np.array(feats, dtype=np.float32), anchor_start_bin, anchor_day)


def build_target_day_offsets(
    weeks: list[WeekRecord],
    target_week_index: int,
    max_occurrences_per_task: int | None = None,
) -> dict[tuple[int, int], int]:
    target_week = weeks[target_week_index]
    per_sample_anchor: dict[tuple[int, int], tuple[int, int]] = {}
    offsets_by_day: dict[int, list[int]] = {day: [] for day in range(7)}

    for task_id, events in target_week.events_by_task.items():
        sorted_events = sorted(events, key=lambda e: (e.start_bin, e.start_time))
        for occurrence_index, event in enumerate(sorted_events):
            context = build_temporal_context(
                weeks,
                target_week_index,
                task_id,
                occurrence_index,
                max_occurrences_per_task=max_occurrences_per_task,
            )
            per_sample_anchor[(task_id, occurrence_index)] = (context.anchor_start_bin, context.anchor_day)
            offsets_by_day[context.anchor_day].append(int(event.start_bin) - int(context.anchor_start_bin))

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
    df = df.copy()
    if 'device_uid' not in df.columns:
        df['device_uid'] = DEFAULT_DEVICE_UID
    df['device_uid'] = df['device_uid'].where(df['device_uid'].notna(), DEFAULT_DEVICE_UID)
    df['device_uid'] = df['device_uid'].astype(str).str.strip()
    df.loc[df['device_uid'] == '', 'device_uid'] = DEFAULT_DEVICE_UID

    series_pairs = sorted(
        {
            (str(device_uid), str(task_name))
            for device_uid, task_name in zip(df['device_uid'].tolist(), df['task_name'].astype(str).tolist())
        },
        key=lambda item: (item[0], item[1]),
    )
    task_names = [build_series_name(device_uid, task_name) for device_uid, task_name in series_pairs]
    task_to_id = {name: idx for idx, name in enumerate(task_names)}
    task_device_uids = [device_uid for device_uid, _ in series_pairs]
    task_base_names = [task_name for _, task_name in series_pairs]
    has_explicit_device_uids = any(device_uid != DEFAULT_DEVICE_UID for device_uid in task_device_uids)
    duration_min = float(df['duration_minutes'].min())
    duration_max = float(df['duration_minutes'].max())
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

    for week_index, week_start in enumerate(week_starts):
        if show_progress:
            print_progress_inline(
                "Preprocesando semanas",
                week_index + 1,
                total_weeks,
            )

        week_df = df_events[df_events['week_start'] == week_start].copy()
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

    sample_feature = week_to_feature_vector(weeks[0]) if weeks else np.zeros(expected_week_feature_dim(len(task_names)), dtype=np.float32)
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
        task_base_names=task_base_names,
        task_device_uids=task_device_uids,
        has_explicit_device_uids=bool(has_explicit_device_uids),
        weeks=weeks,
        duration_min=duration_min,
        duration_max=duration_max,
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
        'task_base_names': prepared.task_base_names,
        'task_device_uids': prepared.task_device_uids,
        'has_explicit_device_uids': prepared.has_explicit_device_uids,
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
    }
