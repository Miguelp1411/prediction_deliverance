from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ANCHOR_LOOKBACK_WEEKS,
    BIN_MINUTES,
    GLOBAL_DAY_OFFSET_RADIUS_BINS,
    HISTORY_SCALES,
    LOCAL_START_OFFSET_RADIUS_BINS,
    MAX_OCCURRENCES_PER_TASK,
    MAX_TASKS_PER_WEEK,
    WINDOW_WEEKS,
    bins_per_day,
    num_time_bins,
)

RECENT_FREQ_SCALES = (2, 4, 12)
RECENT_OFFSET_SCALES = (1, 2, 4, 8)


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
    weeks: list[WeekRecord]
    duration_min: float
    duration_max: float
    max_count_cap: int
    week_feature_dim: int
    history_feature_dim: int


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    day_start = ts.normalize()
    return day_start - pd.Timedelta(days=int(ts.dayofweek))


def _duration_to_norm(duration: float, dur_min: float, dur_max: float) -> float:
    span = max(dur_max - dur_min, 1e-6)
    return float(np.clip((duration - dur_min) / span, 0.0, 1.0))


def denormalize_duration(norm_value: float, dur_min: float, dur_max: float) -> float:
    span = max(dur_max - dur_min, 1e-6)
    return float(np.clip(norm_value, 0.0, 1.0) * span + dur_min)


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    if period <= 0:
        return 0.0, 0.0
    angle = 2.0 * np.pi * float(value) / float(period)
    return float(np.sin(angle)), float(np.cos(angle))


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


def _compute_week_features(grouped: pd.DataFrame, week_index: int, week_start: pd.Timestamp, num_tasks: int, max_count_cap: int) -> WeekRecord:
    counts = np.zeros(num_tasks, dtype=np.float32)
    mean_start_sin = np.zeros(num_tasks, dtype=np.float32)
    mean_start_cos = np.zeros(num_tasks, dtype=np.float32)
    mean_duration_norm = np.zeros(num_tasks, dtype=np.float32)
    events_by_task: dict[int, list[EventRecord]] = {task_id: [] for task_id in range(num_tasks)}

    if not grouped.empty:
        grouped = grouped.sort_values(['task_id', 'start_bin', 'start_time']).reset_index(drop=True)
        if len(grouped) > MAX_TASKS_PER_WEEK:
            raise ValueError(f'La semana {week_start} tiene {len(grouped)} tareas y supera MAX_TASKS_PER_WEEK={MAX_TASKS_PER_WEEK}.')

        for task_id, task_df in grouped.groupby('task_id'):
            task_df = task_df.sort_values(['start_bin', 'start_time'])
            n = int(len(task_df))
            counts[task_id] = float(min(n, max_count_cap))
            angles = 2.0 * np.pi * task_df['start_bin'].to_numpy(dtype=np.float32) / num_time_bins()
            mean_start_sin[task_id] = float(np.mean(np.sin(angles)))
            mean_start_cos[task_id] = float(np.mean(np.cos(angles)))
            mean_duration_norm[task_id] = float(task_df['duration_norm'].mean())
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
    total_tasks_norm = float(np.sum(counts) / max(MAX_TASKS_PER_WEEK, 1))

    return WeekRecord(
        week_index=week_index,
        week_start=week_start,
        counts=counts,
        mean_start_sin=mean_start_sin,
        mean_start_cos=mean_start_cos,
        mean_duration_norm=mean_duration_norm,
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


def _recent_count_frequency(history: list[WeekRecord], task_id: int, k: int) -> float:
    if not history:
        return 0.0
    window = history[-min(k, len(history)):]
    total = float(sum(week.counts[task_id] for week in window))
    denom = float(len(window) * MAX_OCCURRENCES_PER_TASK)
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
    return float(np.mean(values)) if values else None


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


def _build_occurrence_prototypes(history: list[WeekRecord], task_id: int) -> dict[int, list[int]]:
    prototypes: dict[int, list[int]] = {}
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))
        for occ_idx, event in enumerate(events[:MAX_OCCURRENCES_PER_TASK]):
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
) -> tuple[int, float, float, float, float]:
    same_occ_recent = _task_occurrence_history_bins(recent_history, task_id, occurrence_index)
    same_occ_long = _task_occurrence_history_bins(anchor_history, task_id, occurrence_index)

    if same_occ_long:
        base_anchor = _median_bin(same_occ_long)
        recent_support = float(np.clip(len(same_occ_recent) / max(len(recent_history), 1), 0.0, 1.0))
        long_support = float(np.clip(len(same_occ_long) / max(len(anchor_history), 1), 0.0, 1.0))
        if same_occ_recent:
            recent_anchor = _median_bin(same_occ_recent)
            anchor = clip_start_bin(round(0.75 * base_anchor + 0.25 * recent_anchor))
        else:
            anchor = base_anchor
        return anchor, recent_support, float(len(same_occ_recent)), long_support, 1.0

    prototypes = _build_occurrence_prototypes(anchor_history, task_id)
    proto_idx, proto_bins = _nearest_occurrence_prototype(prototypes, occurrence_index)
    if proto_bins:
        anchor = _median_bin(proto_bins)
        long_support = float(np.clip(len(proto_bins) / max(len(anchor_history), 1), 0.0, 1.0))
        distance_penalty = 1.0 / (1.0 + float(abs((proto_idx or 0) - occurrence_index)))
        return anchor, 0.0, 0.0, long_support, distance_penalty

    all_bins = _task_history_bins(anchor_history, task_id)
    if all_bins:
        anchor = _median_bin(all_bins)
        long_support = float(np.clip(len(all_bins) / max(len(anchor_history) * MAX_OCCURRENCES_PER_TASK, 1), 0.0, 1.0))
        return anchor, 0.0, 0.0, long_support, 0.0

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


def build_temporal_context(weeks: list[WeekRecord], target_week_index: int, task_id: int, occurrence_index: int, duration_min: float | None = None, duration_max: float | None = None) -> TemporalContext:
    history = _history_slice(weeks, target_week_index)
    anchor_history = _anchor_history_slice(weeks, target_week_index)
    target_week_start = _infer_target_week_start(weeks, target_week_index)
    duration_min = 0.0 if duration_min is None else duration_min
    duration_max = 1.0 if duration_max is None else duration_max

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

    anchor_start_bin, anchor_support, same_occ_support, anchor_long_support, anchor_proto_match = _compute_anchor_start_bin(history, anchor_history, task_id, occurrence_index)
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
        feats.append(float(window_counts.sum() / (MAX_OCCURRENCES_PER_TASK * scale_len)))
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
            float(counts_per_week.mean() / MAX_OCCURRENCES_PER_TASK),
            float(counts_per_week.std() / max(MAX_OCCURRENCES_PER_TASK, 1)),
            float((counts_per_week > 0).mean()),
            float(min(weeks_since_last, WINDOW_WEEKS) / max(WINDOW_WEEKS, 1)),
            float(np.clip(np.sum(counts_per_week) / (WINDOW_WEEKS * MAX_OCCURRENCES_PER_TASK), 0.0, 1.0)),
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
        feats.append(_recent_count_frequency(history, task_id, scale))

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
        float(np.clip(occurrence_index / max(MAX_OCCURRENCES_PER_TASK - 1, 1), 0.0, 1.0)),
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

    return TemporalContext(np.array(feats, dtype=np.float32), anchor_start_bin, anchor_day)


def build_target_day_offsets(weeks: list[WeekRecord], target_week_index: int) -> dict[tuple[int, int], int]:
    target_week = weeks[target_week_index]
    per_sample_anchor: dict[tuple[int, int], tuple[int, int]] = {}
    offsets_by_day: dict[int, list[int]] = {day: [] for day in range(7)}

    for task_id, events in target_week.events_by_task.items():
        sorted_events = sorted(events, key=lambda e: (e.start_bin, e.start_time))
        for occurrence_index, event in enumerate(sorted_events):
            context = build_temporal_context(weeks, target_week_index, task_id, occurrence_index)
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


def prepare_data(df: pd.DataFrame, train_ratio: float) -> PreparedData:
    task_names = sorted(df['task_name'].astype(str).unique().tolist())
    task_to_id = {name: idx for idx, name in enumerate(task_names)}
    duration_min = float(df['duration_minutes'].min())
    duration_max = float(df['duration_minutes'].max())
    df_events = _build_events(df, task_to_id, duration_min, duration_max)

    week_starts = _continuous_week_starts(df_events)
    split_week_idx = max(int(len(week_starts) * train_ratio), WINDOW_WEEKS + 1)
    train_weeks_df = df_events[df_events['week_start'].isin(week_starts[:split_week_idx])]
    max_count_cap = int(train_weeks_df.groupby(['week_start', 'task_id']).size().max()) if not train_weeks_df.empty else 1
    max_count_cap = max(1, min(max_count_cap, MAX_OCCURRENCES_PER_TASK))

    weeks: list[WeekRecord] = []
    for week_index, week_start in enumerate(week_starts):
        week_df = df_events[df_events['week_start'] == week_start].copy()
        weeks.append(_compute_week_features(week_df, week_index, week_start, len(task_names), max_count_cap))

    sample_feature = week_to_feature_vector(weeks[0]) if weeks else np.zeros(4 * len(task_names) + 7, dtype=np.float32)
    sample_history = build_temporal_context(weeks, min(WINDOW_WEEKS, len(weeks)), 0, 0, duration_min, duration_max).history_features if task_names else np.zeros(59, dtype=np.float32)

    return PreparedData(
        df=df_events,
        task_names=task_names,
        task_to_id=task_to_id,
        weeks=weeks,
        duration_min=duration_min,
        duration_max=duration_max,
        max_count_cap=max_count_cap,
        week_feature_dim=int(sample_feature.shape[0]),
        history_feature_dim=int(sample_history.shape[0]),
    )


def serialize_metadata(prepared: PreparedData) -> dict[str, Any]:
    return {
        'task_names': prepared.task_names,
        'duration_min': prepared.duration_min,
        'duration_max': prepared.duration_max,
        'max_count_cap': prepared.max_count_cap,
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
