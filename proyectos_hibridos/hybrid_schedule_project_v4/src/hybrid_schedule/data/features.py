from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "v4_positional_density_ordinal"


@dataclass
class EventItem:
    database_id: str
    robot_id: str
    task_idx: int
    task_type: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_bin: int
    duration_bins: int
    duration_minutes: float
    source_event_id: str


@dataclass
class SlotPrototype:
    task_idx: int
    slot_id: int
    center_bin: int
    duration_bins: int
    support: float


@dataclass
class SeriesBundle:
    database_id: str
    robot_id: str
    task_names: list[str]
    task_to_idx: dict[str, int]
    week_starts: list[pd.Timestamp]
    counts: np.ndarray  # [W, T]
    day_hist: np.ndarray  # [W, T, 7]
    mean_start: np.ndarray  # [W, T]
    mean_duration: np.ndarray  # [W, T]
    events: list[list[EventItem]]


@dataclass
class GlobalContext:
    task_names: list[str]
    task_to_idx: dict[str, int]
    database_to_idx: dict[str, int]
    robot_to_idx: dict[str, int]
    series: dict[tuple[str, str], SeriesBundle]


def week_bin_from_timestamp(ts: pd.Timestamp, bin_minutes: int) -> int:
    minutes = ts.dayofweek * 24 * 60 + ts.hour * 60 + ts.minute
    return int(minutes // bin_minutes)


def build_global_context(df: pd.DataFrame, bin_minutes: int = 5) -> GlobalContext:
    task_names = sorted(df['task_type'].unique().tolist())
    task_to_idx = {name: idx for idx, name in enumerate(task_names)}
    database_to_idx = {name: idx for idx, name in enumerate(sorted(df['database_id'].unique().tolist()))}
    robot_keys = sorted(df[['database_id', 'robot_id']].drop_duplicates().apply(lambda x: f"{x['database_id']}::{x['robot_id']}", axis=1).tolist())
    robot_to_idx = {name: idx for idx, name in enumerate(robot_keys)}

    series: dict[tuple[str, str], SeriesBundle] = {}
    for (database_id, robot_id), group in df.groupby(['database_id', 'robot_id']):
        week_starts = sorted(group['week_start'].drop_duplicates().tolist())
        W = len(week_starts)
        T = len(task_names)
        counts = np.zeros((W, T), dtype=np.int64)
        day_hist = np.zeros((W, T, 7), dtype=np.float32)
        mean_start = np.zeros((W, T), dtype=np.float32)
        mean_duration = np.zeros((W, T), dtype=np.float32)
        events: list[list[EventItem]] = [[] for _ in range(W)]
        week_to_idx = {wk: i for i, wk in enumerate(week_starts)}

        for _, row in group.iterrows():
            widx = week_to_idx[row['week_start']]
            tidx = task_to_idx[row['task_type']]
            start_bin = week_bin_from_timestamp(row['start_time'], bin_minutes)
            duration_bins = max(1, int(round(float(row['duration_minutes']) / float(bin_minutes))))
            evt = EventItem(
                database_id=database_id,
                robot_id=robot_id,
                task_idx=tidx,
                task_type=row['task_type'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                start_bin=start_bin,
                duration_bins=duration_bins,
                duration_minutes=float(row['duration_minutes']),
                source_event_id=str(row['source_event_id']),
            )
            events[widx].append(evt)
            counts[widx, tidx] += 1
            day_hist[widx, tidx, int(row['day_of_week'])] += 1.0

        for widx in range(W):
            events[widx].sort(key=lambda e: (e.start_bin, e.duration_bins, e.task_type))
            for tidx in range(T):
                task_events = [e for e in events[widx] if e.task_idx == tidx]
                if task_events:
                    mean_start[widx, tidx] = float(np.mean([e.start_bin for e in task_events]))
                    mean_duration[widx, tidx] = float(np.mean([e.duration_bins for e in task_events]))
                    day_sum = float(day_hist[widx, tidx].sum())
                    if day_sum > 0:
                        day_hist[widx, tidx] /= day_sum

        series[(database_id, robot_id)] = SeriesBundle(
            database_id=database_id,
            robot_id=robot_id,
            task_names=task_names,
            task_to_idx=task_to_idx,
            week_starts=week_starts,
            counts=counts,
            day_hist=day_hist,
            mean_start=mean_start,
            mean_duration=mean_duration,
            events=events,
        )

    return GlobalContext(
        task_names=task_names,
        task_to_idx=task_to_idx,
        database_to_idx=database_to_idx,
        robot_to_idx=robot_to_idx,
        series=series,
    )


def calendar_features(week_start: pd.Timestamp) -> np.ndarray:
    iso_week = int(week_start.isocalendar().week)
    month = int(week_start.month)
    day_of_year = int(week_start.dayofyear)
    quarter = int(week_start.quarter)
    woy_angle = 2.0 * np.pi * (iso_week - 1) / 52.0
    month_angle = 2.0 * np.pi * (month - 1) / 12.0
    doy_angle = 2.0 * np.pi * (day_of_year - 1) / 366.0
    quarter_angle = 2.0 * np.pi * (quarter - 1) / 4.0
    return np.asarray([
        np.sin(woy_angle), np.cos(woy_angle),
        np.sin(month_angle), np.cos(month_angle),
        np.sin(doy_angle), np.cos(doy_angle),
        np.sin(quarter_angle), np.cos(quarter_angle),
    ], dtype=np.float32)


def _circular_mean_and_dispersion(starts: list[int], bins_per_week: int) -> tuple[float, float, float]:
    if not starts:
        return 0.0, 1.0, 0.0
    arr = np.asarray(starts, dtype=np.float32)
    angles = 2.0 * np.pi * arr / float(bins_per_week)
    mean_sin = float(np.mean(np.sin(angles)))
    mean_cos = float(np.mean(np.cos(angles)))
    dispersion = float(np.clip(1.0 - np.hypot(mean_sin, mean_cos), 0.0, 1.0))
    return mean_sin, mean_cos, dispersion


def _normalized_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    base = max(len(probs), 2)
    return float(np.clip(-(probs * np.log(probs)).sum() / np.log(float(base)), 0.0, 1.0))


def build_history_tensor(series: SeriesBundle, target_week_idx: int, window_weeks: int, bin_minutes: int = 5) -> np.ndarray:
    start = max(0, target_week_idx - window_weeks)
    bins_per_week = int(7 * 24 * 60 / bin_minutes)
    num_tasks = int(series.counts.shape[1])
    rows: list[np.ndarray] = []
    for widx in range(start, target_week_idx):
        week_events = series.events[widx]
        total_tasks_norm = float(len(week_events) / max(1, num_tasks * 4))
        per_task_stats = []
        for tidx in range(num_tasks):
            task_events = [e for e in week_events if e.task_idx == tidx]
            starts = [e.start_bin for e in task_events]
            mean_sin, mean_cos, dispersion = _circular_mean_and_dispersion(starts, bins_per_week)
            active_days = float(len({e.start_time.dayofweek for e in task_events}) / 7.0) if task_events else 0.0
            day_entropy = _normalized_entropy(series.day_hist[widx, tidx]) if series.day_hist[widx, tidx].sum() > 0 else 0.0
            mean_dur = float(series.mean_duration[widx, tidx] / 12.0)
            per_task_stats.extend([
                float(series.counts[widx, tidx]),
                *series.day_hist[widx, tidx].astype(np.float32).tolist(),
                mean_sin,
                mean_cos,
                mean_dur,
                dispersion,
                active_days,
                day_entropy,
            ])
        row = np.asarray(per_task_stats + [total_tasks_norm] + calendar_features(series.week_starts[widx]).tolist(), dtype=np.float32)
        rows.append(row)

    feat_dim = num_tasks * 14 + 9
    if not rows:
        return np.zeros((window_weeks, feat_dim), dtype=np.float32)
    feat_arr = np.stack(rows, axis=0)
    if feat_arr.shape[0] < window_weeks:
        pad = np.zeros((window_weeks - feat_arr.shape[0], feat_arr.shape[1]), dtype=np.float32)
        feat_arr = np.concatenate([pad, feat_arr], axis=0)
    return feat_arr[-window_weeks:]


def build_future_history_tensor(series: SeriesBundle, window_weeks: int, bin_minutes: int = 5) -> np.ndarray:
    return build_history_tensor(series, len(series.week_starts), window_weeks, bin_minutes=bin_minutes)


def per_task_recent_stats(series: SeriesBundle, target_week_idx: int | None, task_idx: int, window: int = 4) -> dict[str, float]:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    start = max(0, target_week_idx - window)
    values = series.counts[start:target_week_idx, task_idx].astype(np.float32)
    if values.size == 0:
        return {'recent_mean': 0.0, 'recent_last': 0.0, 'recent_median': 0.0, 'recent_std': 0.0, 'recent_max': 0.0}
    return {
        'recent_mean': float(values.mean()),
        'recent_last': float(values[-1]),
        'recent_median': float(np.median(values)),
        'recent_std': float(values.std()),
        'recent_max': float(values.max()),
    }


def seasonal_lag_values(series: SeriesBundle, target_week_idx: int | None, task_idx: int, lags: tuple[int, ...] = (4, 26, 52)) -> tuple[np.ndarray, np.ndarray]:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    vals = []
    masks = []
    for lag in lags:
        idx = int(target_week_idx - lag)
        if idx >= 0 and idx < len(series.week_starts):
            vals.append(float(series.counts[idx, task_idx]))
            masks.append(1.0)
        else:
            vals.append(0.0)
            masks.append(0.0)
    return np.asarray(vals, dtype=np.float32), np.asarray(masks, dtype=np.float32)


def seasonal_count_baseline(
    series: SeriesBundle,
    target_week_idx: int | None,
    task_idx: int,
    template_count: int,
    lags: tuple[int, ...] = (4, 26, 52),
    lag_weights: tuple[float, ...] = (0.50, 0.30, 0.20),
    template_weight: float = 0.55,
    recent_weight: float = 0.15,
) -> dict[str, float]:
    lag_vals, lag_mask = seasonal_lag_values(series, target_week_idx, task_idx, lags=lags)
    weights = np.asarray(lag_weights, dtype=np.float32)
    if weights.size != lag_vals.size or weights.sum() <= 0:
        weights = np.ones_like(lag_vals, dtype=np.float32)
    weighted_mask = weights * lag_mask
    seasonal = float((lag_vals * weighted_mask).sum() / max(weighted_mask.sum(), 1e-6)) if weighted_mask.sum() > 0 else 0.0
    recent = per_task_recent_stats(series, target_week_idx, task_idx, window=8)
    blend_weight = template_weight + (1.0 - template_weight) * float(weighted_mask.sum() > 0)
    blended = (
        template_weight * float(template_count)
        + (1.0 - template_weight - recent_weight) * seasonal
        + recent_weight * recent['recent_median']
    )
    return {
        'baseline_count': float(max(0.0, round(blended))),
        'seasonal_count': seasonal,
        'lag_4': float(lag_vals[0]) if lag_vals.size > 0 else 0.0,
        'lag_26': float(lag_vals[1]) if lag_vals.size > 1 else 0.0,
        'lag_52': float(lag_vals[2]) if lag_vals.size > 2 else 0.0,
        'seasonal_signal': float(weighted_mask.sum()),
        'recent_mean': recent['recent_mean'],
        'recent_last': recent['recent_last'],
        'recent_median': recent['recent_median'],
        'recent_std': recent['recent_std'],
        'blend_weight': blend_weight,
    }



def task_count_lag(series: SeriesBundle, target_week_idx: int | None, task_idx: int, lag: int) -> tuple[float, float]:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    idx = int(target_week_idx - lag)
    if 0 <= idx < len(series.week_starts):
        return float(series.counts[idx, task_idx]), 1.0
    return 0.0, 0.0


def _future_week_start(series: SeriesBundle, target_week_idx: int | None) -> pd.Timestamp:
    if target_week_idx is not None:
        return pd.Timestamp(series.week_starts[target_week_idx])
    if len(series.week_starts) == 0:
        return pd.Timestamp.utcnow().tz_localize('UTC')
    if len(series.week_starts) == 1:
        return pd.Timestamp(series.week_starts[-1]) + pd.Timedelta(days=7)
    return pd.Timestamp(series.week_starts[-1]) + (pd.Timestamp(series.week_starts[-1]) - pd.Timestamp(series.week_starts[-2]))


def occurrence_calendar_context(series: SeriesBundle, target_week_idx: int | None) -> dict[str, float]:
    week_start = _future_week_start(series, target_week_idx)
    iso = week_start.isocalendar()
    iso_week = int(iso.week)
    month = int(week_start.month)
    year = int(week_start.year)
    quarter = int(week_start.quarter)
    day_of_year = int(week_start.dayofyear)
    week_angle = 2.0 * np.pi * float(iso_week - 1) / 52.0
    month_angle = 2.0 * np.pi * float(month - 1) / 12.0
    year_fraction = float((day_of_year - 1) / 365.0)
    monthly_load_table = {1: 0.81, 2: 0.84, 3: 0.88, 4: 0.91, 5: 0.94, 6: 0.86, 7: 0.72, 8: 0.68, 9: 0.79, 10: 1.00, 11: 0.98, 12: 0.96}
    peak_months = {10, 11, 12}
    trough_months = {7, 8, 9}
    circular_dist_to_october = min((month - 10) % 12, (10 - month) % 12)
    n_weekdays = float(sum(1 for d in range(7) if (week_start + pd.Timedelta(days=d)).dayofweek < 5))
    n_weekend = 7.0 - n_weekdays
    return {
        'iso_week': float(iso_week),
        'month': float(month),
        'year': float(year),
        'quarter': float(quarter),
        'day_of_year': float(day_of_year),
        'sin_week_of_year': float(np.sin(week_angle)),
        'cos_week_of_year': float(np.cos(week_angle)),
        'sin_month': float(np.sin(month_angle)),
        'cos_month': float(np.cos(month_angle)),
        'is_peak_season': 1.0 if month in peak_months else 0.0,
        'is_trough_season': 1.0 if month in trough_months else 0.0,
        'dist_to_peak_month': float(circular_dist_to_october / 6.0),
        'monthly_load_index': float(monthly_load_table.get(month, 0.85)),
        'n_weekdays_in_week': n_weekdays,
        'n_weekend_days_in_week': n_weekend,
        'n_night_sessions_possible': n_weekdays,
        'expected_night_tasks': 2.0 * n_weekdays,
        'week_of_year_norm': float((iso_week - 1) / 51.0),
        'year_norm': float((year - 2000) / 50.0),
        'year_fraction': year_fraction,
        'is_transition_week': 1.0 if month != int((week_start + pd.Timedelta(days=6)).month) else 0.0,
    }


def series_trend_features(series: SeriesBundle, target_week_idx: int | None, task_idx: int, window: int = 26) -> dict[str, float]:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    start = max(0, target_week_idx - window)
    values = series.counts[start:target_week_idx, task_idx].astype(np.float32)
    if values.size <= 1:
        return {'weeks_elapsed_total': float(target_week_idx), 'trend_tasks_per_week': 0.0}
    x = np.arange(values.size, dtype=np.float32)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    slope = float(np.dot(x, values - values.mean()) / denom) if denom > 0 else 0.0
    return {
        'weeks_elapsed_total': float(target_week_idx),
        'trend_tasks_per_week': slope,
    }


def task_prototype_from_history(series: SeriesBundle, target_week_idx: int, task_idx: int, max_weeks: int = 52) -> dict[str, float]:
    start = max(0, target_week_idx - max_weeks)
    starts: list[int] = []
    durations: list[int] = []
    for week_events in series.events[start:target_week_idx]:
        for e in week_events:
            if e.task_idx == task_idx:
                starts.append(e.start_bin)
                durations.append(e.duration_bins)
    if not starts:
        return {'start_bin': 0.0, 'duration_bins': 1.0}
    return {
        'start_bin': float(np.median(starts)),
        'duration_bins': float(np.median(durations)),
    }


def task_duration_median(series: SeriesBundle, target_week_idx: int | None, task_idx: int, max_weeks: int = 104) -> float:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    start = max(0, target_week_idx - max_weeks)
    durations = [e.duration_bins for week in series.events[start:target_week_idx] for e in week if e.task_idx == task_idx]
    if not durations:
        return 1.0
    return float(np.median(np.asarray(durations, dtype=np.float32)))


def task_temporal_profile(series: SeriesBundle, target_week_idx: int | None, task_idx: int, max_weeks: int = 52, bin_minutes: int = 5) -> dict[str, float]:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    start = max(0, target_week_idx - max_weeks)
    starts: list[int] = []
    durations: list[int] = []
    day_hist = np.zeros(7, dtype=np.float32)
    active_weeks = 0
    for week_events in series.events[start:target_week_idx]:
        task_events = [e for e in week_events if e.task_idx == task_idx]
        if task_events:
            active_weeks += 1
        for e in task_events:
            starts.append(e.start_bin)
            durations.append(e.duration_bins)
            day_hist[int(e.start_time.dayofweek)] += 1.0
    bins_per_week = int(7 * 24 * 60 / bin_minutes)
    mean_sin, mean_cos, dispersion = _circular_mean_and_dispersion(starts, bins_per_week)
    if day_hist.sum() > 0:
        day_probs = day_hist / day_hist.sum()
    else:
        day_probs = day_hist
    return {
        'mean_start_norm': float(np.median(starts) / max(1, bins_per_week - 1)) if starts else 0.0,
        'duration_median_norm': float(np.median(durations) / 12.0) if durations else 0.0,
        'start_sin': mean_sin,
        'start_cos': mean_cos,
        'start_dispersion': dispersion,
        'active_days_norm': float((day_hist > 0).sum() / 7.0),
        'day_multimodality_norm': _normalized_entropy(day_probs) if day_hist.sum() > 0 else 0.0,
        'weekly_frequency': float(len(starts) / max(1, target_week_idx - start)),
        'support_norm': float(active_weeks / max(1, target_week_idx - start)),
    }


def task_slot_prototypes(series: SeriesBundle, target_week_idx: int | None, task_idx: int, max_slots: int = 32, max_weeks: int = 104) -> list[SlotPrototype]:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    start = max(0, target_week_idx - max_weeks)
    slot_starts: dict[int, list[int]] = {}
    slot_durations: dict[int, list[int]] = {}
    weeks_with_task = 0
    for week_events in series.events[start:target_week_idx]:
        task_events = sorted([e for e in week_events if e.task_idx == task_idx], key=lambda e: e.start_bin)
        if task_events:
            weeks_with_task += 1
        for rank, event in enumerate(task_events[:max_slots]):
            slot_starts.setdefault(rank, []).append(int(event.start_bin))
            slot_durations.setdefault(rank, []).append(int(event.duration_bins))
    if not slot_starts:
        proto = task_prototype_from_history(series, target_week_idx, task_idx)
        return [SlotPrototype(task_idx=task_idx, slot_id=0, center_bin=int(round(proto['start_bin'])), duration_bins=max(1, int(round(proto['duration_bins']))), support=0.0)]
    prototypes: list[SlotPrototype] = []
    for slot_id in sorted(slot_starts.keys()):
        starts = slot_starts[slot_id]
        durations = slot_durations.get(slot_id, [1])
        prototypes.append(SlotPrototype(
            task_idx=task_idx,
            slot_id=int(slot_id),
            center_bin=int(round(float(np.median(starts)))),
            duration_bins=max(1, int(round(float(np.median(durations))))),
            support=float(len(starts) / max(1, weeks_with_task)),
        ))
    prototypes.sort(key=lambda p: (p.center_bin, p.slot_id))
    return prototypes[:max_slots]


def assign_events_to_prototypes(events: list[EventItem], prototypes: list[SlotPrototype]) -> list[tuple[int, EventItem, SlotPrototype]]:
    if not events:
        return []
    ordered_events = sorted(events, key=lambda e: e.start_bin)
    ordered_protos = sorted(prototypes, key=lambda p: p.center_bin)
    used: set[int] = set()
    assignments: list[tuple[int, EventItem, SlotPrototype]] = []
    for idx, event in enumerate(ordered_events):
        best_j = None
        best_cost = None
        for j, proto in enumerate(ordered_protos):
            if j in used:
                continue
            cost = abs(int(event.start_bin) - int(proto.center_bin)) + 0.5 * abs(int(event.duration_bins) - int(proto.duration_bins))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_j = j
        if best_j is None:
            proto = SlotPrototype(task_idx=event.task_idx, slot_id=idx, center_bin=int(event.start_bin), duration_bins=int(event.duration_bins), support=0.0)
        else:
            used.add(best_j)
            proto = ordered_protos[best_j]
        assignments.append((proto.slot_id, event, proto))
    assignments.sort(key=lambda x: x[0])
    return assignments


def slot_recent_start_offsets(series: SeriesBundle, target_week_idx: int | None, task_idx: int, slot_id: int, max_slots: int = 32, max_weeks: int = 26, bin_minutes: int = 5) -> dict[str, float]:
    prototypes = task_slot_prototypes(series, target_week_idx, task_idx, max_slots=max_slots, max_weeks=max_weeks)
    proto_map = {p.slot_id: p for p in prototypes}
    anchor = proto_map.get(slot_id)
    if anchor is None:
        return {'offset_mean_days': 0.0, 'offset_mean_local': 0.0, 'offset_std_local': 0.0, 'slot_support': 0.0}
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    start = max(0, target_week_idx - max_weeks)
    offsets = []
    for week_events in series.events[start:target_week_idx]:
        task_events = [e for e in week_events if e.task_idx == task_idx]
        assigned = assign_events_to_prototypes(task_events, prototypes)
        for assigned_slot_id, event, _proto in assigned:
            if assigned_slot_id == slot_id:
                offsets.append(int(event.start_bin) - int(anchor.center_bin))
    if not offsets:
        return {'offset_mean_days': 0.0, 'offset_mean_local': 0.0, 'offset_std_local': 0.0, 'slot_support': float(anchor.support)}
    bins_per_day = int(24 * 60 / bin_minutes)
    offsets_arr = np.asarray(offsets, dtype=np.float32)
    local_offsets = offsets_arr - np.round(offsets_arr / bins_per_day) * bins_per_day
    return {
        'offset_mean_days': float(np.mean(offsets_arr / bins_per_day)),
        'offset_mean_local': float(np.mean(local_offsets) / bins_per_day),
        'offset_std_local': float(np.std(local_offsets) / bins_per_day),
        'slot_support': float(anchor.support),
    }


def task_slot_ordinal_stats(
    series: SeriesBundle,
    target_week_idx: int | None,
    task_idx: int,
    slot_id: int,
    max_slots: int = 32,
    max_weeks: int = 104,
    bin_minutes: int = 5,
) -> dict[str, float]:
    prototypes = task_slot_prototypes(series, target_week_idx, task_idx, max_slots=max_slots, max_weeks=max_weeks)
    proto_map = {p.slot_id: p for p in prototypes}
    if slot_id not in proto_map:
        return {
            'ordinal_support': 0.0,
            'ordinal_start_median_norm': 0.0,
            'ordinal_start_std_norm': 0.0,
            'ordinal_day_mode_norm': 0.0,
            'ordinal_day_entropy': 0.0,
            'ordinal_active_weeks_norm': 0.0,
        }
    if target_week_idx is None:
        target_week_idx = len(series.week_starts)
    start = max(0, target_week_idx - max_weeks)
    starts: list[int] = []
    day_hist = np.zeros(7, dtype=np.float32)
    active_weeks = 0
    total_weeks = max(1, target_week_idx - start)
    bins_per_week = int(7 * 24 * 60 / bin_minutes)
    for week_events in series.events[start:target_week_idx]:
        task_events = [e for e in week_events if e.task_idx == task_idx]
        if task_events:
            active_weeks += 1
        for assigned_slot_id, event, _ in assign_events_to_prototypes(task_events, prototypes):
            if assigned_slot_id == slot_id:
                starts.append(int(event.start_bin))
                day_hist[int(event.start_time.dayofweek)] += 1.0
    ordinal_support = float(len(starts) / max(1, active_weeks))
    day_probs = day_hist / day_hist.sum() if day_hist.sum() > 0 else day_hist
    ordinal_day_mode = float(day_hist.argmax() / 6.0) if day_hist.sum() > 0 else 0.0
    return {
        'ordinal_support': ordinal_support,
        'ordinal_start_median_norm': float(np.median(starts) / max(1, bins_per_week - 1)) if starts else 0.0,
        'ordinal_start_std_norm': float(np.std(np.asarray(starts, dtype=np.float32)) / max(1, bins_per_week - 1)) if starts else 0.0,
        'ordinal_day_mode_norm': ordinal_day_mode,
        'ordinal_day_entropy': _normalized_entropy(day_probs) if day_hist.sum() > 0 else 0.0,
        'ordinal_active_weeks_norm': float(active_weeks / total_weeks),
    }


def build_slot_plan_features(
    planned_slots: list[dict[str, Any]],
    task_dispersion_by_task: dict[int, float] | None = None,
    bin_minutes: int = 5,
) -> dict[tuple[int, int], dict[str, float]]:
    if not planned_slots:
        return {}
    bins_per_day = int(24 * 60 / bin_minutes)
    sorted_slots = sorted(planned_slots, key=lambda x: (int(x['anchor_start_bin']), int(x.get('task_idx', 0)), int(x.get('slot_id', 0))))
    total_events = len(sorted_slots)
    task_groups: dict[int, list[dict[str, Any]]] = {}
    day_groups: dict[int, list[dict[str, Any]]] = {}
    day_task_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for slot in sorted_slots:
        task_groups.setdefault(int(slot['task_idx']), []).append(slot)
        day_idx = int(slot['anchor_start_bin']) // bins_per_day
        day_groups.setdefault(day_idx, []).append(slot)
        day_task_groups.setdefault((day_idx, int(slot['task_idx'])), []).append(slot)
    for rows in task_groups.values():
        rows.sort(key=lambda x: (int(x['anchor_start_bin']), int(x.get('slot_id', 0))))
    for rows in day_groups.values():
        rows.sort(key=lambda x: (int(x['anchor_start_bin']), int(x.get('task_idx', 0)), int(x.get('slot_id', 0))))
    for rows in day_task_groups.values():
        rows.sort(key=lambda x: (int(x['anchor_start_bin']), int(x.get('slot_id', 0))))

    features: dict[tuple[int, int], dict[str, float]] = {}
    for week_pos, slot in enumerate(sorted_slots):
        task_idx = int(slot['task_idx'])
        slot_id = int(slot['slot_id'])
        anchor = int(slot['anchor_start_bin'])
        key = (task_idx, slot_id)
        pred_task_count = max(1, int(slot.get('pred_task_count', 1)))
        template_task_count = max(0, int(slot.get('template_task_count', 0)))
        slot_support = float(slot.get('support', 0.0))
        task_rows = task_groups.get(task_idx, [slot])
        task_pos = next(i for i, row in enumerate(task_rows) if int(row['slot_id']) == slot_id)
        task_pos_pct = float(task_pos / max(pred_task_count - 1, 1))
        day_idx = anchor // bins_per_day
        day_rows = day_groups.get(day_idx, [slot])
        day_pos = next(i for i, row in enumerate(day_rows) if int(row['task_idx']) == task_idx and int(row['slot_id']) == slot_id)
        day_task_rows = day_task_groups.get((day_idx, task_idx), [slot])
        same_task_day_pos = next(i for i, row in enumerate(day_task_rows) if int(row['slot_id']) == slot_id)
        prev_gap = float(anchor - int(sorted_slots[week_pos - 1]['anchor_start_bin'])) if week_pos > 0 else 0.0
        next_gap = float(int(sorted_slots[week_pos + 1]['anchor_start_bin']) - anchor) if week_pos + 1 < total_events else 0.0
        prev_60 = sum(1 for other in sorted_slots if 0 < anchor - int(other['anchor_start_bin']) <= int(round(60 / bin_minutes)))
        next_60 = sum(1 for other in sorted_slots if 0 < int(other['anchor_start_bin']) - anchor <= int(round(60 / bin_minutes)))
        prev_day_same_task = sum(1 for other in task_rows if 0 < anchor - int(other['anchor_start_bin']) <= bins_per_day)
        next_day_same_task = sum(1 for other in task_rows if 0 < int(other['anchor_start_bin']) - anchor <= bins_per_day)
        task_dispersion = float((task_dispersion_by_task or {}).get(task_idx, 0.0))
        day_pct = float(day_pos / max(len(day_rows) - 1, 1))
        same_task_day_pct = float(same_task_day_pos / max(len(day_task_rows) - 1, 1))
        week_pct = float(week_pos / max(total_events - 1, 1))
        features[key] = {
            'task_position_abs': float(task_pos),
            'task_position_pct': task_pos_pct,
            'is_first_of_type': 1.0 if task_pos == 0 else 0.0,
            'is_last_of_type': 1.0 if task_pos == pred_task_count - 1 else 0.0,
            'is_middle_of_type': 1.0 if pred_task_count > 2 and 0 < task_pos < pred_task_count - 1 else 0.0,
            'week_order_abs': float(week_pos),
            'week_order_pct': week_pct,
            'num_events_before': float(week_pos),
            'num_events_after': float(total_events - week_pos - 1),
            'day_order_abs': float(day_pos),
            'day_order_pct': day_pct,
            'same_task_day_order_abs': float(same_task_day_pos),
            'same_task_day_order_pct': same_task_day_pct,
            'gap_prev_anchor_bins': prev_gap,
            'gap_next_anchor_bins': next_gap,
            'has_prev_anchor': 1.0 if week_pos > 0 else 0.0,
            'has_next_anchor': 1.0 if week_pos + 1 < total_events else 0.0,
            'num_events_in_prev_60m': float(prev_60),
            'num_events_in_next_60m': float(next_60),
            'num_same_task_in_prev_day': float(prev_day_same_task),
            'num_same_task_in_next_day': float(next_day_same_task),
            'position_x_pred_task_count': task_pos_pct * float(pred_task_count),
            'position_x_template_task_count': task_pos_pct * float(template_task_count),
            'position_x_slot_support': task_pos_pct * slot_support,
            'position_x_task_dispersion': task_pos_pct * task_dispersion,
            'planned_total_events': float(total_events),
            'planned_events_this_day': float(len(day_rows)),
        }
    return features


def build_temporal_slot_context(
    series: SeriesBundle,
    target_week_idx: int | None,
    task_idx: int,
    slot_id: int,
    max_slots: int = 32,
    bin_minutes: int = 5,
) -> dict[str, Any]:
    bins_per_day = int(24 * 60 / bin_minutes)
    profile = task_temporal_profile(series, target_week_idx, task_idx, bin_minutes=bin_minutes)
    slot_stats = slot_recent_start_offsets(series, target_week_idx, task_idx, slot_id, max_slots=max_slots, bin_minutes=bin_minutes)
    ordinal_stats = task_slot_ordinal_stats(series, target_week_idx, task_idx, slot_id, max_slots=max_slots, max_weeks=104, bin_minutes=bin_minutes)
    duration_med = task_duration_median(series, target_week_idx, task_idx)
    slot_scale = float(max(1, int(max_slots) - 1))
    return {
        'profile': profile,
        'slot_stats': slot_stats,
        'ordinal_stats': ordinal_stats,
        'duration_med': float(duration_med),
        'bins_per_day': bins_per_day,
        'slot_scale': slot_scale,
        'max_start_norm': float(max(1, 7 * bins_per_day - 1)),
    }


def build_occurrence_numeric_features(series: SeriesBundle, target_week_idx: int | None, task_idx: int, template_count: int, support_mean: float, primary_score: float) -> tuple[np.ndarray, dict[str, float]]:
    seasonal = seasonal_count_baseline(series, target_week_idx, task_idx, template_count)
    profile = task_temporal_profile(series, target_week_idx, task_idx)
    recent_4 = per_task_recent_stats(series, target_week_idx, task_idx, window=4)
    recent_12 = per_task_recent_stats(series, target_week_idx, task_idx, window=12)
    lag_1, lag_1_mask = task_count_lag(series, target_week_idx, task_idx, lag=1)
    lag_52, lag_52_mask = task_count_lag(series, target_week_idx, task_idx, lag=52)
    calendar_ctx = occurrence_calendar_context(series, target_week_idx)
    trend_ctx = series_trend_features(series, target_week_idx, task_idx)
    yoy_delta = recent_4['recent_last'] - lag_52 if lag_52_mask > 0 else 0.0
    lag_52_ratio = (recent_4['recent_last'] / max(lag_52, 1.0)) if lag_52_mask > 0 else 0.0
    numeric = np.asarray([
        seasonal['baseline_count'],
        float(template_count),
        seasonal['seasonal_count'],
        lag_1,
        seasonal['lag_4'],
        seasonal['lag_26'],
        seasonal['lag_52'],
        recent_4['recent_mean'],
        recent_4['recent_last'],
        recent_4['recent_median'],
        recent_4['recent_std'],
        recent_4['recent_max'],
        recent_12['recent_mean'],
        recent_12['recent_std'],
        yoy_delta,
        lag_52_ratio,
        profile['weekly_frequency'],
        profile['start_dispersion'],
        profile['active_days_norm'],
        profile['day_multimodality_norm'],
        profile['support_norm'],
        float(support_mean),
        float(primary_score),
        profile['mean_start_norm'],
        profile['duration_median_norm'],
        calendar_ctx['sin_week_of_year'],
        calendar_ctx['cos_week_of_year'],
        calendar_ctx['sin_month'],
        calendar_ctx['cos_month'],
        calendar_ctx['is_peak_season'],
        calendar_ctx['is_trough_season'],
        calendar_ctx['dist_to_peak_month'],
        calendar_ctx['monthly_load_index'],
        calendar_ctx['n_weekdays_in_week'],
        calendar_ctx['n_weekend_days_in_week'],
        calendar_ctx['n_night_sessions_possible'],
        calendar_ctx['expected_night_tasks'],
        calendar_ctx['week_of_year_norm'],
        calendar_ctx['year_norm'],
        calendar_ctx['year_fraction'],
        calendar_ctx['is_transition_week'],
        float(trend_ctx['weeks_elapsed_total']),
        float(trend_ctx['trend_tasks_per_week']),
        lag_1_mask,
        lag_52_mask,
    ], dtype=np.float32)
    info = dict(seasonal)
    info.update(profile)
    info.update(recent_4)
    info.update({f'recent12_{k}': v for k, v in recent_12.items()})
    info.update(calendar_ctx)
    info.update(trend_ctx)
    info['lag_1'] = float(lag_1)
    info['lag_1_mask'] = float(lag_1_mask)
    info['lag_52_mask'] = float(lag_52_mask)
    info['support_mean'] = float(support_mean)
    info['primary_score'] = float(primary_score)
    info['yoy_delta'] = float(yoy_delta)
    info['lag_52_ratio'] = float(lag_52_ratio)
    return numeric, info


def build_temporal_numeric_features(
    series: SeriesBundle,
    target_week_idx: int | None,
    task_idx: int,
    slot_id: int,
    anchor_start: int,
    anchor_duration: int,
    pred_task_count: int,
    template_task_count: int,
    slot_support: float,
    max_slots: int = 32,
    bin_minutes: int = 5,
    slot_context: dict[str, Any] | None = None,
    plan_features: dict[str, float] | None = None,
) -> np.ndarray:
    slot_context = slot_context or build_temporal_slot_context(
        series,
        target_week_idx,
        task_idx,
        slot_id,
        max_slots=max_slots,
        bin_minutes=bin_minutes,
    )
    plan_features = plan_features or {}
    bins_per_day = int(slot_context['bins_per_day'])
    profile = slot_context['profile']
    slot_stats = slot_context['slot_stats']
    ordinal_stats = slot_context['ordinal_stats']
    duration_med = float(slot_context['duration_med'])
    slot_scale = float(slot_context['slot_scale'])
    day_idx = int(anchor_start // bins_per_day)
    local_bin = int(anchor_start % bins_per_day)
    day_angle = 2.0 * np.pi * float(day_idx) / 7.0
    hour_angle = 2.0 * np.pi * float(local_bin) / max(1, bins_per_day)
    return np.asarray([
        float(anchor_start) / float(slot_context['max_start_norm']),
        float(day_idx) / 6.0,
        float(local_bin) / max(1, bins_per_day - 1),
        float(anchor_duration) / 12.0,
        float(slot_id) / slot_scale,
        float(plan_features.get('task_position_abs', 0.0)),
        float(plan_features.get('task_position_pct', 0.0)),
        float(plan_features.get('is_first_of_type', 0.0)),
        float(plan_features.get('is_last_of_type', 0.0)),
        float(plan_features.get('is_middle_of_type', 0.0)),
        float(pred_task_count),
        float(template_task_count),
        float(slot_support),
        float(plan_features.get('week_order_abs', 0.0)),
        float(plan_features.get('week_order_pct', 0.0)),
        float(plan_features.get('num_events_before', 0.0)),
        float(plan_features.get('num_events_after', 0.0)),
        float(plan_features.get('day_order_abs', 0.0)),
        float(plan_features.get('day_order_pct', 0.0)),
        float(plan_features.get('same_task_day_order_abs', 0.0)),
        float(plan_features.get('same_task_day_order_pct', 0.0)),
        float(plan_features.get('gap_prev_anchor_bins', 0.0)) / max(1, bins_per_day),
        float(plan_features.get('gap_next_anchor_bins', 0.0)) / max(1, bins_per_day),
        float(plan_features.get('has_prev_anchor', 0.0)),
        float(plan_features.get('has_next_anchor', 0.0)),
        float(plan_features.get('num_events_in_prev_60m', 0.0)),
        float(plan_features.get('num_events_in_next_60m', 0.0)),
        float(plan_features.get('num_same_task_in_prev_day', 0.0)),
        float(plan_features.get('num_same_task_in_next_day', 0.0)),
        profile['weekly_frequency'],
        profile['start_dispersion'],
        profile['active_days_norm'],
        profile['day_multimodality_norm'],
        profile['mean_start_norm'],
        profile['duration_median_norm'],
        slot_stats['offset_mean_days'],
        slot_stats['offset_mean_local'],
        slot_stats['offset_std_local'],
        float(duration_med) / 12.0,
        ordinal_stats['ordinal_support'],
        ordinal_stats['ordinal_start_median_norm'],
        ordinal_stats['ordinal_start_std_norm'],
        ordinal_stats['ordinal_day_mode_norm'],
        ordinal_stats['ordinal_day_entropy'],
        ordinal_stats['ordinal_active_weeks_norm'],
        float(plan_features.get('position_x_pred_task_count', 0.0)),
        float(plan_features.get('position_x_template_task_count', 0.0)),
        float(plan_features.get('position_x_slot_support', 0.0)),
        float(plan_features.get('position_x_task_dispersion', 0.0)),
        float(plan_features.get('planned_total_events', 0.0)),
        float(plan_features.get('planned_events_this_day', 0.0)),
        float(np.sin(day_angle)),
        float(np.cos(day_angle)),
        float(np.sin(hour_angle)),
        float(np.cos(hour_angle)),
        1.0 if day_idx < 5 else 0.0,
    ], dtype=np.float32)


def build_temporal_candidate_features(
    series: SeriesBundle,
    target_week_idx: int | None,
    task_idx: int,
    slot_id: int,
    anchor_start: int,
    anchor_duration: int,
    candidate_start: int,
    candidate_duration: int,
    support_score: float,
    max_slots: int = 32,
    bin_minutes: int = 5,
    slot_context: dict[str, Any] | None = None,
    plan_features: dict[str, float] | None = None,
) -> np.ndarray:
    slot_context = slot_context or build_temporal_slot_context(
        series,
        target_week_idx,
        task_idx,
        slot_id,
        max_slots=max_slots,
        bin_minutes=bin_minutes,
    )
    plan_features = plan_features or {}
    bins_per_day = int(slot_context['bins_per_day'])
    max_start_norm = float(slot_context['max_start_norm'])
    profile = slot_context['profile']
    slot_stats = slot_context['slot_stats']
    ordinal_stats = slot_context['ordinal_stats']
    duration_med = float(slot_context['duration_med'])

    candidate_start = int(candidate_start)
    anchor_start = int(anchor_start)
    candidate_duration = max(1, int(candidate_duration))
    anchor_duration = max(1, int(anchor_duration))
    diff = float(candidate_start - anchor_start)
    day_gap = float((candidate_start // bins_per_day) - (anchor_start // bins_per_day))
    local_gap = float((candidate_start % bins_per_day) - (anchor_start % bins_per_day))
    angle = 2.0 * np.pi * float(candidate_start) / max_start_norm
    candidate_day = int(candidate_start // bins_per_day)
    candidate_local = int(candidate_start % bins_per_day)
    day_angle = 2.0 * np.pi * float(candidate_day) / 7.0
    hour_angle = 2.0 * np.pi * float(candidate_local) / max(1, bins_per_day)
    return np.asarray([
        float(candidate_start) / max_start_norm,
        float(candidate_day) / 6.0,
        float(candidate_local) / max(1, bins_per_day - 1),
        float(candidate_duration) / 12.0,
        float(np.log1p(max(0.0, float(support_score)))),
        diff / max_start_norm,
        abs(diff) / max_start_norm,
        day_gap / 6.0,
        local_gap / max(1, bins_per_day - 1),
        float(candidate_duration - anchor_duration) / 12.0,
        1.0 if candidate_start == anchor_start else 0.0,
        float(candidate_start) / max_start_norm - profile['mean_start_norm'],
        float(candidate_duration) / 12.0 - profile['duration_median_norm'],
        day_gap - slot_stats['offset_mean_days'],
        float(local_gap) / max(1, bins_per_day - 1) - slot_stats['offset_mean_local'],
        np.sin(angle),
        np.cos(angle),
        float(candidate_duration - duration_med) / 12.0,
        float(candidate_start) / max_start_norm - ordinal_stats['ordinal_start_median_norm'],
        float(candidate_day) / 6.0 - ordinal_stats['ordinal_day_mode_norm'],
        float(np.sin(day_angle)),
        float(np.cos(day_angle)),
        float(np.sin(hour_angle)),
        float(np.cos(hour_angle)),
        1.0 if candidate_day < 5 else 0.0,
        float(plan_features.get('task_position_pct', 0.0)),
        float(plan_features.get('week_order_pct', 0.0)),
        float(plan_features.get('day_order_pct', 0.0)),
    ], dtype=np.float32)
