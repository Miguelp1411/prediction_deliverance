from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


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


def task_slot_prototypes(series: SeriesBundle, target_week_idx: int | None, task_idx: int, max_slots: int = 8, max_weeks: int = 104) -> list[SlotPrototype]:
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


def slot_recent_start_offsets(series: SeriesBundle, target_week_idx: int | None, task_idx: int, slot_id: int, max_weeks: int = 26, bin_minutes: int = 5) -> dict[str, float]:
    prototypes = task_slot_prototypes(series, target_week_idx, task_idx, max_weeks=max_weeks)
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
    return {
        'offset_mean_days': float(np.mean(offsets_arr / bins_per_day)),
        'offset_mean_local': float(np.mean(offsets_arr % bins_per_day) / bins_per_day),
        'offset_std_local': float(np.std(offsets_arr) / bins_per_day),
        'slot_support': float(anchor.support),
    }


def build_occurrence_numeric_features(series: SeriesBundle, target_week_idx: int | None, task_idx: int, template_count: int, support_mean: float, primary_score: float) -> tuple[np.ndarray, dict[str, float]]:
    seasonal = seasonal_count_baseline(series, target_week_idx, task_idx, template_count)
    profile = task_temporal_profile(series, target_week_idx, task_idx)
    numeric = np.asarray([
        seasonal['baseline_count'],
        float(template_count),
        seasonal['seasonal_count'],
        seasonal['lag_4'],
        seasonal['lag_26'],
        seasonal['lag_52'],
        seasonal['recent_mean'],
        seasonal['recent_last'],
        seasonal['recent_median'],
        seasonal['recent_std'],
        profile['weekly_frequency'],
        profile['start_dispersion'],
        profile['active_days_norm'],
        profile['day_multimodality_norm'],
        profile['support_norm'],
        float(support_mean),
        float(primary_score),
        profile['mean_start_norm'],
        profile['duration_median_norm'],
    ], dtype=np.float32)
    info = dict(seasonal)
    info.update(profile)
    info['support_mean'] = float(support_mean)
    info['primary_score'] = float(primary_score)
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
    bin_minutes: int = 5,
) -> np.ndarray:
    bins_per_day = int(24 * 60 / bin_minutes)
    profile = task_temporal_profile(series, target_week_idx, task_idx, bin_minutes=bin_minutes)
    slot_stats = slot_recent_start_offsets(series, target_week_idx, task_idx, slot_id, bin_minutes=bin_minutes)
    duration_med = task_duration_median(series, target_week_idx, task_idx)
    return np.asarray([
        float(anchor_start) / max(1, 7 * bins_per_day - 1),
        float(anchor_start // bins_per_day) / 6.0,
        float(anchor_start % bins_per_day) / max(1, bins_per_day - 1),
        float(anchor_duration) / 12.0,
        float(slot_id) / 7.0,
        float(pred_task_count) / 12.0,
        float(template_task_count) / 12.0,
        float(slot_support),
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
    ], dtype=np.float32)
