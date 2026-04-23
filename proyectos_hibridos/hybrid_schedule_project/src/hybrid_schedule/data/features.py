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
    woy_angle = 2.0 * np.pi * (iso_week - 1) / 52.0
    month_angle = 2.0 * np.pi * (month - 1) / 12.0
    return np.asarray([
        np.sin(woy_angle), np.cos(woy_angle),
        np.sin(month_angle), np.cos(month_angle),
    ], dtype=np.float32)



def build_history_tensor(series: SeriesBundle, target_week_idx: int, window_weeks: int) -> np.ndarray:
    start = max(0, target_week_idx - window_weeks)
    hist_counts = series.counts[start:target_week_idx].astype(np.float32)
    hist_day = series.day_hist[start:target_week_idx].reshape(target_week_idx - start, -1).astype(np.float32)
    hist_start = series.mean_start[start:target_week_idx].astype(np.float32) / (7 * 24 * 60 / 5)
    hist_dur = series.mean_duration[start:target_week_idx].astype(np.float32) / 12.0

    feats = []
    for idx in range(target_week_idx - start):
        week_features = np.concatenate([
            hist_counts[idx],
            hist_day[idx],
            hist_start[idx],
            hist_dur[idx],
            calendar_features(series.week_starts[start + idx]),
        ]).astype(np.float32)
        feats.append(week_features)
    if not feats:
        return np.zeros((window_weeks, series.counts.shape[1] * 10 + 4), dtype=np.float32)
    feat_arr = np.stack(feats, axis=0)
    if feat_arr.shape[0] < window_weeks:
        pad = np.zeros((window_weeks - feat_arr.shape[0], feat_arr.shape[1]), dtype=np.float32)
        feat_arr = np.concatenate([pad, feat_arr], axis=0)
    return feat_arr[-window_weeks:]



def build_future_history_tensor(series: SeriesBundle, window_weeks: int) -> np.ndarray:
    return build_history_tensor(series, len(series.week_starts), window_weeks)



def per_task_recent_stats(series: SeriesBundle, target_week_idx: int, task_idx: int, window: int = 4) -> dict[str, float]:
    if target_week_idx is None:
        target_week_idx = len(series.week_starts) 
        
    start = max(0, target_week_idx - window)
    values = series.counts[start:target_week_idx, task_idx].astype(np.float32)
    if values.size == 0:
        return {'recent_mean': 0.0, 'recent_last': 0.0, 'recent_median': 0.0}
    return {
        'recent_mean': float(values.mean()),
        'recent_last': float(values[-1]),
        'recent_median': float(np.median(values)),
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
