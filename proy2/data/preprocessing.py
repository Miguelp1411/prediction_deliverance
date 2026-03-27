from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config import (
    BIN_MINUTES,
    HISTORY_SCALES,
    MAX_OCCURRENCES_PER_TASK,
    MAX_TASKS_PER_WEEK,
    WINDOW_WEEKS,
    num_time_bins,
)


RECENT_FREQ_SCALES = (2, 4, 12)


@dataclass
class EventRecord:
    task_id: int
    task_name: str
    start_bin: int
    duration_minutes: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


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


def _build_events(df: pd.DataFrame, task_to_id: dict[str, int], dur_min: float, dur_max: float) -> pd.DataFrame:
    out = df.copy()
    out["task_id"] = out["task_name"].map(task_to_id)
    out["week_start"] = out["start_time"].map(_week_start)
    out["day_of_week"] = out["start_time"].dt.dayofweek.astype(int)
    out["hour"] = out["start_time"].dt.hour.astype(int)
    out["minute"] = out["start_time"].dt.minute.astype(int)
    out["minute_bin"] = (out["minute"] // BIN_MINUTES).astype(int)
    out["start_bin"] = (
        out["day_of_week"] * (24 * 60 // BIN_MINUTES)
        + out["hour"] * (60 // BIN_MINUTES)
        + out["minute_bin"]
    ).astype(int)
    out["duration_norm"] = out["duration_minutes"].map(lambda x: _duration_to_norm(x, dur_min, dur_max))
    return out


def _continuous_week_starts(df_events: pd.DataFrame) -> list[pd.Timestamp]:
    first_week = df_events["week_start"].min()
    last_week = df_events["week_start"].max()
    return list(pd.date_range(start=first_week, end=last_week, freq="7D", tz=first_week.tz))


def _compute_week_features(
    grouped: pd.DataFrame,
    week_index: int,
    week_start: pd.Timestamp,
    num_tasks: int,
    max_count_cap: int,
) -> WeekRecord:
    counts = np.zeros(num_tasks, dtype=np.float32)
    mean_start_sin = np.zeros(num_tasks, dtype=np.float32)
    mean_start_cos = np.zeros(num_tasks, dtype=np.float32)
    mean_duration_norm = np.zeros(num_tasks, dtype=np.float32)
    events_by_task: dict[int, list[EventRecord]] = {task_id: [] for task_id in range(num_tasks)}

    if not grouped.empty:
        grouped = grouped.sort_values(["task_id", "start_bin", "start_time"]).reset_index(drop=True)
        if len(grouped) > MAX_TASKS_PER_WEEK:
            raise ValueError(
                f"La semana {week_start} tiene {len(grouped)} tareas y supera MAX_TASKS_PER_WEEK={MAX_TASKS_PER_WEEK}."
            )

        for task_id, task_df in grouped.groupby("task_id"):
            task_df = task_df.sort_values(["start_bin", "start_time"])
            n = int(len(task_df))
            counts[task_id] = float(min(n, max_count_cap))

            angles = 2.0 * np.pi * task_df["start_bin"].to_numpy(dtype=np.float32) / num_time_bins()
            mean_start_sin[task_id] = float(np.mean(np.sin(angles)))
            mean_start_cos[task_id] = float(np.mean(np.cos(angles)))
            mean_duration_norm[task_id] = float(task_df["duration_norm"].mean())

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
    week_number = int(iso.week)
    month = int(week_start.month)
    day_of_year = int(week_start.dayofyear)

    week_of_year_sin, week_of_year_cos = _cyclical_encode(week_number - 1, 52.0)
    month_sin, month_cos = _cyclical_encode(month - 1, 12.0)
    day_of_year_sin, day_of_year_cos = _cyclical_encode(day_of_year - 1, 366.0)

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
    return np.concatenate(
        [
            week.counts,
            week.mean_start_sin,
            week.mean_start_cos,
            week.mean_duration_norm,
            np.array(
                [
                    week.total_tasks_norm,
                    week.week_of_year_sin,
                    week.week_of_year_cos,
                    week.month_sin,
                    week.month_cos,
                    week.day_of_year_sin,
                    week.day_of_year_cos,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)


def _infer_target_week_start(weeks: list[WeekRecord], target_week_index: int) -> pd.Timestamp:
    if not weeks:
        raise ValueError("No hay semanas disponibles para inferir target_week_start.")

    if 0 <= target_week_index < len(weeks):
        return weeks[target_week_index].week_start

    if target_week_index == len(weeks):
        return weeks[-1].week_start + pd.Timedelta(days=7)

    first_week = weeks[0].week_start
    return first_week + pd.Timedelta(days=7 * target_week_index)


def _history_slice(weeks: list[WeekRecord], target_week_index: int) -> list[WeekRecord]:
    start = max(0, target_week_index - WINDOW_WEEKS)
    end = min(target_week_index, len(weeks))
    return weeks[start:end]


def _recent_count_frequency(history: list[WeekRecord], task_id: int, k: int) -> float:
    if not history:
        return 0.0
    window = history[-min(k, len(history)):]
    if not window:
        return 0.0
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
    start_bins: list[int] = []
    for week in history:
        for event in week.events_by_task.get(task_id, []):
            start_bins.append(int(event.start_bin))

    if not start_bins:
        return None
    return float(np.mean(start_bins))


def _recent_mean_duration_norm(history: list[WeekRecord], task_id: int, dur_min: float, dur_max: float) -> float:
    durations: list[float] = []
    for week in history:
        for event in week.events_by_task.get(task_id, []):
            durations.append(float(event.duration_minutes))

    if not durations:
        return 0.0

    mean_duration = float(np.mean(durations))
    return _duration_to_norm(mean_duration, dur_min, dur_max)


def build_history_features(
    weeks: list[WeekRecord],
    target_week_index: int,
    task_id: int,
    duration_min: float | None = None,
    duration_max: float | None = None,
) -> np.ndarray:
    history = _history_slice(weeks, target_week_index)
    target_week_start = _infer_target_week_start(weeks, target_week_index)

    if duration_min is None:
        duration_min = 0.0
    if duration_max is None:
        duration_max = 1.0

    # Bloque histórico base por escalas
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

    # Estadísticos históricos generales
    if len(counts_per_week) > 0:
        nonzero_indices = np.where(counts_per_week > 0)[0]
        weeks_since_last = (
            float(len(history) - 1 - nonzero_indices[-1]) if len(nonzero_indices) > 0 else float(WINDOW_WEEKS)
        )

        feats.extend(
            [
                float(counts_per_week.mean() / MAX_OCCURRENCES_PER_TASK),
                float(counts_per_week.std() / max(MAX_OCCURRENCES_PER_TASK, 1)),
                float((counts_per_week > 0).mean()),
                float(min(weeks_since_last, WINDOW_WEEKS) / max(WINDOW_WEEKS, 1)),
                float(np.clip(np.sum(counts_per_week) / (WINDOW_WEEKS * MAX_OCCURRENCES_PER_TASK), 0.0, 1.0)),
            ]
        )
    else:
        feats.extend([0.0, 0.0, 0.0, 1.0, 0.0])

    # Nuevas features de calendario de la semana objetivo
    iso = target_week_start.isocalendar()
    week_number = int(iso.week)
    month = int(target_week_start.month)
    day_of_year = int(target_week_start.dayofyear)

    week_sin, week_cos = _cyclical_encode(week_number - 1, 52.0)
    month_sin, month_cos = _cyclical_encode(month - 1, 12.0)
    doy_sin, doy_cos = _cyclical_encode(day_of_year - 1, 366.0)

    feats.extend([week_sin, week_cos, month_sin, month_cos, doy_sin, doy_cos])

    # Tiempo desde la última aparición
    days_since_last = _days_since_last_occurrence(history, task_id, target_week_start)
    max_days_norm = float(max(12 * 7, WINDOW_WEEKS * 7))
    days_since_last_norm = float(np.clip(days_since_last / max_days_norm, 0.0, 1.0))
    feats.append(days_since_last_norm)

    # Frecuencia en últimas 2, 4 y 12 semanas
    for scale in RECENT_FREQ_SCALES:
        feats.append(_recent_count_frequency(history, task_id, scale))

    # Media reciente de horario en seno/coseno
    mean_start_bin = _recent_mean_start_bin(history, task_id)
    if mean_start_bin is None:
        feats.extend([0.0, 0.0])
    else:
        start_sin, start_cos = _cyclical_encode(mean_start_bin, num_time_bins())
        feats.extend([start_sin, start_cos])

    # Media reciente de duración
    mean_duration_norm = _recent_mean_duration_norm(history, task_id, duration_min, duration_max)
    feats.append(mean_duration_norm)

    return np.array(feats, dtype=np.float32)


def prepare_data(df: pd.DataFrame, train_ratio: float) -> PreparedData:
    task_names = sorted(df["task_name"].unique().tolist())
    task_to_id = {name: idx for idx, name in enumerate(task_names)}

    duration_min = float(df["duration_minutes"].min())
    duration_max = float(df["duration_minutes"].max())
    df_events = _build_events(df, task_to_id, duration_min, duration_max)

    week_starts = _continuous_week_starts(df_events)
    split_week_idx = max(int(len(week_starts) * train_ratio), WINDOW_WEEKS + 1)
    train_weeks_df = df_events[df_events["week_start"].isin(week_starts[:split_week_idx])]
    max_count_cap = int(train_weeks_df.groupby(["week_start", "task_id"]).size().max()) if not train_weeks_df.empty else 1
    max_count_cap = max(1, min(max_count_cap, MAX_OCCURRENCES_PER_TASK))

    weeks: list[WeekRecord] = []
    for week_index, week_start in enumerate(week_starts):
        week_df = df_events[df_events["week_start"] == week_start].copy()
        weeks.append(
            _compute_week_features(
                grouped=week_df,
                week_index=week_index,
                week_start=week_start,
                num_tasks=len(task_names),
                max_count_cap=max_count_cap,
            )
        )

    sample_feature = (
        week_to_feature_vector(weeks[0])
        if weeks
        else np.zeros(4 * len(task_names) + 7, dtype=np.float32)
    )

    sample_history = (
        build_history_features(
            weeks=weeks,
            target_week_index=min(WINDOW_WEEKS, len(weeks)),
            task_id=0,
            duration_min=duration_min,
            duration_max=duration_max,
        )
        if task_names
        else np.zeros(5 * len(HISTORY_SCALES) + 15, dtype=np.float32)
    )

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
        "task_names": prepared.task_names,
        "duration_min": prepared.duration_min,
        "duration_max": prepared.duration_max,
        "max_count_cap": prepared.max_count_cap,
        "week_feature_dim": prepared.week_feature_dim,
        "history_feature_dim": prepared.history_feature_dim,
        "window_weeks": WINDOW_WEEKS,
        "bin_minutes": BIN_MINUTES,
        "num_time_bins": num_time_bins(),
    }