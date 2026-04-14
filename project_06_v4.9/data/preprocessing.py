"""
Data preprocessing — converts canonical Events into WeekRecords and PreparedData.

Builds the weekly grid, computes per-week features, and assembles
the full PreparedData bundle used by all downstream modules.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from data.schema import (
    DatabaseProfile,
    Event,
    EventRecord,
    PreparedData,
    WeekRecord,
)


# ── Helpers ──────────────────────────────────────────────────────────
def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize() - pd.Timedelta(days=ts.weekday())


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _circular_mean_bin(bins: list[int], total_bins: int) -> tuple[float, float]:
    if not bins:
        return 0.0, 0.0
    angles = [2 * math.pi * b / total_bins for b in bins]
    s = sum(math.sin(a) for a in angles) / len(angles)
    c = sum(math.cos(a) for a in angles) / len(angles)
    return s, c


def _circular_dispersion(mean_sin: float, mean_cos: float) -> float:
    r = math.sqrt(mean_sin ** 2 + mean_cos ** 2)
    return 1.0 - r


# ── Main preparation ────────────────────────────────────────────────
def prepare_data(
    events: list[Event],
    cfg: SimpleNamespace,
    profiles: dict[str, DatabaseProfile] | None = None,
) -> PreparedData:
    """
    Convert a list of canonical Events into PreparedData.

    Supports events from multiple databases (database_id as feature).
    """
    bin_minutes = cfg.project.bin_minutes
    total_weekly_bins = (7 * 24 * 60) // bin_minutes
    bins_per_day = (24 * 60) // bin_minutes

    if not events:
        return PreparedData(bin_minutes=bin_minutes)

    # ── Discover tasks and databases ─────────────────────────────
    task_names = sorted(set(e.task_type for e in events))
    task_to_id = {t: i for i, t in enumerate(task_names)}
    num_tasks = len(task_names)

    database_ids = sorted(set(e.database_id for e in events))
    db_to_id = {d: i for i, d in enumerate(database_ids)}

    # ── Duration range ───────────────────────────────────────────
    all_durations = [e.duration_minutes for e in events if e.duration_minutes > 0]
    duration_min = min(all_durations) if all_durations else 0.0
    duration_max = max(all_durations) if all_durations else 1.0
    dur_span = max(duration_max - duration_min, 1e-6)

    # ── Task duration medians ────────────────────────────────────
    task_durations: dict[str, list[float]] = defaultdict(list)
    for e in events:
        task_durations[e.task_type].append(e.duration_minutes)
    task_duration_medians = {
        t: float(np.median(durs)) for t, durs in task_durations.items()
    }

    # ── Group events by week ─────────────────────────────────────
    events_by_week: dict[pd.Timestamp, list[Event]] = defaultdict(list)
    for e in events:
        ws = _week_start(e.start_time)
        events_by_week[ws].append(e)

    sorted_week_starts = sorted(events_by_week.keys())

    # ── Caps ─────────────────────────────────────────────────────
    max_occ = getattr(cfg.data, "default_max_occurrences_per_task", 50)
    max_tasks_wk = getattr(cfg.data, "default_max_tasks_per_week", 100)

    # Infer from data
    for ws in sorted_week_starts:
        wk_events = events_by_week[ws]
        counts_per_task: dict[str, int] = defaultdict(int)
        for e in wk_events:
            counts_per_task[e.task_type] += 1
        if counts_per_task:
            max_occ = max(max_occ, max(counts_per_task.values()) + 2)
        max_tasks_wk = max(max_tasks_wk, len(wk_events) + 5)

    max_count_cap = max_occ + 1  # +1 for zero class

    # ── Build WeekRecords ────────────────────────────────────────
    weeks: list[WeekRecord] = []
    for week_idx, ws in enumerate(sorted_week_starts):
        wk_events = events_by_week[ws]

        # Counts per task
        counts = np.zeros(num_tasks, dtype=np.int64)
        # Circular mean start per task
        mean_sin = np.zeros(num_tasks, dtype=np.float32)
        mean_cos = np.zeros(num_tasks, dtype=np.float32)
        # Mean duration normalised
        mean_dur = np.zeros(num_tasks, dtype=np.float32)
        # Dispersion
        dispersion = np.zeros(num_tasks, dtype=np.float32)
        # Active days
        active_days = np.zeros(num_tasks, dtype=np.float32)
        # Day distribution (num_tasks, 7)
        day_dist = np.zeros((num_tasks, 7), dtype=np.float32)

        events_by_task: dict[int, list[EventRecord]] = defaultdict(list)
        all_event_records: list[EventRecord] = []

        # Determine dominant database_id for this week
        db_counter: dict[str, int] = defaultdict(int)
        for e in wk_events:
            db_counter[e.database_id] += 1
        week_db_id = max(db_counter, key=db_counter.get) if db_counter else ""

        for e in wk_events:
            tid = task_to_id[e.task_type]

            # Compute start_bin in weekly grid
            delta = e.start_time - ws
            start_bin = int(delta.total_seconds() / 60 / bin_minutes)
            start_bin = max(0, min(start_bin, total_weekly_bins - 1))

            er = EventRecord(
                task_id=tid,
                task_name=e.task_type,
                start_bin=start_bin,
                duration_minutes=e.duration_minutes,
                start_time=e.start_time,
                end_time=e.end_time,
                device_id=e.device_id,
                database_id=e.database_id,
                robot_id=e.robot_id,
            )
            events_by_task[tid].append(er)
            all_event_records.append(er)
            counts[tid] += 1
            day_dist[tid, e.start_time.weekday()] += 1

        # Per-task aggregation
        for tid in range(num_tasks):
            task_evts = events_by_task.get(tid, [])
            if not task_evts:
                continue

            bins_list = [er.start_bin for er in task_evts]
            s, c = _circular_mean_bin(bins_list, total_weekly_bins)
            mean_sin[tid] = s
            mean_cos[tid] = c
            dispersion[tid] = _circular_dispersion(s, c)

            dur_vals = [er.duration_minutes for er in task_evts]
            mean_dur[tid] = (np.mean(dur_vals) - duration_min) / dur_span

            days_set = set(er.start_time.weekday() for er in task_evts)
            active_days[tid] = len(days_set) / 7.0

            # Normalise day distribution
            row_sum = day_dist[tid].sum()
            if row_sum > 0:
                day_dist[tid] /= row_sum

        # Calendar features
        woy = ws.isocalendar()[1]
        woy_sin, woy_cos = _cyclical_encode(woy, 52)
        month_sin, month_cos = _cyclical_encode(ws.month, 12)
        doy = ws.timetuple().tm_yday
        doy_sin, doy_cos = _cyclical_encode(doy, 365)

        total_norm = len(wk_events) / max(max_tasks_wk, 1)

        wr = WeekRecord(
            week_index=week_idx,
            week_start=ws,
            database_id=week_db_id,
            counts=counts,
            mean_start_sin=mean_sin,
            mean_start_cos=mean_cos,
            mean_duration_norm=mean_dur,
            start_circular_dispersion=dispersion,
            active_days_norm=active_days,
            day_distribution=day_dist.flatten(),
            total_tasks_norm=total_norm,
            week_of_year_sin=woy_sin,
            week_of_year_cos=woy_cos,
            month_sin=month_sin,
            month_cos=month_cos,
            day_of_year_sin=doy_sin,
            day_of_year_cos=doy_cos,
            events_by_task=dict(events_by_task),
            events=all_event_records,
        )
        weeks.append(wr)

    # ── Compute feature dimensions ───────────────────────────────
    # Week feature vector: counts + mean_sin + mean_cos + mean_dur + dispersion
    #   + active_days + day_dist(flattened) + total_norm + cal(6) = 6*T + 7*T + 1 + 6
    week_feature_dim = (
        num_tasks * 6  # counts, sin, cos, dur, dispersion, active_days
        + num_tasks * 7  # day_distribution
        + 1  # total_tasks_norm
        + 6  # calendar (woy_sin/cos, month_sin/cos, doy_sin/cos)
    )

    # History feature dim: per-event features for temporal model
    history_feature_dim = _compute_history_feature_dim(cfg)

    prepared = PreparedData(
        task_names=task_names,
        task_to_id=task_to_id,
        num_tasks=num_tasks,
        database_ids=database_ids,
        db_to_id=db_to_id,
        num_databases=len(database_ids),
        weeks=weeks,
        profiles=profiles or {},
        duration_min=duration_min,
        duration_max=duration_max,
        task_duration_medians=task_duration_medians,
        max_count_cap=max_count_cap,
        max_tasks_per_week=max_tasks_wk,
        bin_minutes=bin_minutes,
        week_feature_dim=week_feature_dim,
        history_feature_dim=history_feature_dim,
    )

    return prepared


def _compute_history_feature_dim(cfg: SimpleNamespace) -> int:
    """Compute the size of the per-event history feature vector."""
    # Recent frequency features (3 scales × 2) + recent offsets (4 scales × 2)
    # + lag slot features (4 lags × 2) + anchor features (6)
    # = 6 + 8 + 8 + 6 = 28
    return 28


# ── Week feature vector builder ──────────────────────────────────────
def build_week_feature_vector(week: WeekRecord, num_tasks: int) -> np.ndarray:
    """Build a flat feature vector for a single week."""
    parts = [
        week.counts.astype(np.float32),
        week.mean_start_sin,
        week.mean_start_cos,
        week.mean_duration_norm,
        week.start_circular_dispersion,
        week.active_days_norm,
        week.day_distribution,
        np.array([week.total_tasks_norm], dtype=np.float32),
        np.array([
            week.week_of_year_sin, week.week_of_year_cos,
            week.month_sin, week.month_cos,
            week.day_of_year_sin, week.day_of_year_cos,
        ], dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


# ── Context sequence builder ─────────────────────────────────────────
def build_context_sequence(
    prepared: PreparedData,
    target_week_idx: int,
    window_weeks: int = 16,
) -> np.ndarray:
    """Build the context sequence (window_weeks × feature_dim) for the target week."""
    start_idx = max(0, target_week_idx - window_weeks)
    end_idx = target_week_idx  # exclusive

    vectors = []
    for i in range(start_idx, end_idx):
        vec = build_week_feature_vector(prepared.weeks[i], prepared.num_tasks)
        vectors.append(vec)

    if not vectors:
        # Pad with zeros
        dim = prepared.week_feature_dim
        return np.zeros((1, dim), dtype=np.float32)

    # Pad if necessary
    while len(vectors) < window_weeks:
        vectors.insert(0, np.zeros_like(vectors[0]))

    return np.stack(vectors, axis=0).astype(np.float32)


# ── Train/val split (temporal) ───────────────────────────────────────
def build_split_indices(
    prepared: PreparedData,
    train_ratio: float = 0.80,
    window_weeks: int = 16,
) -> tuple[list[int], list[int]]:
    """Temporal split: first train_ratio weeks for training, rest for validation."""
    n_weeks = len(prepared.weeks)
    min_target = window_weeks  # need at least window_weeks of history
    valid_indices = list(range(min_target, n_weeks))

    split_point = int(len(valid_indices) * train_ratio)
    train_indices = valid_indices[:split_point]
    val_indices = valid_indices[split_point:]
    return train_indices, val_indices
