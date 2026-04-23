"""
Automatic database profiling.

Computes comprehensive statistics for each database:
task types, device counts, overlaps, durations, frequencies,
seasonality, and operating hours.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.schema import DatabaseProfile, Event


def _week_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Monday 00:00 of the week containing *ts*."""
    return ts.normalize() - pd.Timedelta(days=ts.weekday())


def _count_overlaps(events: list[Event]) -> int:
    """Count pairwise overlaps among events on the same device."""
    by_device: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        by_device[ev.device_id].append(ev)

    total = 0
    for dev_events in by_device.values():
        sorted_ev = sorted(dev_events, key=lambda e: e.start_time)
        for i in range(len(sorted_ev)):
            for j in range(i + 1, len(sorted_ev)):
                if sorted_ev[j].start_time < sorted_ev[i].end_time:
                    total += 1
                else:
                    break
    return total


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    """Pearson autocorrelation at given lag."""
    if len(values) <= lag:
        return 0.0
    x = values[:-lag].astype(float)
    y = values[lag:].astype(float)
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def profile_database(
    events: list[Event],
    database_id: str = "",
    seasonal_lags: list[int] | None = None,
) -> DatabaseProfile:
    """Compute comprehensive database statistics."""
    if seasonal_lags is None:
        seasonal_lags = [4, 13, 26, 52]

    profile = DatabaseProfile(database_id=database_id)
    if not events:
        return profile

    profile.num_events = len(events)

    # Task types
    task_types = sorted(set(e.task_type for e in events))
    profile.task_types = task_types
    profile.num_task_types = len(task_types)

    # Devices
    devices = set(e.device_id for e in events if e.device_id and e.device_id != "__unknown_device__")
    profile.num_devices = max(len(devices), 1)
    profile.is_single_device = profile.num_devices <= 1

    # Date range
    starts = [e.start_time for e in events]
    profile.date_range_start = str(min(starts))
    profile.date_range_end = str(max(starts))

    # Overlaps
    profile.overlap_count = _count_overlaps(events)
    profile.overlap_rate = profile.overlap_count / max(len(events), 1)

    # Duration stats per task
    task_durations: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        task_durations[ev.task_type].append(ev.duration_minutes)

    for task in task_types:
        durs = task_durations[task]
        profile.task_duration_mean[task] = float(np.mean(durs))
        profile.task_duration_median[task] = float(np.median(durs))
        profile.task_duration_std[task] = float(np.std(durs))

    # Weekly frequency per task
    week_tasks: dict[str, Counter[str]] = defaultdict(Counter)
    for ev in events:
        ws = _week_start(ev.start_time)
        week_key = ws.isoformat()
        week_tasks[week_key][ev.task_type] += 1

    all_weeks = sorted(week_tasks.keys())
    profile.num_weeks = len(all_weeks)

    for task in task_types:
        counts_per_week = [week_tasks[w][task] for w in all_weeks]
        profile.task_weekly_frequency[task] = float(np.mean(counts_per_week))

    # Seasonality: autocorrelation at each lag per task
    best_lag_scores: dict[int, float] = defaultdict(float)
    for task in task_types:
        counts_arr = np.array([week_tasks[w][task] for w in all_weeks])
        task_seasonality: dict[str, float] = {}
        for lag in seasonal_lags:
            ac = _autocorrelation(counts_arr, lag)
            task_seasonality[f"lag{lag}"] = ac
            best_lag_scores[lag] += ac
        profile.seasonal_strength[task] = max(task_seasonality.values()) if task_seasonality else 0.0

    # Best lags overall
    profile.best_lags = sorted(best_lag_scores, key=lambda l: best_lag_scores[l], reverse=True)

    # Hour distribution (24 bins)
    hour_counts = [0] * 24
    for ev in events:
        h = ev.start_time.hour
        hour_counts[h] += 1
    total_h = sum(hour_counts) or 1
    profile.hour_distribution = [c / total_h for c in hour_counts]

    # Day distribution (7 bins, Mon=0)
    day_counts = [0] * 7
    for ev in events:
        d = ev.start_time.weekday()
        day_counts[d] += 1
    total_d = sum(day_counts) or 1
    profile.day_distribution = [c / total_d for c in day_counts]

    # Operating hours
    start_hours = sorted(ev.start_time.hour for ev in events)
    if start_hours:
        n = len(start_hours)
        lo_idx = max(0, int(n * 0.02))
        hi_idx = min(n - 1, int(n * 0.98))
        profile.operating_hour_start = start_hours[lo_idx]
        profile.operating_hour_end = min(start_hours[hi_idx] + 1, 24)

    # History quality score (0-1)
    # Higher = more weeks, more events, fewer overlaps, stable frequencies
    weeks_score = min(profile.num_weeks / 104, 1.0)  # 2 years = max
    events_score = min(profile.num_events / 10000, 1.0)
    overlap_score = max(0.0, 1.0 - profile.overlap_rate * 10)
    freq_stability = 0.0
    for task in task_types:
        counts_arr = np.array([week_tasks[w][task] for w in all_weeks], dtype=float)
        if counts_arr.mean() > 0:
            cv = counts_arr.std() / counts_arr.mean()
            freq_stability += max(0.0, 1.0 - cv)
    freq_stability /= max(len(task_types), 1)
    profile.history_quality_score = (
        0.30 * weeks_score + 0.20 * events_score + 0.20 * overlap_score + 0.30 * freq_stability
    )

    return profile


def save_profile(profile: DatabaseProfile, path: str | Path) -> None:
    """Persist a DatabaseProfile to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=2, default=str)


def load_profile(path: str | Path) -> DatabaseProfile:
    """Load a DatabaseProfile from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatabaseProfile(**data)


# ── CLI entrypoint ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from data.adapters.json_adapter import load_json_events

    if len(sys.argv) < 2:
        print("Usage: python -m data.profiling <path_to_json_db> [database_id]")
        sys.exit(1)

    db_path = sys.argv[1]
    db_id = sys.argv[2] if len(sys.argv) > 2 else Path(db_path).stem

    events = load_json_events(db_path, database_id=db_id)
    prof = profile_database(events, database_id=db_id)

    print(f"\n{'='*60}")
    print(f"  Database Profile: {db_id}")
    print(f"{'='*60}")
    print(f"  Events:          {prof.num_events:,}")
    print(f"  Weeks:           {prof.num_weeks}")
    print(f"  Task types:      {prof.num_task_types} — {prof.task_types}")
    print(f"  Devices:         {prof.num_devices} ({'single' if prof.is_single_device else 'multi'})")
    print(f"  Overlaps:        {prof.overlap_count} ({prof.overlap_rate:.4f})")
    print(f"  Operating hours: {prof.operating_hour_start}:00 – {prof.operating_hour_end}:00")
    print(f"  Quality score:   {prof.history_quality_score:.3f}")
    print(f"  Best lags:       {prof.best_lags}")
    print(f"  Date range:      {prof.date_range_start} → {prof.date_range_end}")
    print()
    for task in prof.task_types:
        freq = prof.task_weekly_frequency.get(task, 0)
        dur_m = prof.task_duration_median.get(task, 0)
        seas = prof.seasonal_strength.get(task, 0)
        print(f"  {task:20s} freq={freq:.1f}/wk  dur={dur_m:.0f}min  seasonality={seas:.3f}")

    out_path = Path(db_path).parent / f"{db_id}_profile.json"
    save_profile(prof, out_path)
    print(f"\n  Profile saved to: {out_path}")
