"""
Canonical data schema for the hybrid multi-DB prediction system.

All data adapters must convert to these types before entering
the processing pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ── Canonical Event ──────────────────────────────────────────────────
@dataclass
class Event:
    """Single calendar event in canonical form."""
    database_id: str
    robot_id: str
    device_id: str
    task_type: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    timezone: str | None = None
    source_event_id: str | None = None
    duration_minutes: float = 0.0

    def __post_init__(self) -> None:
        if self.duration_minutes <= 0.0:
            delta = (self.end_time - self.start_time).total_seconds() / 60.0
            self.duration_minutes = max(delta, 0.0)


# ── Event Record (within a week) ────────────────────────────────────
@dataclass
class EventRecord:
    """An event mapped into the weekly bin grid."""
    task_id: int
    task_name: str
    start_bin: int
    duration_minutes: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    device_id: str = ""
    database_id: str = ""
    robot_id: str = ""


# ── Week Record ──────────────────────────────────────────────────────
@dataclass
class WeekRecord:
    """Aggregated features for a single week."""
    week_index: int
    week_start: pd.Timestamp
    database_id: str = ""
    # Per-task counts: shape (num_tasks,)
    counts: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-task circular-mean start sin/cos: shape (num_tasks,)
    mean_start_sin: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_start_cos: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-task mean duration normalised
    mean_duration_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    # Per-task start dispersion
    start_circular_dispersion: np.ndarray = field(default_factory=lambda: np.array([]))
    # Active-days normalised (0-1)
    active_days_norm: np.ndarray = field(default_factory=lambda: np.array([]))
    # Day-of-week distribution (num_tasks, 7) or (7,) global
    day_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    # Scalars
    total_tasks_norm: float = 0.0
    week_of_year_sin: float = 0.0
    week_of_year_cos: float = 0.0
    month_sin: float = 0.0
    month_cos: float = 0.0
    day_of_year_sin: float = 0.0
    day_of_year_cos: float = 0.0
    # Raw events for this week, keyed by task_id
    events_by_task: dict[int, list[EventRecord]] = field(default_factory=dict)
    # All events (flat list)
    events: list[EventRecord] = field(default_factory=list)


# ── Database Profile ─────────────────────────────────────────────────
@dataclass
class DatabaseProfile:
    """Auto-computed profile for a single database."""
    database_id: str = ""
    num_events: int = 0
    num_weeks: int = 0
    task_types: list[str] = field(default_factory=list)
    num_task_types: int = 0
    num_devices: int = 0
    is_single_device: bool = True
    # Per-task stats
    task_weekly_frequency: dict[str, float] = field(default_factory=dict)
    task_duration_mean: dict[str, float] = field(default_factory=dict)
    task_duration_median: dict[str, float] = field(default_factory=dict)
    task_duration_std: dict[str, float] = field(default_factory=dict)
    # Overlap stats
    overlap_count: int = 0
    overlap_rate: float = 0.0
    # Seasonality
    seasonal_strength: dict[str, float] = field(default_factory=dict)
    best_lags: list[int] = field(default_factory=list)
    # Time distributions
    hour_distribution: list[float] = field(default_factory=list)
    day_distribution: list[float] = field(default_factory=list)
    # Operating hours
    operating_hour_start: int = 0
    operating_hour_end: int = 24
    # Quality
    history_quality_score: float = 0.0
    date_range_start: str = ""
    date_range_end: str = ""


# ── Prepared Data Bundle ─────────────────────────────────────────────
@dataclass
class PreparedData:
    """Everything needed for training/inference, multi-database."""
    task_names: list[str] = field(default_factory=list)
    task_to_id: dict[str, int] = field(default_factory=dict)
    num_tasks: int = 0
    database_ids: list[str] = field(default_factory=list)
    db_to_id: dict[str, int] = field(default_factory=dict)
    num_databases: int = 0
    weeks: list[WeekRecord] = field(default_factory=list)
    profiles: dict[str, DatabaseProfile] = field(default_factory=dict)
    duration_min: float = 0.0
    duration_max: float = 1.0
    task_duration_medians: dict[str, float] = field(default_factory=dict)
    max_count_cap: int = 50
    max_tasks_per_week: int = 100
    bin_minutes: int = 5
    # Feature dimensions (set during preparation)
    week_feature_dim: int = 0
    history_feature_dim: int = 0

    @property
    def num_time_bins(self) -> int:
        return (7 * 24 * 60) // self.bin_minutes

    @property
    def bins_per_day(self) -> int:
        return (24 * 60) // self.bin_minutes
