"""
Feature engineering for the occurrence residual model.

Builds per-task feature vectors including count windows, lag counts,
calendar signals, and template counts.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from data.schema import PreparedData, WeekRecord


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def build_occurrence_features(
    prepared: PreparedData,
    target_week_idx: int,
    template_counts: dict[str, int] | None = None,
    window_sizes: tuple[int, ...] = (4, 8, 16),
    lag_positions: tuple[int, ...] = (4, 26, 52),
) -> np.ndarray:
    """
    Build feature matrix for the occurrence residual model.

    Returns shape (num_tasks, feature_dim) where each row is the feature
    vector for one task type.
    """
    num_tasks = prepared.num_tasks
    weeks = prepared.weeks
    target_week = weeks[target_week_idx]

    features_per_task: list[list[float]] = []

    for tid in range(num_tasks):
        task_name = prepared.task_names[tid]
        feats: list[float] = []

        # ── Template count ───────────────────────────────────────
        tpl_count = 0
        if template_counts:
            tpl_count = template_counts.get(task_name, 0)
        feats.append(float(tpl_count))

        # ── Rolling count windows ────────────────────────────────
        for w_size in window_sizes:
            start = max(0, target_week_idx - w_size)
            counts_in_window = []
            for i in range(start, target_week_idx):
                counts_in_window.append(float(weeks[i].counts[tid]))
            if counts_in_window:
                feats.append(np.mean(counts_in_window))
                feats.append(np.std(counts_in_window))
                feats.append(np.min(counts_in_window))
                feats.append(np.max(counts_in_window))
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])

        # ── Lag counts ───────────────────────────────────────────
        for lag in lag_positions:
            lag_idx = target_week_idx - lag
            if 0 <= lag_idx < len(weeks):
                feats.append(float(weeks[lag_idx].counts[tid]))
            else:
                feats.append(0.0)

        # ── Calendar signals ─────────────────────────────────────
        feats.extend([
            target_week.week_of_year_sin,
            target_week.week_of_year_cos,
            target_week.month_sin,
            target_week.month_cos,
            target_week.day_of_year_sin,
            target_week.day_of_year_cos,
        ])

        # ── Task global stats ────────────────────────────────────
        median_dur = prepared.task_duration_medians.get(task_name, 0.0)
        dur_span = max(prepared.duration_max - prepared.duration_min, 1e-6)
        feats.append((median_dur - prepared.duration_min) / dur_span)

        # ── Delta from template ──────────────────────────────────
        # Last known count vs template
        last_idx = target_week_idx - 1
        if last_idx >= 0:
            last_count = float(weeks[last_idx].counts[tid])
        else:
            last_count = 0.0
        feats.append(last_count - tpl_count)

        features_per_task.append(feats)

    return np.array(features_per_task, dtype=np.float32)


def occurrence_feature_dim(
    window_sizes: tuple[int, ...] = (4, 8, 16),
    lag_positions: tuple[int, ...] = (4, 26, 52),
) -> int:
    """Return the feature dimension per task for occurrence features."""
    # template_count(1) + windows(4*len) + lags(len) + calendar(6) + dur(1) + delta(1)
    return 1 + 4 * len(window_sizes) + len(lag_positions) + 6 + 1 + 1
