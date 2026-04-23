"""
Feature engineering for the temporal residual model.

Builds per-event features including history features, anchor bins,
template offsets, and context.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from data.schema import EventRecord, PreparedData, WeekRecord


def _cyclical(value: float, period: float) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def build_temporal_features(
    prepared: PreparedData,
    target_week_idx: int,
    event: EventRecord,
    occurrence_slot: int,
    total_occurrences: int,
    template_events: list[EventRecord] | None = None,
    lookback: int = 52,
) -> np.ndarray:
    """
    Build feature vector for a single event in the temporal model.

    Returns shape (feature_dim,).
    """
    weeks = prepared.weeks
    bins_per_day_val = prepared.bins_per_day
    total_bins = prepared.num_time_bins
    tid = event.task_id

    feats: list[float] = []

    # ── Recent frequency features ────────────────────────────────
    for scale in [2, 4, 12]:
        start = max(0, target_week_idx - scale)
        counts = [float(weeks[i].counts[tid]) for i in range(start, target_week_idx)]
        feats.append(np.mean(counts) if counts else 0.0)
        feats.append(np.std(counts) if counts else 0.0)

    # ── Recent start offset features ─────────────────────────────
    for scale in [1, 2, 4, 8]:
        start_idx = max(0, target_week_idx - scale)
        start_bins: list[int] = []
        for i in range(start_idx, target_week_idx):
            w = weeks[i]
            task_evts = w.events_by_task.get(tid, [])
            for ev in task_evts:
                start_bins.append(ev.start_bin)
        if start_bins:
            feats.append(np.mean(start_bins) / total_bins)
            feats.append(np.std(start_bins) / total_bins)
        else:
            feats.extend([0.0, 0.0])

    # ── Lag slot features ────────────────────────────────────────
    for lag in [1, 4, 26, 52]:
        lag_idx = target_week_idx - lag
        if 0 <= lag_idx < len(weeks):
            w = weeks[lag_idx]
            task_evts = w.events_by_task.get(tid, [])
            if occurrence_slot < len(task_evts):
                feats.append(task_evts[occurrence_slot].start_bin / total_bins)
                feats.append(task_evts[occurrence_slot].duration_minutes / max(prepared.duration_max, 1.0))
            else:
                feats.extend([0.0, 0.0])
        else:
            feats.extend([0.0, 0.0])

    # ── Anchor features ──────────────────────────────────────────
    # Compute anchor from template or lag52
    anchor_bin = 0
    if template_events:
        same_task = [e for e in template_events if e.task_id == tid]
        if occurrence_slot < len(same_task):
            anchor_bin = same_task[occurrence_slot].start_bin
    else:
        lag52_idx = target_week_idx - 52
        if 0 <= lag52_idx < len(weeks):
            task_evts = weeks[lag52_idx].events_by_task.get(tid, [])
            if occurrence_slot < len(task_evts):
                anchor_bin = task_evts[occurrence_slot].start_bin

    feats.append(anchor_bin / total_bins)
    anchor_day = anchor_bin // bins_per_day_val
    anchor_time = anchor_bin % bins_per_day_val
    day_sin, day_cos = _cyclical(anchor_day, 7)
    time_sin, time_cos = _cyclical(anchor_time, bins_per_day_val)
    feats.extend([day_sin, day_cos, time_sin, time_cos])

    # ── Occurrence progress ──────────────────────────────────────
    progress = occurrence_slot / max(total_occurrences - 1, 1)
    feats.append(progress)

    return np.array(feats, dtype=np.float32)


def temporal_feature_dim() -> int:
    """Return the feature dimension for a single event."""
    # freq(3*2) + offsets(4*2) + lags(4*2) + anchor(5) + progress(1) = 28
    return 6 + 8 + 8 + 5 + 1
