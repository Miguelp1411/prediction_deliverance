from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hybrid_schedule.data.features import EventItem, SeriesBundle, calendar_features, task_prototype_from_history, weighted_quantile


@dataclass
class RetrievedWeek:
    week_idx: int
    score: float


@dataclass
class TemplateWeek:
    primary_week_idx: int
    source_weeks: list[int]
    week_scores: list[float]
    counts: np.ndarray
    events: list[EventItem]
    support_by_slot: dict[tuple[int, int], float]


WEEK_BINS_5M = 7 * 24 * 60 / 5


def _context_signature(series: SeriesBundle, end_week_idx: int, calendar_week_start: pd.Timestamp, lookback: int = 8) -> np.ndarray:
    start = max(0, end_week_idx - lookback)
    if end_week_idx <= start:
        counts_mean = np.zeros(series.counts.shape[1], dtype=np.float32)
        counts_std = np.zeros(series.counts.shape[1], dtype=np.float32)
        day_mean = np.zeros(series.day_hist.shape[1] * 7, dtype=np.float32)
        start_mean = np.zeros(series.mean_start.shape[1], dtype=np.float32)
        dur_mean = np.zeros(series.mean_duration.shape[1], dtype=np.float32)
    else:
        hist_counts = series.counts[start:end_week_idx].astype(np.float32)
        counts_mean = hist_counts.mean(axis=0)
        counts_std = hist_counts.std(axis=0)
        day_mean = series.day_hist[start:end_week_idx].mean(axis=0).reshape(-1).astype(np.float32)
        start_mean = series.mean_start[start:end_week_idx].mean(axis=0).astype(np.float32) / WEEK_BINS_5M
        dur_mean = series.mean_duration[start:end_week_idx].mean(axis=0).astype(np.float32) / 12.0
    return np.concatenate([
        counts_mean,
        counts_std,
        day_mean,
        start_mean,
        dur_mean,
        calendar_features(calendar_week_start),
    ], axis=0).astype(np.float32)



def _future_signature(series: SeriesBundle, target_week_start: pd.Timestamp) -> np.ndarray:
    return _context_signature(series, len(series.week_starts), target_week_start)



def retrieve_similar_weeks(series: SeriesBundle, target_week_idx: int | None, topk: int = 5) -> list[RetrievedWeek]:
    if len(series.week_starts) < 2:
        return []
    if target_week_idx is None:
        target_week_start = series.week_starts[-1] + pd.Timedelta(days=7)
        target_vec = _future_signature(series, target_week_start)
        candidate_indices = list(range(max(1, len(series.week_starts) - 156), len(series.week_starts)))
    else:
        target_week_start = series.week_starts[target_week_idx]
        target_vec = _context_signature(series, target_week_idx, target_week_start)
        candidate_indices = list(range(max(1, target_week_idx - 156), target_week_idx))

    target_woy = int(target_week_start.isocalendar().week)
    scored: list[RetrievedWeek] = []
    for idx in candidate_indices:
        cand_week_start = series.week_starts[idx]
        cand_vec = _context_signature(series, idx, cand_week_start)
        l1 = float(np.abs(target_vec - cand_vec).mean())
        cand_woy = int(cand_week_start.isocalendar().week)
        circ = min(abs(target_woy - cand_woy), 52 - abs(target_woy - cand_woy)) / 52.0
        score = -(1.00 * l1 + 0.30 * circ)
        scored.append(RetrievedWeek(week_idx=idx, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[: max(1, topk)]



def _softmax_weights(retrieved: list[RetrievedWeek]) -> np.ndarray:
    logits = np.asarray([r.score for r in retrieved], dtype=np.float64)
    logits = logits - logits.max(initial=0.0)
    weights = np.exp(logits)
    denom = max(weights.sum(), 1e-8)
    return weights / denom



def _event_from_consensus(series: SeriesBundle, reference_week_idx: int, task_idx: int, start_bin: int, duration_bins: int, rank: int) -> EventItem:
    week_start = series.week_starts[reference_week_idx]
    start_time = week_start + pd.Timedelta(minutes=int(start_bin * 5))
    end_time = start_time + pd.Timedelta(minutes=int(duration_bins * 5))
    return EventItem(
        database_id=series.database_id,
        robot_id=series.robot_id,
        task_idx=task_idx,
        task_type=series.task_names[task_idx],
        start_time=start_time,
        end_time=end_time,
        start_bin=int(start_bin),
        duration_bins=max(1, int(duration_bins)),
        duration_minutes=float(max(1, int(duration_bins)) * 5),
        source_event_id=f'consensus::{task_idx}::{rank}',
    )



def _consensus_slots(series: SeriesBundle, retrieved: list[RetrievedWeek], weights: np.ndarray, counts: np.ndarray) -> tuple[list[EventItem], dict[tuple[int, int], float]]:
    primary = retrieved[0].week_idx
    events: list[EventItem] = []
    support_by_slot: dict[tuple[int, int], float] = {}
    for task_idx in range(series.counts.shape[1]):
        desired = int(max(0, counts[task_idx]))
        if desired <= 0:
            continue
        starts: list[float] = []
        durations: list[float] = []
        event_weights: list[float] = []
        for weight, item in zip(weights, retrieved):
            for event in series.events[item.week_idx]:
                if event.task_idx != task_idx:
                    continue
                starts.append(float(event.start_bin))
                durations.append(float(event.duration_bins))
                event_weights.append(float(weight))
        if not starts:
            proto = task_prototype_from_history(series, primary + 1, task_idx)
            start_bin = int(round(proto['start_bin']))
            duration_bins = max(1, int(round(proto['duration_bins'])))
            evt = _event_from_consensus(series, primary, task_idx, start_bin, duration_bins, 0)
            events.append(evt)
            support_by_slot[(task_idx, start_bin)] = 0.0
            continue
        starts_arr = np.asarray(starts, dtype=np.float32)
        durations_arr = np.asarray(durations, dtype=np.float32)
        weights_arr = np.asarray(event_weights, dtype=np.float32)
        desired = min(desired, max(1, len(starts_arr)))
        for rank in range(desired):
            q = (rank + 0.5) / desired
            start_bin = int(round(weighted_quantile(starts_arr, weights_arr, q)))
            nearby = np.abs(starts_arr - start_bin) <= 18
            if nearby.any():
                duration_bins = int(round(weighted_quantile(durations_arr[nearby], weights_arr[nearby], 0.5)))
                support = float(weights_arr[nearby].sum())
            else:
                duration_bins = int(round(weighted_quantile(durations_arr, weights_arr, q)))
                support = float(np.exp(-0.1 * np.min(np.abs(starts_arr - start_bin))))
            evt = _event_from_consensus(series, primary, task_idx, start_bin, duration_bins, rank)
            events.append(evt)
            support_by_slot[(task_idx, start_bin)] = max(support_by_slot.get((task_idx, start_bin), 0.0), support)
    events.sort(key=lambda e: (e.start_bin, e.task_type))
    return events, support_by_slot



def build_template_week(series: SeriesBundle, target_week_idx: int | None, topk: int = 5) -> TemplateWeek:
    retrieved = retrieve_similar_weeks(series, target_week_idx, topk=topk)
    if not retrieved:
        raise ValueError('No hay suficientes semanas para construir plantilla')
    weights = _softmax_weights(retrieved)
    primary = retrieved[0].week_idx
    weighted_counts = np.zeros(series.counts.shape[1], dtype=np.float64)
    for weight, item in zip(weights, retrieved):
        weighted_counts += float(weight) * series.counts[item.week_idx].astype(np.float64)
    counts = np.clip(np.rint(weighted_counts), 0, None).astype(np.int64)
    events, support_by_slot = _consensus_slots(series, retrieved, weights, counts)
    return TemplateWeek(
        primary_week_idx=primary,
        source_weeks=[r.week_idx for r in retrieved],
        week_scores=[float(r.score) for r in retrieved],
        counts=counts,
        events=events,
        support_by_slot=support_by_slot,
    )



def propose_extra_slots(series: SeriesBundle, template: TemplateWeek, task_idx: int, target_week_idx: int | None, required: int) -> list[tuple[int, int, float]]:
    extras: list[tuple[int, int, float]] = []
    seen = set((e.start_bin, e.duration_bins) for e in template.events if e.task_idx == task_idx)
    for week_idx, score in zip(template.source_weeks, template.week_scores):
        for event in series.events[week_idx]:
            if event.task_idx != task_idx:
                continue
            key = (event.start_bin, event.duration_bins)
            if key in seen:
                continue
            seen.add(key)
            support = float(template.support_by_slot.get((task_idx, event.start_bin), 0.0))
            extras.append((int(event.start_bin), int(event.duration_bins), float(score + support)))
    proto = task_prototype_from_history(series, len(series.week_starts) if target_week_idx is None else target_week_idx, task_idx)
    proto_key = (int(round(proto['start_bin'])), int(round(proto['duration_bins'])))
    if proto_key not in seen:
        extras.append((proto_key[0], max(1, proto_key[1]), -0.10))
    extras.sort(key=lambda x: x[2], reverse=True)
    return extras[: max(0, required)]
