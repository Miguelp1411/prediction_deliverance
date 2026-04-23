from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from proyectos_hibridos.hybrid_schedule_project.src.hybrid_schedule.data.features import EventItem, SeriesBundle, task_prototype_from_history


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



def _week_signature(series: SeriesBundle, week_idx: int) -> np.ndarray:
    counts = series.counts[week_idx].astype(np.float32)
    day = series.day_hist[week_idx].reshape(-1).astype(np.float32)
    start = series.mean_start[week_idx].astype(np.float32) / (7 * 24 * 60 / 5)
    dur = series.mean_duration[week_idx].astype(np.float32) / 12.0
    return np.concatenate([counts, day, start, dur], axis=0)



def _future_signature(series: SeriesBundle, target_week_start: pd.Timestamp) -> np.ndarray:
    # contexto reciente más la última semana disponible
    if len(series.week_starts) == 0:
        raise ValueError('Serie vacía')
    recent = series.counts[max(0, len(series.week_starts) - 4):].mean(axis=0).astype(np.float32)
    recent_day = series.day_hist[max(0, len(series.week_starts) - 4):].mean(axis=0).reshape(-1).astype(np.float32)
    recent_start = series.mean_start[max(0, len(series.week_starts) - 4):].mean(axis=0).astype(np.float32) / (7 * 24 * 60 / 5)
    recent_dur = series.mean_duration[max(0, len(series.week_starts) - 4):].mean(axis=0).astype(np.float32) / 12.0
    return np.concatenate([recent, recent_day, recent_start, recent_dur], axis=0)



def retrieve_similar_weeks(series: SeriesBundle, target_week_idx: int | None, topk: int = 5) -> list[RetrievedWeek]:
    if len(series.week_starts) < 2:
        return []
    if target_week_idx is None:
        target_vec = _future_signature(series, series.week_starts[-1] + pd.Timedelta(days=7))
        candidate_indices = list(range(len(series.week_starts)))
        target_week_start = series.week_starts[-1] + pd.Timedelta(days=7)
    else:
        target_vec = _week_signature(series, target_week_idx)
        candidate_indices = list(range(max(0, target_week_idx - 104), target_week_idx))
        target_week_start = series.week_starts[target_week_idx]

    scored: list[RetrievedWeek] = []
    target_woy = int(target_week_start.isocalendar().week)
    for idx in candidate_indices:
        cand_vec = _week_signature(series, idx)
        l1 = float(np.abs(target_vec - cand_vec).mean())
        cand_woy = int(series.week_starts[idx].isocalendar().week)
        circ = min(abs(target_woy - cand_woy), 52 - abs(target_woy - cand_woy)) / 52.0
        score = -(1.0 * l1 + 0.35 * circ)
        scored.append(RetrievedWeek(week_idx=idx, score=score))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:max(1, topk)]



def build_template_week(series: SeriesBundle, target_week_idx: int | None, topk: int = 5) -> TemplateWeek:
    retrieved = retrieve_similar_weeks(series, target_week_idx, topk=topk)
    if not retrieved:
        raise ValueError('No hay suficientes semanas para construir plantilla')
    primary = retrieved[0].week_idx
    weights = np.asarray([np.exp(r.score) for r in retrieved], dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-8)
    counts = np.zeros(series.counts.shape[1], dtype=np.int64)
    for weight, item in zip(weights, retrieved):
        counts += np.rint(weight * series.counts[item.week_idx]).astype(np.int64)
    counts = np.clip(counts, 0, None)

    support_by_slot: dict[tuple[int, int], float] = {}
    for weight, item in zip(weights, retrieved):
        for event in series.events[item.week_idx]:
            key = (event.task_idx, event.start_bin)
            support_by_slot[key] = support_by_slot.get(key, 0.0) + float(weight)

    return TemplateWeek(
        primary_week_idx=primary,
        source_weeks=[r.week_idx for r in retrieved],
        week_scores=[float(r.score) for r in retrieved],
        counts=counts,
        events=list(series.events[primary]),
        support_by_slot=support_by_slot,
    )



def propose_extra_slots(series: SeriesBundle, template: TemplateWeek, task_idx: int, target_week_idx: int | None, required: int) -> list[tuple[int, int, float]]:
    extras: list[tuple[int, int, float]] = []
    seen = set()
    for week_idx, score in zip(template.source_weeks, template.week_scores):
        for event in series.events[week_idx]:
            if event.task_idx != task_idx:
                continue
            key = (event.start_bin, event.duration_bins)
            if key in seen:
                continue
            seen.add(key)
            extras.append((event.start_bin, event.duration_bins, float(score)))
    proto = task_prototype_from_history(series, len(series.week_starts) if target_week_idx is None else target_week_idx, task_idx)
    extras.append((int(round(proto['start_bin'])), int(round(proto['duration_bins'])), -0.25))
    extras.sort(key=lambda x: x[2], reverse=True)
    return extras[:required]
