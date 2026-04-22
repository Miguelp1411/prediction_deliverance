from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hybrid_schedule.data.features import (
    EventItem,
    SeriesBundle,
    SlotPrototype,
    assign_events_to_prototypes,
    season_bucket_from_week_start,
    task_prototype_from_history,
    task_slot_prototypes,
    task_slot_prototypes_contextual,
    week_density_bucket,
    week_regime_signature,
    week_type_bucket_from_week_start,
)


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
    slot_prototypes_by_task: dict[int, list[SlotPrototype]]


@dataclass
class CandidateRecord:
    start_bin: int
    duration_bins: int
    total_support: float = 0.0
    source_support: dict[str, float] = field(default_factory=dict)
    slot_support: dict[int, float] = field(default_factory=dict)
    season_counts: dict[str, float] = field(default_factory=dict)
    week_type_counts: dict[str, float] = field(default_factory=dict)
    density_counts: dict[str, float] = field(default_factory=dict)
    regime_counts: dict[str, float] = field(default_factory=dict)
    precedence_counts: dict[tuple[int, int], float] = field(default_factory=dict)

    def add(
        self,
        source: str,
        weight: float,
        slot_id: int | None = None,
        season_bucket: str | None = None,
        week_type: str | None = None,
        density_bucket: str | None = None,
        regime_id: str | None = None,
        precedence_key: tuple[int, int] | None = None,
    ) -> None:
        w = float(weight)
        self.total_support += w
        self.source_support[source] = self.source_support.get(source, 0.0) + w
        if slot_id is not None:
            sid = int(slot_id)
            self.slot_support[sid] = self.slot_support.get(sid, 0.0) + w
        if season_bucket is not None:
            self.season_counts[str(season_bucket)] = self.season_counts.get(str(season_bucket), 0.0) + w
        if week_type is not None:
            self.week_type_counts[str(week_type)] = self.week_type_counts.get(str(week_type), 0.0) + w
        if density_bucket is not None:
            self.density_counts[str(density_bucket)] = self.density_counts.get(str(density_bucket), 0.0) + w
        if regime_id is not None:
            self.regime_counts[str(regime_id)] = self.regime_counts.get(str(regime_id), 0.0) + w
        if precedence_key is not None:
            key = (int(precedence_key[0]), int(precedence_key[1]))
            self.precedence_counts[key] = self.precedence_counts.get(key, 0.0) + w


EmpiricalCandidateBank = dict[int, dict[str, Any]]


DEFAULT_SOURCE_QUOTAS: dict[str, int] = {
    'slot': 12,
    'task': 6,
    'seasonal': 8,
    'week_type': 4,
    'density': 6,
    'regime': 6,
    'precedence': 6,
    'prototype': 8,
}
DEFAULT_PROTOTYPE_START_OFFSETS = [-12, -6, -3, -2, -1, 0, 1, 2, 3, 6, 12]
DEFAULT_PROTOTYPE_DURATION_OFFSETS = [-2, -1, 0, 1, 2]


def _week_signature(series: SeriesBundle, week_idx: int) -> np.ndarray:
    counts = series.counts[week_idx].astype(np.float32)
    day = series.day_hist[week_idx].reshape(-1).astype(np.float32)
    start = series.mean_start[week_idx].astype(np.float32) / (7 * 24 * 60 / 5)
    dur = series.mean_duration[week_idx].astype(np.float32) / 12.0
    return np.concatenate([counts, day, start, dur], axis=0)


def _future_signature(series: SeriesBundle, target_week_start: pd.Timestamp) -> np.ndarray:
    if len(series.week_starts) == 0:
        raise ValueError('Serie vacía')
    recent = series.counts[max(0, len(series.week_starts) - 4):].mean(axis=0).astype(np.float32)
    recent_day = series.day_hist[max(0, len(series.week_starts) - 4):].mean(axis=0).reshape(-1).astype(np.float32)
    recent_start = series.mean_start[max(0, len(series.week_starts) - 4):].mean(axis=0).astype(np.float32) / (7 * 24 * 60 / 5)
    recent_dur = series.mean_duration[max(0, len(series.week_starts) - 4):].mean(axis=0).astype(np.float32) / 12.0
    return np.concatenate([recent, recent_day, recent_start, recent_dur], axis=0)


def _target_week_start(series: SeriesBundle, target_week_idx: int | None) -> pd.Timestamp:
    if target_week_idx is None:
        return pd.Timestamp(series.week_starts[-1]) + pd.Timedelta(days=7)
    return pd.Timestamp(series.week_starts[target_week_idx])


def _density_bucket_from_total(series: SeriesBundle, total_events: float) -> str:
    totals = np.asarray(series.counts.sum(axis=1), dtype=np.float32)
    if totals.size == 0:
        return 'mid'
    low, high = np.quantile(totals, (0.33, 0.66))
    val = float(total_events)
    if val <= float(low):
        return 'low'
    if val >= float(high):
        return 'high'
    return 'mid'


def _target_regime_signature(series: SeriesBundle, target_week_idx: int | None, lookback: int = 8) -> str:
    if len(series.week_starts) == 0:
        return 'unknown'
    ref_idx = len(series.week_starts) - 1 if target_week_idx is None else max(0, int(target_week_idx) - 1)
    return week_regime_signature(series, ref_idx, lookback=lookback)


def _planned_precedence_signature(planned_slots: list[dict[str, Any]], task_idx: int, slot_id: int) -> tuple[int, int]:
    if not planned_slots:
        return (-1, -1)
    ordered = sorted(planned_slots, key=lambda row: (int(row['anchor_start_bin']), int(row.get('task_idx', 0)), int(row.get('slot_id', 0))))
    for idx, row in enumerate(ordered):
        if int(row.get('task_idx', -1)) == int(task_idx) and int(row.get('slot_id', -1)) == int(slot_id):
            prev_task = int(ordered[idx - 1]['task_idx']) if idx > 0 else -1
            next_task = int(ordered[idx + 1]['task_idx']) if idx + 1 < len(ordered) else -1
            return (prev_task, next_task)
    return (-1, -1)


def build_target_candidate_context(
    series: SeriesBundle,
    target_week_idx: int | None,
    planned_slots: list[dict[str, Any]] | None,
    task_idx: int,
    slot_id: int,
    regime_lookback_weeks: int = 8,
) -> dict[str, Any]:
    week_start = _target_week_start(series, target_week_idx)
    planned_slots = list(planned_slots or [])
    total_events = float(len(planned_slots)) if planned_slots else float(np.mean(series.counts[max(0, len(series.week_starts) - 4):].sum(axis=1)))
    return {
        'season_bucket': season_bucket_from_week_start(week_start),
        'week_type': week_type_bucket_from_week_start(week_start),
        'density_bucket': _density_bucket_from_total(series, total_events),
        'regime_id': _target_regime_signature(series, target_week_idx, lookback=regime_lookback_weeks),
        'precedence_key': _planned_precedence_signature(planned_slots, task_idx=task_idx, slot_id=slot_id),
        'planned_total_events': float(total_events),
    }


def retrieve_similar_weeks(series: SeriesBundle, target_week_idx: int | None, topk: int = 5) -> list[RetrievedWeek]:
    if len(series.week_starts) < 2:
        return []
    if target_week_idx is None:
        target_vec = _future_signature(series, series.week_starts[-1] + pd.Timedelta(days=7))
        candidate_indices = list(range(len(series.week_starts)))
        target_week_start = series.week_starts[-1] + pd.Timedelta(days=7)
    else:
        target_vec = _future_signature(series, series.week_starts[target_week_idx])
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


def _merge_prototypes(base: list[SlotPrototype], contextual: list[SlotPrototype], max_slots: int) -> list[SlotPrototype]:
    merged: list[SlotPrototype] = []
    seen: list[tuple[int, int]] = []
    next_slot_id = max((int(proto.slot_id) for proto in base), default=-1) + 1
    for proto in sorted(base, key=lambda p: (p.center_bin, p.slot_id)):
        merged.append(proto)
        seen.append((int(proto.center_bin), int(proto.duration_bins)))
    for proto in sorted(contextual, key=lambda p: (-p.support, p.center_bin, p.slot_id)):
        key = (int(proto.center_bin), int(proto.duration_bins))
        if any(abs(key[0] - prev[0]) <= 2 and abs(key[1] - prev[1]) <= 1 for prev in seen):
            continue
        merged.append(SlotPrototype(
            task_idx=int(proto.task_idx),
            slot_id=int(next_slot_id),
            center_bin=int(proto.center_bin),
            duration_bins=max(1, int(proto.duration_bins)),
            support=float(proto.support),
        ))
        seen.append(key)
        next_slot_id += 1
        if len(merged) >= int(max_slots):
            break
    merged.sort(key=lambda p: (p.center_bin, -p.support, p.slot_id))
    return merged[:max_slots]


def build_template_week(series: SeriesBundle, target_week_idx: int | None, topk: int = 5, max_slot_prototypes: int = 32) -> TemplateWeek:
    retrieved = retrieve_similar_weeks(series, target_week_idx, topk=topk)
    if not retrieved:
        raise ValueError('No hay suficientes semanas para construir plantilla')
    primary = retrieved[0].week_idx
    weights = np.asarray([np.exp(r.score) for r in retrieved], dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-8)
    weighted_counts = np.zeros(series.counts.shape[1], dtype=np.float64)
    for weight, item in zip(weights, retrieved):
        weighted_counts += weight * series.counts[item.week_idx].astype(np.float64)

    counts = np.rint(weighted_counts).astype(np.int64)
    counts = np.clip(counts, 0, None)

    support_by_slot: dict[tuple[int, int], float] = {}
    for weight, item in zip(weights, retrieved):
        for event in series.events[item.week_idx]:
            key = (event.task_idx, event.start_bin)
            support_by_slot[key] = support_by_slot.get(key, 0.0) + float(weight)

    ref_idx = len(series.week_starts) if target_week_idx is None else target_week_idx
    target_context = build_target_candidate_context(series, target_week_idx, planned_slots=[], task_idx=0, slot_id=0)
    slot_prototypes_by_task: dict[int, list[SlotPrototype]] = {}
    for task_idx in range(series.counts.shape[1]):
        base_prototypes = task_slot_prototypes(series, ref_idx, task_idx, max_slots=max_slot_prototypes)
        contextual_prototypes = task_slot_prototypes_contextual(
            series,
            ref_idx,
            task_idx,
            max_slots=max_slot_prototypes,
            season_filter=str(target_context['season_bucket']),
            week_type_filter=str(target_context['week_type']),
            density_filter=str(target_context['density_bucket']),
            regime_filter=str(target_context['regime_id']),
        )
        slot_prototypes_by_task[task_idx] = _merge_prototypes(base_prototypes, contextual_prototypes, max_slots=max_slot_prototypes)

    return TemplateWeek(
        primary_week_idx=primary,
        source_weeks=[r.week_idx for r in retrieved],
        week_scores=[float(r.score) for r in retrieved],
        counts=counts,
        events=list(series.events[primary]),
        support_by_slot=support_by_slot,
        slot_prototypes_by_task=slot_prototypes_by_task,
    )


def _candidate_event_key(event: EventItem) -> tuple[int, int, str]:
    return (int(event.start_bin), int(event.duration_bins), str(event.source_event_id))


def _precedence_map(week_events: list[EventItem]) -> dict[tuple[int, int, str], tuple[int, int]]:
    ordered = sorted(week_events, key=lambda evt: (int(evt.start_bin), int(evt.duration_bins), str(evt.source_event_id)))
    mapping: dict[tuple[int, int, str], tuple[int, int]] = {}
    for idx, evt in enumerate(ordered):
        prev_task = int(ordered[idx - 1].task_idx) if idx > 0 else -1
        next_task = int(ordered[idx + 1].task_idx) if idx + 1 < len(ordered) else -1
        mapping[_candidate_event_key(evt)] = (prev_task, next_task)
    return mapping


def _get_or_create(container: dict[tuple[int, int], CandidateRecord], key: tuple[int, int]) -> CandidateRecord:
    if key not in container:
        container[key] = CandidateRecord(start_bin=int(key[0]), duration_bins=max(1, int(key[1])))
    return container[key]


def _register_candidate(
    container: dict[tuple[int, int], CandidateRecord],
    key: tuple[int, int],
    source: str,
    weight: float,
    slot_id: int | None,
    season_bucket: str,
    week_type: str,
    density_bucket: str,
    regime_id: str,
    precedence_key: tuple[int, int],
) -> None:
    rec = _get_or_create(container, key)
    rec.add(
        source=source,
        weight=weight,
        slot_id=slot_id,
        season_bucket=season_bucket,
        week_type=week_type,
        density_bucket=density_bucket,
        regime_id=regime_id,
        precedence_key=precedence_key,
    )


def build_empirical_candidate_bank(series: SeriesBundle, template: TemplateWeek) -> EmpiricalCandidateBank:
    raw_weights = np.exp(np.asarray(template.week_scores, dtype=np.float64))
    total = float(raw_weights.sum())
    week_weights = raw_weights / total if total > 0 else np.ones_like(raw_weights) / max(len(raw_weights), 1)
    bank: EmpiricalCandidateBank = {}
    for task_idx in range(series.counts.shape[1]):
        prototypes = list(template.slot_prototypes_by_task.get(task_idx, []))
        slot_bank: dict[int, dict[tuple[int, int], CandidateRecord]] = {}
        task_bank: dict[tuple[int, int], CandidateRecord] = {}
        seasonal_bank: dict[tuple[str, int], dict[tuple[int, int], CandidateRecord]] = {}
        week_type_bank: dict[tuple[str, int], dict[tuple[int, int], CandidateRecord]] = {}
        density_bank: dict[tuple[str, int], dict[tuple[int, int], CandidateRecord]] = {}
        regime_bank: dict[tuple[str, int], dict[tuple[int, int], CandidateRecord]] = {}
        precedence_bank: dict[tuple[tuple[int, int], int], dict[tuple[int, int], CandidateRecord]] = {}
        if not prototypes:
            bank[task_idx] = {
                'slot': slot_bank,
                'task': task_bank,
                'seasonal': seasonal_bank,
                'week_type': week_type_bank,
                'density': density_bank,
                'regime': regime_bank,
                'precedence': precedence_bank,
                'prototypes': prototypes,
            }
            continue
        for weight, week_idx in zip(week_weights.tolist(), template.source_weeks):
            week_events = list(series.events[week_idx])
            task_events = [evt for evt in week_events if evt.task_idx == task_idx]
            if not task_events:
                continue
            assignments = assign_events_to_prototypes(task_events, prototypes)
            if not assignments:
                continue
            week_start = pd.Timestamp(series.week_starts[week_idx])
            season_bucket = season_bucket_from_week_start(week_start)
            week_type = week_type_bucket_from_week_start(week_start)
            density_bucket = week_density_bucket(series, week_idx)
            regime_id = week_regime_signature(series, week_idx, lookback=8)
            precedence_map = _precedence_map(week_events)
            for slot_id, evt, _ in assignments:
                key = (int(evt.start_bin), int(evt.duration_bins))
                precedence_key = precedence_map.get(_candidate_event_key(evt), (-1, -1))
                slot_bank.setdefault(int(slot_id), {})
                seasonal_bank.setdefault((str(season_bucket), int(slot_id)), {})
                week_type_bank.setdefault((str(week_type), int(slot_id)), {})
                density_bank.setdefault((str(density_bucket), int(slot_id)), {})
                regime_bank.setdefault((str(regime_id), int(slot_id)), {})
                precedence_bank.setdefault(((int(precedence_key[0]), int(precedence_key[1])), int(slot_id)), {})
                _register_candidate(slot_bank[int(slot_id)], key, 'slot', weight, slot_id, season_bucket, week_type, density_bucket, regime_id, precedence_key)
                _register_candidate(task_bank, key, 'task', weight, slot_id, season_bucket, week_type, density_bucket, regime_id, precedence_key)
                _register_candidate(seasonal_bank[(str(season_bucket), int(slot_id))], key, 'seasonal', weight, slot_id, season_bucket, week_type, density_bucket, regime_id, precedence_key)
                _register_candidate(week_type_bank[(str(week_type), int(slot_id))], key, 'week_type', weight, slot_id, season_bucket, week_type, density_bucket, regime_id, precedence_key)
                _register_candidate(density_bank[(str(density_bucket), int(slot_id))], key, 'density', weight, slot_id, season_bucket, week_type, density_bucket, regime_id, precedence_key)
                _register_candidate(regime_bank[(str(regime_id), int(slot_id))], key, 'regime', weight, slot_id, season_bucket, week_type, density_bucket, regime_id, precedence_key)
                _register_candidate(precedence_bank[((int(precedence_key[0]), int(precedence_key[1])), int(slot_id))], key, 'precedence', weight, slot_id, season_bucket, week_type, density_bucket, regime_id, precedence_key)
        bank[task_idx] = {
            'slot': slot_bank,
            'task': task_bank,
            'seasonal': seasonal_bank,
            'week_type': week_type_bank,
            'density': density_bank,
            'regime': regime_bank,
            'precedence': precedence_bank,
            'prototypes': prototypes,
        }
    return bank


def _ratio_for_match(counts: dict[Any, float], target_value: Any, total_support: float) -> float:
    if target_value is None or total_support <= 0.0:
        return 0.0
    return float(counts.get(target_value, 0.0) / max(total_support, 1e-8))


def _record_priority(rec: CandidateRecord, target_context: dict[str, Any], slot_id: int, source_name: str) -> float:
    exact_slot = float(rec.slot_support.get(int(slot_id), 0.0))
    nearest_support = 0.0
    if rec.slot_support:
        nearest_slot = min(rec.slot_support.keys(), key=lambda sid: abs(int(sid) - int(slot_id)))
        slot_gap = abs(int(nearest_slot) - int(slot_id))
        nearest_support = float(rec.slot_support.get(int(nearest_slot), 0.0)) / float(slot_gap + 1)
    season_match = _ratio_for_match(rec.season_counts, target_context.get('season_bucket'), rec.total_support)
    week_type_match = _ratio_for_match(rec.week_type_counts, target_context.get('week_type'), rec.total_support)
    density_match = _ratio_for_match(rec.density_counts, target_context.get('density_bucket'), rec.total_support)
    regime_match = _ratio_for_match(rec.regime_counts, target_context.get('regime_id'), rec.total_support)
    precedence_match = _ratio_for_match(rec.precedence_counts, target_context.get('precedence_key'), rec.total_support)
    source_support = float(rec.source_support.get(source_name, 0.0))
    return float(
        rec.total_support
        + 0.50 * exact_slot
        + 0.25 * nearest_support
        + 0.30 * season_match
        + 0.20 * week_type_match
        + 0.25 * density_match
        + 0.25 * regime_match
        + 0.20 * precedence_match
        + 0.20 * source_support
    )


def _candidate_row_from_record(
    rec: CandidateRecord,
    target_context: dict[str, Any],
    slot_id: int,
    selected_sources: set[str] | None = None,
) -> dict[str, Any]:
    sources = set(selected_sources or set())
    sources.update(str(name) for name, value in rec.source_support.items() if float(value) > 0.0)
    season_match = _ratio_for_match(rec.season_counts, target_context.get('season_bucket'), rec.total_support)
    week_type_match = _ratio_for_match(rec.week_type_counts, target_context.get('week_type'), rec.total_support)
    density_match = _ratio_for_match(rec.density_counts, target_context.get('density_bucket'), rec.total_support)
    regime_match = _ratio_for_match(rec.regime_counts, target_context.get('regime_id'), rec.total_support)
    precedence_match = _ratio_for_match(rec.precedence_counts, target_context.get('precedence_key'), rec.total_support)
    slot_exact_support = float(rec.slot_support.get(int(slot_id), 0.0))
    nearest_slot_gap = min((abs(int(sid) - int(slot_id)) for sid in rec.slot_support), default=99)
    neighbor_match = 1.0 / float(nearest_slot_gap + 1) if rec.slot_support else 0.0
    context_consistency = 0.28 * season_match + 0.16 * week_type_match + 0.24 * density_match + 0.20 * regime_match + 0.12 * precedence_match
    max_source_support = max((float(value) for value in rec.source_support.values()), default=0.0)
    return {
        'start_bin': int(rec.start_bin),
        'duration_bins': max(1, int(rec.duration_bins)),
        'support': float(rec.total_support),
        'num_sources': float(len(sources) or 1),
        'max_source_support': float(max_source_support),
        'source_strength': float(max_source_support / max(rec.total_support, 1e-8)) if rec.total_support > 0 else 0.0,
        'neighbor_slot_match': float(np.clip(max(slot_exact_support, neighbor_match), 0.0, 1.0)),
        'season_match': float(season_match),
        'week_type_match': float(week_type_match),
        'density_match': float(density_match),
        'regime_match': float(regime_match),
        'precedence_match': float(precedence_match),
        'context_consistency_score': float(np.clip(context_consistency, 0.0, 1.0)),
        'is_slot_source': 1.0 if 'slot' in sources else 0.0,
        'is_task_source': 1.0 if 'task' in sources else 0.0,
        'is_seasonal_source': 1.0 if 'seasonal' in sources else 0.0,
        'is_week_type_source': 1.0 if 'week_type' in sources else 0.0,
        'is_density_source': 1.0 if 'density' in sources else 0.0,
        'is_regime_source': 1.0 if 'regime' in sources else 0.0,
        'is_precedence_source': 1.0 if 'precedence' in sources else 0.0,
        'is_prototype_source': 1.0 if 'prototype' in sources else 0.0,
    }


def generate_prototype_jitter_candidates(
    anchor_start: int,
    anchor_duration: int,
    start_offsets: list[int] | None = None,
    duration_offsets: list[int] | None = None,
) -> list[tuple[int, int]]:
    start_offsets = list(DEFAULT_PROTOTYPE_START_OFFSETS if start_offsets is None else start_offsets)
    duration_offsets = list(DEFAULT_PROTOTYPE_DURATION_OFFSETS if duration_offsets is None else duration_offsets)
    rows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for start_delta in start_offsets:
        for duration_delta in duration_offsets:
            key = (max(0, int(anchor_start) + int(start_delta)), max(1, int(anchor_duration) + int(duration_delta)))
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    rows.sort(key=lambda item: (abs(item[0] - int(anchor_start)), abs(item[1] - int(anchor_duration)), item[0], item[1]))
    return rows


def _weighted_center_from_records(records: dict[tuple[int, int], CandidateRecord]) -> tuple[int, int] | None:
    if not records:
        return None
    starts = []
    durations = []
    weights = []
    for key, rec in records.items():
        starts.append(float(key[0]))
        durations.append(float(key[1]))
        weights.append(max(float(rec.total_support), 1e-6))
    weights_arr = np.asarray(weights, dtype=np.float64)
    starts_arr = np.asarray(starts, dtype=np.float64)
    durations_arr = np.asarray(durations, dtype=np.float64)
    start_val = int(round(float(np.average(starts_arr, weights=weights_arr))))
    duration_val = max(1, int(round(float(np.average(durations_arr, weights=weights_arr)))) )
    return (start_val, duration_val)


def gather_empirical_candidates(
    candidate_bank: EmpiricalCandidateBank,
    task_idx: int,
    slot_id: int,
    target_context: dict[str, Any] | None = None,
    neighbor_radius: int = 1,
    limit: int = 24,
    fallback_anchor: tuple[int, int] | None = None,
    source_quotas: dict[str, int] | None = None,
    prototype_start_offsets: list[int] | None = None,
    prototype_duration_offsets: list[int] | None = None,
) -> list[dict[str, Any]]:
    task_bank = candidate_bank.get(int(task_idx), {'slot': {}, 'task': {}, 'prototypes': []})
    target_context = dict(target_context or {})
    quotas = dict(DEFAULT_SOURCE_QUOTAS)
    quotas.update({str(k): int(v) for k, v in (source_quotas or {}).items() if int(v) > 0})

    merged: dict[tuple[int, int], dict[str, Any]] = {}

    def _source_names_from_row(row: dict[str, Any]) -> set[str]:
        mapping = {
            'is_slot_source': 'slot',
            'is_task_source': 'task',
            'is_seasonal_source': 'seasonal',
            'is_week_type_source': 'week_type',
            'is_density_source': 'density',
            'is_regime_source': 'regime',
            'is_precedence_source': 'precedence',
            'is_prototype_source': 'prototype',
        }
        names = {name for flag_key, name in mapping.items() if float(row.get(flag_key, 0.0)) > 0.0}
        names.update(str(v) for v in row.get('_selected_sources', []) or [])
        return names

    def _ingest(records: dict[tuple[int, int], CandidateRecord], source_name: str, quota: int) -> None:
        if quota <= 0 or not records:
            return
        ordered = sorted(records.items(), key=lambda item: (-_record_priority(item[1], target_context, int(slot_id), source_name), item[0][0], item[0][1]))
        taken = 0
        for key, rec in ordered:
            previous = merged.get(key, {})
            prev_sources = _source_names_from_row(previous)
            prev_sources.add(source_name)
            row = _candidate_row_from_record(rec, target_context, int(slot_id), selected_sources=prev_sources)
            row['support'] = max(float(previous.get('support', 0.0)), float(row.get('support', 0.0)))
            row['max_source_support'] = max(float(previous.get('max_source_support', 0.0)), float(row.get('max_source_support', 0.0)))
            row['_selected_sources'] = sorted(prev_sources)
            merged[key] = row
            taken += 1
            if taken >= int(quota):
                break

    slot_maps = task_bank.get('slot', {})
    slot_quota_total = int(quotas.get('slot', 0))
    per_neighbor_slot_quota = int(np.ceil(slot_quota_total / max(1, 2 * int(neighbor_radius) + 1))) if slot_quota_total > 0 else 0
    for neighbor_slot in range(int(slot_id) - int(neighbor_radius), int(slot_id) + int(neighbor_radius) + 1):
        _ingest(slot_maps.get(int(neighbor_slot), {}), 'slot', per_neighbor_slot_quota)

    _ingest(task_bank.get('task', {}), 'task', quotas.get('task', 0))
    _ingest(task_bank.get('seasonal', {}).get((str(target_context.get('season_bucket')), int(slot_id)), {}), 'seasonal', quotas.get('seasonal', 0))
    _ingest(task_bank.get('week_type', {}).get((str(target_context.get('week_type')), int(slot_id)), {}), 'week_type', quotas.get('week_type', 0))
    _ingest(task_bank.get('density', {}).get((str(target_context.get('density_bucket')), int(slot_id)), {}), 'density', quotas.get('density', 0))
    _ingest(task_bank.get('regime', {}).get((str(target_context.get('regime_id')), int(slot_id)), {}), 'regime', quotas.get('regime', 0))
    _ingest(task_bank.get('precedence', {}).get((tuple(target_context.get('precedence_key', (-1, -1))), int(slot_id)), {}), 'precedence', quotas.get('precedence', 0))

    prototype_anchors: list[tuple[int, int]] = []
    for proto in task_bank.get('prototypes', []):
        if abs(int(proto.slot_id) - int(slot_id)) <= max(1, int(neighbor_radius)):
            prototype_anchors.append((int(proto.center_bin), max(1, int(proto.duration_bins))))
    precedence_center = _weighted_center_from_records(task_bank.get('precedence', {}).get((tuple(target_context.get('precedence_key', (-1, -1))), int(slot_id)), {}))
    if precedence_center is not None:
        prototype_anchors.append(precedence_center)
    if fallback_anchor is not None:
        prototype_anchors.append((int(fallback_anchor[0]), max(1, int(fallback_anchor[1]))))

    proto_quota = quotas.get('prototype', 0)
    if proto_quota > 0 and prototype_anchors:
        seen_proto: set[tuple[int, int]] = set()
        for anchor_start, anchor_duration in prototype_anchors:
            for cand_start, cand_duration in generate_prototype_jitter_candidates(
                anchor_start,
                anchor_duration,
                start_offsets=prototype_start_offsets,
                duration_offsets=prototype_duration_offsets,
            ):
                key = (int(cand_start), int(cand_duration))
                if key in seen_proto:
                    continue
                seen_proto.add(key)
                if key not in merged:
                    pseudo = CandidateRecord(start_bin=key[0], duration_bins=key[1], total_support=0.05)
                    pseudo.add(
                        source='prototype',
                        weight=0.05,
                        slot_id=int(slot_id),
                        season_bucket=str(target_context.get('season_bucket', 'unknown')),
                        week_type=str(target_context.get('week_type', 'unknown')),
                        density_bucket=str(target_context.get('density_bucket', 'mid')),
                        regime_id=str(target_context.get('regime_id', 'unknown')),
                        precedence_key=tuple(target_context.get('precedence_key', (-1, -1))),
                    )
                    row = _candidate_row_from_record(pseudo, target_context, int(slot_id), selected_sources={'prototype'})
                    row['_selected_sources'] = ['prototype']
                    merged[key] = row
                if len([row for row in merged.values() if float(row.get('is_prototype_source', 0.0)) > 0.0]) >= int(proto_quota):
                    break
            if len([row for row in merged.values() if float(row.get('is_prototype_source', 0.0)) > 0.0]) >= int(proto_quota):
                break

    rows = list(merged.values())
    for row in rows:
        row.pop('_selected_sources', None)
    rows.sort(
        key=lambda row: (
            -(
                float(row.get('support', 0.0))
                + 0.30 * float(row.get('season_match', 0.0))
                + 0.20 * float(row.get('week_type_match', 0.0))
                + 0.25 * float(row.get('density_match', 0.0))
                + 0.25 * float(row.get('regime_match', 0.0))
                + 0.20 * float(row.get('precedence_match', 0.0))
                + 0.15 * float(row.get('neighbor_slot_match', 0.0))
                + 0.10 * float(row.get('source_strength', 0.0))
            ),
            int(row.get('start_bin', 0)),
            int(row.get('duration_bins', 1)),
        )
    )
    rows = rows[:max(1, int(limit))]
    if not rows and fallback_anchor is not None:
        rows = [{
            'start_bin': int(fallback_anchor[0]),
            'duration_bins': max(1, int(fallback_anchor[1])),
            'support': 0.0,
            'num_sources': 1.0,
            'max_source_support': 0.0,
            'source_strength': 0.0,
            'neighbor_slot_match': 1.0,
            'season_match': 0.0,
            'week_type_match': 0.0,
            'density_match': 0.0,
            'regime_match': 0.0,
            'precedence_match': 0.0,
            'context_consistency_score': 0.0,
            'is_slot_source': 0.0,
            'is_task_source': 0.0,
            'is_seasonal_source': 0.0,
            'is_week_type_source': 0.0,
            'is_density_source': 0.0,
            'is_regime_source': 0.0,
            'is_precedence_source': 0.0,
            'is_prototype_source': 1.0,
        }]
    return rows


def propose_extra_slots(
    series: SeriesBundle,
    template: TemplateWeek,
    task_idx: int,
    target_week_idx: int | None,
    required: int,
    used_keys: set[tuple[int, int]] | None = None,
    start_slot_id: int | None = None,
) -> list[tuple[int, int, float, int]]:
    extras: list[tuple[int, int, float, int]] = []
    prototypes = template.slot_prototypes_by_task.get(task_idx, [])
    seen: set[tuple[int, int]] = set(used_keys or set())
    if not seen:
        for proto in prototypes:
            seen.add((int(proto.center_bin), int(proto.duration_bins)))
    next_slot_id = int(start_slot_id if start_slot_id is not None else (max((int(proto.slot_id) for proto in prototypes), default=-1) + 1))
    for week_idx, score in zip(template.source_weeks, template.week_scores):
        for event in series.events[week_idx]:
            if event.task_idx != task_idx:
                continue
            key = (event.start_bin, event.duration_bins)
            if key in seen:
                continue
            seen.add(key)
            extras.append((int(event.start_bin), int(event.duration_bins), float(score), next_slot_id))
            next_slot_id += 1
    proto = task_prototype_from_history(series, len(series.week_starts) if target_week_idx is None else target_week_idx, task_idx)
    proto_key = (int(round(proto['start_bin'])), int(round(proto['duration_bins'])))
    if proto_key not in seen:
        extras.append((proto_key[0], proto_key[1], -0.25, next_slot_id))
    extras.sort(key=lambda x: x[2], reverse=True)
    return extras[:required]


def build_planned_slots_from_counts(
    series: SeriesBundle,
    template: TemplateWeek,
    task_names: list[str],
    count_predictions: dict[int, int],
    target_week_idx: int | None = None,
) -> list[dict[str, int | float | str]]:
    planned: list[dict[str, int | float | str]] = []
    for task_idx, pred_count_raw in count_predictions.items():
        pred_count = max(0, int(pred_count_raw))
        prototypes = sorted(template.slot_prototypes_by_task.get(task_idx, []), key=lambda p: (p.center_bin, -p.support))
        chosen: list[dict[str, int | float | str]] = []
        used_keys: set[tuple[int, int]] = set()
        next_slot_id = max((int(proto.slot_id) for proto in prototypes), default=-1) + 1
        for proto in prototypes[:pred_count]:
            chosen.append({
                'task_idx': int(task_idx),
                'task_type': str(task_names[task_idx]),
                'slot_id': int(proto.slot_id),
                'anchor_start_bin': int(proto.center_bin),
                'anchor_duration_bins': int(proto.duration_bins),
                'support': float(proto.support),
                'template_task_count': int(template.counts[task_idx]),
                'pred_task_count': int(pred_count),
            })
            used_keys.add((int(proto.center_bin), int(proto.duration_bins)))
        if pred_count > len(chosen):
            extras = propose_extra_slots(
                series,
                template,
                int(task_idx),
                target_week_idx,
                required=pred_count - len(chosen),
                used_keys=used_keys,
                start_slot_id=next_slot_id,
            )
            for start_bin, duration_bins, score, slot_id in extras:
                chosen.append({
                    'task_idx': int(task_idx),
                    'task_type': str(task_names[task_idx]),
                    'slot_id': int(slot_id),
                    'anchor_start_bin': int(start_bin),
                    'anchor_duration_bins': max(1, int(duration_bins)),
                    'support': float(score),
                    'template_task_count': int(template.counts[task_idx]),
                    'pred_task_count': int(pred_count),
                })
        planned.extend(chosen[:pred_count])
    planned.sort(key=lambda x: (int(x['anchor_start_bin']), str(x['task_type']), int(x['slot_id'])))
    return planned
