from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hybrid_schedule.data.features import EventItem, SeriesBundle, SlotPrototype, assign_events_to_prototypes, task_prototype_from_history, task_slot_prototypes


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


EmpiricalCandidateBank = dict[int, dict[str, dict]]


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
    slot_prototypes_by_task = {
        task_idx: task_slot_prototypes(series, ref_idx, task_idx, max_slots=max_slot_prototypes)
        for task_idx in range(series.counts.shape[1])
    }

    return TemplateWeek(
        primary_week_idx=primary,
        source_weeks=[r.week_idx for r in retrieved],
        week_scores=[float(r.score) for r in retrieved],
        counts=counts,
        events=list(series.events[primary]),
        support_by_slot=support_by_slot,
        slot_prototypes_by_task=slot_prototypes_by_task,
    )


def build_empirical_candidate_bank(series: SeriesBundle, template: TemplateWeek) -> EmpiricalCandidateBank:
    raw_weights = np.exp(np.asarray(template.week_scores, dtype=np.float64))
    total = float(raw_weights.sum())
    week_weights = raw_weights / total if total > 0 else np.ones_like(raw_weights) / max(len(raw_weights), 1)
    bank: EmpiricalCandidateBank = {}
    for task_idx in range(series.counts.shape[1]):
        prototypes = template.slot_prototypes_by_task.get(task_idx, [])
        slot_bank: dict[int, dict[tuple[int, int], float]] = {}
        task_bank: dict[tuple[int, int], float] = {}
        if not prototypes:
            bank[task_idx] = {'slot': slot_bank, 'task': task_bank}
            continue
        for weight, week_idx in zip(week_weights.tolist(), template.source_weeks):
            task_events = [evt for evt in series.events[week_idx] if evt.task_idx == task_idx]
            if not task_events:
                continue
            assignments = assign_events_to_prototypes(task_events, prototypes)
            for slot_id, evt, _ in assignments:
                key = (int(evt.start_bin), int(evt.duration_bins))
                slot_bank.setdefault(int(slot_id), {})
                slot_bank[int(slot_id)][key] = slot_bank[int(slot_id)].get(key, 0.0) + float(weight)
                task_bank[key] = task_bank.get(key, 0.0) + float(weight)
        bank[task_idx] = {'slot': slot_bank, 'task': task_bank}
    return bank


def gather_empirical_candidates(
    candidate_bank: EmpiricalCandidateBank,
    task_idx: int,
    slot_id: int,
    neighbor_radius: int = 1,
    limit: int = 24,
    fallback_anchor: tuple[int, int] | None = None,
) -> list[tuple[int, int, float]]:
    task_bank = candidate_bank.get(int(task_idx), {'slot': {}, 'task': {}})
    candidates: dict[tuple[int, int], float] = {}
    for neighbor_slot in range(int(slot_id) - int(neighbor_radius), int(slot_id) + int(neighbor_radius) + 1):
        for key, support_score in task_bank.get('slot', {}).get(int(neighbor_slot), {}).items():
            candidates[key] = max(candidates.get(key, 0.0), float(support_score))
    if not candidates:
        candidates = {tuple(key): float(score) for key, score in task_bank.get('task', {}).items()}
    ordered = sorted(candidates.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    rows = [(int(start_bin), int(duration_bins), float(score)) for (start_bin, duration_bins), score in ordered[:max(1, int(limit))]]
    if not rows and fallback_anchor is not None:
        rows = [(int(fallback_anchor[0]), max(1, int(fallback_anchor[1])), 0.0)]
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
