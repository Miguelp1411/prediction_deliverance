from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    BIN_MINUTES,
    CAP_INFERENCE_SCOPE,
    CHECKPOINT_DIR,
    DATA_PATH,
    DEFAULT_MAX_OCCURRENCES_PER_TASK,
    DEFAULT_MAX_TASKS_PER_WEEK,
    DEVICE,
    FEATURE_SCHEMA_VERSION,
    PREDICTION_DAY_TOPK,
    PREDICTION_TIME_TOPK,
    PREDICTION_USE_DURATION_MEDIAN_BLEND,
    PREDICTION_USE_REPAIR,
    POSTPROCESS_OVERRIDE_PATH,
    TEMPORAL_ANCHOR_MAX_SHIFT_BINS,
    TEMPORAL_ANCHOR_PROXIMITY_WEIGHT,
    TEMPORAL_DAY_CANDIDATE_TOPK,
    TEMPORAL_FINAL_STAGE,
    TEMPORAL_GATING_ALLOW_RERANK_FOR_CONFLICTS_AT_LEAST,
    TEMPORAL_GATING_ANCHOR_CLOSE_BINS,
    TEMPORAL_GATING_ENABLE,
    TEMPORAL_GATING_HIGH_CONFIDENCE_MARGIN,
    TEMPORAL_GATING_LOW_CONFIDENCE_MARGIN,
    TEMPORAL_GATING_MAX_RAW_CONFLICTS,
    TEMPORAL_GATING_MAX_STAGE_MOVE_RATE,
    TEMPORAL_GATING_MIN_ANCHOR_AGREEMENT,
    TEMPORAL_GATING_PREFER_RAW_WHEN_CLEAN,
    TEMPORAL_GATING_SCORE_ANCHOR_AGREEMENT_WEIGHT,
    TEMPORAL_GATING_SCORE_ANCHOR_DISTANCE_WEIGHT,
    TEMPORAL_GATING_SCORE_CONFIDENCE_WEIGHT,
    TEMPORAL_GATING_SCORE_CONFLICT_IMPROVEMENT_WEIGHT,
    TEMPORAL_GATING_SCORE_CONFLICT_WEIGHT,
    TEMPORAL_GATING_SCORE_MOVE_WEIGHT,
    TEMPORAL_NUM_ANCHOR_CANDIDATES,
    TEMPORAL_RERANK_BEAM_WIDTH,
    TEMPORAL_RERANK_MAX_CANDIDATES,
    TEMPORAL_RERANK_MIN_GAP_BINS,
    TEMPORAL_RERANK_MOVE_PENALTY,
    TEMPORAL_RERANK_ONLY_ON_CONFLICT,
    TEMPORAL_RERANK_ORDER_PENALTY,
    TEMPORAL_RERANK_OVERLAP_PENALTY,
    TEMPORAL_TIME_CANDIDATE_TOPK,
    TIMEZONE,
    TRAIN_RATIO,
    WINDOW_WEEKS,
    bins_per_day,
    num_time_bins,
)
from data.io import load_tasks_dataframe
from data.preprocessing import (
    build_context_sequence_features,
    build_prediction_occurrence_slots,
    build_temporal_context,
    clip_start_bin,
    denormalize_duration,
    infer_preprocessing_caps,
    prepare_data,
)
from models.occurrence_model import StructuredOccurrenceModel, TaskOccurrenceModel
from models.temporal_model import TemporalAssignmentModel
from utils.runtime import resolve_device
from utils.serialization import load_checkpoint


def _build_sequence(prepared, target_week_idx):
    return build_context_sequence_features(prepared.weeks, target_week_idx, WINDOW_WEEKS, len(prepared.task_names))


def _ensure_checkpoint_compatibility(ckpt: dict, prepared, checkpoint_name: str) -> None:
    metadata = ckpt.get('metadata', {}) or {}
    model_hparams = ckpt.get('model_hparams', {}) or {}
    issues: list[str] = []

    schema_version = int(metadata.get('feature_schema_version', 1))
    if schema_version != FEATURE_SCHEMA_VERSION:
        issues.append(
            f"feature_schema_version={schema_version} y el código actual espera {FEATURE_SCHEMA_VERSION}"
        )

    checkpoint_week_dim = int(metadata.get('week_feature_dim', -1))
    if checkpoint_week_dim != prepared.week_feature_dim:
        issues.append(
            f"week_feature_dim={checkpoint_week_dim} pero ahora el proyecto genera {prepared.week_feature_dim}"
        )

    checkpoint_history_dim = int(metadata.get('history_feature_dim', -1))
    if checkpoint_history_dim != prepared.history_feature_dim:
        issues.append(
            f"history_feature_dim={checkpoint_history_dim} pero ahora el proyecto genera {prepared.history_feature_dim}"
        )

    checkpoint_max_occurrences = metadata.get(
        'max_occurrences_per_task',
        model_hparams.get('max_occurrences_per_task', model_hparams.get('max_occurrences', metadata.get('max_count_cap', model_hparams.get('max_count_cap')))),
    )
    if checkpoint_max_occurrences is not None and _positive_int(checkpoint_max_occurrences, prepared.max_occurrences_per_task) != prepared.max_occurrences_per_task:
        issues.append(
            f"max_occurrences_per_task={checkpoint_max_occurrences} pero los datos preparados usan {prepared.max_occurrences_per_task}"
        )

    checkpoint_max_tasks = metadata.get('max_tasks_per_week', model_hparams.get('max_tasks_per_week'))
    if checkpoint_max_tasks is not None and _positive_int(checkpoint_max_tasks, prepared.max_tasks_per_week) != prepared.max_tasks_per_week:
        issues.append(
            f"max_tasks_per_week={checkpoint_max_tasks} pero los datos preparados usan {prepared.max_tasks_per_week}"
        )

    if issues:
        joined = '; '.join(issues)
        raise ValueError(
            f"Checkpoint incompatible ({checkpoint_name}): {joined}. Reentrena los modelos con el código actual antes de predecir."
        )


def _task_duration_median(prepared, task_name: str) -> float:
    return float(prepared.task_duration_medians.get(task_name, prepared.duration_min))


def _positive_int(value, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def _safe_log_prob(value: float) -> float:
    return math.log(max(float(value), 1e-9))


_ALLOWED_RUNTIME_OVERRIDES = {
    'TEMPORAL_ANCHOR_PROXIMITY_WEIGHT',
    'TEMPORAL_ANCHOR_MAX_SHIFT_BINS',
    'TEMPORAL_NUM_ANCHOR_CANDIDATES',
    'TEMPORAL_RERANK_BEAM_WIDTH',
    'TEMPORAL_RERANK_MAX_CANDIDATES',
    'TEMPORAL_RERANK_OVERLAP_PENALTY',
    'TEMPORAL_RERANK_ORDER_PENALTY',
    'TEMPORAL_RERANK_MOVE_PENALTY',
    'TEMPORAL_RERANK_ONLY_ON_CONFLICT',
    'TEMPORAL_GATING_ENABLE',
    'TEMPORAL_GATING_PREFER_RAW_WHEN_CLEAN',
    'TEMPORAL_GATING_HIGH_CONFIDENCE_MARGIN',
    'TEMPORAL_GATING_LOW_CONFIDENCE_MARGIN',
    'TEMPORAL_GATING_MAX_RAW_CONFLICTS',
    'TEMPORAL_GATING_ALLOW_RERANK_FOR_CONFLICTS_AT_LEAST',
    'TEMPORAL_GATING_MIN_ANCHOR_AGREEMENT',
    'TEMPORAL_GATING_ANCHOR_CLOSE_BINS',
    'TEMPORAL_GATING_MAX_STAGE_MOVE_RATE',
    'TEMPORAL_GATING_SCORE_CONFLICT_WEIGHT',
    'TEMPORAL_GATING_SCORE_CONFIDENCE_WEIGHT',
    'TEMPORAL_GATING_SCORE_MOVE_WEIGHT',
    'TEMPORAL_GATING_SCORE_ANCHOR_DISTANCE_WEIGHT',
    'TEMPORAL_GATING_SCORE_CONFLICT_IMPROVEMENT_WEIGHT',
    'TEMPORAL_GATING_SCORE_ANCHOR_AGREEMENT_WEIGHT',
}


def apply_runtime_overrides(overrides: dict | None) -> dict[str, object]:
    if not overrides:
        return {}
    applied: dict[str, object] = {}
    global_vars = globals()
    for key, value in dict(overrides).items():
        if key not in _ALLOWED_RUNTIME_OVERRIDES:
            continue
        global_vars[key] = value
        applied[key] = value
    return applied


def load_runtime_overrides(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        return {}
    overrides = payload.get('best_params', payload)
    if not isinstance(overrides, dict):
        return {}
    return apply_runtime_overrides(overrides)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _candidate_from_absolute_logits(day_logits_row: torch.Tensor, time_logits_row: torch.Tensor) -> list[tuple[int, float]]:
    day_probs = torch.softmax(day_logits_row, dim=-1)
    time_probs = torch.softmax(time_logits_row, dim=-1)
    day_k = min(max(PREDICTION_DAY_TOPK, TEMPORAL_DAY_CANDIDATE_TOPK), day_probs.numel())
    time_k = min(max(PREDICTION_TIME_TOPK, TEMPORAL_TIME_CANDIDATE_TOPK), time_probs.numel())
    day_values, day_indices = torch.topk(day_probs, k=day_k)
    time_values, time_indices = torch.topk(time_probs, k=time_k)

    candidates: list[tuple[int, float]] = []
    seen: set[int] = set()
    for d_prob, d_idx in zip(day_values.tolist(), day_indices.tolist()):
        for t_prob, t_idx in zip(time_values.tolist(), time_indices.tolist()):
            start_bin = clip_start_bin(int(d_idx) * bins_per_day() + int(t_idx))
            if start_bin in seen:
                continue
            seen.add(start_bin)
            score = _safe_log_prob(d_prob) + _safe_log_prob(t_prob)
            candidates.append((start_bin, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _normalize_stage_name(stage: str | None) -> str:
    stage = str(stage or 'final').strip().lower()
    if stage == 'final':
        stage = str(TEMPORAL_FINAL_STAGE).strip().lower()
    return 'gated' if stage in {'auto', 'gated'} else stage


def _required_stages_for(stage: str | None, include_repair: bool = False) -> tuple[str, ...]:
    normalized = _normalize_stage_name(stage)
    if normalized == 'raw':
        return ('raw',)
    if normalized == 'anchors':
        return ('raw', 'anchors')
    if normalized == 'rerank':
        return ('raw', 'anchors', 'rerank')
    if normalized == 'gated':
        return ('raw', 'anchors', 'rerank', 'gated')
    if normalized == 'repair':
        return ('raw', 'anchors', 'rerank', 'repair')
    available = 'raw, anchors, rerank, gated, repair'
    raise ValueError(f"ensemble_stage='{stage}' no soportado. Disponibles: {available}")


def _merge_candidate_scores(
    model_candidates: list[tuple[int, float]],
    anchor_candidates: tuple[int, ...],
    anchor_candidate_weights: tuple[float, ...],
    anchor_start_bin: int,
) -> list[tuple[int, float]]:
    merged: dict[int, float] = {}
    best_model_bin = int(model_candidates[0][0]) if model_candidates else int(anchor_start_bin)
    max_anchor_shift = max(0, int(TEMPORAL_ANCHOR_MAX_SHIFT_BINS))
    for start_bin, score in model_candidates:
        distance_penalty = TEMPORAL_ANCHOR_PROXIMITY_WEIGHT * (abs(int(start_bin) - int(anchor_start_bin)) / max(bins_per_day(), 1))
        merged[int(start_bin)] = max(merged.get(int(start_bin), -float('inf')), float(score) - float(distance_penalty))

    for idx, start_bin in enumerate(anchor_candidates[: max(1, TEMPORAL_NUM_ANCHOR_CANDIDATES)]):
        start_bin = int(start_bin)
        if max_anchor_shift > 0 and abs(start_bin - best_model_bin) > max_anchor_shift:
            continue
        weight = float(anchor_candidate_weights[idx]) if idx < len(anchor_candidate_weights) else 0.0
        if weight <= 0.0:
            continue
        anchor_score = math.log(max(weight, 1e-6))
        distance_penalty = 0.5 * TEMPORAL_ANCHOR_PROXIMITY_WEIGHT * (abs(start_bin - int(anchor_start_bin)) / max(bins_per_day(), 1))
        merged[start_bin] = max(merged.get(start_bin, -float('inf')), anchor_score - distance_penalty)

    if not merged and model_candidates:
        merged[int(model_candidates[0][0])] = float(model_candidates[0][1])

    ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    return [(int(start_bin), float(score)) for start_bin, score in ranked[: max(1, TEMPORAL_RERANK_MAX_CANDIDATES)]]


def _events_overlap(start_a: int, dur_a: float, start_b: int, dur_b: float) -> bool:
    dur_a_bins = max(1, int(round(float(dur_a) / BIN_MINUTES)))
    dur_b_bins = max(1, int(round(float(dur_b) / BIN_MINUTES)))
    end_a = int(start_a) + dur_a_bins
    end_b = int(start_b) + dur_b_bins
    min_gap = max(0, int(TEMPORAL_RERANK_MIN_GAP_BINS))
    return not (end_a + min_gap <= int(start_b) or end_b + min_gap <= int(start_a))


def _same_task_order_penalty(state_items: list[dict], item: dict, candidate_start_bin: int) -> float:
    penalty = 0.0
    for prev in state_items:
        if int(prev['task_id']) != int(item['task_id']):
            continue
        if int(prev['occurrence_slot']) < int(item['occurrence_slot']) and int(candidate_start_bin) < int(prev['start_bin']):
            penalty += TEMPORAL_RERANK_ORDER_PENALTY
        if int(prev['occurrence_slot']) > int(item['occurrence_slot']) and int(candidate_start_bin) > int(prev['start_bin']):
            penalty += TEMPORAL_RERANK_ORDER_PENALTY
    return penalty


def _beam_rerank_weekly(items: list[dict]) -> list[dict]:
    if not items:
        return []

    ordered_items = sorted(items, key=lambda item: (int(item['anchor_start_bin']), int(item['task_id']), int(item['occurrence_slot'])))
    beam: list[tuple[float, list[dict]]] = [(0.0, [])]
    beam_width = max(1, int(TEMPORAL_RERANK_BEAM_WIDTH))

    for item in ordered_items:
        next_beam: list[tuple[float, list[dict]]] = []
        candidates = item['candidate_starts'][: max(1, TEMPORAL_RERANK_MAX_CANDIDATES)]
        if not candidates:
            candidates = [(int(item['anchor_start_bin']), -1.0)]
        raw_start_bin = int(item.get('raw_start_bin', item.get('start_bin', item['anchor_start_bin'])))
        for state_score, state_items in beam:
            for candidate_start_bin, candidate_score in candidates:
                overlap_penalty = 0.0
                for prev in state_items:
                    if _events_overlap(int(candidate_start_bin), float(item['duration']), int(prev['start_bin']), float(prev['duration'])):
                        overlap_penalty += TEMPORAL_RERANK_OVERLAP_PENALTY
                order_penalty = _same_task_order_penalty(state_items, item, int(candidate_start_bin))
                move_penalty = TEMPORAL_RERANK_MOVE_PENALTY * (abs(int(candidate_start_bin) - raw_start_bin) / max(bins_per_day(), 1))
                total_score = float(state_score) + float(candidate_score) - float(overlap_penalty) - float(order_penalty) - float(move_penalty)
                next_item = dict(item)
                next_item['start_bin'] = int(candidate_start_bin)
                next_beam.append((total_score, state_items + [next_item]))
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    best_items = max(beam, key=lambda x: x[0])[1]
    return sorted(best_items, key=lambda x: (int(x['start_bin']), x['task_name']))


def _collect_conflict_components(items: list[dict]) -> list[list[int]]:
    if not items:
        return []
    n = len(items)
    adjacency = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _events_overlap(int(items[i]['start_bin']), float(items[i]['duration']), int(items[j]['start_bin']), float(items[j]['duration'])):
                adjacency[i].add(j)
                adjacency[j].add(i)
    visited = [False] * n
    components: list[list[int]] = []
    for i in range(n):
        if visited[i] or not adjacency[i]:
            continue
        stack = [i]
        comp: list[int] = []
        visited[i] = True
        while stack:
            node = stack.pop()
            comp.append(node)
            for nei in sorted(adjacency[node]):
                if not visited[nei]:
                    visited[nei] = True
                    stack.append(nei)
        components.append(sorted(comp))
    return components


def _rerank_conflict_components(items: list[dict]) -> list[dict]:
    if not items or not TEMPORAL_RERANK_ONLY_ON_CONFLICT:
        return _beam_rerank_weekly(items)
    ordered = _sort_predictions(_clone_predictions(items))
    components = _collect_conflict_components(ordered)
    if not components:
        return ordered
    reranked = [dict(item) for item in ordered]
    for indices in components:
        component_items = [ordered[idx] for idx in indices]
        reranked_component = _beam_rerank_weekly(component_items)
        for idx, item in zip(indices, reranked_component):
            reranked[idx] = item
    return _sort_predictions(_clone_predictions(reranked))


def _repair_predictions(predictions: list[dict]) -> list[dict]:
    if not predictions:
        return []
    repaired = [dict(item) for item in sorted(predictions, key=lambda x: (int(x['start_bin']), x['task_name']))]
    last_end = None
    for item in repaired:
        start_bin = int(item['start_bin'])
        duration_bins = max(1, int(round(float(item['duration']) / BIN_MINUTES)))
        if last_end is not None and start_bin < last_end:
            item['start_bin'] = int(last_end)
            start_bin = int(item['start_bin'])
        last_end = start_bin + duration_bins
    return repaired


def _sort_predictions(predictions: list[dict]) -> list[dict]:
    return sorted(predictions, key=lambda x: (int(x['start_bin']), x['task_name'], int(x.get('occurrence_slot', 0))))


def _clone_predictions(predictions: list[dict]) -> list[dict]:
    cloned: list[dict] = []
    for item in predictions:
        cloned_item = dict(item)
        if 'candidate_starts' in cloned_item:
            cloned_item['candidate_starts'] = [(int(s), float(sc)) for s, sc in cloned_item['candidate_starts']]
        if 'model_candidate_starts' in cloned_item:
            cloned_item['model_candidate_starts'] = [(int(s), float(sc)) for s, sc in cloned_item['model_candidate_starts']]
        cloned.append(cloned_item)
    return cloned


def _candidate_margin(item: dict) -> float:
    candidates = item.get('candidate_starts') or item.get('model_candidate_starts') or []
    if not candidates:
        return 0.0
    top_score = float(candidates[0][1])
    next_score = float(candidates[1][1]) if len(candidates) > 1 else top_score - 1.0
    return top_score - next_score


def _count_global_overlaps(items: list[dict]) -> int:
    if len(items) < 2:
        return 0
    ordered = _sort_predictions(_clone_predictions(items))
    overlaps = 0
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if _events_overlap(int(ordered[i]['start_bin']), float(ordered[i]['duration']), int(ordered[j]['start_bin']), float(ordered[j]['duration'])):
                overlaps += 1
    return overlaps


def _summarize_stage(stage_name: str, predictions: list[dict], raw_predictions: list[dict]) -> dict[str, float]:
    raw_map = {
        (int(item['task_id']), int(item['occurrence_slot'])): item
        for item in raw_predictions
    }
    margins = []
    move_rates = []
    anchor_distances = []
    anchor_agreements = []
    for item in predictions:
        margins.append(_candidate_margin(item))
        raw_item = raw_map.get((int(item['task_id']), int(item['occurrence_slot'])))
        raw_start = int(raw_item.get('start_bin', item.get('raw_start_bin', item['start_bin']))) if raw_item else int(item.get('raw_start_bin', item['start_bin']))
        move_rates.append(abs(int(item['start_bin']) - raw_start) / max(bins_per_day(), 1))
        anchor_start = int(item.get('anchor_start_bin', raw_start))
        anchor_distances.append(abs(int(item['start_bin']) - anchor_start) / max(bins_per_day(), 1))
        anchor_agreements.append(1.0 if abs(raw_start - anchor_start) <= max(0, int(TEMPORAL_GATING_ANCHOR_CLOSE_BINS)) else 0.0)

    conflict_count = float(_count_global_overlaps(predictions))
    raw_conflicts = float(_count_global_overlaps(raw_predictions))
    conflict_improvement = max(raw_conflicts - conflict_count, 0.0)
    summary = {
        'stage_name': stage_name,
        'conflict_count': conflict_count,
        'avg_margin': _mean(margins),
        'move_rate': _mean(move_rates),
        'anchor_distance': _mean(anchor_distances),
        'anchor_agreement_rate': _mean(anchor_agreements),
        'conflict_improvement': conflict_improvement,
    }
    summary['selection_score'] = (
        TEMPORAL_GATING_SCORE_CONFIDENCE_WEIGHT * summary['avg_margin']
        - TEMPORAL_GATING_SCORE_CONFLICT_WEIGHT * summary['conflict_count']
        - TEMPORAL_GATING_SCORE_MOVE_WEIGHT * summary['move_rate']
        - TEMPORAL_GATING_SCORE_ANCHOR_DISTANCE_WEIGHT * summary['anchor_distance']
        + TEMPORAL_GATING_SCORE_CONFLICT_IMPROVEMENT_WEIGHT * summary['conflict_improvement']
        + TEMPORAL_GATING_SCORE_ANCHOR_AGREEMENT_WEIGHT * summary['anchor_agreement_rate']
    )
    return summary


def _select_gated_stage(stage_outputs: dict[str, list[dict]]) -> tuple[str, dict[str, dict[str, float]]]:
    candidate_names = [name for name in ('raw', 'anchors', 'rerank', 'repair') if name in stage_outputs]
    if not candidate_names:
        return 'raw', {}
    raw_predictions = stage_outputs.get('raw', [])
    snapshots = {name: _summarize_stage(name, stage_outputs[name], raw_predictions) for name in candidate_names}
    raw_snapshot = snapshots.get('raw', {'conflict_count': 0.0, 'avg_margin': 0.0, 'selection_score': -float('inf')})

    if TEMPORAL_GATING_ENABLE and TEMPORAL_GATING_PREFER_RAW_WHEN_CLEAN:
        if raw_snapshot['conflict_count'] <= float(TEMPORAL_GATING_MAX_RAW_CONFLICTS) and raw_snapshot['avg_margin'] >= float(TEMPORAL_GATING_HIGH_CONFIDENCE_MARGIN):
            return 'raw', snapshots

    viable: list[tuple[str, float]] = []
    for name in candidate_names:
        snap = snapshots[name]
        if name == 'rerank' and raw_snapshot['conflict_count'] < float(TEMPORAL_GATING_ALLOW_RERANK_FOR_CONFLICTS_AT_LEAST):
            continue
        if name == 'anchors' and snap['anchor_agreement_rate'] < float(TEMPORAL_GATING_MIN_ANCHOR_AGREEMENT) and raw_snapshot['avg_margin'] >= float(TEMPORAL_GATING_LOW_CONFIDENCE_MARGIN):
            continue
        if name in {'anchors', 'rerank', 'repair'} and snap['move_rate'] > float(TEMPORAL_GATING_MAX_STAGE_MOVE_RATE) and snap['conflict_improvement'] <= 0.0:
            continue
        viable.append((name, float(snap['selection_score'])))

    if not viable:
        return 'raw', snapshots
    best_stage = max(viable, key=lambda item: item[1])[0]
    return best_stage, snapshots


def _build_stage_predictions(base_predictions: list[dict], include_repair: bool, requested_stages: tuple[str, ...] | None = None) -> dict[str, list[dict]]:
    requested_stages = tuple(requested_stages or ('raw', 'anchors', 'rerank', 'gated', 'repair'))
    stage_outputs: dict[str, list[dict]] = {
        'raw': _sort_predictions(_clone_predictions(base_predictions)),
    }

    need_anchors = any(stage in requested_stages for stage in ('anchors', 'rerank', 'gated', 'repair'))
    need_rerank = any(stage in requested_stages for stage in ('rerank', 'gated', 'repair'))
    need_repair = include_repair and 'repair' in requested_stages
    need_gated = 'gated' in requested_stages

    if need_anchors:
        anchor_predictions: list[dict] = []
        for item in base_predictions:
            merged_candidates = _merge_candidate_scores(
                item['model_candidate_starts'],
                item['anchor_candidates'],
                item['anchor_candidate_weights'],
                int(item['anchor_start_bin']),
            )
            anchor_item = dict(item)
            anchor_item['candidate_starts'] = merged_candidates
            anchor_item['start_bin'] = int(merged_candidates[0][0] if merged_candidates else item['start_bin'])
            anchor_predictions.append(anchor_item)
        stage_outputs['anchors'] = _sort_predictions(_clone_predictions(anchor_predictions))

        if need_rerank:
            reranked = _rerank_conflict_components(_clone_predictions(anchor_predictions))
            stage_outputs['rerank'] = _sort_predictions(_clone_predictions(reranked))

            if need_repair:
                repaired = _repair_predictions(_clone_predictions(reranked))
                stage_outputs['repair'] = _sort_predictions(_clone_predictions(repaired))

    if need_gated:
        gated_stage_name, gated_snapshots = _select_gated_stage(stage_outputs)
        stage_outputs['gated'] = _sort_predictions(_clone_predictions(stage_outputs.get(gated_stage_name, stage_outputs['raw'])))
        stage_outputs['_meta_gated_stage_name'] = gated_stage_name
        stage_outputs['_meta_gated_snapshots'] = gated_snapshots

    return {stage: stage_outputs[stage] for stage in requested_stages if stage in stage_outputs} | {k: v for k, v in stage_outputs.items() if str(k).startswith('_meta_')}


def predict_next_week_staged(occurrence_model, temporal_model, prepared, target_week_idx, device, include_repair: bool | None = None, requested_stage: str | None = None):
    include_repair = PREDICTION_USE_REPAIR if include_repair is None else bool(include_repair)
    occurrence_model.eval()
    temporal_model.eval()

    with torch.no_grad():
        seq = _build_sequence(prepared, target_week_idx)
        base_sequence = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
        count_logits = occurrence_model(base_sequence)
        pred_counts = torch.argmax(count_logits, dim=-1).cpu().numpy()[0]

        task_ids, occurrence_slots, predicted_count_norms = [], [], []
        occurrence_progresses = []
        history_features, anchor_days, anchor_time_bins, anchor_start_bins = [], [], [], []
        anchor_candidates_list, anchor_candidate_weights_list = [], []

        for task_id, count in enumerate(pred_counts):
            count = int(min(count, prepared.max_count_cap))
            if count == 0:
                continue
            selected_slots = build_prediction_occurrence_slots(
                prepared.weeks,
                target_week_idx,
                task_id,
                count,
                max_occurrences_per_task=prepared.max_occurrences_per_task,
            )
            for slot_position, occurrence_slot in enumerate(selected_slots):
                context = build_temporal_context(
                    prepared.weeks,
                    target_week_idx,
                    task_id,
                    occurrence_slot,
                    prepared.duration_min,
                    prepared.duration_max,
                    max_occurrences_per_task=prepared.max_occurrences_per_task,
                    predicted_count=count,
                )
                task_ids.append(task_id)
                occurrence_slots.append(int(occurrence_slot))
                predicted_count_norms.append(float(count / prepared.max_count_cap))
                denom = max(count - 1, 1)
                occurrence_progresses.append(float(slot_position / denom) if count > 1 else 0.0)
                history_features.append(context.history_features)
                anchor_days.append(context.anchor_day)
                anchor_time_bins.append(context.anchor_time_bin)
                anchor_start_bins.append(int(context.anchor_start_bin))
                anchor_candidates_list.append(tuple(int(x) for x in context.anchor_candidates))
                anchor_candidate_weights_list.append(tuple(float(x) for x in context.anchor_candidate_weights))

        if not task_ids:
            requested_stages = _required_stages_for(requested_stage or ('repair' if include_repair else TEMPORAL_FINAL_STAGE), include_repair=include_repair)
            return {stage: [] for stage in requested_stages}

        batch_size = len(task_ids)
        sequence_context = temporal_model.encode_sequence(base_sequence).repeat(batch_size, 1)
        outputs = temporal_model.forward_with_context(
            sequence_context=sequence_context,
            task_id=torch.tensor(task_ids, dtype=torch.long, device=device),
            occurrence_slot=torch.tensor(occurrence_slots, dtype=torch.long, device=device),
            history_features=torch.tensor(np.stack(history_features), dtype=torch.float32, device=device),
            predicted_count_norm=torch.tensor(predicted_count_norms, dtype=torch.float32, device=device),
            occurrence_progress=torch.tensor(occurrence_progresses, dtype=torch.float32, device=device),
            anchor_day=torch.tensor(anchor_days, dtype=torch.long, device=device),
            anchor_time_bin=torch.tensor(anchor_time_bins, dtype=torch.long, device=device),
        )

        base_predictions = []
        inferred_device_uid = _infer_prediction_device_uid(prepared)
        for i in range(batch_size):
            task_name = prepared.task_names[task_ids[i]]
            pred_duration = denormalize_duration(
                float(outputs['pred_duration_norm'][i].item()),
                prepared.duration_min,
                prepared.duration_max,
            )
            median_duration = _task_duration_median(prepared, task_name)
            duration = (
                (1.0 - PREDICTION_USE_DURATION_MEDIAN_BLEND) * pred_duration
                + PREDICTION_USE_DURATION_MEDIAN_BLEND * median_duration
            )
            model_candidates = _candidate_from_absolute_logits(
                outputs['day_logits'][i],
                outputs['time_of_day_logits'][i],
            )
            raw_candidates = model_candidates[: max(1, TEMPORAL_RERANK_MAX_CANDIDATES)]
            raw_start_bin = int(raw_candidates[0][0] if raw_candidates else anchor_start_bins[i])
            base_predictions.append({
                'task_id': int(task_ids[i]),
                'task_name': task_name,
                'type': task_name,
                'occurrence_slot': int(occurrence_slots[i]),
                'anchor_start_bin': int(anchor_start_bins[i]),
                'anchor_candidates': tuple(int(x) for x in anchor_candidates_list[i]),
                'anchor_candidate_weights': tuple(float(x) for x in anchor_candidate_weights_list[i]),
                'start_bin': raw_start_bin,
                'raw_start_bin': raw_start_bin,
                'duration': float(duration),
                'device_uid': inferred_device_uid,
                'model_candidate_starts': raw_candidates,
                'candidate_starts': raw_candidates,
            })

    requested_stages = _required_stages_for(requested_stage or ('repair' if include_repair else TEMPORAL_FINAL_STAGE), include_repair=include_repair)
    return _build_stage_predictions(base_predictions, include_repair=include_repair, requested_stages=requested_stages)


def predict_next_week(
    occurrence_model,
    temporal_model,
    prepared,
    target_week_idx,
    device,
    ensemble_stage: str = 'final',
    return_stage_outputs: bool = False,
    include_repair: bool | None = None,
):
    selected_stage = _normalize_stage_name(ensemble_stage)
    stage_for_request = 'repair' if (selected_stage == 'repair' and (include_repair is None or include_repair)) else selected_stage
    if return_stage_outputs:
        stage_outputs = predict_next_week_staged(
            occurrence_model,
            temporal_model,
            prepared,
            target_week_idx,
            device,
            include_repair=include_repair,
            requested_stage='repair' if (include_repair is None or include_repair) else 'gated',
        )
        return stage_outputs

    stage_outputs = predict_next_week_staged(
        occurrence_model,
        temporal_model,
        prepared,
        target_week_idx,
        device,
        include_repair=include_repair,
        requested_stage=stage_for_request,
    )
    if selected_stage not in stage_outputs:
        available = ', '.join(stage_outputs.keys())
        raise ValueError(f"ensemble_stage='{selected_stage}' no soportado. Disponibles: {available}")
    return _sort_predictions(_clone_predictions(stage_outputs[selected_stage]))


def _extract_preprocessing_caps(occ_ckpt: dict, tmp_ckpt: dict, df: pd.DataFrame | None = None) -> tuple[int, int]:
    occ_metadata = occ_ckpt.get('metadata', {}) or {}
    tmp_metadata = tmp_ckpt.get('metadata', {}) or {}
    occ_hparams = occ_ckpt.get('model_hparams', {}) or {}
    tmp_hparams = tmp_ckpt.get('model_hparams', {}) or {}

    data_occ_cap, data_tasks_cap = (None, None)
    if df is not None and not df.empty:
        df_for_caps = df.copy()
        df_for_caps['week_start'] = df_for_caps['start_time'].dt.normalize() - pd.to_timedelta(df_for_caps['start_time'].dt.dayofweek, unit='D')
        task_names = sorted(df_for_caps['task_name'].astype(str).unique().tolist())
        task_to_id = {name: idx for idx, name in enumerate(task_names)}
        df_for_caps['task_id'] = df_for_caps['task_name'].map(task_to_id)
        data_occ_cap, data_tasks_cap = infer_preprocessing_caps(df_for_caps)

    occ_cap = _positive_int(
        tmp_metadata.get(
            'max_occurrences_per_task',
            occ_metadata.get(
                'max_occurrences_per_task',
                tmp_hparams.get(
                    'max_occurrences_per_task',
                    occ_hparams.get(
                        'max_occurrences_per_task',
                        tmp_hparams.get(
                            'max_occurrences',
                            occ_hparams.get(
                                'max_count_cap',
                                tmp_metadata.get(
                                    'max_count_cap',
                                    occ_metadata.get('max_count_cap', data_occ_cap or DEFAULT_MAX_OCCURRENCES_PER_TASK),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        data_occ_cap or DEFAULT_MAX_OCCURRENCES_PER_TASK,
    )
    tasks_cap = _positive_int(
        tmp_metadata.get(
            'max_tasks_per_week',
            occ_metadata.get(
                'max_tasks_per_week',
                tmp_hparams.get(
                    'max_tasks_per_week',
                    occ_hparams.get('max_tasks_per_week', data_tasks_cap or DEFAULT_MAX_TASKS_PER_WEEK),
                ),
            ),
        ),
        data_tasks_cap or DEFAULT_MAX_TASKS_PER_WEEK,
    )
    return occ_cap, tasks_cap


def _load_models(prepared, device: torch.device, occ_ckpt: dict | None = None, tmp_ckpt: dict | None = None):
    occ_ckpt = occ_ckpt or load_checkpoint(CHECKPOINT_DIR / 'occurrence_model.pt', map_location=device)
    tmp_ckpt = tmp_ckpt or load_checkpoint(CHECKPOINT_DIR / 'temporal_model.pt', map_location=device)

    _ensure_checkpoint_compatibility(occ_ckpt, prepared, 'occurrence_model.pt')
    _ensure_checkpoint_compatibility(tmp_ckpt, prepared, 'temporal_model.pt')

    occ_h = occ_ckpt.get('model_hparams', {})
    tmp_h = tmp_ckpt.get('model_hparams', {})

    occ_kind = str(occ_h.get('model_kind', occ_ckpt.get('metadata', {}).get('occurrence_model_kind', 'neural')))
    if occ_kind == 'structured_lag4':
        occurrence_model = StructuredOccurrenceModel(
            input_dim=int(occ_h.get('input_dim', prepared.week_feature_dim)),
            num_tasks=int(occ_h.get('num_tasks', len(prepared.task_names))),
            max_count_cap=int(occ_h.get('max_count_cap', prepared.max_count_cap)),
            lag_weeks=int(occ_h.get('lag_weeks', 4)),
        ).to(device)
    elif occ_kind == 'task_gru':
        occurrence_model = TaskOccurrenceModel(
            input_dim=int(occ_h.get('input_dim', prepared.week_feature_dim)),
            num_tasks=int(occ_h.get('num_tasks', len(prepared.task_names))),
            max_count_cap=int(occ_h.get('max_count_cap', prepared.max_count_cap)),
            hidden_size=int(occ_h.get('hidden_size', 128)),
            num_layers=int(occ_h.get('num_layers', 2)),
            dropout=float(occ_h.get('dropout', 0.15)),
            seasonal_lags=tuple(occ_h.get('seasonal_lags', occ_ckpt.get('metadata', {}).get('occurrence_seasonal_lags', [4, 26, 52]))),
            seasonal_lag_weights=tuple(occ_h.get('seasonal_lag_weights', occ_ckpt.get('metadata', {}).get('occurrence_seasonal_lag_weights', [0.10, 0.25, 0.65]))),
            seasonal_baseline_logit=float(occ_h.get('seasonal_baseline_logit', occ_ckpt.get('metadata', {}).get('occurrence_seasonal_baseline_logit', 2.50))),
            task_delta_bounds=[tuple(bounds) for bounds in occ_h.get('task_delta_bounds', occ_ckpt.get('metadata', {}).get('occurrence_task_delta_bounds', []))],
            delta_outside_range_logit_penalty_per_step=float(occ_h.get('delta_outside_range_logit_penalty_per_step', occ_ckpt.get('metadata', {}).get('occurrence_delta_outside_range_logit_penalty_per_step', -2.0))),
        ).to(device)
    else:
        raise ValueError(f"occurrence model kind no soportado: {occ_kind}")
    occurrence_model.load_state_dict(occ_ckpt['state_dict'], strict=False)

    temporal_model = TemporalAssignmentModel(
        sequence_dim=int(tmp_h.get('sequence_dim', prepared.week_feature_dim)),
        history_feature_dim=int(tmp_h.get('history_feature_dim', prepared.history_feature_dim)),
        num_tasks=int(tmp_h.get('num_tasks', len(prepared.task_names))),
        max_occurrences=int(tmp_h.get('max_occurrences', prepared.max_count_cap)),
        hidden_size=int(tmp_h.get('hidden_size', 160)),
        num_layers=int(tmp_h.get('num_layers', 2)),
        dropout=float(tmp_h.get('dropout', 0.15)),
        task_embed_dim=int(tmp_h.get('task_embed_dim', 32)),
        occurrence_embed_dim=int(tmp_h.get('occurrence_embed_dim', 16)),
        day_embed_dim=int(tmp_h.get('day_embed_dim', 8)),
    ).to(device)
    temporal_model.load_state_dict(tmp_ckpt['state_dict'])

    return occurrence_model, temporal_model


def _prediction_week_start(prepared) -> pd.Timestamp:
    if not prepared.weeks:
        raise ValueError('No hay semanas preparadas para generar predicción')
    return prepared.weeks[-1].week_start + pd.Timedelta(days=7)


def _to_timestamp(week_start: pd.Timestamp, start_bin: int) -> pd.Timestamp:
    return week_start + pd.Timedelta(minutes=int(start_bin) * BIN_MINUTES)


def _format_utc_timestamp(ts: pd.Timestamp) -> str:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts.strftime('%Y-%m-%dT%H:%M:%S.000Z')


def _infer_prediction_device_uid(prepared) -> str | None:
    if not hasattr(prepared, 'df') or prepared.df is None or prepared.df.empty or 'device_uid' not in prepared.df.columns:
        return None
    values = sorted({str(v) for v in prepared.df['device_uid'].dropna().astype(str) if str(v).strip()})
    return values[0] if len(values) == 1 else None


def _materialize_prediction_times(prepared, predictions: list[dict]) -> list[dict]:
    pred_week_start = _prediction_week_start(prepared)
    materialized = []
    inferred_device_uid = _infer_prediction_device_uid(prepared)

    for item in predictions:
        start_ts = _to_timestamp(pred_week_start, int(item['start_bin']))
        end_ts = start_ts + pd.Timedelta(minutes=float(item['duration']))

        task_type = item.get('type', item['task_name'])

        materialized.append({
            'uid': None,
            'device_uid': inferred_device_uid,
            'task_name': None,
            'type': task_type,
            'status': None,
            'start_time': _format_utc_timestamp(start_ts),
            'end_time': _format_utc_timestamp(end_ts),
            'mileage': 0,
            'misc': None,
            'waypoints': [],
        })

    return sorted(materialized, key=lambda x: x['start_time'])


def main():
    parser = argparse.ArgumentParser(
        description='Genera la predicción de la siguiente semana usando los checkpoints guardados.'
    )
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='Ruta al JSON de datos históricos.')
    parser.add_argument('--output', type=str, default='predicted_next_week.json', help='Ruta del JSON de salida.')
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Dispositivo: 'auto' o cualquier dispositivo valido de torch ('cpu', 'cuda', 'mps', 'xpu'...).",
    )
    parser.add_argument('--stage', type=str, default='final', help='Etapa del ensamblado: final|raw|anchors|rerank|gated|repair')
    parser.add_argument('--overrides', type=str, default=None, help='JSON opcional con overrides de postproceso/gating.')
    args = parser.parse_args()

    device = resolve_device(args.device or DEVICE)
    override_path = args.overrides or str(POSTPROCESS_OVERRIDE_PATH)
    applied = load_runtime_overrides(override_path)
    if applied:
        print(f'Overrides aplicados: {sorted(applied.keys())}')
    df = load_tasks_dataframe(args.data, timezone=TIMEZONE)
    occ_ckpt = load_checkpoint(CHECKPOINT_DIR / 'occurrence_model.pt', map_location=device)
    tmp_ckpt = load_checkpoint(CHECKPOINT_DIR / 'temporal_model.pt', map_location=device)
    max_occurrences_per_task, max_tasks_per_week = _extract_preprocessing_caps(occ_ckpt, tmp_ckpt, df=df)
    prepared = prepare_data(
        df,
        train_ratio=TRAIN_RATIO,
        max_occurrences_per_task=max_occurrences_per_task,
        max_tasks_per_week=max_tasks_per_week,
        cap_inference_scope=CAP_INFERENCE_SCOPE,
    )
    occurrence_model, temporal_model = _load_models(prepared, device, occ_ckpt=occ_ckpt, tmp_ckpt=tmp_ckpt)

    predictions = predict_next_week(
        occurrence_model,
        temporal_model,
        prepared,
        len(prepared.weeks),
        device,
        ensemble_stage=args.stage,
    )
    materialized = _materialize_prediction_times(prepared, predictions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(materialized, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Semana generada: {output_path}')
    print(f'Tareas predichas: {len(materialized)}')


if __name__ == '__main__':
    main()
