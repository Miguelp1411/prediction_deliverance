from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CAP_INFERENCE_SCOPE,
    BIN_MINUTES,
    CHECKPOINT_DIR,
    DATA_PATH,
    DATALOADER_NUM_WORKERS,
    DATALOADER_PERSISTENT_WORKERS,
    DATALOADER_PIN_MEMORY,
    DATALOADER_PREFETCH_FACTOR,
    DAY_EMBED_DIM,
    DEVICE,
    ENABLE_CUDNN_BENCHMARK,
    ENABLE_TF32,
    OCC_BATCH_SIZE,
    OCC_DROPOUT,
    OCC_EMBED_DIM,
    OCC_HIDDEN_SIZE,
    OCC_LAG_WEEKS,
    OCC_NUM_LAYERS,
    OCC_USE_AMP,
    OCC_SEASONAL_BASELINE_LOGIT,
    OCC_SEASONAL_LAGS,
    OCC_SEASONAL_LAG_WEIGHTS,
    OCC_EXPECTED_COUNT_MAE_WEIGHT,
    OCC_USE_TASK_DELTA_RANGES,
    OCC_DELTA_RANGE_LOW_QUANTILE,
    OCC_DELTA_RANGE_HIGH_QUANTILE,
    OCC_DELTA_RANGE_MARGIN,
    OCC_DELTA_RANGE_MIN_RADIUS,
    OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP,
    OCCURRENCE_MODEL_KIND,
    PREDICTION_USE_REPAIR,
    REPORTS_DIR,
    SEED,
    TASK_EMBED_DIM,
    TEMPORAL_COUNT_BLEND_TARGET_WEIGHT,
    TIMEZONE,
    TMP_AMP_DTYPE,
    TMP_BATCH_SIZE,
    TMP_DROPOUT,
    TMP_E2E_EVAL_EVERY,
    TMP_HIDDEN_SIZE,
    TMP_LR,
    TMP_MAX_EPOCHS,
    TMP_NUM_LAYERS,
    TMP_PATIENCE,
    TMP_SCHEDULER_PATIENCE,
    TMP_USE_AMP,
    TMP_WEIGHT_DECAY,
    TRAIN_RATIO,
    num_time_bins,
    num_day_classes,
    num_time_of_day_classes,
    OCC_LR,
    OCC_WEIGHT_DECAY,
    OCC_MAX_EPOCHS,
    OCC_PATIENCE,
    OCC_SCHEDULER_PATIENCE,
    OCC_MONITOR_NAME,
    OCC_MONITOR_MODE,
    OCC_MONITOR_MIN_DELTA
)
from data.datasets import OccurrenceDataset, TemporalDataset, build_occurrence_count_lookup, build_split_indices
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data, serialize_metadata
from evaluation.weekly_stats import evaluate_weekly_predictions
from models.occurrence_model import StructuredOccurrenceModel, TaskOccurrenceModel
from models.temporal_model import TemporalAssignmentModel
from predict import predict_next_week, predict_next_week_staged
from training.engine import evaluate_epoch, fit_model
from training.losses import OccurrenceLoss, TemporalLoss
from training.metrics import occurrence_metrics, temporal_metrics
from utils.runtime import resolve_device
from utils.serialization import save_checkpoint


OCCURRENCE_REPORT_KEYS = (
    'count_exact_acc',
    'close_acc_1',
    'close_acc_2',
    'count_mae',
    'weekly_total_mae',
)

OCCURRENCE_AUX_REPORT_KEYS = (
    'presence_precision',
    'presence_recall',
    'presence_f1',
    'occurrence_selection_score',
    'occurrence_realistic_score',
)

TEMPORAL_REPORT_KEYS = (
    'start_exact_acc',
    'start_tol_acc_5m',
    'start_tol_acc_10m',
    'start_mae_minutes',
    'duration_mae_minutes',
    'day_exact_acc',
    'time_of_day_exact_acc',
)

ENSEMBLE_REPORT_KEYS = (
    'task_precision',
    'task_recall',
    'task_f1',
    'time_exact_accuracy',
    'time_close_accuracy_5m',
    'time_close_accuracy_10m',
    'duration_close_accuracy',
    'start_mae_minutes',
    'overlap_same_device_count',
    'overlap_global_count',
    'unknown_device_count',
)

PERCENTAGE_REPORT_KEYS = {
    'task_precision',
    'task_recall',
    'task_f1',
    'time_exact_accuracy',
    'time_close_accuracy_5m',
    'time_close_accuracy_10m',
    'duration_close_accuracy',
}

WEEKLY_AGGREGATE_SUM_KEYS = (
    'total_tasks',
    'predicted_tasks',
    'correct_tasks',
    'time_exact_count',
    'time_close_5m_count',
    'time_close_10m_count',
    'duration_close_count',
    'start_abs_error_sum',
    'matched_pairs',
    'overlap_same_device_count',
    'overlap_global_count',
    'unknown_device_count',
    'matching_reassigned_pairs',
    'matching_assignment_shift_sum',
    'matching_compared_pairs',
    'matching_ordered_exact_count',
    'matching_hungarian_exact_count',
    'matching_ordered_close_5m_count',
    'matching_hungarian_close_5m_count',
    'matching_ordered_cost_sum',
    'matching_hungarian_cost_sum',
    'matching_cost_gain_sum',
)

ABLATION_STAGE_ORDER = ('raw', 'anchors', 'rerank')
ABLATION_STAGE_REPORT_KEYS = (
    'task_precision',
    'task_recall',
    'task_f1',
    'time_exact_accuracy',
    'time_close_accuracy_5m',
    'time_close_accuracy_10m',
    'start_mae_minutes',
    'matching_reassignment_rate',
    'matching_cost_gain_sum',
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    if device.type == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = ENABLE_TF32
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = ENABLE_TF32


def make_dataloader(dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    num_workers = max(0, int(DATALOADER_NUM_WORKERS))
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': bool(DATALOADER_PIN_MEMORY and device.type == 'cuda'),
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(DATALOADER_PERSISTENT_WORKERS)
        kwargs['prefetch_factor'] = max(2, int(DATALOADER_PREFETCH_FACTOR))
    return DataLoader(dataset, **kwargs)


class ZeroLoss(torch.nn.Module):
    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            device = next(iter(outputs.values())).device
        else:
            device = outputs.device
        return torch.zeros((), dtype=torch.float32, device=device)


def _round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _select_metrics(metrics: dict[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    selected: dict[str, float] = {}
    for key in keys:
        if key in metrics:
            selected[key] = _round_float(metrics[key])
    return selected


def _build_per_task_report(per_task_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for task_name, stats in sorted(per_task_stats.items()):
        report[task_name] = {
            'total': _round_float(stats.get('total', 0.0)),
            'task_accuracy': _round_float(stats.get('task_accuracy', 0.0) * 100.0),
            'time_exact_accuracy': _round_float(stats.get('time_exact_accuracy', 0.0) * 100.0),
        }
    return report


def _empty_weekly_totals() -> dict[str, float]:
    return {key: 0.0 for key in WEEKLY_AGGREGATE_SUM_KEYS}


def _accumulate_weekly_stats(
    totals: dict[str, float],
    per_task_accumulator: dict[str, dict[str, float]],
    stats: dict[str, float],
) -> None:
    for key in WEEKLY_AGGREGATE_SUM_KEYS:
        totals[key] += float(stats.get(key, 0.0))
    for task_name, task_stats in stats.get('per_task', {}).items():
        accumulator = per_task_accumulator.setdefault(task_name, {'total': 0.0, 'task_correct': 0.0, 'time_exact': 0.0})
        accumulator['total'] += float(task_stats.get('total', 0.0))
        accumulator['task_correct'] += float(task_stats.get('task_correct', 0.0))
        accumulator['time_exact'] += float(task_stats.get('time_exact', 0.0))


def _finalize_weekly_totals(
    totals: dict[str, float],
    per_task_accumulator: dict[str, dict[str, float]],
) -> dict[str, float]:
    total_tasks = totals['total_tasks']
    predicted_tasks = totals['predicted_tasks']
    correct_tasks = totals['correct_tasks']
    matched_pairs = totals['matched_pairs']

    precision = correct_tasks / predicted_tasks if predicted_tasks > 0 else 0.0
    recall = correct_tasks / total_tasks if total_tasks > 0 else 0.0
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    result = {
        **totals,
        'task_accuracy': recall,
        'task_precision': precision,
        'task_recall': recall,
        'task_f1': f1,
        'time_exact_accuracy': totals['time_exact_count'] / total_tasks if total_tasks > 0 else 0.0,
        'time_close_accuracy': totals['time_close_5m_count'] / total_tasks if total_tasks > 0 else 0.0,
        'time_close_accuracy_5m': totals['time_close_5m_count'] / total_tasks if total_tasks > 0 else 0.0,
        'time_close_accuracy_10m': totals['time_close_10m_count'] / total_tasks if total_tasks > 0 else 0.0,
        'duration_close_accuracy': totals['duration_close_count'] / total_tasks if total_tasks > 0 else 0.0,
        'start_mae_minutes': totals['start_abs_error_sum'] / matched_pairs if matched_pairs > 0 else 0.0,
        'overlap_count': totals['overlap_same_device_count'],
        'per_task': {},
        'matching_reassignment_rate': totals['matching_reassigned_pairs'] / totals['matching_compared_pairs'] if totals['matching_compared_pairs'] > 0 else 0.0,
        'matching_avg_assignment_shift': totals['matching_assignment_shift_sum'] / totals['matching_reassigned_pairs'] if totals['matching_reassigned_pairs'] > 0 else 0.0,
    }

    for task_name, counts in sorted(per_task_accumulator.items()):
        task_total = counts['total']
        result['per_task'][task_name] = {
            'total': counts['total'],
            'task_correct': counts['task_correct'],
            'time_exact': counts['time_exact'],
            'task_accuracy': counts['task_correct'] / task_total if task_total > 0 else 0.0,
            'time_exact_accuracy': counts['time_exact'] / task_total if task_total > 0 else 0.0,
        }

    result['e2e_task_acc'] = result['task_accuracy'] * 100.0
    result['e2e_start_exact_acc'] = result['time_exact_accuracy'] * 100.0
    result['e2e_start_tol_acc_5m'] = result['time_close_accuracy_5m'] * 100.0
    result['e2e_overlap_count'] = result['overlap_same_device_count']
    result['e2e_joint_score'] = result['e2e_task_acc'] + result['e2e_start_exact_acc']
    return result


def _prediction_identity(item: dict) -> tuple[str, int]:
    return str(item.get('task_name', item.get('type'))), int(item.get('occurrence_slot', 0))


def _stage_transition_stats(prev_stage: list[dict], next_stage: list[dict]) -> dict[str, float]:
    prev_map = {_prediction_identity(item): item for item in prev_stage}
    next_map = {_prediction_identity(item): item for item in next_stage}
    shared_keys = sorted(set(prev_map) & set(next_map))
    if not shared_keys:
        return {
            'compared_predictions': 0.0,
            'events_moved': 0.0,
            'movement_sum_minutes': 0.0,
            'movement_mean_minutes': 0.0,
            'movement_mean_moved_minutes': 0.0,
        }

    moved = 0.0
    movement_sum_minutes = 0.0
    for key in shared_keys:
        delta_bins = abs(int(next_map[key]['start_bin']) - int(prev_map[key]['start_bin']))
        delta_minutes = float(delta_bins * BIN_MINUTES)
        movement_sum_minutes += delta_minutes
        if delta_bins > 0:
            moved += 1.0

    compared = float(len(shared_keys))
    return {
        'compared_predictions': compared,
        'events_moved': moved,
        'movement_sum_minutes': movement_sum_minutes,
        'movement_mean_minutes': movement_sum_minutes / compared if compared > 0 else 0.0,
        'movement_mean_moved_minutes': movement_sum_minutes / moved if moved > 0 else 0.0,
    }


def _to_json_ready(value):
    if isinstance(value, dict):
        return {str(k): _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_final_report(report: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(_to_json_ready(report), fh, ensure_ascii=False, indent=2)


def print_final_report(report: dict):
    print("\nReporte final de entrenamiento")
    print('-' * 50)

    occurrence = report['occurrence']
    print(f"OccurrenceModel ({occurrence['type']})")
    for key, value in occurrence['metrics'].items():
        suffix = '%' if 'acc' in key or key.endswith('f1') else ''
        print(f'  {key:<26} {value:.4f}{suffix}')

    temporal = report['temporal']
    print("\nTemporalModel")
    for key, value in temporal['metrics'].items():
        unit = ' min' if 'mae_minutes' in key else ('%' if 'acc' in key else '')
        print(f'  {key:<26} {value:.4f}{unit}')

    ensemble = report['ensemble']
    print("\nEnsamblado final")
    for key, value in ensemble['metrics'].items():
        if key in PERCENTAGE_REPORT_KEYS:
            print(f'  {key:<26} {value:.4f}%')
        elif key == 'start_mae_minutes':
            print(f'  {key:<26} {value:.4f} min')
        else:
            print(f'  {key:<26} {value:.4f}')

    ablation = report.get('ensemble_ablation', {})
    if ablation.get('stage_order'):
        print("\nAblación del ensamblado (validación)")
        for stage in ablation['stage_order']:
            metrics = ablation.get('stages', {}).get(stage, {})
            if not metrics:
                continue
            exact = metrics.get('time_exact_accuracy', 0.0)
            tol5 = metrics.get('time_close_accuracy_5m', 0.0)
            mae = metrics.get('start_mae_minutes', 0.0)
            print(f"  {stage:<8} exact={exact:.2f}% | ±5m={tol5:.2f}% | mae={mae:.2f} min")
        for transition_name, metrics in ablation.get('transitions', {}).items():
            print(
                f"  {transition_name:<16} movidos={metrics.get('events_moved', 0.0):.0f}/"
                f"{metrics.get('compared_predictions', 0.0):.0f} "
                f"({metrics.get('events_moved_rate', 0.0):.2f}%) | "
                f"despl medio={metrics.get('movement_mean_moved_minutes', 0.0):.2f} min"
            )

    print(f"\nReporte guardado en: {report['report_path']}")


def aggregate_weekly_stats(prepared, week_indices, occurrence_model, temporal_model, device):
    if not week_indices:
        return {}

    totals = _empty_weekly_totals()
    per_task_accumulator: dict[str, dict[str, float]] = {}

    for idx in week_indices:
        pred_week = predict_next_week(occurrence_model, temporal_model, prepared, idx, device)
        stats = evaluate_weekly_predictions(prepared.weeks[idx], pred_week)
        _accumulate_weekly_stats(totals, per_task_accumulator, stats)

    return _finalize_weekly_totals(totals, per_task_accumulator)


def aggregate_weekly_ablation_stats(prepared, week_indices, occurrence_model, temporal_model, device, include_repair: bool = False):
    if not week_indices:
        return {'stage_order': [], 'stages': {}, 'transitions': {}}

    stage_order = list(ABLATION_STAGE_ORDER)
    if include_repair:
        stage_order.append('repair')

    stage_totals = {stage: _empty_weekly_totals() for stage in stage_order}
    stage_per_task = {stage: {} for stage in stage_order}
    transition_totals: dict[str, dict[str, float]] = {
        f'{prev}_to_{curr}': {
            'compared_predictions': 0.0,
            'events_moved': 0.0,
            'movement_sum_minutes': 0.0,
        }
        for prev, curr in zip(stage_order, stage_order[1:])
    }

    for idx in week_indices:
        staged_predictions = predict_next_week_staged(
            occurrence_model,
            temporal_model,
            prepared,
            idx,
            device,
            include_repair=include_repair,
        )
        true_week = prepared.weeks[idx]
        for stage in stage_order:
            stats = evaluate_weekly_predictions(true_week, staged_predictions[stage])
            _accumulate_weekly_stats(stage_totals[stage], stage_per_task[stage], stats)
        for prev, curr in zip(stage_order, stage_order[1:]):
            key = f'{prev}_to_{curr}'
            delta = _stage_transition_stats(staged_predictions[prev], staged_predictions[curr])
            transition_totals[key]['compared_predictions'] += float(delta['compared_predictions'])
            transition_totals[key]['events_moved'] += float(delta['events_moved'])
            transition_totals[key]['movement_sum_minutes'] += float(delta['movement_sum_minutes'])

    stages_report = {
        stage: _finalize_weekly_totals(stage_totals[stage], stage_per_task[stage])
        for stage in stage_order
    }
    transitions_report = {}
    for key, totals in transition_totals.items():
        compared = float(totals['compared_predictions'])
        moved = float(totals['events_moved'])
        movement_sum = float(totals['movement_sum_minutes'])
        transitions_report[key] = {
            'compared_predictions': compared,
            'events_moved': moved,
            'events_moved_rate': moved / compared if compared > 0 else 0.0,
            'movement_sum_minutes': movement_sum,
            'movement_mean_minutes': movement_sum / compared if compared > 0 else 0.0,
            'movement_mean_moved_minutes': movement_sum / moved if moved > 0 else 0.0,
        }

    return {
        'stage_order': stage_order,
        'stages': stages_report,
        'transitions': transitions_report,
    }


class TemporalE2EEvaluator:

    def __init__(self, prepared, week_indices, occurrence_model, device):
        self.prepared = prepared
        self.week_indices = list(week_indices)
        self.occurrence_model = occurrence_model
        self.device = device

    @torch.no_grad()
    def __call__(self, temporal_model):
        temporal_model.eval()
        stats = aggregate_weekly_stats(self.prepared, self.week_indices, self.occurrence_model, temporal_model, self.device)
        compound_score = (
            float(stats.get('time_close_accuracy_5m', 0.0)) * 100.0
            - 0.05 * float(stats.get('start_mae_minutes', 0.0))
        )
        enriched = dict(stats)
        enriched['e2e_compound_score'] = compound_score
        return {
            k: enriched.get(k, 0.0)
            for k in [
                'e2e_task_acc',
                'e2e_start_exact_acc',
                'e2e_start_tol_acc_5m',
                'e2e_overlap_count',
                'e2e_joint_score',
                'e2e_compound_score',
                'time_close_accuracy_5m',
                'start_mae_minutes',
            ]
        }


def _build_ablation_report(ablation_stats: dict) -> dict:
    if not ablation_stats:
        return {'stage_order': [], 'stages': {}, 'transitions': {}}

    stages_report: dict[str, dict[str, float]] = {}
    for stage in ablation_stats.get('stage_order', []):
        metrics = dict(ablation_stats.get('stages', {}).get(stage, {}))
        selected = _select_metrics(metrics, ABLATION_STAGE_REPORT_KEYS)
        for key in ('task_precision', 'task_recall', 'task_f1', 'time_exact_accuracy', 'time_close_accuracy_5m', 'time_close_accuracy_10m', 'matching_reassignment_rate'):
            if key in selected:
                selected[key] = _round_float(selected[key] * 100.0)
        stages_report[stage] = selected

    transitions_report: dict[str, dict[str, float]] = {}
    for key, metrics in ablation_stats.get('transitions', {}).items():
        transitions_report[key] = {
            'compared_predictions': _round_float(metrics.get('compared_predictions', 0.0)),
            'events_moved': _round_float(metrics.get('events_moved', 0.0)),
            'events_moved_rate': _round_float(metrics.get('events_moved_rate', 0.0) * 100.0),
            'movement_mean_minutes': _round_float(metrics.get('movement_mean_minutes', 0.0)),
            'movement_mean_moved_minutes': _round_float(metrics.get('movement_mean_moved_minutes', 0.0)),
        }

    return {
        'stage_order': list(ablation_stats.get('stage_order', [])),
        'stages': stages_report,
        'transitions': transitions_report,
    }


def build_final_report(occ_state, occ_val_metrics, tmp_state, tmp_val_metrics, ensemble_stats, ablation_stats, report_path: Path) -> dict:
    best_gap_count_exact_acc = float('nan')
    if (
        getattr(occ_state, 'best_train_metrics', None)
        and getattr(occ_state, 'best_val_metrics', None)
        and 'count_exact_acc' in occ_state.best_train_metrics
        and 'count_exact_acc' in occ_state.best_val_metrics
    ):
        best_gap_count_exact_acc = (
            occ_state.best_train_metrics['count_exact_acc']
            - occ_state.best_val_metrics['count_exact_acc']
        )

    occurrence_block = {
        'type': 'rule_based' if OCCURRENCE_MODEL_KIND == 'structured_lag4' else 'trainable',
        'metrics': _select_metrics(occ_val_metrics, OCCURRENCE_REPORT_KEYS),
        'aux_metrics': _select_metrics(occ_val_metrics, OCCURRENCE_AUX_REPORT_KEYS),
    }
    if OCCURRENCE_MODEL_KIND != 'structured_lag4':
        occurrence_block['meta'] = {
            'best_epoch': int(occ_state.best_epoch),
            'monitor_name': occ_state.monitor_name,
            'monitor_value': _round_float(occ_state.best_metric),
            'occurrence_train_val_gap_count_exact_acc': _round_float(best_gap_count_exact_acc),
        }

    temporal_metrics_report = _select_metrics(tmp_val_metrics, TEMPORAL_REPORT_KEYS)
    ensemble_metrics_report = _select_metrics(ensemble_stats, ENSEMBLE_REPORT_KEYS)
    for key in PERCENTAGE_REPORT_KEYS:
        if key in ensemble_metrics_report:
            ensemble_metrics_report[key] = _round_float(ensemble_metrics_report[key] * 100.0)

    return {
        'occurrence': occurrence_block,
        'temporal': {
            'metrics': temporal_metrics_report,
            'meta': {
                'best_epoch': int(tmp_state.best_epoch),
                'monitor_name': tmp_state.monitor_name,
                'monitor_value': _round_float(tmp_state.best_metric),
            },
        },
        'ensemble': {
            'metrics': ensemble_metrics_report,
        },
        'ensemble_ablation': _build_ablation_report(ablation_stats),
        'per_task_final': _build_per_task_report(ensemble_stats.get('per_task', {})),
        'meta': {
            'best_epoch_temporal': int(tmp_state.best_epoch),
            'monitor_name': tmp_state.monitor_name,
            'monitor_value': _round_float(tmp_state.best_metric),
            'prediction_use_repair': bool(PREDICTION_USE_REPAIR),
            'occurrence_model_kind': OCCURRENCE_MODEL_KIND,
        },
        'report_path': str(report_path),
    }


def _compute_task_delta_bounds(prepared, target_week_indices: list[int]) -> list[tuple[int, int]]:
    full_range = [(-prepared.max_count_cap, prepared.max_count_cap) for _ in prepared.task_names]
    if not OCC_USE_TASK_DELTA_RANGES or not target_week_indices:
        return full_range

    seasonal_weights = np.asarray(OCC_SEASONAL_LAG_WEIGHTS, dtype=np.float32)
    if seasonal_weights.size != len(OCC_SEASONAL_LAGS) or float(seasonal_weights.sum()) <= 0.0:
        seasonal_weights = np.asarray([float(lag) for lag in OCC_SEASONAL_LAGS], dtype=np.float32)
    seasonal_weights = seasonal_weights / max(float(seasonal_weights.sum()), 1e-6)

    per_task_deltas: list[list[int]] = [[] for _ in prepared.task_names]
    for target_week_idx in target_week_indices:
        lag_counts: list[np.ndarray] = []
        lag_masks: list[np.ndarray] = []
        for lag in OCC_SEASONAL_LAGS:
            source_idx = int(target_week_idx) - int(lag)
            if 0 <= source_idx < len(prepared.weeks):
                lag_counts.append(prepared.weeks[source_idx].counts.astype(np.float32, copy=False))
                lag_masks.append(np.ones(len(prepared.task_names), dtype=np.float32))
            else:
                lag_counts.append(np.zeros(len(prepared.task_names), dtype=np.float32))
                lag_masks.append(np.zeros(len(prepared.task_names), dtype=np.float32))
        lag_counts_arr = np.stack(lag_counts, axis=0)
        lag_masks_arr = np.stack(lag_masks, axis=0)
        weighted_masks = lag_masks_arr * seasonal_weights[:, None]
        denom = weighted_masks.sum(axis=0)
        has_signal = denom > 0
        baseline = np.where(
            has_signal,
            (lag_counts_arr * weighted_masks).sum(axis=0) / np.clip(denom, 1e-6, None),
            0.0,
        )
        baseline = np.rint(baseline).astype(np.int64)
        actual_counts = prepared.weeks[int(target_week_idx)].counts.astype(np.int64, copy=False)
        deltas = actual_counts - baseline
        for task_id, valid in enumerate(has_signal.tolist()):
            if valid:
                per_task_deltas[task_id].append(int(deltas[task_id]))

    bounds: list[tuple[int, int]] = []
    low_q = float(np.clip(OCC_DELTA_RANGE_LOW_QUANTILE, 0.0, 1.0))
    high_q = float(np.clip(OCC_DELTA_RANGE_HIGH_QUANTILE, low_q, 1.0))
    margin = max(0, int(OCC_DELTA_RANGE_MARGIN))
    min_radius = max(0, int(OCC_DELTA_RANGE_MIN_RADIUS))
    for task_deltas in per_task_deltas:
        if not task_deltas:
            bounds.append((-prepared.max_count_cap, prepared.max_count_cap))
            continue
        deltas_arr = np.asarray(task_deltas, dtype=np.float32)
        lower = int(np.floor(np.quantile(deltas_arr, low_q))) - margin
        upper = int(np.ceil(np.quantile(deltas_arr, high_q))) + margin
        lower = min(lower, 0, -min_radius)
        upper = max(upper, 0, min_radius)
        lower = max(-prepared.max_count_cap, lower)
        upper = min(prepared.max_count_cap, upper)
        if lower > upper:
            lower, upper = -prepared.max_count_cap, prepared.max_count_cap
        bounds.append((int(lower), int(upper)))
    return bounds


def build_occurrence_model(prepared, occ_train: OccurrenceDataset, device: torch.device):
    if OCCURRENCE_MODEL_KIND == 'structured_lag4':
        model = StructuredOccurrenceModel(
            prepared.week_feature_dim,
            len(prepared.task_names),
            prepared.max_count_cap,
            lag_weeks=OCC_LAG_WEEKS,
        ).to(device)

        target_counts = (
            torch.stack([item['target_counts'] for item in occ_train], dim=0)
            if len(occ_train) > 0 else None
        )
        model.fit(target_counts)

        state = SimpleNamespace(
            best_epoch=0,
            best_metric=float('nan'),
            best_train_loss=0.0,
            best_val_loss=0.0,
            final_train_loss=0.0,
            final_val_loss=0.0,
            best_train_metrics={},
            best_val_metrics={},
            final_train_metrics={},
            final_val_metrics={},
            monitor_name='rule_based',
            monitor_mode='max',
        )
        return model, state

    if OCCURRENCE_MODEL_KIND == 'task_gru':
        task_delta_bounds = _compute_task_delta_bounds(prepared, list(occ_train.week_indices))
        model = TaskOccurrenceModel(
            prepared.week_feature_dim,
            len(prepared.task_names),
            prepared.max_count_cap,
            OCC_HIDDEN_SIZE,
            OCC_NUM_LAYERS,
            OCC_DROPOUT,
            seasonal_lags=OCC_SEASONAL_LAGS,
            seasonal_lag_weights=OCC_SEASONAL_LAG_WEIGHTS,
            seasonal_baseline_logit=OCC_SEASONAL_BASELINE_LOGIT,
            task_delta_bounds=task_delta_bounds,
            delta_outside_range_logit_penalty_per_step=OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP,
        ).to(device)
        return model, None

    raise ValueError(f"OCCURRENCE_MODEL_KIND no soportado: {OCCURRENCE_MODEL_KIND}")

def main():
    train_start_time = time.perf_counter()
    set_seed(SEED)
    device = resolve_device(DEVICE)
    configure_runtime(device)
    print(f"\nDevice de entrenamiento: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"AMP temporal: {'ON' if TMP_USE_AMP else 'OFF'} ({TMP_AMP_DTYPE})")

    print("\n[1/6] Cargando datos...", flush=True)
    df = load_tasks_dataframe(DATA_PATH, timezone=TIMEZONE)
    print("[2/6] Preprocesando datos...", flush=True)
    prepared = prepare_data(df, train_ratio=TRAIN_RATIO, cap_inference_scope=CAP_INFERENCE_SCOPE, show_progress=True)
    print("[3/6] Construyendo partición train/val...", flush=True)
    split = build_split_indices(prepared, train_ratio=TRAIN_RATIO)
    print("[4/6] Construyendo OccurrenceDataset...", flush=True)
    occ_train = OccurrenceDataset(prepared, split.train_target_week_indices)
    occ_val = OccurrenceDataset(prepared, split.val_target_week_indices)
    print("[5/6] Preparando OccurrenceModel...", flush=True)
    occurrence_model, occ_state = build_occurrence_model(prepared, occ_train, device)

    occ_train_loader = make_dataloader(
        occ_train,
        batch_size=OCC_BATCH_SIZE,
        shuffle=(OCCURRENCE_MODEL_KIND == 'task_gru'),
        device=device,
    )
    occ_val_loader = make_dataloader(
        occ_val,
        batch_size=OCC_BATCH_SIZE,
        shuffle=False,
        device=device,
    )

    if OCCURRENCE_MODEL_KIND == 'structured_lag4':
        occ_loss_fn = ZeroLoss()
        occ_train_loss, occ_train_metrics = evaluate_epoch(
            occurrence_model, occ_train_loader, occ_loss_fn, occurrence_metrics, device, amp_enabled=OCC_USE_AMP
        )
        occ_val_loss, occ_val_metrics = evaluate_epoch(
            occurrence_model, occ_val_loader, occ_loss_fn, occurrence_metrics, device, amp_enabled=OCC_USE_AMP
        )
        occ_state.final_train_metrics = dict(occ_train_metrics)
        occ_state.final_val_metrics = dict(occ_val_metrics)
        occ_state.final_train_loss = float(occ_train_loss)
        occ_state.final_val_loss = float(occ_val_loss)

    else:
        occ_loss_fn = OccurrenceLoss()
        occ_optimizer = torch.optim.AdamW(
            occurrence_model.parameters(),
            lr=OCC_LR,
            weight_decay=OCC_WEIGHT_DECAY,
        )
        occ_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            occ_optimizer,
            mode=OCC_MONITOR_MODE,
            factor=0.5,
            patience=OCC_SCHEDULER_PATIENCE,
        )

        occ_state = fit_model(
            occurrence_model,
            occ_train_loader,
            occ_val_loader,
            occ_optimizer,
            occ_scheduler,
            occ_loss_fn,
            occurrence_metrics,
            device,
            OCC_MAX_EPOCHS,
            OCC_PATIENCE,
            'OccurrenceModel',
            monitor_name=OCC_MONITOR_NAME,
            monitor_mode=OCC_MONITOR_MODE,
            min_delta=OCC_MONITOR_MIN_DELTA,
            amp_enabled=OCC_USE_AMP,
        )

        occ_train_loss, occ_train_metrics = evaluate_epoch(
            occurrence_model, occ_train_loader, occ_loss_fn, occurrence_metrics, device, amp_enabled=OCC_USE_AMP
        )
        occ_val_loss, occ_val_metrics = evaluate_epoch(
            occurrence_model, occ_val_loader, occ_loss_fn, occurrence_metrics, device, amp_enabled=OCC_USE_AMP
        )
        occ_state.best_train_metrics = dict(occ_train_metrics)
        occ_state.best_val_metrics = dict(occ_val_metrics)
        occ_state.final_train_metrics = dict(occ_train_metrics)
        occ_state.final_val_metrics = dict(occ_val_metrics)

    print("[6/6] Construyendo TemporalDataset alineado con inferencia...", flush=True)
    train_count_lookup = build_occurrence_count_lookup(prepared, split.train_target_week_indices, occurrence_model, device)
    val_count_lookup = build_occurrence_count_lookup(prepared, split.val_target_week_indices, occurrence_model, device)
    tmp_train = TemporalDataset(prepared, split.train_target_week_indices, count_lookup=train_count_lookup, count_blend_alpha=TEMPORAL_COUNT_BLEND_TARGET_WEIGHT, show_progress=True, desc='TemporalDataset train')
    tmp_val = TemporalDataset(prepared, split.val_target_week_indices, count_lookup=val_count_lookup, count_blend_alpha=0.0, show_progress=True, desc='TemporalDataset val')

    print('\nResumen del dataset')
    print(f'  Tareas únicas             : {len(prepared.task_names)}')
    print(f'  Semanas totales           : {len(prepared.weeks)}')
    print(f'  Ventanas train            : {len(occ_train)}')
    print(f'  Ventanas val              : {len(occ_val)}')
    print(f'  Muestras temporales train : {len(tmp_train)}')
    print(f'  Muestras temporales val   : {len(tmp_val)}')
    print(f'  max_count_cap             : {prepared.max_count_cap}')
    print(f'  max_occurrences_per_task  : {prepared.max_occurrences_per_task}')
    print(f'  max_tasks_per_week        : {prepared.max_tasks_per_week}')
    print(f'  cap_inference_scope       : {prepared.cap_inference_scope}')
    print(f'  caps train                : occ={prepared.inferred_train_max_occurrences_per_task} | tareas={prepared.inferred_train_max_tasks_per_week}')
    print(f'  caps base completa        : occ={prepared.inferred_full_max_occurrences_per_task} | tareas={prepared.inferred_full_max_tasks_per_week}')
    print(f'  bins temporales           : {num_time_bins()}')
    print(f'  clases día                : {num_day_classes()}')
    print(f'  clases hora del día       : {num_time_of_day_classes()}')
    if getattr(occurrence_model, 'task_delta_bounds', None):
        print('  delta bounds por tarea    :')
        for task_name, bounds in zip(prepared.task_names, occurrence_model.task_delta_bounds):
            print(f'    - {task_name}: [{int(bounds[0])}, {int(bounds[1])}]')

    tmp_train_loader = make_dataloader(tmp_train, batch_size=TMP_BATCH_SIZE, shuffle=True, device=device)
    tmp_val_loader = make_dataloader(tmp_val, batch_size=TMP_BATCH_SIZE, shuffle=False, device=device)

    temporal_model = TemporalAssignmentModel(prepared.week_feature_dim, prepared.history_feature_dim, len(prepared.task_names), prepared.max_occurrences_per_task, TMP_HIDDEN_SIZE, TMP_NUM_LAYERS, TMP_DROPOUT, TASK_EMBED_DIM, OCC_EMBED_DIM, DAY_EMBED_DIM).to(device)
    temporal_loss_fn = TemporalLoss()
    temporal_optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=TMP_LR, weight_decay=TMP_WEIGHT_DECAY)
    temporal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(temporal_optimizer, mode='max', factor=0.5, patience=TMP_SCHEDULER_PATIENCE)
    duration_span = max(prepared.duration_max - prepared.duration_min, 1e-6)

    def temporal_metrics_wrapper(outputs, batch):
        return temporal_metrics(outputs, batch, duration_span)

    e2e_evaluator = TemporalE2EEvaluator(prepared, split.val_target_week_indices, occurrence_model, device)

    tmp_state = fit_model(
        temporal_model,
        tmp_train_loader,
        tmp_val_loader,
        temporal_optimizer,
        temporal_scheduler,
        temporal_loss_fn,
        temporal_metrics_wrapper,
        device,
        TMP_MAX_EPOCHS,
        TMP_PATIENCE,
        'TemporalModel',
        monitor_name='e2e_compound_score',
        monitor_mode='max',
        min_delta=1e-4,
        extra_val_evaluator=e2e_evaluator,
        extra_val_evaluator_every=TMP_E2E_EVAL_EVERY,
        amp_enabled=TMP_USE_AMP,
        amp_dtype=TMP_AMP_DTYPE,
    )

    metadata = serialize_metadata(prepared)
    metadata['occurrence_model_kind'] = OCCURRENCE_MODEL_KIND
    metadata['occurrence_lag_weeks'] = OCC_LAG_WEEKS
    metadata['occurrence_seasonal_lags'] = list(OCC_SEASONAL_LAGS)
    metadata['occurrence_seasonal_lag_weights'] = list(OCC_SEASONAL_LAG_WEIGHTS)
    metadata['occurrence_seasonal_baseline_logit'] = float(OCC_SEASONAL_BASELINE_LOGIT)
    metadata['occurrence_expected_count_mae_weight'] = float(OCC_EXPECTED_COUNT_MAE_WEIGHT)
    metadata['occurrence_use_task_delta_ranges'] = bool(OCC_USE_TASK_DELTA_RANGES)
    metadata['occurrence_delta_range_low_quantile'] = float(OCC_DELTA_RANGE_LOW_QUANTILE)
    metadata['occurrence_delta_range_high_quantile'] = float(OCC_DELTA_RANGE_HIGH_QUANTILE)
    metadata['occurrence_delta_range_margin'] = int(OCC_DELTA_RANGE_MARGIN)
    metadata['occurrence_delta_range_min_radius'] = int(OCC_DELTA_RANGE_MIN_RADIUS)
    metadata['occurrence_delta_outside_range_logit_penalty_per_step'] = float(OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP)
    metadata['occurrence_task_delta_bounds'] = [list(bounds) for bounds in getattr(occurrence_model, 'task_delta_bounds', ())]

    save_checkpoint(CHECKPOINT_DIR / 'occurrence_model.pt', 
                    {'state_dict': occurrence_model.state_dict(), 
                     'metadata': metadata, 
                     'best_epoch': occ_state.best_epoch, 
                     'best_val_loss': occ_state.best_val_loss, 
                     'best_monitor_name': occ_state.monitor_name, 
                     'best_monitor_value': occ_state.best_metric, 
                     'model_hparams': {
                            'model_kind': OCCURRENCE_MODEL_KIND,
                            'input_dim': prepared.week_feature_dim,
                            'num_tasks': len(prepared.task_names),
                            'max_count_cap': prepared.max_count_cap,
                            'lag_weeks': OCC_LAG_WEEKS,
                            'hidden_size': OCC_HIDDEN_SIZE,
                            'num_layers': OCC_NUM_LAYERS,
                            'dropout': OCC_DROPOUT,
                            'seasonal_lags': list(OCC_SEASONAL_LAGS),
                            'seasonal_lag_weights': list(OCC_SEASONAL_LAG_WEIGHTS),
                            'seasonal_baseline_logit': float(OCC_SEASONAL_BASELINE_LOGIT),
                            'expected_count_mae_weight': float(OCC_EXPECTED_COUNT_MAE_WEIGHT),
                            'task_delta_bounds': [list(bounds) for bounds in getattr(occurrence_model, 'task_delta_bounds', ())],
                            'delta_outside_range_logit_penalty_per_step': float(OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP),
                            'use_task_delta_ranges': bool(OCC_USE_TASK_DELTA_RANGES),
                            'delta_range_low_quantile': float(OCC_DELTA_RANGE_LOW_QUANTILE),
                            'delta_range_high_quantile': float(OCC_DELTA_RANGE_HIGH_QUANTILE),
                            'delta_range_margin': int(OCC_DELTA_RANGE_MARGIN),
                            'delta_range_min_radius': int(OCC_DELTA_RANGE_MIN_RADIUS),
                            'max_occurrences_per_task': prepared.max_occurrences_per_task,
                            'max_tasks_per_week': prepared.max_tasks_per_week,
                        }})
    
    save_checkpoint(CHECKPOINT_DIR / 'temporal_model.pt', {'state_dict': temporal_model.state_dict(), 'metadata': metadata, 'best_epoch': tmp_state.best_epoch, 'best_val_loss': tmp_state.best_val_loss, 'best_monitor_name': tmp_state.monitor_name, 'best_monitor_value': tmp_state.best_metric, 'model_hparams': {'sequence_dim': prepared.week_feature_dim, 'history_feature_dim': prepared.history_feature_dim, 'num_tasks': len(prepared.task_names), 'max_occurrences': prepared.max_occurrences_per_task, 'max_occurrences_per_task': prepared.max_occurrences_per_task, 'hidden_size': TMP_HIDDEN_SIZE, 'num_layers': TMP_NUM_LAYERS, 'dropout': TMP_DROPOUT, 'task_embed_dim': TASK_EMBED_DIM, 'occurrence_embed_dim': OCC_EMBED_DIM, 'day_embed_dim': DAY_EMBED_DIM, 'max_tasks_per_week': prepared.max_tasks_per_week}})

    tmp_val_loss, tmp_val_metrics = evaluate_epoch(
        temporal_model,
        tmp_val_loader,
        temporal_loss_fn,
        temporal_metrics_wrapper,
        device,
        amp_enabled=TMP_USE_AMP,
        amp_dtype=TMP_AMP_DTYPE,
    )
    ensemble_val_stats = aggregate_weekly_stats(
        prepared,
        split.val_target_week_indices,
        occurrence_model,
        temporal_model,
        device,
    )
    ensemble_ablation_stats = aggregate_weekly_ablation_stats(
        prepared,
        split.val_target_week_indices,
        occurrence_model,
        temporal_model,
        device,
        include_repair=False,
    )

    report_path = REPORTS_DIR / 'training_report.json'
    report = build_final_report(
        occ_state,
        occ_val_metrics,
        tmp_state,
        tmp_val_metrics,
        ensemble_val_stats,
        ensemble_ablation_stats,
        report_path,
    )
    save_final_report(report, report_path)

    print('\nModelos guardados en:')
    print(f"  - {CHECKPOINT_DIR / 'occurrence_model.pt'}")
    print(f"  - {CHECKPOINT_DIR / 'temporal_model.pt'}")
    print_final_report(report)

    total_training_seconds = time.perf_counter() - train_start_time

    hours = int(total_training_seconds // 3600)
    minutes = int((total_training_seconds % 3600) // 60)
    seconds = total_training_seconds % 60

    print(f"\nTiempo total de entrenamient: {hours:02d}:{minutes:02d}:{seconds:05.2f}")

if __name__ == '__main__':
    main()
