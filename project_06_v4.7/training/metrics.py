from __future__ import annotations

import torch

from config import (
    BIN_MINUTES,
    OCC_MONITOR_COUNT_MAE_WEIGHT,
    OCC_MONITOR_EXACT_ACC_WEIGHT,
    OCC_MONITOR_WEEKLY_TOTAL_MAE_WEIGHT,
    START_TOPK_TOLERANCE_BINS,
    bins_per_day,
)
from config import num_time_bins


@torch.no_grad()
def occurrence_metrics(logits: torch.Tensor, target_counts: torch.Tensor) -> dict[str, float]:
    pred_counts = logits.argmax(dim=-1)

    abs_err = torch.abs(pred_counts.float() - target_counts.float())

    exact_acc = (abs_err == 0).float().mean().item() * 100.0
    close_acc_1 = (abs_err <= 1).float().mean().item() * 100.0
    close_acc_2 = (abs_err <= 2).float().mean().item() * 100.0

    count_mae = abs_err.mean().item()
    weekly_total_mae = torch.abs(
        pred_counts.sum(dim=-1).float() - target_counts.sum(dim=-1).float()
    ).mean().item()

    pred_presence = pred_counts > 0
    true_presence = target_counts > 0
    tp = (pred_presence & true_presence).sum().item()
    fp = (pred_presence & ~true_presence).sum().item()
    fn = (~pred_presence & true_presence).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Métrica antigua
    occurrence_selection_score = (
        OCC_MONITOR_EXACT_ACC_WEIGHT * exact_acc
        - OCC_MONITOR_WEEKLY_TOTAL_MAE_WEIGHT * weekly_total_mae
        - OCC_MONITOR_COUNT_MAE_WEIGHT * count_mae
    )

    # Métrica nueva más realista:
    # premia cercanía y castiga error medio y desajuste total semanal
    occurrence_realistic_score = (
        0.20 * exact_acc
        + 0.50 * close_acc_1
        + 0.30 * close_acc_2
        - 1.50 * count_mae
        - 2.00 * weekly_total_mae
    )

    return {
        'count_exact_acc': exact_acc,
        'close_acc_1': close_acc_1,
        'close_acc_2': close_acc_2,
        'count_mae': count_mae,
        'weekly_total_mae': weekly_total_mae,
        'presence_precision': precision * 100.0,
        'presence_recall': recall * 100.0,
        'presence_f1': f1 * 100.0,
        'occurrence_selection_score': occurrence_selection_score,
        'occurrence_realistic_score': occurrence_realistic_score,
    }


@torch.no_grad()
def decode_predicted_start_bins(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pred_day = outputs['day_logits'].argmax(dim=-1).long().clamp(min=0, max=6)
    pred_time = outputs['time_of_day_logits'].argmax(dim=-1).long().clamp(min=0, max=bins_per_day() - 1)
    pred_start = pred_day * bins_per_day() + pred_time
    return pred_start.clamp(min=0, max=num_time_bins() - 1)


@torch.no_grad()
def temporal_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], duration_span: float) -> dict[str, float]:
    pred_start = decode_predicted_start_bins(outputs, batch)
    target_start = batch['target_start_bin'].long()
    abs_bin_error = torch.abs(pred_start - target_start)

    pred_duration = outputs['pred_duration_norm'].clamp(0.0, 1.0) * duration_span
    target_duration = batch['target_duration_norm'].clamp(0.0, 1.0) * duration_span
    duration_abs_error = torch.abs(pred_duration - target_duration)

    start_exact_acc = (abs_bin_error == 0).float().mean().item() * 100.0
    # Métricas acumuladas: ≤5m incluye exactos y ≤10m incluye exactos + ≤5m.
    start_tol_acc_5m = (abs_bin_error <= START_TOPK_TOLERANCE_BINS).float().mean().item() * 100.0
    start_tol_acc_10m = (abs_bin_error <= 2).float().mean().item() * 100.0
    start_bucket_0_5m_only = ((abs_bin_error > 0) & (abs_bin_error <= START_TOPK_TOLERANCE_BINS)).float().mean().item() * 100.0
    start_bucket_5_10m_only = ((abs_bin_error > START_TOPK_TOLERANCE_BINS) & (abs_bin_error <= 2)).float().mean().item() * 100.0
    start_mae_minutes = abs_bin_error.float().mean().item() * BIN_MINUTES
    duration_mae_minutes = duration_abs_error.float().mean().item()
    day_exact_acc = (outputs['day_logits'].argmax(dim=-1) == batch['target_day_idx']).float().mean().item() * 100.0
    time_of_day_exact_acc = (outputs['time_of_day_logits'].argmax(dim=-1) == batch['target_time_bin_idx']).float().mean().item() * 100.0

    return {
        'start_exact_acc': start_exact_acc,
        'start_tol_acc_5m': start_tol_acc_5m,
        'start_tol_acc_10m': start_tol_acc_10m,
        'start_acc_upto_5m': start_tol_acc_5m,
        'start_acc_upto_10m': start_tol_acc_10m,
        'start_bucket_0_5m_only': start_bucket_0_5m_only,
        'start_bucket_5_10m_only': start_bucket_5_10m_only,
        'start_mae_minutes': start_mae_minutes,
        'duration_mae_minutes': duration_mae_minutes,
        'day_exact_acc': day_exact_acc,
        'time_of_day_exact_acc': time_of_day_exact_acc,
    }
