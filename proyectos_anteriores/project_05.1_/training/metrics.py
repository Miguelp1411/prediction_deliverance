from __future__ import annotations

import torch

from config import BIN_MINUTES, START_TOPK_TOLERANCE_BINS, bins_per_day, num_time_bins


@torch.no_grad()
def occurrence_metrics(logits: torch.Tensor, target_counts: torch.Tensor) -> dict[str, float]:
    pred_counts = logits.argmax(dim=-1)
    exact_acc = (pred_counts == target_counts).float().mean().item() * 100.0
    count_mae = torch.abs(pred_counts.float() - target_counts.float()).mean().item()
    weekly_total_mae = torch.abs(pred_counts.sum(dim=-1).float() - target_counts.sum(dim=-1).float()).mean().item()

    pred_presence = pred_counts > 0
    true_presence = target_counts > 0
    tp = (pred_presence & true_presence).sum().item()
    fp = (pred_presence & ~true_presence).sum().item()
    fn = (~pred_presence & true_presence).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        'count_exact_acc': exact_acc,
        'count_mae': count_mae,
        'weekly_total_mae': weekly_total_mae,
        'presence_f1': f1 * 100.0,
    }


@torch.no_grad()
def decode_predicted_start_bins(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs['start_logits'].argmax(dim=-1).long().clamp(min=0, max=num_time_bins() - 1)


@torch.no_grad()
def temporal_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], duration_span_minutes: float) -> dict[str, float]:
    pred_start = decode_predicted_start_bins(outputs)
    target_start = batch['target_start_bin']
    diff = torch.abs(pred_start - target_start)

    pred_day = pred_start // bins_per_day()
    pred_time = pred_start % bins_per_day()
    target_day = target_start // bins_per_day()
    target_time = target_start % bins_per_day()

    return {
        'start_exact_acc': (pred_start == target_start).float().mean().item() * 100.0,
        'day_acc': (pred_day == target_day).float().mean().item() * 100.0,
        'time_exact_acc': (pred_time == target_time).float().mean().item() * 100.0,
        'start_tol_acc': (diff <= START_TOPK_TOLERANCE_BINS).float().mean().item() * 100.0,
        'start_tol_acc_5m': (diff <= 1).float().mean().item() * 100.0,
        'start_tol_acc_10m': (diff <= 2).float().mean().item() * 100.0,
        'start_mae_minutes': diff.float().mean().item() * BIN_MINUTES,
        'duration_mae_minutes': torch.abs(outputs['pred_duration_norm'] - batch['target_duration_norm']).mean().item() * duration_span_minutes,
    }
