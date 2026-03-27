from __future__ import annotations

import torch

from config import BIN_MINUTES, START_TOPK_TOLERANCE_BINS


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
        "count_exact_acc": exact_acc,
        "count_mae": count_mae,
        "weekly_total_mae": weekly_total_mae,
        "presence_f1": f1 * 100.0,
    }


@torch.no_grad()
def temporal_metrics(start_logits: torch.Tensor, target_start_bin: torch.Tensor, pred_duration: torch.Tensor, target_duration: torch.Tensor, duration_span_minutes: float) -> dict[str, float]:
    pred_start = start_logits.argmax(dim=-1)
    exact_acc = (pred_start == target_start_bin).float().mean().item() * 100.0
    diff = torch.abs(pred_start - target_start_bin)
    num_bins = start_logits.shape[-1]
    diff = torch.minimum(diff, num_bins - diff)
    tol_acc = (diff <= START_TOPK_TOLERANCE_BINS).float().mean().item() * 100.0
    start_mae_minutes = diff.float().mean().item() * BIN_MINUTES
    duration_mae_minutes = torch.abs(pred_duration - target_duration).mean().item() * duration_span_minutes
    return {
        "start_exact_acc": exact_acc,
        "start_tol_acc": tol_acc,
        "start_mae_minutes": start_mae_minutes,
        "duration_mae_minutes": duration_mae_minutes,
    }
