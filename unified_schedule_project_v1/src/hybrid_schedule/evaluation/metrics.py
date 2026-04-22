from __future__ import annotations

import torch


@torch.no_grad()
def unified_slot_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_tasks: int,
    max_slots: int,
    bin_minutes: int,
    bins_per_day: int,
    threshold: float = 0.50,
) -> dict[str, float]:
    active_probs = torch.sigmoid(outputs['active_logits'])
    active_pred = (active_probs >= float(threshold)).float()
    active_true = batch['active_targets']

    tp = float(((active_pred == 1.0) & (active_true == 1.0)).sum().item())
    fp = float(((active_pred == 1.0) & (active_true == 0.0)).sum().item())
    fn = float(((active_pred == 0.0) & (active_true == 1.0)).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    pred_day = outputs['day_logits'].argmax(dim=-1)
    pred_time = outputs['time_logits'].argmax(dim=-1)
    pred_start = pred_day.long() * int(bins_per_day) + pred_time.long()
    positive_mask = active_true > 0.5
    if positive_mask.any():
        diff = (pred_start[positive_mask] - batch['target_start'][positive_mask]).abs().float()
        pred_duration = torch.clamp(torch.round(torch.expm1(outputs['pred_log_duration'][positive_mask])), min=1.0)
        dur_diff = (pred_duration - batch['target_duration'][positive_mask]).abs().float()
        start_mae = diff.mean().item() * bin_minutes
        start_tol_5m = (diff <= 1).float().mean().item() * 100.0
        day_acc = (pred_day[positive_mask] == batch['target_day_idx'][positive_mask]).float().mean().item() * 100.0
        dur_mae = dur_diff.mean().item() * bin_minutes
    else:
        start_mae = 0.0
        start_tol_5m = 0.0
        day_acc = 0.0
        dur_mae = 0.0

    pred_counts = active_probs.view(-1, num_tasks, max_slots).sum(dim=-1)
    true_counts = batch['true_counts']
    count_mae = (pred_counts - true_counts).abs().mean().item()

    return {
        'active_precision': precision * 100.0,
        'active_recall': recall * 100.0,
        'active_f1': f1 * 100.0,
        'count_mae': count_mae,
        'start_mae_minutes': start_mae,
        'start_tol_5m': start_tol_5m,
        'day_acc': day_acc,
        'duration_mae_minutes': dur_mae,
    }
