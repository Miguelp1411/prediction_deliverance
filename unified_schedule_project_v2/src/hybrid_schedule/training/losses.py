from __future__ import annotations

import torch
import torch.nn.functional as F


def unified_slot_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_tasks: int,
    max_slots: int,
    pos_weight_active: float = 5.0,
    active_loss_weight: float = 1.0,
    day_loss_weight: float = 0.70,
    time_loss_weight: float = 1.0,
    duration_loss_weight: float = 0.20,
    count_consistency_weight: float = 0.35,
    day_label_smoothing: float = 0.01,
    time_label_smoothing: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active_targets = batch['active_targets']
    pos_weight = torch.tensor(float(pos_weight_active), device=active_targets.device, dtype=active_targets.dtype)
    active_loss = F.binary_cross_entropy_with_logits(outputs['active_logits'], active_targets, pos_weight=pos_weight)

    positive_mask = active_targets > 0.5
    if positive_mask.any():
        day_loss = F.cross_entropy(
            outputs['day_logits'][positive_mask],
            batch['target_day_idx'][positive_mask],
            label_smoothing=float(day_label_smoothing),
        )
        time_loss = F.cross_entropy(
            outputs['time_logits'][positive_mask],
            batch['target_time_bin_idx'][positive_mask],
            label_smoothing=float(time_label_smoothing),
        )
        duration_loss = F.smooth_l1_loss(
            outputs['pred_log_duration'][positive_mask],
            batch['target_log_duration'][positive_mask],
        )
    else:
        zero = active_loss.new_zeros(())
        day_loss = zero
        time_loss = zero
        duration_loss = zero

    active_probs = torch.sigmoid(outputs['active_logits']).view(-1, num_tasks, max_slots)
    pred_counts = active_probs.sum(dim=-1)
    true_counts = batch['true_counts']
    count_loss = F.l1_loss(pred_counts, true_counts)

    total = (
        float(active_loss_weight) * active_loss
        + float(day_loss_weight) * day_loss
        + float(time_loss_weight) * time_loss
        + float(duration_loss_weight) * duration_loss
        + float(count_consistency_weight) * count_loss
    )
    components = {
        'loss_active': active_loss.detach(),
        'loss_day': day_loss.detach(),
        'loss_time': time_loss.detach(),
        'loss_duration': duration_loss.detach(),
        'loss_count': count_loss.detach(),
    }
    return total, components
