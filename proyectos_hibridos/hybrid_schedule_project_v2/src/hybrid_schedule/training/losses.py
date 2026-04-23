from __future__ import annotations

import torch
import torch.nn.functional as F


def occurrence_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    change_loss_weight: float = 0.25,
    expected_count_mae_weight: float = 0.15,
) -> torch.Tensor:
    loss_delta = F.cross_entropy(outputs['delta_logits'], batch['delta_target'])
    loss_change = F.cross_entropy(outputs['change_logits'], batch['change_target'])
    target_count = batch['target_count'].to(outputs['expected_count'].dtype)
    loss_expected = F.l1_loss(outputs['expected_count'], target_count)
    return loss_delta + float(change_loss_weight) * loss_change + float(expected_count_mae_weight) * loss_expected


def temporal_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    day_label_smoothing: float = 0.05,
    local_label_smoothing: float = 0.03,
    duration_loss_weight: float = 0.2,
) -> torch.Tensor:
    day_loss = F.cross_entropy(outputs['day_offset_logits'], batch['day_offset_target'], label_smoothing=float(day_label_smoothing))
    local_loss = F.cross_entropy(outputs['local_offset_logits'], batch['local_offset_target'], label_smoothing=float(local_label_smoothing))
    dur_loss = F.smooth_l1_loss(outputs['duration_delta'], batch['duration_delta'])
    return day_loss + local_loss + float(duration_loss_weight) * dur_loss
