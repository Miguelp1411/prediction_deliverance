from __future__ import annotations

import torch
import torch.nn.functional as F



def occurrence_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], change_loss_weight: float = 0.25) -> torch.Tensor:
    loss_delta = F.cross_entropy(outputs['delta_logits'], batch['delta_target'])
    loss_change = F.cross_entropy(outputs['change_logits'], batch['change_target'])
    return loss_delta + float(change_loss_weight) * loss_change



def temporal_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    day_loss = F.cross_entropy(outputs['day_logits'], batch['day_target'])
    time_loss = F.cross_entropy(outputs['time_logits'], batch['time_target'])
    dur_loss = F.smooth_l1_loss(outputs['duration_delta'], batch['duration_delta'])
    return day_loss + time_loss + 0.2 * dur_loss
