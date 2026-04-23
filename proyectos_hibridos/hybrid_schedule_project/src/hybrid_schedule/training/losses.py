from __future__ import annotations

import torch
import torch.nn.functional as F



def occurrence_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], change_loss_weight: float = 0.25) -> torch.Tensor:
    loss_delta = F.cross_entropy(outputs['delta_logits'], batch['delta_target'])
    loss_change = F.cross_entropy(outputs['change_logits'], batch['change_target'])
    return loss_delta + float(change_loss_weight) * loss_change



def temporal_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    day_loss = F.cross_entropy(outputs['day_logits'], batch['day_target'])
    macro_loss = F.cross_entropy(outputs['macroblock_logits'], batch['macroblock_target'])
    fine_loss = F.cross_entropy(outputs['fine_offset_logits'], batch['fine_offset_target'])
    dur_loss = F.smooth_l1_loss(outputs['duration_delta'], batch['duration_delta'])
    return day_loss + macro_loss + fine_loss + 0.2 * dur_loss
