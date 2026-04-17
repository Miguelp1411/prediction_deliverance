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
    expected_cost_weight: float = 0.30,
) -> torch.Tensor:
    logits = outputs['candidate_logits']
    mask = batch['candidate_mask'].bool()
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = masked_logits.softmax(dim=-1)
    target_probs = batch['candidate_target_probs']
    listwise_loss = -(target_probs * log_probs).sum(dim=-1).mean()
    expected_cost = (probs * batch['candidate_costs']).sum(dim=-1).mean()
    return listwise_loss + float(expected_cost_weight) * expected_cost
