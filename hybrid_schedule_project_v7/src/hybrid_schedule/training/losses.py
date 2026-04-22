from __future__ import annotations

import torch
import torch.nn.functional as F


def occurrence_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    change_loss_weight: float = 0.25,
    expected_count_mae_weight: float = 0.15,
    delta_reg_weight: float = 0.05,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    loss_delta = F.cross_entropy(
        outputs['delta_logits'],
        batch['delta_target'],
        label_smoothing=float(label_smoothing),
    )
    loss_change = F.cross_entropy(outputs['change_logits'], batch['change_target'])
    target_count = batch['target_count'].to(outputs['expected_count'].dtype)
    baseline_count = batch['baseline_count'].to(outputs['expected_count'].dtype)
    loss_expected = F.l1_loss(outputs['expected_count'], target_count)
    loss_residual_shrink = (outputs['expected_count'] - baseline_count).abs().mean()
    return (
        loss_delta
        + float(change_loss_weight) * loss_change
        + float(expected_count_mae_weight) * loss_expected
        + float(delta_reg_weight) * loss_residual_shrink
    )


def temporal_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    expected_cost_weight: float = 0.30,
    label_smoothing: float = 0.0,
    confidence_penalty_weight: float = 0.02,
    anchor_deviation_weight: float = 0.03,
    duration_deviation_weight: float = 0.01,
    bin_minutes: int = 5,
) -> torch.Tensor:
    logits = outputs['candidate_logits']
    mask = batch['candidate_mask'].bool()
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = log_probs.exp()
    target_probs = batch['candidate_target_probs']
    if float(label_smoothing) > 0.0:
        valid_counts = mask.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        uniform = mask.float() / valid_counts
        target_probs = (1.0 - float(label_smoothing)) * target_probs + float(label_smoothing) * uniform
    listwise_loss = -(target_probs * log_probs).sum(dim=-1).mean()
    expected_cost = (probs * batch['candidate_costs']).sum(dim=-1).mean()

    confidence_penalty = (probs * log_probs).sum(dim=-1).mean()

    bins_per_week = max(1.0, float(7 * 24 * 60) / float(max(1, bin_minutes)))
    duration_scale = max(1.0, float(60) / float(max(1, bin_minutes)))
    anchor_start = batch['anchor_start'].unsqueeze(-1).to(probs.dtype)
    anchor_duration = batch['anchor_duration'].unsqueeze(-1).to(probs.dtype)
    start_shift = (batch['candidate_starts'].to(probs.dtype) - anchor_start).abs() / bins_per_week
    duration_shift = (batch['candidate_durations'].to(probs.dtype) - anchor_duration).abs() / duration_scale
    start_shift = start_shift.masked_fill(~mask, 0.0)
    duration_shift = duration_shift.masked_fill(~mask, 0.0)
    expected_anchor_shift = (probs * start_shift).sum(dim=-1).mean()
    expected_duration_shift = (probs * duration_shift).sum(dim=-1).mean()

    return (
        listwise_loss
        + float(expected_cost_weight) * expected_cost
        + float(confidence_penalty_weight) * confidence_penalty
        + float(anchor_deviation_weight) * expected_anchor_shift
        + float(duration_deviation_weight) * expected_duration_shift
    )



def temporal_direct_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    day_loss_weight: float = 0.80,
    time_loss_weight: float = 1.00,
    duration_loss_weight: float = 0.20,
    day_label_smoothing: float = 0.02,
    time_label_smoothing: float = 0.01,
) -> torch.Tensor:
    day_loss = F.cross_entropy(
        outputs['day_logits'],
        batch['target_day_idx'],
        label_smoothing=float(day_label_smoothing),
    )
    time_loss = F.cross_entropy(
        outputs['time_logits'],
        batch['target_time_bin_idx'],
        label_smoothing=float(time_label_smoothing),
    )
    duration_loss = F.smooth_l1_loss(outputs['pred_log_duration'], batch['target_log_duration'])
    return (
        float(day_loss_weight) * day_loss
        + float(time_loss_weight) * time_loss
        + float(duration_loss_weight) * duration_loss
    )
