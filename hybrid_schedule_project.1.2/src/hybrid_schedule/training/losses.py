from __future__ import annotations

import torch
import torch.nn.functional as F



def _vector_param(batch: dict[str, torch.Tensor], key: str, ref: torch.Tensor, default: float) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        return torch.full((ref.shape[0],), float(default), device=ref.device, dtype=ref.dtype)
    if value.ndim == 0:
        return torch.full((ref.shape[0],), float(value.item()), device=ref.device, dtype=ref.dtype)
    return value.to(device=ref.device, dtype=ref.dtype).view(-1)



def _apply_label_smoothing(target_dist: torch.Tensor, label_smoothing: torch.Tensor) -> torch.Tensor:
    num_classes = target_dist.shape[-1]
    uniform = torch.full_like(target_dist, 1.0 / max(num_classes, 1))
    return (1.0 - label_smoothing.unsqueeze(-1)) * target_dist + label_smoothing.unsqueeze(-1) * uniform



def _gaussian_target_distribution(target: torch.Tensor, num_classes: int, sigma: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(num_classes, device=target.device, dtype=torch.float32).unsqueeze(0)
    center = target.to(dtype=torch.float32).unsqueeze(-1)
    sigma = sigma.to(dtype=torch.float32).clamp_min(1e-3).unsqueeze(-1)
    weights = torch.exp(-0.5 * ((positions - center) / sigma) ** 2)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights



def _soft_cross_entropy(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1)



def _confidence_penalty(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return (probs * log_probs).sum(dim=-1)



def occurrence_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], change_loss_weight: float = 0.25) -> torch.Tensor:
    delta_logits = outputs['delta_logits']
    change_logits = outputs['change_logits']

    delta_sigma = _vector_param(batch, 'delta_target_sigma', delta_logits, default=0.75)
    label_smoothing = _vector_param(batch, 'label_smoothing', delta_logits, default=0.0).clamp(0.0, 0.30)
    shrink_weight = _vector_param(batch, 'count_shrink_weight', delta_logits, default=0.0).clamp_min(0.0)
    confidence_penalty = _vector_param(batch, 'confidence_penalty', delta_logits, default=0.0).clamp_min(0.0)

    delta_target_dist = _gaussian_target_distribution(batch['delta_target'], delta_logits.shape[-1], delta_sigma)
    delta_target_dist = _apply_label_smoothing(delta_target_dist, label_smoothing)
    delta_loss = _soft_cross_entropy(delta_logits, delta_target_dist)

    change_target = F.one_hot(batch['change_target'], num_classes=2).to(dtype=change_logits.dtype)
    change_target = _apply_label_smoothing(change_target, (0.50 * label_smoothing).clamp(0.0, 0.15))
    change_loss = _soft_cross_entropy(change_logits, change_target)

    expected_delta = outputs['expected_delta'].abs()
    entropy_term = 0.5 * (_confidence_penalty(delta_logits) + _confidence_penalty(change_logits))

    total = delta_loss + float(change_loss_weight) * change_loss + shrink_weight * expected_delta + confidence_penalty * entropy_term
    return total.mean()



def temporal_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    day_logits = outputs['day_logits']
    time_logits = outputs['time_logits']
    bins_per_day = int(time_logits.shape[-1])

    day_label_smoothing = _vector_param(batch, 'day_label_smoothing', day_logits, default=0.0).clamp(0.0, 0.30)
    time_target_sigma = _vector_param(batch, 'time_target_sigma', time_logits, default=1.0).clamp(0.25, 8.0)
    anchor_weight = _vector_param(batch, 'anchor_weight', time_logits, default=0.0).clamp_min(0.0)
    day_prior_weight = _vector_param(batch, 'day_prior_weight', day_logits, default=0.0).clamp_min(0.0)
    time_prior_weight = _vector_param(batch, 'time_prior_weight', time_logits, default=0.0).clamp_min(0.0)
    confidence_penalty = _vector_param(batch, 'confidence_penalty', time_logits, default=0.0).clamp_min(0.0)
    time_smoothness_weight = _vector_param(batch, 'time_smoothness_weight', time_logits, default=0.0).clamp_min(0.0)

    day_target = F.one_hot(batch['day_target'], num_classes=7).to(dtype=day_logits.dtype)
    day_target = _apply_label_smoothing(day_target, day_label_smoothing)
    loss_day = _soft_cross_entropy(day_logits, day_target)

    time_target_dist = _gaussian_target_distribution(batch['time_target'], bins_per_day, time_target_sigma)
    time_target_dist = _apply_label_smoothing(time_target_dist, (0.50 * day_label_smoothing).clamp(0.0, 0.15))
    loss_time = _soft_cross_entropy(time_logits, time_target_dist)

    day_log_probs = F.log_softmax(day_logits, dim=-1)
    time_log_probs = F.log_softmax(time_logits, dim=-1)
    day_prior = batch.get('day_prior')
    time_prior = batch.get('time_prior')
    if day_prior is None:
        day_kl = torch.zeros_like(loss_day)
    else:
        day_kl = F.kl_div(day_log_probs, day_prior.to(day_logits.dtype), reduction='none').sum(dim=-1)
    if time_prior is None:
        time_kl = torch.zeros_like(loss_time)
    else:
        time_kl = F.kl_div(time_log_probs, time_prior.to(time_logits.dtype), reduction='none').sum(dim=-1)

    day_probs = day_log_probs.exp()
    time_probs = time_log_probs.exp()
    day_positions = torch.arange(7, device=day_logits.device, dtype=day_logits.dtype).unsqueeze(0)
    time_positions = torch.arange(bins_per_day, device=time_logits.device, dtype=time_logits.dtype).unsqueeze(0)
    expected_day = (day_probs * day_positions).sum(dim=-1)
    expected_time = (time_probs * time_positions).sum(dim=-1)
    expected_start = expected_day * bins_per_day + expected_time
    anchor_distance = (expected_start - batch['anchor_start'].to(expected_start.dtype)).abs() / max(bins_per_day, 1)

    time_diff = time_logits[:, 1:] - time_logits[:, :-1]
    smoothness = time_diff.pow(2).mean(dim=-1)
    entropy_term = 0.5 * (_confidence_penalty(day_logits) + _confidence_penalty(time_logits))

    total = (
        loss_day
        + loss_time
        + day_prior_weight * day_kl
        + time_prior_weight * time_kl
        + anchor_weight * anchor_distance
        + time_smoothness_weight * smoothness
        + confidence_penalty * entropy_term
    )
    return total.mean()
