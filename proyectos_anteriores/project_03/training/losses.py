import torch
import torch.nn.functional as F

from proyectos_anteriores.project_03.config import (
    DAY_OFFSET_LOSS_WEIGHT,
    DURATION_LOSS_WEIGHT,
    GLOBAL_DAY_OFFSET_SOFT_SIGMA,
    LOCAL_OFFSET_LOSS_WEIGHT,
    LOCAL_START_OFFSET_SOFT_SIGMA,
)


class OccurrenceLoss:
    def __init__(self, class_weights: torch.Tensor | None = None):
        self.class_weights = class_weights

    def __call__(self, logits: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        num_tasks = logits.shape[1]
        for task_id in range(num_tasks):
            weight = self.class_weights[task_id].to(logits.device) if self.class_weights is not None else None
            total_loss = total_loss + F.cross_entropy(logits[:, task_id, :], target_counts[:, task_id], weight=weight)
        return total_loss / num_tasks


def _gaussian_soft_targets(num_classes: int, target_indices: torch.Tensor, sigma: float) -> torch.Tensor:
    positions = torch.arange(num_classes, device=target_indices.device, dtype=torch.float32)
    target_positions = target_indices.float().unsqueeze(-1)
    probs = torch.exp(-0.5 * ((positions.unsqueeze(0) - target_positions) / max(sigma, 1e-6)) ** 2)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return probs


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


class TemporalLoss:
    def __call__(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        day_targets = _gaussian_soft_targets(outputs['day_offset_logits'].shape[-1], batch['target_day_offset_idx'], GLOBAL_DAY_OFFSET_SOFT_SIGMA)
        local_targets = _gaussian_soft_targets(outputs['local_offset_logits'].shape[-1], batch['target_local_offset_idx'], LOCAL_START_OFFSET_SOFT_SIGMA)
        day_loss = _soft_cross_entropy(outputs['day_offset_logits'], day_targets)
        local_loss = _soft_cross_entropy(outputs['local_offset_logits'], local_targets)
        duration_loss = F.smooth_l1_loss(outputs['pred_duration_norm'], batch['target_duration_norm'])
        return DAY_OFFSET_LOSS_WEIGHT * day_loss + LOCAL_OFFSET_LOSS_WEIGHT * local_loss + DURATION_LOSS_WEIGHT * duration_loss
