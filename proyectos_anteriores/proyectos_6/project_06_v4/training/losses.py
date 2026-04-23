import torch
import torch.nn.functional as F

from proyectos_anteriores.proyectos_6.project_06_v4.config import (
    DAY_CLASS_LOSS_WEIGHT,
    DAY_LABEL_SMOOTHING,
    DURATION_LOSS_WEIGHT,
    TIME_CLASS_LOSS_WEIGHT,
    TIME_LABEL_SMOOTHING,
)


class OccurrenceLoss:
    def __init__(self, class_weights: torch.Tensor | None = None):
        self.class_weights = class_weights

    def __call__(self, logits: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
        B, T, C = logits.shape
        flat_logits = logits.reshape(B * T, C)
        flat_targets = target_counts.reshape(B * T)
        return F.cross_entropy(flat_logits, flat_targets)


class TemporalLoss:
    def __call__(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        day_loss = F.cross_entropy(
            outputs['day_offset_logits'],
            batch['target_day_offset_idx'],
            label_smoothing=DAY_LABEL_SMOOTHING,
        )
        time_loss = F.cross_entropy(
            outputs['local_offset_logits'],
            batch['target_local_offset_idx'],
            label_smoothing=TIME_LABEL_SMOOTHING,
        )
        duration_loss = F.smooth_l1_loss(outputs['pred_duration_norm'], batch['target_duration_norm'])
        return DAY_CLASS_LOSS_WEIGHT * day_loss + TIME_CLASS_LOSS_WEIGHT * time_loss + DURATION_LOSS_WEIGHT * duration_loss
