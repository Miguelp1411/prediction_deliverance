import torch
import torch.nn.functional as F

from config import (
    DAY_CLASS_LOSS_WEIGHT,
    DAY_LABEL_SMOOTHING,
    DURATION_LOSS_WEIGHT,
    OCC_EXPECTED_COUNT_MAE_WEIGHT,
    TIME_CLASS_LOSS_WEIGHT,
    TIME_LABEL_SMOOTHING,
)


class OccurrenceLoss:
    def __init__(self, class_weights: torch.Tensor | None = None, expected_count_mae_weight: float = OCC_EXPECTED_COUNT_MAE_WEIGHT):
        self.class_weights = class_weights
        self.expected_count_mae_weight = float(expected_count_mae_weight)

    def __call__(self, logits: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
        if self.class_weights is None:
            ce_loss = F.cross_entropy(logits.permute(0, 2, 1), target_counts)
        else:
            total_ce_loss = 0.0
            num_tasks = logits.shape[1]
            for task_id in range(num_tasks):
                weight = self.class_weights[task_id].to(logits.device)
                total_ce_loss = total_ce_loss + F.cross_entropy(logits[:, task_id, :], target_counts[:, task_id], weight=weight)
            ce_loss = total_ce_loss / logits.shape[1]

        if self.expected_count_mae_weight <= 0.0:
            return ce_loss

        probs = torch.softmax(logits, dim=-1)
        count_values = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
        expected_counts = (probs * count_values.view(1, 1, -1)).sum(dim=-1)
        distance_loss = torch.abs(expected_counts - target_counts.to(dtype=logits.dtype)).mean()
        return ce_loss + self.expected_count_mae_weight * distance_loss


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
