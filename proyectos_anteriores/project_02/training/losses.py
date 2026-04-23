import torch
import torch.nn.functional as F

from proyectos_anteriores.project_02.config import DURATION_LOSS_WEIGHT


class OccurrenceLoss:
    def __init__(self, class_weights: torch.Tensor | None = None):
        self.class_weights = class_weights

    def __call__(self, logits: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        num_tasks = logits.shape[1]
        for task_id in range(num_tasks):
            weight = None
            if self.class_weights is not None:
                weight = self.class_weights[task_id].to(logits.device)
            total_loss = total_loss + F.cross_entropy(logits[:, task_id, :], target_counts[:, task_id], weight=weight)
        return total_loss / num_tasks


class TemporalLoss:
    def __call__(self, start_logits: torch.Tensor, target_start_bin: torch.Tensor, pred_duration: torch.Tensor, target_duration: torch.Tensor) -> torch.Tensor:
        start_loss = F.cross_entropy(start_logits, target_start_bin)
        duration_loss = F.smooth_l1_loss(pred_duration, target_duration)
        return start_loss + DURATION_LOSS_WEIGHT * duration_loss
