import torch
import torch.nn.functional as F

from config import (
    DAY_CLASS_LOSS_WEIGHT,
    DAY_LABEL_SMOOTHING,
    DURATION_LOSS_WEIGHT,
    OCC_EXPECTED_COUNT_MAE_WEIGHT,
    OCC_V48_DELTA_LOSS_WEIGHT,
    OCC_V48_EXACT_TARGET_WEIGHT,
    OCC_V48_RADIUS1_TOTAL_WEIGHT,
    OCC_V48_RADIUS2_TOTAL_WEIGHT,
    OCC_V48_SELECTOR_LOSS_WEIGHT,
    TIME_CLASS_LOSS_WEIGHT,
    TIME_LABEL_SMOOTHING,
)


def _occurrence_logits(outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(outputs, dict):
        return outputs['logits']
    return outputs


def _soft_target_distribution(target_counts: torch.Tensor, num_classes: int, dtype: torch.dtype) -> torch.Tensor:
    target = torch.zeros(*target_counts.shape, num_classes, dtype=dtype, device=target_counts.device)
    target.scatter_add_(2, target_counts.unsqueeze(-1), torch.full((*target_counts.shape, 1), OCC_V48_EXACT_TARGET_WEIGHT, dtype=dtype, device=target_counts.device))

    radius1_each = float(OCC_V48_RADIUS1_TOTAL_WEIGHT) / 2.0
    if radius1_each > 0.0:
        lower = (target_counts - 1).clamp(min=0)
        upper = (target_counts + 1).clamp(max=num_classes - 1)
        target.scatter_add_(2, lower.unsqueeze(-1), torch.full((*target_counts.shape, 1), radius1_each, dtype=dtype, device=target_counts.device))
        target.scatter_add_(2, upper.unsqueeze(-1), torch.full((*target_counts.shape, 1), radius1_each, dtype=dtype, device=target_counts.device))

    radius2_each = float(OCC_V48_RADIUS2_TOTAL_WEIGHT) / 2.0
    if radius2_each > 0.0:
        lower2 = (target_counts - 2).clamp(min=0)
        upper2 = (target_counts + 2).clamp(max=num_classes - 1)
        target.scatter_add_(2, lower2.unsqueeze(-1), torch.full((*target_counts.shape, 1), radius2_each, dtype=dtype, device=target_counts.device))
        target.scatter_add_(2, upper2.unsqueeze(-1), torch.full((*target_counts.shape, 1), radius2_each, dtype=dtype, device=target_counts.device))

    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return target


class OccurrenceLoss:
    def __init__(self, class_weights: torch.Tensor | None = None, expected_count_mae_weight: float = OCC_EXPECTED_COUNT_MAE_WEIGHT):
        self.class_weights = class_weights
        self.expected_count_mae_weight = float(expected_count_mae_weight)

    def __call__(self, outputs: torch.Tensor | dict[str, torch.Tensor], target_counts: torch.Tensor) -> torch.Tensor:
        logits = _occurrence_logits(outputs)
        log_probs = torch.log_softmax(logits, dim=-1)
        soft_targets = _soft_target_distribution(target_counts, logits.shape[-1], logits.dtype)
        ce_loss = -(soft_targets * log_probs).sum(dim=-1).mean()

        probs = torch.softmax(logits, dim=-1)
        count_values = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
        expected_counts = (probs * count_values.view(1, 1, -1)).sum(dim=-1)
        distance_loss = torch.abs(expected_counts - target_counts.to(dtype=logits.dtype)).mean()
        total_loss = ce_loss + self.expected_count_mae_weight * distance_loss

        if isinstance(outputs, dict):
            if 'candidate_counts' in outputs and 'candidate_masks' in outputs and 'selector_logits' in outputs:
                candidate_counts = outputs['candidate_counts'].long()
                candidate_masks = outputs['candidate_masks'] > 0
                selector_logits = outputs['selector_logits']
                target_expanded = target_counts.unsqueeze(-1)
                distances = torch.abs(candidate_counts - target_expanded)
                distances = torch.where(candidate_masks, distances, torch.full_like(distances, 10_000))
                oracle_idx = distances.argmin(dim=-1)
                selector_loss = F.cross_entropy(
                    selector_logits.reshape(-1, selector_logits.shape[-1]),
                    oracle_idx.reshape(-1),
                )
                total_loss = total_loss + float(OCC_V48_SELECTOR_LOSS_WEIGHT) * selector_loss

                if 'delta_logits' in outputs:
                    best_candidate = candidate_counts.gather(-1, oracle_idx.unsqueeze(-1)).squeeze(-1)
                    delta_target = (target_counts - best_candidate).clamp(
                        min=-(outputs['delta_logits'].shape[-1] // 2),
                        max=(outputs['delta_logits'].shape[-1] // 2),
                    )
                    delta_index = (delta_target + outputs['delta_logits'].shape[-1] // 2).long()
                    delta_loss = F.cross_entropy(
                        outputs['delta_logits'].reshape(-1, outputs['delta_logits'].shape[-1]),
                        delta_index.reshape(-1),
                    )
                    total_loss = total_loss + float(OCC_V48_DELTA_LOSS_WEIGHT) * delta_loss

        return total_loss


class TemporalLoss:
    def __call__(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        day_loss = F.cross_entropy(
            outputs['day_logits'],
            batch['target_day_idx'],
            label_smoothing=DAY_LABEL_SMOOTHING,
        )
        time_loss = F.cross_entropy(
            outputs['time_of_day_logits'],
            batch['target_time_bin_idx'],
            label_smoothing=TIME_LABEL_SMOOTHING,
        )
        duration_loss = F.smooth_l1_loss(outputs['pred_duration_norm'], batch['target_duration_norm'])
        return DAY_CLASS_LOSS_WEIGHT * day_loss + TIME_CLASS_LOSS_WEIGHT * time_loss + DURATION_LOSS_WEIGHT * duration_loss
