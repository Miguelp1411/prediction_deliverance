"""
Training losses for the residual models.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccurrenceResidualLoss(nn.Module):
    """
    Combined loss for the residual occurrence model.

    Two components:
      1. Binary cross-entropy for changed/unchanged head
      2. Cross-entropy for delta class head
    """

    def __init__(
        self,
        change_weight: float = 0.30,
        delta_weight: float = 0.70,
        delta_range: int = 6,
    ) -> None:
        super().__init__()
        self.change_weight = change_weight
        self.delta_weight = delta_weight
        self.delta_range = delta_range

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Target: actual_counts - template_counts
        actual_counts = batch["target_counts"]  # (B, num_tasks)
        template_counts = batch["template_counts"]  # (B, num_tasks)
        delta = actual_counts - template_counts

        # Changed/unchanged target
        changed = (delta != 0).long()  # (B, num_tasks)
        change_loss = F.cross_entropy(
            outputs["change_logits"].reshape(-1, 2),
            changed.reshape(-1),
        )

        # Delta class target: clamp to [-delta_range, +delta_range]
        clamped_delta = delta.clamp(-self.delta_range, self.delta_range)
        delta_target = (clamped_delta + self.delta_range).long()  # shift to [0, 2*K]
        num_classes = outputs["delta_logits"].shape[-1]
        delta_target = delta_target.clamp(0, num_classes - 1)

        delta_loss = F.cross_entropy(
            outputs["delta_logits"].reshape(-1, num_classes),
            delta_target.reshape(-1),
        )

        return self.change_weight * change_loss + self.delta_weight * delta_loss


class TemporalResidualLoss(nn.Module):
    """
    Combined loss for the temporal residual model.

    Components: day CE + time-of-day CE + duration smooth L1 + confidence.
    """

    def __init__(
        self,
        day_weight: float = 0.80,
        time_weight: float = 1.00,
        duration_weight: float = 0.10,
        day_label_smoothing: float = 0.02,
        time_label_smoothing: float = 0.01,
    ) -> None:
        super().__init__()
        self.day_weight = day_weight
        self.time_weight = time_weight
        self.duration_weight = duration_weight
        self.day_label_smoothing = day_label_smoothing
        self.time_label_smoothing = time_label_smoothing

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        day_loss = F.cross_entropy(
            outputs["day_logits"],
            batch["target_day_idx"],
            label_smoothing=self.day_label_smoothing,
        )
        time_loss = F.cross_entropy(
            outputs["time_of_day_logits"],
            batch["target_time_bin_idx"],
            label_smoothing=self.time_label_smoothing,
        )
        duration_loss = F.smooth_l1_loss(
            outputs["pred_duration_norm"],
            batch["target_duration_norm"],
        )
        return (
            self.day_weight * day_loss
            + self.time_weight * time_loss
            + self.duration_weight * duration_loss
        )
