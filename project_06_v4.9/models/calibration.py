"""Temperature scaling calibration for model outputs."""
from __future__ import annotations

import torch
import torch.nn as nn


class TemperatureScaling(nn.Module):
    """Calibrate model logits via learned temperature."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def calibrate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit temperature on validation data. Returns final NLL."""
        self.temperature.data.fill_(1.5)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        ce = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = ce(self.forward(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return ce(self.forward(logits), targets).item()
