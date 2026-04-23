from __future__ import annotations

import torch
from torch import nn


class AttentivePool(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        hidden = max(32, dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.net(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))
