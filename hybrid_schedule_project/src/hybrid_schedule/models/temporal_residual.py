from __future__ import annotations

import torch
from torch import nn

from .blocks import MLP, SequenceEncoder


class TemporalResidualNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        num_databases: int,
        num_robots: int,
        hidden_size: int = 160,
        num_layers: int = 2,
        dropout: float = 0.20,
        task_embed_dim: int = 24,
        database_embed_dim: int = 12,
        robot_embed_dim: int = 12,
        num_days: int = 7,
        num_macroblocks: int = 24,
        fine_offset_bins: int = 12,
    ):
        super().__init__()
        self.num_days = int(num_days)
        self.num_macroblocks = int(num_macroblocks)
        self.fine_offset_bins = int(fine_offset_bins)
        self.encoder = SequenceEncoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(num_databases, database_embed_dim)
        self.robot_emb = nn.Embedding(num_robots, robot_embed_dim)
        static_dim = 8
        fused_dim = hidden_size * 2 + task_embed_dim + database_embed_dim + robot_embed_dim + static_dim
        self.day_head = MLP(fused_dim, hidden_size, self.num_days, dropout)
        self.macroblock_head = MLP(fused_dim, hidden_size, self.num_macroblocks, dropout)
        self.fine_offset_head = MLP(fused_dim, hidden_size, self.fine_offset_bins, dropout)
        self.duration_head = MLP(fused_dim, hidden_size, 1, dropout)

    def forward(
        self,
        history: torch.Tensor,
        task_id: torch.Tensor,
        database_id: torch.Tensor,
        robot_id: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ctx = self.encoder(history)
        fused = torch.cat([
            ctx,
            self.task_emb(task_id),
            self.db_emb(database_id),
            self.robot_emb(robot_id),
            numeric_features,
        ], dim=-1)
        return {
            'day_logits': self.day_head(fused),
            'macroblock_logits': self.macroblock_head(fused),
            'fine_offset_logits': self.fine_offset_head(fused),
            'duration_delta': self.duration_head(fused).squeeze(-1),
        }
