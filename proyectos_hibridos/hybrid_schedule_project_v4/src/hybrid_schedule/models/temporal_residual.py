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
        day_offset_radius: int = 3,
        local_offset_radius: int = 72,
        numeric_feature_dim: int = 18,
    ):
        super().__init__()
        self.day_offset_radius = int(day_offset_radius)
        self.local_offset_radius = int(local_offset_radius)
        self.encoder = SequenceEncoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(num_databases, database_embed_dim)
        self.robot_emb = nn.Embedding(num_robots, robot_embed_dim)
        fused_dim = hidden_size * 2 + task_embed_dim + database_embed_dim + robot_embed_dim + int(numeric_feature_dim)
        self.day_offset_head = MLP(fused_dim, hidden_size, self.day_offset_radius * 2 + 1, dropout)
        self.local_offset_head = MLP(fused_dim, hidden_size, self.local_offset_radius * 2 + 1, dropout)
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
            'day_offset_logits': self.day_offset_head(fused),
            'local_offset_logits': self.local_offset_head(fused),
            'duration_delta': self.duration_head(fused).squeeze(-1),
        }
