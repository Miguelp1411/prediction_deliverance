from __future__ import annotations

import torch
from torch import nn

from .blocks import FeatureAdapter, MLP, SequenceEncoder


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
        bins_per_day: int = 288,
        numeric_dim: int = 10,
    ):
        super().__init__()
        self.bins_per_day = int(bins_per_day)
        self.encoder = SequenceEncoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(max(1, num_databases), database_embed_dim)
        self.robot_emb = nn.Embedding(max(1, num_robots), robot_embed_dim)
        self.adapter = FeatureAdapter(database_embed_dim + robot_embed_dim, hidden_size * 2, hidden_size)
        fused_dim = hidden_size * 2 + task_embed_dim + database_embed_dim + robot_embed_dim + int(numeric_dim)
        self.day_head = MLP(fused_dim, hidden_size, 7, dropout)
        self.time_head = MLP(fused_dim, hidden_size, self.bins_per_day, dropout)

    def forward(
        self,
        history: torch.Tensor,
        task_id: torch.Tensor,
        database_id: torch.Tensor,
        robot_id: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        db_vec = self.db_emb(database_id)
        robot_vec = self.robot_emb(robot_id)
        condition = torch.cat([db_vec, robot_vec], dim=-1)
        ctx = self.adapter(self.encoder(history), condition)
        fused = torch.cat([
            ctx,
            self.task_emb(task_id),
            db_vec,
            robot_vec,
            numeric_features,
        ], dim=-1)
        day_logits = self.day_head(fused)
        time_logits = self.time_head(fused)
        return {
            'day_logits': day_logits,
            'time_logits': time_logits,
        }
