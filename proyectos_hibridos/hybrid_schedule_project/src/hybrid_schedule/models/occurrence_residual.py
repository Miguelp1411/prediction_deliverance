from __future__ import annotations

import torch
from torch import nn

from .blocks import MLP, SequenceEncoder


class OccurrenceResidualNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        num_databases: int,
        num_robots: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.15,
        task_embed_dim: int = 24,
        database_embed_dim: int = 12,
        robot_embed_dim: int = 12,
        max_delta: int = 12,
    ):
        super().__init__()
        self.max_delta = int(max_delta)
        self.encoder = SequenceEncoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(num_databases, database_embed_dim)
        self.robot_emb = nn.Embedding(num_robots, robot_embed_dim)
        static_dim = 6  # template_count, recent_mean, recent_last, recent_median, support_score, target_woy_sin? simplified
        fused_dim = hidden_size * 2 + task_embed_dim + database_embed_dim + robot_embed_dim + static_dim
        self.change_head = MLP(fused_dim, hidden_size, 2, dropout)
        self.delta_head = MLP(fused_dim, hidden_size, self.max_delta * 2 + 1, dropout)

    def forward(
        self,
        history: torch.Tensor,
        task_id: torch.Tensor,
        database_id: torch.Tensor,
        robot_id: torch.Tensor,
        numeric_features: torch.Tensor,
        template_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ctx = self.encoder(history)
        fused = torch.cat([
            ctx,
            self.task_emb(task_id),
            self.db_emb(database_id),
            self.robot_emb(robot_id),
            numeric_features,
        ], dim=-1)
        change_logits = self.change_head(fused)
        delta_logits = self.delta_head(fused)
        delta_values = torch.arange(-self.max_delta, self.max_delta + 1, device=history.device)
        pred_delta = delta_values[delta_logits.argmax(dim=-1)]
        pred_count = torch.clamp(template_count + pred_delta, min=0)
        return {
            'change_logits': change_logits,
            'delta_logits': delta_logits,
            'pred_count': pred_count,
        }
