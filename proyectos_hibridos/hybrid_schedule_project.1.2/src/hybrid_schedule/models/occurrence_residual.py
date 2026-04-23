from __future__ import annotations

import torch
from torch import nn

from .blocks import FeatureAdapter, MLP, SequenceEncoder


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
        numeric_dim: int = 10,
    ):
        super().__init__()
        self.max_delta = int(max_delta)
        self.encoder = SequenceEncoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(max(1, num_databases), database_embed_dim)
        self.robot_emb = nn.Embedding(max(1, num_robots), robot_embed_dim)
        self.adapter = FeatureAdapter(database_embed_dim + robot_embed_dim, hidden_size * 2, hidden_size)
        fused_dim = hidden_size * 2 + task_embed_dim + database_embed_dim + robot_embed_dim + int(numeric_dim)
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
        change_logits = self.change_head(fused)
        delta_logits = self.delta_head(fused)
        delta_values = torch.arange(-self.max_delta, self.max_delta + 1, device=history.device)
        delta_probs = delta_logits.softmax(dim=-1)
        expected_delta = (delta_probs * delta_values.unsqueeze(0)).sum(dim=-1)
        pred_count = torch.clamp(template_count.float() + expected_delta, min=0.0)
        return {
            'change_logits': change_logits,
            'delta_logits': delta_logits,
            'delta_probs': delta_probs,
            'expected_delta': expected_delta,
            'pred_count': pred_count,
        }
