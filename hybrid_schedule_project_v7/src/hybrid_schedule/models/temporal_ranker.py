from __future__ import annotations

import torch
from torch import nn

from .blocks import SequenceEncoder


class TemporalRankingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        num_databases: int,
        num_robots: int,
        hidden_size: int = 160,
        num_layers: int = 1,
        dropout: float = 0.30,
        task_embed_dim: int = 24,
        database_embed_dim: int = 12,
        robot_embed_dim: int = 12,
        numeric_feature_dim: int = 18,
        candidate_feature_dim: int = 18,
    ):
        super().__init__()
        self.encoder = SequenceEncoder(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.ctx_dropout = nn.Dropout(dropout)
        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(num_databases, database_embed_dim)
        self.robot_emb = nn.Embedding(num_robots, robot_embed_dim)

        context_dim = hidden_size * 2 + task_embed_dim + database_embed_dim + robot_embed_dim + int(numeric_feature_dim)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.candidate_proj = nn.Sequential(
            nn.Linear(int(candidate_feature_dim), hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        task_id: torch.Tensor,
        database_id: torch.Tensor,
        robot_id: torch.Tensor,
        numeric_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        ctx = self.ctx_dropout(self.encoder(history))
        fused = torch.cat([
            ctx,
            self.task_emb(task_id),
            self.db_emb(database_id),
            self.robot_emb(robot_id),
            numeric_features,
        ], dim=-1)
        context_repr = self.context_proj(fused)
        candidate_repr = self.candidate_proj(candidate_features)
        expanded_context = context_repr.unsqueeze(1).expand(-1, candidate_repr.shape[1], -1)
        logits = self.score_head(torch.cat([expanded_context, candidate_repr], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), -1e9)
        return {
            'candidate_logits': logits,
        }
