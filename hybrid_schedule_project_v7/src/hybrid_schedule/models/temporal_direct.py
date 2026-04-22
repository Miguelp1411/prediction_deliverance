from __future__ import annotations

import torch
from torch import nn

from .blocks import SequenceEncoder


class _AttentivePool(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        hidden = max(dim // 2, 32)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class _ResidualFusionBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class TemporalDirectNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        num_databases: int,
        num_robots: int,
        window_weeks: int,
        bins_per_day: int,
        max_slots: int,
        hidden_size: int = 192,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.25,
        task_embed_dim: int = 32,
        database_embed_dim: int = 16,
        robot_embed_dim: int = 16,
        slot_embed_dim: int = 24,
        day_embed_dim: int = 8,
        time_embed_dim: int = 16,
        numeric_feature_dim: int = 18,
    ):
        super().__init__()
        if hidden_size % max(1, num_heads) != 0:
            raise ValueError('hidden_size debe ser divisible por num_heads en TemporalDirectNet')
        self.window_weeks = int(window_weeks)
        self.bins_per_day = int(bins_per_day)
        self.max_slots = int(max_slots)

        self.sequence_encoder = SequenceEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.week_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.window_weeks, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=max(1, int(num_heads)),
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.pool = _AttentivePool(hidden_size, dropout)

        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(num_databases, database_embed_dim)
        self.robot_emb = nn.Embedding(num_robots, robot_embed_dim)
        self.slot_emb = nn.Embedding(max(1, max_slots), slot_embed_dim)
        self.anchor_day_emb = nn.Embedding(7, day_embed_dim)
        self.anchor_time_emb = nn.Embedding(max(1, bins_per_day), time_embed_dim)

        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(int(numeric_feature_dim)),
            nn.Linear(int(numeric_feature_dim), hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fused_dim = (
            hidden_size * 2
            + hidden_size
            + hidden_size
            + task_embed_dim
            + database_embed_dim
            + robot_embed_dim
            + slot_embed_dim
            + day_embed_dim
            + time_embed_dim
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size * 3),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fusion_dim = hidden_size * 3
        self.fusion_blocks = nn.Sequential(
            _ResidualFusionBlock(fusion_dim, dropout),
            _ResidualFusionBlock(fusion_dim, dropout),
            nn.LayerNorm(fusion_dim),
        )
        self.day_head = nn.Linear(fusion_dim, 7)
        self.time_head = nn.Linear(fusion_dim, self.bins_per_day)
        self.duration_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
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
        slot_id: torch.Tensor,
        anchor_day: torch.Tensor,
        anchor_time: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        seq_ctx = self.sequence_encoder(history)
        week_tokens = self.week_proj(history)
        seq_len = week_tokens.shape[1]
        pos = self.pos_embedding[:, :seq_len, :]
        week_tokens = week_tokens + pos
        trans_ctx = self.pool(self.transformer(week_tokens))
        numeric_ctx = self.numeric_proj(numeric_features)
        slot_id = slot_id.clamp(min=0, max=max(0, self.max_slots - 1))
        anchor_day = anchor_day.clamp(min=0, max=6)
        anchor_time = anchor_time.clamp(min=0, max=max(0, self.bins_per_day - 1))
        fused = torch.cat([
            seq_ctx,
            trans_ctx,
            numeric_ctx,
            self.task_emb(task_id),
            self.db_emb(database_id),
            self.robot_emb(robot_id),
            self.slot_emb(slot_id),
            self.anchor_day_emb(anchor_day),
            self.anchor_time_emb(anchor_time),
        ], dim=-1)
        fused = self.fusion_blocks(self.fusion(fused))
        return {
            'day_logits': self.day_head(fused),
            'time_logits': self.time_head(fused),
            'pred_log_duration': self.duration_head(fused).squeeze(-1),
        }
