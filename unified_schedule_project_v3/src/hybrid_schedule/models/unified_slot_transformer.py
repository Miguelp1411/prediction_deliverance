from __future__ import annotations

import torch
from torch import nn

from .blocks import AttentivePool, ResidualMLP


class UnifiedSlotTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        numeric_feature_dim: int,
        num_tasks: int,
        num_databases: int,
        num_robots: int,
        max_slots: int,
        window_weeks: int,
        bins_per_day: int,
        hidden_size: int = 256,
        history_layers: int = 2,
        history_heads: int = 8,
        query_layers: int = 3,
        query_heads: int = 8,
        cross_layers: int = 2,
        dropout: float = 0.20,
        task_embed_dim: int = 48,
        database_embed_dim: int = 24,
        robot_embed_dim: int = 24,
        slot_embed_dim: int = 24,
        day_embed_dim: int = 12,
        time_embed_dim: int = 24,
        numeric_hidden_dim: int = 192,
        active_temperature: float = 1.0,
    ):
        super().__init__()
        if hidden_size % max(1, history_heads) != 0:
            raise ValueError('hidden_size debe ser divisible por history_heads')
        if hidden_size % max(1, query_heads) != 0:
            raise ValueError('hidden_size debe ser divisible por query_heads')

        self.window_weeks = int(window_weeks)
        self.bins_per_day = int(bins_per_day)
        self.max_slots = int(max_slots)
        self.hidden_size = int(hidden_size)
        self.active_temperature = float(active_temperature)

        self.history_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.history_pos = nn.Parameter(torch.zeros(1, self.window_weeks, hidden_size))
        history_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(history_heads),
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(history_layer, num_layers=max(1, int(history_layers)))
        self.history_pool = AttentivePool(hidden_size, dropout)

        self.task_emb = nn.Embedding(num_tasks, task_embed_dim)
        self.db_emb = nn.Embedding(num_databases, database_embed_dim)
        self.robot_emb = nn.Embedding(num_robots, robot_embed_dim)
        self.slot_emb = nn.Embedding(max(1, max_slots), slot_embed_dim)
        self.anchor_day_emb = nn.Embedding(7, day_embed_dim)
        self.anchor_time_emb = nn.Embedding(max(1, bins_per_day), time_embed_dim)

        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_feature_dim),
            nn.Linear(numeric_feature_dim, numeric_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(numeric_hidden_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        query_input_dim = hidden_size + task_embed_dim + database_embed_dim + robot_embed_dim + slot_embed_dim + day_embed_dim + time_embed_dim
        self.query_proj = nn.Sequential(
            nn.LayerNorm(query_input_dim),
            nn.Linear(query_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=max(1, int(query_heads)), dropout=dropout, batch_first=True)
            for _ in range(max(1, int(cross_layers)))
        ])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(max(1, int(cross_layers)))])
        self.cross_ff = nn.ModuleList([ResidualMLP(hidden_size, dropout) for _ in range(max(1, int(cross_layers)))])

        query_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(query_heads),
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.query_encoder = nn.TransformerEncoder(query_layer, num_layers=max(1, int(query_layers)))
        self.global_query_pool = AttentivePool(hidden_size, dropout)

        fused_dim = hidden_size * 3
        self.head_prep = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.active_head = nn.Linear(hidden_size, 1)
        self.day_head = nn.Linear(hidden_size, 7)
        self.time_head = nn.Linear(hidden_size, self.bins_per_day)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        task_ids: torch.Tensor,
        database_ids: torch.Tensor,
        robot_ids: torch.Tensor,
        slot_ids: torch.Tensor,
        anchor_days: torch.Tensor,
        anchor_times: torch.Tensor,
        numeric_features: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        week_tokens = self.history_proj(history)
        seq_len = week_tokens.shape[1]
        week_tokens = week_tokens + self.history_pos[:, :seq_len, :]
        history_tokens = self.history_encoder(week_tokens)
        history_global = self.history_pool(history_tokens)

        slot_ids = slot_ids.clamp(min=0, max=max(0, self.max_slots - 1))
        anchor_days = anchor_days.clamp(min=0, max=6)
        anchor_times = anchor_times.clamp(min=0, max=max(0, self.bins_per_day - 1))

        numeric_ctx = self.numeric_proj(numeric_features)
        query = torch.cat([
            numeric_ctx,
            self.task_emb(task_ids),
            self.db_emb(database_ids),
            self.robot_emb(robot_ids),
            self.slot_emb(slot_ids),
            self.anchor_day_emb(anchor_days),
            self.anchor_time_emb(anchor_times),
        ], dim=-1)
        query = self.query_proj(query)

        for attn, norm, ff in zip(self.cross_attn_layers, self.cross_norms, self.cross_ff):
            attended, _ = attn(query, history_tokens, history_tokens, need_weights=False)
            query = norm(query + attended)
            query = ff(query)

        query = self.query_encoder(query, src_key_padding_mask=(~query_mask.bool()) if query_mask is not None else None)
        query_global = self.global_query_pool(query, mask=query_mask if query_mask is not None else None)
        fused = torch.cat([
            query,
            history_global.unsqueeze(1).expand(-1, query.shape[1], -1),
            query_global.unsqueeze(1).expand(-1, query.shape[1], -1),
        ], dim=-1)
        fused = self.head_prep(fused)

        active_logits = self.active_head(fused).squeeze(-1) / max(self.active_temperature, 1e-6)
        day_logits = self.day_head(fused)
        time_logits = self.time_head(fused)
        pred_log_duration = self.duration_head(fused).squeeze(-1)

        if query_mask is not None:
            invalid = ~query_mask.bool()
            active_logits = active_logits.masked_fill(invalid, -1e4)

        return {
            'active_logits': active_logits,
            'day_logits': day_logits,
            'time_logits': time_logits,
            'pred_log_duration': pred_log_duration,
        }
