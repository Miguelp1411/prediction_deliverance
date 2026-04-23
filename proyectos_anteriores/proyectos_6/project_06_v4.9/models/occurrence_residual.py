"""
Residual Occurrence Model.

Predicts Δ = actual_count - template_count per task type.
Two heads:
  1. changed/unchanged binary classifier
  2. delta_class ordinal classifier (range [-K, +K])

Architecture: BiGRU encoder over historical context + per-task features
→ fusion with task/db embeddings → dual head output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccurrenceResidualModel(nn.Module):
    """Residual occurrence model: predicts Δcount from template."""

    def __init__(
        self,
        feature_dim: int,
        sequence_dim: int,
        num_tasks: int,
        num_databases: int,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout: float = 0.10,
        task_embed_dim: int = 16,
        db_embed_dim: int = 8,
        delta_range: int = 6,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.delta_range = delta_range
        self.num_delta_classes = 2 * delta_range + 1  # [-K, ..., 0, ..., +K]

        # Embeddings
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.db_embedding = nn.Embedding(max(num_databases, 1), db_embed_dim)

        # Context sequence encoder (BiGRU)
        self.seq_norm = nn.LayerNorm(sequence_dim)
        self.seq_rnn = nn.GRU(
            input_size=sequence_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.seq_attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Per-task feature MLP
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Fusion
        fused_dim = hidden_size * 2 + hidden_size + task_embed_dim + db_embed_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
        )

        # Head 1: changed/unchanged (binary)
        self.change_head = nn.Linear(hidden_size, 2)

        # Head 2: delta class (ordinal)
        self.delta_head = nn.Linear(hidden_size, self.num_delta_classes)

    def _encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode context sequence. Input: (B, T, seq_dim) → (B, hidden*2)."""
        x = self.seq_norm(sequence)
        outputs, hidden = self.seq_rnn(x)
        attn_scores = self.seq_attn(outputs).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * attn_weights, dim=1)
        return pooled

    def forward(
        self,
        sequence: torch.Tensor,       # (B, T, seq_dim)
        task_features: torch.Tensor,  # (B, num_tasks, feat_dim)
        task_ids: torch.Tensor,        # (B, num_tasks) — just 0..num_tasks-1
        db_ids: torch.Tensor,          # (B,) — database index
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with:
            - change_logits: (B, num_tasks, 2)
            - delta_logits: (B, num_tasks, num_delta_classes)
            - predicted_delta: (B, num_tasks) — argmax delta as integer
        """
        batch_size = sequence.shape[0]

        # Encode context sequence: (B, hidden*2)
        seq_ctx = self._encode_sequence(sequence)

        # Per-task embeddings: (B, num_tasks, embed_dim)
        task_emb = self.task_embedding(task_ids)  # (B, T, task_embed)
        db_emb = self.db_embedding(db_ids).unsqueeze(1).expand(-1, self.num_tasks, -1)

        # Per-task features: (B, num_tasks, hidden)
        task_feat = self.feature_mlp(task_features)

        # Expand sequence context to per-task: (B, num_tasks, hidden*2)
        seq_expanded = seq_ctx.unsqueeze(1).expand(-1, self.num_tasks, -1)

        # Fuse
        fused = torch.cat([seq_expanded, task_feat, task_emb, db_emb], dim=-1)
        h = self.fusion(fused)  # (B, num_tasks, hidden)

        # Heads
        change_logits = self.change_head(h)  # (B, num_tasks, 2)
        delta_logits = self.delta_head(h)    # (B, num_tasks, num_delta_classes)

        # Predicted delta (argmax - delta_range to center at 0)
        predicted_delta = delta_logits.argmax(dim=-1) - self.delta_range

        return {
            "change_logits": change_logits,
            "delta_logits": delta_logits,
            "predicted_delta": predicted_delta,
        }

    def predict_counts(
        self,
        sequence: torch.Tensor,
        task_features: torch.Tensor,
        task_ids: torch.Tensor,
        db_ids: torch.Tensor,
        template_counts: torch.Tensor,  # (B, num_tasks)
    ) -> torch.Tensor:
        """Predict final counts = template + Δ, gated by change head."""
        outputs = self.forward(sequence, task_features, task_ids, db_ids)

        # Gate: if change_prob < 0.5, delta = 0
        change_prob = F.softmax(outputs["change_logits"], dim=-1)[..., 1]  # prob of changed
        delta = outputs["predicted_delta"].float()
        gated_delta = torch.where(change_prob > 0.5, delta, torch.zeros_like(delta))

        final_counts = template_counts.float() + gated_delta
        return final_counts.clamp(min=0).round().long()
