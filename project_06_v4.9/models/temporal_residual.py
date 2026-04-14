"""
Residual Temporal Model.

Generates top-k time slot candidates with scores for each event.
Predicts day, time-of-day, duration, all as offsets from template anchors.
Produces uncertainty/confidence for downstream scheduler scoring.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResidualModel(nn.Module):
    """
    Temporal model: produces per-event slot predictions.

    For each event: day logits (7), time-of-day logits (bins_per_day),
    duration estimate, and confidence score.
    """

    def __init__(
        self,
        sequence_dim: int,
        history_feature_dim: int,
        num_tasks: int,
        num_databases: int,
        max_occurrences: int,
        hidden_size: int = 192,
        num_layers: int = 3,
        dropout: float = 0.25,
        task_embed_dim: int = 44,
        db_embed_dim: int = 16,
        occ_embed_dim: int = 16,
        day_embed_dim: int = 8,
        num_day_classes: int = 7,
        num_time_classes: int = 288,  # 24*60/5
    ) -> None:
        super().__init__()
        self.num_day_classes = num_day_classes
        self.num_time_classes = num_time_classes

        # BiGRU sequence encoder
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
        self.seq_proj = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Embeddings
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.db_embedding = nn.Embedding(max(num_databases, 1), db_embed_dim)
        self.occ_embedding = nn.Embedding(max_occurrences, occ_embed_dim)
        self.anchor_day_embedding = nn.Embedding(num_day_classes, day_embed_dim)
        self.anchor_time_embedding = nn.Embedding(num_time_classes, day_embed_dim)

        # History feature MLP
        self.history_mlp = nn.Sequential(
            nn.LayerNorm(history_feature_dim + 2),  # +2 for count_norm, progress
            nn.Linear(history_feature_dim + 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

        # Fusion
        fused_dim = (
            hidden_size * 2  # sequence context
            + task_embed_dim
            + db_embed_dim
            + occ_embed_dim
            + day_embed_dim * 2  # anchor day + anchor time
            + hidden_size  # history
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
        )

        # Output heads
        self.day_head = nn.Linear(hidden_size * 2, num_day_classes)
        self.time_head = nn.Linear(hidden_size * 2, num_time_classes)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """Encode context sequence. (B, T, D) → (B, hidden*2)."""
        x = self.seq_norm(sequence)
        outputs, hidden = self.seq_rnn(x)
        attn_scores = self.seq_attn(outputs).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * attn_weights, dim=1)
        last_fwd = hidden[-2]
        last_bwd = hidden[-1]
        last_hidden = torch.cat([last_fwd, last_bwd], dim=-1)
        return self.seq_proj(torch.cat([pooled, last_hidden], dim=-1))

    def forward(
        self,
        sequence: torch.Tensor,           # (B, T, seq_dim)
        task_id: torch.Tensor,             # (B,)
        db_id: torch.Tensor,               # (B,)
        occurrence_slot: torch.Tensor,     # (B,)
        history_features: torch.Tensor,    # (B, hist_dim)
        predicted_count_norm: torch.Tensor, # (B,)
        occurrence_progress: torch.Tensor,  # (B,)
        anchor_day: torch.Tensor,          # (B,)
        anchor_time_bin: torch.Tensor,     # (B,)
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns:
            - day_logits: (B, 7)
            - time_of_day_logits: (B, num_time_classes)
            - pred_duration_norm: (B,)
            - confidence: (B,)
        """
        # Encode context (shared across events from same week)
        seq_ctx = self.encode_sequence(sequence)

        # Embeddings
        task_vec = self.task_embedding(task_id)
        db_vec = self.db_embedding(db_id)
        occ_vec = self.occ_embedding(occurrence_slot)
        anchor_day_vec = self.anchor_day_embedding(anchor_day.clamp(0, self.num_day_classes - 1))
        anchor_time_vec = self.anchor_time_embedding(
            anchor_time_bin.clamp(0, self.num_time_classes - 1)
        )

        # History features
        hist_input = torch.cat([
            history_features,
            predicted_count_norm.unsqueeze(-1),
            occurrence_progress.unsqueeze(-1),
        ], dim=-1)
        hist_vec = self.history_mlp(hist_input)

        # Fusion
        fused = torch.cat([
            seq_ctx, task_vec, db_vec, occ_vec,
            anchor_day_vec, anchor_time_vec, hist_vec,
        ], dim=-1)
        h = self.fusion(fused)

        return {
            "day_logits": self.day_head(h),
            "time_of_day_logits": self.time_head(h),
            "pred_duration_norm": self.duration_head(h).squeeze(-1),
            "confidence": self.confidence_head(h).squeeze(-1),
        }

    def predict_topk(
        self,
        outputs: dict[str, torch.Tensor],
        k: int = 8,
        bins_per_day: int = 288,
    ) -> list[list[dict]]:
        """
        Generate top-k slot candidates from model outputs.

        Returns list (batch) of list (top-k candidates), each candidate is:
        {"start_bin": int, "day": int, "time_bin": int, "score": float,
         "duration_norm": float, "confidence": float}
        """
        batch_size = outputs["day_logits"].shape[0]
        day_probs = F.softmax(outputs["day_logits"], dim=-1)
        time_probs = F.softmax(outputs["time_of_day_logits"], dim=-1)
        dur_norm = outputs["pred_duration_norm"]
        confidence = outputs["confidence"]

        results: list[list[dict]] = []
        for b in range(batch_size):
            # Get top days and top times
            day_topk = torch.topk(day_probs[b], min(3, day_probs.shape[-1]))
            time_topk = torch.topk(time_probs[b], min(k, time_probs.shape[-1]))

            candidates: list[dict] = []
            for di in range(day_topk.values.shape[0]):
                day_idx = day_topk.indices[di].item()
                day_score = day_topk.values[di].item()
                for ti in range(time_topk.values.shape[0]):
                    time_idx = time_topk.indices[ti].item()
                    time_score = time_topk.values[ti].item()
                    start_bin = day_idx * bins_per_day + time_idx
                    combined_score = day_score * time_score * confidence[b].item()
                    candidates.append({
                        "start_bin": start_bin,
                        "day": day_idx,
                        "time_bin": time_idx,
                        "score": combined_score,
                        "duration_norm": dur_norm[b].item(),
                        "confidence": confidence[b].item(),
                    })

            # Sort by score and take top-k
            candidates.sort(key=lambda x: x["score"], reverse=True)
            results.append(candidates[:k])

        return results
