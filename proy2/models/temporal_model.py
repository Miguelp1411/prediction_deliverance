import torch
import torch.nn as nn

from models.blocks import SequenceContextEncoder


class TemporalAssignmentModel(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        history_feature_dim: int,
        num_tasks: int,
        max_occurrences: int,
        num_time_bins: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        task_embed_dim: int,
        occurrence_embed_dim: int,
    ):
        super().__init__()
        self.num_time_bins = num_time_bins
        self.sequence_encoder = SequenceContextEncoder(
            input_dim=sequence_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.occ_embedding = nn.Embedding(max_occurrences, occurrence_embed_dim)
        self.history_mlp = nn.Sequential(
            nn.LayerNorm(history_feature_dim + 1),
            nn.Linear(history_feature_dim + 1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        fused_dim = hidden_size * 2 + task_embed_dim + occurrence_embed_dim + hidden_size
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
        )
        self.start_head = nn.Linear(hidden_size * 2, num_time_bins)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        task_id: torch.Tensor,
        occurrence_index: torch.Tensor,
        history_features: torch.Tensor,
        predicted_count_norm: torch.Tensor,
    ):
        context = self.sequence_encoder(sequence)
        task_vec = self.task_embedding(task_id)
        occ_vec = self.occ_embedding(occurrence_index)
        hist_input = torch.cat([history_features, predicted_count_norm.unsqueeze(-1)], dim=-1)
        hist_vec = self.history_mlp(hist_input)
        fused = self.fusion(torch.cat([context, task_vec, occ_vec, hist_vec], dim=-1))
        start_logits = self.start_head(fused)
        duration_norm = self.duration_head(fused).squeeze(-1)
        return start_logits, duration_norm
