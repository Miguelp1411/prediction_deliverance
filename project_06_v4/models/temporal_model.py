import torch
import torch.nn as nn

from config import num_global_day_offset_classes, num_local_start_offset_classes
from .blocks import SequenceContextEncoder


class TemporalAssignmentModel(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        history_feature_dim: int,
        num_tasks: int,
        max_occurrences: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        task_embed_dim: int,
        occurrence_embed_dim: int,
        day_embed_dim: int,
    ):
        super().__init__()
        self.sequence_encoder = SequenceContextEncoder(
            input_dim=sequence_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.occ_embedding = nn.Embedding(max_occurrences, occurrence_embed_dim)
        self.anchor_day_embedding = nn.Embedding(7, day_embed_dim)
        self.history_mlp = nn.Sequential(
            nn.LayerNorm(history_feature_dim + 2),
            nn.Linear(history_feature_dim + 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        fused_dim = hidden_size * 2 + task_embed_dim + occurrence_embed_dim + day_embed_dim + hidden_size
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
        )
        self.day_offset_head = nn.Linear(hidden_size * 2, num_global_day_offset_classes())
        self.local_offset_head = nn.Linear(hidden_size * 2, num_local_start_offset_classes())
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.sequence_encoder(sequence)

    def forward_with_context(
        self,
        sequence_context: torch.Tensor,
        task_id: torch.Tensor,
        occurrence_slot: torch.Tensor,
        history_features: torch.Tensor,
        predicted_count_norm: torch.Tensor,
        occurrence_progress: torch.Tensor,
        anchor_day: torch.Tensor,
    ):
        task_vec = self.task_embedding(task_id)
        occ_vec = self.occ_embedding(occurrence_slot)
        anchor_day_vec = self.anchor_day_embedding(anchor_day)
        hist_input = torch.cat([history_features, predicted_count_norm.unsqueeze(-1), occurrence_progress.unsqueeze(-1)], dim=-1)
        hist_vec = self.history_mlp(hist_input)
        fused = self.fusion(torch.cat([sequence_context, task_vec, occ_vec, anchor_day_vec, hist_vec], dim=-1))
        return {
            'day_offset_logits': self.day_offset_head(fused),
            'local_offset_logits': self.local_offset_head(fused),
            'pred_duration_norm': self.duration_head(fused).squeeze(-1),
        }

    def forward(
        self,
        sequence: torch.Tensor,
        task_id: torch.Tensor,
        occurrence_slot: torch.Tensor,
        history_features: torch.Tensor,
        predicted_count_norm: torch.Tensor,
        occurrence_progress: torch.Tensor,
        anchor_day: torch.Tensor,
    ):
        sequence_context = self.encode_sequence(sequence)
        return self.forward_with_context(
            sequence_context,
            task_id,
            occurrence_slot,
            history_features,
            predicted_count_norm,
            occurrence_progress,
            anchor_day,
        )
