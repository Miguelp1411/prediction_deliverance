import torch
import torch.nn as nn

from models.blocks import SequenceContextEncoder


class TaskOccurrenceModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        max_count_cap: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.max_count_cap = max_count_cap
        self.encoder = SequenceContextEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_tasks * (max_count_cap + 1)),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        context = self.encoder(sequence)
        logits = self.head(context)
        return logits.view(sequence.shape[0], self.num_tasks, self.max_count_cap + 1)
