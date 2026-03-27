# model/decoder.py
# WeekDecoder: genera la semana objetivo tarea a tarea (autoregresivo).

import torch
import torch.nn as nn

from config import N_NUMERIC_FEATS, N_MIN_BINS


class WeekDecoder(nn.Module):
    """
    LSTM autoregresivo que predice, para cada paso t:
      - tarea   (clasificación)
      - día     (clasificación, 7 clases)
      - hora    (clasificación, 24 clases)
      - minuto  (clasificación, N_MIN_BINS clases)
      - duración (regresión, 1 valor)
    """

    def __init__(self, num_tasks: int, embed_dim: int = 32,
                 hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim + N_NUMERIC_FEATS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.head_task     = nn.Linear(hidden_size, num_tasks)
        self.head_day      = nn.Linear(hidden_size, 7)
        self.head_hour     = nn.Linear(hidden_size, 24)
        self.head_minute   = nn.Linear(hidden_size, N_MIN_BINS)
        self.head_duration = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,          # (batch, seq_len, 8)
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ):
        task_ids  = x[:, :, 0].long()   # (batch, seq_len)
        num_feats = x[:, :, 1:8]         # (batch, seq_len, 7) — cols 8-10 son etiquetas, no input

        embedded = self.task_embedding(task_ids)
        inp      = torch.cat([embedded, num_feats], dim=-1)

        output, (h, c) = self.lstm(inp, (hidden, cell))

        return (
            self.head_task(output),                   # (batch, seq_len, num_tasks)
            self.head_day(output),                    # (batch, seq_len, 7)
            self.head_hour(output),                   # (batch, seq_len, 24)
            self.head_minute(output),                 # (batch, seq_len, N_MIN_BINS)
            self.head_duration(output).squeeze(-1),   # (batch, seq_len)
            h, c,
        )