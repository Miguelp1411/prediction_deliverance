# model/encoder.py
# WeekEncoder: resume un histórico de semanas en un vector de contexto (h, c).

import torch
import torch.nn as nn

from proyectos_anteriores.project_01.config import N_NUMERIC_FEATS


class WeekEncoder(nn.Module):
    """
    Dos LSTMs apilados:
      1. task_lstm  → resume las tareas de cada semana individual.
      2. week_lstm  → procesa la secuencia de resúmenes semanales.

    Devuelve (context_h, context_c) que inicializan el decoder.
    """

    def __init__(self, num_tasks: int, embed_dim: int = 32,
                 hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks + 1, embed_dim, padding_idx=0)

        self.task_lstm = nn.LSTM(
            input_size=embed_dim + N_NUMERIC_FEATS,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.week_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,           # (batch, num_weeks, max_tasks, 8)
        padding_mask: torch.Tensor, # (batch, num_weeks, max_tasks)  bool
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch, num_weeks, max_tasks, _ = x.shape
        week_summaries = []

        for w in range(num_weeks):
            week_data = x[:, w, :, :]                  # (batch, max_tasks, 11)
            task_ids  = week_data[:, :, 0].long()      # (batch, max_tasks)
            num_feats = week_data[:, :, 1:8]           # (batch, max_tasks, 7) — solo features, cols 8-10 son etiquetas

            embedded = self.task_embedding(task_ids) # (batch, max_tasks, embed_dim)
            inp      = torch.cat([embedded, num_feats], dim=-1)

            mask_w  = padding_mask[:, w, :]
            lengths = mask_w.sum(dim=1).clamp(min=1).cpu()

            packed = nn.utils.rnn.pack_padded_sequence(
                inp, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.task_lstm(packed)
            week_summaries.append(hidden[-1])        # última capa → (batch, hidden)

        week_seq = torch.stack(week_summaries, dim=1)  # (batch, num_weeks, hidden)
        _, (context_h, context_c) = self.week_lstm(week_seq)

        return context_h, context_c