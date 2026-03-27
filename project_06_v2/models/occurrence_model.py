import torch
import torch.nn as nn

from .blocks import SequenceContextEncoder


class TaskOccurrenceModel(nn.Module):
    """Modelo original basado en encoder secuencial + clasificación por tarea."""

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


class StructuredOccurrenceModel(nn.Module):
    """Predictor estructurado orientado a series con estacionalidad fuerte.

    Regla principal:
    - usa los conteos de la misma fase hace `lag_weeks` semanas;
    - si no hay suficiente histórico, usa medianas por tarea aprendidas en fit().

    Devuelve logits para mantener compatibilidad con el resto del pipeline.
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        max_count_cap: int,
        lag_weeks: int = 4,
        confidence_logit: float = 12.0,
        off_logit: float = -12.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tasks = int(num_tasks)
        self.max_count_cap = int(max_count_cap)
        self.lag_weeks = int(lag_weeks)
        self.confidence_logit = float(confidence_logit)
        self.off_logit = float(off_logit)
        self.register_buffer('fallback_counts', torch.zeros(self.num_tasks, dtype=torch.long))

    def fit(self, target_counts: torch.Tensor | None = None):
        if target_counts is None or target_counts.numel() == 0:
            self.fallback_counts.zero_()
            return self
        target_counts = target_counts.long().clamp(min=0, max=self.max_count_cap)
        self.fallback_counts.copy_(torch.median(target_counts, dim=0).values)
        return self

    def predict_counts(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError('sequence debe tener forma [batch, window, feature_dim]')
        batch_size, window_size, _ = sequence.shape
        if window_size >= self.lag_weeks:
            lag_source = sequence[:, -self.lag_weeks, : self.num_tasks]
        elif window_size > 0:
            lag_source = sequence[:, 0, : self.num_tasks]
        else:
            lag_source = self.fallback_counts.unsqueeze(0).expand(batch_size, -1).float()
        pred_counts = torch.round(lag_source).long().clamp(min=0, max=self.max_count_cap)
        return pred_counts

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        pred_counts = self.predict_counts(sequence)
        logits = torch.full(
            (sequence.shape[0], self.num_tasks, self.max_count_cap + 1),
            fill_value=self.off_logit,
            dtype=sequence.dtype,
            device=sequence.device,
        )
        logits.scatter_(2, pred_counts.unsqueeze(-1), self.confidence_logit)
        return logits
