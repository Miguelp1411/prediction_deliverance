import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP,
    OCC_SEASONAL_BASELINE_LOGIT,
    OCC_SEASONAL_LAGS,
    OCC_SEASONAL_LAG_WEIGHTS,
)
from .blocks import SequenceContextEncoder


class TaskOccurrenceModel(nn.Module):
    """Modelo secuencial con baseline estacional fuerte + residual ordinal.

    La salida pública sigue siendo logits absolutos por conteo para mantener
    compatibilidad con el resto del pipeline, pero internamente la red aprende
    deltas respecto a un baseline estacional por tarea.

    Mejora v4.4:
    - permite rangos de delta específicos por tarea derivados del histórico;
    - aplica una penalización suave creciente fuera de esos rangos en lugar de
      competir siempre sobre el soporte completo [-max_count_cap, +max_count_cap].
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        max_count_cap: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        seasonal_lags: tuple[int, ...] = OCC_SEASONAL_LAGS,
        seasonal_lag_weights: tuple[float, ...] = OCC_SEASONAL_LAG_WEIGHTS,
        seasonal_baseline_logit: float = OCC_SEASONAL_BASELINE_LOGIT,
        task_delta_bounds: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        delta_outside_range_logit_penalty_per_step: float = OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP,
    ):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.max_count_cap = int(max_count_cap)
        self.seasonal_lags = tuple(int(l) for l in seasonal_lags)
        self.seasonal_baseline_logit = float(seasonal_baseline_logit)
        self.delta_outside_range_logit_penalty_per_step = float(delta_outside_range_logit_penalty_per_step)
        self.seasonal_feature_dim = len(self.seasonal_lags) * self.num_tasks * 2
        self.delta_values = torch.arange(-self.max_count_cap, self.max_count_cap + 1, dtype=torch.long)
        self.delta_num_classes = int(self.delta_values.numel())

        raw_weights = torch.as_tensor(seasonal_lag_weights, dtype=torch.float32)
        if raw_weights.numel() != len(self.seasonal_lags) or float(raw_weights.sum().item()) <= 0.0:
            raw_weights = torch.as_tensor([float(lag) for lag in self.seasonal_lags], dtype=torch.float32)
        raw_weights = raw_weights / raw_weights.sum().clamp(min=1e-6)
        self.register_buffer('seasonal_lag_weights', raw_weights)
        self.register_buffer('delta_index_values', self.delta_values.clone())

        normalized_bounds: list[tuple[int, int]] = []
        for task_id in range(self.num_tasks):
            if task_delta_bounds is not None and task_id < len(task_delta_bounds):
                lower_raw, upper_raw = task_delta_bounds[task_id]
            else:
                lower_raw, upper_raw = -self.max_count_cap, self.max_count_cap
            lower = int(max(-self.max_count_cap, min(lower_raw, upper_raw)))
            upper = int(min(self.max_count_cap, max(lower_raw, upper_raw)))
            normalized_bounds.append((lower, upper))
        self.task_delta_bounds = tuple(normalized_bounds)
        lower_bounds = torch.tensor([b[0] for b in normalized_bounds], dtype=torch.long)
        upper_bounds = torch.tensor([b[1] for b in normalized_bounds], dtype=torch.long)
        delta_values = self.delta_values.view(1, -1)
        lower_distance = (lower_bounds.view(-1, 1) - delta_values).clamp(min=0)
        upper_distance = (delta_values - upper_bounds.view(-1, 1)).clamp(min=0)
        self.register_buffer('delta_lower_bounds', lower_bounds)
        self.register_buffer('delta_upper_bounds', upper_bounds)
        self.register_buffer('delta_outside_distance', lower_distance + upper_distance)

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
            nn.Linear(hidden_size * 2, self.num_tasks * self.delta_num_classes),
        )

    def _extract_lag_tensors(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        batch_size = int(sequence.shape[0])
        if self.seasonal_feature_dim <= 0 or sequence.shape[-1] < self.seasonal_feature_dim:
            return None, None
        extra = sequence[:, -1, -self.seasonal_feature_dim :].view(batch_size, len(self.seasonal_lags), 2, self.num_tasks)
        lag_counts = extra[:, :, 0, :].clamp(min=0.0, max=float(self.max_count_cap))
        lag_masks = extra[:, :, 1, :].clamp(min=0.0, max=1.0)
        return lag_counts, lag_masks

    def _seasonal_baseline_counts_and_mask(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(sequence.shape[0])
        lag_counts, lag_masks = self._extract_lag_tensors(sequence)
        if lag_counts is None or lag_masks is None:
            zero_counts = torch.zeros(batch_size, self.num_tasks, dtype=torch.long, device=sequence.device)
            zero_mask = torch.zeros(batch_size, self.num_tasks, dtype=torch.bool, device=sequence.device)
            return zero_counts, zero_mask

        lag_weights = self.seasonal_lag_weights.to(dtype=sequence.dtype, device=sequence.device)
        weighted_masks = lag_masks * lag_weights.view(1, -1, 1)
        denom = weighted_masks.sum(dim=1)
        has_signal = denom > 0
        seasonal_counts = torch.where(
            has_signal,
            (lag_counts * weighted_masks).sum(dim=1) / denom.clamp(min=1e-6),
            torch.zeros_like(denom),
        )
        seasonal_counts = seasonal_counts.round().long().clamp(min=0, max=self.max_count_cap)
        return seasonal_counts, has_signal

    def _seasonal_baseline_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        seasonal_counts, has_signal = self._seasonal_baseline_counts_and_mask(sequence)
        one_hot = F.one_hot(seasonal_counts, num_classes=self.max_count_cap + 1).to(sequence.dtype)
        return one_hot * has_signal.unsqueeze(-1).to(sequence.dtype) * self.seasonal_baseline_logit

    def _apply_task_delta_range_penalty(self, residual_delta_logits: torch.Tensor) -> torch.Tensor:
        penalty_per_step = self.delta_outside_range_logit_penalty_per_step
        if penalty_per_step == 0.0:
            return residual_delta_logits
        outside_distance = self.delta_outside_distance.to(
            device=residual_delta_logits.device,
            dtype=residual_delta_logits.dtype,
        )
        penalty = outside_distance.unsqueeze(0) * penalty_per_step
        return residual_delta_logits + penalty

    def _residual_absolute_logits(self, residual_delta_logits: torch.Tensor, baseline_counts: torch.Tensor) -> torch.Tensor:
        batch_size = residual_delta_logits.shape[0]
        absolute_logits = torch.zeros(
            batch_size,
            self.num_tasks,
            self.max_count_cap + 1,
            dtype=residual_delta_logits.dtype,
            device=residual_delta_logits.device,
        )
        delta_values = self.delta_index_values.to(device=residual_delta_logits.device)
        absolute_indices = baseline_counts.unsqueeze(-1) + delta_values.view(1, 1, -1)
        valid = (absolute_indices >= 0) & (absolute_indices <= self.max_count_cap)
        clamped_indices = absolute_indices.clamp(min=0, max=self.max_count_cap)
        absolute_logits.scatter_add_(2, clamped_indices, residual_delta_logits * valid.to(residual_delta_logits.dtype))
        return absolute_logits

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        context = self.encoder(sequence)
        residual_delta_logits = self.head(context).view(sequence.shape[0], self.num_tasks, self.delta_num_classes)
        residual_delta_logits = self._apply_task_delta_range_penalty(residual_delta_logits)
        baseline_counts, _ = self._seasonal_baseline_counts_and_mask(sequence)
        residual_absolute_logits = self._residual_absolute_logits(residual_delta_logits, baseline_counts)
        return residual_absolute_logits + self._seasonal_baseline_logits(sequence)


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
