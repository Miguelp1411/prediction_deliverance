import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP,
    OCC_SEASONAL_BASELINE_LOGIT,
    OCC_SEASONAL_LAGS,
    OCC_SEASONAL_LAG_WEIGHTS,
    OCC_V48_RECENT_WINDOW,
    OCC_V48_TASK_EMBED_DIM,
    OCC_V48_USE_LAST_COUNT_CANDIDATE,
    OCC_V48_USE_RECENT_MEAN_CANDIDATE,
    OCC_V48_USE_TASK_MEDIAN_CANDIDATE,
    OCC_V48_ZERO_DELTA_LOGIT_BIAS,
)
from .blocks import SequenceContextEncoder


class TaskOccurrenceModel(nn.Module):
    """Modelo secuencial con baseline estacional fuerte + residual ordinal.

    La salida pública sigue siendo logits absolutos por conteo para mantener
    compatibilidad con el resto del pipeline, pero internamente la red aprende
    deltas respecto a un baseline estacional por tarea.
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


class TaskMoEOccurrenceModelV48(nn.Module):
    """Occurrence v4.8: mixture-of-lags discreto + residual por delta + encoder compacto.

    Objetivos de diseño para este proyecto:
    - usar solo señales específicas de conteo y calendario;
    - copiar candidatos discretos (lags/reciente/mediana) en lugar de promediar conteos;
    - aprender deltas pequeños alrededor de esos candidatos;
    - compartir encoder entre tareas pero con embedding e inferencia por tarea.

    La salida es un dict para poder supervisar también selector y delta. La clave
    `logits` mantiene compatibilidad con el resto del pipeline.
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
        task_delta_bounds: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
        delta_outside_range_logit_penalty_per_step: float = OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP,
        task_embed_dim: int = OCC_V48_TASK_EMBED_DIM,
        recent_window: int = OCC_V48_RECENT_WINDOW,
        zero_delta_logit_bias: float = OCC_V48_ZERO_DELTA_LOGIT_BIAS,
        use_recent_mean_candidate: bool = OCC_V48_USE_RECENT_MEAN_CANDIDATE,
        use_last_count_candidate: bool = OCC_V48_USE_LAST_COUNT_CANDIDATE,
        use_task_median_candidate: bool = OCC_V48_USE_TASK_MEDIAN_CANDIDATE,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_tasks = int(num_tasks)
        self.max_count_cap = int(max_count_cap)
        self.seasonal_lags = tuple(int(l) for l in seasonal_lags)
        self.recent_window = max(1, int(recent_window))
        self.use_recent_mean_candidate = bool(use_recent_mean_candidate)
        self.use_last_count_candidate = bool(use_last_count_candidate)
        self.use_task_median_candidate = bool(use_task_median_candidate)
        self.delta_outside_range_logit_penalty_per_step = float(delta_outside_range_logit_penalty_per_step)

        self.base_week_feature_dim = 14 * self.num_tasks + 7
        self.seasonal_feature_dim = len(self.seasonal_lags) * self.num_tasks * 2
        self.delta_values = torch.arange(-self.max_count_cap, self.max_count_cap + 1, dtype=torch.long)
        self.delta_num_classes = int(self.delta_values.numel())
        self.register_buffer('delta_index_values', self.delta_values.clone())
        self.register_buffer('task_median_counts', torch.zeros(self.num_tasks, dtype=torch.long))

        normalized_bounds: list[tuple[int, int]] = []
        for task_id in range(self.num_tasks):
            if task_delta_bounds is not None and task_id < len(task_delta_bounds):
                lower_raw, upper_raw = task_delta_bounds[task_id]
            else:
                lower_raw, upper_raw = -self.max_count_cap, self.max_count_cap
            lower = int(max(-self.max_count_cap, min(lower_raw, upper_raw)))
            upper = int(min(self.max_count_cap, max(lower_raw, upper_raw)))
            normalized_bounds.append((lower, upper))
        lower_bounds = torch.tensor([b[0] for b in normalized_bounds], dtype=torch.long)
        upper_bounds = torch.tensor([b[1] for b in normalized_bounds], dtype=torch.long)
        delta_values = self.delta_values.view(1, -1)
        lower_distance = (lower_bounds.view(-1, 1) - delta_values).clamp(min=0)
        upper_distance = (delta_values - upper_bounds.view(-1, 1)).clamp(min=0)
        self.register_buffer('delta_outside_distance', lower_distance + upper_distance)

        # Conteo, delta, actividad, multimodalidad, total semana, 6 calendarios.
        self.per_step_dim = 11
        self.step_norm = nn.LayerNorm(self.per_step_dim)
        self.rnn = nn.GRU(
            input_size=self.per_step_dim,
            hidden_size=int(hidden_size),
            num_layers=max(1, int(num_layers)),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=True,
        )
        self.task_embed = nn.Embedding(self.num_tasks, int(task_embed_dim))
        fused_dim = int(hidden_size) * 4 + int(task_embed_dim) + 4
        self.context_proj = nn.Sequential(
            nn.Linear(fused_dim, int(hidden_size) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.delta_head = nn.Linear(int(hidden_size) * 2, self.delta_num_classes)
        self.zero_delta_logit_bias = float(zero_delta_logit_bias)

        self.num_candidate_sources = len(self.seasonal_lags)
        if self.use_recent_mean_candidate:
            self.num_candidate_sources += 1
        if self.use_last_count_candidate:
            self.num_candidate_sources += 1
        if self.use_task_median_candidate:
            self.num_candidate_sources += 1
        self.selector_head = nn.Sequential(
            nn.Linear(int(hidden_size) * 2, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(int(hidden_size), self.num_candidate_sources),
        )

    def fit(self, target_counts: torch.Tensor | None = None):
        if target_counts is None or target_counts.numel() == 0:
            self.task_median_counts.zero_()
            return self
        target_counts = target_counts.long().clamp(min=0, max=self.max_count_cap)
        self.task_median_counts.copy_(torch.median(target_counts, dim=0).values)
        return self

    def _split_sequence(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        base = sequence[:, :, : self.base_week_feature_dim]
        t = self.num_tasks
        counts = base[:, :, 0:t]
        active_days = base[:, :, 5 * t : 6 * t]
        day_multimodality = base[:, :, 6 * t : 7 * t]
        total_tasks = base[:, :, 14 * t : 14 * t + 1]
        calendar = base[:, :, 14 * t + 1 : 14 * t + 7]
        if self.seasonal_feature_dim > 0 and sequence.shape[-1] >= self.base_week_feature_dim + self.seasonal_feature_dim:
            extra = sequence[:, -1, self.base_week_feature_dim : self.base_week_feature_dim + self.seasonal_feature_dim]
            extra = extra.view(sequence.shape[0], len(self.seasonal_lags), 2, self.num_tasks)
            lag_counts = extra[:, :, 0, :].clamp(min=0.0, max=float(self.max_count_cap))
            lag_masks = extra[:, :, 1, :].clamp(min=0.0, max=1.0)
        else:
            lag_counts = torch.zeros(sequence.shape[0], len(self.seasonal_lags), self.num_tasks, dtype=sequence.dtype, device=sequence.device)
            lag_masks = torch.zeros_like(lag_counts)
        return {
            'counts': counts,
            'active_days': active_days,
            'day_multimodality': day_multimodality,
            'total_tasks': total_tasks,
            'calendar': calendar,
            'lag_counts': lag_counts,
            'lag_masks': lag_masks,
        }

    def _build_task_step_tensor(self, sequence: torch.Tensor) -> torch.Tensor:
        parts = self._split_sequence(sequence)
        counts = parts['counts']  # [B, W, T]
        batch_size, window_size, _ = counts.shape
        count_delta = torch.zeros_like(counts)
        if window_size > 1:
            count_delta[:, 1:, :] = counts[:, 1:, :] - counts[:, :-1, :]
        total_tasks = parts['total_tasks'].expand(-1, -1, self.num_tasks)
        calendar = parts['calendar'].unsqueeze(2).expand(-1, -1, self.num_tasks, -1)
        step_tensor = torch.stack(
            [
                counts,
                count_delta,
                parts['active_days'],
                parts['day_multimodality'],
                total_tasks,
            ],
            dim=-1,
        )
        step_tensor = torch.cat([step_tensor, calendar], dim=-1)  # [B, W, T, 11]
        return step_tensor.permute(0, 2, 1, 3).contiguous()  # [B, T, W, 11]

    def _build_candidates(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self._split_sequence(sequence)
        counts = parts['counts']
        batch_size = sequence.shape[0]
        candidates: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        for lag_idx in range(len(self.seasonal_lags)):
            candidates.append(parts['lag_counts'][:, lag_idx, :])
            masks.append(parts['lag_masks'][:, lag_idx, :])

        if self.use_recent_mean_candidate:
            recent_counts = counts[:, -self.recent_window :, :]
            candidates.append(recent_counts.mean(dim=1).round().clamp(min=0.0, max=float(self.max_count_cap)))
            masks.append(torch.ones(batch_size, self.num_tasks, dtype=sequence.dtype, device=sequence.device))

        if self.use_last_count_candidate:
            candidates.append(counts[:, -1, :].round().clamp(min=0.0, max=float(self.max_count_cap)))
            masks.append(torch.ones(batch_size, self.num_tasks, dtype=sequence.dtype, device=sequence.device))

        if self.use_task_median_candidate:
            median = self.task_median_counts.to(device=sequence.device, dtype=sequence.dtype).view(1, self.num_tasks).expand(batch_size, -1)
            candidates.append(median)
            masks.append(torch.ones(batch_size, self.num_tasks, dtype=sequence.dtype, device=sequence.device))

        candidate_counts = torch.stack(candidates, dim=-1).round().long().clamp(min=0, max=self.max_count_cap)
        candidate_masks = torch.stack(masks, dim=-1).to(dtype=sequence.dtype)
        return candidate_counts, candidate_masks

    def _apply_task_delta_range_penalty(self, residual_delta_logits: torch.Tensor) -> torch.Tensor:
        if self.delta_outside_range_logit_penalty_per_step == 0.0:
            return residual_delta_logits
        outside_distance = self.delta_outside_distance.to(
            device=residual_delta_logits.device,
            dtype=residual_delta_logits.dtype,
        )
        penalty = outside_distance.unsqueeze(0) * float(self.delta_outside_range_logit_penalty_per_step)
        return residual_delta_logits + penalty

    def _negative_fill_value(self, dtype: torch.dtype) -> float:
        if dtype in (torch.float16, torch.bfloat16):
            return -1e4
        return -1e9

    def _residual_absolute_logits(self, residual_delta_logits: torch.Tensor, baseline_counts: torch.Tensor) -> torch.Tensor:
        batch_size = residual_delta_logits.shape[0]
        neg_fill = self._negative_fill_value(residual_delta_logits.dtype)
        absolute_logits = torch.full(
            (batch_size, self.num_tasks, self.max_count_cap + 1),
            fill_value=neg_fill,
            dtype=residual_delta_logits.dtype,
            device=residual_delta_logits.device,
        )
        delta_values = self.delta_index_values.to(device=residual_delta_logits.device)
        absolute_indices = baseline_counts.unsqueeze(-1) + delta_values.view(1, 1, -1)
        valid = (absolute_indices >= 0) & (absolute_indices <= self.max_count_cap)
        clamped_indices = absolute_indices.clamp(min=0, max=self.max_count_cap)
        update = torch.where(valid, residual_delta_logits, torch.full_like(residual_delta_logits, neg_fill))
        absolute_logits.scatter_reduce_(2, clamped_indices, update, reduce='amax', include_self=True)
        return absolute_logits

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = sequence.shape[0]
        task_steps = self._build_task_step_tensor(sequence)
        _, _, window_size, feat_dim = task_steps.shape
        encoded_input = self.step_norm(task_steps.view(batch_size * self.num_tasks, window_size, feat_dim))
        outputs, hidden = self.rnn(encoded_input)
        pooled = outputs.mean(dim=1)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        parts = self._split_sequence(sequence)
        counts = parts['counts'].permute(0, 2, 1).contiguous()
        recent = counts[:, :, -self.recent_window :]
        recent_mean = recent.mean(dim=-1)
        recent_std = recent.std(dim=-1, unbiased=False)
        last_count = counts[:, :, -1]
        last_delta = last_count - counts[:, :, -2] if counts.shape[-1] > 1 else torch.zeros_like(last_count)
        summary = torch.stack([recent_mean, recent_std, last_count, last_delta], dim=-1).view(batch_size * self.num_tasks, 4)

        task_ids = torch.arange(self.num_tasks, device=sequence.device).view(1, self.num_tasks).expand(batch_size, -1).reshape(-1)
        task_emb = self.task_embed(task_ids)
        fused = torch.cat([pooled, last_hidden, task_emb, summary], dim=-1)
        fused = self.context_proj(fused)

        delta_logits = self.delta_head(fused).view(batch_size, self.num_tasks, self.delta_num_classes)
        zero_idx = int(self.max_count_cap)
        delta_logits[:, :, zero_idx] = delta_logits[:, :, zero_idx] + self.zero_delta_logit_bias
        delta_logits = self._apply_task_delta_range_penalty(delta_logits)

        candidate_counts, candidate_masks = self._build_candidates(sequence)
        selector_logits = self.selector_head(fused).view(batch_size, self.num_tasks, self.num_candidate_sources)
        invalid_penalty = torch.where(candidate_masks > 0, torch.zeros_like(candidate_masks), torch.full_like(candidate_masks, self._negative_fill_value(candidate_masks.dtype)))
        selector_logits = selector_logits + invalid_penalty

        absolute_logits = torch.full(
            (batch_size, self.num_tasks, self.max_count_cap + 1),
            fill_value=self._negative_fill_value(delta_logits.dtype),
            dtype=delta_logits.dtype,
            device=sequence.device,
        )
        for candidate_idx in range(self.num_candidate_sources):
            candidate_abs_logits = self._residual_absolute_logits(delta_logits, candidate_counts[:, :, candidate_idx])
            absolute_logits = torch.logaddexp(absolute_logits, candidate_abs_logits + selector_logits[:, :, candidate_idx].unsqueeze(-1))

        return {
            'logits': absolute_logits,
            'delta_logits': delta_logits,
            'selector_logits': selector_logits,
            'candidate_counts': candidate_counts,
            'candidate_masks': candidate_masks,
        }


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
