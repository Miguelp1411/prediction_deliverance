# model/predictor.py
# WeekPredictor: ensambla encoder + decoder con teacher forcing opcional.

import math
import torch
import torch.nn as nn

from proyectos_anteriores.project_01.config import MAX_TASKS, N_MIN_BINS
from proyectos_anteriores.project_01.model.encoder import WeekEncoder
from proyectos_anteriores.project_01.model.decoder import WeekDecoder

# Las primeras 8 columnas de los tensores X e y son las features del modelo.
# Las columnas 8-10 de y son etiquetas enteras (day_of_week, hour, minute_bin)
# usadas únicamente por loss.py y metrics.py.
_FEAT_COLS = 8


class WeekPredictor(nn.Module):
    """
    Modelo seq2seq completo.

    Durante el entrenamiento se usa teacher forcing (teacher_forcing_ratio > 0):
    el decoder recibe el token real (primeras 8 cols) en lugar del predicho,
    estabilizando el aprendizaje.

    Durante la validación / inferencia se pasa target=None y ratio=0.0 para que
    el modelo use sus propias predicciones (condición real de producción).
    """

    def __init__(self, num_tasks: int, embed_dim: int = 32,
                 hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.encoder = WeekEncoder(num_tasks, embed_dim, hidden_size, num_layers)
        self.decoder = WeekDecoder(num_tasks, embed_dim, hidden_size, num_layers)

    def forward(
        self,
        x: torch.Tensor,                    # (batch, num_weeks, max_tasks, 8)
        masks_X: torch.Tensor,              # (batch, num_weeks, max_tasks)
        target: torch.Tensor | None = None, # (batch, max_tasks, 11) — None en inferencia
        teacher_forcing_ratio: float = 1.0,
    ):
        context_h, context_c = self.encoder(x, masks_X)

        batch_size = x.shape[0]
        outputs    = []

        # Token de inicio: vector de ceros con solo las 8 feature cols
        decoder_input = torch.zeros(batch_size, 1, _FEAT_COLS, device=x.device)
        h, c = context_h, context_c

        for t in range(MAX_TASKS):
            task_out, day_out, hour_out, min_out, dur_out, h, c = \
                self.decoder(decoder_input, h, c)
            outputs.append((task_out, day_out, hour_out, min_out, dur_out))

            use_teacher = (
                target is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            )

            if use_teacher:
                # Solo las 8 columnas de features — las cols 8-10 son etiquetas,
                # no son una entrada válida para el decoder.
                decoder_input = target[:, t : t + 1, :_FEAT_COLS]
            else:
                decoder_input = self._build_next_input(
                    task_out, day_out, hour_out, min_out, dur_out, x.device
                )

        return (
            torch.cat([o[0] for o in outputs], dim=1),  # task   (batch, MAX_TASKS, num_tasks)
            torch.cat([o[1] for o in outputs], dim=1),  # day    (batch, MAX_TASKS, 7)
            torch.cat([o[2] for o in outputs], dim=1),  # hour   (batch, MAX_TASKS, 24)
            torch.cat([o[3] for o in outputs], dim=1),  # minute (batch, MAX_TASKS, N_MIN_BINS)
            torch.cat([o[4] for o in outputs], dim=1),  # dur    (batch, MAX_TASKS)
        )

    # ── Helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_next_input(
        task_out, day_out, hour_out, min_out, dur_out, device
    ) -> torch.Tensor:
        """Construye el token de entrada del siguiente paso a partir de predicciones."""
        pred_task_id  = task_out[:, 0, :].argmax(-1).float() + 1
        pred_day_idx  = day_out[:, 0, :].argmax(-1).float()
        pred_hour_idx = hour_out[:, 0, :].argmax(-1).float()
        pred_min_idx  = min_out[:, 0, :].argmax(-1).float()
        pred_dur      = dur_out[:, 0]

        sin_d = torch.sin(2 * math.pi * pred_day_idx  / 7)
        cos_d = torch.cos(2 * math.pi * pred_day_idx  / 7)
        sin_h = torch.sin(2 * math.pi * pred_hour_idx / 24)
        cos_h = torch.cos(2 * math.pi * pred_hour_idx / 24)
        sin_m = torch.sin(2 * math.pi * pred_min_idx  / N_MIN_BINS)
        cos_m = torch.cos(2 * math.pi * pred_min_idx  / N_MIN_BINS)

        # (batch, 8) → (batch, 1, 8)
        return torch.stack(
            [pred_task_id, sin_d, cos_d, sin_h, cos_h, sin_m, cos_m, pred_dur],
            dim=-1,
        ).unsqueeze(1)