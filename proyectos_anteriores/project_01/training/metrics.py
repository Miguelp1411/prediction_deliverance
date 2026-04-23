# training/metrics.py
# Métricas de evaluación para el modelo seq2seq.

import torch
from proyectos_anteriores.project_01.config import N_MIN_BINS

# Índices de las etiquetas enteras en el tensor y (cols 8-10)
_COL_DAY = 8   # day_of_week  (0-6)
_COL_HR  = 9   # hour         (0-23)
_COL_MIN = 10  # minute_bin   (0-11)


def compute_metrics(preds, targets, mask) -> dict[str, float]:
    """
    Calcula las métricas sobre las posiciones no-padding (mask=True).

    Devuelve un dict con:
      acc_task        — exactitud de tarea (%)
      acc_day         — exactitud de día (%)
      acc_hour        — exactitud de hora (%)
      acc_minute_bin  — exactitud de bin de minuto exacto (%)
      acc_minute_±5m  — exactitud de minuto con tolerancia ±5 min (%)
      mae_duration    — MAE de duración normalizada
    """
    task_logits, day_logits, hour_logits, min_logits, duration = preds

    target_task = (targets[:, :, 0].long() - 1).clamp(min=0)
    target_day  = targets[:, :, _COL_DAY].long()   # day_of_week entero (0-6)
    target_hour = targets[:, :, _COL_HR].long()    # hour entero (0-23)
    target_min  = targets[:, :, _COL_MIN].long()   # minute_bin entero (0-11)
    target_dur  = targets[:, :, 7]                 # dur_norm (float, col 7)

    def _accuracy(logits, target, mask) -> float:
        correct = (logits.argmax(dim=-1) == target)[mask]
        return correct.float().mean().item() * 100 if correct.numel() > 0 else 0.0

    def _mae(pred, target, mask) -> float:
        return torch.abs(pred[mask] - target[mask]).mean().item() if mask.sum() > 0 else 0.0

    # Exactitud de bin de minuto
    acc_min_bin = _accuracy(min_logits, target_min, mask)

    # Tolerancia ±1 bin (≈ ±5 min), con distancia cíclica
    pred_bin = min_logits.argmax(dim=-1)
    diff_bin = (pred_bin - target_min).abs()
    diff_bin = torch.minimum(diff_bin, N_MIN_BINS - diff_bin)
    within_1 = (diff_bin <= 1)[mask]
    acc_min_5 = within_1.float().mean().item() * 100 if within_1.numel() > 0 else 0.0

    return {
        "acc_task":       _accuracy(task_logits, target_task, mask),
        "acc_day":        _accuracy(day_logits,  target_day,  mask),
        "acc_hour":       _accuracy(hour_logits, target_hour, mask),
        "acc_minute_bin": acc_min_bin,
        "acc_minute_±5m": acc_min_5,
        "mae_duration":   _mae(duration, target_dur, mask),
    }