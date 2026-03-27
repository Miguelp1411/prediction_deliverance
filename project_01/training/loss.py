# training/loss.py
# Función de pérdida multi-tarea con soft labels para outputs ordinales.

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_MIN_BINS

# Índices de las etiquetas enteras en el tensor y (cols 8-10)
_COL_DAY = 8   # day_of_week  (0-6)
_COL_HR  = 9   # hour         (0-23)
_COL_MIN = 10  # minute_bin   (0-11)


# ── Soft labels ───────────────────────────────────────────────────────────────

def _soft_labels(target: torch.Tensor, num_classes: int, sigma: float) -> torch.Tensor:
    """
    Gaussiano cíclico centrado en `target`.
    Predecir clases adyacentes recibe crédito parcial; fallar mucho recibe poco.

    target : (N,)  →  returns (N, num_classes)
    """
    classes = torch.arange(num_classes, device=target.device).float()
    t       = target.float().unsqueeze(1)
    diff    = (classes.unsqueeze(0) - t).abs()
    diff    = torch.minimum(diff, num_classes - diff)          # distancia cíclica
    w       = torch.exp(-0.5 * (diff / sigma) ** 2)
    return w / w.sum(dim=1, keepdim=True)


def _soft_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    sigma: float,
) -> torch.Tensor:
    """KL-divergence entre predicción y gaussiano cíclico, solo en posiciones reales."""
    logits_flat = logits[mask]
    target_flat = target[mask]
    if target_flat.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    soft  = _soft_labels(target_flat, num_classes, sigma)
    log_p = F.log_softmax(logits_flat, dim=-1)
    return -(soft * log_p).sum(dim=-1).mean()


def _masked_ce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    logits_flat = logits[mask]
    target_flat = target[mask]
    if target_flat.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    return nn.CrossEntropyLoss()(logits_flat, target_flat)


def _masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return nn.MSELoss()(pred[mask], target[mask])


# ── Pérdida principal ─────────────────────────────────────────────────────────

def compute_loss(preds, targets, mask) -> torch.Tensor:
    """
    Pérdida multi-tarea con pesos equilibrados.

    Los targets de clasificación (día, hora, minuto) se leen de las columnas
    8-10 del tensor y, que contienen los índices enteros originales.
    Antes (col 1-3) se leían sin_day/cos_day/sin_hour (floats entre -1 y 1)
    que al convertirse a long daban 0 casi siempre, produciendo un accuracy
    espurio del 100 % en día, hora y minuto.

      task   ×1.5  CE estándar          (no es ordinal)
      día    ×1.5  CE soft  σ=0.8       (±1 día recibe crédito)
      hora   ×2.0  CE soft  σ=1.5       (±1-2 h reciben crédito)
      minuto ×2.0  CE soft  σ=1.5       (12 bins de 5 min; ±1 bin ≈ ±5-10 min)
      dur    ×0.5  MSE
    """
    task_logits, day_logits, hour_logits, min_logits, duration = preds

    target_task = (targets[:, :, 0].long() - 1).clamp(min=0)
    target_day  = targets[:, :, _COL_DAY].long()   # day_of_week entero (0-6)
    target_hour = targets[:, :, _COL_HR].long()    # hour entero (0-23)
    target_min  = targets[:, :, _COL_MIN].long()   # minute_bin entero (0-11)
    target_dur  = targets[:, :, 7]                 # dur_norm (float, col 7)

    # Pesos ajustados:
    # - task ×2.0   (objetivo principal, más señal)
    # - día  ×2.0   (muy predecible, buen ancla para el modelo)
    # - hora ×2.0   (importante y aprendible)
    # - min  ×0.5   (muy ruidoso, reducir peso para no confundir el gradiente)
    # - dur  ×0.5   (regresión, escala distinta)
    return (
        _masked_ce(task_logits, target_task, mask)                   * 2.0
        + _soft_ce(day_logits,  target_day,  mask,  7,         0.8) * 2.0
        + _soft_ce(hour_logits, target_hour, mask, 24,         1.5) * 2.0
        + _soft_ce(min_logits,  target_min,  mask, N_MIN_BINS, 1.5) * 0.5
        + _masked_mse(duration, target_dur, mask)                    * 0.5
    )