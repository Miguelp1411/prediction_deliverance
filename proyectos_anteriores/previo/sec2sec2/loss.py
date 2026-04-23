import torch
import torch.nn as nn
import torch.nn.functional as F
from proyectos_anteriores.previo.sec2sec2.dataset_3 import N_MIN_BINS


def _soft_labels(target, num_classes, sigma):
    """
    Gaussiano cíclico centrado en `target`.
    "Casi acertar" recibe crédito; fallar mucho recibe poco.
    target: (N,)  →  returns (N, num_classes)
    """
    classes = torch.arange(num_classes, device=target.device).float()
    t       = target.float().unsqueeze(1)
    diff    = (classes.unsqueeze(0) - t).abs()
    diff    = torch.minimum(diff, num_classes - diff)          # distancia cíclica
    w       = torch.exp(-0.5 * (diff / sigma) ** 2)
    return w / w.sum(dim=1, keepdim=True)


def _soft_ce(logits, target, mask, num_classes, sigma):
    """KL-divergence entre predicción y gaussiano cíclico, solo en posiciones reales."""
    logits_flat = logits[mask]
    target_flat = target[mask]
    if target_flat.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    soft  = _soft_labels(target_flat, num_classes, sigma)
    log_p = F.log_softmax(logits_flat, dim=-1)
    return -(soft * log_p).sum(dim=-1).mean()


def _masked_ce(logits, target, mask):
    logits_flat = logits[mask]
    target_flat = target[mask]
    if target_flat.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    return nn.CrossEntropyLoss()(logits_flat, target_flat)


def _masked_mse(pred, target, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return nn.MSELoss()(pred[mask], target[mask])


def compute_loss(preds, targets, mask):
    """
    Pesos equilibrados — task ya estaba bien, no necesita más peso que el resto:
      task   ×1.5  CE estándar   (no es ordinal)
      día    ×1.5  CE soft σ=0.8  (±1 día recibe crédito)
      hora   ×2.0  CE soft σ=1.5  (±1-2 h reciben crédito)
      minuto ×2.0  CE soft σ=1.5  (ahora son 12 bins de 5 min; σ=1.5 → ±1 bin = ±5-10 min)
      dur    ×0.5  MSE
    """
    task_logits, day_logits, hour_logits, min_logits, duration = preds

    target_task = (targets[:, :, 0].long() - 1).clamp(min=0)
    target_day  = targets[:, :, 1].long()
    target_hour = targets[:, :, 2].long()
    target_min  = targets[:, :, 3].long()   # ya es bin (0-11) desde el Dataset
    target_dur  = targets[:, :, 4]

    return (
        _masked_ce(task_logits, target_task, mask)                    * 1.5 +
        _soft_ce(day_logits,  target_day,  mask,  7,          0.8)   * 1.5 +
        _soft_ce(hour_logits, target_hour, mask, 24,          1.5)   * 2.0 +
        _soft_ce(min_logits,  target_min,  mask, N_MIN_BINS,  1.5)   * 2.0 +
        _masked_mse(duration, target_dur,  mask)                      * 0.5
    )