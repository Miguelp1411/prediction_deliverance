"""
slot_loss.py
────────────
Loss y métricas para el RoutinePredictor.

Diferencias clave respecto al sistema anterior:
  - La métrica principal es F1 de slot (no accuracy de posición)
  - La loss de ocurrencia usa pos_weight dinámico para manejar desbalance
    (los slots inactivos son mayoría → sin corrección el modelo predice siempre 0)
  - Las losses de timing sólo se calculan en slots que REALMENTE ocurren
  - Gaussian soft-CE cíclica para hora y minuto (cerca = parcialmente correcto)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MIN_BINS = 12


# ── Helpers ───────────────────────────────────────────────────────────────────

def _soft_cyclic_ce(logits: torch.Tensor, target: torch.Tensor,
                    n_classes: int, sigma: float = 1.0) -> torch.Tensor:
    """
    Cross-entropy suavizada con kernel gaussiano cíclico.
    'Casi acertar' recibe crédito parcial.

    logits : (N, n_classes)
    target : (N,)  long
    """
    if target.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    classes = torch.arange(n_classes, device=target.device).float()
    t    = target.float().unsqueeze(1)                          # (N, 1)
    diff = (classes.unsqueeze(0) - t).abs()                     # (N, n_classes)
    diff = torch.minimum(diff, n_classes - diff)                # cíclico
    w    = torch.exp(-0.5 * (diff / sigma) ** 2)
    w    = w / w.sum(dim=1, keepdim=True)                       # normalizar

    log_p = F.log_softmax(logits, dim=-1)
    return -(w * log_p).sum(dim=-1).mean()


# ── Loss principal ────────────────────────────────────────────────────────────

def compute_loss(preds, targets, slot_freq: torch.Tensor = None):
    """
    preds   : (occurs, hour, minute, duration)  del modelo
    targets : (batch, K, 4) → [ocurre, hour_norm, min_norm, dur_norm]
    slot_freq: (K,) frecuencia histórica de cada slot (opcional, mejora pos_weight)
    """
    occurs_pred, hour_pred, min_pred, dur_pred = preds

    tgt_occurs  = targets[:, :, 0]                                    # float 0/1
    tgt_hour    = (targets[:, :, 1] * 23).round().long().clamp(0, 23)
    tgt_min_bin = (targets[:, :, 2] * 59 / 5).round().long().clamp(0, 11)
    tgt_dur     = targets[:, :, 3]

    active = tgt_occurs > 0.5                                         # (batch, K)

    # ── 1. Loss de ocurrencia ─────────────────────────────────────────────────
    # pos_weight dinámico: evita que el modelo aprenda a predecir siempre 0
    n_pos = active.sum().float().clamp(min=1)
    n_neg = (~active).sum().float().clamp(min=1)
    pw    = (n_neg / n_pos).clamp(min=1.0, max=15.0)
    pos_weight = torch.ones_like(tgt_occurs)
    pos_weight[active] = pw

    loss_occurs = F.binary_cross_entropy_with_logits(
        occurs_pred, tgt_occurs, weight=pos_weight
    )

    # ── 2. Losses de timing (sólo slots activos) ──────────────────────────────
    if active.sum() == 0:
        return loss_occurs * 3.0

    h_active   = hour_pred[active]   # (N_active, 24)
    m_active   = min_pred[active]    # (N_active, 12)
    dur_active = dur_pred[active]    # (N_active,)

    loss_hour = _soft_cyclic_ce(h_active, tgt_hour[active],    n_classes=24, sigma=1.5)
    loss_min  = _soft_cyclic_ce(m_active, tgt_min_bin[active], n_classes=N_MIN_BINS, sigma=1.0)
    loss_dur  = F.mse_loss(dur_active, tgt_dur[active])

    return (
        loss_occurs * 3.0 +
        loss_hour   * 2.0 +
        loss_min    * 1.5 +
        loss_dur    * 0.5
    )


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets, slots=None):
    """
    Devuelve un dict con las métricas completas desglosadas en tres niveles:

    ① F1 Tarea       — ¿predijo las tareas correctas? (ignora el día)
                        "¿sabía que esta semana había que fregar?"
    ② F1 Slot        — ¿predijo tarea + día correcto?
                        "¿sabía que había que fregar el LUNES?"
    ③ Timing         — hora, minuto y duración (sólo en slots activos)

    slots: lista de (task_id, day_of_week) con len=K, necesario para ① y ②_task.
           Si es None, sólo se calculan las métricas de slot completo.
    """
    occurs_pred, hour_pred, min_pred, dur_pred = preds

    tgt_occurs  = targets[:, :, 0] > 0.5
    tgt_hour    = (targets[:, :, 1] * 23).round().long().clamp(0, 23)
    tgt_min_bin = (targets[:, :, 2] * 59 / 5).round().long().clamp(0, 11)
    tgt_dur     = targets[:, :, 3]

    # ── ① F1 de SLOT (tarea + día exacto) ─────────────────────────────────────
    pred_occurs = occurs_pred > 0.0   # threshold=0.5 en sigmoid

    tp = (pred_occurs & tgt_occurs).float().sum()
    fp = (pred_occurs & ~tgt_occurs).float().sum()
    fn = (~pred_occurs & tgt_occurs).float().sum()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall    = (tp / (tp + fn + 1e-8)).item()
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # ── ② F1 de TAREA sola (ignora el día) ────────────────────────────────────
    # Para cada batch, agrupamos los slots por task_id:
    # si CUALQUIER slot de esa tarea está activo → la tarea "ocurre"
    f1_task = f1  # fallback si no hay info de slots
    if slots is not None:
        import torch as _torch
        # Construir máscara (batch, num_unique_tasks) aggregando por task_id
        task_ids   = _torch.tensor([s[0] for s in slots], device=occurs_pred.device)
        unique_ids = task_ids.unique()

        batch = pred_occurs.shape[0]
        pred_task = _torch.zeros(batch, len(unique_ids), dtype=_torch.bool, device=occurs_pred.device)
        tgt_task  = _torch.zeros(batch, len(unique_ids), dtype=_torch.bool, device=occurs_pred.device)

        for col_i, uid in enumerate(unique_ids):
            mask = (task_ids == uid)           # slots de esta tarea (varios días)
            pred_task[:, col_i] = pred_occurs[:, mask].any(dim=1)
            tgt_task[:, col_i]  = tgt_occurs[:, mask].any(dim=1)

        tp_t = (pred_task & tgt_task).float().sum()
        fp_t = (pred_task & ~tgt_task).float().sum()
        fn_t = (~pred_task & tgt_task).float().sum()
        prec_t = (tp_t / (tp_t + fp_t + 1e-8)).item()
        rec_t  = (tp_t / (tp_t + fn_t + 1e-8)).item()
        f1_task = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-8)
    else:
        prec_t = rec_t = f1_task

    # ── Métricas de timing (sólo slots activos y correctamente predichos) ─────
    # Usamos los slots que REALMENTE ocurren para evaluar timing
    active = tgt_occurs

    if active.sum() > 0:
        pred_hour_cls = hour_pred.argmax(dim=-1)
        acc_hour = (pred_hour_cls[active] == tgt_hour[active]).float().mean().item() * 100

        pred_min_cls = min_pred.argmax(dim=-1)
        diff = (pred_min_cls[active] - tgt_min_bin[active]).abs()
        diff = torch.minimum(diff, N_MIN_BINS - diff)    # cíclico
        acc_min_exact = (diff == 0).float().mean().item() * 100
        acc_min_5     = (diff <= 1).float().mean().item() * 100   # ±1 bin = ±5 min

        mae_dur = torch.abs(dur_pred[active] - tgt_dur[active]).mean().item()
    else:
        acc_hour = acc_min_exact = acc_min_5 = mae_dur = 0.0

    return {
        # ① ¿Acertó qué TAREAS hay esta semana? (ignora el día)
        'f1_tarea'      : f1_task * 100,
        # ② ¿Acertó tarea + día exacto?
        'f1_slot'       : f1 * 100,
        'precision_slot': precision * 100,
        'recall_slot'   : recall * 100,
        # ③ Timing
        'acc_hour'      : acc_hour,
        'acc_min_exact' : acc_min_exact,
        'acc_min_5'     : acc_min_5,
        'mae_dur'       : mae_dur,
    }