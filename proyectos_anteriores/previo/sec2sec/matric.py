import torch
from proyectos_anteriores.previo.sec2sec.dataset_3 import N_MIN_BINS


def compute_metrics(preds, targets, mask):
    task_logits, day_logits, hour_logits, min_logits, duration = preds

    target_task = (targets[:, :, 0].long() - 1).clamp(min=0)
    target_day  = targets[:, :, 1].long()
    target_hour = targets[:, :, 2].long()
    target_min  = targets[:, :, 3].long()   # bins 0-11
    target_dur  = targets[:, :, 4]

    def masked_accuracy(logits, target, mask):
        preds_cls = logits.argmax(dim=-1)
        correct   = (preds_cls == target)[mask]
        return correct.float().mean().item() * 100 if correct.numel() > 0 else 0.0

    def masked_mae(pred, target, mask):
        return torch.abs(pred[mask] - target[mask]).mean().item() if mask.sum() > 0 else 0.0

    # Acc minuto en bins (1 bin = 5 min)
    acc_min_bin = masked_accuracy(min_logits, target_min, mask)

    # Tolerancia ±1 bin (±5 min): predecir bin adyacente también cuenta
    pred_bin  = min_logits.argmax(dim=-1)
    diff_bin  = (pred_bin - target_min).abs()
    diff_bin  = torch.minimum(diff_bin, N_MIN_BINS - diff_bin)   # cíclico
    within_1  = (diff_bin <= 1)[mask]
    acc_min_5 = within_1.float().mean().item() * 100 if within_1.numel() > 0 else 0.0

    return {
        'acc_task':        masked_accuracy(task_logits, target_task, mask),
        'acc_day':         masked_accuracy(day_logits,  target_day,  mask),
        'acc_hour':        masked_accuracy(hour_logits, target_hour, mask),
        'acc_minute_bin':  acc_min_bin,   # exacto al bin de 5 min
        'acc_minute_±5m':  acc_min_5,     # tolerancia ±5 min
        'mae_duration':    masked_mae(duration, target_dur, mask),
    }