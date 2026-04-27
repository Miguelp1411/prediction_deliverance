from __future__ import annotations

import torch

from .matching import hungarian_match


@torch.no_grad()
def unified_slot_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    num_tasks: int,
    max_slots: int,
    bin_minutes: int,
    bins_per_day: int,
    threshold: float = 0.50,
) -> dict[str, float]:
    valid_mask = batch['query_mask']
    active_probs = torch.sigmoid(outputs['active_logits'])
    active_pred_bool = (active_probs >= float(threshold)) & valid_mask
    active_true_bool = (batch['active_targets'] > 0.5) & valid_mask

    tp = float((active_pred_bool & active_true_bool).sum().item())
    fp = float((active_pred_bool & (~active_true_bool)).sum().item())
    fn = float(((~active_pred_bool) & active_true_bool).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)

    if 'week_time_logits' in outputs:
        pred_start = outputs['week_time_logits'].argmax(dim=-1).long()
        pred_day = torch.div(pred_start, int(bins_per_day), rounding_mode='floor')
        pred_time = pred_start % int(bins_per_day)
    else:
        pred_day = outputs['day_logits'].argmax(dim=-1)
        pred_time = outputs['time_logits'].argmax(dim=-1)
        pred_start = pred_day.long() * int(bins_per_day) + pred_time.long()
    positive_mask = active_true_bool
    if positive_mask.any():
        diff = (pred_start[positive_mask] - batch['target_start'][positive_mask]).abs().float()
        pred_duration = torch.clamp(torch.round(torch.expm1(outputs['pred_log_duration'][positive_mask])), min=1.0)
        dur_diff = (pred_duration - batch['target_duration'][positive_mask]).abs().float()
        start_mae = diff.mean().item() * bin_minutes
        start_tol_5m = (diff <= 1).float().mean().item() * 100.0
        day_acc = (pred_day[positive_mask] == batch['target_day_idx'][positive_mask]).float().mean().item() * 100.0
        dur_mae = dur_diff.mean().item() * bin_minutes
    else:
        start_mae = 0.0
        start_tol_5m = 0.0
        day_acc = 0.0
        dur_mae = 0.0

    pred_counts = (
        active_probs.view(-1, num_tasks, max_slots)
        * batch['query_mask'].view(-1, num_tasks, max_slots).float()
    ).sum(dim=-1)
    true_counts = batch['true_counts']
    count_mae = (pred_counts - true_counts).abs().mean().item()

    return {
        'active_precision': precision * 100.0,
        'active_recall': recall * 100.0,
        'active_f1': f1 * 100.0,
        'count_mae': count_mae,
        'start_mae_minutes': start_mae,
        'start_tol_5m': start_tol_5m,
        'day_acc': day_acc,
        'duration_mae_minutes': dur_mae,
    }


def _overlap_count(events: list[dict]) -> int:
    ordered = sorted(events, key=lambda x: (x['robot_id'], x['start_bin'], x['duration_bins']))
    count = 0
    last_end_by_robot: dict[str, int] = {}
    for evt in ordered:
        robot = evt['robot_id']
        start = int(evt['start_bin'])
        end = start + int(evt['duration_bins'])
        if robot in last_end_by_robot and start < last_end_by_robot[robot]:
            count += 1
        last_end_by_robot[robot] = max(last_end_by_robot.get(robot, 0), end)
    return count


def evaluate_week(pred_events: list[dict], true_events: list[dict], bin_minutes: int = 5) -> dict[str, float]:
    matches = hungarian_match(pred_events, true_events)
    tp = len(matches)
    fp = max(0, len(pred_events) - tp)
    fn = max(0, len(true_events) - tp)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    exact = 0
    close5 = 0
    close10 = 0
    duration_close = 0
    abs_err_sum = 0.0
    for pred, truth, _ in matches:
        start_diff = abs(int(pred['start_bin']) - int(truth['start_bin']))
        dur_diff = abs(int(pred['duration_bins']) - int(truth['duration_bins']))
        exact += int(start_diff == 0)
        close5 += int(start_diff <= 1)
        close10 += int(start_diff <= 2)
        duration_close += int(dur_diff <= 1)
        abs_err_sum += float(start_diff * bin_minutes)

    denom = max(tp, 1)
    return {
        'task_precision': precision * 100.0,
        'task_recall': recall * 100.0,
        'task_f1': f1 * 100.0,
        'time_exact_accuracy': exact / denom * 100.0,
        'time_close_accuracy_5m': close5 / denom * 100.0,
        'time_close_accuracy_10m': close10 / denom * 100.0,
        'duration_close_accuracy': duration_close / denom * 100.0,
        'start_mae_minutes': abs_err_sum / denom,
        'overlap_same_robot_count': float(_overlap_count(pred_events)),
        'predicted_tasks': float(len(pred_events)),
        'true_tasks': float(len(true_events)),
        'matched_tasks': float(tp),
    }
