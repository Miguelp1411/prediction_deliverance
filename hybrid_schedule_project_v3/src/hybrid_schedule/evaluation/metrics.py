from __future__ import annotations

import torch

from .matching import hungarian_match


@torch.no_grad()
def occurrence_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], max_delta: int) -> dict[str, float]:
    delta_values = torch.arange(-max_delta, max_delta + 1, device=batch['baseline_count'].device)
    pred_delta = delta_values[outputs['delta_logits'].argmax(dim=-1)]
    pred_count = torch.clamp(batch['baseline_count'] + pred_delta, min=0)
    target_count = batch['target_count']
    abs_err = (pred_count.float() - target_count.float()).abs()
    exact = (abs_err == 0).float().mean().item() * 100.0
    close1 = (abs_err <= 1).float().mean().item() * 100.0
    close2 = (abs_err <= 2).float().mean().item() * 100.0
    mae = abs_err.mean().item()
    change_pred = outputs['change_logits'].argmax(dim=-1)
    change_acc = (change_pred == batch['change_target']).float().mean().item() * 100.0
    expected_count_mae = (outputs['expected_count'] - batch['target_count'].float()).abs().mean().item()
    return {
        'count_exact_acc': exact,
        'close_acc_1': close1,
        'close_acc_2': close2,
        'count_mae': mae,
        'expected_count_mae': expected_count_mae,
        'change_acc': change_acc,
    }


@torch.no_grad()
def temporal_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], bins_per_day: int, local_offset_radius: int, bin_minutes: int) -> dict[str, float]:
    day_radius = int(batch['day_offset_radius'][0].item())
    local_radius = int(batch['local_offset_radius'][0].item())
    pred_day_idx = outputs['day_offset_logits'].argmax(dim=-1)
    pred_local_idx = outputs['local_offset_logits'].argmax(dim=-1)
    day_offsets = pred_day_idx.to(torch.int64) - day_radius
    local_offsets = pred_local_idx.to(torch.int64) - local_radius
    pred_start = batch['anchor_start'] + day_offsets * int(bins_per_day) + local_offsets
    max_start = 7 * bins_per_day - 1
    pred_start = pred_start.clamp(min=0, max=max_start)
    abs_error_bins = (pred_start - batch['target_start']).abs()
    duration_pred = batch['anchor_duration'] + outputs['duration_delta']
    duration_abs = (duration_pred - batch['target_duration']).abs()
    return {
        'start_exact_acc': (abs_error_bins == 0).float().mean().item() * 100.0,
        'start_tol_5m': (abs_error_bins <= 1).float().mean().item() * 100.0,
        'start_tol_10m': (abs_error_bins <= 2).float().mean().item() * 100.0,
        'start_mae_minutes': abs_error_bins.float().mean().item() * bin_minutes,
        'duration_mae_minutes': duration_abs.float().mean().item() * bin_minutes,
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
