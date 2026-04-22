from __future__ import annotations

import time
from typing import Callable

import torch

from hybrid_schedule.evaluation.metrics import occurrence_metrics, temporal_metrics, temporal_direct_metrics
from hybrid_schedule.training.losses import occurrence_loss, temporal_loss, temporal_direct_loss


OCC_EXCLUDE = {'target_count', 'change_target', 'delta_target', 'anchor_start', 'anchor_duration', 'target_start', 'target_duration', 'template_count'}
TMP_EXCLUDE = {
    'target_count',
    'change_target',
    'delta_target',
    'template_count',
    'slot_id',
    'baseline_pred_count',
    'anchor_start',
    'anchor_duration',
    'target_start',
    'target_duration',
    'candidate_starts',
    'candidate_durations',
    'candidate_costs',
    'candidate_target_probs',
}

TMP_DIRECT_EXCLUDE = {
    'baseline_pred_count',
    'anchor_start',
    'anchor_duration',
    'target_start',
    'target_duration',
    'target_day_idx',
    'target_time_bin_idx',
    'target_log_duration',
}


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    return {k: float(sum(r.get(k, 0.0) for r in rows) / len(rows)) for k in keys}


def _default_selection_score(model_name: str, metrics: dict[str, float], val_loss: float) -> float:
    if model_name == 'occurrence_residual':
        return float(metrics.get('count_exact_acc', 0.0) + 0.35 * metrics.get('close_acc_1', 0.0) - 8.0 * metrics.get('count_mae', 0.0) - 4.0 * metrics.get('expected_count_mae', 0.0) + 0.10 * metrics.get('change_acc', 0.0))
    return float(metrics.get('start_tol_5m', 0.0) + 0.5 * metrics.get('start_tol_10m', 0.0) - 0.08 * metrics.get('start_mae_minutes', 0.0) - 0.05 * metrics.get('duration_mae_minutes', 0.0))


def _fit_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn: Callable,
    metrics_fn: Callable,
    forward_keys_exclude: set[str],
    device: torch.device,
    epochs: int,
    patience: int,
    model_name: str,
    logger,
    grad_clip_norm: float | None = None,
    scheduler=None,
):
    best_state = None
    best_score = float('-inf')
    wait = 0
    best_metrics = {}
    all_rows = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        train_metrics_rows = []

        for batch in train_loader:
            batch = _move_batch(batch, device)
            kwargs = {k: batch[k] for k in batch.keys() if k not in forward_keys_exclude}
            outputs = model(**kwargs)
            loss = loss_fn(outputs, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()
            train_losses.append(float(loss.item()))
            train_metrics_rows.append(metrics_fn(outputs, batch))

        model.eval()
        val_losses = []
        val_metrics_rows = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                kwargs = {k: batch[k] for k in batch.keys() if k not in forward_keys_exclude}
                outputs = model(**kwargs)
                loss = loss_fn(outputs, batch)
                val_losses.append(float(loss.item()))
                val_metrics_rows.append(metrics_fn(outputs, batch))

        train_metrics = _mean_dict(train_metrics_rows)
        val_metrics = _mean_dict(val_metrics_rows)
        val_loss_mean = float(sum(val_losses) / max(len(val_losses), 1))
        if scheduler is not None:
            scheduler.step(val_loss_mean)
        selection_score = _default_selection_score(model_name, val_metrics, val_loss_mean)
        row = {
            'model': model_name,
            'epoch': epoch,
            'train_loss': float(sum(train_losses) / max(len(train_losses), 1)),
            'val_loss': val_loss_mean,
            'selection_score': selection_score,
            'elapsed_sec': float(time.time() - t0),
            'lr': float(optimizer.param_groups[0].get('lr', 0.0)),
        }
        row.update({f'train_{k}': v for k, v in train_metrics.items()})
        row.update(val_metrics)
        logger.log_epoch(row)
        all_rows.append(row)

        metrics_str = ', '.join(f'{k}={v:.3f}' for k, v in val_metrics.items())
        print(f'[{model_name}] epoch {epoch:03d} | train_loss={row["train_loss"]:.4f} | val_loss={row["val_loss"]:.4f} | lr={row["lr"]:.6f} | score={selection_score:.3f} | {metrics_str}', flush=True)

        if selection_score > best_score + 1e-5:
            best_score = selection_score
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(val_metrics)
            best_metrics['selection_score'] = float(selection_score)
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics, all_rows


def fit_occurrence_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    patience,
    logger,
    max_delta: int,
    change_loss_weight: float = 0.25,
    expected_count_mae_weight: float = 0.15,
    delta_reg_weight: float = 0.05,
    label_smoothing: float = 0.0,
    grad_clip_norm: float | None = None,
    scheduler=None,
):
    return _fit_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=lambda o, b: occurrence_loss(
            o,
            b,
            change_loss_weight=change_loss_weight,
            expected_count_mae_weight=expected_count_mae_weight,
            delta_reg_weight=delta_reg_weight,
            label_smoothing=label_smoothing,
        ),
        metrics_fn=lambda o, b: occurrence_metrics(o, b, max_delta=max_delta),
        forward_keys_exclude=OCC_EXCLUDE,
        device=device,
        epochs=epochs,
        patience=patience,
        model_name='occurrence_residual',
        logger=logger,
        grad_clip_norm=grad_clip_norm,
        scheduler=scheduler,
    )


def fit_temporal_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    patience,
    logger,
    bin_minutes: int,
    expected_cost_weight: float = 0.30,
    label_smoothing: float = 0.0,
    confidence_penalty_weight: float = 0.02,
    anchor_deviation_weight: float = 0.03,
    duration_deviation_weight: float = 0.01,
    grad_clip_norm: float | None = None,
    scheduler=None,
):
    return _fit_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=lambda o, b: temporal_loss(
            o,
            b,
            expected_cost_weight=expected_cost_weight,
            label_smoothing=label_smoothing,
            confidence_penalty_weight=confidence_penalty_weight,
            anchor_deviation_weight=anchor_deviation_weight,
            duration_deviation_weight=duration_deviation_weight,
            bin_minutes=bin_minutes,
        ),
        metrics_fn=lambda o, b: temporal_metrics(o, b, bin_minutes=bin_minutes),
        forward_keys_exclude=TMP_EXCLUDE,
        device=device,
        epochs=epochs,
        patience=patience,
        model_name='temporal_ranker',
        logger=logger,
        grad_clip_norm=grad_clip_norm,
        scheduler=scheduler,
    )



def fit_temporal_direct_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs,
    patience,
    logger,
    bin_minutes: int,
    bins_per_day: int,
    day_loss_weight: float = 0.80,
    time_loss_weight: float = 1.00,
    duration_loss_weight: float = 0.20,
    day_label_smoothing: float = 0.02,
    time_label_smoothing: float = 0.01,
    grad_clip_norm: float | None = None,
    scheduler=None,
):
    return _fit_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=lambda o, b: temporal_direct_loss(
            o,
            b,
            day_loss_weight=day_loss_weight,
            time_loss_weight=time_loss_weight,
            duration_loss_weight=duration_loss_weight,
            day_label_smoothing=day_label_smoothing,
            time_label_smoothing=time_label_smoothing,
        ),
        metrics_fn=lambda o, b: temporal_direct_metrics(o, b, bin_minutes=bin_minutes, bins_per_day=bins_per_day),
        forward_keys_exclude=TMP_DIRECT_EXCLUDE,
        device=device,
        epochs=epochs,
        patience=patience,
        model_name='temporal_direct',
        logger=logger,
        grad_clip_norm=grad_clip_norm,
        scheduler=scheduler,
    )
