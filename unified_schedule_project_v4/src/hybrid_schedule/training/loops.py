from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from hybrid_schedule.evaluation.metrics import unified_slot_metrics
from .losses import unified_slot_loss


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    return {k: float(sum(row.get(k, 0.0) for row in rows) / max(len(rows), 1)) for k in keys}


def _fmt_metric(value: float | None, suffix: str = '') -> str:
    if value is None:
        return 'n/a'
    return f'{float(value):.4f}{suffix}'


def _print_epoch_metrics(epoch: int, epochs: int, train_mean: dict[str, float], val_mean: dict[str, float]) -> None:
    train_loss = train_mean.get('loss')
    val_loss = val_mean.get('loss')
    train_f1 = train_mean.get('active_f1')
    val_f1 = val_mean.get('active_f1')
    train_count_mae = train_mean.get('count_mae')
    val_count_mae = val_mean.get('count_mae')
    train_start_mae = train_mean.get('start_mae_minutes')
    val_start_mae = val_mean.get('start_mae_minutes')

    print(
        '[entrenamiento] '
        f'Época {epoch}/{epochs} | '
        f'train_loss={_fmt_metric(train_loss)} | '
        f'val_loss={_fmt_metric(val_loss)} | '
        f'train_active_f1={_fmt_metric(train_f1, "%")} | '
        f'val_active_f1={_fmt_metric(val_f1, "%")} | '
        f'train_count_mae={_fmt_metric(train_count_mae)} | '
        f'val_count_mae={_fmt_metric(val_count_mae)} | '
        f'train_start_mae={_fmt_metric(train_start_mae, " min")} | '
        f'val_start_mae={_fmt_metric(val_start_mae, " min")}',
        flush=True,
    )


def _monitor_score(
    val_mean: dict[str, float],
    monitor_metric: str,
    monitor_weights: dict[str, float] | None = None,
) -> tuple[float, str]:
    weights = dict(monitor_weights or {})
    metric_key = str(monitor_metric)

    if metric_key.startswith('val_'):
        metric_key = metric_key[4:]

    def require_metric(key: str) -> float:
        if key.startswith('val_'):
            key = key[4:]
        if key not in val_mean:
            available = ', '.join(sorted(val_mean.keys()))
            raise KeyError(
                f'Métrica de monitorización no encontrada: "{key}". '
                f'Métricas disponibles: {available}'
            )
        return float(val_mean[key])

    if weights:
        score = 0.0
        for metric_name, weight in weights.items():
            score += float(weight) * require_metric(str(metric_name))
        return float(score), 'composite'

    return require_metric(metric_key), str(metric_key)


def _is_better(score: float, best_score: float, monitor_mode: str) -> bool:
    mode = str(monitor_mode).lower()
    if mode == 'max':
        return score > best_score + 1e-6
    return score < best_score - 1e-6


def fit_unified_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    logger,
    num_tasks: int,
    max_slots: int,
    bin_minutes: int,
    bins_per_day: int,
    grad_clip_norm: float | None = None,
    scheduler=None,
    loss_kwargs: dict[str, Any] | None = None,
    print_interval: int = 5,
    monitor_metric: str = 'start_tol_5m',
    monitor_mode: str = 'max',
    monitor_weights: dict[str, float] | None = None,
):
    loss_kwargs = dict(loss_kwargs or {})
    best_state = deepcopy(model.state_dict())
    mode = str(monitor_mode).lower()
    if mode not in {'min', 'max'}:
        raise ValueError('monitor_mode debe ser "min" o "max"')
    best_metric = float('-inf') if mode == 'max' else float('inf')
    best_metric_name = str(monitor_metric)
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improve = 0
    epochs_ran = 0

    for epoch in range(1, int(epochs) + 1):
        epochs_ran = epoch
        model.train()
        train_rows = []
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                history=batch['history'],
                task_ids=batch['task_ids'],
                database_ids=batch['database_ids'],
                robot_ids=batch['robot_ids'],
                slot_ids=batch['slot_ids'],
                anchor_days=batch['anchor_days'],
                anchor_times=batch['anchor_times'],
                numeric_features=batch['numeric_features'],
                query_mask=batch['query_mask'],
            )
            loss, components = unified_slot_loss(outputs, batch, num_tasks=num_tasks, max_slots=max_slots, **loss_kwargs)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            row = {'loss': float(loss.item())}
            row.update({k: float(v.item()) for k, v in components.items()})
            row.update(unified_slot_metrics(outputs, batch, num_tasks=num_tasks, max_slots=max_slots, bin_minutes=bin_minutes, bins_per_day=bins_per_day))
            train_rows.append(row)

        model.eval()
        val_rows = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                outputs = model(
                    history=batch['history'],
                    task_ids=batch['task_ids'],
                    database_ids=batch['database_ids'],
                    robot_ids=batch['robot_ids'],
                    slot_ids=batch['slot_ids'],
                    anchor_days=batch['anchor_days'],
                    anchor_times=batch['anchor_times'],
                    numeric_features=batch['numeric_features'],
                    query_mask=batch['query_mask'],
                )
                loss, components = unified_slot_loss(outputs, batch, num_tasks=num_tasks, max_slots=max_slots, **loss_kwargs)
                row = {'loss': float(loss.item())}
                row.update({k: float(v.item()) for k, v in components.items()})
                row.update(unified_slot_metrics(outputs, batch, num_tasks=num_tasks, max_slots=max_slots, bin_minutes=bin_minutes, bins_per_day=bins_per_day))
                val_rows.append(row)

        train_mean = _mean_metrics(train_rows)
        val_mean = _mean_metrics(val_rows)
        val_loss_for_scheduler = float(val_mean.get('loss', float('inf')))
        score, score_name = _monitor_score(
            val_mean=val_mean,
            monitor_metric=monitor_metric,
            monitor_weights=monitor_weights,
        )
        if scheduler is not None:
            scheduler.step(float(score))

        payload = {'epoch': epoch}
        payload.update({f'train_{k}': v for k, v in train_mean.items()})
        payload.update({f'val_{k}': v for k, v in val_mean.items()})
        payload['val_monitor_score'] = float(score)
        payload['lr'] = float(optimizer.param_groups[0]['lr'])
        logger.log_epoch(payload)

        interval = max(1, int(print_interval))
        if epoch % interval == 0 or epoch == int(epochs):
            _print_epoch_metrics(epoch, int(epochs), train_mean, val_mean)

        if _is_better(score, best_metric, mode):
            best_metric = score
            best_metric_name = score_name
            best_val_loss = float(val_mean.get('loss', float('inf')))
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= int(patience):
                print(
                    f'[entrenamiento] Early stopping en época {epoch}: '
                    f'sin mejora durante {int(patience)} épocas. '
                    f'Mejor época={best_epoch}, '
                    f'best_{best_metric_name}={best_metric:.4f}, '
                    f'best_val_loss={best_val_loss:.4f}',
                    flush=True,
                )
                break

    model.load_state_dict(best_state)
    return {
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_monitor_name': str(best_metric_name),
        'best_monitor_score': float(best_metric),
        'epochs_ran': int(epochs_ran),
    }
