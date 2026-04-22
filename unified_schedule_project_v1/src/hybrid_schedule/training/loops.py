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
):
    loss_kwargs = dict(loss_kwargs or {})
    best_state = deepcopy(model.state_dict())
    best_metric = float('inf')
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, int(epochs) + 1):
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
        score = float(val_mean.get('loss', float('inf')))
        if scheduler is not None:
            scheduler.step(score)

        payload = {'epoch': epoch}
        payload.update({f'train_{k}': v for k, v in train_mean.items()})
        payload.update({f'val_{k}': v for k, v in val_mean.items()})
        logger.log_epoch(payload)

        if score < best_metric - 1e-6:
            best_metric = score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= int(patience):
                break

    model.load_state_dict(best_state)
    return {
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_metric),
    }
