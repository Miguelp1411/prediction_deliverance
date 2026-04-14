from __future__ import annotations

import time
from typing import Callable

import torch

from hybrid_schedule.evaluation.metrics import occurrence_metrics, temporal_metrics
from hybrid_schedule.training.losses import occurrence_loss, temporal_loss


OCC_EXCLUDE = {'target_count', 'change_target', 'delta_target', 'anchor_start', 'anchor_duration', 'target_start', 'target_duration', 'day_target', 'time_target'}
TMP_EXCLUDE = {'target_count', 'change_target', 'delta_target', 'anchor_start', 'anchor_duration', 'target_start', 'target_duration', 'day_target', 'time_target', 'template_count'}



def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}



def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row.keys()})
    return {k: float(sum(r.get(k, 0.0) for r in rows) / len(rows)) for k in keys}



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
    min_delta: float,
    model_name: str,
    logger,
    selector_fn: Callable | None = None,
    selector_every: int = 1,
):
    best_state = None
    best_key = None
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
        row = {
            'model': model_name,
            'epoch': epoch,
            'train_loss': float(sum(train_losses) / max(len(train_losses), 1)),
            'val_loss': float(sum(val_losses) / max(len(val_losses), 1)),
            'elapsed_sec': float(time.time() - t0),
        }
        row.update({f'train_{k}': v for k, v in train_metrics.items()})
        row.update(val_metrics)

        selector_metrics = {}
        selector_score = None
        if selector_fn is not None and epoch % max(1, int(selector_every)) == 0:
            selector_metrics = selector_fn(model)
            selector_score = float(selector_metrics.get('selector_score', 0.0))
            row.update(selector_metrics)

        logger.log_epoch(row)
        all_rows.append(row)
        metrics_str = ', '.join(f'{k}={v:.3f}' for k, v in val_metrics.items())
        selector_str = ''
        if selector_score is not None:
            selector_str = f" | selector_score={selector_score:.3f}"
        print(f'[{model_name}] epoch {epoch:03d} | train_loss={row["train_loss"]:.4f} | val_loss={row["val_loss"]:.4f} | {metrics_str}{selector_str}', flush=True)

        current_key = selector_score if selector_score is not None else -row['val_loss']
        if best_key is None or current_key > best_key + min_delta:
            best_key = current_key
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = dict(val_metrics)
            best_metrics.update(selector_metrics)
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metrics, all_rows



def fit_occurrence_model(model, train_loader, val_loader, optimizer, device, epochs, patience, min_delta, logger, max_delta: int, selector_fn: Callable | None = None, selector_every: int = 1):
    return _fit_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=lambda o, b: occurrence_loss(o, b),
        metrics_fn=lambda o, b: occurrence_metrics(o, b, max_delta=max_delta),
        forward_keys_exclude=OCC_EXCLUDE,
        device=device,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        model_name='occurrence_residual',
        logger=logger,
        selector_fn=selector_fn,
        selector_every=selector_every,
    )



def fit_temporal_model(model, train_loader, val_loader, optimizer, device, epochs, patience, min_delta, logger, bins_per_day: int, bin_minutes: int, selector_fn: Callable | None = None, selector_every: int = 1):
    return _fit_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=lambda o, b: temporal_loss(o, b),
        metrics_fn=lambda o, b: temporal_metrics(o, b, bins_per_day=bins_per_day, bin_minutes=bin_minutes),
        forward_keys_exclude=TMP_EXCLUDE,
        device=device,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        model_name='temporal_residual',
        logger=logger,
        selector_fn=selector_fn,
        selector_every=selector_every,
    )
