from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from proyectos_anteriores.proyectos_6.project_06_v4.config import GRAD_CLIP_MAX_NORM, TEMPORAL_COUNT_NOISE_STD, VERBOSE_EVERY


@dataclass
class EarlyStoppingState:
    best_metric: float = float('inf')
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0
    history: list[dict] = field(default_factory=list)
    best_train_loss: float = float('nan')
    best_val_loss: float = float('nan')
    best_train_metrics: dict[str, float] = field(default_factory=dict)
    best_val_metrics: dict[str, float] = field(default_factory=dict)
    final_train_loss: float = float('nan')
    final_val_loss: float = float('nan')
    final_train_metrics: dict[str, float] = field(default_factory=dict)
    final_val_metrics: dict[str, float] = field(default_factory=dict)
    monitor_name: str = 'val_loss'
    monitor_mode: str = 'min'


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _average_metrics(metrics: dict[str, float], total_samples: int) -> dict[str, float]:
    denominator = max(int(total_samples), 1)
    return {k: v / denominator for k, v in metrics.items()}


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    return torch.bfloat16 if str(amp_dtype).lower() == 'bfloat16' else torch.float16


def _autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: str):
    if amp_enabled and device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=_resolve_amp_dtype(amp_dtype))
    return nullcontext()


def _infer_batch_size(batch: dict[str, torch.Tensor]) -> int:
    if 'target_counts' in batch:
        return int(batch['target_counts'].shape[0])
    if 'task_id' in batch:
        return int(batch['task_id'].shape[0])
    if 'sequence' in batch:
        return int(batch['sequence'].shape[0])
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 1


def model_step(model, batch, training: bool = False):
    if 'target_counts' in batch:
        return model(batch['sequence'])
    predicted_count_norm = batch['predicted_count_norm']
    if training and TEMPORAL_COUNT_NOISE_STD > 0:
        noise = torch.randn_like(predicted_count_norm) * TEMPORAL_COUNT_NOISE_STD
        predicted_count_norm = (predicted_count_norm + noise).clamp(0.0, 1.0)
    return model(
        sequence=batch['sequence'],
        task_id=batch['task_id'],
        occurrence_slot=batch['occurrence_slot'],
        history_features=batch['history_features'],
        predicted_count_norm=predicted_count_norm,
        occurrence_progress=batch['occurrence_progress'],
        anchor_day=batch['anchor_day'],
    )


def loss_step(loss_fn, batch, outputs):
    if 'target_counts' in batch:
        return loss_fn(outputs, batch['target_counts'])
    return loss_fn(outputs, batch)


def metrics_step(metrics_fn, batch, outputs):
    if 'target_counts' in batch:
        return metrics_fn(outputs, batch['target_counts'])
    return metrics_fn(outputs, batch)


def train_epoch(model, loader, optimizer, loss_fn, metrics_fn, device, scaler=None, amp_enabled: bool = False, amp_dtype: str = 'float16'):
    model.train()
    running_loss = 0.0
    total_samples = 0
    running_metrics: dict[str, float] | None = None

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        batch_size = _infer_batch_size(batch)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled, amp_dtype):
            outputs = model_step(model, batch, training=True)
            loss = loss_step(loss_fn, batch, outputs)
        if scaler is not None and amp_enabled and device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            optimizer.step()

        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        batch_metrics = metrics_step(metrics_fn, batch, outputs)
        if running_metrics is None:
            running_metrics = {k: 0.0 for k in batch_metrics}
        for key, value in batch_metrics.items():
            running_metrics[key] += float(value) * batch_size

    denominator = max(total_samples, 1)
    return running_loss / denominator, _average_metrics(running_metrics or {}, denominator)


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, metrics_fn, device, amp_enabled: bool = False, amp_dtype: str = 'float16'):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    running_metrics: dict[str, float] | None = None

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        batch_size = _infer_batch_size(batch)
        with _autocast_context(device, amp_enabled, amp_dtype):
            outputs = model_step(model, batch, training=False)
            loss = loss_step(loss_fn, batch, outputs)

        running_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        batch_metrics = metrics_step(metrics_fn, batch, outputs)
        if running_metrics is None:
            running_metrics = {k: 0.0 for k in batch_metrics}
        for key, value in batch_metrics.items():
            running_metrics[key] += float(value) * batch_size

    denominator = max(total_samples, 1)
    return running_loss / denominator, _average_metrics(running_metrics or {}, denominator)


def _get_monitor_value(val_loss: float, val_metrics: dict[str, float], monitor_name: str | None) -> float:
    if not monitor_name or monitor_name == 'val_loss':
        return float(val_loss)
    return float(val_metrics.get(monitor_name, float('nan')))


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == 'max':
        return current > best + min_delta
    return current < best - min_delta


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    metrics_fn,
    device,
    max_epochs,
    patience,
    model_name: str,
    monitor_name: str | None = None,
    monitor_mode: str = 'min',
    min_delta: float = 0.0,
    extra_val_evaluator=None,
    extra_val_evaluator_every: int = 1,
    amp_enabled: bool = False,
    amp_dtype: str = 'float16',
):
    state = EarlyStoppingState(
        best_metric=(-float('inf') if monitor_mode == 'max' else float('inf')),
        monitor_name=(monitor_name or 'val_loss'),
        monitor_mode=monitor_mode,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_enabled and device.type == 'cuda'))
    extra_every = max(1, int(extra_val_evaluator_every))

    for epoch in range(1, max_epochs + 1):
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            metrics_fn,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        val_loss, val_metrics = evaluate_epoch(
            model,
            val_loader,
            loss_fn,
            metrics_fn,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        should_run_extra_eval = False
        if extra_val_evaluator is not None:
            monitor_requires_extra = bool(monitor_name) and monitor_name not in {'val_loss'} and monitor_name not in val_metrics
            should_run_extra_eval = (epoch == 1 or epoch % extra_every == 0 or monitor_requires_extra)
        if extra_val_evaluator is not None and should_run_extra_eval:
            extra_metrics = extra_val_evaluator(model)
            if extra_metrics:
                val_metrics = {**val_metrics, **extra_metrics}
        monitor_value = _get_monitor_value(val_loss, val_metrics, monitor_name)
        if scheduler is not None:
            try:
                scheduler.step(monitor_value)
            except TypeError:
                scheduler.step()

        state.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': dict(train_metrics),
            'val_metrics': dict(val_metrics),
            'monitor_value': monitor_value,
        })
        state.final_train_loss = train_loss
        state.final_val_loss = val_loss
        state.final_train_metrics = dict(train_metrics)
        state.final_val_metrics = dict(val_metrics)

        if _is_improvement(monitor_value, state.best_metric, monitor_mode, min_delta):
            state.best_metric = monitor_value
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            state.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            state.best_train_loss = train_loss
            state.best_val_loss = val_loss
            state.best_train_metrics = dict(train_metrics)
            state.best_val_metrics = dict(val_metrics)
        else:
            state.epochs_without_improvement += 1

        if epoch == 1 or epoch % VERBOSE_EVERY == 0:
            print(f'\n[{model_name}] Epoch {epoch}')
            print(f'  train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {state.monitor_name}={monitor_value:.4f}')
            if train_metrics:
                print('  Train:', ' | '.join(f'{k}={v:.3f}' for k, v in train_metrics.items()))
            if val_metrics:
                print('  Val  :', ' | '.join(f'{k}={v:.3f}' for k, v in val_metrics.items()))

        if state.epochs_without_improvement >= patience:
            print(
                f'\n[{model_name}] Early stopping en epoch {epoch}. '
                f'Mejor epoch={state.best_epoch}, mejor {state.monitor_name}={state.best_metric:.4f}'
            )
            break

    if state.best_state is not None:
        model.load_state_dict(state.best_state)
    return state
