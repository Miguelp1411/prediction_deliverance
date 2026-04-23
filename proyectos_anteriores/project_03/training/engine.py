from __future__ import annotations

from dataclasses import dataclass, field

import torch

from proyectos_anteriores.project_03.config import TEMPORAL_COUNT_NOISE_STD, VERBOSE_EVERY


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


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _average_metrics(metrics: dict[str, float], n_batches: int) -> dict[str, float]:
    return {k: v / max(n_batches, 1) for k, v in metrics.items()}


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
        occurrence_index=batch['occurrence_index'],
        history_features=batch['history_features'],
        predicted_count_norm=predicted_count_norm,
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


def train_epoch(model, loader, optimizer, loss_fn, metrics_fn, device):
    model.train()
    running_loss = 0.0
    running_metrics: dict[str, float] | None = None
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model_step(model, batch, training=True)
        loss = loss_step(loss_fn, batch, outputs)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        batch_metrics = metrics_step(metrics_fn, batch, outputs)
        if running_metrics is None:
            running_metrics = {k: 0.0 for k in batch_metrics}
        for key, value in batch_metrics.items():
            running_metrics[key] += float(value)
    return running_loss / max(len(loader), 1), _average_metrics(running_metrics or {}, len(loader))


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, metrics_fn, device):
    model.eval()
    running_loss = 0.0
    running_metrics: dict[str, float] | None = None
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        outputs = model_step(model, batch, training=False)
        loss = loss_step(loss_fn, batch, outputs)
        running_loss += float(loss.item())
        batch_metrics = metrics_step(metrics_fn, batch, outputs)
        if running_metrics is None:
            running_metrics = {k: 0.0 for k in batch_metrics}
        for key, value in batch_metrics.items():
            running_metrics[key] += float(value)
    return running_loss / max(len(loader), 1), _average_metrics(running_metrics or {}, len(loader))


def fit_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, metrics_fn, device, max_epochs, patience, model_name: str):
    state = EarlyStoppingState()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, metrics_fn, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, loss_fn, metrics_fn, device)
        scheduler.step(val_loss)

        state.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': dict(train_metrics),
            'val_metrics': dict(val_metrics),
        })
        state.final_train_loss = train_loss
        state.final_val_loss = val_loss
        state.final_train_metrics = dict(train_metrics)
        state.final_val_metrics = dict(val_metrics)

        if val_loss + 1e-6 < state.best_metric:
            state.best_metric = val_loss
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
            print(f'  train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')
            if train_metrics:
                print('  Train:', ' | '.join(f'{k}={v:.3f}' for k, v in train_metrics.items()))
            if val_metrics:
                print('  Val  :', ' | '.join(f'{k}={v:.3f}' for k, v in val_metrics.items()))

        if state.epochs_without_improvement >= patience:
            print(f'\n[{model_name}] Early stopping en epoch {epoch}. Mejor epoch={state.best_epoch}, best_val_loss={state.best_metric:.4f}')
            break

    if state.best_state is not None:
        model.load_state_dict(state.best_state)
    return state
