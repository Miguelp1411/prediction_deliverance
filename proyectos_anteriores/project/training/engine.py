from __future__ import annotations

from dataclasses import dataclass

import torch

from proyectos_anteriores.project.config import VERBOSE_EVERY


@dataclass
class EarlyStoppingState:
    best_metric: float = float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    best_state: dict[str, torch.Tensor] | None = None



def _average_metrics(sum_metrics: dict[str, float], steps: int) -> dict[str, float]:
    if steps == 0:
        return {k: 0.0 for k in sum_metrics}
    return {k: v / steps for k, v in sum_metrics.items()}



def train_epoch(model, loader, optimizer, loss_fn, metrics_fn, device):
    model.train()
    running_loss = 0.0
    running_metrics: dict[str, float] | None = None
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model_step(model, batch)
        loss = loss_step(loss_fn, batch, outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += float(loss.item())
        batch_metrics = metrics_step(metrics_fn, batch, outputs)
        if running_metrics is None:
            running_metrics = {k: 0.0 for k in batch_metrics}
        for key, value in batch_metrics.items():
            running_metrics[key] += float(value)

    if running_metrics is None:
        running_metrics = {}
    return running_loss / max(len(loader), 1), _average_metrics(running_metrics, len(loader))


@torch.no_grad()
def evaluate_epoch(model, loader, loss_fn, metrics_fn, device):
    model.eval()
    running_loss = 0.0
    running_metrics: dict[str, float] | None = None
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model_step(model, batch)
        loss = loss_step(loss_fn, batch, outputs)
        running_loss += float(loss.item())
        batch_metrics = metrics_step(metrics_fn, batch, outputs)
        if running_metrics is None:
            running_metrics = {k: 0.0 for k in batch_metrics}
        for key, value in batch_metrics.items():
            running_metrics[key] += float(value)

    if running_metrics is None:
        running_metrics = {}
    return running_loss / max(len(loader), 1), _average_metrics(running_metrics, len(loader))



def fit_model(model, train_loader, val_loader, optimizer, scheduler, loss_fn, metrics_fn, device, max_epochs, patience, model_name: str):
    state = EarlyStoppingState()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, metrics_fn, device)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, loss_fn, metrics_fn, device)
        scheduler.step(val_loss)

        if val_loss + 1e-6 < state.best_metric:
            state.best_metric = val_loss
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            state.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state.epochs_without_improvement += 1

        if epoch == 1 or epoch % VERBOSE_EVERY == 0:
            print(f"\n[{model_name}] Epoch {epoch}")
            print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
            if train_metrics:
                print("  Train:", " | ".join(f"{k}={v:.3f}" for k, v in train_metrics.items()))
            if val_metrics:
                print("  Val  :", " | ".join(f"{k}={v:.3f}" for k, v in val_metrics.items()))

        if state.epochs_without_improvement >= patience:
            print(
                f"\n[{model_name}] Early stopping en epoch {epoch}. "
                f"Mejor epoch={state.best_epoch}, best_val_loss={state.best_metric:.4f}"
            )
            break

    if state.best_state is not None:
        model.load_state_dict(state.best_state)
    return state



def model_step(model, batch):
    if "target_counts" in batch:
        return model(batch["sequence"])
    return model(
        sequence=batch["sequence"],
        task_id=batch["task_id"],
        occurrence_index=batch["occurrence_index"],
        history_features=batch["history_features"],
        predicted_count_norm=batch["predicted_count_norm"],
    )



def loss_step(loss_fn, batch, outputs):
    if "target_counts" in batch:
        return loss_fn(outputs, batch["target_counts"])
    start_logits, pred_duration = outputs
    return loss_fn(start_logits, batch["target_start_bin"], pred_duration, batch["target_duration_norm"])



def metrics_step(metrics_fn, batch, outputs):
    if "target_counts" in batch:
        return metrics_fn(outputs, batch["target_counts"])
    start_logits, pred_duration = outputs
    return metrics_fn(start_logits, batch["target_start_bin"], pred_duration, batch["target_duration_norm"])
