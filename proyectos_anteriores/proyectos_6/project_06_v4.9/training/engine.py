"""
Training engine — generic fit loop with early stopping, AMP, and logging.

Ported from v4.9 engine.py with multi-model support and YAML config.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader


@dataclass
class EarlyStoppingState:
    """Track training progress and best checkpoint."""
    best_metric: float = float("inf")
    best_epoch: int = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement: int = 0
    history: list[dict] = field(default_factory=list)
    stopped_early: bool = False
    total_epochs: int = 0
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")
    monitor_name: str = "val_loss"
    monitor_mode: str = "min"


def _autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: str = "float16"):
    if not amp_enabled:
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.autocast(device_type=dev_type, dtype=dtype)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    metrics_fn: Callable | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
) -> tuple[float, dict[str, float]]:
    """Train for one epoch. Returns (avg_loss, metrics_dict)."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_metrics: dict[str, float] = {}

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bs = next(iter(batch.values())).shape[0] if batch else 1

        optimizer.zero_grad()
        with _autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(**{k: v for k, v in batch.items() if k.startswith("input_")})
            loss = loss_fn(outputs, batch)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * bs
        total_samples += bs

        if metrics_fn is not None:
            with torch.no_grad():
                m = metrics_fn(outputs, batch)
                for k, v in m.items():
                    all_metrics[k] = all_metrics.get(k, 0.0) + v * bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {k: v / max(total_samples, 1) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    metrics_fn: Callable | None,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
) -> tuple[float, dict[str, float]]:
    """Evaluate for one epoch. Returns (avg_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_metrics: dict[str, float] = {}

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        bs = next(iter(batch.values())).shape[0] if batch else 1

        with _autocast_context(device, amp_enabled, amp_dtype):
            outputs = model(**{k: v for k, v in batch.items() if k.startswith("input_")})
            loss = loss_fn(outputs, batch)

        total_loss += loss.item() * bs
        total_samples += bs

        if metrics_fn is not None:
            m = metrics_fn(outputs, batch)
            for k, v in m.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v * bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {k: v / max(total_samples, 1) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


def _is_improvement(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


def fit_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: Callable,
    metrics_fn: Callable | None,
    device: torch.device,
    max_epochs: int,
    patience: int,
    model_name: str = "model",
    monitor_name: str = "val_loss",
    monitor_mode: str = "min",
    min_delta: float = 0.0,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
    verbose_every: int = 5,
    callback: Callable | None = None,
) -> EarlyStoppingState:
    """
    Full training loop with early stopping.

    Returns EarlyStoppingState with best model state and history.
    """
    state = EarlyStoppingState(
        monitor_name=monitor_name,
        monitor_mode=monitor_mode,
        best_metric=float("inf") if monitor_mode == "min" else float("-inf"),
    )

    scaler = None
    if amp_enabled and device.type == "cuda":
        scaler = torch.amp.GradScaler()

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, metrics_fn,
            device, scaler, amp_enabled, amp_dtype,
        )
        val_loss, val_metrics = evaluate_epoch(
            model, val_loader, loss_fn, metrics_fn,
            device, amp_enabled, amp_dtype,
        )

        elapsed = time.time() - t0

        # Determine monitor value
        if monitor_name == "val_loss":
            current_metric = val_loss
        elif monitor_name.startswith("val_"):
            key = monitor_name[4:]  # strip "val_"
            current_metric = val_metrics.get(key, val_loss)
        else:
            current_metric = val_metrics.get(monitor_name, val_loss)

        # Check improvement
        improved = _is_improvement(current_metric, state.best_metric, monitor_mode, min_delta)
        if improved:
            state.best_metric = current_metric
            state.best_epoch = epoch
            state.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_metric)
            else:
                scheduler.step()

        # Record history
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "monitor_value": current_metric,
            "improved": improved,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        state.history.append(entry)

        # Logging
        if epoch == 1 or epoch % verbose_every == 0 or improved:
            star = " ★" if improved else ""
            print(
                f"  [{model_name}] Epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                f"{monitor_name}={current_metric:.4f}{star} | "
                f"{elapsed:.1f}s"
            )

        # Callback
        if callback is not None:
            callback(epoch, entry)

        # Early stopping
        if state.epochs_without_improvement >= patience:
            print(f"  [{model_name}] Early stopping at epoch {epoch} (patience={patience})")
            state.stopped_early = True
            break

    state.total_epochs = len(state.history)
    state.final_train_loss = state.history[-1]["train_loss"] if state.history else float("nan")
    state.final_val_loss = state.history[-1]["val_loss"] if state.history else float("nan")

    # Restore best state
    if state.best_state is not None:
        model.load_state_dict(state.best_state)
        print(f"  [{model_name}] Restored best checkpoint from epoch {state.best_epoch}")

    return state
