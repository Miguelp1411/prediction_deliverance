from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    DATA_PATH,
    DEVICE,
    OCC_BATCH_SIZE,
    OCC_DROPOUT,
    OCC_HIDDEN_SIZE,
    OCC_LR,
    OCC_MAX_EPOCHS,
    OCC_NUM_LAYERS,
    OCC_PATIENCE,
    OCC_WEIGHT_DECAY,
    SEED,
    TIMEZONE,
    TMP_BATCH_SIZE,
    TMP_DROPOUT,
    TMP_HIDDEN_SIZE,
    TMP_LR,
    TMP_MAX_EPOCHS,
    TMP_NUM_LAYERS,
    TMP_PATIENCE,
    TMP_WEIGHT_DECAY,
    TASK_EMBED_DIM,
    OCC_EMBED_DIM,
    TRAIN_RATIO,
    num_time_bins,
)
from data.datasets import OccurrenceDataset, TemporalDataset, build_split_indices
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data, serialize_metadata
from models.occurrence_model import TaskOccurrenceModel
from models.temporal_model import TemporalAssignmentModel
from training.engine import evaluate_epoch, fit_model
from training.losses import OccurrenceLoss, TemporalLoss
from training.metrics import occurrence_metrics, temporal_metrics
from utils.serialization import save_checkpoint
from evaluation.weekly_stats import evaluate_weekly_predictions
from predict import predict_next_week



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def resolve_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")



def build_occurrence_class_weights(dataset: OccurrenceDataset, max_count_cap: int, num_tasks: int) -> torch.Tensor:
    if len(dataset) == 0:
        return torch.ones(num_tasks, max_count_cap + 1)
    counts = torch.stack([item["target_counts"] for item in dataset], dim=0)
    weights = []
    for task_id in range(num_tasks):
        hist = torch.bincount(counts[:, task_id], minlength=max_count_cap + 1).float()
        hist = hist.clamp(min=1.0)
        inv = hist.sum() / hist
        inv = inv / inv.mean()
        weights.append(inv)
    return torch.stack(weights, dim=0)


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "-"
    return " | ".join(f"{k}={v:.3f}" for k, v in metrics.items())


def print_training_summary(model_name: str, state):
    print(f"\nResumen final de entrenamiento — {model_name}")
    print("-" * 50)
    print(f"Mejor epoch         : {state.best_epoch}")
    print(f"Best train_loss     : {state.best_train_loss:.4f}")
    print(f"Best val_loss       : {state.best_val_loss:.4f}")
    print(f"Último train_loss   : {state.final_train_loss:.4f}")
    print(f"Último val_loss     : {state.final_val_loss:.4f}")
    print(f"Best train metrics  : {_format_metrics(state.best_train_metrics)}")
    print(f"Best val metrics    : {_format_metrics(state.best_val_metrics)}")
    print(f"Último train metrics: {_format_metrics(state.final_train_metrics)}")
    print(f"Último val metrics  : {_format_metrics(state.final_val_metrics)}")


def print_loader_summary(title: str, loss: float, metrics: dict[str, float]):
    print(f"  {title}: loss={loss:.4f}")
    if metrics:
        print(f"    {_format_metrics(metrics)}")


def week_to_task_list(week_record):
    items = []
    for events in week_record.events_by_task.values():
        for e in events:
            items.append({
                "task_name": e.task_name,
                "start_bin": e.start_bin,
                "duration": e.duration_minutes,
            })
    return sorted(items, key=lambda x: (x["start_bin"], x["task_name"]))


def print_week_example(prepared, week_idx, pred_week, stats, max_items: int = 8):
    true_tasks = week_to_task_list(prepared.weeks[week_idx])
    pred_tasks = sorted(pred_week, key=lambda x: (x["start_bin"], x["task_name"]))

    print(f"\n  Semana índice {week_idx}")
    print(
        "    "
        f"tareas={stats['total_tasks']} | "
        f"task_acc={stats['task_accuracy']*100:.1f}% | "
        f"hora_exacta={stats['time_exact_accuracy']*100:.1f}% | "
        f"hora_±10m={stats['time_close_accuracy']*100:.1f}% | "
        f"dur_±2m={stats['duration_close_accuracy']*100:.1f}%"
    )

    print("    Real -> Predicción (primeras tareas)")
    shown = min(max(len(true_tasks), len(pred_tasks)), max_items)
    for i in range(shown):
        real = true_tasks[i] if i < len(true_tasks) else None
        pred = pred_tasks[i] if i < len(pred_tasks) else None

        def fmt(item):
            if item is None:
                return "---"
            total_minutes = int(item["start_bin"]) * 5
            day = total_minutes // (24 * 60)
            minute_of_day = total_minutes % (24 * 60)
            hh = minute_of_day // 60
            mm = minute_of_day % 60
            return f"{item['task_name']} @ d{day} {hh:02d}:{mm:02d} ({item['duration']:.1f}m)"

        print(f"      - {fmt(real)}  ==>  {fmt(pred)}")


def summarize_weekly_predictions(label, week_indices, prepared, occurrence_model, temporal_model, device, sample_weeks: int = 8):
    if not week_indices:
        print(f"\nResumen semanal interpretativo — {label}")
        print("  Sin semanas disponibles.")
        return

    chosen = week_indices[: min(sample_weeks, len(week_indices))]
    stats_list = []
    examples = []

    for idx in chosen:
        pred_week = predict_next_week(occurrence_model, temporal_model, prepared, idx, device)
        stats = evaluate_weekly_predictions(prepared.weeks[idx], pred_week)
        stats_list.append(stats)
        examples.append((idx, pred_week, stats))

    avg = {k: float(np.mean([s[k] for s in stats_list])) for k in stats_list[0].keys()}

    print(f"\nResumen semanal interpretativo — {label}")
    print("-" * 50)
    print(f"Semanas evaluadas   : {len(chosen)}")
    print(f"Tareas por semana   : {avg['total_tasks']:.1f}")
    print(f"Tareas correctas    : {avg['task_accuracy'] * avg['total_tasks']:.1f}/{avg['total_tasks']:.1f}")
    print(f"Horario exacto      : {avg['time_exact_accuracy'] * 100:.1f}%")
    print(f"Horario ±10 min     : {avg['time_close_accuracy'] * 100:.1f}%")
    print(f"Duración ±2 min     : {avg['duration_close_accuracy'] * 100:.1f}%")

    ranked = sorted(examples, key=lambda x: x[2]['time_exact_accuracy'])
    print_week_example(prepared, ranked[0][0], ranked[0][1], ranked[0][2])
    if len(ranked) > 1:
        print_week_example(prepared, ranked[-1][0], ranked[-1][1], ranked[-1][2])



def main():
    set_seed(SEED)
    device = resolve_device()

    df = load_tasks_dataframe(DATA_PATH, timezone=TIMEZONE)
    prepared = prepare_data(df, train_ratio=TRAIN_RATIO)
    split = build_split_indices(prepared, train_ratio=TRAIN_RATIO)

    occ_train = OccurrenceDataset(prepared, split.train_target_week_indices)
    occ_val = OccurrenceDataset(prepared, split.val_target_week_indices)
    tmp_train = TemporalDataset(prepared, split.train_target_week_indices)
    tmp_val = TemporalDataset(prepared, split.val_target_week_indices)

    print("\nResumen del dataset")
    print(f"  Tareas únicas      : {len(prepared.task_names)}")
    print(f"  Semanas totales    : {len(prepared.weeks)}")
    print(f"  Ventanas train     : {len(occ_train)}")
    print(f"  Ventanas val       : {len(occ_val)}")
    print(f"  Muestras temporales train: {len(tmp_train)}")
    print(f"  Muestras temporales val  : {len(tmp_val)}")
    print(f"  max_count_cap      : {prepared.max_count_cap}")
    print(f"  bins temporales    : {num_time_bins()}")

    occ_train_loader = DataLoader(occ_train, batch_size=OCC_BATCH_SIZE, shuffle=True)
    occ_val_loader = DataLoader(occ_val, batch_size=OCC_BATCH_SIZE, shuffle=False)
    tmp_train_loader = DataLoader(tmp_train, batch_size=TMP_BATCH_SIZE, shuffle=True)
    tmp_val_loader = DataLoader(tmp_val, batch_size=TMP_BATCH_SIZE, shuffle=False)

    occurrence_model = TaskOccurrenceModel(
        input_dim=prepared.week_feature_dim,
        num_tasks=len(prepared.task_names),
        max_count_cap=prepared.max_count_cap,
        hidden_size=OCC_HIDDEN_SIZE,
        num_layers=OCC_NUM_LAYERS,
        dropout=OCC_DROPOUT,
    ).to(device)

    class_weights = build_occurrence_class_weights(occ_train, prepared.max_count_cap, len(prepared.task_names))
    occ_loss_fn = OccurrenceLoss(class_weights=class_weights)
    occ_optimizer = torch.optim.AdamW(occurrence_model.parameters(), lr=OCC_LR, weight_decay=OCC_WEIGHT_DECAY)
    occ_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(occ_optimizer, mode="min", factor=0.5, patience=10)

    occ_state = fit_model(
        model=occurrence_model,
        train_loader=occ_train_loader,
        val_loader=occ_val_loader,
        optimizer=occ_optimizer,
        scheduler=occ_scheduler,
        loss_fn=occ_loss_fn,
        metrics_fn=occurrence_metrics,
        device=device,
        max_epochs=OCC_MAX_EPOCHS,
        patience=OCC_PATIENCE,
        model_name="OccurrenceModel",
    )

    temporal_model = TemporalAssignmentModel(
        sequence_dim=prepared.week_feature_dim,
        history_feature_dim=prepared.history_feature_dim,
        num_tasks=len(prepared.task_names),
        max_occurrences=prepared.max_count_cap,
        num_time_bins=num_time_bins(),
        hidden_size=TMP_HIDDEN_SIZE,
        num_layers=TMP_NUM_LAYERS,
        dropout=TMP_DROPOUT,
        task_embed_dim=TASK_EMBED_DIM,
        occurrence_embed_dim=OCC_EMBED_DIM,
    ).to(device)

    temporal_loss_fn = TemporalLoss()
    temporal_optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=TMP_LR, weight_decay=TMP_WEIGHT_DECAY)
    temporal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(temporal_optimizer, mode="min", factor=0.5, patience=10)

    duration_span_minutes = max(prepared.duration_max - prepared.duration_min, 1e-6)

    def temporal_metrics_wrapper(start_logits, target_start_bin, pred_duration, target_duration):
        return temporal_metrics(
            start_logits,
            target_start_bin,
            pred_duration,
            target_duration,
            duration_span_minutes=duration_span_minutes,
        )

    tmp_state = fit_model(
        model=temporal_model,
        train_loader=tmp_train_loader,
        val_loader=tmp_val_loader,
        optimizer=temporal_optimizer,
        scheduler=temporal_scheduler,
        loss_fn=temporal_loss_fn,
        metrics_fn=temporal_metrics_wrapper,
        device=device,
        max_epochs=TMP_MAX_EPOCHS,
        patience=TMP_PATIENCE,
        model_name="TemporalModel",
    )

    metadata = serialize_metadata(prepared)

    save_checkpoint(
        CHECKPOINT_DIR / "occurrence_model.pt",
        {
            "state_dict": occurrence_model.state_dict(),
            "metadata": metadata,
            "best_epoch": occ_state.best_epoch,
            "best_val_loss": occ_state.best_metric,
            "model_hparams": {
                "input_dim": prepared.week_feature_dim,
                "num_tasks": len(prepared.task_names),
                "max_count_cap": prepared.max_count_cap,
                "hidden_size": OCC_HIDDEN_SIZE,
                "num_layers": OCC_NUM_LAYERS,
                "dropout": OCC_DROPOUT,
            },
        },
    )

    save_checkpoint(
        CHECKPOINT_DIR / "temporal_model.pt",
        {
            "state_dict": temporal_model.state_dict(),
            "metadata": metadata,
            "best_epoch": tmp_state.best_epoch,
            "best_val_loss": tmp_state.best_metric,
            "model_hparams": {
                "sequence_dim": prepared.week_feature_dim,
                "history_feature_dim": prepared.history_feature_dim,
                "num_tasks": len(prepared.task_names),
                "max_occurrences": prepared.max_count_cap,
                "num_time_bins": num_time_bins(),
                "hidden_size": TMP_HIDDEN_SIZE,
                "num_layers": TMP_NUM_LAYERS,
                "dropout": TMP_DROPOUT,
                "task_embed_dim": TASK_EMBED_DIM,
                "occurrence_embed_dim": OCC_EMBED_DIM,
            },
        },
    )

    print("\nModelos guardados en:")
    print(f"  - {CHECKPOINT_DIR / 'occurrence_model.pt'}")
    print(f"  - {CHECKPOINT_DIR / 'temporal_model.pt'}")

    print_training_summary("OccurrenceModel", occ_state)
    print_training_summary("TemporalModel", tmp_state)

    print("\nResumen final con mejores pesos cargados")
    print("-" * 50)
    occ_train_loss, occ_train_metrics = evaluate_epoch(occurrence_model, occ_train_loader, occ_loss_fn, occurrence_metrics, device)
    occ_val_loss, occ_val_metrics = evaluate_epoch(occurrence_model, occ_val_loader, occ_loss_fn, occurrence_metrics, device)
    print("OccurrenceModel")
    print_loader_summary("train", occ_train_loss, occ_train_metrics)
    print_loader_summary("val  ", occ_val_loss, occ_val_metrics)

    tmp_train_loss, tmp_train_metrics = evaluate_epoch(temporal_model, tmp_train_loader, temporal_loss_fn, temporal_metrics_wrapper, device)
    tmp_val_loss, tmp_val_metrics = evaluate_epoch(temporal_model, tmp_val_loader, temporal_loss_fn, temporal_metrics_wrapper, device)
    print("TemporalModel")
    print_loader_summary("train", tmp_train_loss, tmp_train_metrics)
    print_loader_summary("val  ", tmp_val_loss, tmp_val_metrics)

    summarize_weekly_predictions(
        label="train",
        week_indices=split.train_target_week_indices[-8:],
        prepared=prepared,
        occurrence_model=occurrence_model,
        temporal_model=temporal_model,
        device=device,
        sample_weeks=8,
    )
    summarize_weekly_predictions(
        label="validación",
        week_indices=split.val_target_week_indices[:8],
        prepared=prepared,
        occurrence_model=occurrence_model,
        temporal_model=temporal_model,
        device=device,
        sample_weeks=8,
    )

if __name__ == "__main__":
    main()
