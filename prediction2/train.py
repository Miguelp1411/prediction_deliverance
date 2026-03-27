from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CAP_INFERENCE_SCOPE,
    CHECKPOINT_DIR,
    DATA_PATH,
    DAY_EMBED_DIM,
    DEVICE,
    OCC_BATCH_SIZE,
    OCC_DROPOUT,
    OCC_EMBED_DIM,
    OCC_HIDDEN_SIZE,
    OCC_LR,
    OCC_MAX_EPOCHS,
    OCC_NUM_LAYERS,
    OCC_PATIENCE,
    OCC_SCHEDULER_PATIENCE,
    OCC_WEIGHT_DECAY,
    SEED,
    TASK_EMBED_DIM,
    TIMEZONE,
    TMP_BATCH_SIZE,
    TMP_DROPOUT,
    TMP_HIDDEN_SIZE,
    TMP_LR,
    TMP_MAX_EPOCHS,
    TMP_NUM_LAYERS,
    TMP_PATIENCE,
    TMP_SCHEDULER_PATIENCE,
    TMP_WEIGHT_DECAY,
    TRAIN_RATIO,
    bins_per_day,
    num_day_classes,
    num_time_bins,
    num_time_of_day_classes,
)
from data.datasets import OccurrenceDataset, TemporalDataset, build_split_indices
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data, serialize_metadata
from evaluation.weekly_stats import evaluate_weekly_predictions
from models.occurrence_model import TaskOccurrenceModel
from models.temporal_model import TemporalAssignmentModel
from predict import predict_next_week
from training.engine import evaluate_epoch, fit_model
from training.losses import OccurrenceLoss, TemporalLoss
from training.metrics import occurrence_metrics, temporal_metrics
from utils.serialization import save_checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if DEVICE == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_occurrence_class_weights(dataset: OccurrenceDataset, max_count_cap: int, num_tasks: int) -> torch.Tensor:
    if len(dataset) == 0:
        return torch.ones(num_tasks, max_count_cap + 1)
    counts = torch.stack([item['target_counts'] for item in dataset], dim=0)
    weights = []
    for task_id in range(num_tasks):
        hist = torch.bincount(counts[:, task_id], minlength=max_count_cap + 1).float().clamp(min=1.0)
        inv = hist.sum() / hist
        weights.append(inv / inv.mean())
    return torch.stack(weights, dim=0)


def _format_metrics(metrics: dict[str, float]) -> str:
    return ' | '.join(f'{k}={v:.3f}' for k, v in metrics.items()) if metrics else '-'


def print_training_summary(model_name: str, state):
    print(f'\nResumen final de entrenamiento — {model_name}')
    print('-' * 50)
    print(f'Mejor epoch         : {state.best_epoch}')
    print(f'Monitor             : {state.monitor_name} ({state.monitor_mode})')
    print(f'Mejor monitor       : {state.best_metric:.4f}')
    print(f'Best train_loss     : {state.best_train_loss:.4f}')
    print(f'Best val_loss       : {state.best_val_loss:.4f}')
    print(f'Último train_loss   : {state.final_train_loss:.4f}')
    print(f'Último val_loss     : {state.final_val_loss:.4f}')
    print(f'Best train metrics  : {_format_metrics(state.best_train_metrics)}')
    print(f'Best val metrics    : {_format_metrics(state.best_val_metrics)}')
    print(f'Último train metrics: {_format_metrics(state.final_train_metrics)}')
    print(f'Último val metrics  : {_format_metrics(state.final_val_metrics)}')


def print_loader_summary(title: str, loss: float, metrics: dict[str, float]):
    print(f'  {title}: loss={loss:.4f}')
    if metrics:
        print(f'    {_format_metrics(metrics)}')


def _event_label(event):
    if hasattr(event, 'task_name'):
        return str(event.task_name)
    if hasattr(event, 'type'):
        return str(event.type)
    raise AttributeError("El evento no tiene ni 'task_name' ni 'type'")


def _item_label(item):
    if 'task_name' in item:
        return str(item['task_name'])
    if 'type' in item:
        return str(item['type'])
    raise KeyError("El item no tiene ni 'task_name' ni 'type'")


def week_to_task_list(week_record):
    items = []
    for events in week_record.events_by_task.values():
        for e in events:
            items.append({
                'task_name': _event_label(e),
                'start_bin': int(e.start_bin),
                'duration': float(e.duration_minutes),
            })
    return sorted(items, key=lambda x: (x['start_bin'], x['task_name']))


def print_week_example(prepared, week_idx, pred_week, stats, max_items: int = 8):
    true_tasks = week_to_task_list(prepared.weeks[week_idx])

    pred_tasks = []
    for item in pred_week:
        pred_tasks.append({
            'task_name': _item_label(item),
            'start_bin': int(item['start_bin']),
            'duration': float(item['duration']),
        })
    pred_tasks = sorted(pred_tasks, key=lambda x: (x['start_bin'], x['task_name']))

    total_tasks = float(stats.get('total_tasks', stats.get('tasks_per_week', 0.0)))
    task_accuracy = float(stats.get('task_accuracy', stats.get('task_acc', 0.0) / 100.0))
    time_exact_accuracy = float(stats.get('time_exact_accuracy', stats.get('start_exact_acc', 0.0) / 100.0))
    time_close_accuracy_5m = float(stats.get('time_close_accuracy_5m', stats.get('start_tol_acc_5m', 0.0) / 100.0))
    time_close_accuracy_10m = float(stats.get('time_close_accuracy_10m', stats.get('start_tol_acc_10m', 0.0) / 100.0))
    duration_close_accuracy = float(stats.get('duration_close_accuracy', stats.get('duration_tol_acc_2m', 0.0) / 100.0))

    """ print(f'\n  Semana índice {week_idx}')
    print(
        '    '
        f"tareas={total_tasks:.0f} | "
        f"task_acc={task_accuracy*100:.1f}% | "
        f"hora_exacta={time_exact_accuracy*100:.1f}% | "
        f"hora_±5m={time_close_accuracy_5m*100:.1f}% | "
        f"hora_±10m={time_close_accuracy_10m*100:.1f}% | "
        f"dur_±2m={duration_close_accuracy*100:.1f}%"
    )
    print('    Real -> Predicción (primeras tareas)')

    shown = min(max(len(true_tasks), len(pred_tasks)), max_items)

    for i in range(shown):
        real = true_tasks[i] if i < len(true_tasks) else None
        pred = pred_tasks[i] if i < len(pred_tasks) else None

        def fmt(item):
            if item is None:
                return '---'
            total_minutes = int(item['start_bin']) * 5
            day = total_minutes // (24 * 60)
            minute_of_day = total_minutes % (24 * 60)
            hh = minute_of_day // 60
            mm = minute_of_day % 60
            return f"{item['task_name']} @ d{day} {hh:02d}:{mm:02d} ({item['duration']:.1f}m)"

        print(f'      - {fmt(real)}  ==>  {fmt(pred)}')"""


def summarize_weekly_predictions(
    label,
    week_indices,
    prepared,
    occurrence_model,
    temporal_model,
    device,
    sample_weeks: int = 8,
    use_repair: bool = False,
):
    if not week_indices:
        print(f'\nResumen semanal interpretativo — {label}')
        print('  Sin semanas disponibles.')
        return

    chosen = list(week_indices)
    stats_list = []
    examples = []

    for idx in chosen:
        pred_week = predict_next_week(
            occurrence_model,
            temporal_model,
            prepared,
            idx,
            device,
            use_repair=use_repair,
        )
        stats = evaluate_weekly_predictions(prepared.weeks[idx], pred_week)
        stats_list.append(stats)
        examples.append((idx, pred_week, stats))

    avg = {k: float(np.mean([s[k] for s in stats_list])) for k in stats_list[0].keys()}

    total_tasks = float(avg.get('total_tasks', avg.get('tasks_per_week', 0.0)))
    task_accuracy = float(avg.get('task_accuracy', avg.get('task_acc', 0.0) / 100.0))
    time_exact_accuracy = float(avg.get('time_exact_accuracy', avg.get('start_exact_acc', 0.0) / 100.0))
    time_close_accuracy_5m = float(avg.get('time_close_accuracy_5m', avg.get('start_tol_acc_5m', 0.0) / 100.0))
    time_close_accuracy_10m = float(avg.get('time_close_accuracy_10m', avg.get('start_tol_acc_10m', 0.0) / 100.0))
    duration_close_accuracy = float(avg.get('duration_close_accuracy', avg.get('duration_tol_acc_2m', 0.0) / 100.0))
    start_mae_minutes = float(avg.get('start_mae_minutes', 0.0))

    print(f'\nResumen semanal interpretativo — {label}')
    print('-' * 50)
    print(f"Semanas evaluadas   : {len(chosen)}")
    print(f"Repair              : {'ON' if use_repair else 'OFF'}")
    print(f"Tareas por semana   : {total_tasks:.1f}")
    print(f"Tareas correctas    : {task_accuracy * total_tasks:.1f}/{total_tasks:.1f}")
    print(f"Horario exacto      : {time_exact_accuracy * 100:.1f}%")
    print(f"Horario ±5 min      : {time_close_accuracy_5m * 100:.1f}%")
    print(f"Horario ±10 min     : {time_close_accuracy_10m * 100:.1f}%")
    print(f"MAE inicio          : {start_mae_minutes:.2f} min")
    print(f"Duración ±2 min     : {duration_close_accuracy * 100:.1f}%")

    ranked = sorted(
        examples,
        key=lambda x: float(x[2].get('time_exact_accuracy', x[2].get('start_exact_acc', 0.0) / 100.0))
    )

    if ranked:
        print_week_example(prepared, ranked[0][0], ranked[0][1], ranked[0][2])
    if len(ranked) > 1:
        print_week_example(prepared, ranked[-1][0], ranked[-1][1], ranked[-1][2])


def main():
    set_seed(SEED)
    device = resolve_device()

    print("\n[1/5] Cargando datos...", flush=True)
    df = load_tasks_dataframe(DATA_PATH, timezone=TIMEZONE)

    print("[2/5] Preprocesando datos...", flush=True)
    prepared = prepare_data(
        df,
        train_ratio=TRAIN_RATIO,
        cap_inference_scope=CAP_INFERENCE_SCOPE,
        show_progress=True,
    )

    print("[3/5] Construyendo partición train/val...", flush=True)
    split = build_split_indices(prepared, train_ratio=TRAIN_RATIO)

    print("[4/5] Construyendo OccurrenceDataset...", flush=True)
    occ_train = OccurrenceDataset(prepared, split.train_target_week_indices)
    occ_val = OccurrenceDataset(prepared, split.val_target_week_indices)

    print("[5/5] Construyendo TemporalDataset...", flush=True)
    tmp_train = TemporalDataset(
        prepared,
        split.train_target_week_indices,
        show_progress=True,
        desc="TemporalDataset train",
    )
    tmp_val = TemporalDataset(
    prepared,
    split.val_target_week_indices,
    show_progress=True,
    desc="TemporalDataset val",
)

    print('\nResumen del dataset')
    print(f'  Tareas únicas             : {len(prepared.task_names)}')
    print(f'  Semanas totales           : {len(prepared.weeks)}')
    print(f'  Ventanas train            : {len(occ_train)}')
    print(f'  Ventanas val              : {len(occ_val)}')
    print(f'  Muestras temporales train : {len(tmp_train)}')
    print(f'  Muestras temporales val   : {len(tmp_val)}')
    print(f'  max_count_cap             : {prepared.max_count_cap}')
    print(f'  max_occurrences_per_task  : {prepared.max_occurrences_per_task}')
    print(f'  max_tasks_per_week        : {prepared.max_tasks_per_week}')
    print(f'  cap_inference_scope       : {prepared.cap_inference_scope}')
    print(f'  caps train                : occ={prepared.inferred_train_max_occurrences_per_task} | tareas={prepared.inferred_train_max_tasks_per_week}')
    print(f'  caps base completa        : occ={prepared.inferred_full_max_occurrences_per_task} | tareas={prepared.inferred_full_max_tasks_per_week}')
    print(f'  bins temporales           : {num_time_bins()}')
    print(f'  clases día                : {num_day_classes()}')
    print(f'  clases hora del día       : {num_time_of_day_classes()}')

    occ_train_loader = DataLoader(occ_train, batch_size=OCC_BATCH_SIZE, shuffle=True)
    occ_val_loader = DataLoader(occ_val, batch_size=OCC_BATCH_SIZE, shuffle=False)
    tmp_train_loader = DataLoader(tmp_train, batch_size=TMP_BATCH_SIZE, shuffle=True)
    tmp_val_loader = DataLoader(tmp_val, batch_size=TMP_BATCH_SIZE, shuffle=False)

    occurrence_model = TaskOccurrenceModel(prepared.week_feature_dim, len(prepared.task_names), prepared.max_count_cap, OCC_HIDDEN_SIZE, OCC_NUM_LAYERS, OCC_DROPOUT).to(device)
    occ_loss_fn = OccurrenceLoss(build_occurrence_class_weights(occ_train, prepared.max_count_cap, len(prepared.task_names)))
    occ_optimizer = torch.optim.AdamW(occurrence_model.parameters(), lr=OCC_LR, weight_decay=OCC_WEIGHT_DECAY)
    occ_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(occ_optimizer, mode='min', factor=0.5, patience=OCC_SCHEDULER_PATIENCE)
    occ_state = fit_model(occurrence_model, occ_train_loader, occ_val_loader, occ_optimizer, occ_scheduler, occ_loss_fn, occurrence_metrics, device, OCC_MAX_EPOCHS, OCC_PATIENCE, 'OccurrenceModel', monitor_name='val_loss', monitor_mode='min', min_delta=1e-4)

    temporal_model = TemporalAssignmentModel(prepared.week_feature_dim, prepared.history_feature_dim, len(prepared.task_names), prepared.max_count_cap, TMP_HIDDEN_SIZE, TMP_NUM_LAYERS, TMP_DROPOUT, TASK_EMBED_DIM, OCC_EMBED_DIM, DAY_EMBED_DIM).to(device)
    temporal_loss_fn = TemporalLoss()
    temporal_optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=TMP_LR, weight_decay=TMP_WEIGHT_DECAY)
    temporal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(temporal_optimizer, mode='max', factor=0.5, patience=TMP_SCHEDULER_PATIENCE)
    duration_span = max(prepared.duration_max - prepared.duration_min, 1e-6)

    def temporal_metrics_wrapper(outputs, batch):
        return temporal_metrics(outputs, batch, duration_span)

    tmp_state = fit_model(temporal_model, tmp_train_loader, tmp_val_loader, temporal_optimizer, temporal_scheduler, temporal_loss_fn, temporal_metrics_wrapper, device, TMP_MAX_EPOCHS, TMP_PATIENCE, 'TemporalModel', monitor_name='start_tol_acc_5m', monitor_mode='max', min_delta=0.05)

    metadata = serialize_metadata(prepared)
    save_checkpoint(CHECKPOINT_DIR / 'occurrence_model.pt', {
        'state_dict': occurrence_model.state_dict(),
        'metadata': metadata,
        'best_epoch': occ_state.best_epoch,
        'best_val_loss': occ_state.best_val_loss,
        'best_monitor_name': occ_state.monitor_name,
        'best_monitor_value': occ_state.best_metric,
        'model_hparams': {
            'input_dim': prepared.week_feature_dim,
            'num_tasks': len(prepared.task_names),
            'max_count_cap': prepared.max_count_cap,
            'hidden_size': OCC_HIDDEN_SIZE,
            'num_layers': OCC_NUM_LAYERS,
            'dropout': OCC_DROPOUT,
            'max_occurrences_per_task': prepared.max_occurrences_per_task,
            'max_tasks_per_week': prepared.max_tasks_per_week,
        },
    })
    save_checkpoint(CHECKPOINT_DIR / 'temporal_model.pt', {
        'state_dict': temporal_model.state_dict(),
        'metadata': metadata,
        'best_epoch': tmp_state.best_epoch,
        'best_val_loss': tmp_state.best_val_loss,
        'best_monitor_name': tmp_state.monitor_name,
        'best_monitor_value': tmp_state.best_metric,
        'model_hparams': {
            'sequence_dim': prepared.week_feature_dim,
            'history_feature_dim': prepared.history_feature_dim,
            'num_tasks': len(prepared.task_names),
            'max_occurrences': prepared.max_count_cap,
            'max_occurrences_per_task': prepared.max_occurrences_per_task,
            'hidden_size': TMP_HIDDEN_SIZE,
            'num_layers': TMP_NUM_LAYERS,
            'dropout': TMP_DROPOUT,
            'task_embed_dim': TASK_EMBED_DIM,
            'occurrence_embed_dim': OCC_EMBED_DIM,
            'day_embed_dim': DAY_EMBED_DIM,
            'max_tasks_per_week': prepared.max_tasks_per_week,
        },
    })

    print('\nModelos guardados en:')
    print(f"  - {CHECKPOINT_DIR / 'occurrence_model.pt'}")
    print(f"  - {CHECKPOINT_DIR / 'temporal_model.pt'}")
    print_training_summary('OccurrenceModel', occ_state)
    print_training_summary('TemporalModel', tmp_state)

    print('\nResumen final con mejores pesos cargados')
    print('-' * 50)
    occ_train_loss, occ_train_metrics = evaluate_epoch(occurrence_model, occ_train_loader, occ_loss_fn, occurrence_metrics, device)
    occ_val_loss, occ_val_metrics = evaluate_epoch(occurrence_model, occ_val_loader, occ_loss_fn, occurrence_metrics, device)
    print('OccurrenceModel')
    print_loader_summary('train', occ_train_loss, occ_train_metrics)
    print_loader_summary('val  ', occ_val_loss, occ_val_metrics)
    tmp_train_loss, tmp_train_metrics = evaluate_epoch(temporal_model, tmp_train_loader, temporal_loss_fn, temporal_metrics_wrapper, device)
    tmp_val_loss, tmp_val_metrics = evaluate_epoch(temporal_model, tmp_val_loader, temporal_loss_fn, temporal_metrics_wrapper, device)
    print('TemporalModel')
    print_loader_summary('train', tmp_train_loss, tmp_train_metrics)
    print_loader_summary('val  ', tmp_val_loss, tmp_val_metrics)

    summarize_weekly_predictions(
        'train',
        split.train_target_week_indices[-8:],
        prepared,
        occurrence_model,
        temporal_model,
        device,
        8,
        use_repair=False,
    )

    summarize_weekly_predictions(
        'validación',
        split.val_target_week_indices[:8],
        prepared,
        occurrence_model,
        temporal_model,
        device,
        8,
        use_repair=False,
    )

if __name__ == '__main__':
    main()
