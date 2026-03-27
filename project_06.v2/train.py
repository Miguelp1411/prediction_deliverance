from __future__ import annotations

import random
from types import SimpleNamespace

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
    OCC_LAG_WEEKS,
    OCC_NUM_LAYERS,
    OCCURRENCE_MODEL_KIND,
    SEED,
    TASK_EMBED_DIM,
    TEMPORAL_COUNT_BLEND_TARGET_WEIGHT,
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
    num_day_classes,
    num_time_bins,
    num_time_of_day_classes,
)
from data.datasets import OccurrenceDataset, TemporalDataset, build_occurrence_count_lookup, build_split_indices
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data, serialize_metadata
from evaluation.weekly_stats import evaluate_weekly_predictions
from models.occurrence_model import StructuredOccurrenceModel, TaskOccurrenceModel
from models.temporal_model import TemporalAssignmentModel
from predict import predict_next_week
from training.engine import evaluate_epoch, fit_model
from training.losses import TemporalLoss
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


class ZeroLoss(torch.nn.Module):
    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            device = next(iter(outputs.values())).device
        else:
            device = outputs.device
        return torch.zeros((), dtype=torch.float32, device=device)


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


def aggregate_weekly_stats(prepared, week_indices, occurrence_model, temporal_model, device, use_repair: bool = False):
    if not week_indices:
        return {}
    stats_list = []
    per_task_accumulator: dict[str, list[tuple[float, float]]] = {}
    for idx in week_indices:
        pred_week = predict_next_week(occurrence_model, temporal_model, prepared, idx, device, use_repair=use_repair)
        stats = evaluate_weekly_predictions(prepared.weeks[idx], pred_week)
        stats_list.append(stats)
        for task_name, task_stats in stats.get('per_task', {}).items():
            per_task_accumulator.setdefault(task_name, []).append((float(task_stats.get('task_accuracy', 0.0)), float(task_stats.get('time_exact_accuracy', 0.0))))
    avg = {k: float(np.mean([float(s[k]) for s in stats_list])) for k in ['total_tasks','correct_tasks','task_accuracy','time_exact_accuracy','time_close_accuracy_5m','time_close_accuracy_10m','duration_close_accuracy','start_mae_minutes','overlap_count']}
    avg['per_task'] = {task_name: {'task_accuracy': float(np.mean([x[0] for x in values])), 'time_exact_accuracy': float(np.mean([x[1] for x in values]))} for task_name, values in per_task_accumulator.items()}
    avg['e2e_task_acc'] = avg['task_accuracy'] * 100.0
    avg['e2e_start_exact_acc'] = avg['time_exact_accuracy'] * 100.0
    avg['e2e_start_tol_acc_5m'] = avg['time_close_accuracy_5m'] * 100.0
    avg['e2e_overlap_count'] = avg['overlap_count']
    avg['e2e_joint_score'] = avg['e2e_task_acc'] + avg['e2e_start_exact_acc'] - 5.0 * avg['e2e_overlap_count']
    return avg


class TemporalE2EEvaluator:
    def __init__(self, prepared, week_indices, occurrence_model, device, use_repair: bool = False):
        self.prepared = prepared
        self.week_indices = list(week_indices)
        self.occurrence_model = occurrence_model
        self.device = device
        self.use_repair = use_repair

    @torch.no_grad()
    def __call__(self, temporal_model):
        temporal_model.eval()
        stats = aggregate_weekly_stats(self.prepared, self.week_indices, self.occurrence_model, temporal_model, self.device, use_repair=self.use_repair)
        return {k: stats.get(k, 0.0) for k in ['e2e_task_acc','e2e_start_exact_acc','e2e_start_tol_acc_5m','e2e_overlap_count','e2e_joint_score']}


def summarize_weekly_predictions(label, stats: dict):
    if not stats:
        print(f'\nResumen semanal interpretativo — {label}')
        print('  Sin semanas disponibles.')
        return
    print(f'\nResumen semanal interpretativo — {label}')
    print('-' * 50)
    print(f"Tareas por semana   : {stats['total_tasks']:.1f}")
    print(f"Tareas correctas    : {stats['correct_tasks']:.1f}/{stats['total_tasks']:.1f}")
    print(f"Horario exacto      : {stats['time_exact_accuracy'] * 100:.1f}%")
    print(f"Horario ±5 min      : {stats['time_close_accuracy_5m'] * 100:.1f}%")
    print(f"Horario ±10 min     : {stats['time_close_accuracy_10m'] * 100:.1f}%")
    print(f"MAE inicio          : {stats['start_mae_minutes']:.2f} min")
    print(f"Duración ±2 min     : {stats['duration_close_accuracy'] * 100:.1f}%")
    print(f"Overlaps/semana     : {stats['overlap_count']:.2f}")
    if stats.get('per_task'):
        print('  Por tarea:')
        for task_name, task_stats in sorted(stats['per_task'].items()):
            print(f"    - {task_name}: task_acc={task_stats['task_accuracy'] * 100:.1f}% | start_exact={task_stats['time_exact_accuracy'] * 100:.1f}%")


def build_occurrence_model(prepared, occ_train: OccurrenceDataset, device: torch.device):
    if OCCURRENCE_MODEL_KIND == 'structured_lag4':
        model = StructuredOccurrenceModel(prepared.week_feature_dim, len(prepared.task_names), prepared.max_count_cap, lag_weeks=OCC_LAG_WEEKS).to(device)
        target_counts = torch.stack([item['target_counts'] for item in occ_train], dim=0) if len(occ_train) > 0 else None
        model.fit(target_counts)
        state = SimpleNamespace(best_epoch=0, best_metric=float('nan'), best_train_loss=0.0, best_val_loss=0.0, final_train_loss=0.0, final_val_loss=0.0, best_train_metrics={}, best_val_metrics={}, final_train_metrics={}, final_val_metrics={}, monitor_name='rule_based', monitor_mode='max')
        return model, state
    model = TaskOccurrenceModel(prepared.week_feature_dim, len(prepared.task_names), prepared.max_count_cap, OCC_HIDDEN_SIZE, OCC_NUM_LAYERS, OCC_DROPOUT).to(device)
    raise NotImplementedError('Esta versión del proyecto está configurada para OCCURRENCE_MODEL_KIND=structured_lag4')


def main():
    set_seed(SEED)
    device = resolve_device()

    print("\n[1/6] Cargando datos...", flush=True)
    df = load_tasks_dataframe(DATA_PATH, timezone=TIMEZONE)
    print("[2/6] Preprocesando datos...", flush=True)
    prepared = prepare_data(df, train_ratio=TRAIN_RATIO, cap_inference_scope=CAP_INFERENCE_SCOPE, show_progress=True)
    print("[3/6] Construyendo partición train/val...", flush=True)
    split = build_split_indices(prepared, train_ratio=TRAIN_RATIO)
    print("[4/6] Construyendo OccurrenceDataset...", flush=True)
    occ_train = OccurrenceDataset(prepared, split.train_target_week_indices)
    occ_val = OccurrenceDataset(prepared, split.val_target_week_indices)
    print("[5/6] Ajustando predictor de ocurrencias estructurado...", flush=True)
    occurrence_model, occ_state = build_occurrence_model(prepared, occ_train, device)

    zero_loss = ZeroLoss()
    occ_train_loader = DataLoader(occ_train, batch_size=OCC_BATCH_SIZE, shuffle=False)
    occ_val_loader = DataLoader(occ_val, batch_size=OCC_BATCH_SIZE, shuffle=False)
    occ_train_loss, occ_train_metrics = evaluate_epoch(occurrence_model, occ_train_loader, zero_loss, occurrence_metrics, device)
    occ_val_loss, occ_val_metrics = evaluate_epoch(occurrence_model, occ_val_loader, zero_loss, occurrence_metrics, device)
    occ_state.best_train_metrics = dict(occ_train_metrics)
    occ_state.best_val_metrics = dict(occ_val_metrics)
    occ_state.final_train_metrics = dict(occ_train_metrics)
    occ_state.final_val_metrics = dict(occ_val_metrics)

    print("[6/6] Construyendo TemporalDataset alineado con inferencia...", flush=True)
    train_count_lookup = build_occurrence_count_lookup(prepared, split.train_target_week_indices, occurrence_model, device)
    val_count_lookup = build_occurrence_count_lookup(prepared, split.val_target_week_indices, occurrence_model, device)
    tmp_train = TemporalDataset(prepared, split.train_target_week_indices, count_lookup=train_count_lookup, count_blend_alpha=TEMPORAL_COUNT_BLEND_TARGET_WEIGHT, show_progress=True, desc="TemporalDataset train")
    tmp_val = TemporalDataset(prepared, split.val_target_week_indices, count_lookup=val_count_lookup, count_blend_alpha=0.0, show_progress=True, desc="TemporalDataset val")

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

    tmp_train_loader = DataLoader(tmp_train, batch_size=TMP_BATCH_SIZE, shuffle=True)
    tmp_val_loader = DataLoader(tmp_val, batch_size=TMP_BATCH_SIZE, shuffle=False)

    temporal_model = TemporalAssignmentModel(prepared.week_feature_dim, prepared.history_feature_dim, len(prepared.task_names), prepared.max_count_cap, TMP_HIDDEN_SIZE, TMP_NUM_LAYERS, TMP_DROPOUT, TASK_EMBED_DIM, OCC_EMBED_DIM, DAY_EMBED_DIM).to(device)
    temporal_loss_fn = TemporalLoss()
    temporal_optimizer = torch.optim.AdamW(temporal_model.parameters(), lr=TMP_LR, weight_decay=TMP_WEIGHT_DECAY)
    temporal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(temporal_optimizer, mode='max', factor=0.5, patience=TMP_SCHEDULER_PATIENCE)
    duration_span = max(prepared.duration_max - prepared.duration_min, 1e-6)
    def temporal_metrics_wrapper(outputs, batch): return temporal_metrics(outputs, batch, duration_span)
    e2e_evaluator = TemporalE2EEvaluator(prepared, split.val_target_week_indices, occurrence_model, device, use_repair=False)

    tmp_state = fit_model(temporal_model, tmp_train_loader, tmp_val_loader, temporal_optimizer, temporal_scheduler, temporal_loss_fn, temporal_metrics_wrapper, device, TMP_MAX_EPOCHS, TMP_PATIENCE, 'TemporalModel', monitor_name='e2e_joint_score', monitor_mode='max', min_delta=0.10, extra_val_evaluator=e2e_evaluator)

    metadata = serialize_metadata(prepared)
    metadata['occurrence_model_kind'] = OCCURRENCE_MODEL_KIND
    metadata['occurrence_lag_weeks'] = OCC_LAG_WEEKS

    save_checkpoint(CHECKPOINT_DIR / 'occurrence_model.pt', {'state_dict': occurrence_model.state_dict(), 'metadata': metadata, 'best_epoch': occ_state.best_epoch, 'best_val_loss': occ_state.best_val_loss, 'best_monitor_name': occ_state.monitor_name, 'best_monitor_value': occ_state.best_metric, 'model_hparams': {'model_kind': OCCURRENCE_MODEL_KIND, 'input_dim': prepared.week_feature_dim, 'num_tasks': len(prepared.task_names), 'max_count_cap': prepared.max_count_cap, 'lag_weeks': OCC_LAG_WEEKS, 'max_occurrences_per_task': prepared.max_occurrences_per_task, 'max_tasks_per_week': prepared.max_tasks_per_week}})
    save_checkpoint(CHECKPOINT_DIR / 'temporal_model.pt', {'state_dict': temporal_model.state_dict(), 'metadata': metadata, 'best_epoch': tmp_state.best_epoch, 'best_val_loss': tmp_state.best_val_loss, 'best_monitor_name': tmp_state.monitor_name, 'best_monitor_value': tmp_state.best_metric, 'model_hparams': {'sequence_dim': prepared.week_feature_dim, 'history_feature_dim': prepared.history_feature_dim, 'num_tasks': len(prepared.task_names), 'max_occurrences': prepared.max_count_cap, 'max_occurrences_per_task': prepared.max_occurrences_per_task, 'hidden_size': TMP_HIDDEN_SIZE, 'num_layers': TMP_NUM_LAYERS, 'dropout': TMP_DROPOUT, 'task_embed_dim': TASK_EMBED_DIM, 'occurrence_embed_dim': OCC_EMBED_DIM, 'day_embed_dim': DAY_EMBED_DIM, 'max_tasks_per_week': prepared.max_tasks_per_week}})

    print('\nModelos guardados en:')
    print(f"  - {CHECKPOINT_DIR / 'occurrence_model.pt'}")
    print(f"  - {CHECKPOINT_DIR / 'temporal_model.pt'}")
    print_training_summary('OccurrenceModel', occ_state)
    print_training_summary('TemporalModel', tmp_state)
    print('\nResumen final con mejores pesos cargados')
    print('-' * 50)
    print('OccurrenceModel')
    print_loader_summary('train', occ_train_loss, occ_train_metrics)
    print_loader_summary('val  ', occ_val_loss, occ_val_metrics)
    tmp_train_loss, tmp_train_metrics = evaluate_epoch(temporal_model, tmp_train_loader, temporal_loss_fn, temporal_metrics_wrapper, device)
    tmp_val_loss, tmp_val_metrics = evaluate_epoch(temporal_model, tmp_val_loader, temporal_loss_fn, temporal_metrics_wrapper, device)
    print('TemporalModel')
    print_loader_summary('train', tmp_train_loss, tmp_train_metrics)
    print_loader_summary('val  ', tmp_val_loss, tmp_val_metrics)
    summarize_weekly_predictions('train', aggregate_weekly_stats(prepared, split.train_target_week_indices[-8:], occurrence_model, temporal_model, device, use_repair=False))
    summarize_weekly_predictions('validación', aggregate_weekly_stats(prepared, split.val_target_week_indices, occurrence_model, temporal_model, device, use_repair=False))


if __name__ == '__main__':
    main()
