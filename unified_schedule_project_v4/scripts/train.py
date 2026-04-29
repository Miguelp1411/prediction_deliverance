from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry, profile_events_dataframe
from hybrid_schedule.data.features import build_global_context
from hybrid_schedule.data.scaling import (
    apply_occurrence_scaling,
    fit_history_stats_from_samples,
    fit_vector_stats_from_samples,
)
from hybrid_schedule.models import UnifiedSlotTransformer
from hybrid_schedule.reporting import RunLogger, build_final_report, save_summary
from hybrid_schedule.training import UnifiedWeekSlotDataset, build_time_split, fit_unified_model, run_holdout_backtest
from hybrid_schedule.utils import format_duration, get_device, save_checkpoint, seed_everything


NUMERIC_LOG_INDICES = [3, 5, 10, 11, 13, 15, 16, 17, 19, 21, 22, 25, 26, 27, 28, 29, 35, 36, 37, 39, 45, 46, 47, 48, 49, 50]


def _fmt_summary_value(value, suffix: str = '') -> str:
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.4f}{suffix}'
    except (TypeError, ValueError):
        return str(value)


def print_final_training_summary(
    *,
    output_dir: Path,
    fit_info: dict,
    val_metrics: dict,
    backtest_summary: dict,
    timings: dict,
) -> None:
    print('\n===== Resumen final del entrenamiento =====', flush=True)
    print(f"Artefactos guardados en: {output_dir}", flush=True)
    print(
        f"Épocas ejecutadas: {fit_info.get('epochs_ran', 'n/a')} | "
        f"mejor época: {fit_info.get('best_epoch', 'n/a')} | "
        f"best_val_loss: {_fmt_summary_value(fit_info.get('best_val_loss'))} | "
        f"best_{fit_info.get('best_monitor_name', 'monitor')}: {_fmt_summary_value(fit_info.get('best_monitor_score'))}",
        flush=True,
    )
    print(
        "Validación mejor época | "
        f"loss={_fmt_summary_value(val_metrics.get('loss'))} | "
        f"active_f1={_fmt_summary_value(val_metrics.get('active_f1'), '%')} | "
        f"count_mae={_fmt_summary_value(val_metrics.get('count_mae'))} | "
        f"start_mae={_fmt_summary_value(val_metrics.get('start_mae_minutes'), ' min')} | "
        f"duration_mae={_fmt_summary_value(val_metrics.get('duration_mae_minutes'), ' min')}",
        flush=True,
    )
    if backtest_summary:
        print(
            "Backtest holdout | "
            f"task_f1={_fmt_summary_value(backtest_summary.get('task_f1'), '%')} | "
            f"precision={_fmt_summary_value(backtest_summary.get('task_precision'), '%')} | "
            f"recall={_fmt_summary_value(backtest_summary.get('task_recall'), '%')} | "
            f"start_mae={_fmt_summary_value(backtest_summary.get('start_mae_minutes'), ' min')}",
            flush=True,
        )
    print(
        "Tiempos | "
        f"preparación datos={format_duration(timings.get('data_preparation_seconds', 0.0))} | "
        f"entrenamiento épocas={format_duration(timings.get('model_fit_seconds', 0.0))} | "
        f"proceso total={format_duration(timings.get('total_process_seconds', 0.0))}",
        flush=True,
    )
    print('===========================================\n', flush=True)




def infer_max_slots_by_task(
    context,
    train_indices,
    quantile: float = 0.99,
    margin: int = 2,
    min_slots: int = 4,
    hard_cap: int = 64,
) -> tuple[list[int], int]:
    num_tasks = len(context.task_names)
    per_task_counts: list[list[int]] = [[] for _ in range(num_tasks)]

    for database_id, robot_id, week_idx in train_indices:
        series = context.series[(database_id, robot_id)]
        counts = series.counts[week_idx]
        for task_idx in range(num_tasks):
            per_task_counts[task_idx].append(int(counts[task_idx]))

    caps: list[int] = []
    for task_idx in range(num_tasks):
        arr = np.asarray(per_task_counts[task_idx], dtype=np.float32)
        if arr.size == 0:
            caps.append(int(min_slots))
            continue

        qv = float(np.quantile(arr, quantile))
        mu = float(arr.mean())
        sd = float(arr.std())
        cap = max(np.ceil(qv), np.ceil(mu + 2.0 * sd)) + margin
        cap = int(np.clip(cap, min_slots, hard_cap))
        caps.append(cap)

    return caps, int(max(caps)) if caps else int(min_slots)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    process_started_at = time.perf_counter()

    cfg = load_config(args.config)
    if args.smoke:
        cfg['training']['epochs'] = 1
        cfg['training']['batch_size'] = 2
        cfg['models']['unified']['hidden_size'] = 96
        cfg['models']['unified']['numeric_hidden_dim'] = 96
        cfg['models']['unified']['history_layers'] = 1
        cfg['models']['unified']['query_layers'] = 1
        cfg['models']['unified']['cross_layers'] = 1
        cfg['calendar']['max_slot_prototypes'] = min(4, int(cfg['calendar']['max_slot_prototypes']))
        cfg['splits']['backtest_weeks'] = min(1, int(cfg['splits'].get('backtest_weeks', 1)))

    seed_everything(int(cfg['seed']))
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_preparation_started_at = time.perf_counter()
    registry = load_registry(args.registry)
    df = load_all_events(registry, timezone_default=cfg['calendar']['timezone_default'], show_progress=True)
    profile = profile_events_dataframe(df)
    context = build_global_context(df, bin_minutes=int(cfg['calendar']['bin_minutes']), show_progress=True)
    split = build_time_split(context, train_ratio=float(cfg['splits']['train_ratio']), min_history_weeks=int(cfg['calendar']['min_history_weeks']))

    train_indices = split.train_indices
    val_indices = split.val_indices
    if args.smoke:
        train_indices = train_indices[:8]
        val_indices = val_indices[:4]

    if cfg['calendar'].get('max_slots_by_task') is None:
        hard_cap = int(cfg['calendar']['max_slot_prototypes']) if args.smoke else 64
        min_slots = 1 if args.smoke else 4
        max_slots_by_task, global_max_slots = infer_max_slots_by_task(
            context,
            train_indices,
            min_slots=min_slots,
            hard_cap=hard_cap,
        )
        cfg['calendar']['max_slots_by_task'] = max_slots_by_task
        cfg['calendar']['max_slot_prototypes'] = global_max_slots
    else:
        max_slots_by_task = [int(x) for x in cfg['calendar']['max_slots_by_task']]
        global_max_slots = int(max(max_slots_by_task))
        cfg['calendar']['max_slot_prototypes'] = global_max_slots

    if cfg['inference'].get('max_selected_slots_per_task_by_task') is None:
        cfg['inference']['max_selected_slots_per_task_by_task'] = list(cfg['calendar']['max_slots_by_task'])

    warmup_ds = UnifiedWeekSlotDataset(
        context=context,
        indices=train_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        max_slots=int(cfg['calendar']['max_slot_prototypes']),
        max_slots_by_task=cfg['calendar']['max_slots_by_task'],
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        show_progress=True,
        progress_label='Preparación datos - warmup/train stats',
    )
    history_stats = fit_history_stats_from_samples(warmup_ds.samples, key='history')
    numeric_matrix = np.concatenate([sample['numeric_features'] for sample in warmup_ds.samples], axis=0)
    numeric_stats = fit_vector_stats_from_samples([{'numeric_features': row} for row in numeric_matrix], key='numeric_features', log_indices=NUMERIC_LOG_INDICES)

    apply_occurrence_scaling(warmup_ds, history_stats, numeric_stats)
    warmup_ds.history_stats = history_stats
    warmup_ds.numeric_stats = numeric_stats
    train_ds = warmup_ds
    val_ds = UnifiedWeekSlotDataset(
        context=context,
        indices=val_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        max_slots=int(cfg['calendar']['max_slot_prototypes']),
        max_slots_by_task=cfg['calendar']['max_slots_by_task'],
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        history_stats=history_stats,
        numeric_stats=numeric_stats,
        show_progress=True,
        progress_label='Preparación datos - dataset validación',
    )

    data_preparation_seconds = time.perf_counter() - data_preparation_started_at
    print(f"[preparación datos] Completada en {format_duration(data_preparation_seconds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=int(cfg['training']['batch_size']), shuffle=True, num_workers=int(cfg['training']['num_workers']))
    val_loader = DataLoader(val_ds, batch_size=int(cfg['training']['batch_size']), shuffle=False, num_workers=int(cfg['training']['num_workers']))

    sample = train_ds[0]
    model = UnifiedSlotTransformer(
        input_dim=int(sample['history'].shape[-1]),
        numeric_feature_dim=int(sample['numeric_features'].shape[-1]),
        num_tasks=len(context.task_names),
        num_databases=len(context.database_to_idx),
        num_robots=len(context.robot_to_idx),
        max_slots=int(cfg['calendar']['max_slot_prototypes']),
        window_weeks=int(cfg['calendar']['window_weeks']),
        bins_per_day=int(24 * 60 / int(cfg['calendar']['bin_minutes'])),
        **cfg['models']['unified'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['training']['lr']),
        weight_decay=float(cfg['training']['weight_decay']),
    )
    scheduler = None
    if bool(cfg['training'].get('use_lr_scheduler', True)):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(cfg['training'].get('monitor_mode', 'max')),
            factor=float(cfg['training'].get('lr_scheduler_factor', 0.5)),
            patience=int(cfg['training'].get('lr_scheduler_patience', 5)),
            threshold=float(cfg['training'].get('lr_scheduler_threshold', 0.001)),
            threshold_mode=str(cfg['training'].get('lr_scheduler_threshold_mode', 'abs')),
            min_lr=float(cfg['training'].get('lr_scheduler_min_lr', 1.0e-5)),
        )

    logger = RunLogger(
        output_dir,
        save_epoch_jsonl=bool(cfg['reporting'].get('save_epoch_jsonl', True)),
        reset_epoch_jsonl=True,
    )
    model_fit_started_at = time.perf_counter()
    fit_info = fit_unified_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=int(cfg['training']['epochs']),
        patience=int(cfg['training']['patience']),
        logger=logger,
        num_tasks=len(context.task_names),
        max_slots=int(cfg['calendar']['max_slot_prototypes']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        bins_per_day=int(24 * 60 / int(cfg['calendar']['bin_minutes'])),
        grad_clip_norm=float(cfg['training'].get('grad_clip_norm', 1.0)),
        scheduler=scheduler,
        loss_kwargs={
            'pos_weight_active': float(cfg['training'].get('pos_weight_active', 5.0)),
            'active_loss_weight': float(cfg['training'].get('active_loss_weight', 1.0)),
            'day_loss_weight': float(cfg['training'].get('day_loss_weight', 0.20)),
            'time_loss_weight': float(cfg['training'].get('time_loss_weight', 0.20)),
            'week_time_loss_weight': float(cfg['training'].get('week_time_loss_weight', 1.20)),
            'duration_loss_weight': float(cfg['training'].get('duration_loss_weight', 0.20)),
            'count_consistency_weight': float(cfg['training'].get('count_consistency_weight', 0.35)),
            'day_label_smoothing': float(cfg['training'].get('day_label_smoothing', 0.01)),
            'time_label_smoothing': float(cfg['training'].get('time_label_smoothing', 0.01)),
        },
        print_interval=int(cfg['training'].get('progress_log_interval', 5)),
        monitor_metric=str(cfg['training'].get('monitor_metric', 'start_tol_5m')),
        monitor_mode=str(cfg['training'].get('monitor_mode', 'max')),
        monitor_weights=cfg['training'].get('monitor_weights'),
    )
    model_fit_seconds = time.perf_counter() - model_fit_started_at

    if bool(cfg['reporting'].get('save_csv', True)):
        logger.save_csv()

    metrics_rows = logger.rows
    val_metrics = {}
    if metrics_rows:
        target_epoch = int(fit_info.get('best_epoch', -1))
        best_row = next((row for row in metrics_rows if int(row.get('epoch', -1)) == target_epoch), metrics_rows[-1])
        val_metrics = {k[4:]: v for k, v in best_row.items() if k.startswith('val_')}

    save_checkpoint(
        output_dir / 'unified_model.pt',
        model,
        metadata={
            'config': cfg,
            'history_dim': int(sample['history'].shape[-1]),
            'numeric_feature_dim': int(sample['numeric_features'].shape[-1]),
            'feature_scaling': {
                'history': history_stats,
                'numeric': numeric_stats,
            },
            'num_tasks': len(context.task_names),
            'num_databases': len(context.database_to_idx),
            'num_robots': len(context.robot_to_idx),
            'max_slots': int(cfg['calendar']['max_slot_prototypes']),
            'max_slots_by_task': cfg['calendar'].get('max_slots_by_task'),
            'max_selected_slots_per_task_by_task': cfg['inference'].get('max_selected_slots_per_task_by_task'),
        },
    )

    backtest_indices = val_indices[-int(cfg['splits'].get('backtest_weeks', 0)):] if val_indices else []
    _, backtest_summary = run_holdout_backtest(
        context=context,
        val_indices=backtest_indices,
        config=cfg,
        model=model,
        device=device,
        output_dir=output_dir,
        feature_scaling={
            'history': history_stats,
            'numeric': numeric_stats,
        },
    )

    total_process_seconds = time.perf_counter() - process_started_at
    timings = {
        'data_preparation_seconds': float(data_preparation_seconds),
        'model_fit_seconds': float(model_fit_seconds),
        'total_process_seconds': float(total_process_seconds),
    }
    fit_info.update(timings)

    summary = {
        'profile': profile,
        'training': fit_info,
        'validation': val_metrics,
        'backtest': backtest_summary,
        'timings': timings,
    }
    save_summary(summary, output_dir)
    build_final_report(summary, output_dir)
    (output_dir / 'used_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print_final_training_summary(
        output_dir=output_dir,
        fit_info=fit_info,
        val_metrics=val_metrics,
        backtest_summary=backtest_summary,
        timings=timings,
    )
    print(f'Training completado. Artefactos en {output_dir}')


if __name__ == '__main__':
    main()
