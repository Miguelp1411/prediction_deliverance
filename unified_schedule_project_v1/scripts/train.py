from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry, profile_events_dataframe
from hybrid_schedule.data.features import build_global_context
from hybrid_schedule.data.scaling import (
    fit_history_stats_from_samples,
    fit_vector_stats_from_samples,
)
from hybrid_schedule.models import UnifiedSlotTransformer
from hybrid_schedule.reporting import RunLogger, build_final_report, save_summary
from hybrid_schedule.training import UnifiedWeekSlotDataset, build_time_split, fit_unified_model
from hybrid_schedule.utils import get_device, save_checkpoint, seed_everything


NUMERIC_LOG_INDICES = [3, 5, 10, 11, 13, 15, 16, 17, 19, 21, 22, 25, 26, 27, 28, 29, 35, 36, 37, 39, 45, 46, 47, 48, 49, 50]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

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

    seed_everything(int(cfg['seed']))
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(args.registry)
    df = load_all_events(registry, timezone_default=cfg['calendar']['timezone_default'])
    profile = profile_events_dataframe(df)
    context = build_global_context(df, bin_minutes=int(cfg['calendar']['bin_minutes']))
    split = build_time_split(context, train_ratio=float(cfg['splits']['train_ratio']), min_history_weeks=int(cfg['calendar']['min_history_weeks']))

    train_indices = split.train_indices
    val_indices = split.val_indices
    if args.smoke:
        train_indices = train_indices[:8]
        val_indices = val_indices[:4]

    warmup_ds = UnifiedWeekSlotDataset(
        context=context,
        indices=train_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        max_slots=int(cfg['calendar']['max_slot_prototypes']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
    )
    history_stats = fit_history_stats_from_samples(warmup_ds.samples, key='history')
    numeric_matrix = np.concatenate([sample['numeric_features'] for sample in warmup_ds.samples], axis=0)
    numeric_stats = fit_vector_stats_from_samples([{'numeric_features': row} for row in numeric_matrix], key='numeric_features', log_indices=NUMERIC_LOG_INDICES)

    train_ds = UnifiedWeekSlotDataset(
        context=context,
        indices=train_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        max_slots=int(cfg['calendar']['max_slot_prototypes']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        history_stats=history_stats,
        numeric_stats=numeric_stats,
    )
    val_ds = UnifiedWeekSlotDataset(
        context=context,
        indices=val_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        max_slots=int(cfg['calendar']['max_slot_prototypes']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        history_stats=history_stats,
        numeric_stats=numeric_stats,
    )

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
            mode='min',
            factor=float(cfg['training'].get('lr_scheduler_factor', 0.5)),
            patience=int(cfg['training'].get('lr_scheduler_patience', 5)),
            threshold=float(cfg['training'].get('lr_scheduler_threshold', 0.001)),
            min_lr=float(cfg['training'].get('lr_scheduler_min_lr', 1.0e-5)),
        )

    logger = RunLogger(output_dir)
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
            'day_loss_weight': float(cfg['training'].get('day_loss_weight', 0.70)),
            'time_loss_weight': float(cfg['training'].get('time_loss_weight', 1.0)),
            'duration_loss_weight': float(cfg['training'].get('duration_loss_weight', 0.20)),
            'count_consistency_weight': float(cfg['training'].get('count_consistency_weight', 0.35)),
            'day_label_smoothing': float(cfg['training'].get('day_label_smoothing', 0.01)),
            'time_label_smoothing': float(cfg['training'].get('time_label_smoothing', 0.01)),
        },
    )

    if bool(cfg['reporting'].get('save_csv', True)):
        logger.save_csv()

    metrics_rows = logger.rows
    val_metrics = {}
    if metrics_rows:
        best_row = min(metrics_rows, key=lambda row: float(row.get('val_loss', float('inf'))))
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
        },
    )

    summary = {
        'profile': profile,
        'training': fit_info,
        'validation': val_metrics,
    }
    save_summary(summary, output_dir)
    build_final_report(summary, output_dir)
    (output_dir / 'used_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Training completado. Artefactos en {output_dir}')


if __name__ == '__main__':
    main()
