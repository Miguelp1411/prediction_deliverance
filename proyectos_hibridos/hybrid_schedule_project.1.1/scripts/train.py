from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry, profile_events_dataframe
from hybrid_schedule.data.features import build_global_context
from hybrid_schedule.models import OccurrenceResidualNet, TemporalResidualNet
from hybrid_schedule.reporting import RunLogger, build_final_report, save_summary, save_training_plots
from hybrid_schedule.training import (
    OccurrenceDataset,
    TemporalDataset,
    build_balanced_sample_weights,
    build_time_split,
    fit_occurrence_model,
    fit_temporal_model,
    run_holdout_backtest,
    run_leave_one_database_out_backtest,
    run_occurrence_selector_backtest,
    run_schedule_selector_backtest,
)
from hybrid_schedule.utils import get_device, save_checkpoint, seed_everything



def _selector_subset(indices: list[tuple[str, str, int]], max_weeks_per_series: int) -> list[tuple[str, str, int]]:
    grouped: dict[tuple[str, str], list[tuple[str, str, int]]] = defaultdict(list)
    for row in indices:
        grouped[(row[0], row[1])].append(row)
    subset: list[tuple[str, str, int]] = []
    for rows in grouped.values():
        rows = sorted(rows, key=lambda x: x[2])
        subset.extend(rows[-max_weeks_per_series:])
    subset.sort(key=lambda x: (x[0], x[1], x[2]))
    return subset



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg['training']['epochs_occurrence'] = 1
        cfg['training']['epochs_temporal'] = 1
        cfg['training']['batch_size_occurrence'] = 16
        cfg['training']['batch_size_temporal'] = 32
        cfg['models']['occurrence']['hidden_size'] = 48
        cfg['models']['temporal']['hidden_size'] = 64
        cfg['splits']['backtest_weeks'] = min(4, int(cfg['splits']['backtest_weeks']))
        cfg['splits']['selector_weeks'] = min(2, int(cfg['splits']['selector_weeks']))
    seed_everything(int(cfg['seed']))
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
        train_indices = train_indices[:40]
        val_indices = val_indices[:12]

    occurrence_ds_train = OccurrenceDataset(context, train_indices, window_weeks=int(cfg['calendar']['window_weeks']), topk_templates=int(cfg['calendar']['topk_templates']), max_delta=int(cfg['models']['occurrence']['max_delta']))
    occurrence_ds_val = OccurrenceDataset(context, val_indices, window_weeks=int(cfg['calendar']['window_weeks']), topk_templates=int(cfg['calendar']['topk_templates']), max_delta=int(cfg['models']['occurrence']['max_delta']))
    temporal_ds_train = TemporalDataset(context, train_indices, window_weeks=int(cfg['calendar']['window_weeks']), topk_templates=int(cfg['calendar']['topk_templates']), bins_per_day=(24 * 60) // int(cfg['calendar']['bin_minutes']))
    temporal_ds_val = TemporalDataset(context, val_indices, window_weeks=int(cfg['calendar']['window_weeks']), topk_templates=int(cfg['calendar']['topk_templates']), bins_per_day=(24 * 60) // int(cfg['calendar']['bin_minutes']))

    if len(occurrence_ds_train) == 0 or len(occurrence_ds_val) == 0:
        raise RuntimeError('No hay suficientes semanas para entrenar occurrence')
    if len(temporal_ds_train) == 0 or len(temporal_ds_val) == 0:
        raise RuntimeError('No hay suficientes semanas para entrenar temporal')

    occ_sampler = None
    tmp_sampler = None
    if bool(cfg['training'].get('balanced_sampling', True)):
        occ_weights = build_balanced_sample_weights(train_indices)
        occ_sampler = WeightedRandomSampler(weights=torch.tensor(np.repeat(occ_weights, len(context.task_names)), dtype=torch.double), num_samples=len(occurrence_ds_train), replacement=True) if len(occurrence_ds_train) > 0 else None

    train_loader_occ = DataLoader(occurrence_ds_train, batch_size=int(cfg['training']['batch_size_occurrence']), shuffle=(occ_sampler is None), sampler=occ_sampler, num_workers=int(cfg['training']['num_workers']))
    val_loader_occ = DataLoader(occurrence_ds_val, batch_size=int(cfg['training']['batch_size_occurrence']), shuffle=False, num_workers=int(cfg['training']['num_workers']))
    train_loader_tmp = DataLoader(temporal_ds_train, batch_size=int(cfg['training']['batch_size_temporal']), shuffle=(tmp_sampler is None), sampler=tmp_sampler, num_workers=int(cfg['training']['num_workers']))
    val_loader_tmp = DataLoader(temporal_ds_val, batch_size=int(cfg['training']['batch_size_temporal']), shuffle=False, num_workers=int(cfg['training']['num_workers']))

    sample_occ = occurrence_ds_train[0]
    input_dim = int(sample_occ['history'].shape[-1])
    bins_per_day = (24 * 60) // int(cfg['calendar']['bin_minutes'])

    occ_model = OccurrenceResidualNet(
        input_dim=input_dim,
        num_tasks=len(context.task_names),
        num_databases=len(context.database_to_idx),
        num_robots=len(context.robot_to_idx),
        hidden_size=int(cfg['models']['occurrence']['hidden_size']),
        num_layers=int(cfg['models']['occurrence']['num_layers']),
        dropout=float(cfg['models']['occurrence']['dropout']),
        task_embed_dim=int(cfg['models']['occurrence']['task_embed_dim']),
        database_embed_dim=int(cfg['models']['occurrence']['database_embed_dim']),
        robot_embed_dim=int(cfg['models']['occurrence']['robot_embed_dim']),
        max_delta=int(cfg['models']['occurrence']['max_delta']),
        numeric_dim=int(sample_occ['numeric_features'].shape[-1]),
    ).to(device)

    sample_tmp = temporal_ds_train[0]
    tmp_model = TemporalResidualNet(
        input_dim=input_dim,
        num_tasks=len(context.task_names),
        num_databases=len(context.database_to_idx),
        num_robots=len(context.robot_to_idx),
        hidden_size=int(cfg['models']['temporal']['hidden_size']),
        num_layers=int(cfg['models']['temporal']['num_layers']),
        dropout=float(cfg['models']['temporal']['dropout']),
        task_embed_dim=int(cfg['models']['temporal']['task_embed_dim']),
        database_embed_dim=int(cfg['models']['temporal']['database_embed_dim']),
        robot_embed_dim=int(cfg['models']['temporal']['robot_embed_dim']),
        bins_per_day=bins_per_day,
        numeric_dim=int(sample_tmp['numeric_features'].shape[-1]),
    ).to(device)

    logger = RunLogger(output_dir)

    occ_optimizer = torch.optim.AdamW(occ_model.parameters(), lr=float(cfg['training']['lr_occurrence']), weight_decay=float(cfg['training']['weight_decay']))
    tmp_optimizer = torch.optim.AdamW(tmp_model.parameters(), lr=float(cfg['training']['lr_temporal']), weight_decay=float(cfg['training']['weight_decay']))

    selector_indices = _selector_subset(val_indices, max_weeks_per_series=int(cfg['splits'].get('selector_weeks', 8)))
    selector_every = int(cfg['training'].get('selector_every', 1))

    occ_model, occ_metrics, _occ_rows = fit_occurrence_model(
        occ_model,
        train_loader_occ,
        val_loader_occ,
        occ_optimizer,
        device,
        epochs=int(cfg['training']['epochs_occurrence']),
        patience=int(cfg['training']['patience']),
        min_delta=float(cfg['training'].get('min_delta', 1e-5)),
        logger=logger,
        max_delta=int(cfg['models']['occurrence']['max_delta']),
        selector_fn=lambda model: run_occurrence_selector_backtest(context, selector_indices, cfg, model, device),
        selector_every=selector_every,
    )

    tmp_model, tmp_metrics, _tmp_rows = fit_temporal_model(
        tmp_model,
        train_loader_tmp,
        val_loader_tmp,
        tmp_optimizer,
        device,
        epochs=int(cfg['training']['epochs_temporal']),
        patience=int(cfg['training']['patience']),
        min_delta=float(cfg['training'].get('min_delta', 1e-5)),
        logger=logger,
        bins_per_day=bins_per_day,
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        selector_fn=lambda model: run_schedule_selector_backtest(context, selector_indices, cfg, occ_model, model, device),
        selector_every=selector_every,
    )

    logger.save_csv()
    save_training_plots(logger.rows, output_dir)

    backtest_indices = val_indices[-int(cfg['splits']['backtest_weeks']):] if val_indices else []
    _, backtest_summary = run_holdout_backtest(context, backtest_indices, cfg, occ_model, tmp_model, device, output_dir)
    leave_one_out = run_leave_one_database_out_backtest(context, cfg, occ_model, tmp_model, device, output_dir)

    save_checkpoint(output_dir / 'occurrence_model.pt', occ_model, {'config': cfg, 'input_dim': input_dim})
    save_checkpoint(output_dir / 'temporal_model.pt', tmp_model, {'config': cfg, 'input_dim': input_dim})

    summary = {
        'profile': profile,
        'occurrence': occ_metrics,
        'temporal': tmp_metrics,
        'backtest': backtest_summary,
        'leave_one_database_out': leave_one_out,
        'artifacts': {
            'occurrence_checkpoint': str((output_dir / 'occurrence_model.pt').resolve()),
            'temporal_checkpoint': str((output_dir / 'temporal_model.pt').resolve()),
        },
    }
    save_summary(summary, output_dir)
    build_final_report(summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
