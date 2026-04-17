from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry, profile_events_dataframe
from hybrid_schedule.data.features import FEATURE_SCHEMA_VERSION, build_global_context
from hybrid_schedule.models import OccurrenceResidualNet, TemporalRankingNet
from hybrid_schedule.reporting import RunLogger, build_final_report, save_summary, save_training_plots
from hybrid_schedule.training import OccurrenceDataset, TemporalDataset, build_time_split, fit_occurrence_model, fit_temporal_model
from hybrid_schedule.training.datasets import predict_occurrence_counts_for_indices
from hybrid_schedule.training.backtesting import run_holdout_backtest
from hybrid_schedule.utils import get_device, save_checkpoint, seed_everything


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
        cfg['splits']['backtest_weeks'] = min(1, int(cfg['splits']['backtest_weeks']))
        cfg['scheduler']['use_exact_milp'] = False
        cfg['scheduler']['max_solver_seconds'] = 3
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
        train_indices = train_indices[:40]
        val_indices = val_indices[:12]

    occurrence_ds_train = OccurrenceDataset(context, train_indices, window_weeks=int(cfg['calendar']['window_weeks']), topk_templates=int(cfg['calendar']['topk_templates']), max_delta=int(cfg['models']['occurrence']['max_delta']), bin_minutes=int(cfg['calendar']['bin_minutes']))
    occurrence_ds_val = OccurrenceDataset(context, val_indices, window_weeks=int(cfg['calendar']['window_weeks']), topk_templates=int(cfg['calendar']['topk_templates']), max_delta=int(cfg['models']['occurrence']['max_delta']), bin_minutes=int(cfg['calendar']['bin_minutes']))

    if len(occurrence_ds_train) == 0 or len(occurrence_ds_val) == 0:
        raise RuntimeError('No hay suficientes semanas para entrenar occurrence')

    train_loader_occ = DataLoader(occurrence_ds_train, batch_size=int(cfg['training']['batch_size_occurrence']), shuffle=True, num_workers=int(cfg['training']['num_workers']))
    val_loader_occ = DataLoader(occurrence_ds_val, batch_size=int(cfg['training']['batch_size_occurrence']), shuffle=False, num_workers=int(cfg['training']['num_workers']))

    sample_occ = occurrence_ds_train[0]
    input_dim = int(sample_occ['history'].shape[-1])
    occ_numeric_dim = int(sample_occ['numeric_features'].shape[-1])

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
        numeric_feature_dim=occ_numeric_dim,
    ).to(device)

    logger = RunLogger(output_dir)
    occ_optimizer = torch.optim.AdamW(occ_model.parameters(), lr=float(cfg['training']['lr_occurrence']), weight_decay=float(cfg['training']['weight_decay']))
    occ_model, occ_metrics, occ_rows = fit_occurrence_model(
        occ_model,
        train_loader_occ,
        val_loader_occ,
        occ_optimizer,
        device,
        epochs=int(cfg['training']['epochs_occurrence']),
        patience=int(cfg['training']['patience']),
        logger=logger,
        max_delta=int(cfg['models']['occurrence']['max_delta']),
    )

    count_lookup = predict_occurrence_counts_for_indices(
        context,
        train_indices + val_indices,
        occ_model,
        device,
        window_weeks=int(cfg['calendar']['window_weeks']),
        topk_templates=int(cfg['calendar']['topk_templates']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
    )

    temporal_ds_train = TemporalDataset(
        context,
        train_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        topk_templates=int(cfg['calendar']['topk_templates']),
        candidate_topk_templates=int(cfg['models']['temporal'].get('candidate_topk_templates', cfg['calendar'].get('temporal_candidate_topk_templates', 20))),
        candidate_neighbor_radius=int(cfg['models']['temporal'].get('candidate_neighbor_radius', 1)),
        max_candidates=int(cfg['models']['temporal'].get('max_candidates', 32)),
        start_temperature_bins=float(cfg['models']['temporal'].get('start_temperature_bins', 2.0)),
        duration_temperature_bins=float(cfg['models']['temporal'].get('duration_temperature_bins', 1.0)),
        duration_cost_weight=float(cfg['models']['temporal'].get('duration_cost_weight', 0.35)),
        max_slot_prototypes=int(cfg['calendar'].get('max_slot_prototypes', 32)),
        count_lookup=count_lookup,
        bin_minutes=int(cfg['calendar']['bin_minutes']),
    )
    temporal_ds_val = TemporalDataset(
        context,
        val_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        topk_templates=int(cfg['calendar']['topk_templates']),
        candidate_topk_templates=int(cfg['models']['temporal'].get('candidate_topk_templates', cfg['calendar'].get('temporal_candidate_topk_templates', 20))),
        candidate_neighbor_radius=int(cfg['models']['temporal'].get('candidate_neighbor_radius', 1)),
        max_candidates=int(cfg['models']['temporal'].get('max_candidates', 32)),
        start_temperature_bins=float(cfg['models']['temporal'].get('start_temperature_bins', 2.0)),
        duration_temperature_bins=float(cfg['models']['temporal'].get('duration_temperature_bins', 1.0)),
        duration_cost_weight=float(cfg['models']['temporal'].get('duration_cost_weight', 0.35)),
        max_slot_prototypes=int(cfg['calendar'].get('max_slot_prototypes', 32)),
        count_lookup=count_lookup,
        bin_minutes=int(cfg['calendar']['bin_minutes']),
    )
    if len(temporal_ds_train) == 0 or len(temporal_ds_val) == 0:
        raise RuntimeError('No hay suficientes semanas para entrenar temporal')

    train_loader_tmp = DataLoader(temporal_ds_train, batch_size=int(cfg['training']['batch_size_temporal']), shuffle=True, num_workers=int(cfg['training']['num_workers']))
    val_loader_tmp = DataLoader(temporal_ds_val, batch_size=int(cfg['training']['batch_size_temporal']), shuffle=False, num_workers=int(cfg['training']['num_workers']))

    sample_tmp = temporal_ds_train[0]
    tmp_numeric_dim = int(sample_tmp['numeric_features'].shape[-1])
    tmp_candidate_dim = int(sample_tmp['candidate_features'].shape[-1])
    tmp_model = TemporalRankingNet(
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
        numeric_feature_dim=tmp_numeric_dim,
        candidate_feature_dim=tmp_candidate_dim,
    ).to(device)

    tmp_optimizer = torch.optim.AdamW(tmp_model.parameters(), lr=float(cfg['training']['lr_temporal']), weight_decay=float(cfg['training']['weight_decay']))
    tmp_model, tmp_metrics, tmp_rows = fit_temporal_model(
        tmp_model,
        train_loader_tmp,
        val_loader_tmp,
        tmp_optimizer,
        device,
        epochs=int(cfg['training']['epochs_temporal']),
        patience=int(cfg['training']['patience']),
        logger=logger,
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        expected_cost_weight=float(cfg['models']['temporal'].get('expected_cost_weight', 0.30)),
    )

    logger.save_csv()
    save_training_plots(logger.rows, output_dir)

    backtest_indices = val_indices[-int(cfg['splits']['backtest_weeks']):] if val_indices else []
    _, backtest_summary = run_holdout_backtest(context, backtest_indices, cfg, occ_model, tmp_model, device, output_dir)

    save_checkpoint(output_dir / 'occurrence_model.pt', occ_model, {'config': cfg, 'input_dim': input_dim, 'occ_numeric_dim': occ_numeric_dim, 'feature_schema_version': FEATURE_SCHEMA_VERSION})
    save_checkpoint(output_dir / 'temporal_model.pt', tmp_model, {'config': cfg, 'input_dim': input_dim, 'tmp_numeric_dim': tmp_numeric_dim, 'tmp_candidate_dim': tmp_candidate_dim, 'temporal_model_type': 'ranking', 'feature_schema_version': FEATURE_SCHEMA_VERSION})

    summary = {
        'profile': profile,
        'occurrence': occ_metrics,
        'temporal': tmp_metrics,
        'backtest': backtest_summary,
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
