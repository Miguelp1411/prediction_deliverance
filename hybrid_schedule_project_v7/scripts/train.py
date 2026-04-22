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
from hybrid_schedule.data.scaling import (
    OCCURRENCE_NUMERIC_LOG_INDICES,
    TEMPORAL_CANDIDATE_LOG_INDICES,
    TEMPORAL_NUMERIC_LOG_INDICES,
    apply_occurrence_scaling,
    apply_temporal_scaling,
    fit_candidate_stats_from_samples,
    fit_history_stats_from_samples,
    fit_vector_stats_from_samples,
)
from hybrid_schedule.models import OccurrenceResidualNet, TemporalDirectNet, TemporalRankingNet
from hybrid_schedule.reporting import RunLogger, build_final_report, save_summary, save_training_plots
from hybrid_schedule.training import (
    DirectTemporalDataset,
    OccurrenceDataset,
    TemporalDataset,
    build_time_split,
    fit_occurrence_model,
    fit_temporal_direct_model,
    fit_temporal_model,
)
from hybrid_schedule.training.datasets import predict_occurrence_counts_for_indices
from hybrid_schedule.training.backtesting import run_holdout_backtest
from hybrid_schedule.utils import get_device, save_checkpoint, seed_everything


_DIRECT_ARCHS = {'direct', 'direct_day_time', 'day_time_direct'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    cfg = load_config(args.config)
    temporal_arch = str(cfg['models']['temporal'].get('architecture', 'direct_day_time')).strip().lower()
    if args.smoke:
        cfg['training']['epochs_occurrence'] = 1
        cfg['training']['epochs_temporal'] = 1
        cfg['training']['batch_size_occurrence'] = 16
        cfg['training']['batch_size_temporal'] = 32
        cfg['models']['occurrence']['hidden_size'] = 48
        cfg['models']['temporal']['hidden_size'] = 96 if temporal_arch in _DIRECT_ARCHS else 64
        cfg['splits']['backtest_weeks'] = min(1, int(cfg['splits']['backtest_weeks']))
        cfg['scheduler']['backend'] = 'greedy'
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

    occurrence_ds_train = OccurrenceDataset(
        context,
        train_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        topk_templates=int(cfg['calendar']['topk_templates']),
        max_delta=int(cfg['models']['occurrence']['max_delta']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        use_lag52=bool(cfg['models']['occurrence'].get('use_lag52', False)),
    )
    occurrence_ds_val = OccurrenceDataset(
        context,
        val_indices,
        window_weeks=int(cfg['calendar']['window_weeks']),
        topk_templates=int(cfg['calendar']['topk_templates']),
        max_delta=int(cfg['models']['occurrence']['max_delta']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        use_lag52=bool(cfg['models']['occurrence'].get('use_lag52', False)),
    )
    if len(occurrence_ds_train) == 0 or len(occurrence_ds_val) == 0:
        raise RuntimeError('No hay suficientes semanas para entrenar occurrence')

    history_stats = fit_history_stats_from_samples(occurrence_ds_train.samples, key='history')
    occurrence_numeric_stats = fit_vector_stats_from_samples(occurrence_ds_train.samples, key='numeric_features', log_indices=OCCURRENCE_NUMERIC_LOG_INDICES)
    apply_occurrence_scaling(occurrence_ds_train, history_stats, occurrence_numeric_stats)
    apply_occurrence_scaling(occurrence_ds_val, history_stats, occurrence_numeric_stats)
    cfg.setdefault('feature_scaling', {})
    cfg['feature_scaling']['history'] = history_stats
    cfg['feature_scaling']['occurrence_numeric'] = occurrence_numeric_stats

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
    occ_scheduler = None
    if bool(cfg['training'].get('use_lr_scheduler', True)):
        occ_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            occ_optimizer,
            mode='min',
            factor=float(cfg['training'].get('lr_scheduler_factor', 0.5)),
            patience=int(cfg['training'].get('lr_scheduler_patience', 6)),
            min_lr=float(cfg['training'].get('lr_scheduler_min_lr', 1e-5)),
            threshold=float(cfg['training'].get('lr_scheduler_threshold', 1e-3)),
        )
    occ_model, occ_metrics, _occ_rows = fit_occurrence_model(
        occ_model,
        train_loader_occ,
        val_loader_occ,
        occ_optimizer,
        device,
        epochs=int(cfg['training']['epochs_occurrence']),
        patience=int(cfg['training']['patience']),
        logger=logger,
        max_delta=int(cfg['models']['occurrence']['max_delta']),
        change_loss_weight=float(cfg['models']['occurrence'].get('change_loss_weight', 0.25)),
        expected_count_mae_weight=float(cfg['models']['occurrence'].get('expected_count_mae_weight', 0.15)),
        delta_reg_weight=float(cfg['models']['occurrence'].get('delta_reg_weight', 0.05)),
        label_smoothing=float(cfg['models']['occurrence'].get('label_smoothing', 0.0)),
        grad_clip_norm=float(cfg['training'].get('grad_clip_norm', 0.0)),
        scheduler=occ_scheduler,
    )

    count_lookup = predict_occurrence_counts_for_indices(
        context,
        train_indices + val_indices,
        occ_model,
        device,
        window_weeks=int(cfg['calendar']['window_weeks']),
        topk_templates=int(cfg['calendar']['topk_templates']),
        bin_minutes=int(cfg['calendar']['bin_minutes']),
        feature_scaling=cfg.get('feature_scaling'),
        use_lag52=bool(cfg['models']['occurrence'].get('use_lag52', False)),
    )

    temporal_numeric_stats = None
    temporal_candidate_stats = None
    tmp_candidate_dim = None
    bins_per_day = int((24 * 60) / int(cfg['calendar']['bin_minutes']))

    if temporal_arch in _DIRECT_ARCHS:
        temporal_ds_train = DirectTemporalDataset(
            context,
            train_indices,
            window_weeks=int(cfg['calendar']['window_weeks']),
            topk_templates=int(cfg['calendar']['topk_templates']),
            max_slot_prototypes=int(cfg['calendar'].get('max_slot_prototypes', 32)),
            count_lookup=count_lookup,
            bin_minutes=int(cfg['calendar']['bin_minutes']),
        )
        temporal_ds_val = DirectTemporalDataset(
            context,
            val_indices,
            window_weeks=int(cfg['calendar']['window_weeks']),
            topk_templates=int(cfg['calendar']['topk_templates']),
            max_slot_prototypes=int(cfg['calendar'].get('max_slot_prototypes', 32)),
            count_lookup=count_lookup,
            bin_minutes=int(cfg['calendar']['bin_minutes']),
        )
        if len(temporal_ds_train) == 0 or len(temporal_ds_val) == 0:
            raise RuntimeError('No hay suficientes semanas para entrenar temporal directo')
        temporal_numeric_stats = fit_vector_stats_from_samples(temporal_ds_train.samples, key='numeric_features', log_indices=TEMPORAL_NUMERIC_LOG_INDICES)
        apply_occurrence_scaling(temporal_ds_train, history_stats, temporal_numeric_stats)
        apply_occurrence_scaling(temporal_ds_val, history_stats, temporal_numeric_stats)
        cfg['feature_scaling']['temporal_numeric'] = temporal_numeric_stats
        cfg['feature_scaling']['temporal_candidate'] = None

        train_loader_tmp = DataLoader(temporal_ds_train, batch_size=int(cfg['training']['batch_size_temporal']), shuffle=True, num_workers=int(cfg['training']['num_workers']))
        val_loader_tmp = DataLoader(temporal_ds_val, batch_size=int(cfg['training']['batch_size_temporal']), shuffle=False, num_workers=int(cfg['training']['num_workers']))
        sample_tmp = temporal_ds_train[0]
        tmp_numeric_dim = int(sample_tmp['numeric_features'].shape[-1])
        tmp_model = TemporalDirectNet(
            input_dim=input_dim,
            num_tasks=len(context.task_names),
            num_databases=len(context.database_to_idx),
            num_robots=len(context.robot_to_idx),
            window_weeks=int(cfg['calendar']['window_weeks']),
            bins_per_day=bins_per_day,
            max_slots=int(cfg['calendar'].get('max_slot_prototypes', 32)),
            hidden_size=int(cfg['models']['temporal']['hidden_size']),
            num_layers=int(cfg['models']['temporal']['num_layers']),
            num_heads=int(cfg['models']['temporal'].get('num_heads', 4)),
            dropout=float(cfg['models']['temporal']['dropout']),
            task_embed_dim=int(cfg['models']['temporal'].get('task_embed_dim', 32)),
            database_embed_dim=int(cfg['models']['temporal'].get('database_embed_dim', 16)),
            robot_embed_dim=int(cfg['models']['temporal'].get('robot_embed_dim', 16)),
            slot_embed_dim=int(cfg['models']['temporal'].get('slot_embed_dim', 24)),
            day_embed_dim=int(cfg['models']['temporal'].get('day_embed_dim', 8)),
            time_embed_dim=int(cfg['models']['temporal'].get('time_embed_dim', 16)),
            numeric_feature_dim=tmp_numeric_dim,
        ).to(device)
    else:
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
            candidate_source_quotas=cfg['models']['temporal'].get('candidate_source_quotas', {}),
            prototype_start_offsets=cfg['models']['temporal'].get('prototype_start_offsets_bins'),
            prototype_duration_offsets=cfg['models']['temporal'].get('prototype_duration_offsets_bins'),
            regime_lookback_weeks=int(cfg['models']['temporal'].get('regime_lookback_weeks', 8)),
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
            candidate_source_quotas=cfg['models']['temporal'].get('candidate_source_quotas', {}),
            prototype_start_offsets=cfg['models']['temporal'].get('prototype_start_offsets_bins'),
            prototype_duration_offsets=cfg['models']['temporal'].get('prototype_duration_offsets_bins'),
            regime_lookback_weeks=int(cfg['models']['temporal'].get('regime_lookback_weeks', 8)),
        )
        if len(temporal_ds_train) == 0 or len(temporal_ds_val) == 0:
            raise RuntimeError('No hay suficientes semanas para entrenar temporal')
        temporal_numeric_stats = fit_vector_stats_from_samples(temporal_ds_train.samples, key='numeric_features', log_indices=TEMPORAL_NUMERIC_LOG_INDICES)
        temporal_candidate_stats = fit_candidate_stats_from_samples(temporal_ds_train.samples, feature_key='candidate_features', mask_key='candidate_mask', log_indices=TEMPORAL_CANDIDATE_LOG_INDICES)
        apply_temporal_scaling(temporal_ds_train, history_stats, temporal_numeric_stats, temporal_candidate_stats)
        apply_temporal_scaling(temporal_ds_val, history_stats, temporal_numeric_stats, temporal_candidate_stats)
        cfg['feature_scaling']['temporal_numeric'] = temporal_numeric_stats
        cfg['feature_scaling']['temporal_candidate'] = temporal_candidate_stats

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
    tmp_scheduler = None
    if bool(cfg['training'].get('use_lr_scheduler', True)):
        tmp_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            tmp_optimizer,
            mode='min',
            factor=float(cfg['training'].get('lr_scheduler_factor', 0.5)),
            patience=int(cfg['training'].get('lr_scheduler_patience', 6)),
            min_lr=float(cfg['training'].get('lr_scheduler_min_lr', 1e-5)),
            threshold=float(cfg['training'].get('lr_scheduler_threshold', 1e-3)),
        )

    if temporal_arch in _DIRECT_ARCHS:
        tmp_model, tmp_metrics, _tmp_rows = fit_temporal_direct_model(
            tmp_model,
            train_loader_tmp,
            val_loader_tmp,
            tmp_optimizer,
            device,
            epochs=int(cfg['training']['epochs_temporal']),
            patience=int(cfg['training']['patience']),
            logger=logger,
            bin_minutes=int(cfg['calendar']['bin_minutes']),
            bins_per_day=bins_per_day,
            day_loss_weight=float(cfg['models']['temporal'].get('day_loss_weight', 0.80)),
            time_loss_weight=float(cfg['models']['temporal'].get('time_loss_weight', 1.00)),
            duration_loss_weight=float(cfg['models']['temporal'].get('duration_loss_weight', 0.20)),
            day_label_smoothing=float(cfg['models']['temporal'].get('day_label_smoothing', 0.02)),
            time_label_smoothing=float(cfg['models']['temporal'].get('time_label_smoothing', 0.01)),
            grad_clip_norm=float(cfg['training'].get('grad_clip_norm', 0.0)),
            scheduler=tmp_scheduler,
        )
        temporal_model_type = 'direct_day_time'
    else:
        tmp_model, tmp_metrics, _tmp_rows = fit_temporal_model(
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
            label_smoothing=float(cfg['models']['temporal'].get('label_smoothing', 0.0)),
            confidence_penalty_weight=float(cfg['models']['temporal'].get('confidence_penalty_weight', 0.02)),
            anchor_deviation_weight=float(cfg['models']['temporal'].get('anchor_deviation_weight', 0.03)),
            duration_deviation_weight=float(cfg['models']['temporal'].get('duration_deviation_weight', 0.01)),
            grad_clip_norm=float(cfg['training'].get('grad_clip_norm', 0.0)),
            scheduler=tmp_scheduler,
        )
        temporal_model_type = 'ranking'

    logger.save_csv()
    save_training_plots(logger.rows, output_dir)

    backtest_indices = val_indices[-int(cfg['splits']['backtest_weeks']):] if val_indices else []
    _, backtest_summary = run_holdout_backtest(context, backtest_indices, cfg, occ_model, tmp_model, device, output_dir)

    save_checkpoint(output_dir / 'occurrence_model.pt', occ_model, {'config': cfg, 'input_dim': input_dim, 'occ_numeric_dim': occ_numeric_dim, 'feature_schema_version': FEATURE_SCHEMA_VERSION})
    tmp_metadata = {
        'config': cfg,
        'input_dim': input_dim,
        'tmp_numeric_dim': tmp_numeric_dim,
        'temporal_model_type': temporal_model_type,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
    }
    if tmp_candidate_dim is not None:
        tmp_metadata['tmp_candidate_dim'] = tmp_candidate_dim
    save_checkpoint(output_dir / 'temporal_model.pt', tmp_model, tmp_metadata)

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
