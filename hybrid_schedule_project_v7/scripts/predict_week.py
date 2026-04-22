from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry
from hybrid_schedule.data.features import FEATURE_SCHEMA_VERSION, build_global_context, build_future_history_tensor
from hybrid_schedule.inference import HybridWeekPredictor
from hybrid_schedule.models import OccurrenceResidualNet, TemporalDirectNet, TemporalRankingNet
from hybrid_schedule.utils import get_device, read_checkpoint


_DIRECT_TYPES = {'direct_day_time', 'direct', 'day_time_direct'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--database-id', required=True)
    parser.add_argument('--occurrence-ckpt', default=None)
    parser.add_argument('--temporal-ckpt', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    device = get_device()

    registry = load_registry(args.registry)
    df = load_all_events(registry, timezone_default=cfg['calendar']['timezone_default'])
    context = build_global_context(df, bin_minutes=int(cfg['calendar']['bin_minutes']))

    occ_ckpt = Path(args.occurrence_ckpt or (output_dir / 'occurrence_model.pt'))
    tmp_ckpt = Path(args.temporal_ckpt or (output_dir / 'temporal_model.pt'))
    occ_payload = read_checkpoint(occ_ckpt, map_location=device.type)
    tmp_payload = read_checkpoint(tmp_ckpt, map_location=device.type)
    occ_schema = occ_payload.get('metadata', {}).get('feature_schema_version')
    tmp_schema = tmp_payload.get('metadata', {}).get('feature_schema_version')
    if occ_schema != FEATURE_SCHEMA_VERSION or tmp_schema != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            f'Los checkpoints no son compatibles con el esquema de features actual ({FEATURE_SCHEMA_VERSION}). '
            'Reentrena occurrence y temporal con esta versión del proyecto antes de predecir.'
        )
    ckpt_cfg = occ_payload.get('metadata', {}).get('config') or cfg
    temporal_model_type = str(tmp_payload.get('metadata', {}).get('temporal_model_type', ckpt_cfg['models']['temporal'].get('architecture', 'ranking')))
    sample_series = next(iter(context.series.values()))
    history_dim = int(build_future_history_tensor(sample_series, int(ckpt_cfg['calendar']['window_weeks']), bin_minutes=int(ckpt_cfg['calendar']['bin_minutes'])).shape[-1])
    occ_numeric_dim = int(occ_payload.get('metadata', {}).get('occ_numeric_dim', 19))
    tmp_numeric_dim = int(tmp_payload.get('metadata', {}).get('tmp_numeric_dim', 18))

    occ_model = OccurrenceResidualNet(
        input_dim=history_dim,
        num_tasks=len(context.task_names),
        num_databases=len(context.database_to_idx),
        num_robots=len(context.robot_to_idx),
        hidden_size=int(ckpt_cfg['models']['occurrence']['hidden_size']),
        num_layers=int(ckpt_cfg['models']['occurrence']['num_layers']),
        dropout=float(ckpt_cfg['models']['occurrence']['dropout']),
        task_embed_dim=int(ckpt_cfg['models']['occurrence']['task_embed_dim']),
        database_embed_dim=int(ckpt_cfg['models']['occurrence']['database_embed_dim']),
        robot_embed_dim=int(ckpt_cfg['models']['occurrence']['robot_embed_dim']),
        max_delta=int(ckpt_cfg['models']['occurrence']['max_delta']),
        numeric_feature_dim=occ_numeric_dim,
    ).to(device)

    if temporal_model_type in _DIRECT_TYPES:
        bins_per_day = int((24 * 60) / int(ckpt_cfg['calendar']['bin_minutes']))
        tmp_model = TemporalDirectNet(
            input_dim=history_dim,
            num_tasks=len(context.task_names),
            num_databases=len(context.database_to_idx),
            num_robots=len(context.robot_to_idx),
            window_weeks=int(ckpt_cfg['calendar']['window_weeks']),
            bins_per_day=bins_per_day,
            max_slots=int(ckpt_cfg['calendar'].get('max_slot_prototypes', 32)),
            hidden_size=int(ckpt_cfg['models']['temporal']['hidden_size']),
            num_layers=int(ckpt_cfg['models']['temporal']['num_layers']),
            num_heads=int(ckpt_cfg['models']['temporal'].get('num_heads', 4)),
            dropout=float(ckpt_cfg['models']['temporal']['dropout']),
            task_embed_dim=int(ckpt_cfg['models']['temporal'].get('task_embed_dim', 32)),
            database_embed_dim=int(ckpt_cfg['models']['temporal'].get('database_embed_dim', 16)),
            robot_embed_dim=int(ckpt_cfg['models']['temporal'].get('robot_embed_dim', 16)),
            slot_embed_dim=int(ckpt_cfg['models']['temporal'].get('slot_embed_dim', 24)),
            day_embed_dim=int(ckpt_cfg['models']['temporal'].get('day_embed_dim', 8)),
            time_embed_dim=int(ckpt_cfg['models']['temporal'].get('time_embed_dim', 16)),
            numeric_feature_dim=tmp_numeric_dim,
        ).to(device)
    else:
        tmp_candidate_dim = int(tmp_payload.get('metadata', {}).get('tmp_candidate_dim', 18))
        tmp_model = TemporalRankingNet(
            input_dim=history_dim,
            num_tasks=len(context.task_names),
            num_databases=len(context.database_to_idx),
            num_robots=len(context.robot_to_idx),
            hidden_size=int(ckpt_cfg['models']['temporal']['hidden_size']),
            num_layers=int(ckpt_cfg['models']['temporal']['num_layers']),
            dropout=float(ckpt_cfg['models']['temporal']['dropout']),
            task_embed_dim=int(ckpt_cfg['models']['temporal']['task_embed_dim']),
            database_embed_dim=int(ckpt_cfg['models']['temporal']['database_embed_dim']),
            robot_embed_dim=int(ckpt_cfg['models']['temporal']['robot_embed_dim']),
            numeric_feature_dim=tmp_numeric_dim,
            candidate_feature_dim=tmp_candidate_dim,
        ).to(device)

    occ_model.load_state_dict(occ_payload['state_dict'])
    tmp_model.load_state_dict(tmp_payload['state_dict'])

    predictor = HybridWeekPredictor(context, ckpt_cfg, occ_model, tmp_model, device)
    predictor.save_prediction(args.database_id, output_dir)
    print(f'Predicción guardada en {output_dir}')


if __name__ == '__main__':
    main()
