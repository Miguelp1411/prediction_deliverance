from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry
from hybrid_schedule.data.features import build_global_context, build_future_history_tensor
from hybrid_schedule.inference import HybridWeekPredictor
from hybrid_schedule.models import OccurrenceResidualNet, TemporalResidualNet
from hybrid_schedule.utils import get_device, load_checkpoint, read_checkpoint



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
    ckpt_cfg = occ_payload.get('metadata', {}).get('config') or cfg
    sample_series = next(iter(context.series.values()))
    history_dim = int(build_future_history_tensor(sample_series, int(ckpt_cfg['calendar']['window_weeks'])).shape[-1])

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
    ).to(device)
    tmp_model = TemporalResidualNet(
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
        day_radius=int(ckpt_cfg['models']['temporal']['day_radius']),
        time_radius_bins=int(ckpt_cfg['models']['temporal']['time_radius_bins']),
    ).to(device)

    occ_model.load_state_dict(occ_payload['state_dict'])
    tmp_model.load_state_dict(tmp_payload['state_dict'])

    predictor = HybridWeekPredictor(context, ckpt_cfg, occ_model, tmp_model, device)
    predictor.save_prediction(args.database_id, output_dir)
    print(f'Predicción guardada en {output_dir}')


if __name__ == '__main__':
    main()
