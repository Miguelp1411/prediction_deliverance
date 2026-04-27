from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_schedule.config import load_config
from hybrid_schedule.data import load_all_events, load_registry
from hybrid_schedule.data.features import build_global_context
from hybrid_schedule.inference import UnifiedWeekPredictor
from hybrid_schedule.models import UnifiedSlotTransformer
from hybrid_schedule.utils import get_device, read_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--database-id', required=True)
    parser.add_argument('--robot-id', required=True)
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    device = get_device()

    registry = load_registry(args.registry)
    df = load_all_events(registry, timezone_default=cfg['calendar']['timezone_default'])
    context = build_global_context(df, bin_minutes=int(cfg['calendar']['bin_minutes']))

    ckpt_path = Path(args.checkpoint or (output_dir / 'unified_model.pt'))
    payload = read_checkpoint(ckpt_path, map_location=device.type)
    metadata = payload.get('metadata', {})
    ckpt_cfg = metadata.get('config', cfg)

    model = UnifiedSlotTransformer(
        input_dim=int(metadata['history_dim']),
        numeric_feature_dim=int(metadata['numeric_feature_dim']),
        num_tasks=int(metadata['num_tasks']),
        num_databases=int(metadata['num_databases']),
        num_robots=int(metadata['num_robots']),
        max_slots=int(metadata['max_slots']),
        window_weeks=int(ckpt_cfg['calendar']['window_weeks']),
        bins_per_day=int(24 * 60 / int(ckpt_cfg['calendar']['bin_minutes'])),
        **ckpt_cfg['models']['unified'],
    ).to(device)
    missing, unexpected = model.load_state_dict(payload['state_dict'], strict=False)
    if missing:
        print(f'[predict_week] Aviso: parámetros faltantes al cargar checkpoint: {sorted(missing)}')
    if unexpected:
        print(f'[predict_week] Aviso: parámetros inesperados en checkpoint: {sorted(unexpected)}')

    predictor = UnifiedWeekPredictor(
        context=context,
        config=ckpt_cfg,
        model=model,
        device=device,
        feature_scaling=metadata.get('feature_scaling', {}),
    )
    out_path = predictor.save_prediction(args.database_id, args.robot_id, output_dir)
    print(f'Predicción guardada en {out_path}')


if __name__ == '__main__':
    main()
