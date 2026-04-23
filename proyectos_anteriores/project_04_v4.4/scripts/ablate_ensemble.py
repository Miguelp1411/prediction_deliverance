from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CAP_INFERENCE_SCOPE, CHECKPOINT_DIR, DATA_PATH, DEVICE, REPORTS_DIR, TIMEZONE, TRAIN_RATIO
from data.datasets import build_split_indices
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data
from predict import _extract_preprocessing_caps, _load_models
from train import aggregate_weekly_ablation_stats
from utils.runtime import resolve_device
from utils.serialization import load_checkpoint


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def main():
    parser = argparse.ArgumentParser(description='Ablación del ensamblado para checkpoints congelados del proyecto v4.3.')
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='Ruta al JSON de datos históricos.')
    parser.add_argument('--device', type=str, default=None, help="Dispositivo: 'auto', 'cpu', 'cuda', etc.")
    parser.add_argument('--include-repair', action='store_true', help='Incluye también la etapa repair solo para diagnóstico.')
    parser.add_argument('--output', type=str, default=str(REPORTS_DIR / 'ensemble_ablation_report.json'), help='Ruta del JSON de salida.')
    args = parser.parse_args()

    device = resolve_device(args.device or DEVICE)
    df = load_tasks_dataframe(args.data, timezone=TIMEZONE)
    occ_ckpt = load_checkpoint(CHECKPOINT_DIR / 'occurrence_model.pt', map_location=device)
    tmp_ckpt = load_checkpoint(CHECKPOINT_DIR / 'temporal_model.pt', map_location=device)
    max_occurrences_per_task, max_tasks_per_week = _extract_preprocessing_caps(occ_ckpt, tmp_ckpt, df=df)
    prepared = prepare_data(
        df,
        train_ratio=TRAIN_RATIO,
        max_occurrences_per_task=max_occurrences_per_task,
        max_tasks_per_week=max_tasks_per_week,
        cap_inference_scope=CAP_INFERENCE_SCOPE,
    )
    split = build_split_indices(prepared, train_ratio=TRAIN_RATIO)
    occurrence_model, temporal_model = _load_models(prepared, device, occ_ckpt=occ_ckpt, tmp_ckpt=tmp_ckpt)

    report = aggregate_weekly_ablation_stats(
        prepared,
        split.val_target_week_indices,
        occurrence_model,
        temporal_model,
        device,
        include_repair=bool(args.include_repair),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2), encoding='utf-8')

    print('Ablación del ensamblado completada')
    print(f'  salida: {output_path}')
    for stage in report.get('stage_order', []):
        metrics = report.get('stages', {}).get(stage, {})
        print(
            f"  {stage:<8} exact={100.0 * float(metrics.get('time_exact_accuracy', 0.0)):.2f}% | "
            f"±5m={100.0 * float(metrics.get('time_close_accuracy_5m', 0.0)):.2f}% | "
            f"mae={float(metrics.get('start_mae_minutes', 0.0)):.2f} min"
        )
    for transition_name, metrics in report.get('transitions', {}).items():
        print(
            f"  {transition_name:<16} movidos={float(metrics.get('events_moved', 0.0)):.0f}/"
            f"{float(metrics.get('compared_predictions', 0.0)):.0f} | "
            f"despl medio={float(metrics.get('movement_mean_moved_minutes', 0.0)):.2f} min"
        )


if __name__ == '__main__':
    main()
