from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from auxiliary_corrector import AuxiliaryCorrector, maybe_load_auxiliary_corrector
from config import CAP_INFERENCE_SCOPE, DATA_PATH, TIMEZONE, TRAIN_RATIO, WINDOW_WEEKS
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data
from evaluation.weekly_stats import evaluate_weekly_predictions
from predict import _extract_preprocessing_caps, _load_models, resolve_device, load_checkpoint, CHECKPOINT_DIR, predict_next_week


def _avg(stats_list: list[dict[str, float]], key: str) -> float:
    return float(np.mean([float(s.get(key, 0.0)) for s in stats_list])) if stats_list else 0.0


def evaluate_dataset(data_path: str, device_name: str | None, retrain_auxiliary: bool, use_repair: bool) -> dict:
    device = resolve_device(device_name)
    df = load_tasks_dataframe(data_path, timezone=TIMEZONE)

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
    occurrence_model, temporal_model = _load_models(prepared, device, occ_ckpt=occ_ckpt, tmp_ckpt=tmp_ckpt)

    auxiliary_corrector = None
    aux_status = 'disabled'
    if retrain_auxiliary:
        auxiliary_corrector = AuxiliaryCorrector.fit_from_history(
            prepared,
            occurrence_model,
            temporal_model,
            device,
            week_indices=list(range(WINDOW_WEEKS, len(prepared.weeks))),
        )
        aux_status = 'retrained_from_history'
    else:
        auxiliary_corrector = maybe_load_auxiliary_corrector()
        aux_status = 'loaded_checkpoint' if auxiliary_corrector is not None else 'missing_checkpoint'

    week_indices = list(range(WINDOW_WEEKS, len(prepared.weeks)))
    base_stats: list[dict[str, float]] = []
    aux_stats: list[dict[str, float]] = []

    for week_idx in week_indices:
        base_pred = predict_next_week(
            occurrence_model,
            temporal_model,
            prepared,
            week_idx,
            device,
            use_repair=use_repair,
            auxiliary_corrector=None,
        )
        base_stats.append(evaluate_weekly_predictions(prepared.weeks[week_idx], base_pred))

        if auxiliary_corrector is not None:
            aux_pred = predict_next_week(
                occurrence_model,
                temporal_model,
                prepared,
                week_idx,
                device,
                use_repair=use_repair,
                auxiliary_corrector=auxiliary_corrector,
            )
            aux_stats.append(evaluate_weekly_predictions(prepared.weeks[week_idx], aux_pred))

    report = {
        'data_path': str(data_path),
        'weeks_evaluated': len(week_indices),
        'auxiliary_status': aux_status,
        'base': {
            'task_accuracy': _avg(base_stats, 'task_accuracy'),
            'time_exact_accuracy': _avg(base_stats, 'time_exact_accuracy'),
            'time_close_accuracy_5m': _avg(base_stats, 'time_close_accuracy_5m'),
            'start_mae_minutes': _avg(base_stats, 'start_mae_minutes'),
            'duration_close_accuracy': _avg(base_stats, 'duration_close_accuracy'),
        },
        'auxiliary': None,
    }
    if aux_stats:
        report['auxiliary'] = {
            'task_accuracy': _avg(aux_stats, 'task_accuracy'),
            'time_exact_accuracy': _avg(aux_stats, 'time_exact_accuracy'),
            'time_close_accuracy_5m': _avg(aux_stats, 'time_close_accuracy_5m'),
            'start_mae_minutes': _avg(aux_stats, 'start_mae_minutes'),
            'duration_close_accuracy': _avg(aux_stats, 'duration_close_accuracy'),
        }
    return report


def print_report(report: dict) -> None:
    print(f"\nDataset: {report['data_path']}")
    print('-' * 72)
    print(f"Semanas evaluadas : {report['weeks_evaluated']}")
    print(f"Corrector auxiliar: {report['auxiliary_status']}")
    base = report['base']
    print('Base')
    print(f"  task_acc        : {base['task_accuracy']*100:.1f}%")
    print(f"  hora exacta     : {base['time_exact_accuracy']*100:.1f}%")
    print(f"  hora ±5m        : {base['time_close_accuracy_5m']*100:.1f}%")
    print(f"  MAE inicio      : {base['start_mae_minutes']:.2f} min")
    print(f"  dur ±2m         : {base['duration_close_accuracy']*100:.1f}%")
    aux = report.get('auxiliary')
    if aux is not None:
        print('Auxiliar')
        print(f"  task_acc        : {aux['task_accuracy']*100:.1f}%")
        print(f"  hora exacta     : {aux['time_exact_accuracy']*100:.1f}%")
        print(f"  hora ±5m        : {aux['time_close_accuracy_5m']*100:.1f}%")
        print(f"  MAE inicio      : {aux['start_mae_minutes']:.2f} min")
        print(f"  dur ±2m         : {aux['duration_close_accuracy']*100:.1f}%")
        print('Delta auxiliar - base')
        print(f"  task_acc        : {(aux['task_accuracy'] - base['task_accuracy'])*100:+.1f} pp")
        print(f"  hora exacta     : {(aux['time_exact_accuracy'] - base['time_exact_accuracy'])*100:+.1f} pp")
        print(f"  hora ±5m        : {(aux['time_close_accuracy_5m'] - base['time_close_accuracy_5m'])*100:+.1f} pp")
        print(f"  MAE inicio      : {aux['start_mae_minutes'] - base['start_mae_minutes']:+.2f} min")
        print(f"  dur ±2m         : {(aux['duration_close_accuracy'] - base['duration_close_accuracy'])*100:+.1f} pp")


def main() -> None:
    parser = argparse.ArgumentParser(description='Compara el modelo base contra el corrector auxiliar en una o varias bases históricas.')
    parser.add_argument('data', nargs='*', default=[str(DATA_PATH)], help='Uno o varios JSON históricos a evaluar.')
    parser.add_argument('--device', type=str, default=None, help="Dispositivo: 'cpu' o 'cuda'.")
    parser.add_argument('--retrain-auxiliary', action='store_true', help='Reentrena el corrector auxiliar con cada dataset antes de evaluar.')
    parser.add_argument('--disable-repair', action='store_true', help='Evalúa sin la fase de repair.')
    parser.add_argument('--output', type=str, default=None, help='Ruta opcional para guardar un informe JSON agregado.')
    args = parser.parse_args()

    reports = []
    for data_path in args.data:
        report = evaluate_dataset(
            data_path=data_path,
            device_name=args.device,
            retrain_auxiliary=args.retrain_auxiliary,
            use_repair=not args.disable_repair,
        )
        reports.append(report)
        print_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"\nInforme guardado en: {output_path}")


if __name__ == '__main__':
    main()
