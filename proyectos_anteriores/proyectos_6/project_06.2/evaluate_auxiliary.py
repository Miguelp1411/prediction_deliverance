from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from auxiliary_corrector import AuxiliaryCorrector, maybe_load_auxiliary_corrector
from config import BIN_MINUTES, CAP_INFERENCE_SCOPE, CHECKPOINT_DIR, DATA_PATH, TIMEZONE, TRAIN_RATIO, WINDOW_WEEKS
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data
from evaluation.matching import hungarian_match
from evaluation.weekly_stats import evaluate_weekly_predictions
from predict import _extract_preprocessing_caps, _load_models, load_checkpoint, predict_next_week, resolve_device


def _avg(stats_list: list[dict[str, float]], key: str) -> float:
    return float(np.mean([float(s.get(key, 0.0)) for s in stats_list])) if stats_list else 0.0


def _normalize_pred(pred_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in pred_tasks:
        out.append({
            'task_name': str(item.get('task_name', item.get('type'))),
            'start_bin': int(item['start_bin']),
            'duration': float(item['duration']),
        })
    return out


def _repair_effect(raw_pred: list[dict[str, Any]], repaired_pred: list[dict[str, Any]]) -> dict[str, float]:
    raw_norm = sorted(_normalize_pred(raw_pred), key=lambda x: (x['start_bin'], x['task_name']))
    repaired_norm = sorted(_normalize_pred(repaired_pred), key=lambda x: (x['start_bin'], x['task_name']))
    if not raw_norm or not repaired_norm:
        return {
            'repair_moved_fraction': 0.0,
            'repair_avg_displacement_minutes': 0.0,
        }
    pairs = hungarian_match(raw_norm, repaired_norm)
    if not pairs:
        return {
            'repair_moved_fraction': 0.0,
            'repair_avg_displacement_minutes': 0.0,
        }
    moved = 0
    moved_distances: list[float] = []
    for raw_item, repaired_item in pairs:
        displacement = abs(int(raw_item['start_bin']) - int(repaired_item['start_bin'])) * BIN_MINUTES
        if displacement > 0:
            moved += 1
            moved_distances.append(float(displacement))
    return {
        'repair_moved_fraction': float(moved / max(len(pairs), 1)),
        'repair_avg_displacement_minutes': float(np.mean(moved_distances)) if moved_distances else 0.0,
    }


def _summarize_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    if not stats_list:
        return {}
    keys = [
        'task_accuracy',
        'day_exact_accuracy',
        'day_close_accuracy_1d',
        'time_exact_accuracy',
        'time_close_accuracy_5m',
        'time_close_accuracy_10m',
        'start_mae_minutes',
        'start_mae_when_day_correct_minutes',
        'duration_close_accuracy',
        'repair_moved_fraction',
        'repair_avg_displacement_minutes',
    ]
    return {key: _avg(stats_list, key) for key in keys}


def _delta_metrics(candidate: dict[str, float] | None, base: dict[str, float] | None) -> dict[str, float] | None:
    if not candidate or not base:
        return None
    return {
        'task_accuracy_pp': float((candidate.get('task_accuracy', 0.0) - base.get('task_accuracy', 0.0)) * 100.0),
        'day_exact_accuracy_pp': float((candidate.get('day_exact_accuracy', 0.0) - base.get('day_exact_accuracy', 0.0)) * 100.0),
        'time_exact_accuracy_pp': float((candidate.get('time_exact_accuracy', 0.0) - base.get('time_exact_accuracy', 0.0)) * 100.0),
        'time_close_accuracy_5m_pp': float((candidate.get('time_close_accuracy_5m', 0.0) - base.get('time_close_accuracy_5m', 0.0)) * 100.0),
        'start_mae_minutes_delta': float(candidate.get('start_mae_minutes', 0.0) - base.get('start_mae_minutes', 0.0)),
        'start_mae_when_day_correct_minutes_delta': float(candidate.get('start_mae_when_day_correct_minutes', 0.0) - base.get('start_mae_when_day_correct_minutes', 0.0)),
        'duration_close_accuracy_pp': float((candidate.get('duration_close_accuracy', 0.0) - base.get('duration_close_accuracy', 0.0)) * 100.0),
    }


def _run_predictions(
    prepared,
    occurrence_model,
    temporal_model,
    auxiliary_corrector,
    device,
    week_idx: int,
    *,
    use_repair: bool,
    aux_correct_count: bool,
    aux_correct_time: bool,
    aux_correct_duration: bool,
):
    return predict_next_week(
        occurrence_model,
        temporal_model,
        prepared,
        week_idx,
        device,
        use_repair=use_repair,
        auxiliary_corrector=auxiliary_corrector,
        aux_correct_count=aux_correct_count,
        aux_correct_time=aux_correct_time,
        aux_correct_duration=aux_correct_duration,
    )


def evaluate_dataset(data_path: str, device_name: str | None, retrain_auxiliary: bool) -> dict:
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
    scenario_stats: dict[str, list[dict[str, float]]] = {
        'base': [],
        'aux_count_duration': [],
        'aux_full': [],
    }

    for week_idx in week_indices:
        truth = prepared.weeks[week_idx]
        base_pred = _run_predictions(
            prepared, occurrence_model, temporal_model, None, device, week_idx,
            use_repair=False, aux_correct_count=True, aux_correct_time=False, aux_correct_duration=True,
        )
        base_stats = evaluate_weekly_predictions(truth, base_pred)
        base_stats.update({'repair_moved_fraction': 0.0, 'repair_avg_displacement_minutes': 0.0})
        scenario_stats['base'].append(base_stats)

        if auxiliary_corrector is None:
            continue

        aux_cd_pred = _run_predictions(
            prepared, occurrence_model, temporal_model, auxiliary_corrector, device, week_idx,
            use_repair=False, aux_correct_count=True, aux_correct_time=False, aux_correct_duration=True,
        )
        aux_full_pred = _run_predictions(
            prepared, occurrence_model, temporal_model, auxiliary_corrector, device, week_idx,
            use_repair=False, aux_correct_count=True, aux_correct_time=True, aux_correct_duration=True,
        )

        aux_cd_stats = evaluate_weekly_predictions(truth, aux_cd_pred)
        aux_cd_stats.update({'repair_moved_fraction': 0.0, 'repair_avg_displacement_minutes': 0.0})
        aux_full_stats = evaluate_weekly_predictions(truth, aux_full_pred)
        aux_full_stats.update({'repair_moved_fraction': 0.0, 'repair_avg_displacement_minutes': 0.0})

        scenario_stats['aux_count_duration'].append(aux_cd_stats)
        scenario_stats['aux_full'].append(aux_full_stats)

    scenarios = {name: _summarize_stats(values) for name, values in scenario_stats.items() if values}
    report = {
        'data_path': str(data_path),
        'weeks_evaluated': len(week_indices),
        'auxiliary_status': aux_status,
        'scenarios': scenarios,
        'auxiliary_improvement': {},
    }

    if 'aux_count_duration' in scenarios:
        report['auxiliary_improvement']['count_duration_vs_base'] = _delta_metrics(
            scenarios.get('aux_count_duration'),
            scenarios.get('base'),
        )
    if 'aux_full' in scenarios:
        report['auxiliary_improvement']['full_vs_base'] = _delta_metrics(
            scenarios.get('aux_full'),
            scenarios.get('base'),
        )
    return report



def _format_pct(value: float | None) -> str:
    return 'n/a' if value is None else f'{value*100:.1f}%'


def _format_pp(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:+.1f} pp'


def _format_min(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1f}'


def _scenario_table(report: dict) -> None:
    scenarios = report['scenarios']
    columns = [
        ('base', 'base/sin repair'),
        ('aux_count_duration', 'aux c+d/sin repair'),
        ('aux_full', 'aux full/sin repair'),
    ]
    metrics = [
        ('task_accuracy', 'task_acc', 'pct'),
        ('day_exact_accuracy', 'día exacto', 'pct'),
        ('day_close_accuracy_1d', 'día ±1', 'pct'),
        ('time_exact_accuracy', 'hora exacta', 'pct'),
        ('time_close_accuracy_5m', 'hora ±5m', 'pct'),
        ('start_mae_minutes', 'MAE inicio', 'min'),
        ('start_mae_when_day_correct_minutes', 'MAE inicio | día ok', 'min'),
        ('duration_close_accuracy', 'dur ±2m', 'pct'),
        ('repair_moved_fraction', '% movidas repair', 'pct'),
        ('repair_avg_displacement_minutes', 'despl. medio repair', 'min'),
    ]

    header = ['métrica'.ljust(22)] + [title.rjust(20) for _, title in columns]
    print('\nTabla comparativa por dataset')
    print('-' * (22 + 21 * len(columns)))
    print(''.join(header))
    print('-' * (22 + 21 * len(columns)))
    for metric_key, label, kind in metrics:
        row = [label.ljust(22)]
        for scenario_key, _ in columns:
            value = scenarios.get(scenario_key, {}).get(metric_key)
            if value is None:
                cell = 'n/a'
            elif kind == 'pct':
                cell = f'{value*100:.1f}%'
            else:
                cell = f'{value:.1f}'
            row.append(cell.rjust(20))
        print(''.join(row))


def print_report(report: dict) -> None:
    print(f"\nDataset: {report['data_path']}")
    print('-' * 88)
    print(f"Semanas evaluadas : {report['weeks_evaluated']}")
    print(f"Corrector auxiliar: {report['auxiliary_status']}")
    _scenario_table(report)

    improvements = report.get('auxiliary_improvement', {}) or {}
    if improvements:
        print('\nMejora del auxiliar frente a base, sin repair')
        print('-' * 88)
        for label_key, title in [
            ('count_duration_vs_base', 'Aux count+dur vs base (sin repair)'),
            ('full_vs_base', 'Aux full vs base (sin repair)'),
        ]:
            metrics = improvements.get(label_key)
            if not metrics:
                continue
            print(title)
            print(f"  task_acc        : {_format_pp(metrics.get('task_accuracy_pp'))}")
            print(f"  día exacto      : {_format_pp(metrics.get('day_exact_accuracy_pp'))}")
            print(f"  hora exacta     : {_format_pp(metrics.get('time_exact_accuracy_pp'))}")
            print(f"  hora ±5m        : {_format_pp(metrics.get('time_close_accuracy_5m_pp'))}")
            print(f"  MAE inicio      : {_format_min(metrics.get('start_mae_minutes_delta'))} min")
            print(f"  MAE día correcto: {_format_min(metrics.get('start_mae_when_day_correct_minutes_delta'))} min")
            print(f"  dur ±2m         : {_format_pp(metrics.get('duration_close_accuracy_pp'))}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark del modelo base frente al corrector auxiliar sin aplicar repair en ningún momento.')
    parser.add_argument('data', nargs='*', default=[str(DATA_PATH)], help='Uno o varios JSON históricos a evaluar.')
    parser.add_argument('--device', type=str, default=None, help="Dispositivo: 'cpu' o 'cuda'.")
    parser.add_argument('--retrain-auxiliary', action='store_true', help='Reentrena el corrector auxiliar con cada dataset antes de evaluar.')
    parser.add_argument('--output', type=str, default=None, help='Ruta opcional para guardar un informe JSON agregado.')
    args = parser.parse_args()

    reports = []
    for data_path in args.data:
        report = evaluate_dataset(
            data_path=data_path,
            device_name=args.device,
            retrain_auxiliary=args.retrain_auxiliary,
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
