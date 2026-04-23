from __future__ import annotations

import argparse
import json
import math
from bisect import insort
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from proyectos_anteriores.project_04.config import (
    BIN_MINUTES,
    CHECKPOINT_DIR,
    DATA_PATH,
    DEVICE,
    PREDICTION_DAY_TOPK,
    PREDICTION_REPAIR_RADIUS_BINS,
    PREDICTION_TIME_TOPK,
    PREDICTION_USE_DURATION_MEDIAN_BLEND,
    TIMEZONE,
    TRAIN_RATIO,
    WINDOW_WEEKS,
    bins_per_day,
    num_time_bins,
)
from proyectos_anteriores.project_04.data.io import load_tasks_dataframe
from proyectos_anteriores.project_04.data.preprocessing import build_temporal_context, denormalize_duration, prepare_data, week_to_feature_vector
from proyectos_anteriores.project_04.models.occurrence_model import TaskOccurrenceModel
from proyectos_anteriores.project_04.models.temporal_model import TemporalAssignmentModel
from proyectos_anteriores.project_04.utils.serialization import load_checkpoint


def resolve_device(device_name: str | None = None) -> torch.device:
    chosen = device_name or DEVICE
    if chosen == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _build_sequence(prepared, target_week_idx):
    context_weeks = prepared.weeks[max(0, target_week_idx - WINDOW_WEEKS):target_week_idx]
    if not context_weeks:
        return np.zeros((WINDOW_WEEKS, prepared.week_feature_dim), dtype=np.float32)
    seq = np.stack([week_to_feature_vector(week) for week in context_weeks]).astype(np.float32)
    if seq.shape[0] < WINDOW_WEEKS:
        pad = np.zeros((WINDOW_WEEKS - seq.shape[0], prepared.week_feature_dim), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)
    return seq


def _task_duration_median(prepared, task_name: str) -> float:
    mask = prepared.df['task_name'] == task_name
    if not mask.any():
        return float(prepared.duration_min)
    return float(prepared.df.loc[mask, 'duration_minutes'].median())


def _duration_to_bins(duration_minutes: float) -> int:
    return max(1, int(round(float(duration_minutes) / float(BIN_MINUTES))))


def _quantize_duration_minutes(duration_minutes: float) -> float:
    return float(_duration_to_bins(duration_minutes) * BIN_MINUTES)


def _build_candidates(day_logits_row: torch.Tensor, time_logits_row: torch.Tensor) -> list[tuple[int, float]]:
    day_probs = torch.softmax(day_logits_row, dim=-1)
    time_probs = torch.softmax(time_logits_row, dim=-1)
    day_k = min(PREDICTION_DAY_TOPK, day_probs.numel())
    time_k = min(PREDICTION_TIME_TOPK, time_probs.numel())
    day_values, day_indices = torch.topk(day_probs, k=day_k)
    time_values, time_indices = torch.topk(time_probs, k=time_k)

    candidates: list[tuple[int, float]] = []
    seen: set[int] = set()
    for d_prob, d_idx in zip(day_values.tolist(), day_indices.tolist()):
        for t_prob, t_idx in zip(time_values.tolist(), time_indices.tolist()):
            start_bin = int(d_idx) * bins_per_day() + int(t_idx)
            if start_bin in seen:
                continue
            seen.add(start_bin)
            score = math.log(max(float(d_prob), 1e-9)) + math.log(max(float(t_prob), 1e-9))
            candidates.append((start_bin, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def _overlaps(start_bin: int, duration_bins: int, placed_intervals: list[tuple[int, int]]) -> bool:
    end_bin = start_bin + duration_bins
    for other_start, other_end in placed_intervals:
        if start_bin < other_end and end_bin > other_start:
            return True
    return False


def _nearest_valid_start(
    preferred_start: int,
    duration_bins: int,
    placed_intervals: list[tuple[int, int]],
    lower_bound: int = 0,
) -> int:
    preferred_start = int(np.clip(preferred_start, lower_bound, num_time_bins() - duration_bins))
    if not _overlaps(preferred_start, duration_bins, placed_intervals):
        return preferred_start

    max_radius = min(PREDICTION_REPAIR_RADIUS_BINS, num_time_bins())
    for radius in range(1, max_radius + 1):
        for candidate in (preferred_start - radius, preferred_start + radius):
            if candidate < lower_bound or candidate > num_time_bins() - duration_bins:
                continue
            if not _overlaps(candidate, duration_bins, placed_intervals):
                return candidate

    fallback = lower_bound
    while fallback <= num_time_bins() - duration_bins:
        if not _overlaps(fallback, duration_bins, placed_intervals):
            return fallback
        fallback += 1
    return int(np.clip(preferred_start, 0, num_time_bins() - 1))


def _repair_predictions(raw_predictions: list[dict]) -> list[dict]:
    placed: list[tuple[int, int]] = []
    last_start_by_task: dict[int, int] = {}
    final_predictions: list[dict] = []

    for item in sorted(raw_predictions, key=lambda x: (x['preferred_start_bin'], x['task_id'], x['occurrence_index'])):
        duration_bins = _duration_to_bins(item['duration'])
        quantized_duration = float(duration_bins * BIN_MINUTES)
        lower_bound = last_start_by_task.get(item['task_id'], 0)
        chosen_start = None

        for candidate_start, _ in item['candidate_starts']:
            candidate_start = int(np.clip(candidate_start, lower_bound, num_time_bins() - duration_bins))
            if not _overlaps(candidate_start, duration_bins, placed):
                chosen_start = candidate_start
                break

        if chosen_start is None:
            chosen_start = _nearest_valid_start(
                item['preferred_start_bin'],
                duration_bins,
                placed,
                lower_bound=lower_bound,
            )

        last_start_by_task[item['task_id']] = chosen_start
        interval = (chosen_start, chosen_start + duration_bins)
        insort(placed, interval)
        final_predictions.append({
            'task_name': item['task_name'],
            'type': item['task_name'],
            'start_bin': int(chosen_start),
            'duration': quantized_duration,
        })

    return sorted(final_predictions, key=lambda x: (x['start_bin'], x['task_name']))


def predict_next_week(occurrence_model, temporal_model, prepared, target_week_idx, device, use_repair: bool = True):
    occurrence_model.eval()
    temporal_model.eval()

    with torch.no_grad():
        seq = _build_sequence(prepared, target_week_idx)
        base_sequence = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
        count_logits = occurrence_model(base_sequence)
        pred_counts = torch.argmax(count_logits, dim=-1).cpu().numpy()[0]

        task_ids, occurrence_indices, predicted_count_norms = [], [], []
        history_features, anchor_days = [], []

        for task_id, count in enumerate(pred_counts):
            count = int(min(count, prepared.max_count_cap))
            if count == 0:
                continue
            for occ_idx in range(count):
                context = build_temporal_context(
                    prepared.weeks,
                    target_week_idx,
                    task_id,
                    occ_idx,
                    prepared.duration_min,
                    prepared.duration_max,
                )
                task_ids.append(task_id)
                occurrence_indices.append(occ_idx)
                predicted_count_norms.append(float(count / prepared.max_count_cap))
                history_features.append(context.history_features)
                anchor_days.append(context.anchor_day)

        if not task_ids:
            return []

        batch_size = len(task_ids)
        sequence_context = temporal_model.encode_sequence(base_sequence).repeat(batch_size, 1)
        outputs = temporal_model.forward_with_context(
            sequence_context=sequence_context,
            task_id=torch.tensor(task_ids, dtype=torch.long, device=device),
            occurrence_index=torch.tensor(occurrence_indices, dtype=torch.long, device=device),
            history_features=torch.tensor(np.stack(history_features), dtype=torch.float32, device=device),
            predicted_count_norm=torch.tensor(predicted_count_norms, dtype=torch.float32, device=device),
            anchor_day=torch.tensor(anchor_days, dtype=torch.long, device=device),
        )

        raw_predictions = []
        for i in range(batch_size):
            task_name = prepared.task_names[task_ids[i]]
            pred_duration = denormalize_duration(
                float(outputs['pred_duration_norm'][i].item()),
                prepared.duration_min,
                prepared.duration_max,
            )
            median_duration = _task_duration_median(prepared, task_name)
            duration = (
                (1.0 - PREDICTION_USE_DURATION_MEDIAN_BLEND) * pred_duration
                + PREDICTION_USE_DURATION_MEDIAN_BLEND * median_duration
            )
            candidates = _build_candidates(outputs['day_logits'][i], outputs['time_logits'][i])
            preferred_start_bin = candidates[0][0] if candidates else 0
            raw_predictions.append({
                'task_id': task_ids[i],
                'task_name': task_name,
                'occurrence_index': occurrence_indices[i],
                'preferred_start_bin': int(preferred_start_bin),
                'candidate_starts': candidates,
                'duration': float(duration),
            })

    if use_repair:
        return _repair_predictions(raw_predictions)

    raw_no_repair = []
    for item in raw_predictions:
            raw_no_repair.append({
                'task_name': item['task_name'],
                'type': item['task_name'],
                'start_bin': int(item['preferred_start_bin']),
                'duration': _quantize_duration_minutes(item['duration']),
            })

    return sorted(raw_no_repair, key=lambda x: (x['start_bin'], x['task_name']))


def _load_models(prepared, device: torch.device, checkpoint_dir: Path):
    occ_ckpt = load_checkpoint(checkpoint_dir / 'occurrence_model.pt', map_location=device)
    tmp_ckpt = load_checkpoint(checkpoint_dir / 'temporal_model.pt', map_location=device)

    occ_h = occ_ckpt.get('model_hparams', {})
    tmp_h = tmp_ckpt.get('model_hparams', {})

    occurrence_model = TaskOccurrenceModel(
        input_dim=int(occ_h.get('input_dim', prepared.week_feature_dim)),
        num_tasks=int(occ_h.get('num_tasks', len(prepared.task_names))),
        max_count_cap=int(occ_h.get('max_count_cap', prepared.max_count_cap)),
        hidden_size=int(occ_h.get('hidden_size', 128)),
        num_layers=int(occ_h.get('num_layers', 2)),
        dropout=float(occ_h.get('dropout', 0.15)),
    ).to(device)
    occurrence_model.load_state_dict(occ_ckpt['state_dict'])

    temporal_model = TemporalAssignmentModel(
        sequence_dim=int(tmp_h.get('sequence_dim', prepared.week_feature_dim)),
        history_feature_dim=int(tmp_h.get('history_feature_dim', prepared.history_feature_dim)),
        num_tasks=int(tmp_h.get('num_tasks', len(prepared.task_names))),
        max_occurrences=int(tmp_h.get('max_occurrences', prepared.max_count_cap)),
        hidden_size=int(tmp_h.get('hidden_size', 160)),
        num_layers=int(tmp_h.get('num_layers', 2)),
        dropout=float(tmp_h.get('dropout', 0.15)),
        task_embed_dim=int(tmp_h.get('task_embed_dim', 32)),
        occurrence_embed_dim=int(tmp_h.get('occurrence_embed_dim', 16)),
        day_embed_dim=int(tmp_h.get('day_embed_dim', 8)),
    ).to(device)
    temporal_model.load_state_dict(tmp_ckpt['state_dict'])

    return occurrence_model, temporal_model


def _prediction_week_start(prepared) -> pd.Timestamp:
    if not prepared.weeks:
        raise ValueError('No hay semanas preparadas para generar predicción')
    return prepared.weeks[-1].week_start + pd.Timedelta(days=7)


def _to_timestamp(week_start: pd.Timestamp, start_bin: int) -> pd.Timestamp:
    return week_start + pd.Timedelta(minutes=int(start_bin) * BIN_MINUTES)


def _format_utc_timestamp(ts: pd.Timestamp) -> str:
    """
    Devuelve formato:
    2028-12-31T18:00:00.000Z
    """
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts.strftime('%Y-%m-%dT%H:%M:%S.000Z')


def _materialize_prediction_times(prepared, predictions: list[dict]) -> list[dict]:
    pred_week_start = _prediction_week_start(prepared)
    materialized = []

    for item in predictions:
        start_ts = _to_timestamp(pred_week_start, int(item['start_bin']))
        quantized_duration = _quantize_duration_minutes(float(item['duration']))
        end_ts = start_ts + pd.Timedelta(minutes=quantized_duration)

        task_type = item.get('type', item['task_name'])

        materialized.append({
            'uid': None,
            'device_uid': None,
            'task_name': None,
            'type': task_type,
            'status': None,
            'start_time': _format_utc_timestamp(start_ts),
            'end_time': _format_utc_timestamp(end_ts),
            'mileage': 0,
            'misc': None,
            'waypoints': [],
        })

    return sorted(materialized, key=lambda x: x['start_time'])


def _normalize_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')


def _build_sanity_warnings(prepared, predictions: list[dict], materialized: list[dict]) -> list[str]:
    warnings: list[str] = []
    if not materialized:
        return warnings

    expected_week_start = _normalize_utc(_prediction_week_start(prepared))
    start_series = pd.to_datetime([item['start_time'] for item in materialized], utc=True)
    observed_week_start = _normalize_utc(start_series.min().normalize() - pd.Timedelta(days=int(start_series.min().dayofweek)))
    if observed_week_start != expected_week_start:
        warnings.append(
            f"Semana predicha fuera de rango esperado: esperada {expected_week_start.date()}, observada {observed_week_start.date()}."
        )

    weekly_totals = np.array([int(week.counts.sum()) for week in prepared.weeks], dtype=np.int32)
    hist_total_mean = float(weekly_totals.mean()) if len(weekly_totals) > 0 else 0.0
    pred_total = len(predictions)
    total_lower = max(1, int(np.floor(hist_total_mean * 0.65)))
    total_upper = max(total_lower, int(np.ceil(hist_total_mean * 1.35)))
    if pred_total < total_lower or pred_total > total_upper:
        warnings.append(
            f"Total de tareas inusual: pred={pred_total}, esperado aprox entre {total_lower} y {total_upper} (media hist={hist_total_mean:.1f})."
        )

    pred_by_task: dict[str, int] = {}
    for item in predictions:
        task_name = str(item.get('task_name', item.get('type')))
        pred_by_task[task_name] = pred_by_task.get(task_name, 0) + 1

    for task_id, task_name in enumerate(prepared.task_names):
        hist_counts = np.array([int(week.counts[task_id]) for week in prepared.weeks], dtype=np.int32)
        hist_mean = float(hist_counts.mean()) if len(hist_counts) > 0 else 0.0
        lower = int(np.floor(hist_mean * 0.50))
        upper = int(np.ceil(hist_mean * 1.50))
        pred_count = int(pred_by_task.get(task_name, 0))
        if pred_count < lower or pred_count > upper:
            warnings.append(
                f"Conteo por tipo fuera de rango: {task_name} pred={pred_count}, esperado aprox {lower}-{upper} (media hist={hist_mean:.1f})."
            )

    pred_day_counts = start_series.dayofweek.value_counts().sort_index()
    hist_day_week = prepared.df.groupby(['week_start', 'day_of_week']).size().unstack(fill_value=0)
    hist_day_means = hist_day_week.mean(axis=0)
    for day in range(7):
        pred_day = int(pred_day_counts.get(day, 0))
        hist_mean_day = float(hist_day_means.get(day, 0.0))
        lower = int(np.floor(hist_mean_day * 0.50))
        upper = int(np.ceil(hist_mean_day * 1.60))
        if pred_day < lower or pred_day > upper:
            warnings.append(
                f"Carga diaria inusual (día {day}): pred={pred_day}, esperado aprox {lower}-{upper} (media hist={hist_mean_day:.1f})."
            )

    return warnings


def main():
    parser = argparse.ArgumentParser(
        description='Genera la predicción de la siguiente semana usando los checkpoints guardados.'
    )
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='Ruta al JSON de datos históricos.')
    parser.add_argument('--output', type=str, default=None, help='Ruta del JSON de salida. Por defecto se guarda junto al dataset.')
    parser.add_argument('--checkpoint-dir', type=str, default=str(CHECKPOINT_DIR), help='Carpeta con occurrence_model.pt y temporal_model.pt.')
    parser.add_argument('--device', type=str, default=None, help="Dispositivo: 'cpu' o 'cuda'.")
    parser.add_argument('--strict-sanity', action='store_true', help='Si hay advertencias de sanidad, termina con error.')
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_path = Path(args.output).resolve() if args.output else data_path.parent / 'predicted_next_week.json'

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f'No existe el directorio de checkpoints: {checkpoint_dir}')
    if not (checkpoint_dir / 'occurrence_model.pt').exists():
        raise FileNotFoundError(f'Falta occurrence_model.pt en {checkpoint_dir}')
    if not (checkpoint_dir / 'temporal_model.pt').exists():
        raise FileNotFoundError(f'Falta temporal_model.pt en {checkpoint_dir}')

    device = resolve_device(args.device)
    df = load_tasks_dataframe(data_path, timezone=TIMEZONE)
    prepared = prepare_data(df, train_ratio=TRAIN_RATIO)
    occurrence_model, temporal_model = _load_models(prepared, device, checkpoint_dir)

    predictions = predict_next_week(
        occurrence_model,
        temporal_model,
        prepared,
        len(prepared.weeks),
        device,
        use_repair=False,
    )
    materialized = _materialize_prediction_times(prepared, predictions)
    warnings = _build_sanity_warnings(prepared, predictions, materialized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(materialized, ensure_ascii=False, indent=2), encoding='utf-8')

    metadata_path = output_path.with_suffix(output_path.suffix + '.meta.json')
    metadata = {
        'generated_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'data_path': str(data_path),
        'checkpoint_dir': str(checkpoint_dir),
        'output_path': str(output_path),
        'prediction_week_start_utc': _format_utc_timestamp(_prediction_week_start(prepared)),
        'repair_enabled': False,
        'bin_minutes': BIN_MINUTES,
        'total_tasks': len(materialized),
        'sanity_warnings': warnings,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Semana generada: {output_path}')
    print(f'Metadata: {metadata_path}')
    print(f'Dataset: {data_path}')
    print(f'Checkpoints: {checkpoint_dir}')
    print(f'Tareas predichas: {len(materialized)}')
    if warnings:
        print('Advertencias de sanidad:')
        for msg in warnings:
            print(f'  - {msg}')
        if args.strict_sanity:
            raise ValueError('Sanity check estricto activado y se detectaron advertencias.')


if __name__ == '__main__':
    main()
