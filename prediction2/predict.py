from __future__ import annotations

import argparse
import json
import math
from bisect import insort
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    BIN_MINUTES,
    CAP_INFERENCE_SCOPE,
    CHECKPOINT_DIR,
    DATA_PATH,
    DEFAULT_MAX_OCCURRENCES_PER_TASK,
    DEFAULT_MAX_TASKS_PER_WEEK,
    DEVICE,
    FEATURE_SCHEMA_VERSION,
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
from data.io import load_tasks_dataframe
from data.preprocessing import (
    build_temporal_context,
    denormalize_duration,
    infer_preprocessing_caps,
    prepare_data,
    week_to_feature_vector,
)
from models.occurrence_model import TaskOccurrenceModel
from models.temporal_model import TemporalAssignmentModel
from utils.serialization import load_checkpoint


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


def _ensure_checkpoint_compatibility(ckpt: dict, prepared, checkpoint_name: str) -> None:
    metadata = ckpt.get('metadata', {}) or {}
    model_hparams = ckpt.get('model_hparams', {}) or {}
    issues: list[str] = []

    schema_version = int(metadata.get('feature_schema_version', 1))
    if schema_version != FEATURE_SCHEMA_VERSION:
        issues.append(
            f"feature_schema_version={schema_version} y el código actual espera {FEATURE_SCHEMA_VERSION}"
        )

    checkpoint_week_dim = int(metadata.get('week_feature_dim', -1))
    if checkpoint_week_dim != prepared.week_feature_dim:
        issues.append(
            f"week_feature_dim={checkpoint_week_dim} pero ahora el proyecto genera {prepared.week_feature_dim}"
        )

    checkpoint_history_dim = int(metadata.get('history_feature_dim', -1))
    if checkpoint_history_dim != prepared.history_feature_dim:
        issues.append(
            f"history_feature_dim={checkpoint_history_dim} pero ahora el proyecto genera {prepared.history_feature_dim}"
        )

    checkpoint_max_occurrences = metadata.get(
        'max_occurrences_per_task',
        model_hparams.get('max_occurrences_per_task', model_hparams.get('max_occurrences', metadata.get('max_count_cap', model_hparams.get('max_count_cap')))),
    )
    if checkpoint_max_occurrences is not None and _positive_int(checkpoint_max_occurrences, prepared.max_occurrences_per_task) != prepared.max_occurrences_per_task:
        issues.append(
            f"max_occurrences_per_task={checkpoint_max_occurrences} pero los datos preparados usan {prepared.max_occurrences_per_task}"
        )

    checkpoint_max_tasks = metadata.get('max_tasks_per_week', model_hparams.get('max_tasks_per_week'))
    if checkpoint_max_tasks is not None and _positive_int(checkpoint_max_tasks, prepared.max_tasks_per_week) != prepared.max_tasks_per_week:
        issues.append(
            f"max_tasks_per_week={checkpoint_max_tasks} pero los datos preparados usan {prepared.max_tasks_per_week}"
        )

    if issues:
        joined = '; '.join(issues)
        raise ValueError(
            f"Checkpoint incompatible ({checkpoint_name}): {joined}. Reentrena los modelos con el código actual antes de predecir."
        )


def _task_duration_median(prepared, task_name: str) -> float:
    mask = prepared.df['task_name'] == task_name
    if not mask.any():
        return float(prepared.duration_min)
    return float(prepared.df.loc[mask, 'duration_minutes'].median())


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
        duration_bins = max(1, int(round(item['duration'] / BIN_MINUTES)))
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
            'duration': float(item['duration']),
        })

    return sorted(final_predictions, key=lambda x: (x['start_bin'], x['task_name']))


def predict_next_week(occurrence_model, temporal_model, prepared, target_week_idx, device, use_repair: bool = False):
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
                    max_occurrences_per_task=prepared.max_occurrences_per_task,
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
            'duration': float(item['duration']),
        })

    return sorted(raw_no_repair, key=lambda x: (x['start_bin'], x['task_name']))


def _positive_int(value, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def _extract_preprocessing_caps(occ_ckpt: dict, tmp_ckpt: dict, df: pd.DataFrame | None = None) -> tuple[int, int]:
    occ_metadata = occ_ckpt.get('metadata', {}) or {}
    tmp_metadata = tmp_ckpt.get('metadata', {}) or {}
    occ_hparams = occ_ckpt.get('model_hparams', {}) or {}
    tmp_hparams = tmp_ckpt.get('model_hparams', {}) or {}

    data_occ_cap, data_tasks_cap = (None, None)
    if df is not None and not df.empty:
        df_for_caps = df.copy()
        df_for_caps['week_start'] = df_for_caps['start_time'].dt.normalize() - pd.to_timedelta(df_for_caps['start_time'].dt.dayofweek, unit='D')
        task_names = sorted(df_for_caps['task_name'].astype(str).unique().tolist())
        task_to_id = {name: idx for idx, name in enumerate(task_names)}
        df_for_caps['task_id'] = df_for_caps['task_name'].map(task_to_id)
        data_occ_cap, data_tasks_cap = infer_preprocessing_caps(df_for_caps)

    occ_cap = _positive_int(
        tmp_metadata.get(
            'max_occurrences_per_task',
            occ_metadata.get(
                'max_occurrences_per_task',
                tmp_hparams.get(
                    'max_occurrences_per_task',
                    occ_hparams.get(
                        'max_occurrences_per_task',
                        tmp_hparams.get(
                            'max_occurrences',
                            occ_hparams.get(
                                'max_count_cap',
                                tmp_metadata.get(
                                    'max_count_cap',
                                    occ_metadata.get('max_count_cap', data_occ_cap or DEFAULT_MAX_OCCURRENCES_PER_TASK),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        data_occ_cap or DEFAULT_MAX_OCCURRENCES_PER_TASK,
    )
    tasks_cap = _positive_int(
        tmp_metadata.get(
            'max_tasks_per_week',
            occ_metadata.get(
                'max_tasks_per_week',
                tmp_hparams.get(
                    'max_tasks_per_week',
                    occ_hparams.get('max_tasks_per_week', data_tasks_cap or DEFAULT_MAX_TASKS_PER_WEEK),
                ),
            ),
        ),
        data_tasks_cap or DEFAULT_MAX_TASKS_PER_WEEK,
    )
    return occ_cap, tasks_cap


def _load_models(prepared, device: torch.device, occ_ckpt: dict | None = None, tmp_ckpt: dict | None = None):
    occ_ckpt = occ_ckpt or load_checkpoint(CHECKPOINT_DIR / 'occurrence_model.pt', map_location=device)
    tmp_ckpt = tmp_ckpt or load_checkpoint(CHECKPOINT_DIR / 'temporal_model.pt', map_location=device)

    _ensure_checkpoint_compatibility(occ_ckpt, prepared, 'occurrence_model.pt')
    _ensure_checkpoint_compatibility(tmp_ckpt, prepared, 'temporal_model.pt')

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
        end_ts = start_ts + pd.Timedelta(minutes=float(item['duration']))

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


def main():
    parser = argparse.ArgumentParser(
        description='Genera la predicción de la siguiente semana usando los checkpoints guardados.'
    )
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='Ruta al JSON de datos históricos.')
    parser.add_argument('--output', type=str, default='predicted_next_week.json', help='Ruta del JSON de salida.')
    parser.add_argument('--device', type=str, default=None, help="Dispositivo: 'cpu' o 'cuda'.")
    args = parser.parse_args()

    device = resolve_device(args.device)
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
    occurrence_model, temporal_model = _load_models(prepared, device, occ_ckpt=occ_ckpt, tmp_ckpt=tmp_ckpt)

    predictions = predict_next_week(
        occurrence_model,
        temporal_model,
        prepared,
        len(prepared.weeks),
        device,
        use_repair=False,
    )
    materialized = _materialize_prediction_times(prepared, predictions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(materialized, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Semana generada: {output_path}')
    print(f'Tareas predichas: {len(materialized)}')


if __name__ == '__main__':
    main()