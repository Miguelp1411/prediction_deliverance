from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatabaseRegularizationProfile:
    database_id: str
    events: int
    weeks: int
    robots: int
    sample_share: float
    weekly_cv: float
    task_count_cv_mean: float
    day_entropy_mean: float
    time_entropy_mean: float
    duration_modal_share_mean: float
    scarcity_score: float
    volatility_score: float
    rigidity_score: float
    history_dropout: float
    history_noise_std: float
    feature_dropout: float
    feature_noise_std: float
    occurrence_label_smoothing: float
    occurrence_delta_target_sigma: float
    occurrence_count_shrink_weight: float
    occurrence_confidence_penalty: float
    temporal_day_label_smoothing: float
    temporal_time_target_sigma: float
    temporal_anchor_weight: float
    temporal_day_prior_weight: float
    temporal_time_prior_weight: float
    temporal_confidence_penalty: float
    temporal_time_smoothness_weight: float


def _normalize_series(series: pd.Series) -> pd.Series:
    series = series.astype(float)
    span = float(series.max() - series.min())
    if span <= 1e-12:
        return pd.Series(np.zeros(len(series), dtype=np.float64), index=series.index)
    return (series - float(series.min())) / span


def _safe_entropy(values: pd.Series, classes: list[int] | None = None) -> float:
    counts = values.value_counts(normalize=True)
    if classes is not None:
        counts = counts.reindex(classes, fill_value=0.0)
    probs = counts.to_numpy(dtype=np.float64)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


_DEFAULT_PROFILE = DatabaseRegularizationProfile(
    database_id='__default__',
    events=0,
    weeks=0,
    robots=1,
    sample_share=0.0,
    weekly_cv=0.0,
    task_count_cv_mean=0.0,
    day_entropy_mean=0.0,
    time_entropy_mean=0.0,
    duration_modal_share_mean=1.0,
    scarcity_score=0.25,
    volatility_score=0.25,
    rigidity_score=0.50,
    history_dropout=0.04,
    history_noise_std=0.01,
    feature_dropout=0.04,
    feature_noise_std=0.01,
    occurrence_label_smoothing=0.03,
    occurrence_delta_target_sigma=1.00,
    occurrence_count_shrink_weight=0.04,
    occurrence_confidence_penalty=0.003,
    temporal_day_label_smoothing=0.03,
    temporal_time_target_sigma=1.50,
    temporal_anchor_weight=0.06,
    temporal_day_prior_weight=0.10,
    temporal_time_prior_weight=0.10,
    temporal_confidence_penalty=0.003,
    temporal_time_smoothness_weight=0.0015,
)


def default_regularization_profile() -> dict[str, float]:
    return asdict(_DEFAULT_PROFILE)


def build_database_regularization_profiles(df: pd.DataFrame, scales: dict[str, Any] | None = None) -> dict[str, dict[str, float]]:
    if df.empty:
        return {}

    scales = scales or {}
    input_scales = scales.get('input', {}) if isinstance(scales.get('input', {}), dict) else {}
    occ_scales = scales.get('occurrence', {}) if isinstance(scales.get('occurrence', {}), dict) else {}
    tmp_scales = scales.get('temporal', {}) if isinstance(scales.get('temporal', {}), dict) else {}

    total_events = float(len(df))
    rows: list[dict[str, float | int | str]] = []

    for database_id, group in df.groupby('database_id'):
        weekly = group.groupby(['robot_id', 'week_start']).size()
        per_week_task = group.groupby(['week_start', 'task_type']).size().unstack(fill_value=0).sort_index()
        minute_bin = (group['minute_of_day'].astype(int) // 5).astype(int)

        day_entropies = []
        time_entropies = []
        duration_modes = []
        for task_type, task_group in group.groupby('task_type'):
            day_entropies.append(_safe_entropy(task_group['day_of_week'], classes=list(range(7))))
            time_entropies.append(_safe_entropy((task_group['minute_of_day'].astype(int) // 5).astype(int)))
            duration_modes.append(float(task_group['duration_minutes'].value_counts(normalize=True).iloc[0]))

        rows.append({
            'database_id': database_id,
            'events': int(len(group)),
            'weeks': int(group['week_start'].nunique()),
            'robots': int(group['robot_id'].nunique()),
            'sample_share': float(len(group) / max(total_events, 1.0)),
            'weekly_cv': float(weekly.std() / max(weekly.mean(), 1e-6)) if len(weekly) else 0.0,
            'task_count_cv_mean': float(per_week_task.std().div(per_week_task.mean().replace(0, np.nan)).fillna(0.0).mean()) if not per_week_task.empty else 0.0,
            'day_entropy_mean': float(np.mean(day_entropies)) if day_entropies else 0.0,
            'time_entropy_mean': float(np.mean(time_entropies)) if time_entropies else _safe_entropy(minute_bin),
            'duration_modal_share_mean': float(np.mean(duration_modes)) if duration_modes else 1.0,
        })

    stats = pd.DataFrame(rows).set_index('database_id')
    max_weeks = max(float(stats['weeks'].max()), 1.0)
    max_share = max(float(stats['sample_share'].max()), 1e-8)

    scarcity = (
        0.65 * (1.0 - np.log1p(stats['weeks']) / np.log1p(max_weeks))
        + 0.35 * (1.0 - np.sqrt(stats['sample_share'] / max_share))
    )
    volatility = 0.55 * _normalize_series(stats['weekly_cv']) + 0.45 * _normalize_series(stats['task_count_cv_mean'])
    rigidity = (
        0.60 * (1.0 - _normalize_series(stats['time_entropy_mean']))
        + 0.25 * (1.0 - _normalize_series(stats['day_entropy_mean']))
        + 0.15 * stats['duration_modal_share_mean'].astype(float)
    )

    history_dropout_scale = float(input_scales.get('history_dropout_scale', 1.0))
    history_noise_scale = float(input_scales.get('history_noise_scale', 1.0))
    feature_dropout_scale = float(input_scales.get('feature_dropout_scale', 1.0))
    feature_noise_scale = float(input_scales.get('feature_noise_scale', 1.0))

    occ_label_smoothing_scale = float(occ_scales.get('label_smoothing_scale', 1.0))
    occ_sigma_scale = float(occ_scales.get('target_sigma_scale', 1.0))
    occ_shrink_scale = float(occ_scales.get('shrink_scale', 1.0))
    occ_conf_scale = float(occ_scales.get('confidence_scale', 1.0))

    tmp_label_smoothing_scale = float(tmp_scales.get('label_smoothing_scale', 1.0))
    tmp_sigma_scale = float(tmp_scales.get('time_sigma_scale', 1.0))
    tmp_anchor_scale = float(tmp_scales.get('anchor_scale', 1.0))
    tmp_prior_scale = float(tmp_scales.get('prior_scale', 1.0))
    tmp_conf_scale = float(tmp_scales.get('confidence_scale', 1.0))
    tmp_smooth_scale = float(tmp_scales.get('smoothness_scale', 1.0))

    profiles: dict[str, dict[str, float]] = {}
    for database_id, row in stats.iterrows():
        scarcity_score = float(scarcity.loc[database_id])
        volatility_score = float(volatility.loc[database_id])
        rigidity_score = float(rigidity.loc[database_id])

        history_dropout = _clamp((0.02 + 0.12 * scarcity_score) * history_dropout_scale, 0.0, 0.30)
        history_noise_std = _clamp((0.002 + 0.025 * scarcity_score) * history_noise_scale, 0.0, 0.08)
        feature_dropout = _clamp((0.02 + 0.10 * scarcity_score) * feature_dropout_scale, 0.0, 0.25)
        feature_noise_std = _clamp((0.001 + 0.02 * scarcity_score) * feature_noise_scale, 0.0, 0.06)

        occurrence_label_smoothing = _clamp((0.01 + 0.05 * scarcity_score + 0.04 * volatility_score) * occ_label_smoothing_scale, 0.0, 0.20)
        occurrence_delta_target_sigma = _clamp((0.75 + 1.25 * volatility_score) * occ_sigma_scale, 0.35, 3.50)
        occurrence_count_shrink_weight = _clamp((0.015 + 0.09 * (0.5 * scarcity_score + 0.5 * rigidity_score)) * occ_shrink_scale, 0.0, 0.25)
        occurrence_confidence_penalty = _clamp((0.0005 + 0.008 * scarcity_score) * occ_conf_scale, 0.0, 0.03)

        temporal_day_label_smoothing = _clamp((0.01 + 0.05 * scarcity_score + 0.02 * volatility_score) * tmp_label_smoothing_scale, 0.0, 0.20)
        temporal_time_target_sigma = _clamp((1.0 + 2.5 * volatility_score + 1.0 * scarcity_score) * tmp_sigma_scale, 0.50, 6.00)
        temporal_anchor_weight = _clamp((0.02 + 0.12 * rigidity_score * (1.0 - 0.5 * volatility_score)) * tmp_anchor_scale, 0.0, 0.25)
        temporal_day_prior_weight = _clamp((0.04 + 0.18 * (0.6 * rigidity_score + 0.4 * scarcity_score)) * tmp_prior_scale, 0.0, 0.35)
        temporal_time_prior_weight = _clamp((0.05 + 0.20 * (0.6 * rigidity_score + 0.4 * scarcity_score)) * tmp_prior_scale, 0.0, 0.40)
        temporal_confidence_penalty = _clamp((0.0005 + 0.007 * scarcity_score) * tmp_conf_scale, 0.0, 0.025)
        temporal_time_smoothness_weight = _clamp((0.0004 + 0.004 * (0.5 * scarcity_score + 0.5 * volatility_score)) * tmp_smooth_scale, 0.0, 0.02)

        profile = DatabaseRegularizationProfile(
            database_id=database_id,
            events=int(row['events']),
            weeks=int(row['weeks']),
            robots=int(row['robots']),
            sample_share=float(row['sample_share']),
            weekly_cv=float(row['weekly_cv']),
            task_count_cv_mean=float(row['task_count_cv_mean']),
            day_entropy_mean=float(row['day_entropy_mean']),
            time_entropy_mean=float(row['time_entropy_mean']),
            duration_modal_share_mean=float(row['duration_modal_share_mean']),
            scarcity_score=scarcity_score,
            volatility_score=volatility_score,
            rigidity_score=rigidity_score,
            history_dropout=history_dropout,
            history_noise_std=history_noise_std,
            feature_dropout=feature_dropout,
            feature_noise_std=feature_noise_std,
            occurrence_label_smoothing=occurrence_label_smoothing,
            occurrence_delta_target_sigma=occurrence_delta_target_sigma,
            occurrence_count_shrink_weight=occurrence_count_shrink_weight,
            occurrence_confidence_penalty=occurrence_confidence_penalty,
            temporal_day_label_smoothing=temporal_day_label_smoothing,
            temporal_time_target_sigma=temporal_time_target_sigma,
            temporal_anchor_weight=temporal_anchor_weight,
            temporal_day_prior_weight=temporal_day_prior_weight,
            temporal_time_prior_weight=temporal_time_prior_weight,
            temporal_confidence_penalty=temporal_confidence_penalty,
            temporal_time_smoothness_weight=temporal_time_smoothness_weight,
        )
        profiles[database_id] = asdict(profile)
    return profiles


def render_regularization_markdown(profiles: dict[str, dict[str, float]]) -> str:
    if not profiles:
        return '# Perfiles de regularización\n\nNo hay perfiles disponibles.'
    lines = [
        '# Perfiles de regularización por dataset',
        '',
        'Estos perfiles se derivan del tamaño histórico, volatilidad semanal y rigidez temporal de cada base.',
        '',
    ]
    for database_id in sorted(profiles.keys()):
        p = profiles[database_id]
        lines.extend([
            f'## {database_id}',
            '',
            f"- eventos / semanas / robots: **{p['events']} / {p['weeks']} / {p['robots']}**",
            f"- escasez / volatilidad / rigidez: **{p['scarcity_score']:.3f} / {p['volatility_score']:.3f} / {p['rigidity_score']:.3f}**",
            f"- ruido de entrada: history_dropout={p['history_dropout']:.3f}, history_noise={p['history_noise_std']:.3f}, feature_dropout={p['feature_dropout']:.3f}, feature_noise={p['feature_noise_std']:.3f}",
            f"- occurrence: label_smoothing={p['occurrence_label_smoothing']:.3f}, delta_sigma={p['occurrence_delta_target_sigma']:.3f}, shrink={p['occurrence_count_shrink_weight']:.3f}, confidence_penalty={p['occurrence_confidence_penalty']:.4f}",
            f"- temporal: day_smoothing={p['temporal_day_label_smoothing']:.3f}, time_sigma={p['temporal_time_target_sigma']:.3f}, anchor={p['temporal_anchor_weight']:.3f}, day_prior={p['temporal_day_prior_weight']:.3f}, time_prior={p['temporal_time_prior_weight']:.3f}, confidence_penalty={p['temporal_confidence_penalty']:.4f}, smoothness={p['temporal_time_smoothness_weight']:.4f}",
            '',
        ])
    return '\n'.join(lines)
