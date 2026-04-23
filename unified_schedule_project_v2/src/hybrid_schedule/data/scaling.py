from __future__ import annotations

from typing import Any

import numpy as np


def _signed_log1p(arr: np.ndarray) -> np.ndarray:
    return np.sign(arr) * np.log1p(np.abs(arr))


def _to_float_array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _sanitize_std(std: np.ndarray) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return std


def _history_log_indices(feature_dim: int) -> list[int]:
    if feature_dim < 9:
        return []
    num_tasks = max(0, (feature_dim - 9) // 14)
    return [task_idx * 14 for task_idx in range(num_tasks)]


def fit_feature_stats(matrix: np.ndarray, log_indices: list[int] | None = None) -> dict[str, Any]:
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'fit_feature_stats espera una matriz 2D y recibió shape={arr.shape}')
    transformed = arr.copy()
    log_indices = [int(i) for i in (log_indices or []) if 0 <= int(i) < transformed.shape[1]]
    if log_indices:
        transformed[:, log_indices] = _signed_log1p(transformed[:, log_indices])
    mean = transformed.mean(axis=0).astype(np.float32)
    std = _sanitize_std(transformed.std(axis=0).astype(np.float32))
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'log_indices': [int(i) for i in log_indices],
    }


def transform_feature_matrix(matrix: np.ndarray, stats: dict[str, Any] | None) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if stats is None:
        return arr.astype(np.float32, copy=True)
    transformed = arr.astype(np.float32, copy=True)
    log_indices = [int(i) for i in stats.get('log_indices', []) if 0 <= int(i) < transformed.shape[-1]]
    if transformed.ndim == 1:
        transformed = transformed[None, :]
        squeeze_back = True
    else:
        squeeze_back = False
    if log_indices:
        transformed[:, log_indices] = _signed_log1p(transformed[:, log_indices])
    mean = _to_float_array(stats['mean'])
    std = _sanitize_std(_to_float_array(stats['std']))
    transformed = (transformed - mean) / std
    if squeeze_back:
        transformed = transformed[0]
    return transformed.astype(np.float32, copy=False)


def fit_history_stats_from_samples(samples: list[dict[str, Any]], key: str = 'history') -> dict[str, Any]:
    history = np.concatenate([np.asarray(sample[key], dtype=np.float32) for sample in samples], axis=0)
    return fit_feature_stats(history, log_indices=_history_log_indices(history.shape[1]))


def fit_vector_stats_from_samples(samples: list[dict[str, Any]], key: str, log_indices: list[int] | None = None) -> dict[str, Any]:
    matrix = np.stack([np.asarray(sample[key], dtype=np.float32) for sample in samples], axis=0)
    return fit_feature_stats(matrix, log_indices=log_indices)


def fit_candidate_stats_from_samples(samples: list[dict[str, Any]], feature_key: str = 'candidate_features', mask_key: str = 'candidate_mask', log_indices: list[int] | None = None) -> dict[str, Any]:
    rows = []
    for sample in samples:
        features = np.asarray(sample[feature_key], dtype=np.float32)
        mask = np.asarray(sample[mask_key], dtype=bool)
        if mask.any():
            rows.append(features[mask])
    if not rows:
        raise RuntimeError('No hay candidatos válidos para ajustar la normalización temporal')
    matrix = np.concatenate(rows, axis=0)
    return fit_feature_stats(matrix, log_indices=log_indices)


def apply_occurrence_scaling(dataset, history_stats: dict[str, Any] | None, numeric_stats: dict[str, Any] | None) -> None:
    for sample in dataset.samples:
        sample['history'] = transform_feature_matrix(sample['history'], history_stats)
        sample['numeric_features'] = transform_feature_matrix(sample['numeric_features'], numeric_stats)


def apply_temporal_scaling(dataset, history_stats: dict[str, Any] | None, numeric_stats: dict[str, Any] | None, candidate_stats: dict[str, Any] | None) -> None:
    for sample in dataset.samples:
        sample['history'] = transform_feature_matrix(sample['history'], history_stats)
        sample['numeric_features'] = transform_feature_matrix(sample['numeric_features'], numeric_stats)
        features = np.asarray(sample['candidate_features'], dtype=np.float32)
        mask = np.asarray(sample['candidate_mask'], dtype=bool)
        if mask.any():
            features = features.copy()
            features[mask] = transform_feature_matrix(features[mask], candidate_stats)
            sample['candidate_features'] = features.astype(np.float32, copy=False)


OCCURRENCE_NUMERIC_LOG_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 21, 22, 31, 32, 33, 34, 35, 36, 38]
TEMPORAL_NUMERIC_LOG_INDICES = [3, 5, 10, 11, 13, 15, 16, 17, 19, 21, 22, 25, 26, 27, 28, 29, 35, 36, 37, 39, 45, 46, 47, 48, 49, 50]
TEMPORAL_CANDIDATE_LOG_INDICES = [4, 5, 6, 7, 8, 9, 13, 14, 17, 18]
