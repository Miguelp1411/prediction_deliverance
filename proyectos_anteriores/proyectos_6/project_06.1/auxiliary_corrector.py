from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from config import (
    BIN_MINUTES,
    CHECKPOINT_DIR,
    PREDICTION_REPAIR_RADIUS_BINS,
    WINDOW_WEEKS,
    bins_per_day,
    num_time_bins,
)
from data.preprocessing import build_temporal_context
from predict import predict_next_week
from utils.serialization import load_checkpoint, save_checkpoint

AUXILIARY_CHECKPOINT_PATH = CHECKPOINT_DIR / 'auxiliary_corrector.pt'
_COUNT_SCALES = (1, 2, 4, 8, 16)


@dataclass
class TinyRidgeRegressor:
    alpha: float = 1.0
    coef_: np.ndarray | None = None
    intercept_: np.ndarray | None = None
    feature_mean_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        if len(x) == 0:
            self.feature_mean_ = np.zeros(x.shape[1], dtype=np.float64)
            self.feature_scale_ = np.ones(x.shape[1], dtype=np.float64)
            self.coef_ = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
            self.intercept_ = np.zeros(y.shape[1], dtype=np.float64)
            return self
        self.feature_mean_ = x.mean(axis=0)
        self.feature_scale_ = x.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-6] = 1.0
        xs = (x - self.feature_mean_) / self.feature_scale_
        x_aug = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float64)], axis=1)
        reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(self.alpha)
        reg[-1, -1] = 0.0
        weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
        self.coef_ = weights[:-1]
        self.intercept_ = weights[-1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError('Regresor auxiliar no entrenado')
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        xs = (x - self.feature_mean_) / self.feature_scale_
        out = xs @ self.coef_ + self.intercept_
        return out[:, 0] if out.shape[1] == 1 else out

    def to_dict(self) -> dict[str, Any]:
        return {
            'alpha': float(self.alpha),
            'coef_': None if self.coef_ is None else self.coef_.tolist(),
            'intercept_': None if self.intercept_ is None else self.intercept_.tolist(),
            'feature_mean_': None if self.feature_mean_ is None else self.feature_mean_.tolist(),
            'feature_scale_': None if self.feature_scale_ is None else self.feature_scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'TinyRidgeRegressor':
        model = cls(alpha=float(payload.get('alpha', 1.0)))
        if payload.get('coef_') is not None:
            model.coef_ = np.asarray(payload['coef_'], dtype=np.float64)
        if payload.get('intercept_') is not None:
            model.intercept_ = np.asarray(payload['intercept_'], dtype=np.float64)
        if payload.get('feature_mean_') is not None:
            model.feature_mean_ = np.asarray(payload['feature_mean_'], dtype=np.float64)
        if payload.get('feature_scale_') is not None:
            model.feature_scale_ = np.asarray(payload['feature_scale_'], dtype=np.float64)
        return model


@dataclass
class TinyLogisticRegressor:
    l2: float = 1.0
    lr: float = 0.05
    max_iter: int = 250
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    feature_mean_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None
    constant_prob_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(x) == 0:
            self.constant_prob_ = 0.5
            self.feature_mean_ = np.zeros(x.shape[1], dtype=np.float64)
            self.feature_scale_ = np.ones(x.shape[1], dtype=np.float64)
            self.coef_ = np.zeros(x.shape[1], dtype=np.float64)
            self.intercept_ = 0.0
            return self
        ratio = float(np.clip(y.mean(), 0.0, 1.0))
        self.feature_mean_ = x.mean(axis=0)
        self.feature_scale_ = x.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-6] = 1.0
        if ratio <= 1e-6 or ratio >= 1.0 - 1e-6:
            self.constant_prob_ = ratio
            self.coef_ = np.zeros(x.shape[1], dtype=np.float64)
            self.intercept_ = float(np.log((ratio + 1e-6) / (1.0 - ratio + 1e-6)))
            return self
        xs = (x - self.feature_mean_) / self.feature_scale_
        w = np.zeros(xs.shape[1], dtype=np.float64)
        b = 0.0
        for _ in range(int(self.max_iter)):
            logits = xs @ w + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            err = probs - y
            grad_w = (xs.T @ err) / max(len(xs), 1) + self.l2 * w
            grad_b = float(np.mean(err))
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_ = w
        self.intercept_ = float(b)
        self.constant_prob_ = None
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if self.constant_prob_ is not None:
            return np.full(x.shape[0], float(self.constant_prob_), dtype=np.float64)
        if self.coef_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError('Clasificador auxiliar no entrenado')
        xs = (x - self.feature_mean_) / self.feature_scale_
        logits = xs @ self.coef_ + float(self.intercept_)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            'l2': float(self.l2),
            'lr': float(self.lr),
            'max_iter': int(self.max_iter),
            'coef_': None if self.coef_ is None else self.coef_.tolist(),
            'intercept_': float(self.intercept_),
            'feature_mean_': None if self.feature_mean_ is None else self.feature_mean_.tolist(),
            'feature_scale_': None if self.feature_scale_ is None else self.feature_scale_.tolist(),
            'constant_prob_': self.constant_prob_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'TinyLogisticRegressor':
        model = cls(
            l2=float(payload.get('l2', 1.0)),
            lr=float(payload.get('lr', 0.05)),
            max_iter=int(payload.get('max_iter', 250)),
        )
        if payload.get('coef_') is not None:
            model.coef_ = np.asarray(payload['coef_'], dtype=np.float64)
        if payload.get('feature_mean_') is not None:
            model.feature_mean_ = np.asarray(payload['feature_mean_'], dtype=np.float64)
        if payload.get('feature_scale_') is not None:
            model.feature_scale_ = np.asarray(payload['feature_scale_'], dtype=np.float64)
        model.intercept_ = float(payload.get('intercept_', 0.0))
        model.constant_prob_ = payload.get('constant_prob_')
        return model


@dataclass
class AuxiliaryCorrector:
    task_names: list[str]
    max_count_cap: int
    max_occurrences_per_task: int
    duration_min: float
    duration_max: float
    count_regressor: TinyRidgeRegressor
    count_gate: TinyLogisticRegressor
    temporal_regressor: TinyRidgeRegressor
    time_gate: TinyLogisticRegressor
    duration_gate: TinyLogisticRegressor
    count_conf_threshold: float = 0.55
    time_conf_threshold: float = 0.60
    duration_conf_threshold: float = 0.55
    count_min_abs_delta: float = 0.35
    time_min_abs_delta_bins: float = 1.0
    duration_min_abs_delta_minutes: float = 2.0
    metadata: dict[str, Any] | None = None

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            'metadata': {
                'task_names': list(self.task_names),
                'max_count_cap': int(self.max_count_cap),
                'max_occurrences_per_task': int(self.max_occurrences_per_task),
                'duration_min': float(self.duration_min),
                'duration_max': float(self.duration_max),
                'count_conf_threshold': float(self.count_conf_threshold),
                'time_conf_threshold': float(self.time_conf_threshold),
                'duration_conf_threshold': float(self.duration_conf_threshold),
                'count_min_abs_delta': float(self.count_min_abs_delta),
                'time_min_abs_delta_bins': float(self.time_min_abs_delta_bins),
                'duration_min_abs_delta_minutes': float(self.duration_min_abs_delta_minutes),
                'training_metadata': self.metadata or {},
            },
            'count_regressor': self.count_regressor.to_dict(),
            'count_gate': self.count_gate.to_dict(),
            'temporal_regressor': self.temporal_regressor.to_dict(),
            'time_gate': self.time_gate.to_dict(),
            'duration_gate': self.duration_gate.to_dict(),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> 'AuxiliaryCorrector':
        meta = payload.get('metadata', {}) or {}
        return cls(
            task_names=list(meta.get('task_names', [])),
            max_count_cap=int(meta.get('max_count_cap', 1)),
            max_occurrences_per_task=int(meta.get('max_occurrences_per_task', meta.get('max_count_cap', 1))),
            duration_min=float(meta.get('duration_min', 0.0)),
            duration_max=float(meta.get('duration_max', 1.0)),
            count_regressor=TinyRidgeRegressor.from_dict(payload.get('count_regressor', {})),
            count_gate=TinyLogisticRegressor.from_dict(payload.get('count_gate', {})),
            temporal_regressor=TinyRidgeRegressor.from_dict(payload.get('temporal_regressor', {})),
            time_gate=TinyLogisticRegressor.from_dict(payload.get('time_gate', {})),
            duration_gate=TinyLogisticRegressor.from_dict(payload.get('duration_gate', {})),
            count_conf_threshold=float(meta.get('count_conf_threshold', 0.55)),
            time_conf_threshold=float(meta.get('time_conf_threshold', 0.60)),
            duration_conf_threshold=float(meta.get('duration_conf_threshold', 0.55)),
            count_min_abs_delta=float(meta.get('count_min_abs_delta', 0.35)),
            time_min_abs_delta_bins=float(meta.get('time_min_abs_delta_bins', 1.0)),
            duration_min_abs_delta_minutes=float(meta.get('duration_min_abs_delta_minutes', 2.0)),
            metadata=meta.get('training_metadata', {}),
        )

    def save(self, path: str | None = None):
        save_checkpoint(path or AUXILIARY_CHECKPOINT_PATH, self.to_checkpoint())

    @classmethod
    def load(cls, path: str | None = None) -> 'AuxiliaryCorrector':
        return cls.from_checkpoint(load_checkpoint(path or AUXILIARY_CHECKPOINT_PATH, map_location='cpu'))

    @classmethod
    def fit_from_history(
        cls,
        prepared,
        occurrence_model,
        temporal_model,
        device,
        week_indices: list[int] | None = None,
    ) -> 'AuxiliaryCorrector':
        if week_indices is None:
            recent_start = max(WINDOW_WEEKS, len(prepared.weeks) - 2)
            week_indices = list(range(recent_start, len(prepared.weeks)))
        week_indices = [int(idx) for idx in week_indices if WINDOW_WEEKS <= int(idx) < len(prepared.weeks)]
        if not week_indices:
            raise ValueError('No hay semanas cerradas suficientes para entrenar el corrector auxiliar')

        count_rows: list[np.ndarray] = []
        count_targets: list[float] = []
        temporal_rows: list[np.ndarray] = []
        temporal_targets: list[np.ndarray] = []
        temporal_truth_rows: list[dict[str, Any]] = []
        replay_stats: list[dict[str, Any]] = []

        for week_idx in week_indices:
            base_predictions = predict_next_week(
                occurrence_model,
                temporal_model,
                prepared,
                week_idx,
                device,
                use_repair=True,
                auxiliary_corrector=None,
            )
            true_week = prepared.weeks[week_idx]
            count_rows.extend(_collect_count_examples(prepared, week_idx, base_predictions, true_week, count_targets))
            rows, targets, truths = _collect_temporal_examples(prepared, week_idx, base_predictions, true_week)
            temporal_rows.extend(rows)
            temporal_targets.extend(targets)
            temporal_truth_rows.extend(truths)
            replay_stats.append({
                'week_idx': int(week_idx),
                'predicted_items': len(base_predictions),
                'true_items': int(np.sum(true_week.counts)),
            })

        count_x = np.asarray(count_rows, dtype=np.float64)
        count_y = np.asarray(count_targets, dtype=np.float64)
        temporal_x = np.asarray(temporal_rows, dtype=np.float64)
        temporal_y = np.asarray(temporal_targets, dtype=np.float64)

        count_regressor = TinyRidgeRegressor(alpha=1.5).fit(count_x, count_y)
        temporal_regressor = TinyRidgeRegressor(alpha=2.0).fit(temporal_x, temporal_y)

        count_pred_delta = np.asarray(count_regressor.predict(count_x), dtype=np.float64).reshape(-1)
        count_gate_x, count_gate_y = _build_count_gate_dataset(
            count_x,
            count_pred_delta,
            count_targets,
            prepared.max_count_cap,
            len(prepared.task_names),
        )
        count_gate = TinyLogisticRegressor(l2=0.10, lr=0.08, max_iter=300).fit(count_gate_x, count_gate_y)

        temporal_pred = np.asarray(temporal_regressor.predict(temporal_x), dtype=np.float64)
        time_gate_x, time_gate_y, duration_gate_x, duration_gate_y = _build_temporal_gate_dataset(
            temporal_x,
            temporal_pred,
            temporal_truth_rows,
        )
        time_gate = TinyLogisticRegressor(l2=0.10, lr=0.08, max_iter=300).fit(time_gate_x, time_gate_y)
        duration_gate = TinyLogisticRegressor(l2=0.10, lr=0.08, max_iter=300).fit(duration_gate_x, duration_gate_y)

        return cls(
            task_names=list(prepared.task_names),
            max_count_cap=int(prepared.max_count_cap),
            max_occurrences_per_task=int(prepared.max_occurrences_per_task),
            duration_min=float(prepared.duration_min),
            duration_max=float(prepared.duration_max),
            count_regressor=count_regressor,
            count_gate=count_gate,
            temporal_regressor=temporal_regressor,
            time_gate=time_gate,
            duration_gate=duration_gate,
            metadata={
                'weeks_used': len(week_indices),
                'count_examples': int(len(count_x)),
                'temporal_examples': int(len(temporal_x)),
                'replay_stats': replay_stats[-5:],
            },
        )

    def apply(self, prepared, target_week_idx: int, base_predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        task_groups = _group_predictions_by_task(base_predictions, prepared.task_to_id)
        count_adjusted = self._apply_count_corrections(prepared, target_week_idx, task_groups)
        temporal_adjusted = self._apply_temporal_corrections(prepared, target_week_idx, count_adjusted)
        return _repair_final_predictions(temporal_adjusted)

    def _apply_count_corrections(self, prepared, target_week_idx: int, task_groups: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        adjusted: list[dict[str, Any]] = []
        for task_id, task_name in enumerate(self.task_names):
            current_items = sorted(task_groups.get(task_id, []), key=lambda x: (x['start_bin'], x['task_name']))
            predicted_count = len(current_items)
            feat = _build_count_features(prepared, target_week_idx, task_id, predicted_count)
            raw_delta = float(self.count_regressor.predict(feat)[0])
            conf = float(self.count_gate.predict_proba(_augment_count_gate_features(feat[0], raw_delta))[0])
            corrected_count = predicted_count
            if conf >= self.count_conf_threshold and abs(raw_delta) >= self.count_min_abs_delta:
                corrected_count = int(np.clip(round(predicted_count + raw_delta), 0, self.max_count_cap))
            corrected_items = _resize_task_group(
                prepared,
                target_week_idx,
                task_id,
                task_name,
                current_items,
                corrected_count,
            )
            adjusted.extend(corrected_items)
        return adjusted

    def _apply_temporal_corrections(self, prepared, target_week_idx: int, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped = _group_predictions_by_task(predictions, prepared.task_to_id)
        corrected: list[dict[str, Any]] = []
        for task_id, task_name in enumerate(self.task_names):
            items = sorted(grouped.get(task_id, []), key=lambda x: (x['start_bin'], x['task_name']))
            predicted_count = len(items)
            for occurrence_index, item in enumerate(items):
                feat = _build_temporal_features(
                    prepared,
                    target_week_idx,
                    task_id,
                    occurrence_index,
                    int(item['start_bin']),
                    float(item['duration']),
                    predicted_count,
                )
                raw = np.asarray(self.temporal_regressor.predict(feat), dtype=np.float64)[0]
                raw_day = int(np.clip(round(float(raw[0])), -2, 2))
                raw_local = int(np.clip(round(float(raw[1])), -36, 36))
                raw_duration = float(np.clip(float(raw[2]), -90.0, 90.0))
                gate_feat = _augment_temporal_gate_features(feat[0], raw_day, raw_local, raw_duration)
                time_conf = float(self.time_gate.predict_proba(gate_feat)[0])
                duration_conf = float(self.duration_gate.predict_proba(gate_feat)[0])
                new_item = dict(item)
                if time_conf >= self.time_conf_threshold and (abs(raw_day) > 0 or abs(raw_local) >= self.time_min_abs_delta_bins):
                    delta_bins = raw_day * bins_per_day() + raw_local
                    new_item['start_bin'] = int(np.clip(int(item['start_bin']) + delta_bins, 0, num_time_bins() - 1))
                if duration_conf >= self.duration_conf_threshold and abs(raw_duration) >= self.duration_min_abs_delta_minutes:
                    new_item['duration'] = float(max(BIN_MINUTES, float(item['duration']) + raw_duration))
                corrected.append(new_item)
        return corrected


def train_and_save_auxiliary_corrector(
    prepared,
    occurrence_model,
    temporal_model,
    device,
    week_indices: list[int] | None = None,
    output_path: str | None = None,
) -> AuxiliaryCorrector:
    model = AuxiliaryCorrector.fit_from_history(
        prepared,
        occurrence_model,
        temporal_model,
        device,
        week_indices=week_indices,
    )
    model.save(output_path or AUXILIARY_CHECKPOINT_PATH)
    return model


def maybe_load_auxiliary_corrector(path: str | None = None) -> AuxiliaryCorrector | None:
    try:
        return AuxiliaryCorrector.load(path or AUXILIARY_CHECKPOINT_PATH)
    except Exception:
        return None


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float64)
    if 0 <= int(index) < size:
        out[int(index)] = 1.0
    return out


def _cyclical_encode(value: float, period: float) -> tuple[float, float]:
    if period <= 0:
        return 0.0, 0.0
    angle = 2.0 * np.pi * float(value) / float(period)
    return float(np.sin(angle)), float(np.cos(angle))


def _history_slice(weeks, target_week_idx: int):
    return weeks[max(0, target_week_idx - WINDOW_WEEKS):target_week_idx]


def _infer_target_week_start(weeks, target_week_idx: int):
    if 0 <= target_week_idx < len(weeks):
        return weeks[target_week_idx].week_start
    if target_week_idx == len(weeks):
        return weeks[-1].week_start + pd.Timedelta(days=7)
    return weeks[0].week_start + pd.Timedelta(days=7 * target_week_idx)


def _task_history_duration_stats(history, task_id: int, occurrence_index: int) -> tuple[float, float, float, float]:
    same_occ: list[float] = []
    all_values: list[float] = []
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))
        for idx, event in enumerate(events):
            all_values.append(float(event.duration_minutes))
            if idx == occurrence_index:
                same_occ.append(float(event.duration_minutes))
    values = same_occ or all_values
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values)) if len(values) > 1 else 0.0,
        float(len(values) / max(len(history), 1)),
    )


def _collect_task_counts(history, task_id: int) -> np.ndarray:
    return np.asarray([float(week.counts[task_id]) for week in history], dtype=np.float64) if history else np.zeros(0, dtype=np.float64)


def _build_count_features(prepared, target_week_idx: int, task_id: int, predicted_count: int) -> np.ndarray:
    history = _history_slice(prepared.weeks, target_week_idx)
    counts = _collect_task_counts(history, task_id)
    target_week_start = _infer_target_week_start(prepared.weeks, target_week_idx)
    feats: list[float] = []
    feats.extend(_one_hot(task_id, len(prepared.task_names)).tolist())
    feats.extend([
        float(predicted_count / max(prepared.max_count_cap, 1)),
        float(predicted_count == 0),
        float(predicted_count >= prepared.max_count_cap),
    ])
    for scale in _COUNT_SCALES:
        window = counts[-scale:]
        if len(window) == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend([
                float(window.mean() / max(prepared.max_count_cap, 1)),
                float(window.std() / max(prepared.max_count_cap, 1)),
                float((window > 0).mean()),
            ])
    if len(counts) > 0:
        nonzero = np.where(counts > 0)[0]
        weeks_since_last = float(len(counts) - 1 - nonzero[-1]) if len(nonzero) else float(WINDOW_WEEKS)
        trend = float(counts[-1] - counts.mean()) / max(prepared.max_count_cap, 1)
        mean_count = float(counts.mean() / max(prepared.max_count_cap, 1))
    else:
        weeks_since_last = float(WINDOW_WEEKS)
        trend = 0.0
        mean_count = 0.0
    feats.extend([
        float(min(weeks_since_last, WINDOW_WEEKS) / max(WINDOW_WEEKS, 1)),
        trend,
        mean_count,
    ])
    week_sin, week_cos = _cyclical_encode(int(target_week_start.isocalendar().week) - 1, 52.0)
    month_sin, month_cos = _cyclical_encode(int(target_week_start.month) - 1, 12.0)
    feats.extend([week_sin, week_cos, month_sin, month_cos])
    return np.asarray(feats, dtype=np.float64)[None, :]


def _build_temporal_features(
    prepared,
    target_week_idx: int,
    task_id: int,
    occurrence_index: int,
    pred_start_bin: int,
    pred_duration: float,
    predicted_count: int,
) -> np.ndarray:
    history = _history_slice(prepared.weeks, target_week_idx)
    context = build_temporal_context(
        prepared.weeks,
        target_week_idx,
        task_id,
        occurrence_index,
        duration_min=prepared.duration_min,
        duration_max=prepared.duration_max,
        max_occurrences_per_task=prepared.max_occurrences_per_task,
    )
    pred_day = int(pred_start_bin // bins_per_day())
    pred_time = int(pred_start_bin % bins_per_day())
    start_sin, start_cos = _cyclical_encode(pred_start_bin, num_time_bins())
    day_sin, day_cos = _cyclical_encode(pred_day, 7.0)
    time_sin, time_cos = _cyclical_encode(pred_time, bins_per_day())
    span = max(prepared.duration_max - prepared.duration_min, 1e-6)
    duration_norm = float(np.clip((pred_duration - prepared.duration_min) / span, 0.0, 1.0))
    anchor_delta = float(pred_start_bin - context.anchor_start_bin)
    dur_mean, dur_median, dur_std, dur_support = _task_history_duration_stats(history, task_id, occurrence_index)
    feats: list[float] = []
    feats.extend(_one_hot(task_id, len(prepared.task_names)).tolist())
    feats.extend([
        float(occurrence_index / max(prepared.max_occurrences_per_task - 1, 1)),
        float(predicted_count / max(prepared.max_count_cap, 1)),
        float(pred_start_bin / max(num_time_bins() - 1, 1)),
        start_sin,
        start_cos,
        day_sin,
        day_cos,
        time_sin,
        time_cos,
        duration_norm,
        float(anchor_delta / max(num_time_bins(), 1)),
        float(abs(anchor_delta) / max(num_time_bins(), 1)),
        float(context.anchor_day / 6.0),
        float((dur_mean - prepared.duration_min) / span) if dur_support > 0 else 0.0,
        float((dur_median - prepared.duration_min) / span) if dur_support > 0 else 0.0,
        float(dur_std / max(span, 1.0)),
        float(np.clip(dur_support, 0.0, 1.0)),
    ])
    feats.extend(np.asarray(context.history_features, dtype=np.float64).tolist())
    return np.asarray(feats, dtype=np.float64)[None, :]


def _group_predictions_by_task(predictions: list[dict[str, Any]], task_to_id: dict[str, int]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in predictions:
        task_name = str(item.get('task_name', item.get('type')))
        task_id = int(task_to_id[task_name])
        grouped.setdefault(task_id, []).append({
            'task_id': task_id,
            'task_name': task_name,
            'type': item.get('type', task_name),
            'start_bin': int(item['start_bin']),
            'duration': float(item['duration']),
        })
    return grouped


def _true_week_items(week_record) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task_id, events in week_record.events_by_task.items():
        for event in events:
            out.append({
                'task_id': int(task_id),
                'task_name': str(event.task_name),
                'type': str(event.task_name),
                'start_bin': int(event.start_bin),
                'duration': float(event.duration_minutes),
            })
    return sorted(out, key=lambda x: (x['task_id'], x['start_bin']))


def _collect_count_examples(prepared, week_idx: int, base_predictions: list[dict[str, Any]], true_week, count_targets: list[float]) -> list[np.ndarray]:
    pred_groups = _group_predictions_by_task(base_predictions, prepared.task_to_id)
    rows: list[np.ndarray] = []
    for task_id, _ in enumerate(prepared.task_names):
        predicted_count = len(pred_groups.get(task_id, []))
        true_count = len(true_week.events_by_task.get(task_id, []))
        rows.append(_build_count_features(prepared, week_idx, task_id, predicted_count)[0])
        count_targets.append(float(true_count - predicted_count))
    return rows


def _match_task_group(true_items: list[dict[str, Any]], pred_items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    n, m = len(true_items), len(pred_items)
    if n == 0 or m == 0:
        return []
    size = max(n, m)
    cost = np.full((size, size), 25000.0, dtype=np.float64)
    for i, t in enumerate(true_items):
        for j, p in enumerate(pred_items):
            cost[i, j] = abs(int(t['start_bin']) - int(p['start_bin'])) * BIN_MINUTES + abs(float(t['duration']) - float(p['duration']))
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = []
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r < n and c < m and cost[r, c] < 25000.0:
            pairs.append((true_items[r], pred_items[c]))
    return pairs


def _collect_temporal_examples(prepared, week_idx: int, base_predictions: list[dict[str, Any]], true_week):
    pred_groups = _group_predictions_by_task(base_predictions, prepared.task_to_id)
    true_groups = _group_predictions_by_task(_true_week_items(true_week), prepared.task_to_id)
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    truths: list[dict[str, Any]] = []
    for task_id, _ in enumerate(prepared.task_names):
        pred_items = sorted(pred_groups.get(task_id, []), key=lambda x: x['start_bin'])
        true_items = sorted(true_groups.get(task_id, []), key=lambda x: x['start_bin'])
        if not pred_items or not true_items:
            continue
        pred_count = len(pred_items)
        indexed_pred = []
        indexed_true = []
        for occ_idx, item in enumerate(pred_items):
            indexed_pred.append({**item, 'occurrence_index': occ_idx})
        for occ_idx, item in enumerate(true_items):
            indexed_true.append({**item, 'occurrence_index': occ_idx})
        for true_item, pred_item in _match_task_group(indexed_true, indexed_pred):
            feat = _build_temporal_features(
                prepared,
                week_idx,
                task_id,
                int(pred_item['occurrence_index']),
                int(pred_item['start_bin']),
                float(pred_item['duration']),
                pred_count,
            )[0]
            start_delta_bins = int(true_item['start_bin']) - int(pred_item['start_bin'])
            day_delta = int(np.clip(np.round(start_delta_bins / max(bins_per_day(), 1)), -2, 2))
            local_delta = int(np.clip(start_delta_bins - day_delta * bins_per_day(), -36, 36))
            duration_delta = float(np.clip(float(true_item['duration']) - float(pred_item['duration']), -90.0, 90.0))
            rows.append(feat)
            targets.append(np.asarray([float(day_delta), float(local_delta), duration_delta], dtype=np.float64))
            truths.append({
                'true_start_bin': int(true_item['start_bin']),
                'pred_start_bin': int(pred_item['start_bin']),
                'true_duration': float(true_item['duration']),
                'pred_duration': float(pred_item['duration']),
            })
    return rows, targets, truths


def _augment_count_gate_features(base_feat: np.ndarray, raw_delta: float) -> np.ndarray:
    return np.concatenate([base_feat, np.asarray([raw_delta, abs(raw_delta)], dtype=np.float64)])[None, :]


def _build_count_gate_dataset(
    count_x: np.ndarray,
    count_pred_delta: np.ndarray,
    count_targets: list[float],
    max_count_cap: int,
    num_tasks: int,
):
    gate_x: list[np.ndarray] = []
    gate_y: list[float] = []
    for feat, pred_delta, target_delta in zip(count_x, count_pred_delta, count_targets):
        current_count = int(np.clip(round(float(feat[num_tasks] * max_count_cap)), 0, max_count_cap)) if len(feat) > num_tasks else 0
        corrected_count = int(np.clip(round(current_count + pred_delta), 0, max_count_cap))
        true_count = int(np.clip(round(current_count + target_delta), 0, max_count_cap))
        before = abs(true_count - current_count)
        after = abs(true_count - corrected_count)
        gate_x.append(_augment_count_gate_features(feat, float(pred_delta))[0])
        gate_y.append(float(after + 1e-6 < before))
    return np.asarray(gate_x, dtype=np.float64), np.asarray(gate_y, dtype=np.float64)


def _augment_temporal_gate_features(base_feat: np.ndarray, raw_day: float, raw_local: float, raw_duration: float) -> np.ndarray:
    extra = np.asarray([raw_day, abs(raw_day), raw_local, abs(raw_local), raw_duration, abs(raw_duration)], dtype=np.float64)
    return np.concatenate([base_feat, extra])[None, :]


def _build_temporal_gate_dataset(temporal_x: np.ndarray, temporal_pred: np.ndarray, temporal_truth_rows: list[dict[str, Any]]):
    time_gate_x: list[np.ndarray] = []
    time_gate_y: list[float] = []
    duration_gate_x: list[np.ndarray] = []
    duration_gate_y: list[float] = []
    for feat, pred_vec, truth in zip(temporal_x, temporal_pred, temporal_truth_rows):
        pred_day = int(np.clip(round(float(pred_vec[0])), -2, 2))
        pred_local = int(np.clip(round(float(pred_vec[1])), -36, 36))
        pred_duration_delta = float(np.clip(float(pred_vec[2]), -90.0, 90.0))
        gate_feat = _augment_temporal_gate_features(feat, pred_day, pred_local, pred_duration_delta)[0]
        before_time = abs(int(truth['true_start_bin']) - int(truth['pred_start_bin']))
        after_time = abs(int(truth['true_start_bin']) - (int(truth['pred_start_bin']) + pred_day * bins_per_day() + pred_local))
        before_duration = abs(float(truth['true_duration']) - float(truth['pred_duration']))
        after_duration = abs(float(truth['true_duration']) - (float(truth['pred_duration']) + pred_duration_delta))
        time_gate_x.append(gate_feat)
        time_gate_y.append(float(after_time + 1e-6 < before_time))
        duration_gate_x.append(gate_feat)
        duration_gate_y.append(float(after_duration + 1e-6 < before_duration))
    return (
        np.asarray(time_gate_x, dtype=np.float64),
        np.asarray(time_gate_y, dtype=np.float64),
        np.asarray(duration_gate_x, dtype=np.float64),
        np.asarray(duration_gate_y, dtype=np.float64),
    )


def _task_anchor_and_duration(prepared, target_week_idx: int, task_id: int, occurrence_index: int) -> tuple[int, float]:
    context = build_temporal_context(
        prepared.weeks,
        target_week_idx,
        task_id,
        occurrence_index,
        duration_min=prepared.duration_min,
        duration_max=prepared.duration_max,
        max_occurrences_per_task=prepared.max_occurrences_per_task,
    )
    history = _history_slice(prepared.weeks, target_week_idx)
    same_occ_durations: list[float] = []
    all_durations: list[float] = []
    for week in history:
        events = sorted(week.events_by_task.get(task_id, []), key=lambda e: (e.start_bin, e.start_time))
        for idx, event in enumerate(events):
            all_durations.append(float(event.duration_minutes))
            if idx == occurrence_index:
                same_occ_durations.append(float(event.duration_minutes))
    duration = float(np.median(same_occ_durations or all_durations or [prepared.duration_min]))
    return int(context.anchor_start_bin), duration


def _resize_task_group(prepared, target_week_idx: int, task_id: int, task_name: str, current_items: list[dict[str, Any]], corrected_count: int) -> list[dict[str, Any]]:
    current_items = sorted(current_items, key=lambda x: (x['start_bin'], x['task_name']))
    if corrected_count == len(current_items):
        return [dict(item) for item in current_items]
    if corrected_count < len(current_items):
        scored = []
        for occ_idx, item in enumerate(current_items):
            anchor_start, _ = _task_anchor_and_duration(prepared, target_week_idx, task_id, occ_idx)
            scored.append((abs(int(item['start_bin']) - int(anchor_start)), occ_idx, item))
        keep_idx = {occ_idx for _, occ_idx, _ in sorted(scored, key=lambda x: (x[0], x[1]))[:corrected_count]}
        return [dict(item) for occ_idx, item in enumerate(current_items) if occ_idx in keep_idx]
    expanded = [dict(item) for item in current_items]
    for occ_idx in range(len(current_items), corrected_count):
        anchor_start, duration = _task_anchor_and_duration(prepared, target_week_idx, task_id, occ_idx)
        expanded.append({
            'task_id': int(task_id),
            'task_name': task_name,
            'type': task_name,
            'start_bin': int(anchor_start),
            'duration': float(duration),
        })
    return sorted(expanded, key=lambda x: (x['start_bin'], x['task_name']))


def _overlaps(start_bin: int, duration_bins: int, placed_intervals: list[tuple[int, int]]) -> bool:
    end_bin = start_bin + duration_bins
    for other_start, other_end in placed_intervals:
        if start_bin < other_end and end_bin > other_start:
            return True
    return False


def _nearest_valid_start(preferred_start: int, duration_bins: int, placed_intervals: list[tuple[int, int]], lower_bound: int = 0) -> int:
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


def _repair_final_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    placed: list[tuple[int, int]] = []
    last_start_by_task: dict[int, int] = {}
    final_predictions: list[dict[str, Any]] = []
    ordered = sorted(predictions, key=lambda x: (int(x['start_bin']), int(x.get('task_id', 0)), x.get('task_name', '')))
    for item in ordered:
        duration_bins = max(1, int(round(float(item['duration']) / BIN_MINUTES)))
        task_id = int(item.get('task_id', 0))
        lower_bound = last_start_by_task.get(task_id, 0)
        chosen_start = _nearest_valid_start(int(item['start_bin']), duration_bins, placed, lower_bound=lower_bound)
        placed.append((chosen_start, chosen_start + duration_bins))
        placed.sort()
        last_start_by_task[task_id] = chosen_start
        final_predictions.append({
            'task_name': item['task_name'],
            'type': item.get('type', item['task_name']),
            'start_bin': int(chosen_start),
            'duration': float(max(BIN_MINUTES, item['duration'])),
        })
    return sorted(final_predictions, key=lambda x: (x['start_bin'], x['task_name']))
