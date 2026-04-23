from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BIN_MINUTES,
    CHECKPOINT_DIR,
    PREDICTION_REPAIR_DAY_CHANGE_PENALTY,
    PREDICTION_REPAIR_GLOBAL_FALLBACK,
    PREDICTION_REPAIR_MAX_DAY_SHIFT,
    PREDICTION_REPAIR_RADIUS_BINS,
    WINDOW_WEEKS,
    bins_per_day,
    num_time_bins,
)
from data.preprocessing import build_temporal_context
from predict import predict_next_week
from utils.serialization import load_checkpoint, save_checkpoint

AUXILIARY_CHECKPOINT_PATH = CHECKPOINT_DIR / 'auxiliary_corrector.pt'
AUXILIARY_POLICY_PATH = CHECKPOINT_DIR / 'auxiliary_policy.json'
_COUNT_SCALES = (1, 2, 4, 8, 16)

@dataclass
class AuxiliaryUsagePolicy:
    use_auxiliary: bool = False
    use_aux_count: bool = False
    use_aux_duration: bool = False
    use_aux_time: bool = False
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'use_auxiliary': bool(self.use_auxiliary),
            'use_aux_count': bool(self.use_aux_count),
            'use_aux_duration': bool(self.use_aux_duration),
            'use_aux_time': bool(self.use_aux_time),
            'metrics': self.metrics or {},
        }

    def save(self, path: str | None = None):
        import json
        from pathlib import Path

        target = Path(path or AUXILIARY_POLICY_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> 'AuxiliaryUsagePolicy':
        payload = payload or {}
        return cls(
            use_auxiliary=bool(payload.get('use_auxiliary', False)),
            use_aux_count=bool(payload.get('use_aux_count', False)),
            use_aux_duration=bool(payload.get('use_aux_duration', False)),
            use_aux_time=bool(payload.get('use_aux_time', False)),
            metrics=dict(payload.get('metrics', {}) or {}),
        )

    @classmethod
    def load(cls, path: str | None = None) -> 'AuxiliaryUsagePolicy':
        import json
        from pathlib import Path

        payload = json.loads(Path(path or AUXILIARY_POLICY_PATH).read_text(encoding='utf-8'))
        return cls.from_dict(payload)




class _AuxMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...] = (32, 16), dropout: float = 0.0):
        super().__init__()
        dims = [int(input_dim), *[int(h) for h in hidden_dims if int(h) > 0], int(output_dim)]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TinyNeuralRegressor:
    hidden_dims: tuple[int, ...] = (32, 16)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 140
    batch_size: int = 64
    dropout: float = 0.0
    feature_mean_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    target_scale_: np.ndarray | None = None
    state_dict_: dict[str, torch.Tensor] | None = None
    input_dim_: int | None = None
    output_dim_: int | None = None
    constant_output_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        if x.ndim == 1:
            x = x[:, None]
        self.input_dim_ = int(x.shape[1]) if x.size else 0
        self.output_dim_ = int(y.shape[1]) if y.size else 1
        if len(x) == 0:
            self.feature_mean_ = np.zeros(self.input_dim_, dtype=np.float32)
            self.feature_scale_ = np.ones(self.input_dim_, dtype=np.float32)
            self.target_mean_ = np.zeros(self.output_dim_, dtype=np.float32)
            self.target_scale_ = np.ones(self.output_dim_, dtype=np.float32)
            self.constant_output_ = np.zeros(self.output_dim_, dtype=np.float32)
            self.state_dict_ = None
            return self
        self.feature_mean_ = x.mean(axis=0).astype(np.float32)
        self.feature_scale_ = x.std(axis=0).astype(np.float32)
        self.feature_scale_[self.feature_scale_ < 1e-6] = 1.0
        self.target_mean_ = y.mean(axis=0).astype(np.float32)
        self.target_scale_ = y.std(axis=0).astype(np.float32)
        self.target_scale_[self.target_scale_ < 1e-6] = 1.0
        xs = ((x - self.feature_mean_) / self.feature_scale_).astype(np.float32)
        ys = ((y - self.target_mean_) / self.target_scale_).astype(np.float32)
        if len(x) < 2 or float(np.max(np.abs(ys))) < 1e-6:
            self.constant_output_ = self.target_mean_.copy()
            self.state_dict_ = None
            return self
        dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
        loader = DataLoader(dataset, batch_size=min(int(self.batch_size), len(dataset)), shuffle=True)
        model = _AuxMLP(self.input_dim_, self.output_dim_, self.hidden_dims, dropout=self.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        loss_fn = nn.SmoothL1Loss(beta=0.5)
        best_loss = float('inf')
        best_state = None
        patience = max(10, int(self.epochs) // 6)
        stale = 0
        model.train()
        for _ in range(int(self.epochs)):
            epoch_loss = 0.0
            seen = 0
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                batch_n = int(xb.shape[0])
                epoch_loss += float(loss.detach().cpu()) * batch_n
                seen += batch_n
            epoch_loss = epoch_loss / max(seen, 1)
            if epoch_loss + 1e-6 < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.state_dict_ = best_state
        self.constant_output_ = None
        return self

    def _build_model(self) -> _AuxMLP:
        if self.input_dim_ is None or self.output_dim_ is None:
            raise RuntimeError('Regresor neuronal auxiliar no entrenado')
        model = _AuxMLP(self.input_dim_, self.output_dim_, self.hidden_dims, dropout=self.dropout)
        if self.state_dict_ is not None:
            model.load_state_dict(self.state_dict_)
        model.eval()
        return model

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None or self.target_mean_ is None or self.target_scale_ is None:
            raise RuntimeError('Regresor neuronal auxiliar no entrenado')
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.constant_output_ is not None:
            out = np.repeat(self.constant_output_[None, :], x.shape[0], axis=0)
            return out[:, 0] if out.shape[1] == 1 else out
        xs = ((x - self.feature_mean_) / self.feature_scale_).astype(np.float32)
        model = self._build_model()
        with torch.no_grad():
            pred = model(torch.from_numpy(xs)).cpu().numpy()
        out = pred * self.target_scale_ + self.target_mean_
        return out[:, 0] if out.shape[1] == 1 else out

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': 'tiny_neural_regressor',
            'hidden_dims': list(self.hidden_dims),
            'lr': float(self.lr),
            'weight_decay': float(self.weight_decay),
            'epochs': int(self.epochs),
            'batch_size': int(self.batch_size),
            'dropout': float(self.dropout),
            'feature_mean_': None if self.feature_mean_ is None else self.feature_mean_.tolist(),
            'feature_scale_': None if self.feature_scale_ is None else self.feature_scale_.tolist(),
            'target_mean_': None if self.target_mean_ is None else self.target_mean_.tolist(),
            'target_scale_': None if self.target_scale_ is None else self.target_scale_.tolist(),
            'input_dim_': self.input_dim_,
            'output_dim_': self.output_dim_,
            'constant_output_': None if self.constant_output_ is None else self.constant_output_.tolist(),
            'state_dict_': self.state_dict_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'TinyNeuralRegressor':
        model = cls(
            hidden_dims=tuple(int(v) for v in payload.get('hidden_dims', [32, 16])),
            lr=float(payload.get('lr', 1e-3)),
            weight_decay=float(payload.get('weight_decay', 1e-4)),
            epochs=int(payload.get('epochs', 140)),
            batch_size=int(payload.get('batch_size', 64)),
            dropout=float(payload.get('dropout', 0.0)),
        )
        if payload.get('feature_mean_') is not None:
            model.feature_mean_ = np.asarray(payload['feature_mean_'], dtype=np.float32)
        if payload.get('feature_scale_') is not None:
            model.feature_scale_ = np.asarray(payload['feature_scale_'], dtype=np.float32)
        if payload.get('target_mean_') is not None:
            model.target_mean_ = np.asarray(payload['target_mean_'], dtype=np.float32)
        if payload.get('target_scale_') is not None:
            model.target_scale_ = np.asarray(payload['target_scale_'], dtype=np.float32)
        model.input_dim_ = payload.get('input_dim_')
        model.output_dim_ = payload.get('output_dim_')
        if payload.get('constant_output_') is not None:
            model.constant_output_ = np.asarray(payload['constant_output_'], dtype=np.float32)
        state = payload.get('state_dict_')
        if state is not None:
            model.state_dict_ = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v) for k, v in state.items()}
        return model


@dataclass
class TinyNeuralClassifier:
    hidden_dims: tuple[int, ...] = (32, 16)
    lr: float = 1e-3
    weight_decay: float = 5e-4
    epochs: int = 120
    batch_size: int = 64
    dropout: float = 0.05
    feature_mean_: np.ndarray | None = None
    feature_scale_: np.ndarray | None = None
    state_dict_: dict[str, torch.Tensor] | None = None
    input_dim_: int | None = None
    constant_prob_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if x.ndim == 1:
            x = x[:, None]
        self.input_dim_ = int(x.shape[1]) if x.size else 0
        if len(x) == 0:
            self.feature_mean_ = np.zeros(self.input_dim_, dtype=np.float32)
            self.feature_scale_ = np.ones(self.input_dim_, dtype=np.float32)
            self.constant_prob_ = 0.5
            self.state_dict_ = None
            return self
        ratio = float(np.clip(y.mean(), 0.0, 1.0))
        self.feature_mean_ = x.mean(axis=0).astype(np.float32)
        self.feature_scale_ = x.std(axis=0).astype(np.float32)
        self.feature_scale_[self.feature_scale_ < 1e-6] = 1.0
        if ratio <= 1e-6 or ratio >= 1.0 - 1e-6 or len(x) < 8:
            self.constant_prob_ = ratio
            self.state_dict_ = None
            return self
        xs = ((x - self.feature_mean_) / self.feature_scale_).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(y[:, None].astype(np.float32)))
        loader = DataLoader(dataset, batch_size=min(int(self.batch_size), len(dataset)), shuffle=True)
        model = _AuxMLP(self.input_dim_, 1, self.hidden_dims, dropout=self.dropout)
        pos_weight = (1.0 - ratio) / max(ratio, 1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))
        best_loss = float('inf')
        best_state = None
        patience = max(10, int(self.epochs) // 6)
        stale = 0
        model.train()
        for _ in range(int(self.epochs)):
            epoch_loss = 0.0
            seen = 0
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                batch_n = int(xb.shape[0])
                epoch_loss += float(loss.detach().cpu()) * batch_n
                seen += batch_n
            epoch_loss = epoch_loss / max(seen, 1)
            if epoch_loss + 1e-6 < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.state_dict_ = best_state
        self.constant_prob_ = None
        return self

    def _build_model(self) -> _AuxMLP:
        if self.input_dim_ is None:
            raise RuntimeError('Clasificador neuronal auxiliar no entrenado')
        model = _AuxMLP(self.input_dim_, 1, self.hidden_dims, dropout=self.dropout)
        if self.state_dict_ is not None:
            model.load_state_dict(self.state_dict_)
        model.eval()
        return model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.constant_prob_ is not None:
            return np.full(x.shape[0], float(self.constant_prob_), dtype=np.float32)
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError('Clasificador neuronal auxiliar no entrenado')
        xs = ((x - self.feature_mean_) / self.feature_scale_).astype(np.float32)
        model = self._build_model()
        with torch.no_grad():
            logits = model(torch.from_numpy(xs)).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': 'tiny_neural_classifier',
            'hidden_dims': list(self.hidden_dims),
            'lr': float(self.lr),
            'weight_decay': float(self.weight_decay),
            'epochs': int(self.epochs),
            'batch_size': int(self.batch_size),
            'dropout': float(self.dropout),
            'feature_mean_': None if self.feature_mean_ is None else self.feature_mean_.tolist(),
            'feature_scale_': None if self.feature_scale_ is None else self.feature_scale_.tolist(),
            'input_dim_': self.input_dim_,
            'constant_prob_': self.constant_prob_,
            'state_dict_': self.state_dict_,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'TinyNeuralClassifier':
        model = cls(
            hidden_dims=tuple(int(v) for v in payload.get('hidden_dims', [32, 16])),
            lr=float(payload.get('lr', 1e-3)),
            weight_decay=float(payload.get('weight_decay', 5e-4)),
            epochs=int(payload.get('epochs', 120)),
            batch_size=int(payload.get('batch_size', 64)),
            dropout=float(payload.get('dropout', 0.05)),
        )
        if payload.get('feature_mean_') is not None:
            model.feature_mean_ = np.asarray(payload['feature_mean_'], dtype=np.float32)
        if payload.get('feature_scale_') is not None:
            model.feature_scale_ = np.asarray(payload['feature_scale_'], dtype=np.float32)
        model.input_dim_ = payload.get('input_dim_')
        model.constant_prob_ = payload.get('constant_prob_')
        state = payload.get('state_dict_')
        if state is not None:
            model.state_dict_ = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(v) for k, v in state.items()}
        return model


@dataclass
class AuxiliaryCorrector:
    task_names: list[str]
    max_count_cap: int
    max_occurrences_per_task: int
    duration_min: float
    duration_max: float
    count_regressor: TinyNeuralRegressor
    count_gate: TinyNeuralClassifier
    temporal_regressor: TinyNeuralRegressor
    time_gate: TinyNeuralClassifier
    duration_gate: TinyNeuralClassifier
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
            count_regressor=TinyNeuralRegressor.from_dict(payload.get('count_regressor', {})),
            count_gate=TinyNeuralClassifier.from_dict(payload.get('count_gate', {})),
            temporal_regressor=TinyNeuralRegressor.from_dict(payload.get('temporal_regressor', {})),
            time_gate=TinyNeuralClassifier.from_dict(payload.get('time_gate', {})),
            duration_gate=TinyNeuralClassifier.from_dict(payload.get('duration_gate', {})),
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
                use_repair=False,
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

        count_regressor = TinyNeuralRegressor(hidden_dims=(32, 16), lr=8e-4, weight_decay=1e-4, epochs=160, batch_size=64, dropout=0.0).fit(count_x, count_y)
        temporal_regressor = TinyNeuralRegressor(hidden_dims=(48, 24), lr=7e-4, weight_decay=2e-4, epochs=180, batch_size=96, dropout=0.05).fit(temporal_x, temporal_y)

        count_pred_delta = np.asarray(count_regressor.predict(count_x), dtype=np.float64).reshape(-1)
        count_gate_x, count_gate_y = _build_count_gate_dataset(
            count_x,
            count_pred_delta,
            count_targets,
            prepared.max_count_cap,
            len(prepared.task_names),
        )
        count_gate = TinyNeuralClassifier(hidden_dims=(24, 12), lr=9e-4, weight_decay=8e-4, epochs=140, batch_size=64, dropout=0.05).fit(count_gate_x, count_gate_y)

        temporal_pred = np.asarray(temporal_regressor.predict(temporal_x), dtype=np.float64)
        time_gate_x, time_gate_y, duration_gate_x, duration_gate_y = _build_temporal_gate_dataset(
            temporal_x,
            temporal_pred,
            temporal_truth_rows,
        )
        time_gate = TinyNeuralClassifier(hidden_dims=(32, 16), lr=9e-4, weight_decay=8e-4, epochs=150, batch_size=96, dropout=0.05).fit(time_gate_x, time_gate_y)
        duration_gate = TinyNeuralClassifier(hidden_dims=(32, 16), lr=9e-4, weight_decay=8e-4, epochs=150, batch_size=96, dropout=0.05).fit(duration_gate_x, duration_gate_y)

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
                'auxiliary_backend': 'tiny_neural_mlp',
                'replay_stats': replay_stats[-5:],
            },
        )

    def apply(
        self,
        prepared,
        target_week_idx: int,
        base_predictions: list[dict[str, Any]],
        *,
        correct_count: bool = True,
        correct_time: bool = False,
        correct_duration: bool = True,
        apply_repair: bool = False,
    ) -> list[dict[str, Any]]:
        task_groups = _group_predictions_by_task(base_predictions, prepared.task_to_id)
        if correct_count:
            adjusted = self._apply_count_corrections(prepared, target_week_idx, task_groups)
        else:
            adjusted = _flatten_task_groups(task_groups)
        adjusted = self._apply_temporal_corrections(
            prepared,
            target_week_idx,
            adjusted,
            correct_time=correct_time,
            correct_duration=correct_duration,
        )
        adjusted = sorted(adjusted, key=lambda x: (int(x['start_bin']), str(x['task_name'])))
        return _repair_final_predictions(adjusted) if apply_repair else adjusted

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

    def _apply_temporal_corrections(
        self,
        prepared,
        target_week_idx: int,
        predictions: list[dict[str, Any]],
        *,
        correct_time: bool = False,
        correct_duration: bool = True,
    ) -> list[dict[str, Any]]:
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
                if correct_time and time_conf >= self.time_conf_threshold and (abs(raw_day) > 0 or abs(raw_local) >= self.time_min_abs_delta_bins):
                    delta_bins = raw_day * bins_per_day() + raw_local
                    new_item['start_bin'] = int(np.clip(int(item['start_bin']) + delta_bins, 0, num_time_bins() - 1))
                if correct_duration and duration_conf >= self.duration_conf_threshold and abs(raw_duration) >= self.duration_min_abs_delta_minutes:
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



def _flatten_task_groups(task_groups: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for group in task_groups.values():
        for item in group:
            items.append(dict(item))
    return items

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


def _day_bounds(day_idx: int, duration_bins: int) -> tuple[int, int]:
    day_start = max(0, int(day_idx) * bins_per_day())
    day_end = min(num_time_bins() - duration_bins, day_start + bins_per_day() - duration_bins)
    return day_start, max(day_start, day_end)


def _candidate_cost(candidate_start: int, preferred_start: int) -> tuple[int, int, int]:
    pref_day = int(preferred_start // bins_per_day())
    cand_day = int(candidate_start // bins_per_day())
    pref_tod = int(preferred_start % bins_per_day())
    cand_tod = int(candidate_start % bins_per_day())
    day_diff = abs(cand_day - pref_day)
    tod_diff = abs(cand_tod - pref_tod)
    total_diff = abs(candidate_start - preferred_start)
    return (
        day_diff * int(PREDICTION_REPAIR_DAY_CHANGE_PENALTY) + tod_diff,
        day_diff,
        total_diff,
    )


def _search_nearest_within_bounds(preferred_start: int, duration_bins: int, placed_intervals: list[tuple[int, int]], lower_bound: int, min_start: int, max_start: int) -> int | None:
    min_start = max(int(min_start), int(lower_bound), 0)
    max_start = min(int(max_start), num_time_bins() - duration_bins)
    if min_start > max_start:
        return None
    preferred = int(np.clip(preferred_start, min_start, max_start))
    if not _overlaps(preferred, duration_bins, placed_intervals):
        return preferred
    max_radius = min(PREDICTION_REPAIR_RADIUS_BINS, max_start - min_start)
    for radius in range(1, max_radius + 1):
        left = preferred - radius
        right = preferred + radius
        if left >= min_start and not _overlaps(left, duration_bins, placed_intervals):
            return left
        if right <= max_start and not _overlaps(right, duration_bins, placed_intervals):
            return right
    return None


def _same_day_or_adjacent_valid_start(preferred_start: int, duration_bins: int, placed_intervals: list[tuple[int, int]], lower_bound: int = 0) -> int | None:
    preferred_start = int(np.clip(preferred_start, lower_bound, num_time_bins() - duration_bins))
    pref_day = int(preferred_start // bins_per_day())
    for day_delta in range(0, int(PREDICTION_REPAIR_MAX_DAY_SHIFT) + 1):
        candidate_days = [pref_day] if day_delta == 0 else [pref_day - day_delta, pref_day + day_delta]
        day_candidates: list[tuple[tuple[int, int, int], int]] = []
        for day_idx in candidate_days:
            if day_idx < 0 or day_idx >= 7:
                continue
            day_min, day_max = _day_bounds(day_idx, duration_bins)
            preferred_tod = preferred_start % bins_per_day()
            anchor = day_idx * bins_per_day() + preferred_tod
            candidate = _search_nearest_within_bounds(anchor, duration_bins, placed_intervals, lower_bound, day_min, day_max)
            if candidate is not None:
                day_candidates.append((_candidate_cost(candidate, preferred_start), candidate))
        if day_candidates:
            day_candidates.sort(key=lambda x: x[0])
            return int(day_candidates[0][1])
    return None


def _best_global_valid_start(preferred_start: int, duration_bins: int, placed_intervals: list[tuple[int, int]], lower_bound: int = 0) -> int:
    preferred_start = int(np.clip(preferred_start, lower_bound, num_time_bins() - duration_bins))
    best_start: int | None = None
    best_cost: tuple[int, int, int] | None = None
    for candidate in range(max(0, lower_bound), num_time_bins() - duration_bins + 1):
        if _overlaps(candidate, duration_bins, placed_intervals):
            continue
        cost = _candidate_cost(candidate, preferred_start)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_start = candidate
    if best_start is not None:
        return int(best_start)
    return preferred_start


def _repair_final_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    placed: list[tuple[int, int]] = []
    last_start_by_task: dict[int, int] = {}
    final_predictions: list[dict[str, Any]] = []
    ordered = sorted(predictions, key=lambda x: (int(x['start_bin']), int(x.get('task_id', 0)), x.get('task_name', '')))
    for item in ordered:
        duration_bins = max(1, int(round(float(item['duration']) / BIN_MINUTES)))
        task_id = int(item.get('task_id', 0))
        preferred_start = int(item['start_bin'])
        lower_bound = min(last_start_by_task.get(task_id, 0), preferred_start)
        chosen_start = _same_day_or_adjacent_valid_start(preferred_start, duration_bins, placed, lower_bound=lower_bound)
        if chosen_start is None and PREDICTION_REPAIR_GLOBAL_FALLBACK:
            chosen_start = _best_global_valid_start(preferred_start, duration_bins, placed, lower_bound=lower_bound)
        elif chosen_start is None:
            chosen_start = int(np.clip(preferred_start, lower_bound, num_time_bins() - duration_bins))
        placed.append((chosen_start, chosen_start + duration_bins))
        placed.sort()
        last_start_by_task[task_id] = chosen_start
        final_predictions.append({
            'task_name': item['task_name'],
            'type': item.get('type', item['task_name']),
            'start_bin': int(chosen_start),
            'duration': float(max(BIN_MINUTES, item['duration'])),
            'repair_preferred_start_bin': int(preferred_start),
            'repair_day_shift': int(chosen_start // bins_per_day()) - int(preferred_start // bins_per_day()),
            'repair_displacement_bins': int(chosen_start - preferred_start),
        })
    return sorted(final_predictions, key=lambda x: (x['start_bin'], x['task_name']))


def maybe_load_auxiliary_policy(path: str | None = None) -> AuxiliaryUsagePolicy | None:
    try:
        return AuxiliaryUsagePolicy.load(path or AUXILIARY_POLICY_PATH)
    except Exception:
        return None
