from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    optuna = None  # type: ignore
    HAS_OPTUNA = False

import predict as predict_module
from config import (
    CAP_INFERENCE_SCOPE,
    CHECKPOINT_DIR,
    DATA_PATH,
    DEVICE,
    OPTUNA_N_TRIALS,
    OPTUNA_OBJECTIVE_MODE,
    OPTUNA_STAGE_CANDIDATES,
    OPTUNA_STUDY_NAME,
    OPTUNA_TIMEOUT_SECONDS,
    POSTPROCESS_OVERRIDE_PATH,
    SEED,
    TIMEZONE,
    TRAIN_RATIO,
)
from data.datasets import build_split_indices
from data.io import load_tasks_dataframe
from data.preprocessing import prepare_data
from predict import _extract_preprocessing_caps, _load_models, load_runtime_overrides
from train import (
    aggregate_weekly_ablation_stats,
    configure_runtime,
    select_best_validation_stage,
    set_seed,
    stage_objective_from_metrics,
)
from utils.runtime import resolve_device
from utils.serialization import load_checkpoint


TUNABLE_KEYS = (
    'TEMPORAL_ANCHOR_PROXIMITY_WEIGHT',
    'TEMPORAL_ANCHOR_MAX_SHIFT_BINS',
    'TEMPORAL_NUM_ANCHOR_CANDIDATES',
    'TEMPORAL_RERANK_BEAM_WIDTH',
    'TEMPORAL_RERANK_MAX_CANDIDATES',
    'TEMPORAL_RERANK_OVERLAP_PENALTY',
    'TEMPORAL_RERANK_ORDER_PENALTY',
    'TEMPORAL_RERANK_MOVE_PENALTY',
    'TEMPORAL_GATING_HIGH_CONFIDENCE_MARGIN',
    'TEMPORAL_GATING_LOW_CONFIDENCE_MARGIN',
    'TEMPORAL_GATING_MIN_ANCHOR_AGREEMENT',
    'TEMPORAL_GATING_ANCHOR_CLOSE_BINS',
    'TEMPORAL_GATING_MAX_STAGE_MOVE_RATE',
)


class RandomTrial:
    def __init__(self, number: int, rng: random.Random):
        self.number = int(number)
        self._rng = rng
        self.params: dict[str, object] = {}
        self.user_attrs: dict[str, object] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        value = int(self._rng.randint(int(low), int(high)))
        self.params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float) -> float:
        value = float(self._rng.uniform(float(low), float(high)))
        self.params[name] = value
        return value

    def set_user_attr(self, name: str, value) -> None:
        self.user_attrs[name] = value


def sample_params(trial) -> dict[str, object]:
    beam_width = trial.suggest_int('TEMPORAL_RERANK_BEAM_WIDTH', 4, 12)
    max_candidates = trial.suggest_int('TEMPORAL_RERANK_MAX_CANDIDATES', 4, 10)
    num_anchor_candidates = trial.suggest_int('TEMPORAL_NUM_ANCHOR_CANDIDATES', 3, 8)
    return {
        'TEMPORAL_ANCHOR_PROXIMITY_WEIGHT': trial.suggest_float('TEMPORAL_ANCHOR_PROXIMITY_WEIGHT', 0.0, 0.12),
        'TEMPORAL_ANCHOR_MAX_SHIFT_BINS': trial.suggest_int('TEMPORAL_ANCHOR_MAX_SHIFT_BINS', 6, 36),
        'TEMPORAL_NUM_ANCHOR_CANDIDATES': min(num_anchor_candidates, max_candidates),
        'TEMPORAL_RERANK_BEAM_WIDTH': beam_width,
        'TEMPORAL_RERANK_MAX_CANDIDATES': max_candidates,
        'TEMPORAL_RERANK_OVERLAP_PENALTY': trial.suggest_float('TEMPORAL_RERANK_OVERLAP_PENALTY', 2.0, 16.0),
        'TEMPORAL_RERANK_ORDER_PENALTY': trial.suggest_float('TEMPORAL_RERANK_ORDER_PENALTY', 0.0, 2.5),
        'TEMPORAL_RERANK_MOVE_PENALTY': trial.suggest_float('TEMPORAL_RERANK_MOVE_PENALTY', 0.5, 6.0),
        'TEMPORAL_GATING_ENABLE': True,
        'TEMPORAL_GATING_PREFER_RAW_WHEN_CLEAN': True,
        'TEMPORAL_GATING_HIGH_CONFIDENCE_MARGIN': trial.suggest_float('TEMPORAL_GATING_HIGH_CONFIDENCE_MARGIN', 0.20, 1.40),
        'TEMPORAL_GATING_LOW_CONFIDENCE_MARGIN': trial.suggest_float('TEMPORAL_GATING_LOW_CONFIDENCE_MARGIN', 0.02, 0.40),
        'TEMPORAL_GATING_MIN_ANCHOR_AGREEMENT': trial.suggest_float('TEMPORAL_GATING_MIN_ANCHOR_AGREEMENT', 0.30, 0.90),
        'TEMPORAL_GATING_ANCHOR_CLOSE_BINS': trial.suggest_int('TEMPORAL_GATING_ANCHOR_CLOSE_BINS', 3, 24),
        'TEMPORAL_GATING_MAX_STAGE_MOVE_RATE': trial.suggest_float('TEMPORAL_GATING_MAX_STAGE_MOVE_RATE', 0.05, 0.60),
    }


def snapshot_predict_globals() -> dict[str, object]:
    return {key: getattr(predict_module, key) for key in TUNABLE_KEYS if hasattr(predict_module, key)}


def restore_predict_globals(snapshot: dict[str, object]) -> None:
    for key, value in snapshot.items():
        setattr(predict_module, key, value)


def build_objective(prepared, split, occurrence_model, temporal_model, device, objective_mode: str):
    def objective(trial) -> float:
        params = sample_params(trial)
        original = snapshot_predict_globals()
        try:
            predict_module.apply_runtime_overrides(params)
            ablation = aggregate_weekly_ablation_stats(
                prepared,
                split.val_target_week_indices,
                occurrence_model,
                temporal_model,
                device,
                include_repair=False,
                include_gated=True,
            )
            selection = select_best_validation_stage(ablation)
            gated_metrics = dict(ablation.get('stages', {}).get('gated', {}))
            gated_score = stage_objective_from_metrics(gated_metrics) if gated_metrics else float('-inf')
            selected_stage = str(selection.get('selected_stage', 'raw'))
            selected_score = float(selection.get('selected_stage_score', float('-inf')))
            trial.set_user_attr('selected_stage', selected_stage)
            trial.set_user_attr('selected_stage_score', selected_score)
            trial.set_user_attr('stage_scores', selection.get('stage_scores', {}))
            return selected_score if objective_mode == 'validation_best' else gated_score
        finally:
            restore_predict_globals(original)
    return objective


def run_search(objective, n_trials: int, timeout: int | None):
    if HAS_OPTUNA:
        sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
        study = optuna.create_study(direction='maximize', study_name=OPTUNA_STUDY_NAME, sampler=sampler)
        study.optimize(objective, n_trials=max(1, int(n_trials)), timeout=timeout)
        return {
            'engine': 'optuna',
            'best_value': float(study.best_value),
            'best_trial_number': int(study.best_trial.number),
            'best_params': dict(study.best_trial.params),
            'best_user_attrs': dict(study.best_trial.user_attrs),
        }

    rng = random.Random(SEED)
    best_trial = None
    best_value = float('-inf')
    for trial_number in range(max(1, int(n_trials))):
        trial = RandomTrial(trial_number, rng)
        value = float(objective(trial))
        if best_trial is None or value > best_value:
            best_trial = trial
            best_value = value
    assert best_trial is not None
    return {
        'engine': 'random_search_fallback',
        'best_value': float(best_value),
        'best_trial_number': int(best_trial.number),
        'best_params': dict(best_trial.params),
        'best_user_attrs': dict(best_trial.user_attrs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='v4.7: tuning de postproceso, gating y selección automática de etapa final.')
    parser.add_argument('--data', type=str, default=str(DATA_PATH), help='Ruta al JSON de datos históricos.')
    parser.add_argument('--device', type=str, default=None, help='Dispositivo: auto/cpu/cuda/...')
    parser.add_argument('--trials', type=int, default=OPTUNA_N_TRIALS, help='Número de trials.')
    parser.add_argument('--timeout', type=int, default=OPTUNA_TIMEOUT_SECONDS, help='Timeout total en segundos.')
    parser.add_argument('--objective-mode', type=str, default=OPTUNA_OBJECTIVE_MODE, choices=('gated', 'validation_best'))
    parser.add_argument('--output', type=str, default=str(POSTPROCESS_OVERRIDE_PATH), help='JSON de salida con los mejores parámetros.')
    parser.add_argument('--warm-start-overrides', type=str, default=None, help='JSON opcional con overrides iniciales.')
    args = parser.parse_args()

    set_seed(SEED)
    device = resolve_device(args.device or DEVICE)
    configure_runtime(device)

    if args.warm_start_overrides:
        applied = load_runtime_overrides(args.warm_start_overrides)
        if applied:
            print(f'Warm start aplicado: {sorted(applied.keys())}')

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
    occurrence_model.eval()
    temporal_model.eval()

    search_result = run_search(
        build_objective(prepared, split, occurrence_model, temporal_model, device, args.objective_mode),
        n_trials=args.trials,
        timeout=args.timeout,
    )

    best_params = dict(search_result['best_params'])
    best_params['TEMPORAL_NUM_ANCHOR_CANDIDATES'] = min(
        int(best_params.get('TEMPORAL_NUM_ANCHOR_CANDIDATES', 6)),
        int(best_params.get('TEMPORAL_RERANK_MAX_CANDIDATES', 6)),
    )
    best_params['TEMPORAL_GATING_ENABLE'] = True
    best_params['TEMPORAL_GATING_PREFER_RAW_WHEN_CLEAN'] = True

    user_attrs = dict(search_result.get('best_user_attrs', {}))
    result = {
        'study_name': OPTUNA_STUDY_NAME,
        'search_engine': search_result['engine'],
        'objective_mode': args.objective_mode,
        'best_value': float(search_result['best_value']),
        'best_params': best_params,
        'best_trial_number': int(search_result['best_trial_number']),
        'best_selected_stage': user_attrs.get('selected_stage'),
        'best_selected_stage_score': user_attrs.get('selected_stage_score'),
        'best_stage_scores': user_attrs.get('stage_scores', {}),
        'candidate_stages': list(OPTUNA_STAGE_CANDIDATES),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Búsqueda completada con '{search_result['engine']}'. Mejor valor: {search_result['best_value']:.6f}")
    print(f"Mejor etapa seleccionada en validación: {result.get('best_selected_stage')}")
    print(f'Overrides guardados en: {output_path}')


if __name__ == '__main__':
    main()
