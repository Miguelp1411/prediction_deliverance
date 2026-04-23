# Patch notes: Occurrence ordinal + residual seasonal baseline

Cambios aplicados con impacto mínimo en la interfaz externa:

- `training/losses.py`
  - `OccurrenceLoss` ahora usa `cross_entropy + lambda * MAE(expected_count, target_count)`.
- `models/occurrence_model.py`
  - `TaskOccurrenceModel` ya no predice conteo absoluto de forma directa.
  - Ahora construye logits absolutos como:
    - baseline estacional por tarea, basado en lags `(4, 26, 52)` con pesos configurables
    - más logits residuales sobre deltas respecto al baseline
- `config.py`
  - nuevos hiperparámetros:
    - `OCC_SEASONAL_LAG_WEIGHTS = (0.10, 0.25, 0.65)`
    - `OCC_EXPECTED_COUNT_MAE_WEIGHT = 0.35`
    - `OCC_SEASONAL_BASELINE_LOGIT = 2.50`
  - `FEATURE_SCHEMA_VERSION = 6` para invalidar checkpoints antiguos incompatibles
- `train.py`
  - guarda en checkpoint los nuevos hiperparámetros del occurrence model
  - instancia `TaskOccurrenceModel` con pesos y fuerza del baseline estacional
- `predict.py`
  - carga los nuevos hiperparámetros desde checkpoint si existen

Compatibilidad:
- la salida pública del occurrence model sigue siendo `[batch, num_tasks, max_count_cap + 1]`
- datasets, métricas, evaluación e inferencia no requieren cambios adicionales
- los checkpoints anteriores al parche ya no son compatibles y hay que reentrenar
