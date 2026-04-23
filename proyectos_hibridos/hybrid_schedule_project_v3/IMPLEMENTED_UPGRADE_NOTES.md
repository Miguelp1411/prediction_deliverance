# Hybrid Schedule Project v2

Esta versión integra las fases pedidas sobre la arquitectura híbrida original.

## Fase 1
- Alineación train/inference en temporal usando `count_lookup` predicho por el modelo de ocurrencias.
- Temporal relativo al anchor con `day_offset` y `local_offset`.
- Slots estables por prototipo con `task_slot_prototypes` y `assign_events_to_prototypes`.

## Fase 2
- Features más ricas en `data/features.py` para occurrence y temporal.
- Baseline estacional mezclado con retrieval + lags 4/26/52 + mediana reciente.
- `expected_count_mae` añadido a la loss de ocurrencias.

## Fase 3
- Selección del mejor checkpoint por `selection_score` orientado a resultado final, no solo por `val_loss`.
- Label smoothing en temporal.
- Blend de duración con mediana histórica por tarea.
- Temporal simplificado a menos capas y menor dropout.

## Cambios principales
- `src/hybrid_schedule/data/features.py`
- `src/hybrid_schedule/retrieval/template_retriever.py`
- `src/hybrid_schedule/training/datasets.py`
- `src/hybrid_schedule/training/losses.py`
- `src/hybrid_schedule/training/loops.py`
- `src/hybrid_schedule/models/occurrence_residual.py`
- `src/hybrid_schedule/models/temporal_residual.py`
- `src/hybrid_schedule/inference/predictor.py`
- `scripts/train.py`
- `scripts/predict_week.py`
- `configs/default.yaml`
