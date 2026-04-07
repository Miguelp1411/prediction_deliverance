# Cambios aplicados

## 1. Métrica objetivo temporal
- Se eliminó la penalización por `overlap_same_device_count` de:
  - `e2e_joint_score`
  - `e2e_compound_score`
- El `compound_score` ahora monitoriza únicamente:
  - `time_close_accuracy_5m`
  - `start_mae_minutes`

## 2. Evaluación de overlaps
- `overlap_same_device_count` ya no agrupa eventos con `device_uid=None` dentro de un bucket ficticio.
- Los eventos sin device quedan fuera del cómputo `same_device`.
- Se conserva `unknown_device_count` como métrica diagnóstica separada.

## 3. Reformulación temporal relativa al ancla
- El modelo temporal dejó de predecir:
  - `day_logits`
  - `time_logits`
- Ahora predice:
  - `day_offset_logits`
  - `local_offset_logits`
- La decodificación del `start_bin` se hace relativa a `anchor_start_bin`.

## 4. Reducción de label switching
- Se introdujo una asignación por slots/prototipos históricos por tarea.
- El dataset temporal y la inferencia ya no usan el orden bruto por `start_bin` como identidad principal de ocurrencia.
- La selección de slots para inferencia utiliza prototipos aprendidos del histórico.

## 5. Compatibilidad de checkpoints
- `FEATURE_SCHEMA_VERSION` se incrementó a `5`.
- Los checkpoints antiguos quedan marcados como incompatibles y requieren reentrenamiento.

## Validaciones ejecutadas
- `python -m compileall`
- Smoke tests de:
  - `prepare_data(...)`
  - `TemporalDataset`
  - forward del `TemporalAssignmentModel`
  - `predict_next_week(...)`
  - `aggregate_weekly_stats(...)`
