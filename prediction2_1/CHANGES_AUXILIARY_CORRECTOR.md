# Cambios: integración del corrector auxiliar residual

## Archivos nuevos

- `auxiliary_corrector.py`
- `evaluate_auxiliary.py`
- `requirements.txt`
- `CHANGES_AUXILIARY_CORRECTOR.md`

## Archivos modificados

- `predict.py`
- `train.py`
- `README.md`

## Resumen funcional

### `auxiliary_corrector.py`
Añade una capa residual ligera que aprende a partir de semanas ya cerradas:

- ajuste de conteo por tarea,
- ajuste temporal del inicio,
- ajuste de duración,
- compuertas de confianza para decidir cuándo corregir.

### `predict.py`
Ahora:

- acepta un corrector auxiliar opcional en `predict_next_week(...)`,
- carga automáticamente `checkpoints/auxiliary_corrector.pt` si existe,
- permite desactivarlo con `--disable-auxiliary-corrector`,
- aplica `repair` y luego la corrección auxiliar antes de materializar la salida.

### `train.py`
Ahora:

- acepta `--data` para entrenar con distintos históricos,
- puede omitir el corrector con `--skip-auxiliary-corrector`,
- entrena el corrector auxiliar después de los modelos principales,
- resume en validación `base vs auxiliar`,
- guarda `checkpoints/auxiliary_corrector.pt`.

### `evaluate_auxiliary.py`
Nuevo script para probar el valor real del corrector sobre uno o varios JSON:

- evalúa el modelo base,
- evalúa el modelo base + corrector auxiliar,
- compara métricas agregadas,
- puede reentrenar el corrector con cada dataset usando `--retrain-auxiliary`.

## Métricas usadas

- `task_accuracy`
- `time_exact_accuracy`
- `time_close_accuracy_5m`
- `start_mae_minutes`
- `duration_close_accuracy`

## Checkpoints esperados

- `checkpoints/occurrence_model.pt`
- `checkpoints/temporal_model.pt`
- `checkpoints/auxiliary_corrector.pt`
