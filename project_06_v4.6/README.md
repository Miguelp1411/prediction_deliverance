# Project 06 v3

Esta versión regenera el proyecto con foco en **single-device / multi-device**, visibilidad del `OccurrenceModel` y evaluación más interpretable.

## Novedades

- `device_uid` se conserva desde el dataset cuando existe.
- Los overlaps se reportan de dos formas:
  - `overlap_global_count`
  - `overlap_same_device_count`
- El `OccurrenceModel` ahora expone también:
  - `presence_precision`
  - `presence_recall`
  - `presence_f1`
- Se habilita `PREDICTION_USE_REPAIR=True` por defecto.
- Nuevo script de análisis previo: `python scripts/analyze_dataset.py --data nexus_schedule_5years.json`

## Recomendación de uso

1. Analiza primero el dataset.
2. Entrena unas pocas epochs para validar shapes y reporting.
3. Revisa los overlaps por `device_uid` antes de interpretar la métrica conjunta.

# Predicción semanal por ocurrencias + asignación temporal

Este proyecto replantea el forecasting en dos etapas:

1. **Modelo de ocurrencias**: predice cuántas veces aparece cada tarea la semana siguiente.
2. **Modelo temporal**: para cada ocurrencia predicha, asigna un `start_time` y una duración.

## Estructura

- `train.py`: entrena ambos modelos y guarda checkpoints.
- `predict.py`: genera la próxima semana completa a partir del histórico.
- `data/`: carga, preprocesado y datasets.
- `models/`: modelos PyTorch.
- `training/`: losses, métricas y engine.
- `checkpoints/`: salida de entrenamiento.

## Formato del JSON

Cada entrada debe incluir al menos:

```json
{
  "task_name": "Limpiar mesa comedor",
  "start_time": "2025-12-31T13:40:00.000Z",
  "end_time": "2025-12-31T13:44:00.000Z"
}
```

## Entrenar

```bash
cd project
python train.py
```

## Predecir la siguiente semana

```bash
cd project
python predict.py
```

El resultado se guarda en `checkpoints/predicted_next_week.json`.
