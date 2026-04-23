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
