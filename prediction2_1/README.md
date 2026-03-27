# Predicción semanal por ocurrencias + asignación temporal + corrector auxiliar

Este proyecto ahora tiene tres capas:

1. **OccurrenceModel**: predice cuántas veces aparece cada tarea la semana siguiente.
2. **TemporalAssignmentModel**: para cada ocurrencia predicha, asigna día, hora y duración.
3. **AuxiliaryCorrector**: aprende errores residuales históricos del modelo principal y compensa sesgos de conteo, inicio y duración cuando tiene confianza suficiente.

## Qué añade el corrector auxiliar

El corrector auxiliar es muy pequeño y CPU-friendly. Aprende sobre semanas ya cerradas:

- **errores de conteo**: cuándo el modelo principal sobrepredice o infrapredice,
- **errores temporales**: cuándo suele desplazar una tarea algunos bins o incluso un día,
- **errores de duración**: cuándo suele inflar o recortar duraciones,
- **errores de confianza**: cuándo conviene corregir y cuándo es mejor no tocar la salida original.

No sustituye a los modelos principales. Solo aplica una compensación residual sobre sus predicciones.

## Estructura

- `train.py`: entrena los dos modelos principales y, salvo que se desactive, también el corrector auxiliar.
- `predict.py`: genera la próxima semana y aplica automáticamente el corrector auxiliar si encuentra su checkpoint.
- `evaluate_auxiliary.py`: compara base vs corrector auxiliar sobre uno o varios históricos.
- `auxiliary_corrector.py`: implementación del corrector residual ligero.
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

También se acepta `type` en lugar de `task_name`.

## Dependencias

```bash
pip install -r requirements.txt
```

## Entrenar

```bash
python train.py --data robot_schedule_v4_2025_2026.json
```

Para entrenar solo el modelo principal:

```bash
python train.py --data robot_schedule_v4_2025_2026.json --skip-auxiliary-corrector
```

Al terminar, se guardan:

- `checkpoints/occurrence_model.pt`
- `checkpoints/temporal_model.pt`
- `checkpoints/auxiliary_corrector.pt` (si no se desactiva)

## Predecir la siguiente semana

```bash
python predict.py --data robot_schedule_v4_2025_2026.json --output predicted_next_week.json
```

Por defecto intentará cargar `checkpoints/auxiliary_corrector.pt` y aplicarlo.

Para desactivarlo:

```bash
python predict.py --data robot_schedule_v4_2025_2026.json --disable-auxiliary-corrector
```

## Evaluar si el corrector auxiliar aporta valor

Comparación base vs auxiliar sobre un histórico:

```bash
python evaluate_auxiliary.py robot_schedule_v4_2025_2026.json --retrain-auxiliary
```

Comparación sobre varios históricos:

```bash
python evaluate_auxiliary.py robot_schedule_v4_2025_2026.json robot_schedule_gamma.json nexus_schedule_5years.json --retrain-auxiliary --output reports/aux_eval.json
```

Métricas incluidas:

- `task_accuracy`
- `time_exact_accuracy`
- `time_close_accuracy_5m`
- `start_mae_minutes`
- `duration_close_accuracy`

## Flujo recomendado de pruebas

1. Entrena con un histórico concreto usando `train.py`.
2. Evalúa `base vs auxiliar` con `evaluate_auxiliary.py`.
3. Repite con tus distintos JSON históricos.
4. Solo pasa a producción si el corrector mejora de forma consistente o, al menos, no degrada las métricas clave.
