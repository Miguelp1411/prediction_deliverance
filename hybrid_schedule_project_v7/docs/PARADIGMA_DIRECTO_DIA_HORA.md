# Paradigma directo día + hora para `hybrid_schedule_project_v5`

## Qué cambia

La rama temporal deja de ser un **ranker de candidatos históricos** y pasa a ser un **predictor directo** de:

- día de la semana,
- hora dentro del día,
- duración.

El solver se mantiene, pero ahora solo como capa de **consistencia global** para evitar solapes y escoger entre las combinaciones más probables del modelo.

## Nueva arquitectura

Se ha añadido `TemporalDirectNet` con estas piezas:

1. **Encoder secuencial bidireccional (GRU)** para capturar la evolución semanal.
2. **Encoder Transformer sobre semanas** con embeddings posicionales para captar patrones de largo alcance.
3. **Embeddings estructurales** para:
   - tarea,
   - base de datos,
   - robot,
   - slot ordinal,
   - día ancla,
   - hora ancla.
4. **Torre numérica profunda** para las features temporales del slot.
5. **Bloques residuales de fusión** antes de las cabezas de salida.
6. **Tres cabezas de salida**:
   - `day_logits`
   - `time_logits`
   - `pred_log_duration`

## Dataset temporal nuevo

Se ha añadido `DirectTemporalDataset`, que para cada evento objetivo entrena sobre:

- historia de semanas,
- identidad de tarea/robot/base,
- slot ordinal,
- ancla temporal del slot,
- features numéricas del slot,
- targets directos:
  - `target_day_idx`
  - `target_time_bin_idx`
  - `target_log_duration`

## Inference

En predicción:

1. `occurrence` sigue estimando cuántas veces aparece cada tarea.
2. Se construyen los `planned_slots`.
3. El nuevo temporal directo predice distribuciones de día y hora para cada slot.
4. Se generan candidatos combinando el `top-k` de días y horas.
5. El solver resuelve conflictos y entrega un calendario sin solapes.

## Ficheros principales tocados

- `src/hybrid_schedule/models/temporal_direct.py`
- `src/hybrid_schedule/training/datasets.py`
- `src/hybrid_schedule/training/losses.py`
- `src/hybrid_schedule/evaluation/metrics.py`
- `src/hybrid_schedule/training/loops.py`
- `src/hybrid_schedule/inference/predictor.py`
- `scripts/train.py`
- `scripts/predict_week.py`
- `configs/default.yaml`

## Configuración por defecto

El `default.yaml` ahora deja la arquitectura temporal en:

```yaml
models:
  temporal:
    architecture: direct_day_time
```

## Estado de validación

Se ha validado:

- compilación de todo el proyecto,
- construcción de datasets,
- forward pass del nuevo modelo,
- cálculo de loss y métricas,
- generación de predicción con el nuevo predictor.

No se ha dejado un entrenamiento largo completado dentro del paquete; hay que reentrenar para obtener checkpoints reales con esta arquitectura.
