# Cambios aplicados en prediction2_v2

## Objetivo
Alinear el entrenamiento con la inferencia real y reducir la caída de rendimiento al encadenar ambos modelos.

## Cambios principales

### 1. Predictor de ocurrencias estructurado
Se ha sustituido el entrenamiento del modelo de ocurrencias por un predictor `structured_lag4`.

- Archivo: `models/occurrence_model.py`
- Configuración: `config.py`
- Entrenamiento/guardado: `train.py`

Qué hace:
- usa los conteos de la misma fase hace 4 semanas;
- usa medianas por tarea solo como fallback;
- sigue devolviendo logits para no romper el pipeline existente.

Por qué ayuda:
- el dataset tiene una periodicidad de 4 semanas muy fuerte;
- evita el colapso a una semana promedio;
- mejora la estabilidad del primer bloque, que era el principal origen de error acumulado.

### 2. Entrenamiento del temporal con conteos predichos reales
El `TemporalDataset` ya no condiciona por defecto con el conteo real del target.

- Archivo: `data/datasets.py`

Qué cambia:
- se construye un `count_lookup` usando el predictor de ocurrencias;
- cada muestra temporal recibe el conteo que vería realmente en inferencia;
- `target_count_norm` se conserva solo para monitorización.

Por qué ayuda:
- elimina el desajuste train/inference;
- evita validar el temporal con una señal privilegiada que no existe en producción.

### 3. Nueva feature de posición relativa de la ocurrencia
Se añade `occurrence_progress`.

- Archivos: `data/datasets.py`, `models/temporal_model.py`, `training/engine.py`, `predict.py`

Qué representa:
- posición relativa de la ocurrencia dentro del total predicho de esa tarea en la semana.

Por qué ayuda:
- hace más estable la representación en tareas densas como `Assist`;
- reduce la dependencia de un índice absoluto frágil (`occurrence_index`).

### 4. Monitor end-to-end semanal para seleccionar el temporal
Se añade evaluación semanal agregada durante el entrenamiento del temporal.

- Archivos: `train.py`, `training/engine.py`, `evaluation/weekly_stats.py`

Qué mide:
- `e2e_task_acc`
- `e2e_start_exact_acc`
- `e2e_start_tol_acc_5m`
- `e2e_overlap_count`
- `e2e_joint_score`

Por qué ayuda:
- el checkpoint se selecciona por comportamiento real del pipeline completo, no solo por una métrica local del segundo modelo.

### 5. Métricas semanales por tarea y overlaps
Se amplía la evaluación semanal.

- Archivo: `evaluation/weekly_stats.py`

Qué añade:
- `overlap_count`
- métricas por tarea (`task_accuracy`, `time_exact_accuracy`)

Por qué ayuda:
- hace visible si el sistema mejora de verdad en las tareas difíciles y no solo en el promedio.

### 6. Repair desactivado por defecto
- Configuración: `PREDICTION_USE_REPAIR = False`
- Archivo: `predict.py`

Por qué ayuda:
- en el proyecto original el repair podía introducir desplazamientos grandes;
- ahora el flujo por defecto prioriza predicción directa sin post-procesado agresivo.

## Compatibilidad
La versión incrementa `FEATURE_SCHEMA_VERSION` a 3. Los checkpoints antiguos no son compatibles y deben regenerarse con `train.py`.

## Resultado esperado
1. Mejor estabilidad en el modelo de ocurrencias.
2. Menor diferencia entre validación e inferencia real.
3. Mejor comportamiento semanal del pipeline completo.
4. Diagnóstico más claro por tarea y por tipo de error.
