# Análisis y optimización aplicada

## Resumen ejecutivo

Se revisó el proyecto para detectar:

- código muerto o no referenciado;
- sobrecostes innecesarios en entrenamiento;
- artefactos generados que no deberían formar parte del código fuente;
- fallos funcionales en scripts auxiliares.

## Hallazgos principales

### 1) Evaluación end-to-end demasiado frecuente en entrenamiento

En `train.py` el modelo temporal ejecutaba el evaluador extra con:

- `extra_val_evaluator_every=1`

aunque ya existía la constante de configuración `TMP_E2E_EVAL_EVERY` en `config.py`.

Ese evaluador llama a `aggregate_weekly_stats(...)`, que a su vez ejecuta predicción completa sobre semanas de validación. En una prueba local de CPU, evaluar solo 3 semanas tardó ~19 segundos, por lo que hacerlo en cada epoch penaliza mucho el tiempo total.

### 2) Script documentado en README no era ejecutable tal cual

El comando:

```bash
python scripts/analyze_dataset.py --data nexus_schedule_5years.json
```

fallaba con `ModuleNotFoundError: No module named 'data'` porque el script no añadía la raíz del proyecto al `sys.path`.

### 3) Código muerto detectado

Se eliminaron helpers no referenciados en `data/preprocessing.py`:

- `global_day_offset_to_index`
- `global_day_index_to_offset`
- `local_start_offset_to_index`
- `local_start_index_to_offset`
- `build_target_day_offsets`

También se retiraron imports no usados en:

- `evaluation/weekly_stats.py`
- `scripts/analyze_dataset.py`

### 4) Artefactos no fuente dentro del proyecto

Se eliminaron del paquete optimizado:

- `__pycache__/`
- `*.pyc`
- `project_06_v3_reporting_metrics_fixed.zip`
- `reports/training_report.json`
- `predicted_database.json`
- `predicted_tasks.json`

Además se añadió `.gitignore` para evitar que vuelvan a contaminar el repo.

### 5) Micro-optimizaciones seguras

- `data/datasets.py`: se sustituyeron varias conversiones con `torch.tensor(...)` por `torch.from_numpy(...)` para evitar copias innecesarias de arrays ya normalizados.
- `data/preprocessing.py`: `prepare_data(...)` ahora agrupa `df_events` por `week_start` una sola vez, en lugar de filtrar el DataFrame completo dentro de cada iteración semanal.
- `predict.py`: las medianas de duración por tarea ya no se recalculan filtrando el DataFrame repetidamente; se precalculan en `PreparedData` y se reutilizan.

## Cambios aplicados

### Código

- `train.py`
  - se usa `TMP_E2E_EVAL_EVERY` en vez de `1`.
- `scripts/analyze_dataset.py`
  - fix de import path;
  - limpieza de imports no usados.
- `data/preprocessing.py`
  - eliminación de funciones muertas;
  - precálculo de medianas por tarea;
  - agrupación de semanas más eficiente.
- `data/datasets.py`
  - reducción de copias tensoriales innecesarias.
- `evaluation/weekly_stats.py`
  - limpieza de import no usado.

### Higiene del proyecto

- `.gitignore` nuevo.
- limpieza de cachés y outputs generados.

## Riesgo / compatibilidad

Los cambios son conservadores:

- no alteran la interfaz pública principal (`train.py`, `predict.py`);
- no cambian el formato de entrada de datos;
- no eliminan datasets base ni checkpoints esperados por el código;
- la principal modificación de comportamiento es que la evaluación end-to-end pasa a respetar la frecuencia configurada.

## Candidatos adicionales para una segunda ronda

No se eliminaron automáticamente, pero conviene revisar si realmente deben vivir en el repo:

- `aux_database.json`
- `aux_database_1.json`
- `aux_database_2.json`
- `robot_schedule_5years.json`
- `robot_schedule_gamma.json`
- `robot_schedule_v4_2025_2026.json`

No están referenciados por el pipeline Python actual. Podrían ser datasets alternativos o artefactos de integración externa.

## Validaciones ejecutadas

- compilación del proyecto: `python -m compileall`
- ejecución del script de análisis corregido
- smoke test de:
  - carga de datos
  - `prepare_data(...)`
  - construcción de `OccurrenceDataset`
  - construcción de `TemporalDataset`
  - `predict_next_week(...)`
