# PATCH NOTES — Occurrence residual delta bounds by task (v4.4)

## Qué cambia

Se aplica la **Opción B** sobre el `OccurrenceModel` actual:

- el modelo sigue usando **baseline estacional por tarea** con lags `(4, 26, 52)`;
- sigue aprendiendo un **residual ordinal**;
- pero deja de tratar todos los deltas `[-max_count_cap, +max_count_cap]` como igualmente plausibles para todas las tareas.

Ahora se calcula un **rango de delta por tarea** a partir del histórico de entrenamiento y se añade una **penalización suave creciente** cuando la red intenta salir de ese rango.

## Implementación

### 1. Cálculo de rangos por tarea

En `train.py` se añade `_compute_task_delta_bounds(...)`.

Para cada tarea:

1. se reconstruye el baseline estacional con los mismos lags y pesos del modelo;
2. se calcula el residual histórico:

   `delta = count_real - baseline_count`

3. se obtiene un rango robusto por cuantiles:

   - `low = quantile(delta, OCC_DELTA_RANGE_LOW_QUANTILE)`
   - `high = quantile(delta, OCC_DELTA_RANGE_HIGH_QUANTILE)`

4. se amplía con un margen configurable;
5. se fuerza un radio mínimo alrededor de `0` para no estrangular la red.

### 2. Penalización suave fuera de rango

En `models/occurrence_model.py`:

- se registran `task_delta_bounds` por tarea;
- se calcula la distancia de cada clase de delta al rango válido;
- se suma a los logits un término:

  `penalty = distance_to_valid_range * OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP`

Con el valor por defecto `-2.0`, cuanto más lejos esté un delta del rango histórico plausible, peor queda su logit.

No es un hard-mask estricto, así que el modelo todavía puede salir del rango en semanas anómalas, pero le cuesta bastante más.

## Config nueva

Añadida en `config.py`:

```python
OCC_USE_TASK_DELTA_RANGES = True
OCC_DELTA_RANGE_LOW_QUANTILE = 0.025
OCC_DELTA_RANGE_HIGH_QUANTILE = 0.975
OCC_DELTA_RANGE_MARGIN = 1
OCC_DELTA_RANGE_MIN_RADIUS = 4
OCC_DELTA_OUTSIDE_RANGE_LOGIT_PENALTY_PER_STEP = -2.0
```

## Persistencia / compatibilidad

Los bounds por tarea y la configuración asociada se guardan en:

- `metadata`
- `model_hparams`

Eso permite que `predict.py` reconstruya el mismo `OccurrenceModel` durante inferencia.

La carga de checkpoints antiguos sigue siendo tolerante (`strict=False`).

## Efecto esperado

- menos capacidad desperdiciada en deltas extremos irreales;
- predicciones más estables por tarea;
- menor ruido en los conteos que consume el modelo temporal;
- mejor ajuste con la arquitectura residual existente, sin rehacer el pipeline.

## Bounds observados en este dataset

Con la configuración por defecto y usando el split de entrenamiento de este proyecto, los bounds derivados fueron:

- Assist: `[-9, +10]`
- Clean: `[-8, +9]`
- Delivery: `[-10, +11]`
- Patroll: `[-10, +11]`

Estos valores no están fijados a mano: se recalculan desde el histórico cuando se vuelve a entrenar.
