# OccurrenceModel v4.8 — Mixture-of-Lags + Residual Delta + Compact Task Encoder

## Qué cambia

La v4.8 sustituye el `task_gru` grande y generalista por un modelo más compacto y específico para forecasting de conteos semanales:

- **encoder compacto por tarea** con GRU pequeña compartida;
- **features específicas de occurrence**: conteo, delta reciente, actividad, multimodalidad y calendario;
- **mixture-of-lags discreto** en lugar de promedio de lags;
- **residual por delta** sobre candidatos discretos;
- **loss suavizada** para premiar exactitud y cercanía (`exact`, `±1`, `±2`);
- **aux losses** para supervisar:
  - qué candidato estacional conviene copiar;
  - qué delta residual conviene aplicar.

## Motivación

En este proyecto el problema del occurrence no parece venir de falta de capacidad, sino de:

1. demasiados parámetros para muy pocas semanas de train;
2. prior estacional basado en promedio de lags, que degrada exactitud;
3. loss poco alineada con las métricas reales (`exact`, `±1`, `±2`);
4. mezcla de features demasiado generalistas para un problema puramente de conteo.

## Arquitectura

Para cada tarea, el modelo construye una secuencia temporal compacta con:

- conteo semanal;
- delta contra la semana anterior;
- actividad semanal (`active_days_norm`);
- multimodalidad diaria;
- total de tareas en la semana;
- calendario cíclico.

Esa secuencia se codifica con una **BiGRU pequeña** compartida entre tareas.
Luego se concatena con:

- embedding de tarea;
- resumen reciente (`mean`, `std`, último conteo, último delta).

A partir de ahí salen dos ramas:

### 1) `selector_head`
Decide qué candidato discreto copiar entre:

- `lag4`
- `lag26`
- `lag52`
- `recent_mean`
- `last_count`
- `task_median`

(según flags activas)

### 2) `delta_head`
Predice un delta residual discreto sobre soporte completo `[-max_count_cap, +max_count_cap]`, con sesgo positivo hacia `delta=0` y penalización fuera del rango histórico por tarea.

## Cómo se combinan

No se promedian los lags.
En vez de eso se construye una **mezcla discreta de candidatos**:

- cada candidato propone un baseline de conteo exacto;
- el `delta_head` aprende cuánto desplazar ese candidato;
- la mezcla final se forma con `logaddexp` sobre todos los candidatos válidos.

Así, si `lag52` ya era exactamente correcto, el modelo puede copiarlo sin degradarlo con un promedio.

## Loss nueva

La loss principal ahora usa un target suave sobre el conteo final:

- `exact`: 0.62
- `±1` total: 0.28
- `±2` total: 0.10

además de una penalización por MAE del conteo esperado.

Y se añaden dos pérdidas auxiliares:

- **selector loss**: enseña al selector a elegir el candidato cuya distancia al target sea mínima;
- **delta loss**: enseña al residual a corregir el mejor candidato disponible.

## Cambios de entrenamiento

Se reduce la complejidad del OccurrenceModel:

- `hidden_size`: 128 → 64
- `num_layers`: 2 → 1
- `dropout`: 0.30 → 0.20
- `batch_size`: 24 → 32
- `max_epochs`: 180 → 120
- `patience`: 24 → 18

Con esto se busca menos overfitting y entrenamiento más rápido.

## Compatibilidad

El modelo devuelve un dict con:

- `logits`
- `delta_logits`
- `selector_logits`
- `candidate_counts`
- `candidate_masks`

Pero el resto del pipeline sigue funcionando porque `predict.py`, `datasets.py`, `losses.py` y `metrics.py` ya extraen `logits` cuando hace falta.
