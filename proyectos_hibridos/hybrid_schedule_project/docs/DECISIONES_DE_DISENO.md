# Decisiones de diseño frente al proyecto original

## 1. La predicción base sale de memoria histórica, no de generación libre

El proyecto nuevo asume que, para este tipo de datos, la señal dominante está en la repetición de semanas históricas similares. Por eso la primera decisión de arquitectura es:

- recuperar semanas análogas,
- construir una plantilla base,
- aprender solo la corrección residual.

Esto evita que la red compita innecesariamente contra un baseline estacional fuerte.

## 2. Occurrence se modela como corrección residual

En vez de predecir conteos absolutos desde cero:

- `template_count` actúa como baseline,
- la red predice `changed / unchanged`,
- la red predice `delta` acotado.

Eso está alineado con el problema que describías: el valor está en capturar dónde falla la plantilla, no en reconstruir toda la semana desde cero.

## 3. El modelo temporal no decide la agenda final

El temporal residual solo produce:

- offsets de día,
- offsets finos de tiempo,
- ajuste de duración,
- top-k candidatos.

La decisión final se aplaza a un optimizador global para no romper la coherencia semanal.

## 4. El cierre semanal se hace con optimización global

Se construye una formulación MILP con restricciones de exclusión temporal por robot. Para tamaños moderados el solver es exacto. Para tamaños grandes, el proyecto cae a un fallback greedy controlado para garantizar operatividad. En producción, la recomendación es sustituir ese fallback por OR-Tools CP-SAT.

## 5. Multi-base de datos desde el diseño

El proyecto no depende de un `DATA_PATH` fijo. Todas las fuentes entran por:

- un `registry.yaml`,
- adapters por tipo de fuente,
- normalización a esquema canónico,
- embeddings explícitos de `database_id` y `robot_id`.

Eso hace viable entrenar un modelo global y luego especializarlo por base.

## 6. Reporting como parte del producto, no como añadido

Cada entrenamiento guarda:

- JSONL por epoch,
- CSV consolidado,
- plots,
- checkpoints,
- resumen final,
- backtest semanal.

El objetivo es que el proyecto no solo prediga, sino que también pueda auditarse, compararse y depurarse.
