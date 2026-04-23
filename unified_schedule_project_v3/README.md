# Unified schedule project v1

Versión re-diseñada del proyecto original para forecasting semanal de agendas, pero con una sola red grande en lugar de una arquitectura partida en `occurrence + temporal + solver exacto`.

## Idea central

El proyecto original tiene una lógica fuerte, pero parte el problema en dos decisiones consecutivas:

1. cuántas veces aparece cada tarea;
2. dónde colocar cada aparición.

Ese desacoplamiento introduce dos riesgos:

- **propagación de error**: si falla el conteo, el temporal ya parte mal;
- **subóptimo estructural**: el modelo temporal no razona desde el principio sobre toda la semana completa.

En esta versión nueva se reemplaza esa tubería por un **único modelo grande, conjunto y a nivel de semana**, que decide por cada `task × slot ordinal`:

- activación del slot,
- día de la semana,
- hora concreta,
- duración.

## Componentes

- `UnifiedSlotTransformer`
  - encoder transformer de historial semanal,
  - embeddings estructurales (task, db, robot, slot, día ancla, hora ancla),
  - proyección numérica profunda,
  - capas de cross-attention query→historial,
  - transformer sobre queries de la semana para modelar competencia y coocurrencia,
  - heads conjuntos para presencia, día, hora y duración.

- `UnifiedWeekSlotDataset`
  - construye una semana como un conjunto fijo de queries `num_tasks × max_slot_prototypes`,
  - usa prototipos históricos por ordinal para conservar la memoria estructural del proyecto original,
  - asigna eventos reales a esos prototipos para entrenar targets conjuntos.

- `decode_week_with_constraints`
  - decodificación con restricciones suaves de ocupación,
  - sin depender de un segundo modelo,
  - con búsqueda por beam sobre candidatos top-k de día y hora.

## Qué mantiene del proyecto original

- ingestión multi-base desde `registry.yaml`,
- normalización de eventos,
- construcción de contexto global por base y robot,
- features históricas y temporales fuertes,
- entrenamiento y predicción por scripts.

## Entrenamiento

```bash
PYTHONPATH=src python scripts/train.py \
  --registry examples/registry_template.yaml \
  --output-dir reports/run_01
```

## Predicción

```bash
PYTHONPATH=src python scripts/predict_week.py \
  --registry examples/registry_template.yaml \
  --output-dir reports/run_01 \
  --database-id TU_BASE \
  --robot-id TU_ROBOT
```

## Honestidad técnica

Esta arquitectura está **diseñada para ser mejor** que la original en este problema porque elimina el desacoplamiento entre conteo y localización temporal y modela la semana completa de forma conjunta. Pero **no puedo afirmar una mejora empírica ya demostrada** sin reentrenar y comparar en backtest con los mismos datos y el mismo protocolo de evaluación.
