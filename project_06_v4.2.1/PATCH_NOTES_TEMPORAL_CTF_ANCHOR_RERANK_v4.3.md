# Patch notes v4.3 — coarse-to-fine temporal + anchor ensemble + weekly reranker

## Cambios principales

### 1) Temporal model: salida absoluta coarse-to-fine
Se eliminó la formulación relativa `day_offset + local_offset`.
Ahora el modelo temporal predice:
- `day_logits`: 7 clases (día absoluto dentro de la semana)
- `time_of_day_logits`: 288 clases (bin absoluto de 5 minutos dentro del día)
- `pred_duration_norm`

Esto evita el cuello de botella anterior de offsets locales limitados.

### 2) Anchor mejorado y relajado
`build_temporal_context()` ahora acepta `predicted_count` y construye un conjunto de anchors candidatos mezclando:
- misma ocurrencia reciente
- misma ocurrencia histórica larga
- `lag52`
- `lag26`
- medianas históricas
- prototype del slot más cercano
- prototype condicionado por la posición relativa del slot y el recuento previsto

El contexto temporal ahora expone:
- `anchor_start_bin`
- `anchor_day`
- `anchor_time_bin`
- `anchor_candidates`
- `anchor_candidate_weights`

### 3) Reranking semanal conjunto
`predict.py` ya no usa greedy independiente por ocurrencia.
Ahora:
- genera candidatos temporales desde las salidas coarse-to-fine
- mezcla esos candidatos con anchors candidatos
- aplica un beam reranker semanal con penalización por:
  - solapes
  - orden incoherente entre ocurrencias de la misma tarea
  - distancia al anchor
- aplica reparación final opcional de calendario (`PREDICTION_USE_REPAIR=True`)

### 4) Occurrence model: mayor peso de `count_exact_acc` en validación
La selección del mejor checkpoint ya no se hace con `weekly_total_mae` solamente.
Ahora se monitoriza `occurrence_selection_score`, una métrica compuesta que favorece:
- mayor `count_exact_acc`
- menor `weekly_total_mae`
- menor `count_mae`

También se ajustó el scheduler para optimizar esa métrica en modo `max`.

## Ficheros tocados
- `config.py`
- `data/preprocessing.py`
- `data/datasets.py`
- `models/temporal_model.py`
- `training/losses.py`
- `training/metrics.py`
- `training/engine.py`
- `predict.py`
- `train.py`

## Compatibilidad
Este parche cambia el esquema temporal y el `FEATURE_SCHEMA_VERSION`.
Los checkpoints anteriores deben considerarse incompatibles para inferencia con el nuevo temporal model. Hay que reentrenar.
