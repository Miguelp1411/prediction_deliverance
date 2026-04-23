# Cambios aplicados: cálculo automático de caps y compatibilidad de checkpoints

## Objetivo
Dejar el proyecto robusto para que `MAX_OCCURRENCES_PER_TASK` y `MAX_TASKS_PER_WEEK`:

1. se infieran automáticamente desde la base de datos,
2. se guarden en los checkpoints al entrenar,
3. se recuperen correctamente al predecir,
4. y sigan funcionando con checkpoints antiguos.

## Cambios principales

### 1) `config.py`
- Nuevo parámetro: `CAP_INFERENCE_SCOPE = 'full_dataset'`
- Por defecto, los caps se infieren sobre toda la base histórica cargada.

### 2) `data/preprocessing.py`
- Añadidas funciones:
  - `infer_preprocessing_caps(...)`
  - `resolve_preprocessing_caps(...)`
- `prepare_data(...)` ahora:
  - acepta `cap_inference_scope`,
  - calcula caps de `train` y de `full_dataset`,
  - usa por defecto `full_dataset`,
  - devuelve también trazabilidad interna de cómo se infirieron.
- `serialize_metadata(...)` ahora guarda:
  - `max_occurrences_per_task`
  - `max_tasks_per_week`
  - `cap_inference_scope`
  - caps inferidos en train y en base completa

### 3) `train.py`
- Entrena usando `CAP_INFERENCE_SCOPE`.
- Imprime en consola:
  - caps finales usados,
  - caps de train,
  - caps de base completa.
- Guarda más metadata útil en ambos checkpoints.

### 4) `predict.py`
- Recupera caps desde:
  1. metadata del checkpoint,
  2. hyperparams del checkpoint,
  3. y si faltan, los infiere desde el JSON de entrada.
- Se vuelve compatible con checkpoints antiguos que no guardaban `max_tasks_per_week`.
- Refuerza la comprobación de compatibilidad usando también `model_hparams`.

### 5) Checkpoints incluidos
Se han actualizado los checkpoints incluidos para que ya lleven metadata consistente.
Con `aux_database.json` los valores quedan:
- `max_occurrences_per_task = 30`
- `max_tasks_per_week = 97`

## Resultado
- El proyecto ya no depende de escribir esos caps a mano.
- La predicción deja de caer al fallback incorrecto de `100` para `max_tasks_per_week` en el caso de los checkpoints incluidos.
- El flujo train -> save checkpoint -> predict queda coherente.
