# Mejoras Candidate Bank V6

## Cambios implementados

### 1. Candidate bank multifuente
Se amplió `build_empirical_candidate_bank(...)` para indexar candidatos por:

- `slot`
- `task`
- `seasonal`
- `week_type`
- `density`
- `regime`
- `precedence`
- `prototypes`

Cada candidato ahora acumula soporte contextual, fuentes de recuperación y compatibilidad con el slot objetivo.

### 2. Contexto de objetivo
Se añadió `build_target_candidate_context(...)` para inferir, tanto en entrenamiento como en inferencia:

- estación (`season_bucket`)
- tipo de semana (`week_type`)
- densidad prevista (`density_bucket`)
- régimen reciente (`regime_id`)
- precedencia esperada (`precedence_key`)

### 3. Prototipos contextuales
Se añadió `task_slot_prototypes_contextual(...)` y se integró dentro de `build_template_week(...)`.
La plantilla mezcla prototipos históricos base con prototipos filtrados por contexto objetivo.

### 4. Recuperación diversa
`gather_empirical_candidates(...)` ahora:

- mezcla varias fuentes con cuotas por tipo
- recupera por vecinos de slot y por contexto
- añade candidatos sintéticos alrededor de anchors/prototipos (`prototype jitter`)
- usa precedencia para recuperar y para generar anchors extra

### 5. Features temporales ampliadas
`build_temporal_candidate_features(...)` incorpora ahora metadatos del candidate bank:

- flags por fuente
- matches de estación, tipo de semana, densidad, régimen y precedencia
- fuerza de fuente
- número de fuentes activas
- score de consistencia contextual

### 6. Integración end-to-end
Se propagaron los cambios a:

- `training/datasets.py`
- `inference/predictor.py`
- `scripts/train.py`
- `configs/default.yaml`

## Reentrenamiento requerido
No hace falta tocar occurrence ni scheduler, pero **sí reentrenar el modelo temporal**, porque cambia:

- el candidate set
- la dimensionalidad de `candidate_features`
- la distribución de soporte/contexto
