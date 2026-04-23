# Regularización útil por dataset

Esta ampliación introduce **regularización guiada por perfil de base de datos** en lugar de aplicar el mismo castigo a todas las fuentes.

## Problemas detectados en el proyecto original

1. **Regularización global y casi uniforme**: solo había `dropout` y `weight_decay` comunes.
2. **Pérdidas demasiado duras**:
   - `occurrence` trataba todos los errores de delta como clases totalmente separadas.
   - `temporal` castigaba igual una predicción a 5 minutos del target y otra a varias horas.
3. **Sin regularización estructural**:
   - no existía anclaje explícito a la plantilla,
   - no se penalizaba sobreconfianza,
   - no se usaban priors recientes de día/hora en la pérdida,
   - no había ruido de entrada adaptado al tamaño del dataset.
4. **Sampling desbalanceado incompleto**: `occurrence` tenía sampler opcional, `temporal` no.
5. **`change_loss_weight` estaba en config pero no se estaba aplicando realmente en entrenamiento**.

## Qué se ha añadido

### 1) Perfiles de regularización por base de datos
Se calcula para cada dataset:

- **escasez**: menos semanas / menos masa histórica → más regularización de entrada y más smoothing;
- **volatilidad**: mayor variabilidad semanal → targets más suaves en `occurrence` y `temporal`;
- **rigidez**: si el horario es repetitivo → más anclaje a plantilla y más peso de los priors.

### 2) Regularización en `occurrence`

- **soft targets gaussianos para delta**: una diferencia de ±1 deja de ser penalizada como si fuera totalmente distinta.
- **label smoothing adaptativo por dataset**.
- **shrinkage al residual**: penaliza deltas esperados grandes cuando la base es más rígida o escasa.
- **confidence penalty**: reduce predicciones excesivamente picudas en bases pequeñas.

### 3) Regularización en `temporal`

- **soft targets temporales**: la hora objetivo se modela con una distribución alrededor del bin real.
- **day label smoothing**.
- **prior regularization** con distribución reciente de día y hora.
- **anchor regularization**: la predicción no se aleja gratuitamente del slot plantilla.
- **smoothness regularization** en logits temporales para evitar distribuciones irregulares y memorísticas.
- **confidence penalty** también en la cabeza temporal.

### 4) Regularización de entrada

Durante entrenamiento:

- `history dropout`
- `history gaussian noise`
- `feature dropout`
- `feature gaussian noise`

Todo ello **adaptado a cada dataset**.

### 5) Sampling balanceado también para temporal

Ahora ambos datasets de entrenamiento pueden usar pesos balanceados por base, robot y tarea.

## Criterio práctico por dataset

### `robot_schedule_gamma`
- Mucho volumen y muchos robots.
- Regularización útil: **moderada**, con énfasis en evitar memorizar combinaciones robot-base y en conservar coherencia temporal.

### `nexus_schedule_10years` y `nexus_schedule_5years`
- Mucha historia, pero **más volatilidad semanal** que el resto.
- Regularización útil: **targets suaves y menos castigo por pequeños desplazamientos**, no tanto anclaje rígido a plantilla.

### `robot_schedule_5years`
- Régimen intermedio: volumen medio, 2 robots y volatilidad media-alta.
- Regularización útil: combinación de **soft targets**, priors temporales y shrinkage moderado.

### `aux_database`, `aux_database_1`, `aux_database_2`
- Menor volumen y horizontes más cortos.
- Especialmente `aux_database_2` parece más estable.
- Regularización útil:
  - más ruido de entrada y smoothing en las bases pequeñas,
  - más anclaje a plantilla donde la agenda es rígida,
  - menos libertad residual cuando el patrón histórico es muy repetitivo.

### `robot_schedule_v4_2025_2026`
- Es la base con menos semanas.
- Debe recibir la **regularización más fuerte** para no sobreajustar a un único ciclo anual corto.

## Artefactos nuevos

- `src/hybrid_schedule/training/regularization.py`
- `examples/registry_all_datasets.yaml`
- `reports/.../database_regularization_profiles.json`
- `reports/.../database_regularization_profiles.md`

## Recomendación de uso

Para entrenar sobre todas las bases y verificar los perfiles nuevos:

```bash
python scripts/train.py \
  --registry examples/registry_all_datasets.yaml \
  --config configs/default.yaml \
  --output-dir reports/run_all_regularized
```

Para una verificación rápida:

```bash
python scripts/train.py \
  --registry examples/registry_all_datasets.yaml \
  --config configs/default.yaml \
  --output-dir reports/smoke_regularized \
  --smoke
```
