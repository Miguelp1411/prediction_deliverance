# Arquitectura unificada propuesta

## Objetivo

Sustituir:

- red de occurrence,
- red temporal,
- acoplamiento posterior entre ambas,

por una única arquitectura que capture el problema real: **predecir una semana completa de eventos correlacionados**.

## Diseño

### 1. Representación del problema

Para cada pareja `(database_id, robot_id)` y para cada semana objetivo, construyo un conjunto fijo de queries:

`Q = número_de_tareas × max_slot_prototypes`

Cada query representa un posible slot ordinal de una tarea. Por ejemplo:

- tarea A, slot 0
- tarea A, slot 1
- ...
- tarea B, slot 0
- ...

Eso permite que el mismo modelo aprenda a decidir cuántos slots activar y cómo ubicarlos.

### 2. Entradas del modelo

Cada query recibe:

- historia semanal de la serie,
- embedding de tarea,
- embedding de base de datos,
- embedding de robot,
- embedding de slot ordinal,
- día ancla e instante ancla procedentes de prototipos históricos,
- vector numérico temporal construido con el mismo contexto rico del proyecto original.

### 3. Núcleo del modelo

El modelo usa tres niveles:

1. **History encoder**: transformer sobre semanas históricas.
2. **Cross-attention**: cada query atiende al historial para recuperar evidencia temporal relevante.
3. **Query interaction encoder**: las queries interactúan entre sí para modelar competencia, co-ocurrencia y densidad semanal.

Ese tercer nivel es la principal diferencia estructural frente al diseño original.

### 4. Salidas

Por cada query:

- `active_logits`
- `day_logits`
- `time_logits`
- `pred_log_duration`

La suma de probabilidades activas por tarea actúa como estimador continuo del conteo, sin necesidad de una segunda red.

## Pérdida

La loss total combina:

- BCE para activación,
- CE para día,
- CE para hora,
- SmoothL1 para duración,
- L1 de consistencia de conteos por tarea.

## Decodificación

No uso un segundo modelo. La inferencia hace:

1. cálculo de probabilidad de activación por slot;
2. selección de slots por tarea con suma esperada de probabilidades;
3. generación de candidatos top-k día × top-k hora;
4. beam search con penalización por ocupación y por desviación respecto al ancla.

## Por qué puede superar al proyecto original

- evita cascada de error entre occurrence y temporal;
- aprende dependencias cruzadas entre slots de la misma semana;
- la señal de conteo y la señal temporal comparten representación;
- el decoder de ocupación parte de una semántica unificada, no de dos modelos desacoplados.
