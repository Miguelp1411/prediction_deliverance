# Cambios — reparación temporal endurecida

Se ha corregido la política de `repair` para reducir desplazamientos grandes e impedir el efecto dominó entre ocurrencias de la misma tarea.

## Qué se ha cambiado

### 1. Reparación con prioridad por mismo día
En `predict.py` y `auxiliary_corrector.py` el `repair` ya no hace un barrido lineal con fallback temprano desde `lower_bound`.

Ahora sigue esta política:
1. intentar mantener la tarea en su **mismo día**,
2. si no hay hueco, buscar en **días adyacentes**,
3. solo como último recurso usar un **fallback global con coste**, penalizando mucho cambiar de día.

## 2. Coste de recolocación
Se añade una función de coste que penaliza:
- cambio de día,
- cambio de hora dentro del día,
- desplazamiento total.

Esto hace que el `repair` prefiera una recolocación cercana antes que mandar una tarea varios días después.

## 3. Eliminado el efecto dominó de `last_start_by_task`
Antes, si una ocurrencia de una tarea era desplazada muy lejos, las siguientes ocurrencias de esa misma tarea heredaban ese desplazamiento como cota inferior dura.

Ahora esa cota se recorta contra el `preferred_start` actual:
- una mala recolocación previa ya **no arrastra** toda la serie.

## 4. Metadatos diagnósticos por tarea reparada
Las predicciones reparadas incluyen:
- `repair_preferred_start_bin`
- `repair_day_shift`
- `repair_displacement_bins`

Esto facilita diagnosticar cuánto está moviendo realmente el `repair`.

## 5. Nueva configuración
En `config.py` se han añadido:
- `PREDICTION_REPAIR_MAX_DAY_SHIFT = 1`
- `PREDICTION_REPAIR_DAY_CHANGE_PENALTY = 144`
- `PREDICTION_REPAIR_GLOBAL_FALLBACK = True`

## Validación rápida
En una comprobación rápida sobre una semana retenida de `robot_schedule_v4_2025_2026` con CPU:
- `repair` seguía evitando solapes,
- el MAE de inicio del calendario reparado quedó en ~155 min,
- frente a una versión intermedia del repair que estaba produciendo ~1432 min y cascadas multi-día.

Esto no convierte todavía el `repair` en óptimo global, pero sí elimina el comportamiento más destructivo que estaba rompiendo el calendario.
