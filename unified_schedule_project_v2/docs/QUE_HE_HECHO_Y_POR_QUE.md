# Qué he hecho y por qué

## 1. He preservado el contexto útil del proyecto original

He mantenido la parte valiosa del proyecto:

- carga multi-fuente por registry,
- normalización y esquema canónico,
- construcción de series por base y robot,
- historial semanal,
- prototipos ordinales de slots,
- features temporales ricas.

No tenía sentido reescribir eso porque ahí ya había trabajo bueno y alineado con el dominio.

## 2. He eliminado el desacoplamiento de modelos

Antes:

- un modelo para occurrence,
- otro modelo para temporal,
- una capa posterior para cerrar conflictos.

Ahora:

- un único modelo conjunto que decide existencia + localización + duración.

Eso reduce el acoplamiento frágil entre etapas.

## 3. He movido la unidad de decisión desde “evento aislado” a “semana completa”

La mejora importante no es solo “hacer el modelo más grande”. Es cambiar la geometría del problema.

El modelo nuevo ve una parrilla completa de slots potenciales de la semana y los procesa con interacción entre queries. Eso le deja aprender:

- densidad semanal,
- patrones de secuencia entre tareas,
- competencia por huecos temporales,
- redundancia entre slots del mismo task.

## 4. He dejado un decoder coherente con la petición de un único modelo

No he montado otro predictor auxiliar. La decodificación es una búsqueda restringida sobre las salidas del mismo modelo, con beam search y penalización de ocupación.

## 5. He documentado la honestidad del cambio

El nuevo proyecto está preparado para ser más fuerte, pero la superioridad real solo se demuestra reentrenando ambos sistemas y comparando backtests homogéneos.
