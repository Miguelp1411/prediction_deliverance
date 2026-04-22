# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**

## Arquitectura

Se ha sustituido la pipeline de dos redes (`occurrence` + `temporal`) por un único modelo de gran capacidad que opera a nivel de semana completa y slot potencial. Ese modelo decide simultáneamente:

- si un slot debe existir,
- en qué día debe ir,
- en qué minuto del día debe empezar,
- cuánto debe durar.

## Mejor métrica de validación registrada

- best_epoch: **76**
- best_val_loss: **3.887867835851816**
- val active_f1: **98.27%**
- val count_mae: **5.895**
- val start_mae_minutes: **265.94 min**
- val duration_mae_minutes: **1.23 min**

## Interpretación

La mejora conceptual frente al diseño original es que desaparece el error de acoplamiento entre conteo y colocación temporal. En este diseño, la misma red ve la semana completa, compite internamente por slots, y se decodifica con restricciones de ocupación en un único paso de inferencia.
