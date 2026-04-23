# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Validación del modelo unificado

- best_epoch: **76**
- best_val_loss: **3.887867835851816**
- val active_f1: **98.27%**
- val count_mae: **5.895**
- val day_acc: **88.64%**
- val start_tol_5m: **65.37%**
- val start_mae_minutes: **265.94 min**
- val duration_mae_minutes: **1.23 min**

## Weekly backtest

- task_f1: **79.34%**
- time_exact_accuracy: **75.80%**
- time_close_accuracy_5m: **76.26%**
- time_close_accuracy_10m: **76.82%**
- start_mae_minutes: **131.26 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
