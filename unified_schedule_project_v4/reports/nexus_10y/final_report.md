# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Validación del modelo unificado

- best_epoch: **124**
- best_val_loss: **3.0019806898557224**
- best_composite: **61.2779038668101**
- val active_f1: **90.94%**
- val count_mae: **0.820**
- val day_acc: **94.10%**
- val start_tol_5m: **69.72%**
- val start_mae_minutes: **88.00 min**
- val duration_mae_minutes: **1.64 min**

## Weekly backtest

- task_f1: **96.41%**
- time_exact_accuracy: **76.52%**
- time_close_accuracy_5m: **76.66%**
- time_close_accuracy_10m: **77.25%**
- start_mae_minutes: **125.41 min**
- overlap_same_robot_count: **5**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
