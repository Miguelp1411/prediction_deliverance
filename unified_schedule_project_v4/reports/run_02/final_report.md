# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **16368**
- bases de datos: **1**
- robots: **1**
- semanas: **261**
- overlap histórico mismo robot: **0**

## Validación del modelo unificado

- best_epoch: **118**
- best_val_loss: **4.931566306522915**
- best_composite: **47.2462268841164**
- val active_f1: **83.84%**
- val count_mae: **1.934**
- val day_acc: **88.09%**
- val start_tol_5m: **53.20%**
- val start_mae_minutes: **181.64 min**
- val duration_mae_minutes: **4.21 min**

## Weekly backtest

- task_f1: **92.62%**
- time_exact_accuracy: **67.61%**
- time_close_accuracy_5m: **68.13%**
- time_close_accuracy_10m: **69.18%**
- start_mae_minutes: **188.70 min**
- overlap_same_robot_count: **3**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
