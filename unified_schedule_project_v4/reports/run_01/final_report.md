# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **16368**
- bases de datos: **1**
- robots: **1**
- semanas: **261**
- overlap histórico mismo robot: **0**

## Validación del modelo unificado

- best_epoch: **108**
- best_val_loss: **4.908065250941685**
- best_composite: **47.471751980456006**
- val active_f1: **83.92%**
- val count_mae: **1.910**
- val day_acc: **88.22%**
- val start_tol_5m: **53.30%**
- val start_mae_minutes: **178.33 min**
- val duration_mae_minutes: **4.13 min**

## Weekly backtest

- task_f1: **92.48%**
- time_exact_accuracy: **66.25%**
- time_close_accuracy_5m: **66.84%**
- time_close_accuracy_10m: **67.88%**
- start_mae_minutes: **197.76 min**
- overlap_same_robot_count: **3**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
