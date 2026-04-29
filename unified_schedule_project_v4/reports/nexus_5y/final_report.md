# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **16368**
- bases de datos: **1**
- robots: **1**
- semanas: **261**
- overlap histórico mismo robot: **0**

## Validación del modelo unificado

- best_epoch: **117**
- best_val_loss: **4.803776741027832**
- best_composite: **47.437517775670315**
- val active_f1: **84.47%**
- val count_mae: **1.845**
- val day_acc: **88.35%**
- val start_tol_5m: **52.39%**
- val start_mae_minutes: **180.42 min**
- val duration_mae_minutes: **3.90 min**

## Weekly backtest

- task_f1: **92.50%**
- time_exact_accuracy: **67.19%**
- time_close_accuracy_5m: **68.22%**
- time_close_accuracy_10m: **68.96%**
- start_mae_minutes: **203.02 min**
- overlap_same_robot_count: **4**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
