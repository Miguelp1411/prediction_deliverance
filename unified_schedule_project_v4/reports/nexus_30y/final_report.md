# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **98195**
- bases de datos: **1**
- robots: **1**
- semanas: **1566**
- overlap histórico mismo robot: **14**

## Validación del modelo unificado

- best_epoch: **101**
- best_val_loss: **3.639736092090607**
- best_composite: **56.45027478918827**
- val active_f1: **88.10%**
- val count_mae: **1.331**
- val day_acc: **92.07%**
- val start_tol_5m: **66.75%**
- val start_mae_minutes: **140.81 min**
- val duration_mae_minutes: **1.33 min**

## Weekly backtest

- task_f1: **94.72%**
- time_exact_accuracy: **78.58%**
- time_close_accuracy_5m: **78.81%**
- time_close_accuracy_10m: **79.18%**
- start_mae_minutes: **116.78 min**
- overlap_same_robot_count: **5**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
