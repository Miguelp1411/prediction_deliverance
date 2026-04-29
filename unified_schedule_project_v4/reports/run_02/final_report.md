# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **98195**
- bases de datos: **1**
- robots: **1**
- semanas: **1566**
- overlap histórico mismo robot: **14**

## Validación del modelo unificado

- best_epoch: **110**
- best_val_loss: **4.139825832843781**
- best_composite: **54.5220297780303**
- val active_f1: **87.51%**
- val count_mae: **1.365**
- val day_acc: **91.08%**
- val start_tol_5m: **64.49%**
- val start_mae_minutes: **165.46 min**
- val duration_mae_minutes: **1.83 min**

## Weekly backtest

- task_f1: **95.09%**
- time_exact_accuracy: **75.75%**
- time_close_accuracy_5m: **75.86%**
- time_close_accuracy_10m: **76.47%**
- start_mae_minutes: **142.24 min**
- overlap_same_robot_count: **6**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
