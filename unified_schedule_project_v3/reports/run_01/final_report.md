# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Validación del modelo unificado

- best_epoch: **87**
- best_val_loss: **2.421086696478037**
- val active_f1: **90.44%**
- val count_mae: **0.989**
- val day_acc: **94.20%**
- val start_tol_5m: **67.28%**
- val start_mae_minutes: **112.54 min**
- val duration_mae_minutes: **1.54 min**

## Weekly backtest

- task_f1: **95.42%**
- time_exact_accuracy: **75.76%**
- time_close_accuracy_5m: **76.07%**
- time_close_accuracy_10m: **77.16%**
- start_mae_minutes: **138.67 min**
- overlap_same_robot_count: **5**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
