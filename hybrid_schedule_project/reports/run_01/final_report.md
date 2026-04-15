# Final training report

## Dataset summary

- eventos: **49100**
- bases de datos: **2**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **38.28%**
- val close_acc_1: **79.06%**
- val close_acc_2: **92.81%**
- val count_mae: **0.944**
- val change_acc: **74.53%**

## Temporal residual model

- val start_exact_acc: **35.37%**
- val start_tol_5m: **36.16%**
- val start_tol_10m: **40.07%**
- val start_mae_minutes: **206.70 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **82.45%**
- time_exact_accuracy: **24.74%**
- time_close_accuracy_5m: **26.84%**
- time_close_accuracy_10m: **29.23%**
- start_mae_minutes: **401.32 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
