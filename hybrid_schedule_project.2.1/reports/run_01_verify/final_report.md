# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **12.50%**
- val close_acc_1: **43.75%**
- val close_acc_2: **58.33%**
- val count_mae: **2.208**
- val change_acc: **77.08%**

## Temporal residual model

- val start_exact_acc: **0.25%**
- val start_tol_5m: **0.25%**
- val start_tol_10m: **0.25%**
- val start_mae_minutes: **987.94 min**
- val duration_mae_minutes: **0.10 min**

## Weekly backtest

- task_f1: **84.56%**
- time_exact_accuracy: **22.06%**
- time_close_accuracy_5m: **23.46%**
- time_close_accuracy_10m: **27.87%**
- start_mae_minutes: **284.52 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
