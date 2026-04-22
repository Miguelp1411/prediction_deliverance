# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **27.16%**
- val close_acc_1: **61.06%**
- val close_acc_2: **77.64%**
- val count_mae: **1.762**
- val change_acc: **93.27%**

## Temporal residual model

- val start_exact_acc: **68.04%**
- val start_tol_5m: **68.34%**
- val start_tol_10m: **71.08%**
- val start_mae_minutes: **77.03 min**
- val duration_mae_minutes: **1.25 min**

## Weekly backtest

- task_f1: **86.83%**
- time_exact_accuracy: **43.97%**
- time_close_accuracy_5m: **44.43%**
- time_close_accuracy_10m: **44.89%**
- start_mae_minutes: **541.21 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
