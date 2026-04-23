# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **27.31%**
- val close_acc_1: **62.16%**
- val close_acc_2: **82.66%**
- val count_mae: **1.390**
- val change_acc: **75.91%**

## Temporal residual model

- val start_exact_acc: **25.05%**
- val start_tol_5m: **25.21%**
- val start_tol_10m: **26.10%**
- val start_mae_minutes: **557.73 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **82.50%**
- time_exact_accuracy: **27.91%**
- time_close_accuracy_5m: **29.58%**
- time_close_accuracy_10m: **31.70%**
- start_mae_minutes: **377.97 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
