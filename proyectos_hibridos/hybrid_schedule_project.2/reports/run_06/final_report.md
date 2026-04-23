# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **26.47%**
- val close_acc_1: **61.41%**
- val close_acc_2: **82.28%**
- val count_mae: **1.418**
- val change_acc: **73.94%**

## Temporal residual model

- val start_exact_acc: **28.37%**
- val start_tol_5m: **28.58%**
- val start_tol_10m: **29.63%**
- val start_mae_minutes: **548.07 min**
- val duration_mae_minutes: **0.01 min**

## Weekly backtest

- task_f1: **82.52%**
- time_exact_accuracy: **27.52%**
- time_close_accuracy_5m: **29.32%**
- time_close_accuracy_10m: **31.62%**
- start_mae_minutes: **413.97 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
