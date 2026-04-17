# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **27.88%**
- val close_acc_1: **62.50%**
- val close_acc_2: **81.73%**
- val count_mae: **1.425**
- val change_acc: **94.47%**

## Temporal residual model

- val start_exact_acc: **36.98%**
- val start_tol_5m: **40.75%**
- val start_tol_10m: **47.98%**
- val start_mae_minutes: **111.26 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **88.66%**
- time_exact_accuracy: **31.45%**
- time_close_accuracy_5m: **33.42%**
- time_close_accuracy_10m: **36.71%**
- start_mae_minutes: **499.56 min**
- overlap_same_robot_count: **1**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
