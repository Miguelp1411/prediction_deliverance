# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **38.70%**
- val close_acc_1: **78.12%**
- val close_acc_2: **92.07%**
- val count_mae: **0.974**
- val change_acc: **91.83%**

## Temporal residual model

- val start_exact_acc: **18.78%**
- val start_tol_5m: **22.30%**
- val start_tol_10m: **26.77%**
- val start_mae_minutes: **267.11 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **87.49%**
- time_exact_accuracy: **19.14%**
- time_close_accuracy_5m: **21.47%**
- time_close_accuracy_10m: **23.61%**
- start_mae_minutes: **619.92 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
