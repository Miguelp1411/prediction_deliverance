# Final training report

## Dataset summary

- eventos: **49100**
- bases de datos: **2**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **34.22%**
- val close_acc_1: **74.38%**
- val close_acc_2: **90.62%**
- val count_mae: **1.059**
- val change_acc: **73.91%**

## Temporal residual model

- val start_exact_acc: **35.99%**
- val start_tol_5m: **36.76%**
- val start_tol_10m: **41.17%**
- val start_mae_minutes: **200.14 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **82.66%**
- time_exact_accuracy: **26.32%**
- time_close_accuracy_5m: **28.41%**
- time_close_accuracy_10m: **31.14%**
- start_mae_minutes: **393.96 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
