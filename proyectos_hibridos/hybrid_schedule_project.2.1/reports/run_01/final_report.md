# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **24.47%**
- val close_acc_1: **60.69%**
- val close_acc_2: **80.47%**
- val count_mae: **1.461**
- val change_acc: **76.06%**

## Temporal residual model

- val start_exact_acc: **27.04%**
- val start_tol_5m: **27.19%**
- val start_tol_10m: **28.10%**
- val start_mae_minutes: **540.24 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **82.05%**
- time_exact_accuracy: **26.07%**
- time_close_accuracy_5m: **27.70%**
- time_close_accuracy_10m: **30.11%**
- start_mae_minutes: **394.42 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
