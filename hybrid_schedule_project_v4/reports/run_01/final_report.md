# Final training report

## Dataset summary

- eventos: **19542**
- bases de datos: **1**
- robots: **2**
- semanas: **261**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **55.29%**
- val close_acc_1: **87.26%**
- val close_acc_2: **94.71%**
- val count_mae: **0.685**
- val change_acc: **83.89%**

## Temporal residual model

- val start_exact_acc: **73.85%**
- val start_tol_5m: **75.24%**
- val start_tol_10m: **81.01%**
- val start_mae_minutes: **75.42 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **92.28%**
- time_exact_accuracy: **61.88%**
- time_close_accuracy_5m: **64.44%**
- time_close_accuracy_10m: **67.06%**
- start_mae_minutes: **407.91 min**
- overlap_same_robot_count: **1**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
