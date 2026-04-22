# Final training report

## Dataset summary

- eventos: **19542**
- bases de datos: **1**
- robots: **2**
- semanas: **261**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **56.73%**
- val close_acc_1: **83.17%**
- val close_acc_2: **89.90%**
- val count_mae: **0.841**
- val change_acc: **85.82%**

## Temporal residual model

- val start_exact_acc: **87.19%**
- val start_tol_5m: **87.45%**
- val start_tol_10m: **90.31%**
- val start_mae_minutes: **38.78 min**
- val duration_mae_minutes: **1.34 min**

## Weekly backtest

- task_f1: **91.54%**
- time_exact_accuracy: **50.60%**
- time_close_accuracy_5m: **51.30%**
- time_close_accuracy_10m: **55.14%**
- start_mae_minutes: **595.02 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
