# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **4.17%**
- val close_acc_1: **14.58%**
- val close_acc_2: **16.67%**
- val count_mae: **9.792**
- val change_acc: **93.75%**

## Temporal residual model

- val start_exact_acc: **0.26%**
- val start_tol_5m: **0.26%**
- val start_tol_10m: **0.26%**
- val start_mae_minutes: **611.09 min**
- val duration_mae_minutes: **0.02 min**

## Weekly backtest

- task_f1: **68.52%**
- time_exact_accuracy: **8.11%**
- time_close_accuracy_5m: **8.11%**
- time_close_accuracy_10m: **8.11%**
- start_mae_minutes: **356.76 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
