# Final training report

## Dataset summary

- eventos: **49100**
- bases de datos: **2**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **27.08%**
- val close_acc_1: **45.83%**
- val close_acc_2: **62.50%**
- val count_mae: **2.167**
- val change_acc: **75.00%**

## Temporal residual model

- val start_exact_acc: **0.00%**
- val start_tol_5m: **0.13%**
- val start_tol_10m: **0.13%**
- val start_mae_minutes: **833.92 min**
- val duration_mae_minutes: **0.09 min**

## Weekly backtest

- task_f1: **81.87%**
- time_exact_accuracy: **3.29%**
- time_close_accuracy_5m: **6.33%**
- time_close_accuracy_10m: **13.51%**
- start_mae_minutes: **520.56 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
