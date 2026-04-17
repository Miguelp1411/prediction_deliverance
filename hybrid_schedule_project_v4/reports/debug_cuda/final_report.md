# Final training report

## Dataset summary

- eventos: **49100**
- bases de datos: **2**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **28.91%**
- val close_acc_1: **68.44%**
- val close_acc_2: **85.00%**
- val count_mae: **1.256**
- val change_acc: **76.09%**

## Temporal residual model

- val start_exact_acc: **14.93%**
- val start_tol_5m: **15.40%**
- val start_tol_10m: **18.29%**
- val start_mae_minutes: **385.68 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **81.39%**
- time_exact_accuracy: **26.30%**
- time_close_accuracy_5m: **28.66%**
- time_close_accuracy_10m: **32.01%**
- start_mae_minutes: **337.27 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
