# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **4.17%**
- val close_acc_1: **16.67%**
- val close_acc_2: **27.08%**
- val count_mae: **7.354**
- val change_acc: **95.83%**

## Temporal residual model

- val start_exact_acc: **7.99%**
- val start_tol_5m: **8.38%**
- val start_tol_10m: **9.43%**
- val start_mae_minutes: **847.01 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **77.17%**
- time_exact_accuracy: **18.37%**
- time_close_accuracy_5m: **18.37%**
- time_close_accuracy_10m: **22.45%**
- start_mae_minutes: **523.98 min**
- overlap_same_robot_count: **7**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
