# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **27.16%**
- val close_acc_1: **57.21%**
- val close_acc_2: **78.12%**
- val count_mae: **1.572**
- val change_acc: **95.43%**

## Temporal residual model

- val start_exact_acc: **38.09%**
- val start_tol_5m: **42.34%**
- val start_tol_10m: **49.47%**
- val start_mae_minutes: **108.44 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **87.70%**
- time_exact_accuracy: **32.62%**
- time_close_accuracy_5m: **34.22%**
- time_close_accuracy_10m: **37.58%**
- start_mae_minutes: **482.28 min**
- overlap_same_robot_count: **0**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
