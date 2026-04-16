# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence residual model

- val count_exact_acc: **37.26%**
- val close_acc_1: **76.68%**
- val close_acc_2: **91.35%**
- val count_mae: **1.012**
- val change_acc: **92.07%**

## Temporal residual model

- val start_exact_acc: **67.11%**
- val start_tol_5m: **69.30%**
- val start_tol_10m: **71.53%**
- val start_mae_minutes: **116.76 min**
- val duration_mae_minutes: **0.00 min**

## Weekly backtest

- task_f1: **87.76%**
- time_exact_accuracy: **7.09%**
- time_close_accuracy_5m: **7.94%**
- time_close_accuracy_10m: **9.40%**
- start_mae_minutes: **1516.57 min**
- overlap_same_robot_count: **1**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
