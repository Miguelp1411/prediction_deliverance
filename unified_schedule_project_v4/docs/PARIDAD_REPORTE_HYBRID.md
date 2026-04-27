# Paridad de reporte con el proyecto híbrido

Este ajuste mantiene la arquitectura unificada, pero añade el bloque de evaluación necesario para que el informe final exponga las mismas métricas agregadas de backtest semanal que el proyecto híbrido.

## Cambios realizados

1. Se añadió `splits.backtest_weeks` al config.
2. Se incorporó `training/backtesting.py` con evaluación holdout semana a semana.
3. Se añadió `evaluate_week(...)` en `evaluation/metrics.py` usando el mismo matching y las mismas métricas agregadas que en el híbrido.
4. `scripts/train.py` ahora ejecuta el backtest holdout y guarda `summary['backtest']`.
5. `report_builder.py` ahora incluye la sección `Weekly backtest` con:
   - `task_f1`
   - `time_exact_accuracy`
   - `time_close_accuracy_5m`
   - `time_close_accuracy_10m`
   - `start_mae_minutes`
   - `overlap_same_robot_count`
6. Se guarda `backtest_weekly_metrics.csv` en el directorio de salida.

## Importante

La validación original del modelo unificado a nivel de slots se conserva. El nuevo bloque de backtest es adicional y sirve para comparar ambos proyectos sobre una base homogénea.
