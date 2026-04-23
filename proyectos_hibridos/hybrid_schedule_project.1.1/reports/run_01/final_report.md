# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence model (causal + ensemble)

- selector count_exact_acc: **18.75%**
- selector close_acc_1: **25.00%**
- selector close_acc_2: **40.62%**
- selector count_mae: **3.344**
- selector score: **-1.562**

## Temporal model (matching bipartito + día/hora absolutos)

- selector start_exact_acc: **34.72%**
- selector start_tol_5m: **35.01%**
- selector start_tol_10m: **37.06%**
- selector start_mae_minutes: **224.48 min**
- selector score: **104.250**

## Weekly holdout backtest

- task_f1: **90.78%**
- time_exact_accuracy: **45.65%**
- time_close_accuracy_5m: **46.75%**
- time_close_accuracy_10m: **48.15%**
- start_mae_minutes: **247.90 min**
- overlap_same_robot_count: **0**

## Leave-one-database-out

- **nexus_10y**: task_f1=90.78% | time_close_10m=48.15% | start_mae=247.90 min

## Interpretación

La versión final elimina fuga de información en retrieval y temporal, selecciona checkpoints por backtest walk-forward real, construye la plantilla semanal por consenso causal de vecinos, combina conteos con un ensemble histórico+estacional+residual, predice tiempos como día/hora absolutos con matching bipartito y resuelve la coherencia final con un scheduler exacto compatible con CP-SAT.
