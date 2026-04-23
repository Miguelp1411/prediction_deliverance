# Final training report

## Dataset summary

- eventos: **32732**
- bases de datos: **1**
- robots: **1**
- semanas: **522**
- overlap histórico mismo robot: **0**

## Occurrence model (causal + ensemble)

- selector count_exact_acc: **0.00%**
- selector close_acc_1: **25.00%**
- selector close_acc_2: **25.00%**
- selector count_mae: **3.875**
- selector score: **-23.750**

## Temporal model (matching bipartito + día/hora absolutos)

- selector start_exact_acc: **0.00%**
- selector start_tol_5m: **0.91%**
- selector start_tol_10m: **1.56%**
- selector start_mae_minutes: **187.04 min**
- selector score: **88.778**

## Weekly holdout backtest

- task_f1: **85.27%**
- time_exact_accuracy: **21.00%**
- time_close_accuracy_5m: **22.39%**
- time_close_accuracy_10m: **27.22%**
- start_mae_minutes: **242.99 min**
- overlap_same_robot_count: **2**

## Leave-one-database-out

- **nexus_10y**: task_f1=84.50% | time_close_10m=45.64% | start_mae=148.58 min

## Interpretación

La versión final elimina fuga de información en retrieval y temporal, selecciona checkpoints por backtest walk-forward real, construye la plantilla semanal por consenso causal de vecinos, combina conteos con un ensemble histórico+estacional+residual, predice tiempos como día/hora absolutos con matching bipartito y resuelve la coherencia final con un scheduler exacto compatible con CP-SAT.
