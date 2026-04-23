# Mejoras implementadas en la versión causal v2

## 1. Limpieza causal del pipeline

- El retrieval de semanas similares ya no usa información de la semana objetivo real durante train/val.
- La firma objetivo se construye solo con contexto histórico anterior y calendario de la semana a predecir.
- El dataset temporal ya no introduce `series.counts[target_week_idx, task_idx]` ni ninguna feature equivalente derivada del target real.
- La selección de checkpoints se hace por **walk-forward backtest real**:
  - `run_occurrence_selector_backtest()` para conteos.
  - `run_schedule_selector_backtest()` para la agenda completa.

## 2. Bloque temporal reformulado

- Se elimina la formulación residual limitada por ventana local.
- El temporal ahora predice **día de la semana absoluto** y **hora intra-día absoluta**.
- Se elimina la cabeza de duración aprendida; la duración se obtiene como prior determinista/estable por tarea.
- El dataset temporal usa **matching bipartito** (`pair_template_slots_to_events`) para alinear plantilla y target.

## 3. Plantilla coherente por consenso

- La plantilla semanal ya no copia simplemente la semana primaria.
- `build_template_week()` construye:
  - conteos de consenso por promedio ponderado de vecinos,
  - slots de consenso por cuantiles ponderados sobre los vecinos recuperados,
  - soporte por slot para usarlo como prior.

## 4. Conteos con ensemble causal

- El predictor mezcla:
  - plantilla histórica,
  - baseline estacional,
  - media reciente,
  - salida del modelo residual.
- También expone intervalo de conteos (`count_low`, `count_high`) a partir de la distribución del modelo.

## 5. Preparación multi-base

- Muestreo balanceado por base/robot en occurrence.
- Adaptadores ligeros condicionados por base/robot dentro de las redes (`FeatureAdapter`).
- Backtest `leave-one-database-out` disponible.

## 6. Scheduler mejorado

- `solve_week_schedule()` intenta resolver con **CP-SAT** si `ortools` está instalado.
- Si no está disponible, cae a MILP de SciPy y, en último caso, a greedy.
- Se añaden agendas alternativas plausibles en la explicación de predicción.

## 7. Incertidumbre útil

- Intervalos de conteo por tarea.
- Confianza local por slot temporal.
- Varias agendas alternativas en `prediction_explanation.json`.

## 8. Scripts y artefactos

- `scripts/train.py` entrena la nueva versión causal y guarda:
  - checkpoints,
  - métricas por época,
  - backtest holdout,
  - leave-one-database-out,
  - informe final.
- `scripts/predict_week.py` genera la semana futura con la nueva arquitectura.
