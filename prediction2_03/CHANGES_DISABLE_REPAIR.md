Cambios: desactivar repair en todo el flujo
=========================================

Se ha dejado el proyecto para que no aplique repair en ningún momento.

Cambios principales
-------------------
- `predict.py`
  - La predicción CLI llama siempre a `predict_next_week(..., use_repair=False)`.
- `train.py`
  - Los resúmenes de train y validación se evalúan con `use_repair=False`.
  - El resumen del corrector auxiliar también evalúa base / c+d / full sin repair.
- `auxiliary_corrector.py`
  - El replay histórico para entrenar el corrector usa `use_repair=False`.
  - `AuxiliaryCorrector.apply(..., apply_repair=False)` queda desactivado por defecto.
- `evaluate_auxiliary.py`
  - Se elimina la comparativa con repair.
  - La tabla compara solo:
    - base/sin repair
    - aux c+d/sin repair
    - aux full/sin repair
  - Las métricas de repair quedan a 0.0 para dejar explícito que no se aplicó.

Resultado esperado
------------------
- Ya no aparecerá la mezcla confusa de train OFF y validación ON.
- Todo el pipeline de evaluación, benchmark e inferencia trabaja sin repair.
