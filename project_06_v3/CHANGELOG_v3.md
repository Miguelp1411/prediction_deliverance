# project_06 v3

Cambios principales de esta versión regenerada:

- Soporte de evaluación **resource-aware**: los overlaps se calculan tanto a nivel global como por `device_uid`.
- Preservación de `device_uid` desde la carga del JSON cuando exista.
- Métricas ampliadas para el `OccurrenceModel`: `presence_precision`, `presence_recall` y `presence_f1`.
- Activación por defecto de `PREDICTION_USE_REPAIR=True`.
- Nuevos flags de configuración: `SCHEDULING_MODE`, `OVERLAP_SCOPE`, `REPORTS_DIR`, `ANALYSIS_DIR`.
- Script `scripts/analyze_dataset.py` para perfilar el dataset antes de entrenar.

Nota: la arquitectura base del proyecto original se mantiene. Esta versión prioriza observabilidad, semántica correcta de overlaps y preparación para escenarios single-device y multi-device.
