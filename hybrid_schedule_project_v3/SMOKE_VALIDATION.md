# Smoke validation

Se ejecutó una prueba de humo sobre el pipeline de entrenamiento:

```bash
PYTHONPATH=src python scripts/train.py   --registry examples/registry.yaml   --output-dir /tmp/hybrid_smoke   --smoke
```

Resultado: ejecución completada con `EXIT:0`.

Se verificó que:
- el entrenamiento de occurrence arranca y produce métricas,
- el entrenamiento temporal arranca y produce métricas,
- se generan checkpoints,
- se genera `summary.json`,
- se genera `backtest_weekly_metrics.csv`.

La validación rápida usó parámetros reducidos y backtest simplificado de humo. No sustituye un entrenamiento completo ni una comparación final frente a `project_06_v4.2`.
