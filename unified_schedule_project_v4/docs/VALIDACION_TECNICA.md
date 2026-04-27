# Validación técnica realizada

Se ha ejecutado una validación técnica mínima para comprobar que el proyecto es operativo.

## Entrenamiento smoke

Comando ejecutado:

```bash
PYTHONPATH=src python scripts/train.py --registry /tmp/unified_registry.yaml --output-dir /tmp/unified_smoke --smoke
```

Resultado:
- entrenamiento completado sin errores,
- checkpoint guardado en `unified_model.pt`,
- resumen generado (`summary.json`),
- informe final generado (`final_report.md`).

## Predicción smoke

Comando ejecutado:

```bash
PYTHONPATH=src python scripts/predict_week.py   --registry /tmp/unified_registry.yaml   --output-dir /tmp/unified_smoke   --database-id nexus_5y   --robot-id nexus-r0b0t-0001-4aaa-8bbb-c0mplex001x
```

Resultado:
- predicción JSON generada correctamente.

## Importante

Esta validación es **técnica**, no de rendimiento real. Sirve para verificar que:

- el pipeline carga datos,
- construye contexto,
- entrena,
- serializa el modelo,
- recarga el checkpoint,
- y produce una predicción futura.

No demuestra todavía superioridad frente al proyecto original. Para eso hace falta entrenamiento completo y comparación de backtest homogénea.
