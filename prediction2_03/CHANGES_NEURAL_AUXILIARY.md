# Cambios: corrector auxiliar con red neuronal pequeña

Se sustituyó el corrector auxiliar lineal por una versión neuronal ligera en `auxiliary_corrector.py`.

## Qué cambia

- Los regresores residuales ahora usan `TinyNeuralRegressor` (MLP pequeña en PyTorch).
- Las compuertas de confianza ahora usan `TinyNeuralClassifier` (MLP pequeña en PyTorch).
- Se mantiene la misma API externa:
  - `AuxiliaryCorrector.fit_from_history(...)`
  - `AuxiliaryCorrector.apply(...)`
  - `train_and_save_auxiliary_corrector(...)`
  - `maybe_load_auxiliary_corrector(...)`
  - `AuxiliaryUsagePolicy`
- La policy automática sigue funcionando igual: el auxiliar solo se usa en predicción si mejora validación.

## Arquitectura del auxiliar neuronal

### Conteo
- Regressor MLP: `32 -> 16`
- Gate MLP: `24 -> 12`

### Temporal
- Regressor MLP: `48 -> 24`
- Gates MLP: `32 -> 16`

Todas las redes:
- CPU-friendly
- entrenamiento corto
- inferencia instantánea
- normalización de features
- early stopping simple por pérdida

## Compatibilidad

El checkpoint antiguo del auxiliar lineal **no es compatible** con esta nueva implementación.

Debes reentrenar para generar un nuevo:
- `checkpoints/auxiliary_corrector.pt`
- `checkpoints/auxiliary_policy.json`

## Flujo recomendado

```bash
python train.py --data TU_BASE.json
python predict.py --data TU_BASE.json --output predicted.json
```

## Qué no cambia

- `OccurrenceModel` y `TemporalAssignmentModel` siguen intactos.
- No se reintroduce `repair`.
- La activación en predicción sigue dependiendo de la policy automática guardada tras validación.
