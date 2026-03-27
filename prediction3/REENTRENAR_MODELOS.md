# Reentrenado obligatorio tras el cambio a `joint_start_bin`

Este proyecto ya está migrado para que el modelo temporal prediga un único `start_bin` semanal conjunto en lugar de separar `día` y `hora`.

## Qué implica

Los checkpoints temporales antiguos dejan de ser compatibles con el código nuevo porque la capa de salida cambia de:

- `day_head + time_head`

a:

- `start_head`

## Qué hacer

1. Reentrena el proyecto con `train.py`.
2. Usa los nuevos checkpoints generados en `checkpoints/`.
3. Después ejecuta `predict.py` normalmente.

## Archivos afectados por esta migración

- `config.py`
- `models/temporal_model.py`
- `data/datasets.py`
- `training/losses.py`
- `training/metrics.py`
- `predict.py`
- `train.py`
- `data/preprocessing.py`

## Protección añadida

`predict.py` detecta checkpoints temporales antiguos y lanza un error claro indicando que hay que reentrenar, en vez de fallar con un `load_state_dict` confuso.
