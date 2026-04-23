# Actualización temporal jerárquica

Se ha sustituido el decoder temporal plano por un decoder jerárquico en tres niveles:

1. `day_head`: predice el día de la semana objetivo (0..6).
2. `macroblock_head`: predice el bloque horario dentro del día. En esta versión se ha elegido la variante de bloques de **60 minutos** (`macroblock_minutes: 60`).
3. `fine_offset_head`: predice el bin fino de 5 minutos dentro del macrobloque.

## Motivo

Las métricas anteriores mostraban errores gruesos de varias horas. Por eso se cambió la formulación desde un único head de tiempo a una predicción jerárquica:

- primero estructura gruesa (`day_head`),
- luego franja horaria (`macroblock_head`),
- y por último ajuste fino (`fine_offset_head`).

## Cambios aplicados

- `models/temporal_residual.py`
  - nuevo decoder con `day_logits`, `macroblock_logits` y `fine_offset_logits`.
- `training/datasets.py`
  - nuevos targets jerárquicos: `day_target`, `macroblock_target`, `fine_offset_target`.
- `training/losses.py`
  - nueva loss temporal = CE(día) + CE(macrobloque) + CE(offset fino) + SmoothL1(duración).
- `evaluation/metrics.py`
  - la reconstrucción del `start_bin` ahora se hace componiendo día + macrobloque + offset fino.
- `inference/predictor.py`
  - generación jerárquica de candidatos `topk_day × topk_macro × topk_fine`.
- `configs/default.yaml`
  - nuevos parámetros: `macroblock_minutes`, `topk_macro`, `topk_fine`.
- `scripts/train.py` y `scripts/predict_week.py`
  - instanciación del nuevo temporal jerárquico.

## Validación técnica realizada

Se validó que:

- el dataset temporal genera correctamente los targets jerárquicos,
- el modelo produce salidas con shapes coherentes,
- la loss hace `backward()` sin errores,
- la inferencia y la reconstrucción de bins usan la nueva jerarquía temporal.
