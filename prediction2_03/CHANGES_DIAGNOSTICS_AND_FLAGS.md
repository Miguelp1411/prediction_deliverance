# Cambios aplicados: flags separados, benchmark 4 escenarios y métricas diagnósticas

## 1. Flags separados del corrector auxiliar

Se ha desacoplado la aplicación del corrector auxiliar en tres componentes independientes:

- `aux_correct_count=True`
- `aux_correct_duration=True`
- `aux_correct_time=False` por defecto

### Archivos afectados
- `auxiliary_corrector.py`
- `predict.py`

### Efecto
- En inferencia, el corrector temporal fino queda desactivado por defecto.
- El corrector de conteo y duración sigue activo por defecto.
- `predict.py` admite ahora:
  - `--disable-aux-count`
  - `--enable-aux-time`
  - `--disable-aux-duration`

## 2. Benchmark reestructurado por dataset

`evaluate_auxiliary.py` ahora calcula y presenta, por dataset, estos cuatro escenarios:

- `base / sin repair`
- `base / con repair`
- `aux count+dur / con repair`
- `aux completo / con repair`

Además calcula también los escenarios auxiliares sin repair para poder medir la mejora del auxiliar antes y después del repair.

## 3. Métricas diagnósticas añadidas

`evaluation/weekly_stats.py` ahora devuelve también:

- `day_exact_accuracy`
- `day_close_accuracy_1d`
- `start_mae_when_day_correct_minutes`

`evaluate_auxiliary.py` añade también métricas específicas del repair:

- `repair_moved_fraction`
- `repair_avg_displacement_minutes`

Y resume explícitamente la mejora del auxiliar:

- antes del repair
- después del repair

## 4. Salida de entrenamiento enriquecida

`train.py` ahora muestra en los resúmenes semanales:

- día exacto
- día ±1
- MAE de inicio condicionado a día correcto

Y en la validación del corrector compara:

- base
- auxiliar count+duration
- auxiliar full
