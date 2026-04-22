# Actualización v4: features posicionales, densidad y soporte ordinal

Esta versión amplía el proyecto para que el modelo temporal no vea solo el `anchor_start_bin`, sino también el contexto ordinal, la densidad local y la estabilidad histórica del slot.

## Lo añadido al modelo temporal

### 1. Posición intra-task
- `task_position_abs`
- `task_position_pct`
- `is_first_of_type`
- `is_last_of_type`
- `is_middle_of_type`

### 2. Posición global dentro de la semana
- `week_order_abs`
- `week_order_pct`
- `num_events_before`
- `num_events_after`

### 3. Posición dentro del día
- `day_order_abs`
- `day_order_pct`
- `same_task_day_order_abs`
- `same_task_day_order_pct`

### 4. Separaciones y densidad
- `gap_prev_anchor_bins`
- `gap_next_anchor_bins`
- `has_prev_anchor`
- `has_next_anchor`
- `num_events_in_prev_60m`
- `num_events_in_next_60m`
- `num_same_task_in_prev_day`
- `num_same_task_in_next_day`

### 5. Soporte histórico dependiente del ordinal
- `ordinal_support`
- `ordinal_start_median_norm`
- `ordinal_start_std_norm`
- `ordinal_day_mode_norm`
- `ordinal_day_entropy`
- `ordinal_active_weeks_norm`

### 6. Interacciones posición × carga / soporte / dispersión
- `position_x_pred_task_count`
- `position_x_template_task_count`
- `position_x_slot_support`
- `position_x_task_dispersion`

### 7. Señales temporales adicionales del candidato
- seno/coseno del día de semana del candidato
- seno/coseno de la hora local del candidato
- desviación del candidato respecto al ordinal histórico

## Lo añadido al modelo de occurrence

Se amplió el vector numérico con señales del documento de feature engineering:
- lags adicionales (`lag_1`, `lag_4`, `lag_26`, `lag_52`)
- rolling stats de 4 y 12 semanas
- `yoy_delta`, `lag_52_ratio`
- estacionalidad explícita (`sin/cos` de semana y mes)
- banderas de pico / valle
- `dist_to_peak_month`
- `monthly_load_index`
- estructura semanal (`n_weekdays_in_week`, `n_weekend_days_in_week`, `expected_night_tasks`)
- tendencia (`weeks_elapsed_total`, `trend_tasks_per_week`)

## Cambios de arquitectura / configuración

- Se amplió `max_slot_prototypes` a 48.
- Se amplió la exploración temporal a más templates y más candidatos.
- Se incrementó la capacidad del ranker temporal para absorber el nuevo bloque de features.
- Los checkpoints antiguos quedan invalidados: esta versión exige reentrenar para que las dimensiones coincidan.

## Compatibilidad

Los scripts de predicción ahora verifican `feature_schema_version`. Si intentas usar un checkpoint previo a esta actualización, el proyecto devuelve un error explícito y te obliga a reentrenar, evitando inferencias corruptas por mismatch de features.
