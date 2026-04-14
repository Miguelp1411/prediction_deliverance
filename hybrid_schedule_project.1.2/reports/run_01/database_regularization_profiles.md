# Perfiles de regularización por dataset

Estos perfiles se derivan del tamaño histórico, volatilidad semanal y rigidez temporal de cada base.

## nexus_10y

- eventos / semanas / robots: **32732 / 522 / 1**
- escasez / volatilidad / rigidez: **0.000 / 0.000 / 1.000**
- ruido de entrada: history_dropout=0.020, history_noise=0.002, feature_dropout=0.020, feature_noise=0.001
- occurrence: label_smoothing=0.010, delta_sigma=0.750, shrink=0.060, confidence_penalty=0.0005
- temporal: day_smoothing=0.010, time_sigma=1.000, anchor=0.140, day_prior=0.148, time_prior=0.170, confidence_penalty=0.0005, smoothness=0.0004
