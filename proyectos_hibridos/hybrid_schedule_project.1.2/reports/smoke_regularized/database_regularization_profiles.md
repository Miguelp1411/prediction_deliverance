# Perfiles de regularización por dataset

Estos perfiles se derivan del tamaño histórico, volatilidad semanal y rigidez temporal de cada base.

## aux_database

- eventos / semanas / robots: **8975 / 104 / 2**
- escasez / volatilidad / rigidez: **0.394 / 0.213 / 0.656**
- ruido de entrada: history_dropout=0.067, history_noise=0.012, feature_dropout=0.059, feature_noise=0.009
- occurrence: label_smoothing=0.038, delta_sigma=1.016, shrink=0.062, confidence_penalty=0.0036
- temporal: day_smoothing=0.034, time_sigma=1.926, anchor=0.090, day_prior=0.139, time_prior=0.160, confidence_penalty=0.0033, smoothness=0.0016

## aux_database_1

- eventos / semanas / robots: **4712 / 104 / 1**
- escasez / volatilidad / rigidez: **0.427 / 0.335 / 0.748**
- ruido de entrada: history_dropout=0.071, history_noise=0.013, feature_dropout=0.063, feature_noise=0.010
- occurrence: label_smoothing=0.045, delta_sigma=1.169, shrink=0.068, confidence_penalty=0.0039
- temporal: day_smoothing=0.038, time_sigma=2.265, anchor=0.095, day_prior=0.152, time_prior=0.174, confidence_penalty=0.0035, smoothness=0.0019

## aux_database_2

- eventos / semanas / robots: **4263 / 104 / 1**
- escasez / volatilidad / rigidez: **0.432 / 0.000 / 0.924**
- ruido de entrada: history_dropout=0.072, history_noise=0.013, feature_dropout=0.063, feature_noise=0.010
- occurrence: label_smoothing=0.032, delta_sigma=0.750, shrink=0.076, confidence_penalty=0.0040
- temporal: day_smoothing=0.032, time_sigma=1.432, anchor=0.131, day_prior=0.171, time_prior=0.195, confidence_penalty=0.0035, smoothness=0.0013

## nexus_schedule_10years

- eventos / semanas / robots: **32732 / 522 / 1**
- escasez / volatilidad / rigidez: **0.115 / 0.998 / 0.158**
- ruido de entrada: history_dropout=0.034, history_noise=0.005, feature_dropout=0.031, feature_noise=0.003
- occurrence: label_smoothing=0.056, delta_sigma=1.997, shrink=0.027, confidence_penalty=0.0014
- temporal: day_smoothing=0.036, time_sigma=3.610, anchor=0.030, day_prior=0.065, time_prior=0.078, confidence_penalty=0.0013, smoothness=0.0026

## nexus_schedule_5years

- eventos / semanas / robots: **16368 / 261 / 1**
- escasez / volatilidad / rigidez: **0.255 / 0.999 / 0.159**
- ruido de entrada: history_dropout=0.051, history_noise=0.008, feature_dropout=0.046, feature_noise=0.006
- occurrence: label_smoothing=0.063, delta_sigma=1.999, shrink=0.034, confidence_penalty=0.0025
- temporal: day_smoothing=0.043, time_sigma=3.753, anchor=0.030, day_prior=0.076, time_prior=0.090, confidence_penalty=0.0023, smoothness=0.0029

## robot_schedule_5years

- eventos / semanas / robots: **19542 / 261 / 2**
- escasez / volatilidad / rigidez: **0.240 / 0.606 / 0.481**
- ruido de entrada: history_dropout=0.049, history_noise=0.008, feature_dropout=0.044, feature_noise=0.006
- occurrence: label_smoothing=0.046, delta_sigma=1.507, shrink=0.047, confidence_penalty=0.0024
- temporal: day_smoothing=0.034, time_sigma=2.754, anchor=0.060, day_prior=0.109, time_prior=0.127, confidence_penalty=0.0022, smoothness=0.0021

## robot_schedule_gamma

- eventos / semanas / robots: **72495 / 111 / 15**
- escasez / volatilidad / rigidez: **0.160 / 0.263 / 0.834**
- ruido de entrada: history_dropout=0.039, history_noise=0.006, feature_dropout=0.036, feature_noise=0.004
- occurrence: label_smoothing=0.029, delta_sigma=1.078, shrink=0.060, confidence_penalty=0.0018
- temporal: day_smoothing=0.023, time_sigma=1.817, anchor=0.107, day_prior=0.142, time_prior=0.163, confidence_penalty=0.0016, smoothness=0.0012

## robot_schedule_v4_2025_2026

- eventos / semanas / robots: **2356 / 52 / 1**
- escasez / volatilidad / rigidez: **0.525 / 0.341 / 0.750**
- ruido de entrada: history_dropout=0.083, history_noise=0.015, feature_dropout=0.072, feature_noise=0.011
- occurrence: label_smoothing=0.050, delta_sigma=1.177, shrink=0.072, confidence_penalty=0.0047
- temporal: day_smoothing=0.043, time_sigma=2.378, anchor=0.095, day_prior=0.159, time_prior=0.182, confidence_penalty=0.0042, smoothness=0.0021
