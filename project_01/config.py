# config.py
# Todas las constantes del proyecto en un único lugar.
# Importa desde aquí en cualquier otro módulo para evitar imports circulares.

WINDOW          = 12    # semanas de contexto que ve el encoder
MAX_TASKS       = 45   # máximo de tareas por semana (padding target)
N_MIN_BINS      = 12   # bins de minuto (cada bin = 5 min → 60/5 = 12)
N_NUMERIC_FEATS = 7    # sin_day, cos_day, sin_hour, cos_hour, sin_min, cos_min, duration
