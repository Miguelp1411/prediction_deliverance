import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

WINDOW = 4          # semanas de contexto
MAX_TASKS = 60      # máximo de tareas por semana (ajusta según tus datos)
PAD_TOKEN = -1      # valor de relleno

def semanas_a_tensor(semana_df, max_tasks=MAX_TASKS):
    """Convierte un DataFrame de semana a tensor numpy (MAX_TASKS, 5)"""
    feats = semana_df[['task_id', 'day_of_week', 'hour', 'minute', 'duration_mins']].values
    
    # Truncamos si hay más tareas que MAX_TASKS
    feats = feats[:max_tasks]
    
    # Padding con ceros hasta MAX_TASKS
    pad_len = max_tasks - len(feats)
    if pad_len > 0:
        padding = np.zeros((pad_len, 5))
        feats = np.vstack([feats, padding])
    
    return feats.astype(np.float32)

