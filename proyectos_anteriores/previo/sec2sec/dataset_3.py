import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

WINDOW      = 4
MAX_TASKS   = 60
N_MIN_BINS  = 12

def _encode_cyclic(value, max_value):
    """Convierte un valor a sus componentes seno/coseno"""
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


class WeekDataset(Dataset):
    def __init__(self, semanas_separadas, window=WINDOW):
        all_durations = np.concatenate(
            [s['duration_mins'].values for s in semanas_separadas]
        ).reshape(-1, 1)
        self.scaler = MinMaxScaler()
        self.scaler.fit(all_durations)

        X_list, y_list, mask_X_list, mask_y_list = [], [], [], []

        for i in range(len(semanas_separadas) - window):
            entrada  = [self._to_tensor(semanas_separadas[i + j]) for j in range(window)]
            masks_in = [self._to_mask(semanas_separadas[i + j])   for j in range(window)]
            objetivo = self._to_tensor(semanas_separadas[i + window])
            mask_out = self._to_mask(semanas_separadas[i + window])

            X_list.append(np.stack(entrada))
            mask_X_list.append(np.stack(masks_in))
            y_list.append(objetivo)
            mask_y_list.append(mask_out)

        self.X       = torch.tensor(np.array(X_list),      dtype=torch.float32)
        self.masks_X = torch.tensor(np.array(mask_X_list), dtype=torch.bool)
        self.y       = torch.tensor(np.array(y_list),      dtype=torch.float32)
        self.masks_y = torch.tensor(np.array(mask_y_list), dtype=torch.bool)

    def _to_tensor(self, semana_df, max_tasks=MAX_TASKS):
        cols = ['task_id', 'day_of_week', 'hour', 'minute', 'duration_mins']
        rows = semana_df[cols].values.copy().astype(np.float32)
        rows = rows[:max_tasks]

        task_ids = (rows[:, 0] + 1).reshape(-1, 1)          # task_id con offset

        sin_day, cos_day     = _encode_cyclic(rows[:, 1], 7)   # día (0-6)
        sin_hour, cos_hour   = _encode_cyclic(rows[:, 2], 24)  # hora (0-23)
        
        # minuto a bin ANTES de codificar cíclicamente
        minute_bin           = (rows[:, 3] // 5)              # bin 0-11
        sin_min, cos_min     = _encode_cyclic(minute_bin, N_MIN_BINS)

        dur_norm = self.scaler.transform(rows[:, 4].reshape(-1, 1)).ravel()

        # Apilamos: [task_id, sin_day, cos_day, sin_hour, cos_hour, sin_min, cos_min, duration]
        result = np.column_stack([
            task_ids,
            sin_day, cos_day,
            sin_hour, cos_hour,
            sin_min, cos_min,
            dur_norm
        ]).astype(np.float32)

        pad_len = max_tasks - len(result)
        if pad_len > 0:
            result = np.vstack([result, np.zeros((pad_len, 8), dtype=np.float32)])

        return result  # (MAX_TASKS, 8)  ← ahora son 8 features

    def _to_mask(self, semana_df, max_tasks=MAX_TASKS):
        n = min(len(semana_df), max_tasks)
        mask = np.zeros(max_tasks, dtype=bool)
        mask[:n] = True
        return mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks_X[idx], self.masks_y[idx]