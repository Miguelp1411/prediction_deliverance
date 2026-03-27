# data/dataset.py
# WeekDataset: convierte las semanas en tensores (X, y, máscaras) listos para entrenar.

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from config import WINDOW, MAX_TASKS, N_MIN_BINS


def _encode_cyclic(value: np.ndarray, max_value: int) -> tuple[np.ndarray, np.ndarray]:
    """Codificación cíclica seno/coseno para features periódicas."""
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


class WeekDataset(Dataset):
    """
    Dado una lista de DataFrames (uno por semana), genera ventanas de tamaño
    `window` como entrada y la semana siguiente como objetivo.

    Cada semana se representa como un tensor (MAX_TASKS, 8):
        [task_id, sin_day, cos_day, sin_hour, cos_hour, sin_min, cos_min, duration_norm]
    """

    def __init__(self, semanas_separadas: list, window: int = WINDOW):
        # Ajuste global de duración con MinMaxScaler
        all_durations = np.concatenate(
            [s["duration_mins"].values for s in semanas_separadas]
        ).reshape(-1, 1)
        self.scaler = MinMaxScaler()
        self.scaler.fit(all_durations)

        X_list, y_list, mask_X_list, mask_y_list = [], [], [], []

        for i in range(len(semanas_separadas) - window):
            entrada   = [self._to_tensor(semanas_separadas[i + j]) for j in range(window)]
            masks_in  = [self._to_mask(semanas_separadas[i + j])   for j in range(window)]
            objetivo  = self._to_tensor(semanas_separadas[i + window])
            mask_out  = self._to_mask(semanas_separadas[i + window])

            X_list.append(np.stack(entrada))
            mask_X_list.append(np.stack(masks_in))
            y_list.append(objetivo)
            mask_y_list.append(mask_out)

        self.X       = torch.tensor(np.array(X_list),      dtype=torch.float32)
        self.masks_X = torch.tensor(np.array(mask_X_list), dtype=torch.bool)
        self.y       = torch.tensor(np.array(y_list),      dtype=torch.float32)
        self.masks_y = torch.tensor(np.array(mask_y_list), dtype=torch.bool)

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _to_tensor(self, semana_df, max_tasks: int = MAX_TASKS) -> np.ndarray:
        cols = ["task_id", "day_of_week", "hour", "minute", "duration_mins"]
        rows = semana_df[cols].values.copy().astype(np.float32)[:max_tasks]

        task_ids = (rows[:, 0] + 1).reshape(-1, 1)              # offset: 0 reservado para padding

        sin_day,  cos_day  = _encode_cyclic(rows[:, 1], 7)      # día de semana (0-6)
        sin_hour, cos_hour = _encode_cyclic(rows[:, 2], 24)     # hora (0-23)

        minute_bin         = rows[:, 3] // 5                    # bin de 5 min → 0-11
        sin_min,  cos_min  = _encode_cyclic(minute_bin, N_MIN_BINS)

        dur_norm = self.scaler.transform(rows[:, 4].reshape(-1, 1)).ravel()

        result = np.column_stack([
            task_ids,
            sin_day,  cos_day,
            sin_hour, cos_hour,
            sin_min,  cos_min,
            dur_norm,
        ]).astype(np.float32)

        # Padding con ceros hasta MAX_TASKS
        pad_len = max_tasks - len(result)
        if pad_len > 0:
            result = np.vstack([result, np.zeros((pad_len, 8), dtype=np.float32)])

        return result  # (MAX_TASKS, 8)

    def _to_mask(self, semana_df, max_tasks: int = MAX_TASKS) -> np.ndarray:
        """True en posiciones reales, False en padding."""
        mask = np.zeros(max_tasks, dtype=bool)
        mask[: min(len(semana_df), max_tasks)] = True
        return mask

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.masks_X[idx], self.masks_y[idx]
