# data/loader.py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from proyectos_anteriores.project_01.config import WINDOW, MAX_TASKS, N_MIN_BINS

# ──────────────────────────────────────────────────────────────────────────────
# Estructura del tensor de 11 columnas
# ──────────────────────────────────────────────────────────────────────────────
#  col 0  : task_id       (int, 1-indexed)          ← entrada del modelo
#  col 1  : sin_day       (float)                   ← entrada del modelo
#  col 2  : cos_day       (float)                   ← entrada del modelo
#  col 3  : sin_hour      (float)                   ← entrada del modelo
#  col 4  : cos_hour      (float)                   ← entrada del modelo
#  col 5  : sin_min       (float)                   ← entrada del modelo
#  col 6  : cos_min       (float)                   ← entrada del modelo
#  col 7  : dur_norm      (float)                   ← entrada del modelo
#  col 8  : day_of_week   (int, 0-6)               ← SOLO para loss/metrics
#  col 9  : hour          (int, 0-23)              ← SOLO para loss/metrics
#  col 10 : minute_bin    (int, 0-11, cada 5 min)  ← SOLO para loss/metrics
# ──────────────────────────────────────────────────────────────────────────────

FEAT_COLS = 8


def _encode_cyclic(value, max_value):
    angle = 2 * np.pi * value / max_value
    return np.sin(angle), np.cos(angle)


class WeekDataset(Dataset):
    """
    Dataset con augmentation opcional para el split de entrenamiento.

    augment=True añade ruido gaussiano pequeño sobre duración y hora en las
    semanas de contexto (X), sin tocar el target (y). Esto evita que el modelo
    memorice secuencias exactas y mejora la generalización con datasets pequeños.

    Parámetros de ruido:
      dur_noise_std  : ruido sobre dur_norm  (default 0.02, ≈ ±2% rango normalizado)
      hour_noise_std : ruido sobre sin/cos hora (default 0.05, ≈ ±7 min)
    """

    def __init__(
        self,
        semanas_separadas,
        window: int = WINDOW,
        augment: bool = False,
        dur_noise_std: float = 0.02,
        hour_noise_std: float = 0.05,
    ):
        self.augment        = augment
        self.dur_noise_std  = dur_noise_std
        self.hour_noise_std = hour_noise_std

        all_durations = np.concatenate(
            [s["duration_mins"].values for s in semanas_separadas]
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
        cols = ["task_id", "day_of_week", "hour", "minute", "duration_mins"]
        rows = semana_df[cols].values.copy().astype(np.float32)[:max_tasks]

        task_ids             = (rows[:, 0] + 1).reshape(-1, 1)
        sin_day,  cos_day   = _encode_cyclic(rows[:, 1], 7)
        sin_hour, cos_hour  = _encode_cyclic(rows[:, 2], 24)
        minute_bin          = rows[:, 3] // 5
        sin_min,  cos_min   = _encode_cyclic(minute_bin, N_MIN_BINS)
        dur_norm = self.scaler.transform(rows[:, 4].reshape(-1, 1)).ravel()

        features = np.column_stack([
            task_ids, sin_day, cos_day, sin_hour, cos_hour, sin_min, cos_min, dur_norm
        ]).astype(np.float32)

        labels = np.column_stack([
            rows[:, 1].reshape(-1, 1),   # col 8 : day_of_week (0-6)
            rows[:, 2].reshape(-1, 1),   # col 9 : hour        (0-23)
            minute_bin.reshape(-1, 1),   # col 10: minute_bin  (0-11)
        ]).astype(np.float32)

        result = np.hstack([features, labels])

        pad_len = max_tasks - len(result)
        if pad_len > 0:
            result = np.vstack([result, np.zeros((pad_len, 11), dtype=np.float32)])

        return result  # (MAX_TASKS, 11)

    def _to_mask(self, semana_df, max_tasks=MAX_TASKS):
        mask = np.zeros(max_tasks, dtype=bool)
        mask[: min(len(semana_df), max_tasks)] = True
        return mask

    def _augment(self, X: torch.Tensor, masks_X: torch.Tensor) -> torch.Tensor:
        """
        Ruido gaussiano sobre semanas de contexto (X), solo posiciones reales.
        NO toca: target (y), etiquetas enteras (cols 8-10), task_id (col 0).
        """
        X    = X.clone()
        real = masks_X.unsqueeze(-1).float()  # (num_weeks, max_tasks, 1)

        # Ruido en duración (col 7)
        dur_noise = torch.randn_like(X[:, :, 7]) * self.dur_noise_std
        X[:, :, 7] = (X[:, :, 7] + dur_noise * real.squeeze(-1)).clamp(0.0, 1.0)

        # Ruido en sin/cos de hora (cols 3-4)
        hour_noise = torch.randn(X.shape[0], X.shape[1], 2) * self.hour_noise_std
        X[:, :, 3:5] = X[:, :, 3:5] + hour_noise * real

        # Re-normalizar al círculo unitario para mantener la estructura cíclica
        norm = X[:, :, 3:5].norm(dim=-1, keepdim=True).clamp(min=1e-6)
        X[:, :, 3:5] = X[:, :, 3:5] / norm

        return X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, y, masks_X, masks_y = (
            self.X[idx], self.y[idx], self.masks_X[idx], self.masks_y[idx]
        )
        if self.augment:
            X = self._augment(X, masks_X)
        return X, y, masks_X, masks_y