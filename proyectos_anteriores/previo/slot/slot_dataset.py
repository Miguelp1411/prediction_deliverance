"""
slot_dataset.py
───────────────
Representación basada en SLOTS en lugar de secuencias.

Un slot = (task_id, day_of_week).
Ejemplo: ("Fregar suelo", lunes) es un slot.

Para cada ventana de WINDOW semanas construimos:
  X: (WINDOW, K, 6)  →  K slots × 6 features del contexto
  y: (K, 4)          →  para cada slot: [ocurre?, hour_norm, min_norm, dur_norm]

Ventajas sobre Seq2Seq:
  ✓ Elimina el problema de orden (no hay posición N que memorizar)
  ✓ Predicción paralela: no hay error compounding
  ✓ El modelo aprende "los martes a las 9h suele haber tarea X"
  ✓ Métricas más interpretables (F1 de slot, no acc de posición)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

WINDOW        = 4   # semanas de contexto
MIN_SLOT_FREQ = 2   # el slot debe aparecer al menos N veces en todo el dataset
N_MIN_BINS    = 12  # bins de 5 minutos (0-55 min → bins 0-11)


class SlotWeekDataset(Dataset):
    """
    Dataset donde cada muestra es (contexto_4_semanas, target_semana).
    Ambos representados como tensores de slots, no secuencias.
    """

    def __init__(self, semanas_separadas, window=WINDOW, min_slot_freq=MIN_SLOT_FREQ):
        all_data = pd.concat(semanas_separadas, ignore_index=True)
        all_data = all_data.sort_values('start_time').reset_index(drop=True)

        # ── 1. Descubrir slots frecuentes ─────────────────────────────────────
        slot_counts = all_data.groupby(['task_id', 'day_of_week']).size()
        slot_series = slot_counts[slot_counts >= min_slot_freq]
        slot_list   = list(zip(slot_series.index.get_level_values('task_id'),
                               slot_series.index.get_level_values('day_of_week')))

        self.slots        = slot_list                                  # [(task_id, day), ...]
        self.num_slots    = len(slot_list)
        self.slot_to_idx  = {s: i for i, s in enumerate(slot_list)}

        # ── 2. Estadísticas para normalización ───────────────────────────────
        self.dur_mean = float(all_data['duration_mins'].mean())
        self.dur_std  = float(all_data['duration_mins'].std()) + 1e-6

        # Frecuencia histórica de cada slot (prior para la loss de ocurrencia)
        self.slot_freq = np.zeros(self.num_slots, dtype=np.float32)
        total_weeks = len(semanas_separadas)
        for i, slot in enumerate(self.slots):
            task_id, day = slot
            weeks_active = sum(
                1 for w in semanas_separadas
                if ((w['task_id'] == task_id) & (w['day_of_week'] == day)).any()
            )
            self.slot_freq[i] = weeks_active / total_weeks

        # ── 3. Construir tensores ─────────────────────────────────────────────
        X_list, y_list = [], []

        for i in range(len(semanas_separadas) - window):
            ctx_weeks = [semanas_separadas[i + j] for j in range(window)]
            tgt_week  = semanas_separadas[i + window]

            # Contexto: (WINDOW, K, 6)
            # Features: [ocurre, hour_norm, min_norm, dur_norm, recency, slot_freq]
            ctx_tensor = np.stack([
                self._encode_week(w, recency=(j + 1) / window)
                for j, w in enumerate(ctx_weeks)
            ])

            # Target: (K, 4) → [ocurre, hour_norm, min_norm, dur_norm]
            tgt_tensor = self._encode_week(tgt_week, recency=1.0)[:, :4]

            X_list.append(ctx_tensor)
            y_list.append(tgt_tensor)

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)  # (N, WINDOW, K, 6)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32)  # (N, K, 4)

        print(f"SlotWeekDataset: {len(self.X)} ventanas | {self.num_slots} slots")
        active_slots = (self.slot_freq > 0.3).sum()
        print(f"  Slots activos >30% del tiempo: {active_slots}/{self.num_slots}")

    def _encode_week(self, week_df, recency=1.0):
        """
        Codifica una semana como (num_slots, 6).

        Features por slot:
          [0] ocurre       — 1 si la tarea aparece este día esta semana
          [1] hour_norm    — hora / 23
          [2] min_norm     — minuto / 59
          [3] dur_norm     — (dur - mean) / std
          [4] recency      — qué tan reciente es esta semana (0=antigua, 1=reciente)
          [5] slot_freq    — frecuencia histórica del slot (prior)
        """
        result = np.zeros((self.num_slots, 6), dtype=np.float32)
        result[:, 4] = recency
        result[:, 5] = self.slot_freq  # prior histórico

        # Ordenar por tiempo para quedarnos con la primera ocurrencia del slot
        week_sorted = week_df.sort_values('start_time')

        seen_slots = set()
        for _, row in week_sorted.iterrows():
            key = (int(row['task_id']), int(row['day_of_week']))
            if key not in self.slot_to_idx or key in seen_slots:
                continue

            seen_slots.add(key)
            idx = self.slot_to_idx[key]
            result[idx, 0] = 1.0
            result[idx, 1] = row['hour'] / 23.0
            result[idx, 2] = row['minute'] / 59.0
            result[idx, 3] = (row['duration_mins'] - self.dur_mean) / self.dur_std

        return result

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
