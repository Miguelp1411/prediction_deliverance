from __future__ import annotations

import numpy as np
import torch

from proyectos_anteriores.project.config import WINDOW_WEEKS
from proyectos_anteriores.project.data.preprocessing import (
    build_history_features,
    denormalize_duration,
    week_to_feature_vector,
)


def predict_next_week(
    occurrence_model,
    temporal_model,
    prepared,
    target_week_idx,
    device,
):
    """
    Genera la predicción completa de una semana.
    """

    occurrence_model.eval()
    temporal_model.eval()

    with torch.no_grad():
        # Construimos la secuencia de semanas de contexto (ventana)
        context_weeks = prepared.weeks[max(0, target_week_idx - WINDOW_WEEKS): target_week_idx]
        if not context_weeks:
            # Fallback si no hay historia suficiente
            seq = np.zeros((WINDOW_WEEKS, prepared.week_feature_dim), dtype=np.float32)
        else:
            seq = np.stack([week_to_feature_vector(week) for week in context_weeks]).astype(np.float32)
            # Pad si hubiese menos semanas
            if seq.shape[0] < WINDOW_WEEKS:
                pad = np.zeros((WINDOW_WEEKS - seq.shape[0], prepared.week_feature_dim), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)

        sequence_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        # =========================
        # 1) Predicción de conteos
        # =========================
        count_logits = occurrence_model(sequence_tensor)
        pred_counts = torch.argmax(count_logits, dim=-1).cpu().numpy()[0]

        predictions = []

        # =========================
        # 2) Predicción temporal
        # =========================
        for task_id, count in enumerate(pred_counts):
            if count == 0:
                continue
                
            history_features_np = build_history_features(
                weeks=prepared.weeks,
                target_week_index=target_week_idx,
                task_id=task_id,
                duration_min=prepared.duration_min,
                duration_max=prepared.duration_max,
            )
            history_tensor = torch.tensor(history_features_np, dtype=torch.float32).unsqueeze(0).to(device)
            predicted_count_norm_tensor = torch.tensor(
                [float(count / prepared.max_count_cap)], dtype=torch.float32
            ).to(device)

            for occ_idx in range(count):
                task_tensor = torch.tensor([task_id]).to(device)
                occ_tensor = torch.tensor([occ_idx]).to(device)

                start_logits, pred_duration_norm = temporal_model(
                    sequence=sequence_tensor,
                    task_id=task_tensor,
                    occurrence_index=occ_tensor,
                    history_features=history_tensor,
                    predicted_count_norm=predicted_count_norm_tensor,
                )

                start_bin = torch.argmax(start_logits, dim=-1).item()
                duration = denormalize_duration(
                    pred_duration_norm.item(),
                    prepared.duration_min,
                    prepared.duration_max,
                )

                predictions.append(
                    {
                        "task_name": prepared.task_names[task_id],
                        "start_bin": int(start_bin),
                        "duration": float(duration),
                    }
                )

    return predictions