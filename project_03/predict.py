from __future__ import annotations

import numpy as np
import torch

from config import WINDOW_WEEKS, num_time_bins
from data.preprocessing import (
    build_temporal_context,
    denormalize_duration,
    global_day_index_to_offset,
    local_start_index_to_offset,
    week_to_feature_vector,
)


def predict_next_week(occurrence_model, temporal_model, prepared, target_week_idx, device):
    occurrence_model.eval()
    temporal_model.eval()

    with torch.no_grad():
        context_weeks = prepared.weeks[max(0, target_week_idx - WINDOW_WEEKS):target_week_idx]
        if not context_weeks:
            seq = np.zeros((WINDOW_WEEKS, prepared.week_feature_dim), dtype=np.float32)
        else:
            seq = np.stack([week_to_feature_vector(week) for week in context_weeks]).astype(np.float32)
            if seq.shape[0] < WINDOW_WEEKS:
                pad = np.zeros((WINDOW_WEEKS - seq.shape[0], prepared.week_feature_dim), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)

        base_sequence = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
        count_logits = occurrence_model(base_sequence)
        pred_counts = torch.argmax(count_logits, dim=-1).cpu().numpy()[0]

        task_ids, occurrence_indices, predicted_count_norms = [], [], []
        history_features, anchor_days, anchor_start_bins = [], [], []

        for task_id, count in enumerate(pred_counts):
            count = int(min(count, prepared.max_count_cap))
            if count == 0:
                continue
            for occ_idx in range(count):
                context = build_temporal_context(
                    prepared.weeks,
                    target_week_idx,
                    task_id,
                    occ_idx,
                    prepared.duration_min,
                    prepared.duration_max,
                )
                task_ids.append(task_id)
                occurrence_indices.append(occ_idx)
                predicted_count_norms.append(float(count / prepared.max_count_cap))
                history_features.append(context.history_features)
                anchor_days.append(context.anchor_day)
                anchor_start_bins.append(context.anchor_start_bin)

        if not task_ids:
            return []

        batch_size = len(task_ids)
        sequence_context = temporal_model.encode_sequence(base_sequence).repeat(batch_size, 1)
        outputs = temporal_model.forward_with_context(
            sequence_context=sequence_context,
            task_id=torch.tensor(task_ids, dtype=torch.long, device=device),
            occurrence_index=torch.tensor(occurrence_indices, dtype=torch.long, device=device),
            history_features=torch.tensor(np.stack(history_features), dtype=torch.float32, device=device),
            predicted_count_norm=torch.tensor(predicted_count_norms, dtype=torch.float32, device=device),
            anchor_day=torch.tensor(anchor_days, dtype=torch.long, device=device),
        )

        day_offset_idx = outputs['day_offset_logits'].argmax(dim=-1).cpu().tolist()
        local_offset_idx = outputs['local_offset_logits'].argmax(dim=-1).cpu().tolist()
        duration_norms = outputs['pred_duration_norm'].cpu().tolist()

        predictions = []
        for i in range(batch_size):
            day_offset = global_day_index_to_offset(day_offset_idx[i])
            local_offset = local_start_index_to_offset(local_offset_idx[i])
            start_bin = int(np.clip(anchor_start_bins[i] + day_offset + local_offset, 0, num_time_bins() - 1))
            duration = denormalize_duration(duration_norms[i], prepared.duration_min, prepared.duration_max)
            predictions.append({
                'task_name': prepared.task_names[task_ids[i]],
                'start_bin': start_bin,
                'duration': float(duration),
            })

    return sorted(predictions, key=lambda x: (x['start_bin'], x['task_name']))
