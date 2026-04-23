"""
Datasets for PyTorch training of occurrence and temporal models.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing import build_context_sequence
from data.schema import PreparedData
from features.occurrence_features import build_occurrence_features
from features.temporal_features import build_temporal_features
from retrieval.template_builder import TemplateBuilder
from retrieval.template_retriever import TemplateRetriever


class OccurrenceDataset(Dataset):
    """Dataset for training the residual occurrence model."""

    def __init__(
        self,
        prepared: PreparedData,
        target_week_indices: list[int],
        window_weeks: int = 16,
        retriever: TemplateRetriever | None = None,
        show_progress: bool = False,
    ) -> None:
        self.samples: list[dict[str, Any]] = []

        if retriever is None:
            retriever = TemplateRetriever(prepared)
        builder = TemplateBuilder(prepared, retriever)

        total = len(target_week_indices)
        for i, idx in enumerate(target_week_indices):
            if show_progress and (i + 1) % 50 == 0:
                print(f"\r  Building OccurrenceDataset: {i+1}/{total}", end="", flush=True)

            # Build template
            _, template_counts, _ = builder.build_template(idx, strategy="topk_blend")

            # Build features
            context_seq = build_context_sequence(prepared, idx, window_weeks)
            occ_features = build_occurrence_features(prepared, idx, template_counts)

            # Target: actual counts
            actual_counts = prepared.weeks[idx].counts.astype(np.int64)
            template_counts_arr = np.array(
                [template_counts.get(t, 0) for t in prepared.task_names],
                dtype=np.int64,
            )

            # Database ID
            db_id = prepared.db_to_id.get(prepared.weeks[idx].database_id, 0)

            self.samples.append({
                "context_sequence": context_seq,
                "task_features": occ_features,
                "target_counts": actual_counts,
                "template_counts": template_counts_arr,
                "db_id": db_id,
            })

        if show_progress:
            print()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.samples[index]
        num_tasks = s["task_features"].shape[0]
        return {
            "input_sequence": torch.tensor(s["context_sequence"], dtype=torch.float32),
            "input_task_features": torch.tensor(s["task_features"], dtype=torch.float32),
            "input_task_ids": torch.arange(num_tasks, dtype=torch.long),
            "input_db_ids": torch.tensor(s["db_id"], dtype=torch.long),
            "target_counts": torch.tensor(s["target_counts"], dtype=torch.long),
            "template_counts": torch.tensor(s["template_counts"], dtype=torch.long),
        }


class TemporalDataset(Dataset):
    """Dataset for training the temporal residual model."""

    def __init__(
        self,
        prepared: PreparedData,
        target_week_indices: list[int],
        window_weeks: int = 16,
        retriever: TemplateRetriever | None = None,
        show_progress: bool = False,
    ) -> None:
        self.samples: list[dict[str, Any]] = []

        if retriever is None:
            retriever = TemplateRetriever(prepared)
        builder = TemplateBuilder(prepared, retriever)

        bins_per_day = prepared.bins_per_day

        total = len(target_week_indices)
        for i, idx in enumerate(target_week_indices):
            if show_progress and (i + 1) % 50 == 0:
                print(f"\r  Building TemporalDataset: {i+1}/{total}", end="", flush=True)

            context_seq = build_context_sequence(prepared, idx, window_weeks)
            template_events, template_counts, _ = builder.build_template(idx)
            week = prepared.weeks[idx]
            db_id = prepared.db_to_id.get(week.database_id, 0)

            # For each real event in this week, create a sample
            for tid in range(prepared.num_tasks):
                task_events = week.events_by_task.get(tid, [])
                count = len(task_events)
                if count == 0:
                    continue

                for slot, event in enumerate(task_events):
                    hist_feats = build_temporal_features(
                        prepared, idx, event,
                        occurrence_slot=slot,
                        total_occurrences=count,
                        template_events=[e for e in template_events if e.task_name == event.task_name],
                    )

                    # Anchor from template
                    task_tpl = [e for e in template_events if e.task_name == event.task_name]
                    anchor_bin = task_tpl[slot].start_bin if slot < len(task_tpl) else 0
                    anchor_day = anchor_bin // bins_per_day
                    anchor_time = anchor_bin % bins_per_day

                    target_day = event.start_bin // bins_per_day
                    target_time = event.start_bin % bins_per_day

                    dur_span = max(prepared.duration_max - prepared.duration_min, 1e-6)
                    target_dur_norm = (event.duration_minutes - prepared.duration_min) / dur_span

                    self.samples.append({
                        "context_sequence": context_seq,
                        "task_id": tid,
                        "db_id": db_id,
                        "occurrence_slot": min(slot, 49),  # cap
                        "history_features": hist_feats,
                        "count_norm": count / max(prepared.max_count_cap, 1),
                        "progress": slot / max(count - 1, 1),
                        "anchor_day": min(anchor_day, 6),
                        "anchor_time_bin": min(anchor_time, bins_per_day - 1),
                        "target_start_bin": event.start_bin,
                        "target_day_idx": min(target_day, 6),
                        "target_time_bin_idx": min(target_time, bins_per_day - 1),
                        "target_duration_norm": target_dur_norm,
                    })

        if show_progress:
            print()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = self.samples[index]
        return {
            "input_sequence": torch.tensor(s["context_sequence"], dtype=torch.float32),
            "input_task_id": torch.tensor(s["task_id"], dtype=torch.long),
            "input_db_id": torch.tensor(s["db_id"], dtype=torch.long),
            "input_occurrence_slot": torch.tensor(s["occurrence_slot"], dtype=torch.long),
            "input_history_features": torch.tensor(s["history_features"], dtype=torch.float32),
            "input_predicted_count_norm": torch.tensor(s["count_norm"], dtype=torch.float32),
            "input_occurrence_progress": torch.tensor(s["progress"], dtype=torch.float32),
            "input_anchor_day": torch.tensor(s["anchor_day"], dtype=torch.long),
            "input_anchor_time_bin": torch.tensor(s["anchor_time_bin"], dtype=torch.long),
            "target_start_bin": torch.tensor(s["target_start_bin"], dtype=torch.long),
            "target_day_idx": torch.tensor(s["target_day_idx"], dtype=torch.long),
            "target_time_bin_idx": torch.tensor(s["target_time_bin_idx"], dtype=torch.long),
            "target_duration_norm": torch.tensor(s["target_duration_norm"], dtype=torch.float32),
        }
