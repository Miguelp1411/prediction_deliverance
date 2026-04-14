"""
Weekly prediction pipeline — full end-to-end inference.

Executes: retrieve template → predict Δ occurrence → predict temporal
candidates → solve schedule via CP-SAT → validate → explain.
"""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from config import resolve_device
from data.preprocessing import build_context_sequence
from data.schema import EventRecord, PreparedData
from features.occurrence_features import build_occurrence_features
from features.temporal_features import build_temporal_features
from retrieval.template_builder import TemplateBuilder
from retrieval.template_retriever import TemplateRetriever
from scheduler.constraints import EventCandidate, SchedulerConstraints
from scheduler.solver import solve_schedule
from scheduler.postcheck import validate_schedule


def predict_week(
    prepared: PreparedData,
    target_week_idx: int,
    occurrence_model: torch.nn.Module,
    temporal_model: torch.nn.Module,
    cfg: SimpleNamespace,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Full prediction pipeline for a single target week.

    Returns dict with:
      - schedule: list of scheduled events
      - template_metadata: info about which template was used
      - occurrence_delta: predicted count changes
      - temporal_candidates: per-event top-k candidates
      - validation: schedule validation report
      - explanation: human-readable explanation
    """
    if device is None:
        device = resolve_device(cfg)

    occurrence_model.eval()
    temporal_model.eval()

    retriever = TemplateRetriever(prepared)
    builder = TemplateBuilder(prepared, retriever)

    # ── Step 1: Retrieve template ────────────────────────────────
    strategy = getattr(cfg.retrieval, "strategy", "topk_blend")
    top_k = getattr(cfg.retrieval, "top_k", 5)

    template_events, template_counts, template_meta = builder.build_template(
        target_week_idx, strategy=strategy, top_k=top_k
    )

    # ── Step 2: Predict occurrence residuals ─────────────────────
    window_weeks = getattr(cfg.data, "window_weeks", 16)
    context_seq = build_context_sequence(prepared, target_week_idx, window_weeks)
    occ_features = build_occurrence_features(prepared, target_week_idx, template_counts)

    with torch.no_grad():
        seq_tensor = torch.tensor(context_seq, dtype=torch.float32).unsqueeze(0).to(device)
        feat_tensor = torch.tensor(occ_features, dtype=torch.float32).unsqueeze(0).to(device)
        task_ids = torch.arange(prepared.num_tasks).unsqueeze(0).to(device)
        db_idx = prepared.db_to_id.get(prepared.weeks[target_week_idx].database_id, 0)
        db_ids = torch.tensor([db_idx], dtype=torch.long).to(device)

        template_counts_tensor = torch.tensor(
            [template_counts.get(t, 0) for t in prepared.task_names],
            dtype=torch.long,
        ).unsqueeze(0).to(device)

        predicted_counts = occurrence_model.predict_counts(
            seq_tensor, feat_tensor, task_ids, db_ids, template_counts_tensor
        )[0].cpu().numpy()

    # Build final counts per task
    final_counts: dict[str, int] = {}
    occurrence_delta: dict[str, int] = {}
    for tid, task_name in enumerate(prepared.task_names):
        final_counts[task_name] = max(int(predicted_counts[tid]), 0)
        occurrence_delta[task_name] = final_counts[task_name] - template_counts.get(task_name, 0)

    # ── Step 3: Generate temporal candidates ─────────────────────
    seq_ctx_tensor = seq_tensor  # reuse
    bins_per_day_val = prepared.bins_per_day
    total_bins = prepared.num_time_bins

    event_candidates: list[EventCandidate] = []
    temporal_details: list[dict] = []
    event_idx = 0

    for task_name in prepared.task_names:
        count = final_counts[task_name]
        tid = prepared.task_to_id[task_name]

        # Get template events for this task
        task_template = [e for e in template_events if e.task_name == task_name]

        for slot in range(count):
            # Build history features
            template_ev_for_slot = task_template[slot] if slot < len(task_template) else None
            hist_features = build_temporal_features(
                prepared, target_week_idx,
                EventRecord(
                    task_id=tid, task_name=task_name,
                    start_bin=template_ev_for_slot.start_bin if template_ev_for_slot else 0,
                    duration_minutes=prepared.task_duration_medians.get(task_name, 30),
                    start_time=prepared.weeks[target_week_idx].week_start,
                    end_time=prepared.weeks[target_week_idx].week_start,
                ),
                occurrence_slot=slot,
                total_occurrences=count,
                template_events=template_events,
            )

            # Anchor info
            anchor_bin = template_ev_for_slot.start_bin if template_ev_for_slot else 0
            anchor_day = anchor_bin // bins_per_day_val
            anchor_time = anchor_bin % bins_per_day_val

            with torch.no_grad():
                hist_tensor = torch.tensor(hist_features, dtype=torch.float32).unsqueeze(0).to(device)
                task_tensor = torch.tensor([tid], dtype=torch.long).to(device)
                db_tensor = torch.tensor([db_idx], dtype=torch.long).to(device)
                occ_tensor = torch.tensor([slot], dtype=torch.long).to(device)
                count_norm = torch.tensor([count / max(prepared.max_count_cap, 1)], dtype=torch.float32).to(device)
                progress = torch.tensor([slot / max(count - 1, 1)], dtype=torch.float32).to(device)
                anchor_day_t = torch.tensor([anchor_day], dtype=torch.long).to(device)
                anchor_time_t = torch.tensor([anchor_time], dtype=torch.long).to(device)

                outputs = temporal_model(
                    seq_ctx_tensor, task_tensor, db_tensor, occ_tensor,
                    hist_tensor, count_norm, progress,
                    anchor_day_t, anchor_time_t,
                )
                topk_candidates = temporal_model.predict_topk(outputs, k=8, bins_per_day=bins_per_day_val)

            # Build EventCandidate for scheduler
            duration_norm = outputs["pred_duration_norm"][0].item()
            dur_span = prepared.duration_max - prepared.duration_min
            duration_minutes = duration_norm * dur_span + prepared.duration_min
            median_dur = prepared.task_duration_medians.get(task_name, 30)
            blend = getattr(cfg.scheduler, "duration_median_blend", 0.35)
            final_duration = duration_minutes * (1 - blend) + median_dur * blend
            duration_bins = max(1, round(final_duration / prepared.bin_minutes))

            candidates_for_solver = [
                (c["start_bin"], c["score"])
                for c in topk_candidates[0]
            ]

            ec = EventCandidate(
                event_idx=event_idx,
                task_name=task_name,
                task_id=tid,
                device_id=task_template[slot].device_id if slot < len(task_template) else "__unknown__",
                candidates=candidates_for_solver,
                duration_bins=duration_bins,
                template_start_bin=template_ev_for_slot.start_bin if template_ev_for_slot else None,
                preferred_order=event_idx,
            )
            event_candidates.append(ec)

            temporal_details.append({
                "event_idx": event_idx,
                "task_name": task_name,
                "slot": slot,
                "candidates": topk_candidates[0],
                "confidence": outputs["confidence"][0].item(),
            })
            event_idx += 1

    # ── Step 4: Solve schedule with CP-SAT ───────────────────────
    constraints = SchedulerConstraints.from_config(cfg.scheduler, bins_per_day_val)
    schedule = solve_schedule(event_candidates, constraints, total_bins)

    # ── Step 5: Validate ─────────────────────────────────────────
    validation = validate_schedule(schedule, total_bins)

    return {
        "schedule": schedule,
        "template_metadata": template_meta,
        "template_counts": template_counts,
        "final_counts": final_counts,
        "occurrence_delta": occurrence_delta,
        "temporal_candidates": temporal_details,
        "validation": validation,
    }
