"""
CP-SAT exact scheduler — constructs conflict-free weekly agendas.

Replaces all heuristic gating/rerank/repair logic with a global
optimisation that respects hard constraints (no overlaps, valid times)
and maximises model scores while staying close to the template.
"""
from __future__ import annotations

import math
from typing import Any

from ortools.sat.python import cp_model

from scheduler.constraints import EventCandidate, SchedulerConstraints


def solve_schedule(
    candidates: list[EventCandidate],
    constraints: SchedulerConstraints,
    total_weekly_bins: int = 2016,
) -> list[dict[str, Any]]:
    """
    Solve the weekly scheduling problem.

    For each event candidate with multiple possible start_bins,
    choose exactly one start_bin such that:
      - No two events on the same device overlap (hard)
      - Events fit within 0..total_bins-1 (hard)
      - Maximise model scores (soft objective)
      - Minimise deviation from template (soft)
      - Preserve historical order (soft)

    Returns list of scheduled events:
      {"event_idx", "task_name", "task_id", "device_id",
       "start_bin", "duration_bins", "score", "from_candidate_idx"}
    """
    if not candidates:
        return []

    model = cp_model.CpModel()

    # ── Decision variables ───────────────────────────────────────
    # For each event, one binary variable per candidate slot
    slot_vars: list[list[cp_model.IntVar]] = []
    start_vars: list[cp_model.IntVar] = []

    for ev in candidates:
        if not ev.candidates:
            # No candidates — create a fallback at template or 0
            fallback = ev.template_start_bin if ev.template_start_bin is not None else 0
            ev.candidates = [(fallback, 0.1)]

        ev_slot_vars = []
        for j, (start_bin, score) in enumerate(ev.candidates):
            var = model.NewBoolVar(f"ev{ev.event_idx}_slot{j}")
            ev_slot_vars.append(var)
        slot_vars.append(ev_slot_vars)

        # Exactly one slot must be chosen per event
        model.AddExactlyOne(ev_slot_vars)

        # Create the actual start variable
        start_var = model.NewIntVar(0, total_weekly_bins - 1, f"start_{ev.event_idx}")
        start_vars.append(start_var)

        # Link start_var to chosen slot
        for j, (start_bin, _) in enumerate(ev.candidates):
            model.Add(start_var == start_bin).OnlyEnforceIf(ev_slot_vars[j])

    # ── Hard constraints ─────────────────────────────────────────
    if constraints.no_overlap_per_device:
        # Group events by device
        device_events: dict[str, list[int]] = {}
        for i, ev in enumerate(candidates):
            device_events.setdefault(ev.device_id, []).append(i)

        for device_id, event_indices in device_events.items():
            if len(event_indices) < 2:
                continue
            # Create interval variables for no-overlap
            intervals = []
            for i in event_indices:
                ev = candidates[i]
                interval = model.NewIntervalVar(
                    start_vars[i],
                    ev.duration_bins + constraints.min_gap_bins,
                    start_vars[i] + ev.duration_bins + constraints.min_gap_bins,
                    f"interval_{i}",
                )
                intervals.append(interval)
            model.AddNoOverlap(intervals)

    # ── Objective function ───────────────────────────────────────
    SCORE_SCALE = 1000  # Scale float scores to integer for CP-SAT

    objective_terms: list[cp_model.LinearExpr] = []

    for i, ev in enumerate(candidates):
        for j, (start_bin, score) in enumerate(ev.candidates):
            # Reward: model score
            score_reward = int(score * SCORE_SCALE * constraints.prefer_model_score)
            objective_terms.append(score_reward * slot_vars[i][j])

            # Penalty: deviation from template
            if ev.template_start_bin is not None:
                deviation = abs(start_bin - ev.template_start_bin)
                deviation_penalty = int(
                    deviation * constraints.penalize_template_deviation
                )
                objective_terms.append(-deviation_penalty * slot_vars[i][j])

            # Penalty: order disruption (if start_bin would break original order)
            if constraints.penalize_order_disruption > 0:
                order_penalty = 0
                # Penalize being far from expected position
                expected_bin = ev.preferred_order * (total_weekly_bins // max(len(candidates), 1))
                order_dev = abs(start_bin - expected_bin) // 100
                order_penalty = int(order_dev * constraints.penalize_order_disruption)
                objective_terms.append(-order_penalty * slot_vars[i][j])

    if objective_terms:
        model.Maximize(sum(objective_terms))

    # ── Solve ────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = constraints.solver_time_limit_seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    # ── Extract solution ─────────────────────────────────────────
    results: list[dict[str, Any]] = []

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, ev in enumerate(candidates):
            chosen_start = solver.Value(start_vars[i])
            chosen_slot_idx = -1
            chosen_score = 0.0
            for j, (start_bin, score) in enumerate(ev.candidates):
                if solver.BooleanValue(slot_vars[i][j]):
                    chosen_slot_idx = j
                    chosen_score = score
                    break

            results.append({
                "event_idx": ev.event_idx,
                "task_name": ev.task_name,
                "task_id": ev.task_id,
                "device_id": ev.device_id,
                "start_bin": chosen_start,
                "duration_bins": ev.duration_bins,
                "score": chosen_score,
                "from_candidate_idx": chosen_slot_idx,
                "template_start_bin": ev.template_start_bin,
                "solver_status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            })
    else:
        # Fallback: use best candidate per event (no solver solution)
        for i, ev in enumerate(candidates):
            if ev.candidates:
                best = max(ev.candidates, key=lambda x: x[1])
                results.append({
                    "event_idx": ev.event_idx,
                    "task_name": ev.task_name,
                    "task_id": ev.task_id,
                    "device_id": ev.device_id,
                    "start_bin": best[0],
                    "duration_bins": ev.duration_bins,
                    "score": best[1],
                    "from_candidate_idx": 0,
                    "template_start_bin": ev.template_start_bin,
                    "solver_status": "fallback",
                })

    # Sort by start_bin
    results.sort(key=lambda x: x["start_bin"])
    return results
