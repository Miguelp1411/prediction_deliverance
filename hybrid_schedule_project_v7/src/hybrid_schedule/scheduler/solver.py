from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover - fallback when OR-Tools is not installed
    cp_model = None


def _overlap(a_start: int, a_dur: int, b_start: int, b_dur: int, min_gap_bins: int = 0) -> bool:
    a_end = a_start + a_dur
    b_end = b_start + b_dur
    return not (a_end + min_gap_bins <= b_start or b_end + min_gap_bins <= a_start)


def _candidate_total_score(cand: dict[str, Any], scheduler_cfg: dict[str, Any], bin_minutes: int) -> float:
    bins_per_day = int(24 * 60 / bin_minutes)

    start_bin = int(cand['start_bin'])
    duration_bins = max(1, int(cand['duration_bins']))

    anchor_start_bin = int(cand.get('anchor_start_bin', start_bin))
    anchor_duration_bins = max(1, int(cand.get('anchor_duration_bins', duration_bins)))

    model_score = float(cand.get('model_score', cand.get('score', 0.0)))
    empirical_support = float(cand.get('empirical_support', 0.0))

    cand_day = start_bin // bins_per_day
    anchor_day = anchor_start_bin // bins_per_day

    cand_local = start_bin % bins_per_day
    anchor_local = anchor_start_bin % bins_per_day

    local_gap_bins = abs(cand_local - anchor_local)
    day_gap = abs(cand_day - anchor_day)
    duration_gap_bins = abs(duration_bins - anchor_duration_bins)

    total = (
        float(scheduler_cfg.get('empirical_model_weight', 1.0)) * model_score
        + float(scheduler_cfg.get('empirical_support_weight', 0.0)) * np.log1p(max(0.0, empirical_support))
        - float(scheduler_cfg.get('movement_penalty', 0.0)) * (local_gap_bins / max(1, bins_per_day))
        - float(scheduler_cfg.get('day_penalty', 0.0)) * float(day_gap)
        - float(scheduler_cfg.get('duration_penalty', 0.0)) * (duration_gap_bins / 12.0)
    )
    return float(total)


def _prepare_events(events: list[dict[str, Any]], scheduler_cfg: dict[str, Any], bin_minutes: int) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for event in events:
        prepared_candidates = []
        for cand in event.get('candidates', []):
            cand2 = dict(cand)
            cand2.setdefault('anchor_start_bin', int(event.get('anchor_start_bin', cand2['start_bin'])))
            cand2.setdefault('anchor_duration_bins', int(event.get('anchor_duration_bins', cand2['duration_bins'])))
            cand2['total_score'] = _candidate_total_score(cand2, scheduler_cfg=scheduler_cfg, bin_minutes=bin_minutes)
            prepared_candidates.append(cand2)
        prepared.append({
            **event,
            'candidates': prepared_candidates,
        })
    return prepared


def _greedy_fallback(
    events: list[dict[str, Any]],
    min_gap_bins: int = 0,
    prohibit_overlap: bool = True,
) -> list[dict[str, Any]]:
    chosen = []
    by_robot: dict[str, list[dict[str, Any]]] = {}

    def _priority(event: dict[str, Any]) -> tuple[float, float, int]:
        candidates = sorted(event['candidates'], key=lambda x: x['total_score'], reverse=True)
        best = float(candidates[0]['total_score']) if candidates else float('-inf')
        second = float(candidates[1]['total_score']) if len(candidates) > 1 else best - 1.0
        return (-(best - second), -best, len(candidates))

    for event in sorted(events, key=_priority):
        candidates = sorted(event['candidates'], key=lambda x: x['total_score'], reverse=True)
        robot = event['robot_id']
        robot_schedule = by_robot.setdefault(robot, [])

        selected = None
        if not prohibit_overlap:
            selected = candidates[0] if candidates else None
        else:
            for cand in candidates:
                if all(
                    not _overlap(
                        cand['start_bin'],
                        cand['duration_bins'],
                        other['start_bin'],
                        other['duration_bins'],
                        min_gap_bins=min_gap_bins,
                    )
                    for other in robot_schedule
                ):
                    selected = cand
                    break

            if selected is None:
                selected = min(
                    candidates,
                    key=lambda cand: (
                        sum(
                            _overlap(
                                cand['start_bin'],
                                cand['duration_bins'],
                                other['start_bin'],
                                other['duration_bins'],
                                min_gap_bins=min_gap_bins,
                            )
                            for other in robot_schedule
                        ),
                        -float(cand['total_score']),
                    ),
                )

        if selected is None:
            continue

        placed = {
            'robot_id': robot,
            'task_type': event['task_type'],
            'task_idx': event['task_idx'],
            'start_bin': int(selected['start_bin']),
            'duration_bins': int(selected['duration_bins']),
            'score': float(selected['total_score']),
            'model_score': float(selected.get('model_score', selected.get('score', 0.0))),
            'empirical_support': float(selected.get('empirical_support', 0.0)),
            'anchor_start_bin': int(event.get('anchor_start_bin', selected['start_bin'])),
            'anchor_duration_bins': int(event.get('anchor_duration_bins', selected['duration_bins'])),
        }
        robot_schedule.append(placed)
        chosen.append(placed)

    return chosen


def solve_week_schedule(
    events: list[dict[str, Any]],
    scheduler_cfg: dict[str, Any],
    bin_minutes: int,
) -> list[dict[str, Any]]:
    if not events:
        return []

    backend = str(scheduler_cfg.get('backend', 'cp_sat')).lower()
    min_gap_bins = int(scheduler_cfg.get('min_gap_bins', 0))
    prohibit_overlap = bool(scheduler_cfg.get('prohibit_overlap', True))
    max_solver_seconds = int(scheduler_cfg.get('max_solver_seconds', 20))
    max_exact_events = scheduler_cfg.get('max_exact_events', 128)
    max_exact_variables = scheduler_cfg.get('max_exact_variables', 2500)
    num_search_workers = int(scheduler_cfg.get('num_search_workers', 8))
    objective_scale = int(scheduler_cfg.get('objective_scale', 1000))

    prepared = _prepare_events(events, scheduler_cfg=scheduler_cfg, bin_minutes=bin_minutes)

    total_variables = sum(len(e.get('candidates', [])) for e in prepared)
    over_event_limit = max_exact_events is not None and len(prepared) > int(max_exact_events)
    over_variable_limit = max_exact_variables is not None and total_variables > int(max_exact_variables)

    if backend == 'greedy' or cp_model is None or over_event_limit or over_variable_limit:
        return _greedy_fallback(
            prepared,
            min_gap_bins=min_gap_bins,
            prohibit_overlap=prohibit_overlap,
        )

    model = cp_model.CpModel()
    var_map: dict[tuple[int, int], cp_model.IntVar] = {}

    for event_idx, event in enumerate(prepared):
        vars_for_event = []
        for cand_idx, cand in enumerate(event['candidates']):
            var = model.NewBoolVar(f'x_{event_idx}_{cand_idx}')
            var_map[(event_idx, cand_idx)] = var
            vars_for_event.append(var)
        if not vars_for_event:
            return _greedy_fallback(
                prepared,
                min_gap_bins=min_gap_bins,
                prohibit_overlap=prohibit_overlap,
            )
        model.AddExactlyOne(vars_for_event)

    if prohibit_overlap:
        for (i, event_a), (j, event_b) in combinations(list(enumerate(prepared)), 2):
            if event_a['robot_id'] != event_b['robot_id']:
                continue
            for a_idx, cand_a in enumerate(event_a['candidates']):
                for b_idx, cand_b in enumerate(event_b['candidates']):
                    if _overlap(
                        cand_a['start_bin'],
                        cand_a['duration_bins'],
                        cand_b['start_bin'],
                        cand_b['duration_bins'],
                        min_gap_bins=min_gap_bins,
                    ):
                        model.Add(var_map[(i, a_idx)] + var_map[(j, b_idx)] <= 1)

    objective_terms = []
    for event_idx, event in enumerate(prepared):
        for cand_idx, cand in enumerate(event['candidates']):
            coeff = int(round(float(cand['total_score']) * objective_scale))
            objective_terms.append(coeff * var_map[(event_idx, cand_idx)])

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_solver_seconds)
    solver.parameters.num_search_workers = int(num_search_workers)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _greedy_fallback(
            prepared,
            min_gap_bins=min_gap_bins,
            prohibit_overlap=prohibit_overlap,
        )

    chosen = []
    for event_idx, event in enumerate(prepared):
        chosen_cand = None
        for cand_idx, cand in enumerate(event['candidates']):
            if solver.Value(var_map[(event_idx, cand_idx)]) == 1:
                chosen_cand = cand
                break
        if chosen_cand is None:
            chosen_cand = max(event['candidates'], key=lambda x: x['total_score'])

        chosen.append({
            'robot_id': event['robot_id'],
            'task_type': event['task_type'],
            'task_idx': event['task_idx'],
            'start_bin': int(chosen_cand['start_bin']),
            'duration_bins': int(chosen_cand['duration_bins']),
            'score': float(chosen_cand['total_score']),
            'model_score': float(chosen_cand.get('model_score', chosen_cand.get('score', 0.0))),
            'empirical_support': float(chosen_cand.get('empirical_support', 0.0)),
            'anchor_start_bin': int(event.get('anchor_start_bin', chosen_cand['start_bin'])),
            'anchor_duration_bins': int(event.get('anchor_duration_bins', chosen_cand['duration_bins'])),
        })

    return chosen