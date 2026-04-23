from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

try:
    from ortools.sat.python import cp_model
except Exception:  # pragma: no cover - dependencia opcional
    cp_model = None



def _overlap(a_start: int, a_dur: int, b_start: int, b_dur: int, min_gap_bins: int = 0) -> bool:
    return not (a_start + a_dur + min_gap_bins <= b_start or b_start + b_dur + min_gap_bins <= a_start)



def _greedy_fallback(events: list[dict], min_gap_bins: int = 0) -> list[dict]:
    chosen = []
    by_robot: dict[str, list[dict]] = {}
    for event in sorted(events, key=lambda x: max((c['score'] for c in x.get('candidates', [{'score': 0.0}])), default=0.0), reverse=True):
        candidates = sorted(event['candidates'], key=lambda x: x['score'], reverse=True)
        robot = event['robot_id']
        robot_schedule = by_robot.setdefault(robot, [])
        selected = None
        for cand in candidates:
            if all(not _overlap(cand['start_bin'], cand['duration_bins'], other['start_bin'], other['duration_bins'], min_gap_bins=min_gap_bins) for other in robot_schedule):
                selected = cand
                break
        if selected is None:
            selected = candidates[0]
        placed = {
            'robot_id': robot,
            'task_type': event['task_type'],
            'task_idx': event['task_idx'],
            'start_bin': int(selected['start_bin']),
            'duration_bins': int(selected['duration_bins']),
            'score': float(selected['score']),
            'anchor_start_bin': int(event.get('anchor_start_bin', selected['start_bin'])),
        }
        robot_schedule.append(placed)
        chosen.append(placed)
    return chosen



def _solve_cp_sat(events: list[dict], min_gap_bins: int = 0, max_solver_seconds: int = 20):
    if cp_model is None:
        return None
    model = cp_model.CpModel()
    vars_by_event: list[list] = []
    all_vars = []
    event_refs = []
    for event_idx, event in enumerate(events):
        local_vars = []
        for cand_idx, cand in enumerate(event['candidates']):
            var = model.NewBoolVar(f'x_{event_idx}_{cand_idx}')
            local_vars.append(var)
            all_vars.append((var, float(cand['score'])))
            event_refs.append((event_idx, cand_idx, var))
        model.Add(sum(local_vars) == 1)
        vars_by_event.append(local_vars)

    for (i, event_a), (j, event_b) in combinations(list(enumerate(events)), 2):
        if event_a['robot_id'] != event_b['robot_id']:
            continue
        for a_idx, cand_a in enumerate(event_a['candidates']):
            for b_idx, cand_b in enumerate(event_b['candidates']):
                if _overlap(cand_a['start_bin'], cand_a['duration_bins'], cand_b['start_bin'], cand_b['duration_bins'], min_gap_bins=min_gap_bins):
                    model.Add(vars_by_event[i][a_idx] + vars_by_event[j][b_idx] <= 1)

    scale = 1000
    model.Maximize(sum(int(round(score * scale)) * var for var, score in all_vars))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_solver_seconds)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    chosen = []
    for event_idx, event in enumerate(events):
        chosen_cand = None
        for cand_idx, cand in enumerate(event['candidates']):
            if solver.BooleanValue(vars_by_event[event_idx][cand_idx]):
                chosen_cand = cand
                break
        if chosen_cand is None:
            chosen_cand = event['candidates'][0]
        chosen.append({
            'robot_id': event['robot_id'],
            'task_type': event['task_type'],
            'task_idx': event['task_idx'],
            'start_bin': int(chosen_cand['start_bin']),
            'duration_bins': int(chosen_cand['duration_bins']),
            'score': float(chosen_cand['score']),
            'anchor_start_bin': int(event.get('anchor_start_bin', chosen_cand['start_bin'])),
        })
    return chosen



def _solve_scipy_milp(events: list[dict], min_gap_bins: int = 0, max_solver_seconds: int = 20):
    total_variables = sum(len(e.get('candidates', [])) for e in events)
    if total_variables == 0:
        return []
    variable_index = {}
    objective = []
    integrality = []
    event_constraints = []
    bounds_lb = []
    bounds_ub = []
    idx = 0

    for event_idx, event in enumerate(events):
        row = []
        for cand_idx, cand in enumerate(event['candidates']):
            variable_index[(event_idx, cand_idx)] = idx
            objective.append(-float(cand['score']))
            integrality.append(1)
            bounds_lb.append(0.0)
            bounds_ub.append(1.0)
            row.append(idx)
            idx += 1
        event_constraints.append(row)

    c = np.asarray(objective, dtype=np.float64)
    integrality = np.asarray(integrality, dtype=np.int8)
    bounds = Bounds(bounds_lb, bounds_ub)
    constraints = []

    for row in event_constraints:
        A = np.zeros((1, len(c)), dtype=np.float64)
        A[0, row] = 1.0
        constraints.append(LinearConstraint(A, [1.0], [1.0]))

    for (i, event_a), (j, event_b) in combinations(list(enumerate(events)), 2):
        if event_a['robot_id'] != event_b['robot_id']:
            continue
        for a_idx, cand_a in enumerate(event_a['candidates']):
            for b_idx, cand_b in enumerate(event_b['candidates']):
                if _overlap(cand_a['start_bin'], cand_a['duration_bins'], cand_b['start_bin'], cand_b['duration_bins'], min_gap_bins=min_gap_bins):
                    A = np.zeros((1, len(c)), dtype=np.float64)
                    A[0, variable_index[(i, a_idx)]] = 1.0
                    A[0, variable_index[(j, b_idx)]] = 1.0
                    constraints.append(LinearConstraint(A, [-np.inf], [1.0]))

    try:
        result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options={'time_limit': float(max_solver_seconds)})
        if not result.success or result.x is None:
            return None
        x = result.x
    except Exception:
        return None

    chosen = []
    for event_idx, event in enumerate(events):
        best_cand_idx = None
        best_val = -1.0
        for cand_idx, _ in enumerate(event['candidates']):
            val = x[variable_index[(event_idx, cand_idx)]]
            if val > best_val:
                best_val = float(val)
                best_cand_idx = cand_idx
        chosen_cand = event['candidates'][int(best_cand_idx or 0)]
        chosen.append({
            'robot_id': event['robot_id'],
            'task_type': event['task_type'],
            'task_idx': event['task_idx'],
            'start_bin': int(chosen_cand['start_bin']),
            'duration_bins': int(chosen_cand['duration_bins']),
            'score': float(chosen_cand['score']),
            'anchor_start_bin': int(event.get('anchor_start_bin', chosen_cand['start_bin'])),
        })
    return chosen



def solve_week_schedule(events: list[dict], use_exact_milp: bool = True, min_gap_bins: int = 0, max_solver_seconds: int = 20) -> list[dict]:
    if not events:
        return []
    total_variables = sum(len(e.get('candidates', [])) for e in events)
    if (not use_exact_milp) or total_variables > 800 or len(events) > 120:
        return _greedy_fallback(events, min_gap_bins=min_gap_bins)
    solved = _solve_cp_sat(events, min_gap_bins=min_gap_bins, max_solver_seconds=max_solver_seconds)
    if solved is not None:
        return solved
    solved = _solve_scipy_milp(events, min_gap_bins=min_gap_bins, max_solver_seconds=max_solver_seconds)
    if solved is not None:
        return solved
    return _greedy_fallback(events, min_gap_bins=min_gap_bins)
