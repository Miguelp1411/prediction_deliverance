from __future__ import annotations

from typing import Any

import math


def _overlap(a_start: int, a_dur: int, b_start: int, b_dur: int) -> bool:
    return not (a_start + a_dur <= b_start or b_start + b_dur <= a_start)


def _candidate_score(
    active_prob: float,
    day_logprob: float,
    time_logprob: float,
    duration_penalty: float,
    anchor_penalty: float,
) -> float:
    return (
        math.log(max(active_prob, 1e-8))
        + day_logprob
        + time_logprob
        - duration_penalty
        - anchor_penalty
    )


def build_event_candidates(
    event_row: dict[str, Any],
    bins_per_day: int,
    topk_days: int,
    topk_times: int,
    duration_radius_bins: int,
    max_candidates_per_event: int,
    anchor_penalty_weight: float,
    duration_penalty_weight: float,
) -> list[dict[str, Any]]:
    start_items = sorted(event_row.get('start_distribution', []), key=lambda x: x[1], reverse=True)
    day_items = sorted(event_row.get('day_distribution', []), key=lambda x: x[1], reverse=True)[:max(1, int(topk_days))]
    time_items = sorted(event_row.get('time_distribution', []), key=lambda x: x[1], reverse=True)[:max(1, int(topk_times))]
    pred_duration = max(1, int(round(event_row['pred_duration_bins'])))
    duration_candidates = sorted(set(max(1, pred_duration + d) for d in range(-int(duration_radius_bins), int(duration_radius_bins) + 1)))
    out = []
    if start_items:
        for start_bin, start_prob in start_items[:max(1, int(topk_times) * max(1, int(topk_days)))]:
            for duration in duration_candidates:
                anchor_gap = abs(int(start_bin) - int(event_row['anchor_start_bin'])) / max(1.0, float(7 * bins_per_day))
                duration_gap = abs(duration - int(event_row['anchor_duration_bins'])) / 12.0
                out.append({
                    'start_bin': int(start_bin),
                    'duration_bins': int(duration),
                    'score': _candidate_score(
                        active_prob=float(event_row['active_prob']),
                        day_logprob=0.0,
                        time_logprob=float(math.log(max(start_prob, 1e-8))),
                        duration_penalty=float(duration_gap) * float(duration_penalty_weight),
                        anchor_penalty=float(anchor_gap) * float(anchor_penalty_weight),
                    )
                })
    else:
        for day_idx, day_prob in day_items:
            for time_idx, time_prob in time_items:
                start_bin = int(day_idx) * int(bins_per_day) + int(time_idx)
                for duration in duration_candidates:
                    anchor_gap = abs(start_bin - int(event_row['anchor_start_bin'])) / max(1.0, float(7 * bins_per_day))
                    duration_gap = abs(duration - int(event_row['anchor_duration_bins'])) / 12.0
                    out.append({
                        'start_bin': start_bin,
                        'duration_bins': int(duration),
                        'score': _candidate_score(
                            active_prob=float(event_row['active_prob']),
                            day_logprob=float(math.log(max(day_prob, 1e-8))),
                            time_logprob=float(math.log(max(time_prob, 1e-8))),
                            duration_penalty=float(duration_gap) * float(duration_penalty_weight),
                            anchor_penalty=float(anchor_gap) * float(anchor_penalty_weight),
                        )
                    })
    out.sort(key=lambda x: x['score'], reverse=True)
    return out[:max(1, int(max_candidates_per_event))]


def decode_week_with_constraints(
    selected_events: list[dict[str, Any]],
    bins_per_day: int,
    topk_days: int = 3,
    topk_times: int = 12,
    duration_radius_bins: int = 1,
    beam_width: int = 6,
    max_candidates_per_event: int = 18,
    anchor_penalty: float = 0.10,
    duration_penalty: float = 0.05,
    occupancy_soft_penalty: float = 5.0,
) -> list[dict[str, Any]]:
    if not selected_events:
        return []

    expanded = []
    for row in selected_events:
        candidates = build_event_candidates(
            row,
            bins_per_day=bins_per_day,
            topk_days=topk_days,
            topk_times=topk_times,
            duration_radius_bins=duration_radius_bins,
            max_candidates_per_event=max_candidates_per_event,
            anchor_penalty_weight=anchor_penalty,
            duration_penalty_weight=duration_penalty,
        )
        if not candidates:
            continue
        expanded.append({**row, 'candidates': candidates})
    expanded.sort(key=lambda x: float(x['active_prob']), reverse=True)

    beams = [{'score': 0.0, 'events': [], 'by_robot': {}}]
    for row in expanded:
        new_beams = []
        robot = row['robot_id']
        for beam in beams:
            schedule = beam['by_robot'].get(robot, [])
            for cand in row['candidates']:
                overlaps = sum(_overlap(cand['start_bin'], cand['duration_bins'], other['start_bin'], other['duration_bins']) for other in schedule)
                penalty = float(occupancy_soft_penalty) * float(overlaps)
                placed = {
                    'database_id': row['database_id'],
                    'robot_id': row['robot_id'],
                    'task_type': row['task_type'],
                    'task_idx': int(row['task_idx']),
                    'slot_id': int(row['slot_id']),
                    'start_bin': int(cand['start_bin']),
                    'duration_bins': int(cand['duration_bins']),
                    'active_prob': float(row['active_prob']),
                    'anchor_start_bin': int(row['anchor_start_bin']),
                    'anchor_duration_bins': int(row['anchor_duration_bins']),
                    'score': float(cand['score']) - penalty,
                }
                new_by_robot = {k: list(v) for k, v in beam['by_robot'].items()}
                new_by_robot.setdefault(robot, []).append(placed)
                new_beams.append({
                    'score': float(beam['score']) + float(cand['score']) - penalty,
                    'events': beam['events'] + [placed],
                    'by_robot': new_by_robot,
                })
        if not new_beams:
            continue
        new_beams.sort(key=lambda x: x['score'], reverse=True)
        beams = new_beams[:max(1, int(beam_width))]

    best = max(beams, key=lambda x: x['score'])
    final_events = []
    for evt in sorted(best['events'], key=lambda x: (x['robot_id'], x['start_bin'], x['task_type'])):
        robot_schedule = [e for e in final_events if e['robot_id'] == evt['robot_id']]
        if all(not _overlap(evt['start_bin'], evt['duration_bins'], other['start_bin'], other['duration_bins']) for other in robot_schedule):
            final_events.append(evt)

    # Si alguna tarea no cabe sin solape, mantenemos el primer candidato del beam.
    selected_keys = {(e['task_idx'], e['slot_id'], e['robot_id']) for e in final_events}
    for evt in sorted(best['events'], key=lambda x: x['score'], reverse=True):
        key = (evt['task_idx'], evt['slot_id'], evt['robot_id'])
        if key not in selected_keys:
            final_events.append(evt)
            selected_keys.add(key)

    return sorted(final_events, key=lambda x: (x['robot_id'], x['start_bin'], x['duration_bins']))
