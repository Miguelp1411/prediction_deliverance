"""
Post-check validation for scheduled output.

Verifies the solution is conflict-free, all events are valid,
and produces summary statistics.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _events_overlap(
    start_a: int, dur_a: int, start_b: int, dur_b: int
) -> bool:
    end_a = start_a + dur_a
    end_b = start_b + dur_b
    return start_a < end_b and start_b < end_a


def validate_schedule(
    schedule: list[dict[str, Any]],
    total_weekly_bins: int = 2016,
) -> dict[str, Any]:
    """
    Validate a scheduled output.

    Returns a report with:
      - is_valid: bool
      - overlap_same_device_count: int
      - overlap_global_count: int
      - events_out_of_range: int
      - events_placed: int
      - percent_conflict_free: float
      - details: list of issues
    """
    issues: list[str] = []
    overlap_same_device = 0
    overlap_global = 0
    out_of_range = 0

    # Check range
    for ev in schedule:
        start = ev["start_bin"]
        end = start + ev["duration_bins"]
        if start < 0 or end > total_weekly_bins:
            out_of_range += 1
            issues.append(f"Event {ev['event_idx']} out of range: [{start}, {end})")

    # Check overlaps per device
    by_device: dict[str, list[dict]] = defaultdict(list)
    for ev in schedule:
        by_device[ev.get("device_id", "__default__")].append(ev)

    for device_id, dev_events in by_device.items():
        sorted_evs = sorted(dev_events, key=lambda e: e["start_bin"])
        for i in range(len(sorted_evs)):
            for j in range(i + 1, len(sorted_evs)):
                if _events_overlap(
                    sorted_evs[i]["start_bin"], sorted_evs[i]["duration_bins"],
                    sorted_evs[j]["start_bin"], sorted_evs[j]["duration_bins"],
                ):
                    overlap_same_device += 1
                    issues.append(
                        f"Overlap on {device_id}: events {sorted_evs[i]['event_idx']} "
                        f"and {sorted_evs[j]['event_idx']}"
                    )
                else:
                    break

    # Global overlaps (cross-device, informational)
    sorted_all = sorted(schedule, key=lambda e: e["start_bin"])
    for i in range(len(sorted_all)):
        for j in range(i + 1, len(sorted_all)):
            if sorted_all[j]["start_bin"] >= sorted_all[i]["start_bin"] + sorted_all[i]["duration_bins"]:
                break
            if sorted_all[i].get("device_id") != sorted_all[j].get("device_id"):
                overlap_global += 1

    is_valid = overlap_same_device == 0 and out_of_range == 0

    return {
        "is_valid": is_valid,
        "overlap_same_device_count": overlap_same_device,
        "overlap_global_count": overlap_global,
        "events_out_of_range": out_of_range,
        "events_placed": len(schedule),
        "percent_conflict_free": 100.0 if is_valid else 0.0,
        "details": issues,
    }
