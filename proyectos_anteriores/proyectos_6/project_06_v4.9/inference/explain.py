"""
Explainable inference — provides human-readable explanations
of all prediction decisions.
"""
from __future__ import annotations

from typing import Any


def explain_prediction(result: dict[str, Any], bin_minutes: int = 5) -> str:
    """
    Generate human-readable explanation of a weekly prediction.

    Takes the output from predict_week() and produces a narrative.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  Weekly Prediction Explanation")
    lines.append("=" * 60)

    # Template info
    meta = result.get("template_metadata", {})
    lines.append(f"\n  Template strategy: {meta.get('strategy', 'unknown')}")
    if "source_week_idx" in meta:
        lines.append(f"  Source week index: {meta['source_week_idx']}")
    if "source_weeks" in meta:
        lines.append(f"  Top-k similar weeks:")
        for w in meta["source_weeks"][:5]:
            lines.append(f"    - Week {w['idx']} ({w['start']}) score={w['score']:.3f}")

    # Occurrence delta
    lines.append("\n  Occurrence Changes (template → final):")
    template_counts = result.get("template_counts", {})
    final_counts = result.get("final_counts", {})
    deltas = result.get("occurrence_delta", {})
    for task in sorted(final_counts.keys()):
        tpl = template_counts.get(task, 0)
        final = final_counts[task]
        delta = deltas.get(task, 0)
        arrow = "→"
        if delta > 0:
            arrow = f"→ +{delta} →"
        elif delta < 0:
            arrow = f"→ {delta} →"
        lines.append(f"    {task:20s}: {tpl} {arrow} {final}")

    # Schedule
    schedule = result.get("schedule", [])
    lines.append(f"\n  Scheduled Events: {len(schedule)}")
    for ev in schedule[:20]:  # Show first 20
        day = ev["start_bin"] // (1440 // bin_minutes)
        time_bin = ev["start_bin"] % (1440 // bin_minutes)
        hour = (time_bin * bin_minutes) // 60
        minute = (time_bin * bin_minutes) % 60
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_name = days[min(day, 6)]
        tpl_info = ""
        if ev.get("template_start_bin") is not None:
            shift = ev["start_bin"] - ev["template_start_bin"]
            tpl_info = f" (Δ={shift:+d} bins from template)"
        lines.append(
            f"    {ev['task_name']:20s} {day_name} {hour:02d}:{minute:02d} "
            f"dur={ev['duration_bins']*bin_minutes}min "
            f"score={ev.get('score', 0):.3f} [{ev.get('solver_status', '?')}]{tpl_info}"
        )
    if len(schedule) > 20:
        lines.append(f"    ... and {len(schedule) - 20} more events")

    # Validation
    validation = result.get("validation", {})
    lines.append(f"\n  Validation:")
    lines.append(f"    Valid: {validation.get('is_valid', '?')}")
    lines.append(f"    Same-device overlaps: {validation.get('overlap_same_device_count', '?')}")
    lines.append(f"    Events placed: {validation.get('events_placed', '?')}")

    lines.append("")
    return "\n".join(lines)
