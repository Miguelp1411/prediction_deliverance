"""
JSON adapter — loads the project's JSON database format into canonical Events.

Supports autodetection of 'type' vs 'task_name' column style.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.schema import Event

UNKNOWN_DEVICE = "__unknown_device__"


def load_json_events(
    path: str | Path,
    database_id: str | None = None,
    timezone: str | None = None,
) -> list[Event]:
    """Load a JSON file and return a list of canonical Events."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")

    if database_id is None:
        database_id = path.stem

    with open(path, "r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("JSON must contain a list of task records")

    events: list[Event] = []
    for record in raw:
        # Detect task label column
        task_type = record.get("type") or record.get("task_name")
        if not task_type or not str(task_type).strip():
            continue
        task_type = str(task_type).strip()

        start_str = record.get("start_time")
        end_str = record.get("end_time")
        if not start_str or not end_str:
            continue

        start_time = pd.Timestamp(start_str, tz="UTC")
        end_time = pd.Timestamp(end_str, tz="UTC")

        if timezone:
            start_time = start_time.tz_convert(timezone)
            end_time = end_time.tz_convert(timezone)

        duration = (end_time - start_time).total_seconds() / 60.0
        if duration < 0.0:
            continue

        device_id = str(record.get("device_uid", "") or UNKNOWN_DEVICE).strip()
        robot_id = device_id  # In this schema device == robot
        source_id = str(record.get("uid", ""))

        events.append(Event(
            database_id=database_id,
            robot_id=robot_id,
            device_id=device_id,
            task_type=task_type,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
            source_event_id=source_id,
            duration_minutes=duration,
        ))

    events.sort(key=lambda e: (e.start_time, e.end_time, e.task_type))
    return events
