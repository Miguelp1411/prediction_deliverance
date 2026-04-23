from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CANONICAL_COLUMNS = [
    'database_id',
    'robot_id',
    'task_type',
    'start_time',
    'end_time',
    'timezone',
    'source_event_id',
]


@dataclass(slots=True)
class CanonicalEvent:
    database_id: str
    robot_id: str
    task_type: str
    start_time: Any
    end_time: Any
    timezone: str
    source_event_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
