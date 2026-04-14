"""API adapter stub — ready for future REST API data sources."""
from __future__ import annotations

from data.schema import Event


def load_api_events(
    url: str,
    database_id: str,
    headers: dict | None = None,
    timezone: str | None = None,
) -> list[Event]:
    """Placeholder for REST API-based event loading."""
    raise NotImplementedError(
        "API adapter not yet implemented. Provide a URL that returns "
        "JSON with fields: task_type, start_time, end_time, device_uid."
    )
