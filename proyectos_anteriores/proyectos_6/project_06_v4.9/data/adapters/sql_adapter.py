"""SQL adapter stub — ready for future database connections."""
from __future__ import annotations

from data.schema import Event


def load_sql_events(
    connection_string: str,
    query: str,
    database_id: str,
    timezone: str | None = None,
) -> list[Event]:
    """Placeholder for SQL-based event loading."""
    raise NotImplementedError(
        "SQL adapter not yet implemented. Provide a SQLAlchemy connection string "
        "and a query that returns columns: task_type, start_time, end_time, device_uid."
    )
