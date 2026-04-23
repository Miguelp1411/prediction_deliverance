"""
Database registry — manages multiple data sources.

Loads events from multiple databases via adapters, assigns database_id,
and provides a unified interface for the pipeline.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from data.adapters.json_adapter import load_json_events
from data.schema import Event


class DatabaseRegistry:
    """Registry of all data sources available for training/inference."""

    def __init__(self) -> None:
        self._databases: dict[str, dict[str, Any]] = {}
        self._events: dict[str, list[Event]] = {}

    @property
    def database_ids(self) -> list[str]:
        return list(self._databases.keys())

    @property
    def num_databases(self) -> int:
        return len(self._databases)

    def register(
        self,
        database_id: str,
        path: str | Path | None = None,
        adapter: str = "json",
        timezone: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a new database source."""
        self._databases[database_id] = {
            "path": str(path) if path else None,
            "adapter": adapter,
            "timezone": timezone,
            **kwargs,
        }

    def load(self, database_id: str) -> list[Event]:
        """Load events for a registered database."""
        if database_id in self._events:
            return self._events[database_id]

        info = self._databases.get(database_id)
        if info is None:
            raise KeyError(f"Database '{database_id}' not registered")

        adapter = info["adapter"]
        if adapter == "json":
            events = load_json_events(
                path=info["path"],
                database_id=database_id,
                timezone=info.get("timezone"),
            )
        else:
            raise ValueError(f"Unsupported adapter: {adapter}")

        self._events[database_id] = events
        return events

    def load_all(self) -> dict[str, list[Event]]:
        """Load events from all registered databases."""
        for db_id in self._databases:
            self.load(db_id)
        return dict(self._events)

    def all_events(self) -> list[Event]:
        """Return a single flat list of all events from all databases."""
        self.load_all()
        result: list[Event] = []
        for events in self._events.values():
            result.extend(events)
        result.sort(key=lambda e: (e.start_time, e.end_time, e.task_type))
        return result


def build_registry_from_config(cfg: SimpleNamespace) -> DatabaseRegistry:
    """Build a DatabaseRegistry from the config's data.databases list."""
    registry = DatabaseRegistry()
    databases = getattr(cfg.data, "databases", [])
    project_root = getattr(cfg, "project_root", Path.cwd())

    for db_entry in databases:
        if isinstance(db_entry, SimpleNamespace):
            db_id = db_entry.id
            db_path = Path(db_entry.path)
        elif isinstance(db_entry, dict):
            db_id = db_entry["id"]
            db_path = Path(db_entry["path"])
        elif isinstance(db_entry, str):
            db_path = Path(db_entry)
            db_id = db_path.stem
        else:
            continue

        if not db_path.is_absolute():
            db_path = project_root / db_path

        timezone = getattr(cfg.project, "timezone", None)
        registry.register(database_id=db_id, path=db_path, timezone=timezone)

    return registry
