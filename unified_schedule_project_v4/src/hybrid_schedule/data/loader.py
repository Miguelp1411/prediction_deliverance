from __future__ import annotations

import pandas as pd

from hybrid_schedule.utils.progress import TerminalProgress

from .adapters.api_adapter import APIAdapter
from .adapters.json_adapter import JSONAdapter
from .adapters.sql_adapter import SQLiteAdapter
from .normalize import normalize_events


ADAPTERS = {
    'json': JSONAdapter(),
    'sqlite': SQLiteAdapter(),
    'api': APIAdapter(),
}


def load_all_events(
    registry_entries: list[dict],
    timezone_default: str = 'Europe/Madrid',
    show_progress: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    progress = TerminalProgress(
        'Preparación datos - carga de fuentes',
        len(registry_entries),
        enabled=show_progress,
    )
    for idx, entry in enumerate(registry_entries, start=1):
        adapter = ADAPTERS[entry['source_type']]
        raw_df = adapter.load(entry)
        frames.append(normalize_events(raw_df, entry, timezone_default=timezone_default))
        progress.update(idx)
    if not frames:
        raise ValueError('No se cargó ningún dataset')
    return pd.concat(frames, ignore_index=True)
