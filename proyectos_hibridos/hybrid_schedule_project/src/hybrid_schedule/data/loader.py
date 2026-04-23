from __future__ import annotations

import pandas as pd

from .adapters.api_adapter import APIAdapter
from .adapters.json_adapter import JSONAdapter
from .adapters.sql_adapter import SQLiteAdapter
from .normalize import normalize_events


ADAPTERS = {
    'json': JSONAdapter(),
    'sqlite': SQLiteAdapter(),
    'api': APIAdapter(),
}


def load_all_events(registry_entries: list[dict], timezone_default: str = 'Europe/Madrid') -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for entry in registry_entries:
        adapter = ADAPTERS[entry['source_type']]
        raw_df = adapter.load(entry)
        frames.append(normalize_events(raw_df, entry, timezone_default=timezone_default))
    if not frames:
        raise ValueError('No se cargó ningún dataset')
    return pd.concat(frames, ignore_index=True)
