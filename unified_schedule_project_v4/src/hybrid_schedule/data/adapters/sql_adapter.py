from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .base import BaseAdapter


class SQLiteAdapter(BaseAdapter):
    def load(self, entry: dict) -> pd.DataFrame:
        base_dir = Path(entry['_registry_dir'])
        path = Path(entry['path'])
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        query = entry.get('query')
        table = entry.get('table')
        if not query:
            if not table:
                raise ValueError('Para sqlite debes definir query o table en el registry')
            query = f'SELECT * FROM {table}'
        con = sqlite3.connect(path)
        try:
            return pd.read_sql_query(query, con)
        finally:
            con.close()
