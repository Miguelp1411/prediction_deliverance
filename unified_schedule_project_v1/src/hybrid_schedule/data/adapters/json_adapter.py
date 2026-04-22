from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .base import BaseAdapter


class JSONAdapter(BaseAdapter):
    def load(self, entry: dict) -> pd.DataFrame:
        base_dir = Path(entry['_registry_dir'])
        path = Path(entry['path'])
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(raw, list):
            raise ValueError(f'El JSON {path} debe contener una lista de eventos')
        df = pd.DataFrame(raw)
        return df
