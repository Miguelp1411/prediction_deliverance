from __future__ import annotations

from pathlib import Path

import yaml


SUPPORTED_SOURCE_TYPES = {'json', 'sqlite', 'api'}


def load_registry(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'No existe el registry: {path}')
    payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    datasets = payload.get('datasets', [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError('El registry debe contener una lista no vacía en la clave datasets')
    normalized: list[dict] = []
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError('Cada dataset del registry debe ser un objeto')
        source_type = str(item.get('source_type', '')).strip().lower()
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f'source_type no soportado: {source_type}')
        database_id = str(item.get('database_id', '')).strip()
        if not database_id:
            raise ValueError('Cada dataset debe definir database_id')
        record = dict(item)
        record['_registry_dir'] = str(path.parent.resolve())
        normalized.append(record)
    return normalized
