import json
from pathlib import Path

import pandas as pd


def load_tasks_dataframe_from_records(raw: list[dict], timezone: str | None = None) -> pd.DataFrame:
    if not isinstance(raw, list):
        raise ValueError("El JSON debe contener una lista de tareas")

    df = pd.DataFrame(raw)

    if 'type' in df.columns and df['type'].notna().any():
        label_col = 'type'
    elif 'task_name' in df.columns and df['task_name'].notna().any():
        label_col = 'task_name'
    else:
        raise ValueError("El JSON debe tener una columna 'type' o 'task_name' con valores válidos")

    required = {label_col, 'start_time', 'end_time'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el JSON: {sorted(missing)}")

    optional_cols = [c for c in ['device_uid'] if c in df.columns]
    df = df[[label_col, 'start_time', 'end_time', *optional_cols]].copy()
    df = df.rename(columns={label_col: 'task_name'})
    df = df[df['task_name'].notna()].copy()
    df['task_name'] = df['task_name'].astype(str).str.strip()
    df = df[df['task_name'] != ''].copy()

    df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
    df['end_time'] = pd.to_datetime(df['end_time'], utc=True)

    if timezone:
        df['start_time'] = df['start_time'].dt.tz_convert(timezone)
        df['end_time'] = df['end_time'].dt.tz_convert(timezone)

    if 'device_uid' not in df.columns:
        df['device_uid'] = None
    df['device_uid'] = df['device_uid'].astype('object')
    df = df.sort_values(['start_time', 'end_time', 'task_name']).reset_index(drop=True)
    df['duration_minutes'] = ((df['end_time'] - df['start_time']).dt.total_seconds() / 60.0)
    df = df[df['duration_minutes'] >= 0.0].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def load_tasks_dataframe(path: str | Path, timezone: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de datos: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return load_tasks_dataframe_from_records(raw, timezone=timezone)