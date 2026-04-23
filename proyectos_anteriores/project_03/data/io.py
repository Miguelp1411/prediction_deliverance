import json
from pathlib import Path

import pandas as pd


def load_tasks_dataframe(path: str | Path, timezone: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero de datos: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("El JSON debe contener una lista de tareas")

    df = pd.DataFrame(raw)
    required = {"task_name", "start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el JSON: {sorted(missing)}")

    df = df[["task_name", "start_time", "end_time"]].copy()
    df["task_name"] = df["task_name"].astype(str)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

    if timezone:
        df["start_time"] = df["start_time"].dt.tz_convert(timezone)
        df["end_time"] = df["end_time"].dt.tz_convert(timezone)

    df = df.sort_values("start_time").reset_index(drop=True)
    df["duration_minutes"] = (
        (df["end_time"] - df["start_time"]).dt.total_seconds() / 60.0
    )
    df = df[df["duration_minutes"] >= 0.0].copy()
    df.reset_index(drop=True, inplace=True)
    return df
