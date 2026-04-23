from __future__ import annotations

import pandas as pd



def _pick_column(df: pd.DataFrame, explicit: str | None, candidates: list[str], required: bool = True) -> str | None:
    if explicit and explicit in df.columns:
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f'No se encontró ninguna columna válida entre: {candidates}')
    return None



def normalize_events(df: pd.DataFrame, entry: dict, timezone_default: str = 'Europe/Madrid') -> pd.DataFrame:
    task_col = _pick_column(df, entry.get('task_field'), ['task_type', 'type', 'task_name'])
    robot_col = _pick_column(df, entry.get('robot_field'), ['robot_id', 'device_uid', 'device_id'], required=False)
    start_col = _pick_column(df, entry.get('start_field'), ['start_time', 'start', 'start_at'])
    end_col = _pick_column(df, entry.get('end_field'), ['end_time', 'end', 'end_at'])
    event_id_col = _pick_column(df, entry.get('event_id_field'), ['uid', 'id', 'event_id'], required=False)

    timezone = str(entry.get('timezone') or timezone_default)
    task_values = df[task_col].astype(str).str.strip()
    start_values = pd.to_datetime(df[start_col], utc=True)
    end_values = pd.to_datetime(df[end_col], utc=True)
    robot_values = (df[robot_col].fillna('__default_robot__').astype(str) if robot_col else pd.Series(['__default_robot__'] * len(df), index=df.index))
    event_ids = (df[event_id_col].astype(str) if event_id_col else pd.Series([f"{entry['database_id']}_{i}" for i in range(len(df))], index=df.index))

    out = pd.DataFrame({
        'database_id': [str(entry['database_id'])] * len(df),
        'robot_id': robot_values.values,
        'task_type': task_values.values,
        'start_time': start_values.values,
        'end_time': end_values.values,
        'timezone': [timezone] * len(df),
        'source_event_id': event_ids.values,
    })

    out = out[out['task_type'].notna() & (out['task_type'] != '')].copy()
    out['start_time'] = pd.to_datetime(out['start_time'], utc=True).dt.tz_convert(timezone)
    out['end_time'] = pd.to_datetime(out['end_time'], utc=True).dt.tz_convert(timezone)
    out = out[out['end_time'] >= out['start_time']].copy()
    out['duration_minutes'] = (out['end_time'] - out['start_time']).dt.total_seconds() / 60.0
    out['week_start'] = out['start_time'].dt.normalize() - pd.to_timedelta(out['start_time'].dt.dayofweek, unit='D')
    out['day_of_week'] = out['start_time'].dt.dayofweek.astype(int)
    out['minute_of_day'] = out['start_time'].dt.hour * 60 + out['start_time'].dt.minute
    out = out.sort_values(['database_id', 'robot_id', 'start_time', 'end_time', 'task_type']).reset_index(drop=True)
    return out
