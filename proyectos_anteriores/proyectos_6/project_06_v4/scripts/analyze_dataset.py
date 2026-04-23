from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from proyectos_anteriores.proyectos_6.project_06_v4.data.io import load_tasks_dataframe


def count_overlaps(df: pd.DataFrame, by_device: bool = True) -> int:
    groups = ['device_uid'] if by_device and 'device_uid' in df.columns else [None]
    overlaps = 0
    if groups == [None]:
        group_iter = [(None, df)]
    else:
        group_iter = df.groupby('device_uid', dropna=False)
    for _, g in group_iter:
        g = g.sort_values('start_time')
        prev_end = None
        for row in g.itertuples(index=False):
            start = row.start_time
            end = row.end_time
            if prev_end is not None and start < prev_end:
                overlaps += 1
                prev_end = max(prev_end, end)
            else:
                prev_end = end
    return overlaps


def main():
    parser = argparse.ArgumentParser(description='Analiza el dataset para detectar paralelismo, duración y densidad semanal.')
    parser.add_argument('--data', required=True, help='Ruta al JSON de entrada')
    parser.add_argument('--out', default=None, help='Ruta opcional para guardar un resumen JSON')
    args = parser.parse_args()

    df = load_tasks_dataframe(args.data)
    df['week_start'] = (df['start_time'].dt.normalize() - pd.to_timedelta(df['start_time'].dt.dayofweek, unit='D'))
    df['device_uid'] = df.get('device_uid')

    summary = {
        'rows': int(len(df)),
        'tasks': sorted(df['task_name'].astype(str).unique().tolist()),
        'num_tasks': int(df['task_name'].nunique()),
        'num_devices': int(df['device_uid'].dropna().astype(str).nunique()) if 'device_uid' in df.columns else 0,
        'global_overlap_count': int(count_overlaps(df, by_device=False)),
        'same_device_overlap_count': int(count_overlaps(df, by_device=True)),
        'duration_minutes_min': float(df['duration_minutes'].min()) if len(df) else 0.0,
        'duration_minutes_max': float(df['duration_minutes'].max()) if len(df) else 0.0,
        'duration_minutes_mean': float(df['duration_minutes'].mean()) if len(df) else 0.0,
        'duration_minutes_std': float(df['duration_minutes'].std(ddof=0)) if len(df) else 0.0,
        'events_per_week_mean': float(df.groupby('week_start').size().mean()) if len(df) else 0.0,
        'events_per_week_max': int(df.groupby('week_start').size().max()) if len(df) else 0,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
