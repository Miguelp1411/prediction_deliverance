from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _count_overlaps(group: pd.DataFrame) -> int:
    if group.empty:
        return 0
    g = group.sort_values('start_time')
    starts = g['start_time'].to_numpy()
    ends = g['end_time'].to_numpy()
    count = 0
    for i in range(1, len(g)):
        if starts[i] < ends[i - 1]:
            count += 1
    return int(count)


def profile_events_dataframe(df: pd.DataFrame) -> dict:
    profile: dict = {}
    profile['num_events'] = int(len(df))
    profile['num_databases'] = int(df['database_id'].nunique())
    profile['num_robots'] = int(df['robot_id'].nunique())
    profile['num_tasks'] = int(df['task_type'].nunique())
    profile['tasks'] = sorted(df['task_type'].unique().tolist())
    profile['date_min'] = str(df['start_time'].min()) if not df.empty else None
    profile['date_max'] = str(df['start_time'].max()) if not df.empty else None
    profile['num_weeks'] = int(df['week_start'].nunique()) if not df.empty else 0
    profile['duration_minutes_min'] = float(df['duration_minutes'].min()) if not df.empty else 0.0
    profile['duration_minutes_median'] = float(df['duration_minutes'].median()) if not df.empty else 0.0
    profile['duration_minutes_max'] = float(df['duration_minutes'].max()) if not df.empty else 0.0

    weekly_counts = df.groupby(['database_id', 'robot_id', 'week_start']).size()
    profile['weekly_tasks_mean'] = float(weekly_counts.mean()) if len(weekly_counts) else 0.0
    profile['weekly_tasks_std'] = float(weekly_counts.std()) if len(weekly_counts) else 0.0

    overlaps = defaultdict(int)
    for (database_id, robot_id), group in df.groupby(['database_id', 'robot_id']):
        overlaps[f'{database_id}::{robot_id}'] = _count_overlaps(group)
    profile['overlap_same_robot_count'] = int(sum(overlaps.values()))
    profile['overlap_breakdown'] = dict(overlaps)

    by_db = {}
    for database_id, group in df.groupby('database_id'):
        weeks = group['week_start'].nunique()
        by_db[database_id] = {
            'events': int(len(group)),
            'robots': int(group['robot_id'].nunique()),
            'weeks': int(weeks),
            'tasks': sorted(group['task_type'].unique().tolist()),
            'weekly_mean': float(group.groupby(['robot_id', 'week_start']).size().mean()),
            'duration_median': float(group['duration_minutes'].median()),
        }
    profile['by_database'] = by_db
    return profile


def render_profile_markdown(profile: dict) -> str:
    lines = [
        '# Dataset profile',
        '',
        f"- Eventos: **{profile['num_events']}**",
        f"- Bases de datos: **{profile['num_databases']}**",
        f"- Robots/dispositivos: **{profile['num_robots']}**",
        f"- Tipos de tarea: **{profile['num_tasks']}**",
        f"- Semanas: **{profile['num_weeks']}**",
        f"- Duración min/mediana/max: **{profile['duration_minutes_min']:.1f} / {profile['duration_minutes_median']:.1f} / {profile['duration_minutes_max']:.1f} min**",
        f"- Tareas por semana (media±std): **{profile['weekly_tasks_mean']:.2f} ± {profile['weekly_tasks_std']:.2f}**",
        f"- Overlaps mismo robot: **{profile['overlap_same_robot_count']}**",
        '',
        '## Por base de datos',
        '',
    ]
    for database_id, info in profile.get('by_database', {}).items():
        lines.extend([
            f'### {database_id}',
            '',
            f"- eventos: {info['events']}",
            f"- robots: {info['robots']}",
            f"- semanas: {info['weeks']}",
            f"- tareas: {', '.join(info['tasks'])}",
            f"- media semanal: {info['weekly_mean']:.2f}",
            f"- duración mediana: {info['duration_median']:.2f} min",
            '',
        ])
    return '\n'.join(lines)


def save_profile_report(profile: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    (output_dir / 'dataset_profile.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'dataset_profile.md').write_text(render_profile_markdown(profile), encoding='utf-8')
