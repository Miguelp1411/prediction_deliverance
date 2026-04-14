from __future__ import annotations

from pathlib import Path



def build_final_report(summary: dict, output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    occ = summary.get('occurrence', {})
    tmp = summary.get('temporal', {})
    back = summary.get('backtest', {})
    profile = summary.get('profile', {})

    text = f"""# Final training report

## Dataset summary

- eventos: **{profile.get('num_events', 'n/a')}**
- bases de datos: **{profile.get('num_databases', 'n/a')}**
- robots: **{profile.get('num_robots', 'n/a')}**
- semanas: **{profile.get('num_weeks', 'n/a')}**
- overlap histórico mismo robot: **{profile.get('overlap_same_robot_count', 'n/a')}**

## Occurrence residual model

- val count_exact_acc: **{occ.get('count_exact_acc', 0):.2f}%**
- val close_acc_1: **{occ.get('close_acc_1', 0):.2f}%**
- val close_acc_2: **{occ.get('close_acc_2', 0):.2f}%**
- val count_mae: **{occ.get('count_mae', 0):.3f}**
- val change_acc: **{occ.get('change_acc', 0):.2f}%**

## Temporal residual model

- val start_exact_acc: **{tmp.get('start_exact_acc', 0):.2f}%**
- val start_tol_5m: **{tmp.get('start_tol_5m', 0):.2f}%**
- val start_tol_10m: **{tmp.get('start_tol_10m', 0):.2f}%**
- val start_mae_minutes: **{tmp.get('start_mae_minutes', 0):.2f} min**
- val duration_mae_minutes: **{tmp.get('duration_mae_minutes', 0):.2f} min**

## Weekly backtest

- task_f1: **{back.get('task_f1', 0):.2f}%**
- time_exact_accuracy: **{back.get('time_exact_accuracy', 0):.2f}%**
- time_close_accuracy_5m: **{back.get('time_close_accuracy_5m', 0):.2f}%**
- time_close_accuracy_10m: **{back.get('time_close_accuracy_10m', 0):.2f}%**
- start_mae_minutes: **{back.get('start_mae_minutes', 0):.2f} min**
- overlap_same_robot_count: **{back.get('overlap_same_robot_count', 0):.0f}**

## Interpretación

La arquitectura final usa una semana plantilla como base, corrige cantidades con un modelo residual y corrige tiempos con un modelo residual top-k. La coherencia semanal final se resuelve con un scheduler MILP exacto, lo que evita depender de heurísticas locales como estrategia principal.
"""
    (output_dir / 'final_report.md').write_text(text, encoding='utf-8')
    return text
