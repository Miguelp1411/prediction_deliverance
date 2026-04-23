from __future__ import annotations

from pathlib import Path



def build_final_report(summary: dict, output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    occ = summary.get('occurrence', {})
    tmp = summary.get('temporal', {})
    back = summary.get('backtest', {})
    profile = summary.get('profile', {})
    leave_one_out = summary.get('leave_one_database_out', {})

    db_lines = []
    for db, metrics in sorted(leave_one_out.items()):
        db_lines.append(
            f"- **{db}**: task_f1={metrics.get('task_f1', 0):.2f}% | time_close_10m={metrics.get('time_close_accuracy_10m', 0):.2f}% | start_mae={metrics.get('start_mae_minutes', 0):.2f} min"
        )
    db_block = '\n'.join(db_lines) if db_lines else '- n/a'

    text = f"""# Final training report

## Dataset summary

- eventos: **{profile.get('num_events', 'n/a')}**
- bases de datos: **{profile.get('num_databases', 'n/a')}**
- robots: **{profile.get('num_robots', 'n/a')}**
- semanas: **{profile.get('num_weeks', 'n/a')}**
- overlap histórico mismo robot: **{profile.get('overlap_same_robot_count', 'n/a')}**

## Occurrence model (causal + ensemble)

- selector count_exact_acc: **{occ.get('count_exact_acc', 0):.2f}%**
- selector close_acc_1: **{occ.get('close_acc_1', 0):.2f}%**
- selector close_acc_2: **{occ.get('close_acc_2', 0):.2f}%**
- selector count_mae: **{occ.get('count_mae', 0):.3f}**
- selector score: **{occ.get('selector_score', 0):.3f}**

## Temporal model (matching bipartito + día/hora absolutos)

- selector start_exact_acc: **{tmp.get('start_exact_acc', 0):.2f}%**
- selector start_tol_5m: **{tmp.get('start_tol_5m', 0):.2f}%**
- selector start_tol_10m: **{tmp.get('start_tol_10m', 0):.2f}%**
- selector start_mae_minutes: **{tmp.get('start_mae_minutes', 0):.2f} min**
- selector score: **{tmp.get('selector_score', 0):.3f}**

## Weekly holdout backtest

- task_f1: **{back.get('task_f1', 0):.2f}%**
- time_exact_accuracy: **{back.get('time_exact_accuracy', 0):.2f}%**
- time_close_accuracy_5m: **{back.get('time_close_accuracy_5m', 0):.2f}%**
- time_close_accuracy_10m: **{back.get('time_close_accuracy_10m', 0):.2f}%**
- start_mae_minutes: **{back.get('start_mae_minutes', 0):.2f} min**
- overlap_same_robot_count: **{back.get('overlap_same_robot_count', 0):.0f}**

## Leave-one-database-out

{db_block}

## Interpretación

La versión final elimina fuga de información en retrieval y temporal, selecciona checkpoints por backtest walk-forward real, construye la plantilla semanal por consenso causal de vecinos, combina conteos con un ensemble histórico+estacional+residual, predice tiempos como día/hora absolutos con matching bipartito y resuelve la coherencia final con un scheduler exacto compatible con CP-SAT.
"""
    (output_dir / 'final_report.md').write_text(text, encoding='utf-8')
    return text
