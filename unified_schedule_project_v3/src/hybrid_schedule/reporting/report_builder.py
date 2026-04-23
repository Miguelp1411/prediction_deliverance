from __future__ import annotations

from pathlib import Path


def build_final_report(summary: dict, output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = summary.get('profile', {})
    train = summary.get('training', {})
    val = summary.get('validation', {})
    back = summary.get('backtest', {})
    text = f"""# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **{profile.get('num_events', 'n/a')}**
- bases de datos: **{profile.get('num_databases', 'n/a')}**
- robots: **{profile.get('num_robots', 'n/a')}**
- semanas: **{profile.get('num_weeks', 'n/a')}**
- overlap histórico mismo robot: **{profile.get('overlap_same_robot_count', 'n/a')}**

## Validación del modelo unificado

- best_epoch: **{train.get('best_epoch', 'n/a')}**
- best_val_loss: **{train.get('best_val_loss', 'n/a')}**
- val active_f1: **{val.get('active_f1', 0):.2f}%**
- val count_mae: **{val.get('count_mae', 0):.3f}**
- val day_acc: **{val.get('day_acc', 0):.2f}%**
- val start_tol_5m: **{val.get('start_tol_5m', 0):.2f}%**
- val start_mae_minutes: **{val.get('start_mae_minutes', 0):.2f} min**
- val duration_mae_minutes: **{val.get('duration_mae_minutes', 0):.2f} min**

## Weekly backtest

- task_f1: **{back.get('task_f1', 0):.2f}%**
- time_exact_accuracy: **{back.get('time_exact_accuracy', 0):.2f}%**
- time_close_accuracy_5m: **{back.get('time_close_accuracy_5m', 0):.2f}%**
- time_close_accuracy_10m: **{back.get('time_close_accuracy_10m', 0):.2f}%**
- start_mae_minutes: **{back.get('start_mae_minutes', 0):.2f} min**
- overlap_same_robot_count: **{back.get('overlap_same_robot_count', 0):.0f}**

## Interpretación

La arquitectura unificada mantiene su validación interna a nivel de slots, pero además incorpora un backtest semanal end-to-end con el mismo matching y las mismas métricas agregadas que el proyecto híbrido. Así, ambos proyectos pueden compararse sobre una base homogénea en el informe final.
"""
    (output_dir / 'final_report.md').write_text(text, encoding='utf-8')
    return text
