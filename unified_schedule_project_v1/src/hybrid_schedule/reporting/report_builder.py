from __future__ import annotations

from pathlib import Path


def build_final_report(summary: dict, output_dir: str | Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = summary.get('profile', {})
    train = summary.get('training', {})
    val = summary.get('validation', {})
    text = f"""# Informe final del proyecto unificado

## Resumen del dataset

- eventos: **{profile.get('num_events', 'n/a')}**
- bases de datos: **{profile.get('num_databases', 'n/a')}**
- robots: **{profile.get('num_robots', 'n/a')}**
- semanas: **{profile.get('num_weeks', 'n/a')}**

## Arquitectura

Se ha sustituido la pipeline de dos redes (`occurrence` + `temporal`) por un único modelo de gran capacidad que opera a nivel de semana completa y slot potencial. Ese modelo decide simultáneamente:

- si un slot debe existir,
- en qué día debe ir,
- en qué minuto del día debe empezar,
- cuánto debe durar.

## Mejor métrica de validación registrada

- best_epoch: **{train.get('best_epoch', 'n/a')}**
- best_val_loss: **{train.get('best_val_loss', 'n/a')}**
- val active_f1: **{val.get('active_f1', 0):.2f}%**
- val count_mae: **{val.get('count_mae', 0):.3f}**
- val start_mae_minutes: **{val.get('start_mae_minutes', 0):.2f} min**
- val duration_mae_minutes: **{val.get('duration_mae_minutes', 0):.2f} min**

## Interpretación

La mejora conceptual frente al diseño original es que desaparece el error de acoplamiento entre conteo y colocación temporal. En este diseño, la misma red ve la semana completa, compite internamente por slots, y se decodifica con restricciones de ocupación en un único paso de inferencia.
"""
    (output_dir / 'final_report.md').write_text(text, encoding='utf-8')
    return text
