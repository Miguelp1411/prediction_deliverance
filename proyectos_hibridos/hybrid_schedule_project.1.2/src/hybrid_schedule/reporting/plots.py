from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



def _plot(df: pd.DataFrame, x: str, ys: list[str], title: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for col in ys:
        if col in df.columns:
            plt.plot(df[x], df[col], label=col)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel('value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def save_training_plots(rows: list[dict], output_dir: str | Path) -> None:
    if not rows:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if 'model' in df.columns:
        for model_name, sub in df.groupby('model'):
            safe = model_name.replace(' ', '_').lower()
            _plot(sub, 'epoch', [c for c in ['train_loss', 'val_loss', 'count_mae', 'start_mae_minutes', 'task_f1'] if c in sub.columns], f'History {model_name}', output_dir / f'{safe}_history.png')
    else:
        _plot(df, 'epoch', [c for c in ['train_loss', 'val_loss'] if c in df.columns], 'Training history', output_dir / 'training_history.png')
