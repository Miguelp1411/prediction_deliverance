from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / 'epoch_metrics.jsonl'
        self.rows: list[dict[str, Any]] = []

    def log_epoch(self, payload: dict[str, Any]) -> None:
        self.rows.append(dict(payload))
        with self.jsonl_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def save_csv(self) -> None:
        if not self.rows:
            return
        keys = sorted({k for row in self.rows for k in row.keys()})
        path = self.output_dir / 'epoch_metrics.csv'
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)


def save_summary(summary: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
