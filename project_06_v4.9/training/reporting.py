"""
Training reporting — generates reports during and after training.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class TrainingReporter:
    """Generates comprehensive training reports."""

    def __init__(self, reports_dir: str | Path) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_history: list[dict] = []
        self._start_time = time.time()

    def log_epoch(self, entry: dict[str, Any]) -> None:
        """Log a single epoch's metrics."""
        self._epoch_history.append(entry)

        # Append to JSONL
        jsonl_path = self.reports_dir / "training_log.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def save_csv(self, filename: str = "training_log.csv") -> None:
        """Save epoch history as CSV."""
        if not self._epoch_history:
            return
        csv_path = self.reports_dir / filename
        keys = list(self._epoch_history[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for entry in self._epoch_history:
                writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in entry.items()})

    def save_final_report(
        self,
        model_name: str,
        best_epoch: int,
        best_metrics: dict[str, float],
        baseline_results: dict[str, dict[str, float]] | None = None,
        ablation_results: dict[str, dict[str, float]] | None = None,
        per_db_results: dict[str, dict[str, float]] | None = None,
        per_task_results: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Generate final training reports (JSON + Markdown)."""
        total_time = time.time() - self._start_time

        report = {
            "model_name": model_name,
            "total_epochs": len(self._epoch_history),
            "best_epoch": best_epoch,
            "total_training_time_seconds": total_time,
            "best_metrics": best_metrics,
            "baseline_comparison": baseline_results or {},
            "ablation_results": ablation_results or {},
            "per_database_results": per_db_results or {},
            "per_task_results": per_task_results or {},
            "epoch_history": self._epoch_history,
        }

        # JSON report
        json_path = self.reports_dir / "training_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        # Markdown report
        md_path = self.reports_dir / "training_report.md"
        self._write_markdown_report(md_path, report)

        # Final summary
        summary_path = self.reports_dir / "final_summary.md"
        self._write_summary(summary_path, report)

        self.save_csv()

        print(f"\n  Reports saved to: {self.reports_dir}")

    def _write_markdown_report(self, path: Path, report: dict) -> None:
        """Write detailed Markdown training report."""
        lines = [
            f"# Training Report: {report['model_name']}",
            "",
            f"- **Total epochs**: {report['total_epochs']}",
            f"- **Best epoch**: {report['best_epoch']}",
            f"- **Training time**: {report['total_training_time_seconds']:.1f}s",
            "",
            "## Best Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in report["best_metrics"].items():
            lines.append(f"| {k} | {v:.4f} |")

        if report.get("baseline_comparison"):
            lines.extend(["", "## Baseline Comparison", "", "| Baseline | Metric | Value |", "|----------|--------|-------|"])
            for baseline, metrics in report["baseline_comparison"].items():
                for k, v in metrics.items():
                    lines.append(f"| {baseline} | {k} | {v:.4f} |")

        if report.get("per_database_results"):
            lines.extend(["", "## Per-Database Results", "", "| Database | Metric | Value |", "|----------|--------|-------|"])
            for db, metrics in report["per_database_results"].items():
                for k, v in metrics.items():
                    lines.append(f"| {db} | {k} | {v:.4f} |")

        if report.get("per_task_results"):
            lines.extend(["", "## Per-Task Results", "", "| Task | Metric | Value |", "|------|--------|-------|"])
            for task, metrics in report["per_task_results"].items():
                for k, v in metrics.items():
                    lines.append(f"| {task} | {k} | {v:.4f} |")

        lines.extend(["", "## Learning Curve", ""])
        if report.get("epoch_history"):
            lines.extend(["| Epoch | Train Loss | Val Loss | Monitor |", "|-------|------------|----------|---------|"])
            for entry in report["epoch_history"]:
                lines.append(
                    f"| {entry.get('epoch', '?')} | {entry.get('train_loss', 0):.4f} | "
                    f"{entry.get('val_loss', 0):.4f} | {entry.get('monitor_value', 0):.4f} |"
                )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _write_summary(self, path: Path, report: dict) -> None:
        """Write concise final summary."""
        lines = [
            f"# Final Summary: {report['model_name']}",
            "",
            f"Trained for **{report['total_epochs']}** epochs, "
            f"best at epoch **{report['best_epoch']}**.",
            "",
            "## Key Results",
            "",
        ]
        for k, v in report["best_metrics"].items():
            lines.append(f"- **{k}**: {v:.4f}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
