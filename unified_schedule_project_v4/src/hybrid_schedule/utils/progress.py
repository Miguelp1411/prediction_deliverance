from __future__ import annotations

import time


def format_duration(seconds: float) -> str:
    """Devuelve una duración legible en formato compacto."""
    seconds = max(0.0, float(seconds))
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class TerminalProgress:
    """Progreso por terminal sin dependencias externas.

    Imprime una línea cada `step_percent` puntos porcentuales, incluyendo el
    porcentaje completado, el porcentaje restante y una estimación temporal.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        step_percent: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.label = str(label)
        self.total = max(1, int(total))
        self.step_percent = max(0.1, float(step_percent))
        self.enabled = bool(enabled)
        self.started_at = time.perf_counter()
        self._last_bucket = -1
        if self.enabled:
            self.update(0)

    def update(self, done: int) -> None:
        if not self.enabled:
            return
        done = min(max(int(done), 0), self.total)
        percent = 100.0 * float(done) / float(self.total)
        bucket = int(percent // self.step_percent)
        if done < self.total and bucket <= self._last_bucket:
            return

        self._last_bucket = bucket
        elapsed = time.perf_counter() - self.started_at
        remaining_percent = max(0.0, 100.0 - percent)
        if done > 0 and done < self.total:
            estimated_total = elapsed * float(self.total) / float(done)
            remaining_seconds = max(0.0, estimated_total - elapsed)
            remaining_text = format_duration(remaining_seconds)
        elif done >= self.total:
            remaining_text = "0s"
        else:
            remaining_text = "calculando"

        print(
            f"[{self.label}] {percent:5.1f}% completado | "
            f"queda {remaining_percent:5.1f}% | {done}/{self.total} | "
            f"transcurrido {format_duration(elapsed)} | restante estimado {remaining_text}",
            flush=True,
        )
