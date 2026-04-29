from __future__ import annotations

import shutil
import sys
import textwrap
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


class TerminalLiveBlock:
    """Bloque vivo de terminal: se imprime una vez y luego se redibuja encima."""

    def __init__(self, *, enabled: bool = True, width: int | None = None) -> None:
        self.enabled = bool(enabled)
        self.width = width
        self._rendered_lines = 0

    def _terminal_width(self) -> int:
        if self.width is not None:
            return max(20, int(self.width))

        # -1 evita que algunos terminales hagan wrap justo al final de línea.
        return max(20, shutil.get_terminal_size(fallback=(120, 20)).columns - 1)

    def _split_rendered_lines(self, text: str) -> list[str]:
        width = self._terminal_width()
        rendered: list[str] = []

        for raw_line in str(text).splitlines() or ['']:
            if len(raw_line) <= width:
                rendered.append(raw_line)
            else:
                rendered.extend(
                    textwrap.wrap(
                        raw_line,
                        width=width,
                        replace_whitespace=False,
                        drop_whitespace=False,
                    ) or ['']
                )

        return rendered

    def update(self, text: str) -> None:
        if not self.enabled:
            print(text, flush=True)
            return

        lines = self._split_rendered_lines(text)

        if self._rendered_lines > 0:
            # El cursor está justo debajo del bloque anterior.
            # Subimos al inicio del bloque anterior.
            sys.stdout.write(f'\x1b[{self._rendered_lines}F')

            # Borramos desde ahí hasta abajo.
            # Esto evita que queden líneas vacías si el bloque nuevo es más pequeño.
            sys.stdout.write('\x1b[J')

        # Pintamos el bloque nuevo.
        sys.stdout.write('\n'.join(lines))
        sys.stdout.write('\n')
        sys.stdout.flush()

        self._rendered_lines = len(lines)

    def finish(self) -> None:
        """Deja el último bloque visible y evita que futuros prints lo borren."""
        self._rendered_lines = 0


class TerminalProgress:
    """Progreso por terminal sin dependencias externas.

    Muestra un único bloque por fase y actualiza sus valores en sitio cada
    `step_percent` puntos porcentuales.
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
        self._block = TerminalLiveBlock(enabled=self.enabled)

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

        self._block.update(
            f"[{self.label}] {percent:5.1f}% completado | "
            f"queda {remaining_percent:5.1f}% | {done}/{self.total} | "
            f"transcurrido {format_duration(elapsed)}"
        )

        if done >= self.total:
            self._block.finish()
