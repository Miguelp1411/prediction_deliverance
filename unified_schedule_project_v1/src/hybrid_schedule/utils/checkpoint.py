from __future__ import annotations

from pathlib import Path
from typing import Any

import torch



def save_checkpoint(path: str | Path, model, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'metadata': metadata}, path)



def read_checkpoint(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location or 'cpu')


def load_checkpoint(path: str | Path, model, map_location: str | None = None) -> dict[str, Any]:
    payload = read_checkpoint(path, map_location=map_location)
    model.load_state_dict(payload['state_dict'])
    return payload.get('metadata', {})
