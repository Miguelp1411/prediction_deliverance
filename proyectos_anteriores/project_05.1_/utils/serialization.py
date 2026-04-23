from __future__ import annotations

from pathlib import Path

import torch



def save_checkpoint(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)



def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(Path(path), map_location=map_location)
