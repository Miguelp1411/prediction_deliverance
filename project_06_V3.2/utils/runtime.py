from __future__ import annotations

import torch


def get_auto_device() -> torch.device:
    accelerator = getattr(torch, 'accelerator', None)
    if accelerator is not None and accelerator.is_available():
        return accelerator.current_accelerator()
    return torch.device('cpu')


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name and device_name.lower() != 'auto':
        return torch.device(device_name)
    return get_auto_device()
