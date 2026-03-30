from __future__ import annotations

import torch


import torch

def get_auto_device():
    # 1. Comprueba si hay una GPU NVIDIA (CUDA) disponible
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # 2. Comprueba si hay un chip de Apple (MPS) disponible
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    
    # 3. Si no hay aceleradores, usa el procesador normal
    else:
        return torch.device("cpu")

def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name and device_name.lower() != 'auto':
        return torch.device(device_name)
    return get_auto_device()
