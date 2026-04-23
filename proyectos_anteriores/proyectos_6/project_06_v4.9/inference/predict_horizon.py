"""Multi-week rolling prediction."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from data.schema import PreparedData
from inference.predict_week import predict_week


def predict_horizon(
    prepared: PreparedData,
    start_week_idx: int,
    num_weeks: int,
    occurrence_model: torch.nn.Module,
    temporal_model: torch.nn.Module,
    cfg: SimpleNamespace,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Predict multiple weeks in a rolling fashion."""
    results: list[dict[str, Any]] = []
    for offset in range(num_weeks):
        target_idx = start_week_idx + offset
        if target_idx >= len(prepared.weeks):
            break
        result = predict_week(
            prepared, target_idx, occurrence_model, temporal_model, cfg, device
        )
        result["week_index"] = target_idx
        result["week_start"] = str(prepared.weeks[target_idx].week_start)
        results.append(result)
    return results
