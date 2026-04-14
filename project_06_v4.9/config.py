"""
Unified YAML-based configuration loader.

Loads base.yaml, merges overrides from a specified config file,
and exposes the result as a nested SimpleNamespace for attribute access.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


# ── helpers ──────────────────────────────────────────────────────────
def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Convert a nested dict to a nested SimpleNamespace."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        elif isinstance(value, list):
            setattr(ns, key, [
                _dict_to_namespace(v) if isinstance(v, dict) else v
                for v in value
            ])
        else:
            setattr(ns, key, value)
    return ns


def _namespace_to_dict(ns: SimpleNamespace) -> dict:
    """Convert a nested SimpleNamespace back to a dict."""
    d: dict[str, Any] = {}
    for key, value in vars(ns).items():
        if isinstance(value, SimpleNamespace):
            d[key] = _namespace_to_dict(value)
        elif isinstance(value, list):
            d[key] = [
                _namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v
                for v in value
            ]
        else:
            d[key] = value
    return d


# ── main loader ──────────────────────────────────────────────────────
def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Load config from YAML, resolving ``_base_`` inheritance."""
    if path is None:
        path = os.environ.get("PREDICTOR_CONFIG", str(_DEFAULT_CONFIG_PATH))
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Resolve _base_ chain
    if "_base_" in raw:
        base_path = path.parent / raw["_base_"]
        base_cfg = load_config(base_path)
        base_dict = _namespace_to_dict(base_cfg)
        raw = _deep_merge(base_dict, raw)

    cfg = _dict_to_namespace(raw)

    # Inject PROJECT_ROOT
    cfg.project_root = PROJECT_ROOT
    return cfg


# ── derived helpers ──────────────────────────────────────────────────
def num_time_bins(cfg: SimpleNamespace) -> int:
    return (7 * 24 * 60) // cfg.project.bin_minutes


def bins_per_day(cfg: SimpleNamespace) -> int:
    return (24 * 60) // cfg.project.bin_minutes


def num_day_classes() -> int:
    return 7


def num_time_of_day_classes(cfg: SimpleNamespace) -> int:
    return bins_per_day(cfg)


# ── device resolver ──────────────────────────────────────────────────
def resolve_device(cfg: SimpleNamespace) -> "torch.device":
    import torch

    dev = cfg.runtime.device
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)
