"""
Constraint definitions for the CP-SAT scheduler.

Defines the set of hard and soft constraints for schedule construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SchedulerConstraints:
    """Configuration for all scheduler constraints."""
    # Hard constraints
    no_overlap_per_device: bool = True
    valid_time_range: bool = True      # events must fit within 0..total_bins
    # Soft constraints (penalties)
    penalize_template_deviation: float = 1.0
    penalize_order_disruption: float = 0.5
    prefer_model_score: float = 2.0
    min_gap_bins: int = 0
    # Operating hours (optional)
    operating_hours_constraint: bool = False
    operating_hour_start_bin: int = 0
    operating_hour_end_bin: int = 2016  # 7*24*12 = full week
    # Solver
    solver_time_limit_seconds: int = 30
    # Mode
    is_single_device: bool = True

    @classmethod
    def from_config(cls, cfg_scheduler: Any, bins_per_day: int = 288) -> "SchedulerConstraints":
        """Build constraints from config namespace."""
        sc = cls()
        if hasattr(cfg_scheduler, "no_overlap"):
            sc.no_overlap_per_device = cfg_scheduler.no_overlap
        if hasattr(cfg_scheduler, "penalize_template_deviation"):
            sc.penalize_template_deviation = cfg_scheduler.penalize_template_deviation
        if hasattr(cfg_scheduler, "penalize_order_disruption"):
            sc.penalize_order_disruption = cfg_scheduler.penalize_order_disruption
        if hasattr(cfg_scheduler, "prefer_model_score"):
            sc.prefer_model_score = cfg_scheduler.prefer_model_score
        if hasattr(cfg_scheduler, "min_gap_minutes"):
            sc.min_gap_bins = int(cfg_scheduler.min_gap_minutes / 5)
        if hasattr(cfg_scheduler, "operating_hours_constraint"):
            sc.operating_hours_constraint = cfg_scheduler.operating_hours_constraint
        if hasattr(cfg_scheduler, "solver_time_limit_seconds"):
            sc.solver_time_limit_seconds = cfg_scheduler.solver_time_limit_seconds
        return sc


@dataclass
class EventCandidate:
    """An event to be scheduled by the solver."""
    event_idx: int                     # index in the candidate list
    task_name: str
    task_id: int
    device_id: str
    # Candidate slots: list of (start_bin, score)
    candidates: list[tuple[int, float]] = field(default_factory=list)
    duration_bins: int = 1
    template_start_bin: int | None = None
    preferred_order: int = 0  # original order position
