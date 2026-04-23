"""Calendar feature utilities."""
from __future__ import annotations

import math

import numpy as np


def cyclical_encode(value: float, period: float) -> tuple[float, float]:
    """Encode a cyclic value as (sin, cos)."""
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def week_of_year_features(week_number: int) -> tuple[float, float]:
    return cyclical_encode(week_number, 52)


def month_features(month: int) -> tuple[float, float]:
    return cyclical_encode(month, 12)


def day_of_year_features(day_of_year: int) -> tuple[float, float]:
    return cyclical_encode(day_of_year, 365)


def day_of_week_features(day: int) -> tuple[float, float]:
    return cyclical_encode(day, 7)


def hour_features(hour: int) -> tuple[float, float]:
    return cyclical_encode(hour, 24)


def build_calendar_vector(week_of_year: int, month: int, day_of_year: int) -> np.ndarray:
    """Build a 6-dim calendar feature vector."""
    woy_s, woy_c = week_of_year_features(week_of_year)
    m_s, m_c = month_features(month)
    d_s, d_c = day_of_year_features(day_of_year)
    return np.array([woy_s, woy_c, m_s, m_c, d_s, d_c], dtype=np.float32)
