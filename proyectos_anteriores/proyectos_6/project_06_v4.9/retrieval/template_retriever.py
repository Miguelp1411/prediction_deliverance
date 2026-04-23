"""
Template retriever — finds similar historical weeks for a target week.

Uses cosine similarity on week feature vectors, with configurable
lag candidates and similarity features.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from data.preprocessing import build_week_feature_vector
from data.schema import PreparedData, WeekRecord


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return dot / (norm_a * norm_b)


class TemplateRetriever:
    """Retrieve similar historical weeks for a target week."""

    def __init__(self, prepared: PreparedData) -> None:
        self.prepared = prepared
        # Pre-compute feature vectors for all weeks
        self._feature_cache: dict[int, np.ndarray] = {}
        for w in prepared.weeks:
            self._feature_cache[w.week_index] = build_week_feature_vector(
                w, prepared.num_tasks
            )

    def get_lag_week(self, target_idx: int, lag: int) -> WeekRecord | None:
        """Return the week at target_idx - lag, or None."""
        idx = target_idx - lag
        if 0 <= idx < len(self.prepared.weeks):
            return self.prepared.weeks[idx]
        return None

    def get_similar_weeks(
        self,
        target_idx: int,
        top_k: int = 5,
        exclude_recent: int = 1,
        max_history: int | None = None,
        database_id: str | None = None,
    ) -> list[tuple[WeekRecord, float]]:
        """
        Find the top-k most similar historical weeks.

        Returns list of (WeekRecord, similarity_score) sorted by score desc.
        """
        target_vec = self._feature_cache.get(target_idx)
        if target_vec is None:
            return []

        candidates: list[tuple[int, float]] = []
        min_idx = 0
        if max_history is not None:
            min_idx = max(0, target_idx - max_history)

        for idx in range(min_idx, target_idx - exclude_recent + 1):
            week = self.prepared.weeks[idx]
            # Optional: filter by database
            if database_id and week.database_id != database_id:
                continue

            hist_vec = self._feature_cache.get(idx)
            if hist_vec is None:
                continue

            sim = _cosine_similarity(target_vec, hist_vec)
            candidates.append((idx, sim))

        # Sort by similarity (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        results: list[tuple[WeekRecord, float]] = []
        for idx, score in candidates[:top_k]:
            results.append((self.prepared.weeks[idx], score))

        return results

    def get_same_week_of_year(
        self,
        target_idx: int,
        top_k: int = 3,
    ) -> list[tuple[WeekRecord, float]]:
        """Find weeks with the same week-of-year in previous years."""
        target_week = self.prepared.weeks[target_idx]
        target_woy = target_week.week_start.isocalendar()[1]

        candidates: list[tuple[int, float]] = []
        for idx in range(target_idx):
            week = self.prepared.weeks[idx]
            woy = week.week_start.isocalendar()[1]
            if woy == target_woy:
                # Score by recency (more recent = higher)
                recency = 1.0 / max(target_idx - idx, 1)
                candidates.append((idx, recency))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [
            (self.prepared.weeks[idx], score)
            for idx, score in candidates[:top_k]
        ]
