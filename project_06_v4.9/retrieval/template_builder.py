"""
Template builder — constructs a base agenda from retrieved template weeks.

Supports multiple strategies: lag52_copy, best_similar, topk_blend,
weighted_blend.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from data.schema import EventRecord, PreparedData, WeekRecord
from retrieval.template_retriever import TemplateRetriever


def _copy_events(
    source_week: WeekRecord,
    task_names: list[str],
) -> tuple[list[EventRecord], dict[str, int]]:
    """Copy all events from a source week, returning events and counts."""
    events: list[EventRecord] = []
    counts: dict[str, int] = defaultdict(int)
    for task_id, task_events in source_week.events_by_task.items():
        for ev in task_events:
            events.append(EventRecord(
                task_id=ev.task_id,
                task_name=ev.task_name,
                start_bin=ev.start_bin,
                duration_minutes=ev.duration_minutes,
                start_time=ev.start_time,
                end_time=ev.end_time,
                device_id=ev.device_id,
                database_id=ev.database_id,
                robot_id=ev.robot_id,
            ))
            counts[ev.task_name] += 1
    return events, dict(counts)


def _blend_counts(
    week_counts: list[dict[str, int]],
    weights: list[float],
    task_names: list[str],
) -> dict[str, int]:
    """Weighted average of task counts, rounded to nearest int."""
    total_weight = sum(weights)
    if total_weight < 1e-8:
        return {t: 0 for t in task_names}

    result: dict[str, float] = {t: 0.0 for t in task_names}
    for counts, w in zip(week_counts, weights):
        for task in task_names:
            result[task] += counts.get(task, 0) * (w / total_weight)

    return {t: round(v) for t, v in result.items()}


class TemplateBuilder:
    """Build a base agenda from retrieved templates."""

    def __init__(self, prepared: PreparedData, retriever: TemplateRetriever) -> None:
        self.prepared = prepared
        self.retriever = retriever

    def build_template(
        self,
        target_week_idx: int,
        strategy: str = "topk_blend",
        top_k: int = 5,
        lag_candidates: list[int] | None = None,
    ) -> tuple[list[EventRecord], dict[str, int], dict[str, Any]]:
        """
        Build a template for the target week.

        Returns:
            - template_events: list of EventRecord from the template
            - template_counts: dict of task_name -> count
            - metadata: explanation dict (which week(s) used, scores, etc.)
        """
        if lag_candidates is None:
            lag_candidates = [52, 26, 4, 1]

        if strategy == "lag52_copy":
            return self._lag_copy(target_week_idx, lag=52)
        elif strategy == "best_similar":
            return self._best_similar(target_week_idx)
        elif strategy == "topk_blend":
            return self._topk_blend(target_week_idx, top_k)
        elif strategy == "weighted_blend":
            return self._weighted_lag_blend(target_week_idx, lag_candidates)
        else:
            raise ValueError(f"Unknown template strategy: {strategy}")

    def _lag_copy(
        self,
        target_idx: int,
        lag: int = 52,
    ) -> tuple[list[EventRecord], dict[str, int], dict[str, Any]]:
        """Copy the week from lag weeks ago."""
        source = self.retriever.get_lag_week(target_idx, lag)
        if source is None:
            # Fallback: try shorter lags
            for fallback_lag in [26, 4, 1]:
                source = self.retriever.get_lag_week(target_idx, fallback_lag)
                if source is not None:
                    lag = fallback_lag
                    break

        if source is None:
            return [], {t: 0 for t in self.prepared.task_names}, {"strategy": "empty", "reason": "no_history"}

        events, counts = _copy_events(source, self.prepared.task_names)
        metadata = {
            "strategy": f"lag{lag}_copy",
            "source_week_idx": source.week_index,
            "source_week_start": str(source.week_start),
        }
        return events, counts, metadata

    def _best_similar(
        self,
        target_idx: int,
    ) -> tuple[list[EventRecord], dict[str, int], dict[str, Any]]:
        """Use the single most similar historical week."""
        similar = self.retriever.get_similar_weeks(target_idx, top_k=1)
        if not similar:
            return self._lag_copy(target_idx)

        best_week, score = similar[0]
        events, counts = _copy_events(best_week, self.prepared.task_names)
        metadata = {
            "strategy": "best_similar",
            "source_week_idx": best_week.week_index,
            "source_week_start": str(best_week.week_start),
            "similarity_score": score,
        }
        return events, counts, metadata

    def _topk_blend(
        self,
        target_idx: int,
        top_k: int = 5,
    ) -> tuple[list[EventRecord], dict[str, int], dict[str, Any]]:
        """
        Blend the top-k similar weeks.

        Uses the MOST similar week's events as the template,
        but adjusts counts based on weighted average of top-k.
        """
        similar = self.retriever.get_similar_weeks(target_idx, top_k=top_k)
        if not similar:
            return self._lag_copy(target_idx)

        # Use best week's events as the base
        best_week, best_score = similar[0]
        events, _ = _copy_events(best_week, self.prepared.task_names)

        # Blend counts from all top-k
        all_counts = []
        all_weights = []
        for week, score in similar:
            _, counts = _copy_events(week, self.prepared.task_names)
            all_counts.append(counts)
            all_weights.append(max(score, 0.0))

        blended_counts = _blend_counts(all_counts, all_weights, self.prepared.task_names)

        metadata = {
            "strategy": "topk_blend",
            "top_k": len(similar),
            "source_weeks": [
                {"idx": w.week_index, "start": str(w.week_start), "score": s}
                for w, s in similar
            ],
            "base_week_idx": best_week.week_index,
        }
        return events, blended_counts, metadata

    def _weighted_lag_blend(
        self,
        target_idx: int,
        lag_candidates: list[int],
    ) -> tuple[list[EventRecord], dict[str, int], dict[str, Any]]:
        """Blend counts from multiple lag weeks with fixed weights."""
        default_weights = {52: 0.50, 26: 0.20, 4: 0.20, 1: 0.10}

        all_counts = []
        all_weights = []
        best_source = None

        for lag in lag_candidates:
            source = self.retriever.get_lag_week(target_idx, lag)
            if source is not None:
                _, counts = _copy_events(source, self.prepared.task_names)
                all_counts.append(counts)
                all_weights.append(default_weights.get(lag, 0.1))
                if best_source is None:
                    best_source = source

        if not all_counts:
            return [], {t: 0 for t in self.prepared.task_names}, {"strategy": "empty"}

        events, _ = _copy_events(best_source, self.prepared.task_names)
        blended_counts = _blend_counts(all_counts, all_weights, self.prepared.task_names)

        metadata = {
            "strategy": "weighted_lag_blend",
            "lags_used": lag_candidates,
        }
        return events, blended_counts, metadata
