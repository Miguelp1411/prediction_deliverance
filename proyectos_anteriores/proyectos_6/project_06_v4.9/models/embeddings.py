"""Shared embedding layers for task, database, and device."""
from __future__ import annotations

import torch
import torch.nn as nn


class SharedEmbeddings(nn.Module):
    """Shared embedding layers used by both occurrence and temporal models."""

    def __init__(
        self,
        num_tasks: int,
        num_databases: int,
        task_embed_dim: int = 16,
        db_embed_dim: int = 8,
    ) -> None:
        super().__init__()
        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.db_embedding = nn.Embedding(max(num_databases, 1), db_embed_dim)
        self.task_embed_dim = task_embed_dim
        self.db_embed_dim = db_embed_dim

    def embed_task(self, task_id: torch.Tensor) -> torch.Tensor:
        return self.task_embedding(task_id)

    def embed_database(self, db_id: torch.Tensor) -> torch.Tensor:
        return self.db_embedding(db_id)
