from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseAdapter(ABC):
    @abstractmethod
    def load(self, entry: dict) -> pd.DataFrame:
        raise NotImplementedError
