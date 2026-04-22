from __future__ import annotations

import pandas as pd

from .base import BaseAdapter


class APIAdapter(BaseAdapter):
    def load(self, entry: dict) -> pd.DataFrame:
        raise NotImplementedError(
            'APIAdapter es un esqueleto. Integra aquí tu cliente HTTP interno y normaliza la respuesta a DataFrame.'
        )
