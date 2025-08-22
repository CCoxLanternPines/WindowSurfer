from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any

import pandas as pd


@dataclass
class Overlay:
    x: List[int]
    y: List[float]
    s: List[float] | None = None
    c: str | None = None


class Brain:
    name: str = "base"

    def prepare(self, df: pd.DataFrame) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, df: pd.DataFrame, t: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def warmup(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def overlays(self) -> Dict[str, Dict[str, List[Any]]]:  # pragma: no cover - interface
        raise NotImplementedError

    def compute_stats(
        self, df: pd.DataFrame, trend_state: List[int], slopes: List[float]
    ) -> Dict[str, Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError
