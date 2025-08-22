from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Signal:
    t: int
    price: float
    tag: str
    meta: dict


class Brain:
    name: str = "base"

    def warmup(self) -> int:
        raise NotImplementedError

    def prepare(self, df) -> None:
        raise NotImplementedError

    def step(self, df, t: int) -> None:
        raise NotImplementedError

    def overlays(self) -> Dict[str, Dict[str, List[Any]]]:
        raise NotImplementedError
