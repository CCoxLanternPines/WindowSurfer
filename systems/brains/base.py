from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class BrainOutput:
    """Container for arbitrary feature arrays."""

    # arbitrary arrays, len == len(df); used by viz + truth
    features: Dict[str, Any]


class Brain:
    """Abstract brain – implement in concrete modules."""

    name: str = "base"
    # optional defaults under settings["general_settings"]["strategy_settings"]
    settings_key: Optional[str] = None

    def compute(self, df: pd.DataFrame, settings: Dict[str, Any]) -> BrainOutput:
        raise NotImplementedError

    def visualize(self, df: pd.DataFrame, out: BrainOutput, ax: plt.Axes) -> None:
        """Draw on provided axes. No returns."""
        raise NotImplementedError

    # Truth questions return dict[name -> callable] that consumes (df, out) and returns stats dict
    def truth(self) -> Dict[str, Any]:
        return {}
