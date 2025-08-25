"""Visualization engine for simulation outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from systems.scripts.chart import plot_trades


def render_simulation(sim_path: str) -> None:
    """Load simulation data from ``sim_path`` and render plots."""
    path = Path(sim_path)
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    df = pd.DataFrame(data.get("df", {}))
    pts: Dict[str, Dict[str, List[float]]] = data.get("pts", {})
    vol_pts: Dict[str, List[float]] = data.get("vol_pts", {})
    trades: List[Dict[str, float]] = data.get("trades", [])
    meta: Dict[str, Any] = data.get("meta", {})
    start_capital = float(meta.get("start_capital", 0.0))
    final_value = float(meta.get("final_value", 0.0))

    plot_trades(df, pts, vol_pts, trades, start_capital, final_value)

