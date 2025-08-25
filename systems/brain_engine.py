from __future__ import annotations

"""Utility runner for modular brain files."""

import importlib
from typing import Any

import pandas as pd

from .sim_engine import parse_timeframe, apply_time_filter


def run_brain(name: str, timeframe: str, viz: bool) -> None:
    """Load candles and execute a specific brain module."""
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)

    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    mod = importlib.import_module(f"systems.brains.{name}")
    signals = mod.run(df, viz)
    stats: dict[str, Any] = mod.summarize(signals, df)
    print(
        f"[BRAIN][{name}][{timeframe}] "
        f"count={stats['count']} avg_gap={stats['avg_gap']}c "
        f"slope_bias={stats['slope_bias']}"
    )
