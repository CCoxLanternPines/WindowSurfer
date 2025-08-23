from __future__ import annotations

"""Utility runner for modular brain files."""

import importlib
from typing import Any

import pandas as pd
from pathlib import Path

from .sim_engine import parse_timeframe, apply_time_filter


def list_brains() -> list[str]:
    """Return sorted list of available brain modules."""
    brains_dir = Path(__file__).parent / "brains"
    return sorted(
        p.stem for p in brains_dir.glob("*.py") if not p.name.startswith("_")
    )


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
    summary: dict[str, Any] = mod.summarize(signals, df)
    stats: dict[str, Any] = summary.get("stats", summary)
    brain_name = summary.get("brain", name)
    print(
        f"[BRAIN][{brain_name}][{timeframe}] "
        f"count={stats.get('count')} avg_gap={stats.get('avg_gap')}c "
        f"slope_bias={stats.get('slope_bias')}"
    )
