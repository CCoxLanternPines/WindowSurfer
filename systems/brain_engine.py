from __future__ import annotations

"""Utility runner for modular brain files."""

import importlib
from typing import Any

import pandas as pd
from pathlib import Path

from .sim_engine import parse_timeframe, apply_time_filter
from .utils.regime import compute_regimes, load_regime_settings


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

    regime_cfg = load_regime_settings()
    regime_df = compute_regimes(df, **regime_cfg)
    counts: dict[str, int] = {}
    for s in signals:
        idx = s.get("index") or s.get("candle_index")
        if idx is None or idx not in regime_df.index:
            continue
        trend = regime_df.loc[idx, "trend"]
        vol = regime_df.loc[idx, "vol"]
        key = f"{trend}/{vol}"
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total:
        rev = stats.get("reversal_pct")
        cont = stats.get("continuation_pct")
        if rev is not None and cont is not None:
            print(f"  Global resolution: {rev}% rev / {cont}% cont")
        print("  By regime:")
        for key, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            pct = 100 * cnt / total
            print(f"    {key}: {pct:.0f}% ({cnt}/{total})")
