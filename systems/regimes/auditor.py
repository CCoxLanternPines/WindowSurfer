from __future__ import annotations

"""Regime threshold auditor.

This utility analyses historical candle data to suggest balanced
``slope_eps`` and ``vol_eps`` thresholds. Suggestions are written to
``settings/regime_suggestions.json`` so users may manually apply them to
``settings/settings.json``. A dynamic, data-driven approach is used so
that the resulting regimes are roughly balanced across large datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is on the path when executed as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from systems.sim_engine import (
    parse_timeframe,
    apply_time_filter,
    infer_candle_seconds_from_filename,
)


def _load_config() -> tuple[int, int, int, int]:
    """Return configuration for the auditor.

    Returns
    -------
    tuple
        ``(slope_window, vol_window, slope_percentile, vol_percentile)``
    """
    path = Path("settings/settings.json")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    regime = data.get("regime_settings", {})
    detector = regime.get("detector", {})
    audit = data.get("regime_audit", {})

    slope_window = int(detector.get("slope_window", 30))
    vol_window = int(detector.get("vol_window", 30))
    slope_pct = int(audit.get("slope_percentile", 65))
    vol_pct = int(audit.get("vol_percentile", 60))

    return slope_window, vol_window, slope_pct, vol_pct


def _calc_slope(series: pd.Series) -> float:
    if series.size < 2:
        return 0.0
    x = np.arange(series.size)
    return float(np.polyfit(x, series.values, 1)[0])


def audit_history(symbol: str, time: str = "", apply: bool = False) -> dict[str, Any]:
    """Audit candle history for ``symbol`` and return suggestions.

    Parameters
    ----------
    symbol:
        Market symbol to audit.
    time:
        Optional timeframe limitation (e.g. ``"2y"``).
    apply:
        If ``True``, suggested thresholds are written back to
        ``settings/settings.json``.
    """
    slope_window, vol_window, slope_pct, vol_pct = _load_config()

    data_dir = Path("data/sim")
    file_path = data_dir / f"{symbol}_1h.csv"
    if not file_path.exists():
        raise SystemExit(f"missing data file: {file_path}")

    df = pd.read_csv(file_path)

    delta = parse_timeframe(time)
    if delta is not None:
        filtered = apply_time_filter(df, delta, str(file_path))
        if filtered.empty:
            sec = infer_candle_seconds_from_filename(str(file_path)) or 3600
            need = int(delta.total_seconds() // sec)
            filtered = df.tail(need)
        df = filtered

    closes = df["close"]

    slope_short = closes.rolling(slope_window, min_periods=1).apply(
        _calc_slope, raw=False
    )
    slope_long = closes.rolling(slope_window * 3, min_periods=1).apply(
        _calc_slope, raw=False
    )

    use_long = slope_long.abs() > slope_short.abs()
    slope = slope_short.where(~use_long, slope_long)

    returns = closes.pct_change()
    vol = returns.rolling(vol_window, min_periods=1).std()

    slope_eps = float(np.percentile(slope.abs().dropna(), slope_pct))
    vol_eps = float(np.percentile(vol.dropna(), vol_pct))

    regimes = np.where(
        slope > slope_eps,
        "trend_up",
        np.where(
            slope < -slope_eps,
            "trend_down",
            np.where(vol >= vol_eps, "chop", "flat"),
        ),
    )
    counts = pd.Series(regimes).value_counts(normalize=True)
    distribution = counts.reindex(["trend_up", "trend_down", "chop", "flat"], fill_value=0.0)

    print(f"Suggested slope_eps = {slope_eps:.6f}")
    print(f"Suggested vol_eps   = {vol_eps:.6f}")
    print("Regime distribution with these:")
    for r, pct in distribution.items():
        print(f"  {r:<10}: {pct * 100:.0f}%")

    suggestions = {
        "slope_eps": slope_eps,
        "vol_eps": vol_eps,
        "distribution": distribution.to_dict(),
    }
    out_path = Path("settings/regime_suggestions.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(suggestions, fh, indent=2)

    if apply:
        settings_path = Path("settings/settings.json")
        with settings_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        cfg.setdefault("regime_settings", {})["slope_eps"] = slope_eps
        cfg["regime_settings"]["vol_eps"] = vol_eps
        det = cfg["regime_settings"].setdefault("detector", {})
        det["slope_eps"] = slope_eps
        det["vol_eps"] = vol_eps
        with settings_path.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    return suggestions


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime threshold auditor")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--time", default="")
    parser.add_argument("--apply", action="store_true", help="Write suggestions to settings.json")
    args = parser.parse_args()

    audit_history(args.symbol, args.time, args.apply)


if __name__ == "__main__":
    main()
