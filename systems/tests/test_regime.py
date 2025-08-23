from __future__ import annotations

"""Proof-of-concept regime detector harness."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root is on the path when executed as a script
sys.path.append(str(Path(__file__).resolve().parents[2]))

from systems.regimes import define_regimes, detect_regime
from systems.sim_engine import (
    parse_timeframe,
    apply_time_filter,
    infer_candle_seconds_from_filename,
)


def _load_config() -> dict[str, Any]:
    path = Path("settings/settings.json")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("regime_settings", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime detector harness")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--time", default="")
    args = parser.parse_args()

    data_dir = Path("data/sim")
    file_path = data_dir / f"{args.symbol}_1h.csv"
    if not file_path.exists():
        raise SystemExit(f"missing data file: {file_path}")

    df = pd.read_csv(file_path)

    delta = parse_timeframe(args.time)
    if delta is not None:
        filtered = apply_time_filter(df, delta, str(file_path))
        if filtered.empty:
            sec = infer_candle_seconds_from_filename(str(file_path)) or 3600
            need = int(delta.total_seconds() // sec)
            filtered = df.tail(need)
        df = filtered

    df = df.reset_index(drop=True)
    df = define_regimes(df)

    cfg = _load_config()
    window = int(cfg.get("window", 50))

    guesses: list[str] = []
    for t in range(len(df)):
        start = max(0, t - window + 1)
        sub = df.iloc[start : t + 1]
        guess = detect_regime(sub)
        guesses.append(guess)
    df["regime_guess"] = guesses

    valid = df["regime_true_shifted"].notna()
    total = int(valid.sum())
    accuracy = (
        df.loc[valid, "regime_guess"]
        == df.loc[valid, "regime_true_shifted"]
    ).mean() * 100

    regimes = ["trend_up", "trend_down", "chop", "flat"]
    cm = pd.crosstab(
        df.loc[valid, "regime_true_shifted"],
        df.loc[valid, "regime_guess"],
        rownames=["true"],
        colnames=["pred"],
    ).reindex(index=regimes, columns=regimes, fill_value=0)

    print(f"Total candles: {total}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("Precision/Recall per regime:")
    for r in regimes:
        tp = cm.loc[r, r]
        precision = tp / cm[r].sum() if cm[r].sum() > 0 else 0.0
        recall = tp / cm.loc[r].sum() if cm.loc[r].sum() > 0 else 0.0
        print(f"  {r}: P={precision:.2f}, R={recall:.2f}")


if __name__ == "__main__":
    main()
