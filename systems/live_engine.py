from __future__ import annotations

import time
import pandas as pd

from .metabrain.engine_utils import cache_all_brains, trade_all_brains


def get_recent_candles(timeframe: str) -> pd.DataFrame:
    """Stub loader for recent candles."""
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)
    df = df.tail(100).reset_index(drop=True)
    df["candle_index"] = range(len(df))
    return df


def sleep_until_next_candle() -> None:
    """Sleep placeholder."""
    time.sleep(1)


def run_live(timeframe: str = "1m") -> None:
    while True:
        df = get_recent_candles(timeframe)
        all_brains = cache_all_brains(df)
        last_candle = df.iloc[-1]
        decision = trade_all_brains(all_brains, last_candle, position_state="flat")
        print(f"[LIVE] Decision={decision} at candle={last_candle['candle_index']}")
        sleep_until_next_candle()
