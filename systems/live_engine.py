from __future__ import annotations

import time
import numpy as np
import pandas as pd

from .metabrain.engine_utils import cache_all_brains, trade_all_brains


WINDOW_SIZE = 24
WINDOW_STEP = 2
LOOKBACK = WINDOW_SIZE


def normalized_angle(series: pd.Series, lookback: int) -> float:
    """Return slope angle normalized to [-1,1] over the lookback."""
    dy = series.iloc[-1] - series.iloc[0]
    dx = lookback
    angle = np.arctan2(dy, dx)
    norm = angle / (np.pi / 4)
    return max(-1.0, min(1.0, norm))


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
