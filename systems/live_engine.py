from __future__ import annotations

import time
import pandas as pd

from .metabrain.engine_utils import cache_all_brains, extract_features_at_t as _extract_features_at_t
from .metabrain.arbiter import run_arbiter
from .utils.regime import tag_regime_at, load_regime_settings


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
        regime_cfg = load_regime_settings()
        brain_cache = cache_all_brains(df)
        idx = len(df) - 1
        features = _extract_features_at_t(brain_cache, idx)
        features["regime"] = tag_regime_at(df, idx=idx, **regime_cfg)
        decision, _, _, feat_snapshot = run_arbiter(
            features, position_state="flat", return_score=True
        )
        last_candle = df.iloc[-1]
        print(
            f"[LIVE] Decision={decision} at candle={last_candle['candle_index']} "
            f"regime={feat_snapshot.get('regime')}"
        )
        sleep_until_next_candle()
