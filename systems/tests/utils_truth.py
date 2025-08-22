import pandas as pd
import numpy as np

from systems.sim_engine import parse_timeframe, apply_time_filter


def load_candles(tag: str, timeframe: str):
    """Load CSV candles for a market tag and apply timeframe filtering."""
    file_path = f"data/sim/{tag}_1h.csv"
    df = pd.read_csv(file_path)
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)
    return df.reset_index(drop=True)


def slope(series) -> float:
    """Return simple linear regression slope of the given series."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    return float(np.polyfit(x, series, 1)[0])


def percent_results(results_dict: dict) -> str:
    """Format boolean result lists as percentage strings."""
    lines = []
    for question, hits in results_dict.items():
        total = len(hits)
        pct = (sum(hits) / total * 100) if total else 0.0
        lines.append(f"{question} = {pct:.0f}%")
    return "\n".join(lines)
