import pandas as pd
import numpy as np

from systems.sim_engine import parse_timeframe, apply_time_filter, WINDOW_SIZE


def load_candles(tag: str, timeframe: str) -> pd.DataFrame:
    """Load CSV candles for a market tag and apply timeframe filtering."""
    file_path = f"data/sim/{tag}_1h.csv"
    df = pd.read_csv(file_path)

    # Apply timeframe filtering to maintain sim/live parity.
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df_filtered = apply_time_filter(df, delta, file_path)
        # If no rows survive (e.g. historical data far in the past),
        # fall back to a simple tail slice to ensure we have data.
        if not df_filtered.empty:
            df = df_filtered
        else:
            df = df.tail(WINDOW_SIZE)

    df = df.reset_index(drop=True)
    df["candle_index"] = np.arange(len(df))
    return df


def slope(series) -> float:
    """Return simple linear regression slope of the given series."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    return float(np.polyfit(x, series, 1)[0])


def percent_results(results: dict) -> str:
    """Format hits/total pairs into percentage strings."""
    lines = []
    for question, (hits, total) in results.items():
        pct = (hits / total * 100) if total else 0.0
        lines.append(f"{question} = {pct:.0f}%")
    return "\n".join(lines)


def run_truth(df: pd.DataFrame, questions, context_fn) -> dict:
    """Iterate candles and evaluate truth-check functions.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data frame.
    questions : List[Tuple[str, Callable]]
        Sequence of (question, fn) pairs where fn(t, df, ctx) -> bool.
    context_fn : Callable
        Function computing reusable context from the dataframe.

    Returns
    -------
    dict
        Mapping question -> (hits, total).
    """

    ctx = context_fn(df)
    results = {q: [0, 0] for q, _ in questions}

    for t in range(len(df)):
        for q, fn in questions:
            try:
                hit = bool(fn(t, df, ctx))
            except Exception:
                hit = False
            if hit:
                results[q][0] += 1
            results[q][1] += 1

    return {q: (h, tot) for q, (h, tot) in results.items()}
