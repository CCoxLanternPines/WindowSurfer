from __future__ import annotations

import importlib
import pandas as pd

from .sim_engine import parse_timeframe, apply_time_filter
from .metabrain.extractor import extract_features
from .metabrain.arbiter import run_arbiter


def run_metabrain(timeframe: str = "6m", viz: bool = True) -> None:
    file_path = "data/sim/SOLUSD_1h.csv"
    df = pd.read_csv(file_path)
    delta = parse_timeframe(timeframe)
    if delta is not None:
        df = apply_time_filter(df, delta, file_path)
    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    brain_modules = [
        "exhaustion",
        "reversal",
        "momentum_inflection",
        "bottom_catcher",
        "divergence",
        "rolling_peak",
    ]

    all_brains: dict[str, dict] = {}
    for mod_name in brain_modules:
        mod = importlib.import_module(f"systems.brains.{mod_name}")
        signals = mod.run(df, viz=False)
        summary = mod.summarize(signals, df)
        key = summary.get("brain", mod_name)
        all_brains[key] = summary

    features = extract_features(all_brains)
    decision = run_arbiter(features, position_state="flat")

    print(f"[METABRAIN][{timeframe}] Decision={decision}")
    print(" Features snapshot:", features)

    if viz:
        import matplotlib.pyplot as plt

        plt.plot(df["candle_index"], df["close"], lw=1, color="blue")
        if decision == "BUY":
            plt.scatter(
                df["candle_index"].iloc[-1],
                df["close"].iloc[-1],
                color="green",
                marker="^",
                s=150,
                zorder=6,
            )
        elif decision == "SELL":
            plt.scatter(
                df["candle_index"].iloc[-1],
                df["close"].iloc[-1],
                color="red",
                marker="v",
                s=150,
                zorder=6,
            )
        plt.show()

