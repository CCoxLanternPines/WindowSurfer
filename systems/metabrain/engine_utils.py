from __future__ import annotations

import importlib
from .extractor import extract_features
from .arbiter import run_arbiter

BRAIN_MODULES = [
    "exhaustion",
    "reversal",
    "momentum_inflection",
    "bottom_catcher",
    "divergence",
    "rolling_peak",
]

def cache_all_brains(df):
    """Run all brains on a dataframe, return dict of summaries and signals."""
    all_brains = {}
    for mod_name in BRAIN_MODULES:
        mod = importlib.import_module(f"systems.brains.{mod_name}")
        signals = mod.run(df, viz=False)
        summary = mod.summarize(signals, df)
        key = summary.get("brain", mod_name)
        all_brains[key] = summary
    return all_brains

def trade_all_candles(all_brains, df):
    """SIM mode: walk candles and generate decision timeline."""
    features = extract_features(all_brains)
    # For now just apply arbiter once on whole set, expand later to step-by-step
    decision = run_arbiter(features, position_state="flat")
    return decision

def trade_all_brains(all_brains, last_candle, position_state="flat"):
    """LIVE mode: decide based on most recent candle only."""
    features = extract_features(all_brains)
    decision = run_arbiter(features, position_state=position_state)
    return decision


def simulate_metabrain(df, position_state="flat"):
    """Step through candles, make MetaBrain decisions at each step."""
    brain_modules = [
        "exhaustion",
        "reversal",
        "momentum_inflection",
        "bottom_catcher",
        "divergence",
        "rolling_peak",
    ]

    buy_signals, sell_signals = [], []
    state = position_state

    for t in range(50, len(df)):
        sub_df = df.iloc[: t + 1].reset_index(drop=True)
        all_brains = {}
        for mod_name in brain_modules:
            mod = importlib.import_module(f"systems.brains.{mod_name}")
            signals = mod.run(sub_df, viz=False)
            summary = mod.summarize(signals, sub_df)
            key = summary.get("brain", mod_name)
            all_brains[key] = summary

        features = extract_features(all_brains)
        decision = run_arbiter(features, state)

        x = int(df["candle_index"].iloc[t])
        y = float(df["close"].iloc[t])
        if state == "flat" and decision == "BUY":
            buy_signals.append((x, y))
            state = "long"
        elif state == "long" and decision == "SELL":
            sell_signals.append((x, y))
            state = "flat"

    return buy_signals, sell_signals
