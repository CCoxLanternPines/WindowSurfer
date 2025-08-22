from __future__ import annotations
import argparse, json
import pandas as pd
from systems.utils.config import load_settings
from systems.utils.resolve_symbol import resolve_data_path
from systems.brains import get_brain


def load_candles(path, timeframe):
    # your existing loader; simplified here
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True)
    ap.add_argument("--ledger", default="Kris_Ledger")
    ap.add_argument("--time", default="1y")
    args = ap.parse_args()

    settings = load_settings()
    path = resolve_data_path(settings, args.ledger)
    df = load_candles(path, args.time)

    brain = get_brain(args.brain)
    out = brain.compute(df, settings)
    truth = brain.truth()
    for name, fn in truth.items():
        stats = fn(df, out.features)
        print(f"{name}: {stats}")


if __name__ == "__main__":
    main()
