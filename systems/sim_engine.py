from __future__ import annotations

"""Simple historical simulation matching mini-bot trade logic."""

import argparse
import re
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd

from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.trade_apply import paper_execute_buy, paper_execute_sell
from systems.utils.config import load_settings


CFG = {
    "window_size": 24,
    "flat_band_deg": 10.0,
    "strong_move_threshold": 0.15,
    "range_min": 0.08,
    "volume_skew_bias": 0.4,
    "max_pressure": 7,
    "buy_trigger": 2,
    "sell_trigger": 4,
    "flat_sell_fraction": 0.2,
    "flat_sell_threshold": 0.5,
}


def parse_timeframe(tf: str) -> timedelta | None:
    match = re.match(r"(\d+)([dhmw])", tf)
    if not match:
        return None
    val, unit = int(match.group(1)), match.group(2)
    if unit == "d":
        return timedelta(days=val)
    if unit == "w":
        return timedelta(weeks=val)
    if unit == "m":
        return timedelta(days=30 * val)  # rough month
    if unit == "h":
        return timedelta(hours=val)
    return None

def run_simulation(series=None, settings=None, ledger=None, **kwargs):
    """Run a mini-bot style simulation ignoring ledger writes.

    ``series`` and ``settings`` are optional for compatibility with the
    command-line invocation via ``bot.py``.  The ``ledger`` parameter is
    retained but ignored when ``disable_ledger`` is true.
    """

    timeframe = kwargs.get("timeframe") or kwargs.get("time") or "1m"
    viz = kwargs.get("viz", False)

    if settings is None:
        settings = load_settings()
    disable_ledger = settings.get("sim", {}).get("disable_ledger", True)

    ledger_name = ledger if isinstance(ledger, str) else None
    if series is None:
        tag = settings.get("ledger_settings", {}).get(ledger_name, {}).get("tag", "SOLUSD")
        file_path = f"data/sim/{tag}_1h.csv"
        df = pd.read_csv(file_path)
    else:
        df = series.copy()

    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            cutoff = (pd.Timestamp.utcnow().tz_localize(None) - delta).timestamp()
            if "timestamp" in df.columns:
                df = df[df["timestamp"] >= cutoff]

    df = df.reset_index(drop=True)
    df["candle_index"] = range(len(df))

    state = {"buy_pressure": 0.0, "sell_pressure": 0.0, "last_features": None}
    open_notes: list[dict[str, float]] = []  # {price, amount}
    realized_pnl = 0.0

    if viz:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df["candle_index"], df["close"], color="blue", label="Close Price")
    else:
        ax1 = None

    for t in range(len(df)):
        price = float(df.iloc[t]["close"])

        buy_res = evaluate_buy(t, df, cfg=CFG, state=state)
        if buy_res:
            size = buy_res["trade_size"]
            paper_execute_buy(price, size)
            open_notes.append({"price": price, "amount": size})
            # if ledger is not None and not disable_ledger:
            #     ledger.log_event({"action": "buy", "price": price, "amount": size})
            if ax1 is not None:
                ax1.scatter(
                    df.iloc[t]["candle_index"], price, color="green", s=120, zorder=6
                )

        sell_orders = evaluate_sell(t, df, cfg=CFG, open_notes=open_notes, state=state)
        for order in sell_orders:
            note = order["note"]
            amt = order["sell_amount"]
            mode = order.get("sell_mode", "normal")
            entry_price = note["price"]

            paper_execute_sell(price, amt)

            pnl = (price - entry_price) * amt
            realized_pnl += pnl

            if amt >= note["amount"] - 1e-9:
                open_notes.remove(note)
            else:
                note["amount"] -= amt

            # if ledger is not None and not disable_ledger:
            #     ledger.log_event({"action": "sell", "price": price, "amount": amt})

            if ax1 is not None:
                color = "red" if mode == "normal" else "orange"
                size = 120 if mode == "normal" else 90
                ax1.scatter(df.iloc[t]["candle_index"], price, color=color, s=size, zorder=6)

    print(f"[RESULT] PnL={realized_pnl:.2f}, Remaining Notes={len(open_notes)}")

    if ax1 is not None:
        ax1.scatter([], [], color="green", s=120, label="Buy")
        ax1.scatter([], [], color="red", s=120, label="Sell")
        ax1.scatter([], [], color="orange", s=90, label="Flat Sell")

        ax1.set_title("Price with Trades")
        ax1.set_xlabel("Candles (Index)")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")
        ax1.grid(True)
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=str, default="1m")
    parser.add_argument("--viz", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    run_simulation(timeframe=args.time, viz=args.viz)


if __name__ == "__main__":
    main()

