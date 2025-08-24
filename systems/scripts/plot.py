from __future__ import annotations

"""Utility for plotting candles with trade markers from ledger data."""

from pathlib import Path
import json
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd

from systems.utils.load_config import load_config


def _validate_account_market(account: str, market: str) -> None:
    """Raise ``SystemExit`` if account or market are not configured."""
    cfg = load_config()
    accounts = cfg.get("accounts", {})
    if account not in accounts:
        print(f"[ERROR] Unknown account: {account}")
        raise SystemExit(1)
    markets = accounts[account].get("markets", {})
    if market not in markets:
        print(f"[ERROR] Unknown market: {market} for account {account}")
        raise SystemExit(1)


def plot_trades_from_ledger(account: str, market: str, mode: str) -> None:
    """Plot candles with BUY/SELL/PASS markers from ledger data."""
    _validate_account_market(account, market)

    market_file = market
    if mode == "sim":
        ledger_path = Path("data/ledgers/ledger_simulation.json")
        candles_path = Path("data/candles/sim") / f"{market_file}.csv"
    elif mode == "live":
        ledger_path = Path("data/ledgers") / f"{account}_{market_file}.json"
        candles_path = Path("data/candles/live") / f"{market_file}.csv"
    else:
        raise ValueError("mode must be 'sim' or 'live'")

    df = pd.read_csv(candles_path)
    times = pd.to_datetime(df["timestamp"], unit="s")
    fig, ax = plt.subplots()
    ax.plot(times, df["close"], label="Close", color="blue")

    try:
        with ledger_path.open("r", encoding="utf-8") as fh:
            ledger: dict[str, Any] = json.load(fh)
    except FileNotFoundError:
        ledger = {}

    buys_x: list[int] = []
    buys_y: list[float] = []
    sells_x: list[int] = []
    sells_y: list[float] = []
    pass_x: list[int] = []
    pass_y: list[float] = []
    press_buy_x: list[int] = []
    press_buy_y: list[float] = []
    press_sell_x: list[int] = []
    press_sell_y: list[float] = []

    for entry in ledger.get("entries", []):
        ts = entry.get("timestamp")
        price = entry.get("price")
        side = entry.get("side")
        if ts is None or price is None or side is None:
            continue
        if side == "BUY":
            buys_x.append(ts)
            buys_y.append(price)
        elif side == "SELL":
            sells_x.append(ts)
            sells_y.append(price)
        else:
            pass_x.append(ts)
            pass_y.append(price)
        if "pressure_buy" in entry:
            press_buy_x.append(ts)
            press_buy_y.append(entry["pressure_buy"])
        if "pressure_sell" in entry:
            press_sell_x.append(ts)
            press_sell_y.append(entry["pressure_sell"])

    if buys_x:
        ax.scatter(pd.to_datetime(buys_x, unit="s"), buys_y, color="green", marker="^", label="BUY")
    if sells_x:
        ax.scatter(pd.to_datetime(sells_x, unit="s"), sells_y, color="red", marker="v", label="SELL")
    if pass_x:
        ax.scatter(pd.to_datetime(pass_x, unit="s"), pass_y, color="gray", marker=".", label="PASS")

    if press_buy_x or press_sell_x:
        ax2 = ax.twinx()
        if press_buy_x:
            ax2.plot(pd.to_datetime(press_buy_x, unit="s"), press_buy_y, color="purple", alpha=0.3, label="pressure_buy")
        if press_sell_x:
            ax2.plot(pd.to_datetime(press_sell_x, unit="s"), press_sell_y, color="orange", alpha=0.3, label="pressure_sell")
        ax2.set_ylabel("Pressure")
        ax2.legend(loc="upper right")

    ax.legend(loc="upper left")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
