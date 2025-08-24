from __future__ import annotations

"""Plot trades from a ledger alongside candle data."""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd

from systems.utils.config import load_account_settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(account: str, market: str) -> None:
    """Exit if account or market is missing from config."""
    accounts = load_account_settings()
    acct_cfg = accounts.get(account)
    if not acct_cfg:
        print(f"[ERROR] Unknown account {account}")
        sys.exit(1)
    markets = acct_cfg.get("market settings", {})
    if market not in markets:
        print(f"[ERROR] Unknown market {market} for account {account}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Public plotting API
# ---------------------------------------------------------------------------

def plot_trades_from_ledger(
    account: str, market: str, mode: str, ledger_path: str | None = None
) -> None:
    """Plot candles with BUY/SELL/PASS markers from ledger data."""
    _validate(account, market)

    if ledger_path is None:
        if mode == "sim":
            ledger_path = Path("data/temp/sim_data.json")
        elif mode == "live":
            ledger_path = Path("data/ledgers") / f"{account}_{market}.json"
        else:
            raise ValueError("mode must be 'sim' or 'live'")
    else:
        ledger_path = Path(ledger_path)

    if mode == "sim":
        candles_path = Path("data/candles/sim") / f"{market}.csv"
    elif mode == "live":
        candles_path = Path("data/candles/live") / f"{market}.csv"
    else:
        raise ValueError("mode must be 'sim' or 'live'")

    if not ledger_path.exists():
        print(f"[ERROR] Ledger not found at {ledger_path}")
        return
    if not candles_path.exists():
        print(f"[ERROR] Candles not found at {candles_path}")
        return

    df = pd.read_csv(candles_path)
    times = pd.to_datetime(df["timestamp"], unit="s")
    fig, ax = plt.subplots()
    ax.plot(times, df["close"], label="Close", color="blue")

    try:
        with ledger_path.open("r", encoding="utf-8") as fh:
            ledger: dict[str, Any] = json.load(fh)
    except Exception:
        ledger = {}

    buys_x, buys_y = [], []
    sells_x, sells_y = [], []
    pass_x, pass_y = [], []
    press_buy_x, press_buy_y = [], []
    press_sell_x, press_sell_y = [], []

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
        elif side == "PASS":
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
