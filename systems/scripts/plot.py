from __future__ import annotations

"""Plot trades from a ledger alongside candle data."""

import json
import sys
from math import atan, degrees
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd

from systems.utils.config import (
    load_account_settings,
    load_coin_settings,
    resolve_coin_config,
)


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
    market = market.replace("/", "").upper()
    _validate(account, market)

    coin_cfg = resolve_coin_config(market, load_coin_settings())
    buy_trigger = float(coin_cfg.get("buy_trigger", 0.0))
    sell_trigger = float(coin_cfg.get("sell_trigger", 0.0))
    flat_band = float(coin_cfg.get("flat_band_deg", 0.0))

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
    slope_x, slope_y = [], []

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

        feats = entry.get("features", {})
        slope_val = feats.get("slope")
        if slope_val is not None:
            slope_x.append(ts)
            slope_y.append(degrees(atan(float(slope_val))))

    if buys_x:
        ax.scatter(pd.to_datetime(buys_x, unit="s"), buys_y, color="green", marker="^", label="BUY")
    if sells_x:
        ax.scatter(pd.to_datetime(sells_x, unit="s"), sells_y, color="red", marker="v", label="SELL")
    if pass_x:
        ax.scatter(pd.to_datetime(pass_x, unit="s"), pass_y, color="gray", marker=".", label="PASS")

    if slope_x:
        ax2 = ax.twinx()
        slope_times = pd.to_datetime(slope_x, unit="s")
        ax2.plot(slope_times, slope_y, color="gray", label="slope")
        ax2.axhline(buy_trigger, color="green", linestyle="--", label="buy_trigger")
        ax2.axhline(sell_trigger, color="red", linestyle="--", label="sell_trigger")
        if flat_band:
            ax2.fill_between(slope_times, -flat_band, flat_band, color="gray", alpha=0.1, label="flat_band")
        ax2.set_ylabel("Slope (deg)")
        ax2.legend(loc="upper right")

    ax.legend(loc="upper left")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()
