from __future__ import annotations

"""Plot trades from a ledger alongside candle data."""

import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd

from systems.utils.config import load_account_settings


# ---------------------------------------------------------------------------
# Internal helpers


def _validate(account: str, market: str) -> None:
    """Exit if ``account`` or ``market`` is missing from config."""
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


def plot_trades_from_ledger(account: str, market: str, mode: str) -> None:
    """Plot candles with trade markers from ledger data.

    Parameters
    ----------
    account: str
        Account name from ``account_settings.json``.
    market: str
        Market identifier (e.g. ``DOGEUSD``).
    mode: str
        ``"sim"`` for simulation or ``"live"`` for live trading.
    """

    _validate(account, market)

    if mode == "sim":
        ledger_path = Path("data/ledgers/ledger_simulation.json")
        candles_path = Path("data/candles/sim") / f"{market}.csv"
    elif mode == "live":
        ledger_path = Path("data/ledgers") / f"{account}_{market}.json"
        candles_path = Path("data/candles/live") / f"{market}.csv"
    else:  # pragma: no cover - defensive
        raise ValueError("mode must be 'sim' or 'live'")

    if not ledger_path.exists():
        print(f"[ERROR] Ledger not found at {ledger_path}")
        return
    if not candles_path.exists():
        print(f"[ERROR] Candles not found at {candles_path}")
        return

    with ledger_path.open("r", encoding="utf-8") as fh:
        ledger = json.load(fh)
    df = pd.read_csv(candles_path)

    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(df["timestamp"], df["close"], label="Close", color="blue")

    entries: List[dict] = ledger.get("entries", [])
    buys_x: List[float] = []
    buys_y: List[float] = []
    sells_x: List[float] = []
    sells_y: List[float] = []
    passes_x: List[float] = []
    passes_y: List[float] = []
    press_times: List[float] = []
    press_buy: List[float] = []
    press_sell: List[float] = []

    for e in entries:
        ts = e.get("timestamp")
        price = e.get("price")
        side = e.get("side")
        if ts is None or price is None or side is None:
            continue
        if side == "BUY":
            buys_x.append(ts)
            buys_y.append(price)
        elif side == "SELL":
            sells_x.append(ts)
            sells_y.append(price)
        else:
            passes_x.append(ts)
            passes_y.append(price)
        pb = e.get("pressure_buy")
        ps = e.get("pressure_sell")
        if pb is not None or ps is not None:
            press_times.append(ts)
            press_buy.append(pb)
            press_sell.append(ps)

    if buys_x:
        ax.scatter(buys_x, buys_y, color="green", marker="^", label="BUY")
    if sells_x:
        ax.scatter(sells_x, sells_y, color="red", marker="v", label="SELL")
    if passes_x:
        ax.scatter(passes_x, passes_y, color="gray", marker=".", label="PASS")

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")

    if press_times:
        ax2 = ax.twinx()
        if any(p is not None for p in press_buy):
            ax2.plot(press_times, press_buy, color="green", alpha=0.3, label="Buy Pressure")
        if any(p is not None for p in press_sell):
            ax2.plot(press_times, press_sell, color="red", alpha=0.3, label="Sell Pressure")
        ax2.set_ylabel("Pressure")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2)
    else:
        ax.legend()

    title = f"{account} {market} ({mode})"
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

