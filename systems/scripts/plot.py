from __future__ import annotations

"""Plot trades from a ledger alongside candle data."""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd
import numpy as np
from math import atan, degrees

from systems.utils.config import load_account_settings, load_coin_settings
from systems.utils.resolve_symbol import split_tag


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
# Truth overlay helper
# ---------------------------------------------------------------------------

def _truth_overlays(df: pd.DataFrame, coin_cfg: dict) -> dict:
    """
    Compute source-of-truth overlays:
      - exhaustion_up / exhaustion_down bubbles
      - volatility bubbles (sized from the same vol->mult lerp used in sim)
      - angle arrows (normalized -1..+1 from ±45°)
    All sizes are derived from the same linear mappings used by the engine,
    so visuals == math.
    """
    # --- Settings with safe fallbacks (prefer config) ---
    lookback_exh   = int(coin_cfg.get("exhaustion_lookback", 184))
    window_step    = int(coin_cfg.get("window_step", 12))

    buy_min_bub    = float(coin_cfg.get("buy_min_bubble", 100))
    buy_max_bub    = float(coin_cfg.get("buy_max_bubble", 500))

    sell_min_bub   = float(coin_cfg.get("sell_min_bubble", 150))
    sell_max_bub   = float(coin_cfg.get("sell_max_bubble", 800))

    min_note_pct   = float(coin_cfg.get("min_note_size_pct", 0.03))
    max_note_pct   = float(coin_cfg.get("max_note_size_pct", 0.25))

    vol_lb         = float(coin_cfg.get("buy_min_vol_bubble", 0.0))
    vol_ub         = float(coin_cfg.get("buy_max_vol_bubble", 0.01))
    vol_mult_min   = float(coin_cfg.get("buy_mult_vol_min", 2.5))
    vol_mult_max   = float(coin_cfg.get("buy_mult_vol_max", 0.0))
    vol_lookback   = int(coin_cfg.get("vol_lookback", 48))

    angle_lb       = int(coin_cfg.get("angle_lookback", 48))
    mult_up        = float(coin_cfg.get("buy_mult_trend_up", 1.0))
    mult_floor     = float(coin_cfg.get("buy_mult_trend_floor", 0.25))
    mult_down      = float(coin_cfg.get("buy_mult_trend_down", 0.0))

    # Visual tuning (local; not strategy behavior)
    SIZE_SCALAR = 1_000_000.0
    SIZE_POWER  = 3.0

    close = df["close"].astype(float).reset_index(drop=True)
    ts    = df["timestamp"].astype(float).reset_index(drop=True)

    # --- Volatility series (returns rolling std) ---
    returns = close.pct_change().fillna(0.0)
    vol = returns.rolling(vol_lookback).std().fillna(0.0)

    def clamp(x, a, b): 
        return max(a, min(b, x))

    def lerp(x, a, b, c, d):
        if b == a:
            return c
        frac = (clamp(x, a, b) - a) / (b - a)
        return c + frac * (d - c)

    def trend_mult(norm_angle):
        # -1 -> down, 0 -> floor, +1 -> up
        v = max(-1.0, min(1.0, float(norm_angle)))
        return (mult_down + (mult_floor - mult_down) * (v + 1.0)) if v < 0 else (mult_floor + (mult_up - mult_floor) * v)

    # --- Angle (normalized by ±45°) per-candle ---
    norm_angle = np.zeros(len(close), dtype=float)
    for i in range(angle_lb, len(close)):
        dy = close[i] - close[i - angle_lb]
        dx = float(angle_lb)
        angle = np.arctan2(dy, dx)                 # radians
        norm  = angle / (np.pi / 4.0)              # ±45° => ±1
        norm_angle[i] = float(max(-1.0, min(1.0, norm)))

    # --- Exhaustion bubbles (up/down) sampled by window_step ---
    ex_up_x, ex_up_y, ex_up_s   = [], [], []
    ex_dn_x, ex_dn_y, ex_dn_s   = [], [], []
    for t in range(lookback_exh, len(close), window_step):
        now = close[t]; past = close[t - lookback_exh]
        if past <= 0: 
            continue
        if now > past:
            norm_up = (now - past) / past
            size = SIZE_SCALAR * (norm_up ** SIZE_POWER)
            ex_up_x.append(ts[t]); ex_up_y.append(now); ex_up_s.append(size)
        elif now < past:
            norm_dn = (past - now) / past
            size = SIZE_SCALAR * (norm_dn ** SIZE_POWER)
            ex_dn_x.append(ts[t]); ex_dn_y.append(now); ex_dn_s.append(size)

    # --- Volatility bubbles (size from the SAME vol->mult lerp used by strategy) ---
    vol_x, vol_y, vol_s = [], [], []
    for t in range(vol_lookback, len(close), window_step):
        v = float(vol.iloc[t])
        v_mult = lerp(v, vol_lb, vol_ub, vol_mult_min, vol_mult_max)
        # normalize to 0..1 for bubble area; bigger when v_mult is bigger
        num = v_mult - min(vol_mult_min, vol_mult_max)
        den = abs(vol_mult_max - vol_mult_min) or 1.0
        norm_m = max(0.0, min(1.0, num / den))
        size = SIZE_SCALAR * (norm_m ** SIZE_POWER)
        vol_x.append(ts[t]); vol_y.append(close[t]); vol_s.append(size)

    # --- Angle arrows (small segments), color-coded ---
    arrows = []  # list of (x0,y0,x1,y1,color)
    for i in range(angle_lb, len(close)):
        v = norm_angle[i]
        if v > 0.05:
            color = "orange"   # up
        elif v < -0.05:
            color = "purple"   # down
        else:
            color = "gray"     # flat
        x0 = ts[i]; y0 = close[i]
        x1 = ts[i] + (ts.iloc[1] - ts.iloc[0]) * 5 if len(ts) > 1 else x0 + 5
        y1 = y0 + v * 5.0
        arrows.append((x0, y0, x1, y1, color))

    return {
        "ex_up":   (ex_up_x, ex_up_y, ex_up_s),
        "ex_down": (ex_dn_x, ex_dn_y, ex_dn_s),
        "vol":     (vol_x, vol_y, vol_s),
        "arrows":  arrows,
    }


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

    # --- Source-of-truth overlays (match engine math) ---
    coin_settings = load_coin_settings()
    coin_symbol, _ = split_tag(market)
    coin_cfg = coin_settings.get(coin_symbol.upper(), {})
    over = _truth_overlays(df, coin_cfg)

    # Exhaustion bubbles
    ex_dn_x, ex_dn_y, ex_dn_s = over["ex_down"]
    if ex_dn_x:
        ax.scatter(pd.to_datetime(ex_dn_x, unit="s"), ex_dn_y, s=ex_dn_s, c="green", alpha=0.30, edgecolor="black", linewidths=0.5)
    ex_up_x, ex_up_y, ex_up_s = over["ex_up"]
    if ex_up_x:
        ax.scatter(pd.to_datetime(ex_up_x, unit="s"), ex_up_y, s=ex_up_s, c="red", alpha=0.30, edgecolor="black", linewidths=0.5)

    # Volatility bubbles (sized from vol-mult)
    vol_x, vol_y, vol_s = over["vol"]
    if vol_x:
        ax.scatter(pd.to_datetime(vol_x, unit="s"), vol_y, s=vol_s, c="crimson", alpha=0.20, edgecolor="black", linewidths=0.3)

    # Angle arrows
    for (x0, y0, x1, y1, color) in over["arrows"]:
        ax.plot([pd.to_datetime(x0, unit="s"), pd.to_datetime(x1, unit="s")],
                [y0, y1], color=color, lw=1.3, alpha=0.7, zorder=2)

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


def plot_from_json(ledger_path: str) -> None:
    """Plot simulation trades from a ledger JSON file."""
    path = Path(ledger_path)
    if not path.exists():
        print(f"[ERROR] Ledger not found at {path}")
        return

    try:
        with path.open("r", encoding="utf-8") as fh:
            ledger: dict[str, Any] = json.load(fh)
    except Exception:
        ledger = {}

    market = ledger.get("meta", {}).get("coin")
    if not market:
        print("[ERROR] Ledger missing coin metadata")
        return

    candles_path = Path("data/candles/sim") / f"{market}.csv"
    if not candles_path.exists():
        print(f"[ERROR] Candles not found at {candles_path}")
        return

    df = pd.read_csv(candles_path)
    times = pd.to_datetime(df["timestamp"], unit="s")
    fig, ax = plt.subplots()
    ax.plot(times, df["close"], label="Close", color="blue")

    coin_settings = load_coin_settings()
    coin_symbol, _ = split_tag(market)
    coin_cfg = coin_settings.get(coin_symbol.upper(), {})
    over = _truth_overlays(df, coin_cfg)

    ex_dn_x, ex_dn_y, ex_dn_s = over["ex_down"]
    if ex_dn_x:
        ax.scatter(pd.to_datetime(ex_dn_x, unit="s"), ex_dn_y, s=ex_dn_s, c="green", alpha=0.30, edgecolor="black", linewidths=0.5)
    ex_up_x, ex_up_y, ex_up_s = over["ex_up"]
    if ex_up_x:
        ax.scatter(pd.to_datetime(ex_up_x, unit="s"), ex_up_y, s=ex_up_s, c="red", alpha=0.30, edgecolor="black", linewidths=0.5)

    vol_x, vol_y, vol_s = over["vol"]
    if vol_x:
        ax.scatter(pd.to_datetime(vol_x, unit="s"), vol_y, s=vol_s, c="crimson", alpha=0.20, edgecolor="black", linewidths=0.3)

    for (x0, y0, x1, y1, color) in over["arrows"]:
        ax.plot([pd.to_datetime(x0, unit="s"), pd.to_datetime(x1, unit="s")], [y0, y1], color=color, lw=1.3, alpha=0.7, zorder=2)

    buys_x, buys_y, sells_x, sells_y = [], [], [], []
    for entry in ledger.get("trades", []):
        idx = int(entry.get("idx", -1))
        if idx < 0 or idx >= len(df):
            continue
        ts = float(df.iloc[idx]["timestamp"])
        price = entry.get("price")
        side = entry.get("side")
        if price is None or side is None:
            continue
        if side == "BUY":
            buys_x.append(ts); buys_y.append(price)
        elif side == "SELL":
            sells_x.append(ts); sells_y.append(price)

    if buys_x:
        ax.scatter(pd.to_datetime(buys_x, unit="s"), buys_y, color="green", marker="^", label="BUY")
    if sells_x:
        ax.scatter(pd.to_datetime(sells_x, unit="s"), sells_y, color="red", marker="v", label="SELL")

    ax.legend(loc="upper left")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show(block=False)
    plt.close(fig)
