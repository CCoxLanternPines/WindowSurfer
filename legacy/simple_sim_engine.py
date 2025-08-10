from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class SimResult:
    per_block: list[dict]
    summary: dict

def _rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    # simple RSI (wilders-ish)
    delta = np.diff(series, prepend=series[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    au = pd.Series(up).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    ad = pd.Series(dn).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    rs = np.divide(au, np.maximum(ad, 1e-12))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return -dd.min() if len(dd) else 0.0

def _run_block(c: pd.DataFrame, knobs: dict, equity0: float = 1000.0) -> dict:
    # Required columns: timestamp, open, high, low, close, volume
    close = c["close"].to_numpy(dtype=float)
    high  = c["high"].to_numpy(dtype=float)
    low   = c["low"].to_numpy(dtype=float)

    # knobs (with safe defaults)
    rsi_buy        = int(knobs.get("rsi_buy", 35))
    rsi_sell       = int(knobs.get("rsi_sell", 65))
    buy_cooldown   = int(knobs.get("buy_cooldown", 8))
    sell_cooldown  = int(knobs.get("sell_cooldown", 8))
    take_profit    = float(knobs.get("take_profit", 0.03))
    stop_loss      = float(knobs.get("stop_loss", 0.06))
    trailing_stop  = float(knobs.get("trailing_stop", 0.02))
    position_pct   = float(knobs.get("position_pct", 0.06))
    max_concurrent = int(knobs.get("max_concurrent", 2))

    rsi = _rsi(close, period=14)
    equity = np.full(len(close), equity0, dtype=float)
    cash = equity0
    positions = []  # list of dict: {"entry": price, "size": $, "peak": price}
    buy_cd = 0
    sell_cd = 0
    trades = 0

    for i in range(len(close)):
        price = close[i]
        # update trailing peaks
        for p in positions:
            p["peak"] = max(p["peak"], price)

        # exits (TP/SL/trailing) if sell cooldown over
        if sell_cd == 0 and positions:
            keep = []
            for p in positions:
                ret = (price - p["entry"]) / p["entry"]
                trail_hit = (p["peak"] - price) / p["peak"] >= trailing_stop if p["peak"] > 0 else False
                if ret >= take_profit or ret <= -stop_loss or trail_hit or rsi[i] >= rsi_sell:
                    cash += p["size"] * (1 + ret)
                    trades += 1
                else:
                    keep.append(p)
            positions = keep
            sell_cd = sell_cooldown if trades else 0
        else:
            sell_cd = max(0, sell_cd - 1)

        # entries if buy cooldown over
        if buy_cd == 0 and len(positions) < max_concurrent and rsi[i] <= rsi_buy:
            size = cash * position_pct
            if size > 0:
                positions.append({"entry": price, "size": size, "peak": price})
                cash -= size
                trades += 1
                buy_cd = buy_cooldown
        else:
            buy_cd = max(0, buy_cd - 1)

        # mark-to-market
        pos_val = sum(p["size"] * (price / p["entry"]) for p in positions)
        equity[i] = cash + pos_val

    # close any open at last price
    if positions:
        price = close[-1]
        for p in positions:
            ret = (price - p["entry"]) / p["entry"]
            cash += p["size"] * (1 + ret)
        positions.clear()
    equity[-1] = cash

    pnl = (equity[-1] - equity0)
    max_dd = _max_drawdown(equity)
    return {"pnl": float(pnl), "max_dd": float(max_dd), "trades": int(trades)}

def run_sim_blocks(candles: pd.DataFrame, blocks: list[dict], knobs: dict) -> SimResult:
    per = []
    total_pnl = 0.0
    max_dd_all = 0.0
    trades_all = 0
    for b in blocks:
        i0 = int(b["test_index_start"])
        i1 = int(b["test_index_end"])
        if i1 < i0 or i1 >= len(candles): 
            continue
        block_df = candles.iloc[i0:i1+1]
        res = _run_block(block_df, knobs)
        res["block_id"] = int(b.get("block_id", len(per)+1))
        per.append(res)
        total_pnl += res["pnl"]
        max_dd_all = max(max_dd_all, res["max_dd"])
        trades_all += res["trades"]
    return SimResult(
        per_block=per,
        summary={"pnl": float(total_pnl), "max_dd": float(max_dd_all), "trades": int(trades_all)}
    )
