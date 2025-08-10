from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
import pandas as pd

from systems.data_loader import load_or_fetch
from systems.paths import log_file

RUNNER_ID = "systems.simulator.run_sim_blocks"


def _rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(series, prepend=series[0])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    au = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    ad = pd.Series(dn).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = np.divide(au, np.maximum(ad, 1e-12))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-12)
    return float(-dd.min()) if dd.size else 0.0


def _run_block(df: pd.DataFrame, knobs: Dict[str, Any], equity0: float, *, verbosity: int = 0) -> Dict[str, Any]:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=int)

    rsi = _rsi(close, period=14)

    # knobs
    rsi_buy = int(knobs.get("rsi_buy", 35))
    rsi_sell = int(knobs.get("rsi_sell", 65))
    buy_cooldown = int(knobs.get("buy_cooldown", 8))
    sell_cooldown = int(knobs.get("sell_cooldown", 8))
    take_profit = float(knobs.get("take_profit", 0.03))
    stop_loss = float(knobs.get("stop_loss", 0.06))
    trailing_stop = float(knobs.get("trailing_stop", 0.02))
    position_pct = float(knobs.get("position_pct", 0.06))
    max_concurrent = int(knobs.get("max_concurrent", 2))

    equity = np.full(len(close), equity0, dtype=float)
    cash = equity0
    positions: list[Dict[str, float]] = []
    buy_cd = 0
    sell_cd = 0
    trades = 0

    for i in range(len(close)):
        price = close[i]
        timestamp = ts[i]
        if verbosity >= 3:
            print(f"[TICK] {pd.to_datetime(timestamp, unit='s', utc=True)} close={price:.2f} open_notes={len(positions)} cool(buy={buy_cd},sell={sell_cd})")

        for p in positions:
            p["peak"] = max(p["peak"], price)

        if sell_cd == 0 and positions:
            keep = []
            for p in positions:
                ret = (price - p["entry"]) / p["entry"]
                trail_hit = (
                    (p["peak"] - price) / p["peak"] >= trailing_stop if p["peak"] > 0 else False
                )
                if ret >= take_profit or ret <= -stop_loss or trail_hit or rsi[i] >= rsi_sell:
                    cash += p["size"] * (1 + ret)
                    trades += 1
                    if verbosity >= 3:
                        print(f"[SELL] note amt={p['size']/p['entry']:.2f} at={price:.2f} pnl=${p['size']*ret:.2f}")
                else:
                    keep.append(p)
            positions = keep
            sell_cd = sell_cooldown if trades else 0
        else:
            sell_cd = max(0, sell_cd - 1)

        if buy_cd == 0 and len(positions) < max_concurrent and rsi[i] <= rsi_buy:
            size = cash * position_pct
            if size > 0:
                positions.append({"entry": price, "size": size, "peak": price})
                cash -= size
                trades += 1
                buy_cd = buy_cooldown
                if verbosity >= 3:
                    print(f"[BUY] note amt={size/price:.2f} at={price:.2f}")
        else:
            buy_cd = max(0, buy_cd - 1)

        pos_val = sum(p["size"] * (price / p["entry"]) for p in positions)
        equity[i] = cash + pos_val

    if positions:
        price = close[-1]
        for p in positions:
            ret = (price - p["entry"]) / p["entry"]
            cash += p["size"] * (1 + ret)
        positions.clear()
    equity[-1] = cash

    pnl = equity[-1] - equity0
    maxdd = _max_drawdown(equity)
    returns = np.diff(equity, prepend=equity0) / np.maximum(equity0, 1e-12)
    return {
        "pnl": float(pnl),
        "maxdd": float(maxdd),
        "trades": int(trades),
        "equity_series": equity.tolist(),
        "returns": returns.tolist(),
        "ledger": {"cash": float(cash)},
    }


def run_sim_blocks(
    blocks: Iterable[Dict[str, Any]],
    tag: str,
    knobs: Dict[str, Any],
    settings: Dict[str, Any],
    *,
    verbosity: int = 0,
    run_id: str | None = None,
    log_path: str | None = None,
    audit: bool = True,
) -> Dict[str, Any]:
    """Execute simulation over blocks."""

    log_fh = None
    if run_id:
        path = Path(log_path) if log_path else log_file(tag, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = path.open("a", encoding="utf-8", newline="")

    def _log(msg: str, log_fh=log_fh) -> None:
        safe = msg.encode("utf-8", "replace").decode("utf-8")
        print(safe)
        if log_fh is not None:
            log_fh.write(safe + "\n")
            log_fh.flush()

    _log(f"[SIM] Using sim runner: {RUNNER_ID}")
    if audit:
        _log(f"[AUDIT] Sim runner active: {RUNNER_ID} (production parity mode)")

    sim_dir = Path(f"data/tmp/simulation/{tag}")
    if sim_dir.exists():
        for p in sim_dir.glob("*.json"):
            p.unlink()
    else:
        sim_dir.mkdir(parents=True, exist_ok=True)

    candles = load_or_fetch(tag)
    ts = candles["timestamp"].to_numpy()

    capital = float(settings.get("capital", 1000.0))
    equity_curve: list[float] = []
    total_trades = 0
    total_pnl = 0.0

    for k, block in enumerate(blocks, 1):
        if "i0" in block and "i1" in block:
            i0, i1 = int(block["i0"]), int(block["i1"])
        else:
            start = block.get("start_ts") or block.get("start")
            end = block.get("end_ts") or block.get("end")
            i0 = int(np.searchsorted(ts, start, side="left"))
            i1 = int(np.searchsorted(ts, end, side="right") - 1)
        block_df = candles.iloc[i0 : i1 + 1].copy()
        if block_df.empty:
            continue

        res = _run_block(block_df, knobs, capital, verbosity=verbosity)
        capital += res["pnl"]
        total_trades += res["trades"]
        total_pnl += res["pnl"]
        equity_curve.extend(res["equity_series"])

        ledger_path = sim_dir / f"block_{k:02d}.json"
        with ledger_path.open("w") as fh:
            json.dump(res.get("ledger", {}), fh, indent=2)

        if verbosity >= 2:
            start_dt = pd.to_datetime(block_df.iloc[0]["timestamp"], unit="s", utc=True)
            end_dt = pd.to_datetime(block_df.iloc[-1]["timestamp"], unit="s", utc=True)
            _log(
                f"[BLOCK] k={k:02d} | {start_dt.date()} -> {end_dt.date()} | candles={len(block_df)}"
            )
            _log(
                f"[BLOCK] trades={res['trades']} pnl=${res['pnl']:.2f} maxdd={res['maxdd']:.2f}"
            )

    maxdd_all = _max_drawdown(np.asarray(equity_curve, dtype=float))
    result = {
        "trades": int(total_trades),
        "pnl": float(total_pnl),
        "maxdd": float(maxdd_all),
        "equity_series": equity_curve,
    }

    if verbosity >= 1:
        pnl_dd = total_pnl * (1 - 1.5 * maxdd_all)
        _log(
            f"[TRIAL] tag={tag} blocks={k} trades={total_trades} pnl=${total_pnl:,.2f} maxdd={maxdd_all:.2f} pnl_dd=${pnl_dd:,.2f}"
        )

    if log_fh is not None:
        log_fh.close()

    return result


__all__ = ["run_sim_blocks", "RUNNER_ID"]
