from __future__ import annotations

"""Historical simulation engine for position-based strategy."""

from datetime import datetime, timezone, timedelta
import csv
import json
import os
import re

import ccxt
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict

from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.candle_loader import load_candles_df
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import (
    apply_buy,
    apply_sell,
    paper_execute_buy,
    paper_execute_sell,
)
from systems.utils.addlog import addlog
from pathlib import Path

from systems.utils.config import (
    resolve_path,
    load_general,
    load_coin_settings,
    load_account_settings,
    resolve_account_market,
)
from systems.utils.resolve_symbol import (
    split_tag,
    resolve_symbols,
    to_tag,
)
from systems.utils.trade_logger import init_logger as init_trade_logger, record_event
from systems.utils.ledger import init_ledger as ledger_init, append_entry as ledger_append, save_ledger as ledger_save


TIMEFRAME_SECONDS = {
    "s": 1,
    "m": 30 * 24 * 3600,  # 1m = one month
    "w": 7 * 24 * 3600,   # 1w = one week
    "d": 24 * 3600,       # 1d = one day
    "h": 3600,            # 1h = one hour
}


def parse_timeframe(tf: str) -> timedelta | None:
    if not tf:
        return None
    m = re.match(r"(?i)^\s*(\d+)\s*([smhdw])\s*$", tf)
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2).lower()
    return timedelta(seconds=n * TIMEFRAME_SECONDS[u])


def _to_ccxt(market: str) -> str:
    """Convert a noslash market like ``DOGEUSD`` to ``DOGE/USD``."""
    market = market.upper()
    for quote in ("USDT", "USDC", "USD", "EUR", "GBP"):
        if market.endswith(quote):
            base = market[: -len(quote)]
            return f"{base}/{quote}"
    return market


def _run_single_sim(
    *,
    general,
    coin_settings,
    accounts_cfg,
    account: str,
    market: str,
    client: ccxt.Exchange,
    verbose: int = 0,
    timeframe: str = "1m",
    viz: bool = True,
) -> None:
    os.environ["WS_MODE"] = "sim"
    os.environ["WS_ACCOUNT"] = account
    addlog(
        "[PARITY] Running in sim mode — strategy knobs identical, only execution differs",
        verbose_int=1,
        verbose_state=verbose,
    )

    market = market.replace("/", "").upper()
    ccxt_market = _to_ccxt(market)

    symbols = resolve_symbols(client, ccxt_market)
    kraken_name = symbols["kraken_name"]
    kraken_pair = symbols["kraken_pair"]
    binance_name = symbols["binance_name"]

    addlog(
        f"[RESOLVE][{account}][{market}] KrakenName={kraken_name} KrakenPair={kraken_pair} BinanceName={binance_name}",
        verbose_int=1,
        verbose_state=verbose,
    )

    tag = to_tag(kraken_name)
    file_tag = market
    ledger_name = f"{account}_{file_tag}"
    init_trade_logger(ledger_name)
    base, _ = split_tag(tag)
    coin = base.upper()
    csv_path = Path("data/candles/sim") / f"{market}.csv"
    df, removed = load_candles_df(account, market, verbose=verbose)
    ts_col = "timestamp"

    # --- Timestamp unit detection ---
    last_ts_raw = float(df[ts_col].iloc[-1]) if len(df) else 0.0
    is_ms = last_ts_raw > 1e12  # > ~2001-09-09 in ms
    TUNIT = "ms" if is_ms else "s"
    TFACTOR = 1000.0 if is_ms else 1.0  # seconds→ts units

    # Optional hard check
    if not df[ts_col].is_monotonic_increasing:
        raise ValueError(f"Candles not sorted by {ts_col}: {csv_path}")

    delta = None
    if timeframe:
        delta = parse_timeframe(timeframe)
        if delta:
            end_ts = float(df[ts_col].iloc[-1])
            start_ts = end_ts - delta.total_seconds() * TFACTOR
            df = df[df[ts_col] >= start_ts].reset_index(drop=True)
            # Recompute last_ts_raw after slice
            last_ts_raw = float(df[ts_col].iloc[-1]) if len(df) else 0.0

    # --- Time-slice diagnostics (anchored to dataset end) ---
    if timeframe and delta and len(df) >= 2:
        diffs_raw = df[ts_col].diff().dropna()
        med_step_raw = float(diffs_raw.median()) if not diffs_raw.empty else 0.0
        med_step_sec = med_step_raw / TFACTOR if med_step_raw else 0.0
        approx_rows = int(delta.total_seconds() / med_step_sec) if med_step_sec else 0
        print(
            f"[TIMEFILTER][CHECK] delta={delta} med_step={med_step_sec:.0f}s approx_rows~{approx_rows} actual_rows={len(df)}"
        )

    # Log one line so we always know what we ran on
    def _iso(ts_raw: float) -> str:
        try:
            return datetime.fromtimestamp(ts_raw / TFACTOR, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return "n/a"

    first_raw = float(df[ts_col].iloc[0]) if len(df) else None
    last_raw = float(df[ts_col].iloc[-1]) if len(df) else None
    first_iso = _iso(first_raw) if first_raw is not None else "n/a"
    last_iso = _iso(last_raw) if last_raw is not None else "n/a"
    print(
        f"[DATA][SIM] file={csv_path} rows={len(df)} first={first_iso} last={last_iso} dups_removed={removed}"
    )
    if timeframe and delta:
        print(
            f"[TIMEFILTER] Using --time {timeframe} rows={len(df)} first={first_iso} last={last_iso}"
        )

    total = len(df)

    runtime_state = build_runtime_state(
        general,
        coin_settings,
        accounts_cfg,
        account,
        ccxt_market,
        mode="sim",
        client=client,
        prev={"verbose": verbose},
    )
    runtime_state["strategy"].update(
        resolve_account_market(account, market, accounts_cfg)
    )
    runtime_state["mode"] = "sim"
    runtime_state["symbol"] = tag
    runtime_state["buy_unlock_p"] = {}
    strategy_cfg = runtime_state.get("strategy", {})

    # Capture starting capital for ledger meta
    start_capital = float(runtime_state.get("capital", 0.0))

    # simple ledger for reporting/graphing
    trade_ledger = ledger_init(account, market, "sim")


    ledger_obj = Ledger()
    ledger_obj.set_metadata({"capital": start_capital})
    buy_points: list[tuple[float, float]] = []
    sell_points: list[tuple[float, float]] = []
    win_metrics = {
        "strategy": {
            "window_size": str(strategy_cfg.get("window_size", "")),
            "buys": 0,
            "sells": 0,
            "gross_invested": 0.0,
            "realized_cost": 0.0,
            "realized_proceeds": 0.0,
            "realized_trades": 0,
            "realized_roi_accum": 0.0,
        }
    }
    addlog(f"[SIM] Starting simulation for {coin}", verbose_int=1, verbose_state=verbose)

    step = int(strategy_cfg.get("window_step", 1))
    window_size = int(strategy_cfg.get("window_size", 0))
    for start in tqdm(
        range(0, total - window_size, step),
        desc="📉 Sim Progress",
        dynamic_ncols=True,
    ):
        t = start  # anchor candle for this window
        price = float(df.iloc[t]["close"])
        ts = float(df.iloc[t][ts_col]) if ts_col in df.columns else None
        iso_ts = (
            datetime.fromtimestamp(ts / TFACTOR, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if ts is not None
            else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        decision = "HOLD"
        trades_log: list[dict[str, Any]] = []
        ctx = {"ledger": ledger_obj}
        size_usd = 0.0
        entry_price_val = 0.0
        roi_val = 0.0
        buy_res = evaluate_buy(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
            runtime_state=runtime_state,
        )
        if buy_res:
            result = paper_execute_buy(price, buy_res["size_usd"], timestamp=ts)

            note = apply_buy(
                ledger=ledger_obj,
                window_name="strategy",
                t=t,
                meta=buy_res,
                result=result,
                state=runtime_state,
            )

            runtime_state.setdefault("buy_unlock_p", {})["strategy"] = buy_res.get("unlock_p")

            m_buy = win_metrics.get("strategy")
            if m_buy is not None:
                cost = result["filled_amount"] * result["avg_price"]
                m_buy["buys"] += 1
                m_buy["gross_invested"] += cost

            trades_log.append(
                {
                    "action": "BUY",
                    "amount": result["filled_amount"] * result["avg_price"],
                    "price": result["avg_price"],
                    "note_id": note.get("id"),
                }
            )
            decision = "BUY"
            size_usd = buy_res.get("size_usd", 0.0)
            entry_price_val = result.get("avg_price", 0.0)

            if viz:
                buy_points.append((float(df.iloc[t][ts_col]), price))

        open_notes = ledger_obj.get_open_notes()
        sell_notes = evaluate_sell(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
            open_notes=open_notes,
            runtime_state=runtime_state,
        )
        if sell_notes:
            decision = (
                "FLAT"
                if any(n.get("sell_mode") == "flat" for n in sell_notes)
                else "SELL"
            )

        for note in sell_notes:
            result = paper_execute_sell(
                price, note.get("entry_amount", 0.0), timestamp=ts
            )
            if viz:
                mode = note.get("sell_mode", "normal")
                sell_points.append((float(df.iloc[t][ts_col]), price, mode))
            closed = apply_sell(
                ledger=ledger_obj,
                note=note,
                t=t,
                result=result,
                state=runtime_state,
            )
            trades_log.append(
                {
                    "action": "SELL",
                    "amount": result["filled_amount"] * result["avg_price"],
                    "price": result["avg_price"],
                    "note_id": closed.get("id"),
                }
            )

            qty = closed.get("entry_amount", 0.0)
            buy_price = closed.get("entry_price", 0.0)
            exit_price = closed.get("exit_price", 0.0)
            cost = buy_price * qty
            proceeds = exit_price * qty
            roi_trade = (proceeds - cost) / cost if cost > 0 else 0.0
            m_sell = win_metrics.get("strategy")
            if m_sell is not None:
                m_sell["sells"] += 1
                m_sell["realized_cost"] += cost
                m_sell["realized_proceeds"] += proceeds
                m_sell["realized_trades"] += 1
                m_sell["realized_roi_accum"] += roi_trade
        if sell_notes:
            total_cost = sum(
                n.get("entry_amount", 0.0) * n.get("entry_price", 0.0)
                for n in sell_notes
            )
            total_amount = sum(n.get("entry_amount", 0.0) for n in sell_notes)
            entry_price_val = (total_cost / total_amount) if total_amount else 0.0
            size_usd = total_cost
            roi_val = ((price - entry_price_val) / entry_price_val) if entry_price_val else 0.0
        features = runtime_state.get("last_features", {}).get("strategy", {})
        pressures = runtime_state.get("pressures", {})
        event = {
            "timestamp": iso_ts,
            "ledger": ledger_name,
            "pair": tag,
            "window": f"{strategy_cfg.get('window_size', 0)}h",
            "decision": decision,
            "features": {
                "close": price,
                "slope": features.get("slope"),
                "volatility": features.get("volatility"),
                "buy_pressure": pressures.get("buy", {}).get("strategy", 0.0),
                "sell_pressure": pressures.get("sell", {}).get("strategy", 0.0),
                "buy_trigger": strategy_cfg.get("buy_trigger", 0.0),
                "sell_trigger": strategy_cfg.get("sell_trigger", 0.0),
            },
            "trades": trades_log,
        }
        ledger_entry = {
            "candle_idx": t,
            "idx": t,
            "timestamp": ts,
            "side": "BUY" if decision == "BUY" else ("SELL" if sell_notes else "PASS"),
            "price": price,
            "size_usd": size_usd,
            "usd": size_usd,
            "entry_price": entry_price_val,
            "roi": roi_val,
            "pressure_buy": pressures.get("buy", {}).get("strategy", 0.0),
            "pressure_sell": pressures.get("sell", {}).get("strategy", 0.0),
            "features": features,
        }
        ledger_append(trade_ledger, ledger_entry)
        record_event(event)

    final_price = float(df.iloc[-1]["close"]) if total else 0.0
    summary = ledger_obj.get_account_summary(final_price)

    open_value = sum(
        n.get("entry_amount", 0.0) * final_price for n in ledger_obj.get_open_notes()
    )

    wallet_cash = runtime_state["capital"]
    m = win_metrics.get("strategy", {})
    realized_roi = (
        (m.get("realized_proceeds", 0.0) / m.get("realized_cost", 1.0) - 1.0)
        if m.get("realized_cost", 0.0) > 0
        else 0.0
    )
    avg_trade_roi = (
        m.get("realized_roi_accum", 0.0) / m.get("realized_trades", 1)
        if m.get("realized_trades", 0) > 0
        else 0.0
    )
    total_value_at_liq = m.get("realized_proceeds", 0.0) + open_value
    realized_pnl = m.get("realized_proceeds", 0.0) - m.get("realized_cost", 0.0)
    addlog(
        f"[REPORT][strategy] buys={m.get('buys',0)} sells={m.get('sells',0)} realized_pnl=${realized_pnl:.2f} realized_roi={(realized_roi*100):.2f}% avg_trade_roi={(avg_trade_roi*100):.2f}% open_notes_value=${open_value:.2f} strategy_total_at_liq=${total_value_at_liq:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    rows = [
        {
            "window": "strategy",
            "window_size": m.get("window_size", ""),
            "buys": m.get("buys", 0),
            "sells": m.get("sells", 0),
            "gross_invested": m.get("gross_invested", 0.0),
            "realized_cost": m.get("realized_cost", 0.0),
            "realized_proceeds": m.get("realized_proceeds", 0.0),
            "realized_pnl": realized_pnl,
            "realized_roi": realized_roi,
            "avg_trade_roi": avg_trade_roi,
            "open_value_now": open_value,
            "window_total_at_liq": total_value_at_liq,
        }
    ]

    global_total_at_liq = wallet_cash + open_value
    addlog(
        f"[REPORT][GLOBAL] cash=${wallet_cash:.2f} open_value=${open_value:.2f} total_at_liq=${global_total_at_liq:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    addlog(
        f"Final Value (USD): ${summary['total_value']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )
    ledger_obj.set_metadata({"capital": runtime_state.get("capital", 0.0)})
    save_ledger(
        ledger_name,
        ledger_obj,
        sim=True,
        final_tick=total - 1,
        summary=summary,
        tag=file_tag,
    )
    root = resolve_path("")

    # Always write simplified JSON ledger for external visualization
    temp_dir = resolve_path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    full_path = temp_dir / "sim_data.json"
    print(f"[DEBUG][sim_engine] Attempting to write sim ledger → {full_path.resolve()}")
    simple_ledger = {
        "trades": [
            {
                "idx": e.get("idx"),
                "price": e.get("price"),
                "side": e.get("side"),
                "usd": e.get("usd", e.get("size_usd")),
            }
            for e in trade_ledger.get("entries", [])
        ],
        "meta": {
            "coin": market,
            "start_capital": start_capital,
            "final_value": summary.get("total_value", 0.0),
        },
    }
    with full_path.open("w", encoding="utf-8") as fh:
        json.dump(simple_ledger, fh, indent=2)
    print(
        f"[DEBUG][sim_engine] Finished writing ledger, entries={len(simple_ledger['trades'])}"
    )
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = logs_dir / f"sim_report_{ledger_name}_{ts}.csv"
    json_path = logs_dir / f"sim_report_{ledger_name}_{ts}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=[
                "window",
                "window_size",
                "buys",
                "sells",
                "gross_invested",
                "realized_cost",
                "realized_proceeds",
                "realized_pnl",
                "realized_roi",
                "avg_trade_roi",
                "open_value_now",
                "window_total_at_liq",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_data = {
        "windows": rows,
        "global": {
            "cash": wallet_cash,
            "open_value_now": open_value,
            "total_at_liq": global_total_at_liq,
        },
    }
    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(json_data, f_json, indent=2)
    if viz:
        import matplotlib.pyplot as plt
        import numpy as np

        times = pd.to_datetime(df[ts_col], unit=TUNIT)
        plt.figure()
        plt.plot(times, df["close"], label="Close", color="gray", zorder=1)

        # ----- Feature calculations for visualization -----
        ex_lookback = int(strategy_cfg.get("exhaustion_lookback", 184))
        window_step = int(strategy_cfg.get("window_step", 12))
        vol_lookback = int(strategy_cfg.get("vol_lookback", 48))
        angle_lookback = int(strategy_cfg.get("angle_lookback", 48))
        angle_up_min = float(strategy_cfg.get("angle_up_min", 0.05))
        angle_down_min = float(strategy_cfg.get("angle_down_min", -0.05))

        returns = df["close"].pct_change()
        df["volatility"] = returns.rolling(vol_lookback).std().fillna(0.0)
        df["angle"] = 0.0

        SIZE_SCALAR = 1_000_000
        SIZE_POWER = 3

        ex_up = {"x": [], "y": [], "s": []}
        ex_down = {"x": [], "y": [], "s": []}
        vol_pts = {"x": [], "y": [], "s": []}

        for t in range(angle_lookback, len(df)):
            dy = df["close"].iloc[t] - df["close"].iloc[t - angle_lookback]
            dx = angle_lookback
            angle = np.arctan2(dy, dx)
            norm = angle / (np.pi / 4)
            df.at[t, "angle"] = max(-1.0, min(1.0, norm))

        for t in range(ex_lookback, len(df), window_step):
            now_price = float(df["close"].iloc[t])
            past_price = float(df["close"].iloc[t - ex_lookback])
            if now_price > past_price:
                delta_up = now_price - past_price
                norm_up = delta_up / max(1e-9, past_price)
                size = SIZE_SCALAR * (norm_up ** SIZE_POWER)
                ex_up["x"].append(times.iloc[t])
                ex_up["y"].append(now_price)
                ex_up["s"].append(size)
            elif now_price < past_price:
                delta_down = past_price - now_price
                norm_down = delta_down / max(1e-9, past_price)
                size = SIZE_SCALAR * (norm_down ** SIZE_POWER)
                ex_down["x"].append(times.iloc[t])
                ex_down["y"].append(now_price)
                ex_down["s"].append(size)

        for t in range(vol_lookback, len(df), window_step):
            vol = df["volatility"].iloc[t]
            size = SIZE_SCALAR * (vol * (0.4 ** SIZE_POWER))
            vol_pts["x"].append(times.iloc[t])
            vol_pts["y"].append(float(df["close"].iloc[t]))
            vol_pts["s"].append(size)

        arrow_span = 5
        for i in range(angle_lookback, len(df), window_step):
            v = df["angle"].iloc[i]
            if v > angle_up_min:
                color = "orange"
            elif v < angle_down_min:
                color = "purple"
            else:
                color = "gray"
            x0 = times.iloc[i]
            y0 = df["close"].iloc[i]
            j = min(i + arrow_span, len(df) - 1)
            x1 = times.iloc[j]
            y1 = y0 + v * arrow_span
            plt.plot([x0, x1], [y0, y1], color=color, lw=1.5, alpha=0.7)

        if ex_up["x"]:
            plt.scatter(
                ex_up["x"], ex_up["y"], s=ex_up["s"],
                color="green", alpha=0.3, edgecolors="black",
                label="Exhaust Up", zorder=2,
            )
        if ex_down["x"]:
            plt.scatter(
                ex_down["x"], ex_down["y"], s=ex_down["s"],
                color="red", alpha=0.3, edgecolors="black",
                label="Exhaust Down", zorder=2,
            )
        if vol_pts["x"]:
            plt.scatter(
                vol_pts["x"], vol_pts["y"], s=vol_pts["s"],
                facecolors="none", edgecolors="red", alpha=0.3,
                label="Volatility", zorder=2,
            )

        # --- Plot buys ---
        if buy_points:
            b_t, b_p = zip(*buy_points)
            plt.scatter(
                pd.to_datetime(b_t, unit=TUNIT),
                b_p,
                marker="o", color="g", label="Buy", zorder=5,
            )

        # --- Plot sells ---
        if sell_points:
            times_normal = [t for t, p, m in sell_points if m == "normal"]
            prices_normal = [p for t, p, m in sell_points if m == "normal"]
            times_flat   = [t for t, p, m in sell_points if m == "flat"]
            prices_flat  = [p for t, p, m in sell_points if m == "flat"]
            times_all   = [t for t, p, m in sell_points if m == "all"]
            prices_all  = [p for t, p, m in sell_points if m == "all"]

            if times_normal:
                plt.scatter(
                    pd.to_datetime(times_normal, unit=TUNIT),
                    prices_normal,
                    marker="o", color="r", label="Sell", zorder=5,
                )
            if times_flat:
                plt.scatter(
                    pd.to_datetime(times_flat, unit=TUNIT),
                    prices_flat,
                    marker="o", color="orange", label="Flat Sell", zorder=5,
                )
            if times_all:
                plt.scatter(
                    pd.to_datetime(times_all, unit=TUNIT),
                    prices_all,
                    marker="x", color="red", label="All Sell", zorder=6,
                )

        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()


def run_simulation(*, coin: str, timeframe: str = "1m", viz: bool = True) -> None:
    """Run a simulation for the given ``coin`` over ``timeframe``."""
    general = load_general()
    coin_settings = load_coin_settings()
    accounts_cfg = load_account_settings()

    if not accounts_cfg:
        raise ValueError("No account configuration available")

    # Use the first configured account for simulation purposes
    account = next(iter(accounts_cfg.keys()))
    client = ccxt.kraken({"enableRateLimit": True})
    market = coin.replace("/", "").upper()

    addlog(
        f"[RUN][{account}][{market}]",
        verbose_int=1,
        verbose_state=0,
    )
    _run_single_sim(
        general=general,
        coin_settings=coin_settings,
        accounts_cfg=accounts_cfg,
        account=account,
        market=market,
        client=client,
        verbose=0,
        timeframe=timeframe,
        viz=viz,
    )
