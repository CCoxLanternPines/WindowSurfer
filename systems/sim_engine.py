from __future__ import annotations

"""Historical simulation engine for position-based strategy."""

import shutil
from datetime import datetime, timezone
import csv
import json
import os

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
)
from systems.utils.resolve_symbol import (
    split_tag,
    resolve_symbols,
    to_tag,
    sim_path_csv,
    candle_filename,
)
from systems.utils.time import parse_cutoff as parse_timeframe
from systems.utils.trade_logger import init_logger as init_trade_logger, record_event
from systems.utils.ledger import init_ledger as ledger_init, append_entry as ledger_append, save_ledger as ledger_save


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

    symbols = resolve_symbols(client, market)
    kraken_name = symbols["kraken_name"]
    kraken_pair = symbols["kraken_pair"]
    binance_name = symbols["binance_name"]

    addlog(
        f"[RESOLVE][{account}][{market}] KrakenName={kraken_name} KrakenPair={kraken_pair} BinanceName={binance_name}",
        verbose_int=1,
        verbose_state=verbose,
    )

    tag = to_tag(kraken_name)
    file_tag = market.replace("/", "_")
    ledger_name = f"{account}_{file_tag}"
    init_trade_logger(ledger_name)
    base, _ = split_tag(tag)
    coin = base.upper()
    csv_path = candle_filename(account, market)
    df, removed = load_candles_df(account, market, verbose=verbose)
    ts_col = "timestamp"

    # Optional hard check
    if not df[ts_col].is_monotonic_increasing:
        raise ValueError(f"Candles not sorted by {ts_col}: {csv_path}")

    if timeframe:
        delta = parse_timeframe(timeframe)
        cutoff_ts = (datetime.now(tz=timezone.utc) - delta).timestamp()
        df = df[df[ts_col] >= cutoff_ts].reset_index(drop=True)

    # Log one line so we always know what we ran on
    first_ts = int(df[ts_col].iloc[0]) if len(df) else None
    last_ts = int(df[ts_col].iloc[-1]) if len(df) else None
    first_iso = (
        datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if first_ts is not None
        else "n/a"
    )
    last_iso = (
        datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if last_ts is not None
        else "n/a"
    )
    print(
        f"[DATA][SIM] file={csv_path} rows={len(df)} first={first_iso} last={last_iso} dups_removed={removed}"
    )

    total = len(df)

    runtime_state = build_runtime_state(
        general,
        coin_settings,
        accounts_cfg,
        account,
        market,
        mode="sim",
        client=client,
        prev={"verbose": verbose},
    )
    runtime_state["mode"] = "sim"
    runtime_state["symbol"] = tag
    runtime_state["buy_unlock_p"] = {}
    strategy_cfg = runtime_state.get("strategy", {})

    # simple ledger for reporting/graphing
    trade_ledger = ledger_init(account, market, "sim")


    ledger_obj = Ledger()
    ledger_obj.set_metadata({"capital": runtime_state.get("capital", 0.0)})
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
        ts = int(df.iloc[t]["timestamp"]) if "timestamp" in df.columns else None
        iso_ts = (
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
            "timestamp": ts,
            "side": "BUY" if decision == "BUY" else ("SELL" if sell_notes else "PASS"),
            "price": price,
            "size_usd": size_usd,
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
    default_path = root / "data" / "tmp" / "simulation" / f"{ledger_name}.json"
    sim_path = root / "data" / "tmp" / f"simulation_{ledger_name}.json"
    if default_path.exists() and default_path != sim_path:
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_path, sim_path)
    ledger_save(
        trade_ledger,
        str(resolve_path("data/ledgers/ledger_simulation.json")),
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

        times = pd.to_datetime(df[ts_col], unit="s")
        plt.figure()
        plt.plot(times, df["close"], label="Close", color="gray", zorder=1)
        # --- Plot buys (single block, no duplication) ---
        if buy_points:
            b_t, b_p = zip(*buy_points)
            plt.scatter(
                pd.to_datetime(b_t, unit="s"),
                b_p,
                marker="o", color="g", label="Buy", zorder=2,
            )


        # --- Plot sells (normal vs flat) ---
        if sell_points:
            times_normal = [t for t, p, m in sell_points if m == "normal"]
            prices_normal = [p for t, p, m in sell_points if m == "normal"]
            times_flat   = [t for t, p, m in sell_points if m == "flat"]
            prices_flat  = [p for t, p, m in sell_points if m == "flat"]
            times_all   = [t for t, p, m in sell_points if m == "all"]
            prices_all  = [p for t, p, m in sell_points if m == "all"]

            if times_normal:
                plt.scatter(
                    pd.to_datetime(times_normal, unit="s"),
                    prices_normal,
                    marker="o", color="r", label="Sell", zorder=2,
                )
            if times_flat:
                plt.scatter(
                    pd.to_datetime(times_flat, unit="s"),
                    prices_flat,
                    marker="o", color="orange", label="Flat Sell", zorder=2,
                )
            if times_all:
                plt.scatter(
                    pd.to_datetime(times_all, unit="s"),
                    prices_all,
                    marker="x", color="red", label="All Sell", zorder=3,
                )


        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()


def run_simulation(
    *,
    account: str | None = None,
    market: str | None = None,
    all_accounts: bool = False,
    verbose: int = 0,
    timeframe: str = "1m",
    viz: bool = True,
) -> None:
    """Iterate configured accounts/markets and run simulations."""
    general = load_general()
    coin_settings = load_coin_settings()
    accounts_cfg = load_account_settings()
    accounts = accounts_cfg
    targets = accounts.keys() if (all_accounts or not account) else [account]
    for acct_name in targets:
        acct_cfg = accounts.get(acct_name)
        if not acct_cfg:
            addlog(
                f"[ERROR] Unknown account {acct_name}",
                verbose_int=1,
                verbose_state=verbose,
            )
            continue
        client = ccxt.kraken({"enableRateLimit": True})
        markets_cfg = acct_cfg.get("market settings", {})
        m_targets = [market] if market else list(markets_cfg.keys())
        for m in m_targets:
            if m not in markets_cfg:
                continue
            addlog(
                f"[RUN][{acct_name}][{m}]",
                verbose_int=1,
                verbose_state=verbose,
            )
            _run_single_sim(
                general=general,
                coin_settings=coin_settings,
                accounts_cfg=accounts_cfg,
                account=acct_name,
                market=m,
                client=client,
                verbose=verbose,
                timeframe=timeframe,
                viz=viz,
            )
