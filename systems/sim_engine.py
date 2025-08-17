from __future__ import annotations

"""Historical simulation engine for position-based strategy."""

import shutil
from datetime import datetime, timezone
import csv
import json

import pandas as pd
from tqdm import tqdm

from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import (
    apply_buy,
    apply_sell,
    paper_execute_buy,
    paper_execute_sell,
)
from systems.scripts.strategy_jackpot import (
    init_jackpot,
    on_buy_drip,
    maybe_periodic_jackpot_buy,
    maybe_cashout_jackpot,
)
from systems.utils.addlog import addlog
from pathlib import Path

from systems.utils.config import load_settings, load_ledger_config, resolve_path
from systems.utils.resolve_symbol import (
    split_tag,
    resolve_ccxt_symbols,
    to_tag,
    sim_path_csv,
)
from systems.utils.time import parse_cutoff as parse_timeframe


def run_simulation(
    *,
    ledger: str,
    verbose: int = 0,
    timeframe: str = "1m",
    viz: bool = True,
) -> None:
    settings = load_settings()
    ledger_cfg = load_ledger_config(ledger)
    base, _ = split_tag(ledger_cfg["tag"])
    coin = base.upper()
    strategy_cfg = settings.get("general_settings", {}).get("strategy_settings", {})

    kraken_symbol, _ = resolve_ccxt_symbols(settings, ledger)
    tag = to_tag(kraken_symbol)
    csv_path = sim_path_csv(tag)
    if not Path(csv_path).exists():
        print(
            f"[ERROR] Missing data file: {csv_path}. Run: python bot.py --mode fetch --ledger {ledger}"
        )
        raise SystemExit(1)
    df = pd.read_csv(csv_path)

    # Normalize + guard
    ts_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("timestamp", "time", "date"):
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column in {csv_path}")

    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    before = len(df)
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    removed = before - len(df)

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
        settings,
        ledger_cfg,
        mode="sim",
        prev={"verbose": verbose},
    )
    runtime_state["mode"] = "sim"
    runtime_state["buy_unlock_p"] = {}
    init_jackpot(runtime_state, ledger_cfg, df)

    ledger_obj = Ledger()
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

    for t in tqdm(range(total), desc="📉 Sim Progress", dynamic_ncols=True):
        price = float(df.iloc[t]["close"])

        ctx = {"ledger": ledger_obj}
        buy_res = evaluate_buy(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
            runtime_state=runtime_state,
        )
        if buy_res:
            ts = None
            if "timestamp" in df.columns:
                ts = int(df.iloc[t]["timestamp"])

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

            if viz:
                buy_points.append((float(df.iloc[t][ts_col]), price))

        open_notes = ledger_obj.get_open_notes()
        sell_orders = evaluate_sell(
            ctx,
            t,
            df,
            cfg=strategy_cfg,
            open_notes=open_notes,
            runtime_state=runtime_state,
        )

        for order in sell_orders:
            note = order["note"]
            amt = order["sell_amount"]
            mode = order.get("sell_mode", "normal")
            entry_price = note.get("entry_price", 0.0)

            ts = None
            if "timestamp" in df.columns:
                ts = int(df.iloc[t]["timestamp"])
            result = paper_execute_sell(price, amt, timestamp=ts)

            if viz:
                sell_points.append((float(df.iloc[t][ts_col]), price, mode))

            if amt >= note.get("entry_amount", 0.0) - 1e-9:
                note["sell_mode"] = mode
                apply_sell(
                    ledger=ledger_obj,
                    note=note,
                    t=t,
                    result=result,
                    state=runtime_state,
                )
            else:
                partial = note.copy()
                partial["entry_amount"] = amt
                partial["entry_usdt"] = amt * entry_price
                partial["sell_mode"] = mode
                ledger_obj.open_note(partial)
                apply_sell(
                    ledger=ledger_obj,
                    note=partial,
                    t=t,
                    result=result,
                    state=runtime_state,
                )
                note["entry_amount"] -= amt
                note["entry_usdt"] -= amt * entry_price

            cost = entry_price * amt
            proceeds = result.get("avg_price", 0.0) * amt
            roi_trade = (proceeds - cost) / cost if cost > 0 else 0.0
            m_sell = win_metrics.get("strategy")
            if m_sell is not None:
                m_sell["sells"] += 1
                m_sell["realized_cost"] += cost
                m_sell["realized_proceeds"] += proceeds
                m_sell["realized_trades"] += 1
                m_sell["realized_roi_accum"] += roi_trade

        ctx_j = {
            "ledger": ledger_obj,
            "verbosity": runtime_state.get("verbose", 0),
        }
        maybe_periodic_jackpot_buy(
            ctx_j,
            runtime_state,
            t,
            df,
            price,
            runtime_state.get("limits", {}),
            ledger_cfg["tag"],
        )
        maybe_cashout_jackpot(
            ctx_j,
            runtime_state,
            t,
            df,
            price,
            runtime_state.get("limits", {}),
            ledger_cfg["tag"],
        )

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
    j = runtime_state.get("jackpot", {})
    coin_value = sum(
        n.get("entry_amount", 0.0) * final_price for n in j.get("notes_open", [])
    )
    jackpot_total = j.get("pool_usd", 0.0) + coin_value
    if j.get("enabled"):
        addlog(
            f"[REPORT][JACKPOT] drips=${j.get('drips',0.0):.2f} buys={j.get('buys',0)} sells={j.get('sells',0)} realized_pnl=${j.get('realized_pnl',0.0):.2f} pool_left=${j.get('pool_usd',0.0):.2f} coin_value={coin_value:.2f} total={jackpot_total:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
    addlog(
        f"Final Value (USD): ${summary['total_value']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    root = resolve_path("")
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = logs_dir / f"sim_report_{ledger}_{ts}.csv"
    json_path = logs_dir / f"sim_report_{ledger}_{ts}.json"
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
        if j.get("enabled"):
            f_csv.write("\n")
            f_csv.write(
                "jackpot_drips,jackpot_buys,jackpot_sells,jackpot_realized_pnl,jackpot_pool_left,jackpot_coin_value,jackpot_total\n"
            )
            f_csv.write(
                f"{j.get('drips',0.0)},{j.get('buys',0)},{j.get('sells',0)},{j.get('realized_pnl',0.0)},{j.get('pool_usd',0.0)},{coin_value},{jackpot_total}\n"
            )
    json_data = {
        "windows": rows,
        "global": {
            "cash": wallet_cash,
            "open_value_now": open_value,
            "total_at_liq": global_total_at_liq,
        },
    }
    if j.get("enabled"):
        json_data["jackpot"] = {
            "drips": j.get("drips", 0.0),
            "buys": j.get("buys", 0),
            "sells": j.get("sells", 0),
            "realized_pnl": j.get("realized_pnl", 0.0),
            "pool_left": j.get("pool_usd", 0.0),
            "coin_value": coin_value,
            "total": jackpot_total,
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


        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    save_ledger(
        ledger,
        ledger_obj,
        sim=True,
        final_tick=total - 1,
        summary=summary,
        tag=ledger_cfg["tag"],
    )
    default_path = root / "data" / "tmp" / "simulation" / f"{ledger}.json"
    sim_path = root / "data" / "tmp" / f"simulation_{ledger}.json"
    if default_path.exists() and default_path != sim_path:
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_path, sim_path)
