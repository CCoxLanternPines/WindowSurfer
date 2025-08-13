from __future__ import annotations

"""Historical simulation engine for position-based strategy."""

import shutil
from collections import defaultdict
from datetime import datetime
import csv
import json

import pandas as pd
from tqdm import tqdm

from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.runtime_state import build_runtime_state
from systems.scripts.trade_apply import (
    apply_buy_result_to_ledger,
    apply_sell_result_to_ledger,
    paper_execute_buy,
    paper_execute_sell,
)
from systems.scripts import strategy_jackpot
from systems.scripts.report_generator import compute_strategy_report
from systems.utils.addlog import addlog
from systems.utils.config import load_settings, load_ledger_config, resolve_path
from systems.utils.resolve_symbol import split_tag


def _format_jackpot_state(state, price, now_ts, cfg):
    ath = state.get("ath_price") or 0.0
    atl = state.get("atl_price") or 0.0
    _, pos = strategy_jackpot.eligibility(
        price, ath, atl, cfg.get("start_level_frac", 0.5)
    )
    mult = 1.0 + pos * (cfg.get("multiplier_floor", 1.0) - 1.0)
    last_drip_ts = state.get("last_drip_ts")
    period = cfg.get("drip_period_hours", 0.0)
    if last_drip_ts is None:
        next_drip = 0.0
    else:
        next_drip = max(0.0, period - (now_ts - last_drip_ts) / 3600)
    return (
        f"[JACKPOT][STATE] ath=${ath:.2f} atl=${atl:.2f} pos={pos:.2f} mult={mult:.2f} "
        f"total_contributed=${state.get('total_contributed_usd',0.0):.2f} next_drip_in={int(next_drip)}h"
    )

def run_simulation(*, ledger: str, verbose: int = 0) -> None:
    settings = load_settings()
    ledger_cfg = load_ledger_config(ledger)
    base, _ = split_tag(ledger_cfg["tag"])
    coin = base.upper()
    window_settings = ledger_cfg.get("window_settings", {})

    raw_path = resolve_path("") / "data" / "raw" / f"{coin}.csv"
    df = pd.read_csv(raw_path)

    # Normalize + guard
    ts_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("timestamp", "time", "date"):
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column in {raw_path}")

    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    before = len(df)
    df = df.sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    removed = before - len(df)

    # Optional hard check
    if not df[ts_col].is_monotonic_increasing:
        raise ValueError(f"Candles not sorted by {ts_col}: {raw_path}")

    # Log one line so we always know what we ran on
    first_ts = int(df[ts_col].iloc[0]) if len(df) else None
    last_ts = int(df[ts_col].iloc[-1]) if len(df) else None
    print(f"[DATA] file={raw_path} rows={len(df)} first={first_ts} last={last_ts} dups_removed={removed}")

    total = len(df)

    runtime_state = build_runtime_state(
        settings,
        ledger_cfg,
        mode="sim",
        prev={"verbose": verbose},
    )
    runtime_state["buy_unlock_p"] = {}

    ledger_obj = Ledger()
    jackpot_cfg = settings.get("jackpot", {})
    jackpot_state = None
    if jackpot_cfg.get("enabled"):
        first_ts = int(df.iloc[0]["timestamp"]) if "timestamp" in df.columns and len(df) else 0
        jackpot_state = strategy_jackpot.init_state(ledger_cfg.get("tag", ""), ledger_obj, jackpot_cfg, first_ts)
    win_metrics = {}
    for wname, wcfg in window_settings.items():
        win_metrics[wname] = {
            "window_size": str(wcfg.get("window_size", "")),
            "buys": 0,
            "sells": 0,
            "gross_invested": 0.0,
            "realized_cost": 0.0,
            "realized_proceeds": 0.0,
            "realized_trades": 0,
            "realized_roi_accum": 0.0,
        }
    addlog(f"[SIM] Starting simulation for {coin}", verbose_int=1, verbose_state=verbose)

    for t in tqdm(range(total), desc="📉 Sim Progress", dynamic_ncols=True):
        price = float(df.iloc[t]["close"])

        if jackpot_cfg.get("enabled") and jackpot_state is not None:
            ath, atl = strategy_jackpot.update_reference_levels(
                df.iloc[: t + 1], jackpot_cfg.get("ath_scope", "dataset")
            )
            jackpot_state["ath_price"], jackpot_state["atl_price"] = ath, atl
            realized = sum(n.get("gain", 0.0) for n in ledger_obj.get_closed_notes())
            realized += jackpot_state.get("realized_pnl", 0.0)
            now_ts = int(df.iloc[t]["timestamp"]) if "timestamp" in df.columns else t
            sell_sig = strategy_jackpot.evaluate_sell(jackpot_state, price, jackpot_cfg)
            if sell_sig:
                result = paper_execute_sell(price, sell_sig.get("qty", 0.0), timestamp=now_ts)
                strategy_jackpot.apply_fills(
                    jackpot_state,
                    [
                        {
                            **sell_sig,
                            "price": result.get("avg_price", price),
                            "usd": result.get("avg_price", price) * result.get("filled_amount", 0.0),
                            "timestamp": now_ts,
                        }
                    ],
                    ledger=ledger_obj,
                    runtime_state=runtime_state,
                )
            else:
                buy_sig = strategy_jackpot.evaluate_buy(
                    jackpot_state, price, now_ts, realized, jackpot_cfg
                )
                if buy_sig:
                    result = paper_execute_buy(price, buy_sig.get("usd", 0.0), timestamp=now_ts)
                    strategy_jackpot.apply_fills(
                        jackpot_state,
                        [
                            {
                                **buy_sig,
                                "price": result.get("avg_price", price),
                                "timestamp": now_ts,
                            }
                        ],
                        ledger=ledger_obj,
                        runtime_state=runtime_state,
                    )

        for window_name, wcfg in window_settings.items():
            ctx = {"ledger": ledger_obj}
            buy_res = evaluate_buy(
                ctx,
                t,
                df,
                window_name=window_name,
                cfg=wcfg,
                runtime_state=runtime_state,
            )
            if buy_res:
                ts = None
                if "timestamp" in df.columns:
                    ts = int(df.iloc[t]["timestamp"])
                result = paper_execute_buy(price, buy_res["size_usd"], timestamp=ts)
                note = apply_buy_result_to_ledger(
                    ledger=ledger_obj,
                    window_name=window_name,
                    t=t,
                    meta=buy_res,
                    result=result,
                    state=runtime_state,
                )
                runtime_state.setdefault("buy_unlock_p", {})[window_name] = buy_res.get("unlock_p")
                if runtime_state["capital"] < -1e-9:
                    addlog(
                        f"[BUG] capital negative after buy: ${runtime_state['capital']:.2f}",
                        verbose_int=1,
                        verbose_state=verbose,
                    )
                    runtime_state["capital"] = 0.0

                m_buy = win_metrics.get(window_name)
                if m_buy is not None:
                    cost = result["filled_amount"] * result["avg_price"]
                    m_buy["buys"] += 1
                    m_buy["gross_invested"] += cost

            open_notes = ledger_obj.get_open_notes()
            sell_res = evaluate_sell(
                ctx,
                t,
                df,
                window_name=window_name,
                cfg=wcfg,
                open_notes=open_notes,
                runtime_state=runtime_state,
            )

            if __debug__ and not runtime_state.get("_sell_shape_logged"):
                if isinstance(sell_res, list):
                    addlog("[SIM] evaluate_sell returned list", verbose_int=2, verbose_state=verbose)
                elif isinstance(sell_res, dict):
                    addlog("[SIM] evaluate_sell returned dict", verbose_int=2, verbose_state=verbose)
                runtime_state["_sell_shape_logged"] = True

            if isinstance(sell_res, list):
                sell_notes = sell_res
            elif isinstance(sell_res, dict):
                sell_notes = sell_res.get("notes", [])
            else:
                sell_notes = []

            for note in sell_notes:
                ts = None
                if "timestamp" in df.columns:
                    ts = int(df.iloc[t]["timestamp"])
                result = paper_execute_sell(price, note.get("entry_amount", 0.0), timestamp=ts)
                apply_sell_result_to_ledger(
                    ledger=ledger_obj,
                    note=note,
                    t=t,
                    result=result,
                    state=runtime_state,
                )

                w = note.get("window_name")
                qty = note.get("entry_amount", 0.0)
                buy_price = note.get("entry_price", 0.0)
                exit_price = note.get("exit_price", 0.0)
                cost = buy_price * qty
                proceeds = exit_price * qty
                roi_trade = (proceeds - cost) / cost if cost > 0 else 0.0
                m_sell = win_metrics.get(w)
                if m_sell is not None:
                    m_sell["sells"] += 1
                    m_sell["realized_cost"] += cost
                    m_sell["realized_proceeds"] += proceeds
                    m_sell["realized_trades"] += 1
                    m_sell["realized_roi_accum"] += roi_trade


    final_price = float(df.iloc[-1]["close"]) if total else 0.0
    final_ts = int(df.iloc[-1]["timestamp"]) if "timestamp" in df.columns and total else 0
    summary = ledger_obj.get_account_summary(final_price)

    wallet_cash = runtime_state["capital"]
    stats = compute_strategy_report(ledger_obj, {"_": final_price})
    rows = []
    for name in sorted(stats.keys()):
        s = stats[name]
        realized_pnl = s["realized_pnl"]
        realized_roi = s["realized_roi"]
        avg_trade_roi = s["avg_trade_roi"]
        open_value = s["open_value_now"]
        total_at_liq = s["window_total_at_liq"]
        addlog(
            f"[REPORT][{name}] buys={s['buys']} sells={s['sells']} realized_pnl=${realized_pnl:.2f} realized_roi={(realized_roi*100):.2f}% avg_trade_roi={(avg_trade_roi*100):.2f}% open_notes_value=${open_value:.2f} window_total_at_liq=${total_at_liq:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        rows.append(
            {
                "window": name,
                "window_size": s.get("window_size"),
                "buys": s["buys"],
                "sells": s["sells"],
                "gross_invested": s["gross_invested"],
                "realized_cost": s["realized_cost"],
                "realized_proceeds": s["realized_proceeds"],
                "realized_pnl": realized_pnl,
                "realized_roi": realized_roi,
                "avg_trade_roi": avg_trade_roi,
                "open_value_now": open_value,
                "window_total_at_liq": total_at_liq,
            }
        )
    if "jackpot" in stats and jackpot_state is not None:
        addlog(
            _format_jackpot_state(jackpot_state, final_price, final_ts, jackpot_cfg),
            verbose_int=3,
            verbose_state=verbose,
        )
    if "jackpot" in stats and not any(r["window"] == "jackpot" for r in rows):
        raise RuntimeError("jackpot stats missing from windows report")

    global_open_value = sum(s["open_value_now"] for s in stats.values())
    sum_open = sum(r["open_value_now"] for r in rows)
    if abs(sum_open - global_open_value) <= 1e-6:
        addlog("[REPORT][ASSERT] PASS open_value tally", verbose_int=3, verbose_state=verbose)
    else:
        raise AssertionError("global open_value mismatch")
    global_total_at_liq = wallet_cash + global_open_value
    addlog(
        f"[REPORT][GLOBAL] cash=${wallet_cash:.2f} open_value=${global_open_value:.2f} total_at_liq=${global_total_at_liq:.2f}",
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
    json_data = {
        "windows": rows,
        "global": {
            "cash": wallet_cash,
            "open_value_now": global_open_value,
            "total_at_liq": global_total_at_liq,
        },
    }
    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(json_data, f_json, indent=2)

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
