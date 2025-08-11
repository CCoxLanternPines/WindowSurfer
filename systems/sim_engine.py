from __future__ import annotations

"""Historical simulation engine for position-based strategy."""

import shutil
from collections import defaultdict
from datetime import datetime
import csv
import json

from tqdm import tqdm

from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.utils.addlog import addlog
from systems.utils.config import load_settings, load_ledger_config, resolve_path


def run_simulation(*, ledger: str, verbose: int = 0) -> None:
    settings = load_settings()
    ledger_cfg = load_ledger_config(ledger)
    tag = ledger_cfg.get("tag", "").upper()
    window_settings = ledger_cfg.get("window_settings", {})

    df = fetch_candles(tag)
    total = len(df)

    runtime_state = {
        "capital": float(settings.get("simulation_capital", 0.0)),
        "buy_unlock_p": {},
        "verbose": verbose,
    }

    ledger_obj = Ledger()
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
    addlog(f"[SIM] Starting simulation for {tag}", verbose_int=1, verbose_state=verbose)

    for t in tqdm(range(total), desc="📉 Sim Progress", dynamic_ncols=True):
        price = float(df.iloc[t]["close"])

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
                size_usd = buy_res["size_usd"]
                amount = size_usd / price if price else 0.0
                note = {
                    "id": f"{window_name}-{t}",
                    "entry_idx": t,
                    "entry_price": price,
                    "entry_usdt": size_usd,
                    "entry_amount": amount,
                    "window_name": buy_res["window_name"],
                    "window_size": buy_res["window_size"],
                    "p_buy": buy_res["p_buy"],
                    "target_price": buy_res["target_price"],
                    "target_roi": buy_res["target_roi"],
                    "unlock_p": buy_res["unlock_p"],
                }
                if "created_idx" in buy_res:
                    note["created_idx"] = buy_res["created_idx"]
                if "created_ts" in buy_res:
                    note["created_ts"] = buy_res["created_ts"]
                ledger_obj.open_note(note)
                runtime_state["capital"] -= size_usd
                runtime_state["buy_unlock_p"][window_name] = buy_res["unlock_p"]

                m_buy = win_metrics.get(window_name)
                if m_buy is not None:
                    cost = price * amount
                    m_buy["buys"] += 1
                    m_buy["gross_invested"] += cost

            open_notes = ledger_obj.get_open_notes()
            sell_notes = evaluate_sell(
                ctx,
                t,
                df,
                window_name=window_name,
                cfg=wcfg,
                open_notes=open_notes,
                runtime_state=runtime_state,
            )
            for note in sell_notes:
                exit_usdt = note["entry_amount"] * price if price else 0.0
                note["exit_idx"] = t
                note["exit_price"] = price
                note["exit_usdt"] = exit_usdt
                note["gain"] = exit_usdt - note["entry_usdt"]
                note["gain_pct"] = (
                    note["gain"] / note["entry_usdt"] if note["entry_usdt"] else 0.0
                )

                w = note.get("window_name")
                qty = note.get("entry_amount", 0.0)
                buy_price = note.get("entry_price", 0.0)
                cost = buy_price * qty
                proceeds = price * qty
                roi_trade = (proceeds - cost) / cost if cost > 0 else 0.0
                m_sell = win_metrics.get(w)
                if m_sell is not None:
                    m_sell["sells"] += 1
                    m_sell["realized_cost"] += cost
                    m_sell["realized_proceeds"] += proceeds
                    m_sell["realized_trades"] += 1
                    m_sell["realized_roi_accum"] += roi_trade

                ledger_obj.close_note(note)
                runtime_state["capital"] += exit_usdt

    final_price = float(df.iloc[-1]["close"]) if total else 0.0
    summary = ledger_obj.get_account_summary(final_price)

    open_value_by_window = defaultdict(float)
    for note in ledger_obj.get_open_notes():
        w = note.get("window_name")
        qty = note.get("entry_amount", 0.0)
        open_value_by_window[w] += qty * final_price

    wallet_cash = runtime_state["capital"]
    rows = []
    for w, m in win_metrics.items():
        realized_roi = (
            (m["realized_proceeds"] / m["realized_cost"] - 1.0)
            if m["realized_cost"] > 0
            else 0.0
        )
        avg_trade_roi = (
            m["realized_roi_accum"] / m["realized_trades"]
            if m["realized_trades"] > 0
            else 0.0
        )
        open_value = open_value_by_window.get(w, 0.0)
        total_value_at_liq = m["realized_proceeds"] + open_value
        realized_pnl = m["realized_proceeds"] - m["realized_cost"]
        addlog(
            f"[REPORT][{w} {m['window_size']}] buys={m['buys']} sells={m['sells']} realized_pnl=${realized_pnl:.2f} realized_roi={(realized_roi*100):.2f}% avg_trade_roi={(avg_trade_roi*100):.2f}% open_notes_value=${open_value:.2f} window_total_at_liq=${total_value_at_liq:.2f}",
            verbose_int=1,
            verbose_state=verbose,
        )
        rows.append(
            {
                "window": w,
                "window_size": m["window_size"],
                "buys": m["buys"],
                "sells": m["sells"],
                "gross_invested": m["gross_invested"],
                "realized_cost": m["realized_cost"],
                "realized_proceeds": m["realized_proceeds"],
                "realized_pnl": realized_pnl,
                "realized_roi": realized_roi,
                "avg_trade_roi": avg_trade_roi,
                "open_value_now": open_value,
                "window_total_at_liq": total_value_at_liq,
            }
        )

    global_open_value = sum(open_value_by_window.values())
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

    save_ledger(ledger_cfg["tag"], ledger_obj, sim=True, final_tick=total - 1, summary=summary)
    default_path = root / "data" / "tmp" / "simulation" / f"{ledger}.json"
    sim_path = root / "data" / "tmp" / f"simulation_{ledger}.json"
    if default_path.exists() and default_path != sim_path:
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_path, sim_path)
