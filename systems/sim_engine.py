from __future__ import annotations

"""Historical simulation engine for position-based strategy."""

import shutil

from tqdm import tqdm

from systems.scripts.fetch_canles import fetch_candles
from systems.scripts.ledger import Ledger, save_ledger
from systems.scripts.evaluate_buy import evaluate_buy_signal
from systems.scripts.evaluate_sell import evaluate_sell_actions
from systems.utils.addlog import addlog
from systems.utils.config import load_settings, load_ledger_config, resolve_path


def run_simulation(*, ledger: str, verbose: int = 0) -> None:
    settings = load_settings()
    ledger_cfg = load_ledger_config(ledger)
    tag = ledger_cfg.get("tag", "").upper()
    cfg = next(iter(ledger_cfg.get("window_settings", {}).values()))

    df = fetch_candles(tag)
    total = len(df)

    runtime_state = {
        "capital": float(settings.get("simulation_capital", 0.0)),
        "buy_unlock_p": None,
        "verbose": verbose,
    }

    ledger_obj = Ledger()
    addlog(f"[SIM] Starting simulation for {tag}", verbose_int=1, verbose_state=verbose)

    for t in tqdm(range(total), desc="📉 Sim Progress", dynamic_ncols=True):
        price = float(df.iloc[t]["close"])

        ctx = {"ledger": ledger_obj}
        buy_res = evaluate_buy_signal(ctx, t, df, cfg, runtime_state)
        if buy_res:
            size_usd = buy_res["size_usd"]
            note_meta = buy_res["note"]
            amount = size_usd / price if price else 0.0
            note = {
                "id": str(t),
                "window": "default",
                "entry_idx": t,
                "entry_price": price,
                "entry_usdt": size_usd,
                "entry_amount": amount,
                **note_meta,
            }
            ledger_obj.open_note(note)
            runtime_state["capital"] -= size_usd
            runtime_state["buy_unlock_p"] = note_meta["unlock_p"]

        open_notes = ledger_obj.get_open_notes()
        sell_notes = evaluate_sell_actions(ctx, t, df, cfg, open_notes, runtime_state)
        sell_notes = sell_notes[: cfg.get("max_notes_sell_per_candle", 1)]
        for note in sell_notes:
            exit_usdt = note["entry_amount"] * price
            note["exit_idx"] = t
            note["exit_price"] = price
            note["exit_usdt"] = exit_usdt
            note["gain"] = exit_usdt - note["entry_usdt"]
            note["gain_pct"] = (
                note["gain"] / note["entry_usdt"] if note["entry_usdt"] else 0.0
            )
            ledger_obj.close_note(note)
            runtime_state["capital"] += exit_usdt

    final_price = float(df.iloc[-1]["close"]) if total else 0.0
    summary = ledger_obj.get_account_summary(final_price)
    addlog(
        f"Final Value (USD): ${summary['total_value']:.2f}",
        verbose_int=1,
        verbose_state=verbose,
    )

    root = resolve_path("")
    save_ledger(ledger_cfg["tag"], ledger_obj, sim=True, final_tick=total - 1, summary=summary)
    default_path = root / "data" / "tmp" / "simulation" / f"{ledger}.json"
    sim_path = root / "data" / "tmp" / f"simulation_{ledger}.json"
    if default_path.exists() and default_path != sim_path:
        sim_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_path, sim_path)
