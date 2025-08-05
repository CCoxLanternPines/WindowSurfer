from __future__ import annotations

"""Execute trading logic at the top of each hour."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import json
import pandas as pd

from systems.scripts.evaluate_buy import evaluate_buy
from systems.scripts.evaluate_sell import evaluate_sell
from systems.scripts.get_window_data import get_wave_window_data_df
from systems.scripts.kraken_utils import get_live_price
from systems.scripts.execution_handler import (
    execute_buy,
    execute_sell,
    load_or_fetch_snapshot,
)
from systems.scripts.ledger import Ledger, save_ledger
from systems.utils.addlog import addlog, send_telegram_message
from systems.scripts.send_top_hour_report import send_top_hour_report
from systems.utils.path import find_project_root
from systems.utils.top_hour_report import format_top_of_hour_report
from systems.utils.resolve_symbol import split_tag
from systems.scripts.window_position_tools import get_trade_params


def handle_top_of_hour(
    *,
    tick: int | datetime,
    sim: bool,
    settings: dict | None = None,
    candle: dict | None = None,
    ledger: Ledger | None = None,
    ledger_config: dict | None = None,
    **kwargs: Any,
) -> None:
    """Run buy/sell evaluations for all windows on an hourly boundary.

    Parameters
    ----------
    tick:
        Current tick index.
    candle:
        Candle data for this tick.
    ledger:
        Ledger tracking open and closed notes.
    ledger_config:
        Ledger-specific configuration.
    sim:
        ``True`` when running in simulation mode. Live logic is not yet
        implemented.
    **kwargs:
        Additional context. The simulation engine passes a DataFrame ``df`` and
        ``offset`` for window calculations along with a mutable ``state``
        dictionary containing capital and cooldown counters.
    """

    if not sim:
        if settings is None:
            return

        root: Path = find_project_root()
        cooldown_path = root / "data" / "tmp" / "cooldowns.json"
        if cooldown_path.exists():
            with open(cooldown_path, "r") as f:
                cooldowns = json.load(f)
        else:
            cooldowns = {}

        dry_run = kwargs.get("dry", False)
        client = kwargs.get("client")

        general_cfg = settings.get("general_settings", {})

        for ledger_name, ledger_cfg in settings.get("ledger_settings", {}).items():
            tag = ledger_cfg["tag"]
            _, quote = split_tag(tag)
            wallet_code = ledger_cfg["wallet_code"]
            window_settings = ledger_cfg.get("window_settings", {})
            triggered_strategies = {wn.title(): False for wn in window_settings}
            strategy_summary: dict[str, dict] = {}
            ledger = Ledger.load_ledger(tag=ledger_cfg["tag"])

            snapshot = load_or_fetch_snapshot(ledger_name)
            if not snapshot:
                addlog(
                    "[ERROR] Kraken snapshot missing — cannot proceed in live mode.",
                    verbose_int=1,
                    verbose_state=True,
                )
                continue
            balance = snapshot.get("balance", {})

            price = get_live_price(kraken_pair=tag)

            current_ts = (
                int(tick.timestamp()) if isinstance(tick, datetime) else int(tick)
            )

            metadata = ledger.get_metadata()
            ledger_cooldowns = cooldowns.get(ledger_name, {})
            last_buy_tick = ledger_cooldowns.get("last_buy_tick", {})

            for window_name, window_cfg in window_settings.items():
                addlog(
                    f"[EVAL] {ledger_name} | {tag} | {window_name} window → evaluating",
                    verbose_int=3,
                    verbose_state=True,
                )

                buy_count = 0
                sell_count = 0

                try:
                    df = pd.read_csv(root / "data" / "raw" / f"{tag}.csv")
                except Exception:
                    df = None

                wave = get_wave_window_data_df(
                    df,
                    window=window_cfg["window_size"],
                    candle_offset=0,
                )

                if wave:
                    trade = get_trade_params(
                        current_price=price,
                        window_high=wave["ceiling"],
                        window_low=wave["floor"],
                        config=window_cfg,
                    )
                    if trade["in_dead_zone"]:
                        addlog(
                            f"[SKIP] {ledger_name} | {tag} | {window_name} → In dead zone",
                            verbose_int=3,
                            verbose_state=True,
                        )
                        continue

                    position = trade["pos_pct"]
                    buy_cd = window_cfg.get("buy_cooldown", 0) * 3600
                    last_buy = last_buy_tick.get(window_name, float("-inf"))
                    if position <= window_cfg.get("buy_floor", 0) and (
                        dry_run or current_ts - last_buy >= buy_cd
                    ):
                        open_for_window = [
                            n
                            for n in ledger.get_active_notes()
                            if n.get("window") == window_name
                        ]
                        if len(open_for_window) < window_cfg.get("max_open_notes", 0):
                            available = float(balance.get(quote, 0.0))
                            invest = available * window_cfg.get(
                                "investment_fraction", 0
                            )
                            max_usd = general_cfg.get("max_note_usdt", invest)
                            min_usd = general_cfg.get("minimum_note_size", 0.0)
                            invest = min(invest, max_usd)
                            if invest >= min_usd and invest <= available and invest > 0:
                                result = execute_buy(
                                    client=client,
                                    symbol=tag,
                                    price=price,
                                    amount_usd=invest,
                                    ledger_name=ledger_name,
                                    wallet_code=wallet_code,
                                )
                                if result:
                                    note = {
                                        "entry_amount": result["filled_amount"],
                                        "entry_price": result["avg_price"],
                                        "entry_ts": result["timestamp"],
                                        "entry_tick": current_ts,
                                        "window": window_name,
                                        "status": "Open",
                                    }
                                    ledger.open_note(note)
                                    if not dry_run:
                                        last_buy_tick[window_name] = current_ts
                                    buy_count += 1
                                    msg = (
                                        f"[LIVE][BUY] {ledger_name} | {tag} | "
                                        f"{result['filled_amount']:.4f} {wallet_code} @ "
                                        f"${result['avg_price']:.3f}"
                                    )
                                    addlog(msg)
                                    send_telegram_message(msg)
                    else:
                        reasons = []
                        if position > window_cfg.get("buy_floor", 0):
                            reasons.append(
                                f"position={position:.2f} above floor={window_cfg.get('buy_floor', 0)}"
                            )
                        if not dry_run and current_ts - last_buy < buy_cd:
                            remaining = buy_cd - (current_ts - last_buy)
                            reasons.append(
                                f"cooldown active ({remaining // 60}m left)"
                            )
                        if not reasons:
                            reasons.append("unknown gating condition")

                        addlog(
                            f"[SKIP] {ledger_name} | {tag} | {window_name} → Buy blocked: {', '.join(reasons)}",
                            verbose_int=3,
                            verbose_state=True,
                        )


                    for note in list(ledger.get_active_notes()):
                        if note.get("window") != window_name:
                            continue
                        gain_pct = (price - note["entry_price"]) / note["entry_price"]
                        trade_note = get_trade_params(
                            current_price=price,
                            window_high=wave["ceiling"],
                            window_low=wave["floor"],
                            config=window_cfg,
                            entry_price=note["entry_price"],
                        )
                        maturity_roi = trade_note["maturity_roi"]
                        addlog(
                            f"[DEBUG][LIVE SELL] gain_pct={gain_pct:.2%} maturity_roi={maturity_roi:.2%}",
                            verbose_int=3,
                            verbose_state=True,
                        )
                        if maturity_roi is None or gain_pct < maturity_roi:
                            continue
                        result = execute_sell(
                            client=client,
                            symbol=tag,
                            coin_amount=note["entry_amount"],
                            ledger_name=ledger_name,
                        )
                        note["exit_price"] = result["avg_price"]
                        note["exit_ts"] = result["timestamp"]
                        note["exit_tick"] = current_ts
                        gain = (note["exit_price"] - note["entry_price"]) * note["entry_amount"]
                        note["gain"] = gain
                        base = note["entry_price"] * note["entry_amount"] or 1
                        note["gain_pct"] = gain / base
                        note["status"] = "Closed"
                        ledger.close_note(note)
                        sell_count += 1
                        addlog(
                            f"[LIVE][SELL] {ledger_name} | {tag} | Gain: ${gain:.2f} ({note['gain_pct']:.2%})",
                        )
                if buy_count > 0 or sell_count > 0:
                    triggered_strategies[window_name.title()] = True

                summary = ledger.get_account_summary(price)
                idle_capital = float(balance.get(quote, 0.0))
                summary["idle_capital"] = idle_capital
                summary["total_value"] += idle_capital
                hour_str = datetime.now().strftime("%I:%M%p")
                addlog(
                    f"[SUMMARY] {hour_str} | {ledger_name} | \U0001F4B0 ${summary['total_value']:.2f} | "
                    f"\U0001F4B5 ${summary['idle_capital']:.2f} | \U0001FA99 ${summary['open_value']:.2f}",
                    verbose_int=2,
                    verbose_state=True,
                )
                message = (
                    f"[LIVE] {ledger_name} | {tag} | {window_name} window\n"
                    f"✅ Buy attempts: {buy_count} | Sells: {sell_count} | "
                    f"Open Notes: {summary['open_notes']} | Realized Gain: ${summary['realized_gain']:.2f}"
                )
                addlog(
                    message,
                    verbose_int=1,
                    verbose_state=True,
                )

                open_notes_w = [
                    n for n in ledger.get_open_notes() if n.get("window") == window_name
                ]
                closed_notes_w = [
                    n for n in ledger.get_closed_notes() if n.get("window") == window_name
                ]
                unrealized = sum(
                    (price - n.get("entry_price", 0.0)) * n.get("entry_amount", 0.0)
                    for n in open_notes_w
                )
                realized = sum(n.get("gain", 0.0) for n in closed_notes_w)
                total_gain = unrealized + realized
                invested = sum(
                    n.get("entry_price", 0.0) * n.get("entry_amount", 0.0)
                    for n in open_notes_w + closed_notes_w
                )
                roi = (total_gain / invested * 100.0) if invested else 0.0
                strategy_summary[window_name.title()] = {
                    "buys": buy_count,
                    "sells": sell_count,
                    "open": len(open_notes_w),
                    "roi": roi,
                    "total": total_gain,
                }

            if not dry_run:
                metadata["last_buy_tick"] = last_buy_tick
            ledger.set_metadata(metadata)
            save_ledger(ledger_cfg["tag"], ledger)

            usd_balance = float(balance.get(quote, 0.0))
            coin_balance = float(balance.get(wallet_code, 0.0))
            coin_balance_usd = coin_balance * price
            total_liquid_value = usd_balance + coin_balance_usd
            note_counts = {}
            for win in window_settings.keys():
                open_n = sum(
                    1 for n in ledger.get_open_notes() if n.get("window") == win
                )
                closed_n = sum(
                    1 for n in ledger.get_closed_notes() if n.get("window") == win
                )
                note_counts[win.title()] = (open_n, closed_n)
            report = format_top_of_hour_report(
                tag,
                datetime.utcnow(),
                usd_balance,
                coin_balance_usd,
                wallet_code,
                total_liquid_value,
                triggered_strategies,
                note_counts,
            )
            addlog(report, verbose_int=1, verbose_state=True)

            send_top_hour_report(
                ledger_name=ledger_name,
                tag=tag,
                strategy_summary=strategy_summary,
                verbose=general_cfg.get("verbose", 0),
            )

            if not dry_run:
                cooldowns[ledger_name] = {
                    "last_buy_tick": last_buy_tick,
                }

        if not dry_run:
            cooldown_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cooldown_path, "w") as f:
                json.dump(cooldowns, f, indent=2)

        return

    if candle is None or ledger is None or ledger_config is None:
        return

    # Extract common context
    df = kwargs.get("df")
    offset = kwargs.get("offset")
    state: Dict[str, Any] = kwargs.get("state", {})
    verbose = kwargs.get("verbose", 0)

    windows = ledger_config.get("window_settings", {})
    if not windows or df is None or offset is None:
        return

    price = float(candle.get("close", 0.0))

    for name, cfg in windows.items():
        wave = get_wave_window_data_df(
            df,
            window=cfg["window_size"],
            candle_offset=offset,
        )
        if not wave:
            continue

        sim_capital = state.get("capital", 0.0)
        max_note_usdt = kwargs.get("max_note_usdt", sim_capital)
        min_note_usdt = kwargs.get("min_note_usdt", 0.0)

        sim_capital, buy_skipped = evaluate_buy(
            ledger=ledger,
            name=name,
            cfg=cfg,
            wave=wave,
            tick=tick,
            price=price,
            sim_capital=sim_capital,
            last_buy_tick=state.get("last_buy_tick", {}),
            max_note_usdt=max_note_usdt,
            min_note_usdt=min_note_usdt,
            verbose=verbose,
        )
        state["capital"] = sim_capital
        if buy_skipped:
            state.get("buy_cooldown_skips", {}).setdefault(name, 0)
            state["buy_cooldown_skips"][name] += 1

        sim_capital, closed, roi_skipped = evaluate_sell(
            ledger=ledger,
            name=name,
            tick=tick,
            price=price,
            wave=wave,
            cfg=cfg,
            sim_capital=sim_capital,
            verbose=verbose,
        )
        state["capital"] = sim_capital
        state["min_roi_gate_hits"] = state.get("min_roi_gate_hits", 0) + roi_skipped

        if closed:
            for note in closed:
                msg = (
                    f"[SELL] Tick {tick} | Window: {note['window']} | "
                    f"Gain: +${note['gain']:.2f} ({note['gain_pct']:.2%})"
                )
                addlog(msg, verbose_int=2, verbose_state=verbose)
                send_telegram_message(msg)

    if sim:
        # Simulation-specific behaviour already covered through state mutation.
        # Placeholder for future live logic.
        pass
