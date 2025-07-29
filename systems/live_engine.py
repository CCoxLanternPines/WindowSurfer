import time
import sys
import threading
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import ccxt
from systems.utils.logger import addlog
from systems.utils.loggers import logger
from systems.utils.top_hour_report import format_top_of_hour_report
from systems.scripts.get_candle_data import get_candle_data_json
from systems.scripts.get_window_data import get_window_data_json
from systems.fetch import fetch_missing_candles
from systems.utils.settings_loader import get_strategy_cooldown
from systems.scripts.execution_handler import get_available_fiat_balance


try:
    import msvcrt  # Windows-only
except ImportError:
    msvcrt = None


def esc_listener(should_exit_flag):
    if not msvcrt:
        return
    while True:
        if msvcrt and msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':  # ESC
                should_exit_flag.append(True)
                break


def run_live(tag: str, window: str, verbose: int = 0) -> None:
    addlog(f"[LIVE] Running live mode for {tag} on window {window}", verbose_int=1, verbose_state=verbose)

    # Resolve exchange symbols for future use
    from systems.utils.resolve_symbol import resolve_symbol
    symbols = resolve_symbol(tag)

    from systems.scripts.ledger import RamLedger
    ledger = RamLedger()
    cooldowns = {
        "knife_catch": get_strategy_cooldown("knife_catch"),
        "whale_catch": get_strategy_cooldown("whale_catch"),
        "fish_catch": get_strategy_cooldown("fish_catch"),
    }
    last_triggered = {
        "knife_catch": None,
        "whale_catch": None,
        "fish_catch": None,
    }

    should_exit = []

    if msvcrt:
        threading.Thread(target=esc_listener, args=(should_exit,), daemon=True).start()

    while True:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        elapsed_secs = now.minute * 60 + now.second
        remaining_secs = 3600 - elapsed_secs

        with tqdm(
                total=3600,
                initial=elapsed_secs,
                desc="⏳ Time to next hour",
                bar_format="{l_bar}{bar}| {percentage:3.0f}% {remaining}s",
                leave=True,
                dynamic_ncols=True
        ) as pbar:
            for _ in range(remaining_secs):
                if should_exit:
                    addlog("\n🚪 ESC detected — exiting live mode.", verbose_int=1, verbose_state=verbose)
                    return
                time.sleep(1)
                pbar.update(1)

        now = datetime.now(timezone.utc)
        addlog(
            f"\n🕐 Top of hour reached at {now.strftime('%Y-%m-%d %H:%M:%S %Z')} — Restarting countdown...\n",
            verbose_int=0,
            verbose_state=verbose,
        )

        handle_top_of_hour(
            tag=tag,
            window=window,
            ledger=ledger,
            cooldowns=cooldowns,
            last_triggered=last_triggered,
            verbose=verbose,
        )


def handle_top_of_hour(
    tag: str,
    window: str,
    ledger,
    cooldowns: dict,
    last_triggered: dict,
    verbose: int = 0,
) -> None:
    ensure_latest_candles(tag, lookback="48h", verbose=verbose)

    candle = get_candle_data_json(tag, row_offset=0)
    window_data = get_window_data_json(tag, window, candle_offset=0)

    addlog("[TRACE] Candle and window data pulled.", verbose_int=2, verbose_state=verbose)

    if candle and window_data:
        for strat in cooldowns:
            cooldowns[strat] = max(0, cooldowns[strat] - 1)

        prev_triggers = last_triggered.copy()

        evaluate_live_tick(
            candle=candle,
            window_data=window_data,
            ledger=ledger,
            cooldowns=cooldowns,
            last_triggered=last_triggered,
            tag=tag,
            verbose=verbose
        )

        # Summary report -------------------------------------------------
        from systems.utils.resolve_symbol import resolve_symbol

        exchange = ccxt.kraken({"enableRateLimit": True})
        usd_balance = get_available_fiat_balance(exchange, "USD")
        coin_balance_usd = sum(
            float(n.get("entry_usdt", 0.0)) for n in ledger.get_active_notes()
        )
        total_liquid = usd_balance + coin_balance_usd

        trade_counts = ledger.get_trade_counts_by_strategy()

        def _counts(key: str):
            data = trade_counts.get(key, {"total": 0, "open": 0})
            open_n = data.get("open", 0)
            closed_n = data.get("total", 0) - open_n
            return open_n, closed_n

        note_counts = {
            "Fish": _counts("fish_catch"),
            "Whale": _counts("whale_catch"),
            "Knife": _counts("knife_catch"),
        }

        triggered = {
            "Fish": last_triggered.get("fish_catch") != prev_triggers.get("fish_catch"),
            "Whale": last_triggered.get("whale_catch") != prev_triggers.get("whale_catch"),
            "Knife": last_triggered.get("knife_catch") != prev_triggers.get("knife_catch"),
        }

        coin_symbol = resolve_symbol(tag)["kraken"].split("/")[0]

        report = format_top_of_hour_report(
            symbol=tag,
            ts=datetime.now(),
            usd_balance=usd_balance,
            coin_balance_usd=coin_balance_usd,
            coin_symbol=coin_symbol,
            total_liquid_value=total_liquid,
            triggered_strategies=triggered,
            note_counts=note_counts,
        )

        addlog(
            report,
            verbose_int=1,
            verbose_state=verbose,
        )

    else:
        addlog(
            "[WARN] Missing candle or window data. Skipping this cycle.",
            verbose_int=1,
            verbose_state=verbose,
        )


def evaluate_live_tick(
    candle: dict,
    window_data: dict,
    ledger,
    cooldowns: dict,
    last_triggered: dict,
    tag: str,
    verbose: int = 0
) -> None:
    from systems.scripts.evaluate_buy import evaluate_buy_df
    from systems.scripts.evaluate_sell import evaluate_sell_df
    from systems.scripts.execution_handler import sell_order
    from systems.utils.resolve_symbol import resolve_symbol

    live = True
    symbols = resolve_symbol(tag)
    kraken_symbol = symbols["kraken"]

    exchange = ccxt.kraken({"enableRateLimit": True})

    def get_capital():
        return get_available_fiat_balance(exchange, "USD")

    evaluate_buy_df(
        candle=candle,
        window_data=window_data,
        tick=0,  # No time series index in live mode
        cooldowns=cooldowns,
        last_triggered=last_triggered,
        tag=tag,
        sim=False,
        verbose=verbose,
        ledger=ledger,
        get_capital=get_capital
    )

    to_sell = evaluate_sell_df(
        candle=candle,
        window_data=window_data,
        tick=0,
        notes=ledger.get_active_notes(),
        tag=tag,
        verbose=verbose
    )

    exit_price = candle["close"]

    for note in to_sell:
        if live:
            fills = sell_order(kraken_symbol, note["entry_usdt"])
            note["exit_price"] = fills["price"]
            note["exit_amount"] = fills["volume"]
            note["exit_usdt"] = fills["cost"]
            note["fee"] = fills["fee"]
            note["exit_ts"] = fills["timestamp"]
            note["kraken_txid"] = fills["kraken_txid"]
        else:
            note["exit_price"] = exit_price
            note["exit_ts"] = candle.get("ts", 0)
            note["exit_usdt"] = exit_price * note["entry_amount"]
        note["exit_tick"] = 0
        note["gain_pct"] = (note.get("exit_usdt", 0) - note["entry_usdt"]) / note["entry_usdt"]
        note["status"] = "Closed"
        ledger.close_note(note)

        addlog(
            f"[SELL] Live Tick | Strategy: {note['strategy']} | Gain: {note.get('gain_pct', 0):.2%}",
            verbose_int=1,
            verbose_state=verbose,
        )


def ensure_latest_candles(tag: str, lookback: str = "48h", verbose: int = 1) -> None:
    try:
        addlog(
            f"[SYNC] Checking for missing candles in last {lookback} for {tag}",
            verbose_int=1,
            verbose_state=verbose,
        )
        fetch_missing_candles(tag, relative_window=lookback, verbose=verbose)
    except Exception as e:
        addlog(
            f"[ERROR] Failed to fetch missing candles: {e}",
            verbose_int=1,
            verbose_state=verbose,
        )
