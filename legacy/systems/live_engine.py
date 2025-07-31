import time
import sys
import threading
from datetime import datetime, timezone
from tqdm import tqdm
from systems.utils.logger import addlog
from systems.fetch import fetch_missing_candles
from systems.utils.settings_loader import load_settings
from systems.scripts.execution_handler import get_available_fiat_balance, buy_order, sell_order
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.kraken_utils import get_kraken_balance

try:
    import msvcrt  # Windows-only
except ImportError:
    msvcrt = None

def ensure_latest_candles(tag: str, lookback: str = "48h", verbose: int = 1) -> None:
    try:
        addlog(
            f"[SYNC] Checking for missing candles in last {lookback} for {tag}",
            verbose_int=2,
            verbose_state=verbose,
        )
        fetch_missing_candles(tag, relative_window=lookback, verbose=verbose)
    except Exception as e:
        addlog(
            f"[ERROR] Failed to fetch missing candles: {e}",
            verbose_int=2,
            verbose_state=verbose,
        )

def evaluate_live_tick(
    candle: dict,
    window_data: dict,
    ledger,
    cooldowns: dict,
    last_triggered: dict,
    tag: str,
    meta: dict,
    exchange,
    verbose: int = 0
) -> None:
    from systems.scripts.evaluate_buy import evaluate_buy_df
    from systems.scripts.evaluate_sell import evaluate_sell_df

    settings = load_settings()

    def get_capital():
        return get_available_fiat_balance(exchange, meta["fiat"])

    max_note_usdt = meta.get(
        "max_note_usdt",
        settings["general_settings"].get("max_note_usdt", 999999),
    )

    for key in cooldowns:
        cooldowns[key] = max(0, cooldowns[key] - 1)

    active = settings.get(
        "active_strategies",
        ["fish_catch", "whale_catch", "knife_catch"],
    )

    for strat in active:
        evaluate_buy_df(
            candle=candle,
            window_data=window_data,
            tick=0,
            cooldowns=cooldowns,
            last_triggered=last_triggered,
            tag=tag,
            strategy=strat,
            sim=False,
            verbose=verbose,
            ledger=ledger,
            get_capital=get_capital,
            meta=meta,
            max_note_usdt=max_note_usdt,
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
        fills = sell_order(meta["kraken_name"], meta["fiat"], note["entry_usdt"], verbose=verbose)
        note["exit_price"] = fills["price"]
        note["exit_amount"] = fills["volume"]
        note["exit_usdt"] = fills["cost"]
        note["fee"] = fills["fee"]
        note["exit_ts"] = fills["timestamp"]
        note["kraken_txid"] = fills["kraken_txid"]
        note["exit_tick"] = 0
        note["gain_pct"] = (note.get("exit_usdt", 0) - note["entry_usdt"]) / note["entry_usdt"]
        note["status"] = "Closed"
        note["strategy"] = note.get("strategy", "live_entry")
        ledger.close_note(note)

        addlog(
            f"[SELL] Live Tick | Strategy: {note['strategy']} | Gain: {note.get('gain_pct', 0):.2%}",
            verbose_int=1,
            verbose_state=verbose,
        )

def run_live(tag: str | None, window: str, verbose: int = 0) -> None:
    """Continuously execute ``handle_top_of_hour`` every hour."""
    from systems.top_hour import handle_top_of_hour

    settings = load_settings()
    tags = [tag.upper()] if tag else list(settings.get("symbol_settings", {}))

    addlog(
        f"[LIVE] Running live mode on window {window} for: {', '.join(tags)}",
        verbose_int=2,
        verbose_state=verbose,
    )

    should_exit: list[bool] = []

    def esc_listener():
        if not msvcrt:
            return
        while True:
            if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                should_exit.append(True)
                break

    if msvcrt:
        threading.Thread(target=esc_listener, daemon=True).start()

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
                    addlog("[EXIT] ESC pressed. Exiting live mode.", verbose_int=1, verbose_state=verbose)
                    return
                time.sleep(1)
                pbar.update(1)

        for t in tags:
            if t not in settings.get("symbol_settings", {}):
                addlog(f"[WARN] Unknown symbol tag: {t}", verbose_int=1, verbose_state=verbose)
                continue

            try:
                handle_top_of_hour(t, window=window, verbose=verbose)
            except KeyboardInterrupt:
                addlog("[EXIT] KeyboardInterrupt received. Exiting live mode.", verbose_int=1, verbose_state=verbose)
                return

        addlog(
            "[CYCLE] Top-of-hour cycle complete. Waiting for next hour...",
            verbose_int=2,
            verbose_state=verbose,
        )
