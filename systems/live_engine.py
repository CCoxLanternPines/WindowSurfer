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
from systems.utils.settings_loader import get_strategy_cooldown, load_settings
from systems.scripts.execution_handler import get_available_fiat_balance, buy_order, sell_order
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.kraken_utils import get_kraken_balance

try:
    import msvcrt  # Windows-only
except ImportError:
    msvcrt = None

if datetime.utcnow().minute != 0:
    print("[SKIP] Not top of the hour. Exiting.")
    sys.exit(0)

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

    def get_capital():
        return get_available_fiat_balance(exchange, meta["fiat"])

    evaluate_buy_df(
        candle=candle,
        window_data=window_data,
        tick=0,
        cooldowns=cooldowns,
        last_triggered=last_triggered,
        tag=tag,
        sim=False,
        verbose=verbose,
        ledger=ledger,
        get_capital=get_capital,
        meta=meta
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

def run_live(tag: str, window: str, verbose: int = 0) -> None:
    addlog(f"[LIVE] Running live mode for {tag} on window {window}", verbose_int=1, verbose_state=verbose)

    settings = load_settings()
    meta = settings["symbol_settings"][tag]

    ledger = load_ledger(tag)
    cooldowns = {
        "knife_catch": 0,
        "whale_catch": 0,
        "fish_catch": 0,
    }
    last_triggered = {
        "knife_catch": None,
        "whale_catch": None,
        "fish_catch": None,
    }

    ensure_latest_candles(tag, lookback="48h", verbose=verbose)
    candle = get_candle_data_json(tag, row_offset=0)
    window_data = get_window_data_json(tag, window, candle_offset=0)

    exchange = ccxt.kraken({"enableRateLimit": True})
    usd_balance = get_available_fiat_balance(exchange, meta["fiat"])

    kraken_balance = get_kraken_balance(verbose)
    fiat_asset = meta["fiat"]
    wallet_code = meta.get("wallet_code", meta["kraken_name"].replace("USD", ""))

    available_usd = float(kraken_balance.get(fiat_asset, 0.0))
    available_coin = float(kraken_balance.get(wallet_code, 0.0))
    coin_price = candle["close"]
    coin_balance_usd = available_coin * coin_price

    if candle and window_data:
        evaluate_live_tick(
            candle=candle,
            window_data=window_data,
            ledger=ledger,
            cooldowns=cooldowns,
            last_triggered=last_triggered,
            tag=tag,
            meta=meta,
            exchange=exchange,
            verbose=verbose
        )
        save_ledger(tag, ledger)

    report = format_top_of_hour_report(
            symbol=tag,
            ts=datetime.now(),
            usd_balance=available_usd,
            coin_balance_usd=coin_balance_usd,
            coin_symbol=wallet_code,
            total_liquid_value=available_usd + coin_balance_usd,
            triggered_strategies={
                "Fish": last_triggered["fish_catch"] is not None,
                "Whale": last_triggered["whale_catch"] is not None,
                "Knife": last_triggered["knife_catch"] is not None,
            },
            note_counts={
                "Fish": (sum(1 for n in ledger.get_active_notes() if n["strategy"] == "fish_catch"),
                         sum(1 for n in ledger.get_closed_notes() if n["strategy"] == "fish_catch")),
                "Whale": (sum(1 for n in ledger.get_active_notes() if n["strategy"] == "whale_catch"),
                          sum(1 for n in ledger.get_closed_notes() if n["strategy"] == "whale_catch")),
                "Knife": (sum(1 for n in ledger.get_active_notes() if n["strategy"] == "knife_catch"),
                          sum(1 for n in ledger.get_closed_notes() if n["strategy"] == "knife_catch")),
            }
        )
    addlog(report, verbose_int=0, verbose_state=verbose)

    addlog("[EXIT] Completed top-of-hour logic. Exiting live mode.", verbose_int=1, verbose_state=verbose)
