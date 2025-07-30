"""Run top-of-hour logic for one or more symbols."""

from __future__ import annotations

from datetime import datetime
import ccxt

from systems.utils.logger import addlog
from systems.utils.top_hour_report import format_top_of_hour_report
from systems.utils.settings_loader import load_settings
from systems.live_engine import (
    ensure_latest_candles,
    evaluate_live_tick,
)
from systems.scripts.get_candle_data import get_candle_data_json
from systems.scripts.get_window_data import get_window_data_json
from systems.scripts.ledger import load_ledger, save_ledger
from systems.scripts.kraken_utils import get_kraken_balance


DEFAULT_WINDOW = "3d"


def handle_top_of_hour(tag: str, window: str = DEFAULT_WINDOW, verbose: int = 0) -> None:
    """Execute one top-of-hour evaluation cycle for ``tag``."""
    settings = load_settings()
    meta = settings["symbol_settings"][tag]
    meta["window"] = window

    addlog(
        f"[TOP] Processing {tag} | Kraken: {meta['kraken_name']} | "
        f"Wallet: {meta['wallet_code']} | Fiat: {meta['fiat']} (verbose {verbose})",
        verbose_int=1,
        verbose_state=verbose,
    )

    ensure_latest_candles(tag, lookback="48h", verbose=verbose)
    candle = get_candle_data_json(tag, row_offset=0)
    window_data = get_window_data_json(tag, window, candle_offset=0)
    if not candle or not window_data:
        addlog("[ERROR] Missing candle or window data", verbose_int=1, verbose_state=verbose)
        return

    ledger = load_ledger(tag)
    cooldowns = {"knife_catch": 0, "whale_catch": 0, "fish_catch": 0}
    last_triggered = {"knife_catch": None, "whale_catch": None, "fish_catch": None}

    exchange = ccxt.kraken({"enableRateLimit": True})

    evaluate_live_tick(
        candle=candle,
        window_data=window_data,
        ledger=ledger,
        cooldowns=cooldowns,
        last_triggered=last_triggered,
        tag=tag,
        meta=meta,
        exchange=exchange,
        verbose=verbose,
    )
    save_ledger(tag, ledger)

    kraken_balance = get_kraken_balance(verbose)
    fiat_asset = meta["fiat"]
    wallet_code = meta.get("wallet_code", meta["kraken_name"].replace("USD", ""))
    available_usd = float(kraken_balance.get(fiat_asset, 0.0))
    available_coin = float(kraken_balance.get(wallet_code, 0.0))
    coin_price = candle["close"]
    coin_balance_usd = available_coin * coin_price
    total_liquid_value = available_usd + coin_balance_usd

    triggered = {k.title(): v is not None for k, v in last_triggered.items()}
    notes = ledger.get_trade_counts_by_strategy()

    report = format_top_of_hour_report(
        tag,
        datetime.utcnow(),
        available_usd,
        coin_balance_usd,
        wallet_code,
        total_liquid_value,
        triggered,
        notes,
    )
    addlog(report, verbose_int=1, verbose_state=verbose)


def run_top_hour_all(tag: str | None = None, window: str = DEFAULT_WINDOW, verbose: int = 0) -> None:
    """Run ``handle_top_of_hour`` for all configured symbols or a single ``tag``."""
    settings = load_settings()
    symbols = settings.get("symbol_settings", {})
    tags = [tag.upper()] if tag else list(symbols.keys())

    for t in tags:
        if t not in symbols:
            addlog(f"[WARN] Unknown symbol tag: {t}", verbose_int=1, verbose_state=verbose)
            continue
        handle_top_of_hour(t, window=window, verbose=verbose)
